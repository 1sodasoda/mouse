"""Train several architectures on identical data and compare them.

Run from the src/ directory:  python compare_models.py

Loads the logs **once**, then for every seed splits and windows them and
trains each architecture on exactly the same tensors with the same early
stopping and budget, so the only thing that differs is the model. Several
seeds means several different session-level splits: with a few dozen test
recordings a single split can rank two models the wrong way round, so the
spread across seeds matters as much as the best number. Reports held-out pixel error next to the
constant-velocity baseline, plus the share of individual windows each model
wins -- mean error alone hides a model that is better on a few big moves and
worse everywhere else.

Options:  --models --seeds --epochs --patience --hidden  (plus the windowing
flags from train.py: --step-ms --window --horizon --min-move-px ...)
"""
import argparse
import os

import numpy as np
import torch

import train as trainer
from commons import dataset
from ml import baselines, windowing
from ml.sequence_model import ARCHITECTURES, make_model, count_params, pick_device

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(ARCHITECTURES),
                   help="comma-separated architectures (default: all)")
    p.add_argument("--step-ms", type=float, default=windowing.STEP_MS,
                   dest="step_ms")
    p.add_argument("--max-gap-ms", type=float, default=windowing.MAX_GAP_MS,
                   dest="max_gap_ms")
    p.add_argument("--window", type=int, default=windowing.WINDOW)
    p.add_argument("--horizon", type=int, default=windowing.HORIZON)
    p.add_argument("--min-move-px", type=float, default=windowing.MIN_MOVE_PX,
                   dest="min_move_px")
    p.add_argument("--static-frac", type=float, default=0.0,
                   dest="static_frac")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None,
                   help="comma-separated split seeds to sweep; the spread "
                        "across them says whether a ranking is real or luck")
    p.add_argument("--no-rate", action="store_true", dest="no_rate")
    return p.parse_args()


def run_one_seed(sessions, args, device, names, per_model):
    """Train every architecture on one split and record its test error."""
    split = dataset.split_sessions(sessions, seed=args.seed)
    print(f"\n{'=' * 72}\nseed {args.seed}")
    print(dataset.describe(split).split("\n")[0])
    built, tensors, scale = trainer.build_tensors(split, args, device)

    X_te, Y_te, _g = built["test"]
    cv_pred = baselines.predict_constant_velocity(X_te, args.horizon)
    cv_err = np.hypot(*(cv_pred - Y_te).T)
    cv = baselines.euclidean_error(cv_pred, Y_te)
    zero = baselines.euclidean_error(
        baselines.predict_zero(X_te, args.horizon), Y_te)

    for name in names:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = make_model(name, hidden=args.hidden, window=args.window,
                           horizon=args.horizon).to(device)
        n_par = count_params(model)
        print(f"\n--- seed {args.seed}  {name}  ({n_par} params) ---")
        model = trainer.train(model, tensors, args)
        (mean_px, med_px), pred = trainer.evaluate(
            model, tensors["test"][0], Y_te, scale)
        win = 100.0 * (np.hypot(*(pred - Y_te).T) < cv_err).mean()
        per_model.setdefault(name, {"params": n_par, "runs": []})
        per_model[name]["runs"].append((mean_px, med_px, win))
        print(f"  test mean {mean_px:.2f} px  median {med_px:.2f} px  "
              f"beats const-vel on {win:.0f}% of windows")
    return cv, zero


def main():
    args = parse_args()
    device = pick_device()
    print(f"device: {device}\n")

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds
             else [args.seed])

    # the CSV read dominates runtime, so do it once for every seed
    sessions, skipped = dataset.load_sessions(trainer.DATA_RAW)
    print(f"usable sessions: {len(sessions)} "
          f"({len(skipped)} skipped as too short or near-static)")

    per_model, cvs, zeros = {}, [], []
    for seed in seeds:
        args.seed = seed
        cv, zero = run_one_seed(sessions, args, device, names, per_model)
        cvs.append(cv)
        zeros.append(zero)

    def agg(runs, i):
        v = np.array([r[i] for r in runs], dtype=float)
        return v.mean(), v.min(), v.max()

    rows = []
    for name, d in per_model.items():
        m_mean, m_lo, m_hi = agg(d["runs"], 0)
        md_mean, _, _ = agg(d["runs"], 1)
        w_mean, w_lo, w_hi = agg(d["runs"], 2)
        beat = sum(1 for (mp, _md, _w), c in zip(d["runs"], cvs) if mp < c[0])
        rows.append((name, d["params"], m_mean, m_lo, m_hi, md_mean,
                     w_mean, w_lo, w_hi, beat))
    rows.sort(key=lambda r: r[2])

    print("\n" + "=" * 86)
    print(f"held-out comparison over {len(seeds)} split seed(s) -- "
          f"{args.horizon} steps ({args.horizon * args.step_ms:.0f} ms) ahead")
    print(f"{'model':9s} {'params':>7s} {'mean px':>9s} {'(min-max)':>14s} "
          f"{'median':>7s} {'win% vs cv':>11s} {'seeds won':>10s}")
    for (name, n_par, m_mean, m_lo, m_hi, md_mean,
         w_mean, w_lo, w_hi, beat) in rows:
        print(f"{name:9s} {n_par:7d} {m_mean:9.2f} "
              f"{f'({m_lo:.2f}-{m_hi:.2f})':>14s} {md_mean:7.2f} "
              f"{w_mean:10.0f}% {f'{beat}/{len(seeds)}':>10s}")
    cv_m = np.mean([c[0] for c in cvs]); cv_md = np.mean([c[1] for c in cvs])
    z_m = np.mean([z[0] for z in zeros]); z_md = np.mean([z[1] for z in zeros])
    print(f"{'const-vel':9s} {0:7d} {cv_m:9.2f} {'':>14s} {cv_md:7.2f}")
    print(f"{'zero':9s} {0:7d} {z_m:9.2f} {'':>14s} {z_md:7.2f}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "architecture_comparison.csv")
    with open(out, "w") as fh:
        fh.write("model,params,mean_px,mean_px_min,mean_px_max,median_px,"
                 "win_pct_vs_constvel,seeds_beating_constvel,n_seeds\n")
        for (name, n_par, m_mean, m_lo, m_hi, md_mean,
             w_mean, _wl, _wh, beat) in rows:
            fh.write(f"{name},{n_par},{m_mean:.4f},{m_lo:.4f},{m_hi:.4f},"
                     f"{md_mean:.4f},{w_mean:.1f},{beat},{len(seeds)}\n")
        fh.write(f"constant velocity,0,{cv_m:.4f},,,{cv_md:.4f},,,{len(seeds)}\n")
        fh.write(f"zero,0,{z_m:.4f},,,{z_md:.4f},,,{len(seeds)}\n")
    print(f"\nsaved {os.path.relpath(out, os.path.dirname(__file__))}")


if __name__ == "__main__":
    main()
