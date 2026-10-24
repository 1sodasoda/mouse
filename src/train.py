"""Train the sequence model on the collected mouse logs, honestly.

Run from the src/ directory:  python train.py

What "honestly" means here, as opposed to the R2 numbers in
``results/experiment_summary.md``:

* the split is **by session file**, so no test window shares a recording with
  a training window (neighbouring rows are ~1 ms apart -- a row-level split
  would leak and flatter the model);
* the input scale is fitted on train only;
* the reported error is on held-out sessions the model never saw;
* trivial baselines (no movement / constant velocity) are reported next to it,
  because autocorrelated motion makes a bare error number meaningless.

Recordings arrive at different polling rates (~1000 Hz for the early logs,
~40-160 Hz from ``data_collector.py``). ``ml/windowing.py`` resamples every
session onto a fixed real-time grid and hands the model its native rate as a
feature, so both are trained on together and the horizon means the same
number of milliseconds everywhere.

Options:  --step-ms --max-gap-ms --window --horizon --epochs --seed --hidden
"""
import argparse
import os

import numpy as np
import torch
from torch import nn

from commons import dataset
from ml import baselines, windowing
from ml.sequence_model import DeltaGRU, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(HERE, "..", "data", "raw")
RESULTS = os.path.join(HERE, "..", "results")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step-ms", type=float, default=windowing.STEP_MS,
                   dest="step_ms",
                   help="ms of real time per step (default %(default)s)")
    p.add_argument("--max-gap-ms", type=float, default=windowing.MAX_GAP_MS,
                   dest="max_gap_ms",
                   help="drop windows spanning a longer pause "
                        "(default %(default)s)")
    p.add_argument("--window", type=int, default=windowing.WINDOW,
                   help="steps of history (default %(default)s)")
    p.add_argument("--horizon", type=int, default=windowing.HORIZON,
                   help="steps ahead to predict (default %(default)s)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20,
                   help="early-stop after N epochs without val improvement")
    p.add_argument("--min-move-px", type=float, default=windowing.MIN_MOVE_PX,
                   dest="min_move_px",
                   help="drop windows whose history moved less than this; "
                        "0 keeps every window (default %(default)s)")
    p.add_argument("--static-frac", type=float, default=0.0,
                   dest="static_frac",
                   help="fraction of dropped still windows to keep anyway, so "
                        "the model still learns to predict no motion "
                        "(default %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-rate", action="store_true", dest="no_rate",
                   help="ablation: blank the polling-rate channel, so the "
                        "model cannot tell which rate a window came from")
    return p.parse_args()


def load_split(args):
    sessions, skipped = dataset.load_sessions(DATA_RAW)
    print(f"usable sessions: {len(sessions)}")
    for path, reason in skipped:
        print(f"  skipped {os.path.relpath(path, HERE)}: {reason}")
    if len(sessions) < 3:
        raise SystemExit(
            "not enough usable sessions to train; record more with "
            "data_collector.py")

    rates = sorted(windowing.native_hz(s.df) for s in sessions)
    print(f"polling rates: {rates[0]:.0f}-{rates[-1]:.0f} Hz "
          f"(median {rates[len(rates) // 2]:.0f} Hz) -- resampled to "
          f"{args.step_ms:.0f} ms steps")

    split = dataset.split_sessions(sessions, seed=args.seed)
    print("\nsplit by session file (no recording spans two splits):")
    print(dataset.describe(split))
    return split


def drop_still(X, Y, groups, args, name, rng):
    """Remove windows where the pointer was not really moving.

    Over half the recorded sessions are the cursor sitting still, which makes
    "predict zero" the winning strategy and hides whether a model learned
    anything. Filtering is on the window history only (see
    ``windowing.moving_mask``), never on the target.
    """
    if args.min_move_px <= 0:
        return X, Y, groups
    keep = windowing.moving_mask(X, args.min_move_px)
    still = ~keep
    if args.static_frac > 0 and still.any():
        # keep a sample of the still windows so the model still sees rest
        idx = np.flatnonzero(still)
        pick = rng.choice(idx, size=int(len(idx) * args.static_frac),
                          replace=False)
        keep[pick] = True
    print(f"{name:5s} windows: {len(X)} -> {int(keep.sum())} moving "
          f"({100 * (1 - keep.mean()):.0f}% still, dropped)")
    return X[keep], Y[keep], groups[keep]


def build_tensors(split, args, device):
    built = {}
    rng = np.random.default_rng(args.seed)
    for name, part in split.items():
        X, Y, groups = windowing.build(part, args.window, args.horizon,
                                       args.step_ms, args.max_gap_ms)
        X, Y, groups = drop_still(X, Y, groups, args, name, rng)
        built[name] = (X, Y, groups)
        if len(X) == 0:
            raise SystemExit(
                f"no windows in '{name}' -- sessions too short for "
                f"step_ms={args.step_ms} window={args.window} "
                f"horizon={args.horizon}")

    scale = windowing.fit_scale(built["train"][0])   # train only
    print(f"\ninput scale (train RMS delta): {scale:.3f} px")

    if args.no_rate:
        print("ablation: polling-rate channel blanked")
        for X, _Y, _g in built.values():
            X[:, :, windowing.RATE] = 0.0

    tensors = {}
    for name, (X, Y, _groups) in built.items():
        tensors[name] = (
            torch.tensor(windowing.normalise(X, scale),
                         dtype=torch.float32, device=device),
            torch.tensor(Y / scale, dtype=torch.float32, device=device),
        )
    return built, tensors, scale


def evaluate(model, xt, Y_px, scale):
    """Mean/median Euclidean error in pixels on un-normalised targets."""
    model.eval()
    with torch.no_grad():
        pred = model(xt).cpu().numpy() * scale
    return baselines.euclidean_error(pred, Y_px), pred


def train(model, tensors, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    xtr, ytr = tensors["train"]
    xva, yva = tensors["val"]

    best, best_state, bad = float("inf"), None, 0
    n = len(xtr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=xtr.device)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xva), yva))
        if val_loss < best - 1e-6:
            best, bad = val_loss, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  val MSE {val_loss:.5f}"
                  f"  (best {best:.5f})")
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch} "
                  f"({args.patience} epochs without improvement)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def report(model, built, tensors, scale, args, split):
    X_te, Y_te, groups_te = built["test"]
    (mean_px, med_px), pred = evaluate(model, tensors["test"][0], Y_te, scale)

    rows = [("GRU (this model)", mean_px, med_px)]
    for name, fn in baselines.BASELINES.items():
        rows.append((name, *baselines.euclidean_error(
            fn(X_te, args.horizon), Y_te)))

    print(f"\nheld-out test error -- displacement {args.horizon} steps "
          f"({args.horizon * args.step_ms:.0f} ms) ahead")
    print(f"{'predictor':24s} {'mean px':>9s} {'median px':>10s}")
    for name, mean_e, med_e in rows:
        print(f"{name:24s} {mean_e:9.2f} {med_e:10.2f}")

    if dict((n, md) for n, _m, md in rows)["zero (no movement)"] == 0.0:
        print("\nWARNING: the 'do nothing' baseline has zero median error -- "
              "most test windows are motionless, so these numbers cannot "
              "tell a good model from a bad one. Raise --min-move-px.")

    cv = dict((n, m) for n, m, _ in rows)["constant velocity"]
    gru = rows[0][1]
    verdict = ("beats" if gru < cv else "does NOT beat")
    print(f"\nGRU {verdict} constant velocity "
          f"({gru:.2f} vs {cv:.2f} px mean error).")

    # per-session test breakdown, so one odd recording is visible
    print("\nper test session:")
    for i, s in enumerate(split["test"]):
        m = groups_te == i
        if not m.any():
            continue
        e_gru, _ = baselines.euclidean_error(pred[m], Y_te[m])
        e_cv, _ = baselines.euclidean_error(
            baselines.predict_constant_velocity(X_te[m], args.horizon), Y_te[m])
        print(f"  {s.gesture:8s} {os.path.basename(s.path):40s} "
              f"n={int(m.sum()):4d}  GRU {e_gru:7.2f}  const-vel {e_cv:7.2f}")

    os.makedirs(RESULTS, exist_ok=True)
    ckpt = os.path.join(RESULTS, "sequence_model.pt")
    torch.save({"state_dict": model.state_dict(), "scale": scale,
                "window": args.window, "horizon": args.horizon,
                "step_ms": args.step_ms, "max_gap_ms": args.max_gap_ms,
                "hidden": args.hidden}, ckpt)
    csv = os.path.join(RESULTS, "sequence_metrics.csv")
    with open(csv, "w") as fh:
        fh.write("predictor,mean_px,median_px\n")
        for name, mean_e, med_e in rows:
            fh.write(f'"{name}",{mean_e:.4f},{med_e:.4f}\n')
    print(f"\nsaved {os.path.relpath(ckpt, HERE)} and "
          f"{os.path.relpath(csv, HERE)}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"device: {device}\n")

    split = load_split(args)
    built, tensors, scale = build_tensors(split, args, device)

    model = DeltaGRU(hidden=args.hidden).to(device)
    print(f"\ntraining DeltaGRU(hidden={args.hidden}) for up to "
          f"{args.epochs} epochs")
    model = train(model, tensors, args)
    report(model, built, tensors, scale, args, split)


if __name__ == "__main__":
    main()
