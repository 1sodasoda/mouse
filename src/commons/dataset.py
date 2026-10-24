"""Dataset assembly from the raw logs.

``logger/mouse_logger.py`` writes one CSV per recording into
``data/raw/<gesture>/``. This module turns that directory into a dataset:
discover the logs, drop the ones that cannot be used, tag each with the gesture
folder it came from, and split them **by file**.

Splitting by file matters: the logger samples at ~1000 Hz, so neighbouring rows
are ~1 ms and ~1 px apart. A row-level shuffle would put near-duplicate rows in
both train and test and report an error far lower than the real one.
"""
import os
import random
from collections import Counter, namedtuple

import numpy as np
import pandas as pd

from commons.mouse_logger import load_log, POS_COLS, TIME_COL

# Logs shorter than this are fumbles / smoke tests, not usable sessions.
MIN_SAMPLES = 200

# Logs whose pointer travelled less than this in total are the cursor sitting
# still. pynput fires on sub-pixel jitter, so an idle recording still produces
# plenty of rows -- a row count alone does not tell motion from stillness, and
# a dataset of motionless sessions makes "predict zero" the winning strategy.
MIN_PATH_PX = 200.0

# One recorded log: file path, gesture label, loaded frame.
Session = namedtuple("Session", "path gesture df")


def path_length(df):
    """Total distance the pointer travelled over a session, in pixels."""
    p = df[list(POS_COLS)].to_numpy(dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def discover(data_raw):
    """Every .csv under data_raw as (path, gesture), sorted.

    The gesture is the immediate parent directory name, which is how both
    ``data_collector.py`` and ``testmain1.py`` lay recordings out.
    """
    found = []
    for root, _dirs, files in os.walk(data_raw):
        for f in sorted(files):
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                found.append((path, os.path.basename(root)))
    return sorted(found)


def load_sessions(data_raw, min_samples=MIN_SAMPLES, min_path_px=MIN_PATH_PX):
    """Load every usable log under data_raw.

    Returns (sessions, skipped) where skipped is a list of (path, reason), so
    the caller can report exactly what was left out instead of silently
    shrinking the dataset. A log is usable when it has the x/y and time
    columns, at least min_samples rows, and the pointer actually moved
    min_path_px in total.
    """
    sessions, skipped = [], []
    for path, gesture in discover(data_raw):
        try:
            df = load_log(path)
        except Exception as exc:                      # unreadable / malformed
            skipped.append((path, f"unreadable ({exc.__class__.__name__})"))
            continue

        missing = [c for c in (*POS_COLS, TIME_COL) if c not in df.columns]
        if missing:
            skipped.append((path, f"missing columns {missing}"))
            continue
        if len(df) < min_samples:
            skipped.append((path, f"only {len(df)} rows (< {min_samples})"))
            continue

        df = df.dropna(subset=[*POS_COLS, TIME_COL]).reset_index(drop=True)
        if len(df) < min_samples:
            skipped.append((path, f"only {len(df)} rows after dropna"))
            continue

        travelled = path_length(df)
        if travelled < min_path_px:
            skipped.append((path, f"near-static (pointer travelled "
                                  f"{travelled:.0f} px < {min_path_px:.0f})"))
            continue

        sessions.append(Session(path, gesture, df))
    return sessions, skipped


def split_sessions(sessions, fractions=(0.7, 0.15, 0.15), seed=0):
    """Partition whole sessions into train / val / test.

    Deterministic for a given seed. val and test get at least one session each
    when there are enough to go round; the remainder goes to train.
    """
    if len(sessions) < 3:
        raise ValueError(
            f"need at least 3 sessions to split, have {len(sessions)}; "
            "record more with data_collector.py")

    order = sorted(sessions, key=lambda s: s.path)
    rng = random.Random(seed)
    rng.shuffle(order)

    n = len(order)
    n_test = max(1, round(fractions[2] * n))
    n_val = max(1, round(fractions[1] * n))
    if n_val + n_test >= n:                           # keep train non-empty
        n_val, n_test = 1, 1
    return {
        "train": order[: n - n_val - n_test],
        "val": order[n - n_val - n_test: n - n_test],
        "test": order[n - n_test:],
    }


def describe(split):
    """Human-readable summary of a split: sessions, rows, gestures per part."""
    lines = []
    for name, part in split.items():
        rows = sum(len(s.df) for s in part)
        gestures = Counter(s.gesture for s in part)
        lines.append(f"{name:5s} {len(part):2d} sessions  {rows:6d} rows  "
                     f"{dict(gestures)}")
        for s in part:
            lines.append(f"        {s.gesture:8s} {os.path.basename(s.path)}"
                         f"  ({len(s.df)} rows)")
    return "\n".join(lines)
