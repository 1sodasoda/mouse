"""Shared helpers for working with logged mouse data.

This is the *common* side of mouse logging: loading recorded CSVs and the
per-sample timing math that several filters need. The recording side (the live
`pynput` listener) lives in ``logger/mouse_logger.py``.
"""
import numpy as np
import pandas as pd

# Column names written by the logger.
POS_COLS = ('x', 'y')
VELOCITY_COLS = ('Magnitude', 'Direction')
TIME_COL = 'time'


def load_log(path):
    """Read a logged CSV into a DataFrame."""
    return pd.read_csv(path)


def dt_from_times(times, n):
    """Per-sample time deltas (seconds) from a `time` column.

    Shared by the low-pass and Kalman filters so irregular logging intervals
    are handled consistently. Falls back to unit spacing when `times` is None.
    Non-positive / missing gaps are clamped to 1 ms.

    Parsed as ISO8601 rather than letting pandas infer one format from the
    first row: ``str(datetime.now())`` omits the fractional part when the
    microseconds are exactly zero, so a single row in a long log can read
    "2026-10-18 06:11:50" while the rest carry microseconds.
    """
    if times is None:
        return np.ones(n)
    t = pd.to_datetime(times, format="ISO8601")
    dt = t.diff().dt.total_seconds().to_numpy()
    dt[0] = dt[1] if n > 1 else 1.0
    return np.where(np.isnan(dt) | (dt <= 0), 1e-3, dt)
