"""Trajectory-level motion metrics used to compare filters/models.

These are the measures behind the write-ups: how much a filtered path still
reaches the extreme turn positions, and how jittery it is.
"""
import numpy as np


def max_reach(x, y):
    """Largest displacement from the path centroid (overall extent)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cx, cy = x.mean(), y.mean()
    return float(np.hypot(x - cx, y - cy).max())


def reach_retention(fx, fy, rx, ry):
    """Filtered max reach as a fraction of the raw max reach.

    ~1.0 means turn extremes preserved; <1 means the filter undershoots
    corners (measured about the *raw* centroid for a fair comparison).
    """
    rx = np.asarray(rx, dtype=float)
    ry = np.asarray(ry, dtype=float)
    cx, cy = rx.mean(), ry.mean()
    raw = np.hypot(rx - cx, ry - cy).max()
    filt = np.hypot(np.asarray(fx, float) - cx,
                    np.asarray(fy, float) - cy).max()
    return float(filt / raw)


def jitter(a):
    """Mean absolute second difference — high-frequency roughness of a signal."""
    a = np.asarray(a, dtype=float)
    if len(a) < 3:
        return 0.0
    return float(np.abs(np.diff(a, 2)).mean())


def path_jitter(x, y):
    """Average jitter across both coordinates of a trajectory."""
    return (jitter(x) + jitter(y)) / 2
