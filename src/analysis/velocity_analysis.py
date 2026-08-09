"""Velocity (speed + heading) analysis of a trajectory.

The logger already records Magnitude/Direction per step; these helpers
recompute them from raw x/y so any trajectory (filtered or reconstructed) can
be analysed the same way.
"""
import numpy as np


def step_velocity(x, y):
    """Per-step velocity in polar form from position arrays.

    Returns (magnitude, direction) where magnitude = step length and
    direction = atan2(dy, dx). First sample is 0 (no previous point).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    magnitude = np.hypot(dx, dy)
    direction = np.arctan2(dy, dx)
    return magnitude, direction


def mean_speed(x, y):
    """Mean per-step speed (pixels/sample) over a trajectory."""
    mag, _ = step_velocity(x, y)
    return float(mag.mean())
