"""Trivial predictors the learned model has to beat.

A held-out error number means nothing on its own -- mouse motion is highly
autocorrelated, so a rule with no parameters already does well. These are the
reference points reported next to the model in ``train.py``.
"""
import numpy as np

from ml.windowing import DELTAS


def predict_zero(X, horizon):
    """"The mouse does not move": always predict a zero displacement."""
    return np.zeros((len(X), 2))


def predict_constant_velocity(X, horizon):
    """Carry the last observed delta forward for ``horizon`` steps.

    This is the discrete constant-velocity model -- the same assumption the
    Kalman filters in ``kalman/`` make, and the one to beat. Steps are a fixed
    duration after resampling, so this is comparable across polling rates.
    """
    return np.asarray(X, dtype=float)[:, -1, DELTAS] * horizon


def predict_mean_velocity(X, horizon):
    """Carry the window's mean delta forward -- a smoothed velocity estimate."""
    return np.asarray(X, dtype=float)[:, :, DELTAS].mean(axis=1) * horizon


BASELINES = {
    "zero (no movement)": predict_zero,
    "constant velocity": predict_constant_velocity,
    "mean velocity": predict_mean_velocity,
}


def euclidean_error(pred, true):
    """Mean and median Euclidean error in pixels."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    d = np.hypot(pred[:, 0] - true[:, 0], pred[:, 1] - true[:, 1])
    return float(d.mean()), float(np.median(d))
