"""Retuned constant-velocity Kalman smoother (kf2)."""
import numpy as np
import pandas as pd

from commons.mouse_logger import dt_from_times


def tuned_kalman_filter(series, q=5.0, times=None):
    """Retuned constant-velocity Kalman smoother (successor to kalman_filter).

    Two fixes over the original, which over-trusted the motion model and so
    undershot sharp turns (it only reached ~84% of the raw extent on the
    circle):

    * process noise `q` raised (default 5.0 vs the old hard-coded 1.0) so the
      filter can accelerate through corners and reach the turn extremes;
    * initial velocity seeded from the first step instead of 0, removing the
      startup lag.

    `q` is the tunable knob: higher -> tracks fast moves / turns more closely
    (less smoothing), lower -> smoother. Measurement noise R is fixed at 50.
    Uses real per-sample dt from `times` when available.
    """
    z = series.to_numpy(dtype=float)
    n = len(z)
    if n == 0:
        return series

    dt = dt_from_times(times, n)
    R = 50.0
    H = np.array([[1.0, 0.0]])
    I = np.eye(2)

    v0 = (z[1] - z[0]) / dt[1] if n > 1 else 0.0   # seed velocity from step 1
    x = np.array([z[0], v0])
    P = np.array([[R, 0.0], [0.0, 1.0]])
    out = np.empty(n)
    out[0] = x[0]

    for i in range(1, n):
        d = dt[i]
        F = np.array([[1.0, d], [0.0, 1.0]])
        Q = q * np.array([[d ** 3 / 3, d ** 2 / 2],
                          [d ** 2 / 2, d]])
        x = F @ x
        P = F @ P @ F.T + Q
        resid = z[i] - (H @ x)[0]
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T)[:, 0] / S
        x = x + K * resid
        P = (I - np.outer(K, H[0])) @ P
        out[i] = x[0]

    return pd.Series(out, index=series.index)


# Backwards-compatible alias for the previous name.
kalman_filter_v2 = tuned_kalman_filter
