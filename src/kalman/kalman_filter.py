"""Original constant-velocity Kalman smoother (kf1)."""
import numpy as np
import pandas as pd

from commons.mouse_logger import dt_from_times


def kalman_filter(series, meas_var=50.0, times=None):
    """1-D constant-velocity Kalman smoother on a position series.

    State = [position, velocity]. Fuses a constant-velocity motion model with
    the noisy measurements, so it tracks fast moves with less lag than the
    fixed smoothers while still rejecting jitter.

    meas_var (R): measurement noise variance. Larger -> trust the model more
    -> smoother output. Uses real per-sample dt from `times` when available.

    NOTE: over-trusts the model and undershoots sharp turns; see
    ``tuned_kalman_filter`` for the retuned successor.
    """
    z = series.to_numpy(dtype=float)
    n = len(z)
    if n == 0:
        return series

    dt = dt_from_times(times, n)
    q = 1.0                       # process noise (accel) spectral density
    R = float(meas_var)
    H = np.array([[1.0, 0.0]])
    I = np.eye(2)

    x = np.array([z[0], 0.0])                  # state: [pos, vel]
    P = np.array([[R, 0.0], [0.0, 1.0]])       # state covariance
    out = np.empty(n)
    out[0] = x[0]

    for i in range(1, n):
        d = dt[i]
        F = np.array([[1.0, d], [0.0, 1.0]])
        Q = q * np.array([[d ** 3 / 3, d ** 2 / 2],
                          [d ** 2 / 2, d]])
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        resid = z[i] - (H @ x)[0]
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T)[:, 0] / S                 # Kalman gain, 2-vector
        x = x + K * resid
        P = (I - np.outer(K, H[0])) @ P
        out[i] = x[0]

    return pd.Series(out, index=series.index)
