"""1-pole RC low-pass filter."""
import numpy as np
import pandas as pd

from commons.mouse_logger import dt_from_times


def low_pass(series, cutoff=5.0, times=None):
    """1-pole RC low-pass filter with cutoff in Hz.

    Uses real sample spacing from `times` (per-sample alpha) so irregular
    logging intervals are handled correctly. Falls back to dt=1 if no times.
    """
    x = series.to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return series

    dt = dt_from_times(times, n)
    rc = 1.0 / (2 * np.pi * cutoff)
    y = np.empty(n)
    y[0] = x[0]
    for i in range(1, n):
        a = dt[i] / (rc + dt[i])
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return pd.Series(y, index=series.index)
