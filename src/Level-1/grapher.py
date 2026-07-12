import matplotlib.pyplot as plot
import pandas as pd
import numpy as np


def moving_average(series, window=5):
    """Simple moving average over a fixed window (centered)."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def exponential_smoothing(series, alpha=0.2):
    """Exponential smoothing: y[n] = alpha*x[n] + (1-alpha)*y[n-1].

    alpha in (0, 1]. Smaller alpha -> smoother, more lag.
    """
    return series.ewm(alpha=alpha, adjust=False).mean()


def low_pass(series, cutoff=5.0, times=None):
    """1-pole RC low-pass filter with cutoff in Hz.

    Uses real sample spacing from `times` (per-sample alpha) so irregular
    logging intervals are handled correctly. Falls back to dt=1 if no times.
    """
    x = series.to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return series

    if times is not None:
        t = pd.to_datetime(times)
        dt = t.diff().dt.total_seconds().to_numpy()
        dt[0] = dt[1] if n > 1 else 1.0
        dt = np.where(np.isnan(dt) | (dt <= 0), 1e-3, dt)
    else:
        dt = np.ones(n)

    rc = 1.0 / (2 * np.pi * cutoff)
    y = np.empty(n)
    y[0] = x[0]
    for i in range(1, n):
        a = dt[i] / (rc + dt[i])
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return pd.Series(y, index=series.index)


# Registry: name -> (function, param name, default)
FILTERS = {
    'ma': (moving_average, 'window', 5),
    'ema': (exponential_smoothing, 'alpha', 0.2),
    'lpf': (low_pass, 'cutoff', 5.0),
}


class Plotter:

    def __init__(self, file):
        self.file = file
        self.df = pd.read_csv(file)

    def _apply(self, columns, kind=None, param=None):
        """Return a copy of df with the given columns filtered.

        kind: one of FILTERS keys ('ma', 'ema', 'lpf'), or None for raw.
        param: override the filter's parameter (window / alpha / cutoff).
        """
        if kind is None:
            return self.df

        if kind not in FILTERS:
            raise ValueError(f"unknown filter '{kind}', pick from {list(FILTERS)}")

        fn, pname, default = FILTERS[kind]
        value = default if param is None else param
        out = self.df.copy()
        times = self.df['time'] if 'time' in self.df.columns else None

        for col in columns:
            if kind == 'lpf':
                out[col] = fn(self.df[col], value, times=times)
            else:
                out[col] = fn(self.df[col], value)
        return out

    def _plot(self, xcol, ycol, kind, param):
        ax = None
        if kind is not None:
            # raw underlay in red so the filter's effect is visible
            ax = self.df.plot(x=xcol, y=ycol, color='red', alpha=0.4,
                              label=f'{ycol} (raw)')
        df = self._apply([xcol, ycol], kind, param)
        label = ycol if kind is None else f'{ycol} ({kind})'
        df.plot(x=xcol, y=ycol, ax=ax, label=label)
        plot.show()

    def load_pos(self, kind=None, param=None):
        self._plot('x', 'y', kind, param)

    def load_mag(self, kind=None, param=None):
        self._plot('Magnitude', 'Direction', kind, param)
