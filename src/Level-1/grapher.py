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


def kalman_filter(series, meas_var=50.0, times=None):
    """1-D constant-velocity Kalman smoother on a position series.

    State = [position, velocity]. Fuses a constant-velocity motion model with
    the noisy measurements, so it tracks fast moves with less lag than the
    fixed smoothers while still rejecting jitter.

    meas_var (R): measurement noise variance. Larger -> trust the model more
    -> smoother output. Uses real per-sample dt from `times` when available.
    """
    z = series.to_numpy(dtype=float)
    n = len(z)
    if n == 0:
        return series

    if times is not None:
        t = pd.to_datetime(times)
        dt = t.diff().dt.total_seconds().to_numpy()
        dt[0] = dt[1] if n > 1 else 1.0
        dt = np.where(np.isnan(dt) | (dt <= 0), 1e-3, dt)
    else:
        dt = np.ones(n)

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


# Registry: name -> (function, param name, default)
FILTERS = {
    'ma': (moving_average, 'window', 5),
    'ema': (exponential_smoothing, 'alpha', 0.2),
    'lpf': (low_pass, 'cutoff', 5.0),
    'kf': (kalman_filter, 'meas_var', 50.0),
}

# filters that consume the per-sample timestamps for real dt
_TIME_AWARE = ('lpf', 'kf')


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
            if kind in _TIME_AWARE:
                out[col] = fn(self.df[col], value, times=times)
            else:
                out[col] = fn(self.df[col], value)
        return out

    # per-layer style for the 'all' comparison view.
    # widest drawn first (underneath), narrowest last (on top), so every
    # line stays visible even where the filtered curves nearly coincide.
    _STYLES = {
        'raw': dict(color='red',        lw=4.0, alpha=0.30, ls='-'),
        'ma':  dict(color='tab:blue',   lw=3.0, alpha=0.85, ls='-'),
        'ema': dict(color='tab:green',  lw=2.2, alpha=0.85, ls='--'),
        'lpf': dict(color='tab:purple', lw=1.5, alpha=0.90, ls=':'),
        'kf':  dict(color='tab:orange', lw=1.0, alpha=0.95, ls='-.'),
    }

    def _plot(self, xcol, ycol, kind, param):
        if kind == 'all':
            return self._plot_all(xcol, ycol)

        ax = None
        if kind is not None:
            # raw underlay in red so the filter's effect is visible
            ax = self.df.plot(x=xcol, y=ycol, color='red', lw=3.5,
                              alpha=0.35, label=f'{ycol} (raw)')
        df = self._apply([xcol, ycol], kind, param)
        label = ycol if kind is None else f'{ycol} ({kind})'
        df.plot(x=xcol, y=ycol, ax=ax, label=label)
        plot.show()

    def _plot_all(self, xcol, ycol):
        """Overlay raw + every filter (each at its own default) for comparison."""
        ax = self.df.plot(x=xcol, y=ycol, label=f'{ycol} (raw)',
                          **self._STYLES['raw'])
        for kind in FILTERS:
            df = self._apply([xcol, ycol], kind, None)
            df.plot(x=xcol, y=ycol, ax=ax, label=f'{ycol} ({kind})',
                    **self._STYLES[kind])
        ax.legend()
        plot.show()

    def load_pos(self, kind=None, param=None):
        self._plot('x', 'y', kind, param)

    def load_mag(self, kind=None, param=None):
        self._plot('Magnitude', 'Direction', kind, param)
