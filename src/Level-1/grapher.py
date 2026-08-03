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


def kalman_filter_v2(series, q=5.0, times=None):
    """Retuned constant-velocity Kalman smoother (successor to kalman_filter).

    Two fixes over the original, which over-trusted the motion model and so
    undershot sharp turns (see results.md — it only reached ~84% of the raw
    extent on the circle):

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

    if times is not None:
        t = pd.to_datetime(times)
        dt = t.diff().dt.total_seconds().to_numpy()
        dt[0] = dt[1] if n > 1 else 1.0
        dt = np.where(np.isnan(dt) | (dt <= 0), 1e-3, dt)
    else:
        dt = np.ones(n)

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


# Registry: name -> (function, param name, default)
FILTERS = {
    'ma': (moving_average, 'window', 5),
    'ema': (exponential_smoothing, 'alpha', 0.2),
    'lpf': (low_pass, 'cutoff', 5.0),
    'kf': (kalman_filter, 'meas_var', 50.0),      # original (over-smooths turns)
    'kf2': (kalman_filter_v2, 'q', 5.0),          # retuned (reaches turn extremes)
}

# filters that consume the per-sample timestamps for real dt
_TIME_AWARE = ('lpf', 'kf', 'kf2')


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
        'kf':  dict(color='tab:orange', lw=1.2, alpha=0.95, ls='-.'),
        'kf2': dict(color='tab:brown',  lw=0.9, alpha=0.95, ls='-'),
    }

    def _plot(self, xcol, ycol, kind, param):
        if kind == 'all':
            return self._plot_all(xcol, ycol)
        if kind == 'lr':
            return self._plot_lr(xcol, ycol)
        if kind == 'rf':
            return self._plot_rf(xcol, ycol)
        if kind == 'lrt':
            return self._plot_param(xcol, ycol, 'lr')
        if kind == 'rft':
            return self._plot_param(xcol, ycol, 'rf')
        if kind == 'lrf':
            return self._plot_features(xcol, ycol, 'lr')
        if kind == 'rff':
            return self._plot_features(xcol, ycol, 'rf')

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

    def fit_line(self, xcol, ycol):
        """Least-squares linear fit of ycol on xcol via scikit-learn.

        Returns (slope, intercept, r2, model).
        """
        from sklearn.linear_model import LinearRegression

        x = self.df[[xcol]].to_numpy(dtype=float)
        y = self.df[ycol].to_numpy(dtype=float)
        model = LinearRegression().fit(x, y)
        return float(model.coef_[0]), float(model.intercept_), \
            float(model.score(x, y)), model

    def _plot_lr(self, xcol, ycol):
        """Scatter the raw points and overlay the best-fit regression line."""
        m, b, r2, model = self.fit_line(xcol, ycol)
        ax = self.df.plot.scatter(x=xcol, y=ycol, color='red', alpha=0.3,
                                  s=6, label=f'{ycol} (raw)')
        xs = np.linspace(self.df[xcol].min(), self.df[xcol].max(), 200)
        ys = model.predict(xs.reshape(-1, 1))
        ax.plot(xs, ys, color='black', lw=2,
                label=f'fit: y = {m:.3f}x + {b:.1f}   R² = {r2:.3f}')
        ax.legend()
        plot.show()

    def fit_forest(self, xcol, ycol, n_estimators=100, max_depth=5,
                   min_samples_leaf=20):
        """Random-forest regression of ycol on xcol via scikit-learn.

        Nonlinear, so it can follow curved paths a straight line can't.
        max_depth / min_samples_leaf are regularizers: with the defaults each
        leaf averages >=20 points and trees stay shallow, so the fit is smooth
        instead of memorizing the training points (unlimited depth overfits
        badly on this data). Returns (r2, model); random_state fixed.
        """
        from sklearn.ensemble import RandomForestRegressor

        x = self.df[[xcol]].to_numpy(dtype=float)
        y = self.df[ycol].to_numpy(dtype=float)
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=0).fit(x, y)
        return float(model.score(x, y)), model

    def _plot_rf(self, xcol, ycol):
        """Scatter the raw points and overlay the random-forest prediction."""
        r2, model = self.fit_forest(xcol, ycol)
        ax = self.df.plot.scatter(x=xcol, y=ycol, color='red', alpha=0.3,
                                  s=6, label=f'{ycol} (raw)')
        xs = np.linspace(self.df[xcol].min(), self.df[xcol].max(), 400)
        ys = model.predict(xs.reshape(-1, 1))
        ax.plot(xs, ys, color='black', lw=2,
                label=f'random forest   R² = {r2:.3f}')
        ax.legend()
        plot.show()

    def fit_parametric(self, xcol, ycol, kind='rf'):
        """Fit xcol and ycol each as a function of sample index (parametric).

        A trajectory is parametric: x = f(t), y = g(t). Regressing against the
        index t (instead of y on x) gives one (x, y) per t, so multivalued
        paths like a circle or crisscrossing flicks fit correctly.

        kind: 'lr' (straight-line param) or 'rf' (nonlinear, traces curves).
        Returns (x_pred, y_pred, r2x, r2y) over a dense t grid.
        """
        n = len(self.df)
        t = np.arange(n, dtype=float).reshape(-1, 1)
        x = self.df[xcol].to_numpy(dtype=float)
        y = self.df[ycol].to_numpy(dtype=float)

        if kind == 'lr':
            from sklearn.linear_model import LinearRegression
            make = LinearRegression
        else:
            from sklearn.ensemble import RandomForestRegressor
            make = lambda: RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=5,
                random_state=0)

        mx, my = make(), make()
        mx.fit(t, x)
        my.fit(t, y)
        tg = np.linspace(0, n - 1, min(2000, 2 * n)).reshape(-1, 1)
        return (mx.predict(tg), my.predict(tg),
                float(mx.score(t, x)), float(my.score(t, y)))

    # extra logged columns fed to the feature-based fits, when present
    _EXTRA_FEATURES = ('Magnitude', 'Direction')

    def fit_features(self, xcol, ycol, kind='rf', feature_cols=None):
        """Predict xcol, ycol from sample index t plus extra logged columns.

        Feature matrix = [t, Magnitude, Direction, ...] (whichever of
        _EXTRA_FEATURES exist). Predicts on the real feature rows (not a
        synthetic grid, since the extra features are only defined at samples),
        so the fit is the per-sample predicted trajectory in order.

        Returns (x_pred, y_pred, r2x, r2y, used_cols).
        """
        n = len(self.df)
        t = np.arange(n, dtype=float)
        if feature_cols is None:
            feature_cols = [c for c in self._EXTRA_FEATURES
                            if c in self.df.columns]
        cols = ['t'] + feature_cols
        X = np.column_stack(
            [t] + [self.df[c].to_numpy(dtype=float) for c in feature_cols])
        x = self.df[xcol].to_numpy(dtype=float)
        y = self.df[ycol].to_numpy(dtype=float)

        if kind == 'lr':
            from sklearn.linear_model import LinearRegression
            make = LinearRegression
        else:
            from sklearn.ensemble import RandomForestRegressor
            make = lambda: RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=5,
                random_state=0)

        mx, my = make(), make()
        mx.fit(X, x)
        my.fit(X, y)
        return (mx.predict(X), my.predict(X),
                float(mx.score(X, x)), float(my.score(X, y)), cols)

    def _plot_features(self, xcol, ycol, kind):
        """Scatter raw points and overlay the feature-based predicted path."""
        xg, yg, r2x, r2y, cols = self.fit_features(xcol, ycol, kind)
        ax = self.df.plot.scatter(x=xcol, y=ycol, color='red', alpha=0.3,
                                  s=6, label=f'{ycol} (raw)')
        name = 'linear' if kind == 'lr' else 'random forest'
        ax.plot(xg, yg, color='black', lw=1.5,
                label=f"{name} [{'+'.join(cols)}]  R²x={r2x:.3f} R²y={r2y:.3f}")
        ax.legend()
        plot.show()

    def _plot_param(self, xcol, ycol, kind):
        """Scatter raw points and overlay the parametric (index-based) fit."""
        xg, yg, r2x, r2y = self.fit_parametric(xcol, ycol, kind)
        ax = self.df.plot.scatter(x=xcol, y=ycol, color='red', alpha=0.3,
                                  s=6, label=f'{ycol} (raw)')
        name = 'linear' if kind == 'lr' else 'random forest'
        ax.plot(xg, yg, color='black', lw=2,
                label=f'{name} param(t)   R²x={r2x:.3f}  R²y={r2y:.3f}')
        ax.legend()
        plot.show()

    def load_pos(self, kind=None, param=None):
        self._plot('x', 'y', kind, param)

    def load_mag(self, kind=None, param=None):
        self._plot('Magnitude', 'Direction', kind, param)
