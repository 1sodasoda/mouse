"""Plotting front-end: applies filters / ML fits and draws them.

This module wires together the split packages (filters, kalman, ml) behind the
same ``Plotter`` API used before the refactor.
"""
import matplotlib.pyplot as plot
import numpy as np

from commons.mouse_logger import load_log
from filters.moving_average import moving_average
from filters.exponential_smoothing import exponential_smoothing
from filters.low_pass_filter import low_pass
from kalman.kalman_filter import kalman_filter
from kalman.tuned_kalman_filter import tuned_kalman_filter
from ml.linear_regression import fit_line
from ml.random_forest_regression import fit_forest
from ml.trajectory_fitting import fit_parametric
from ml.temporal_prediction import fit_features


# Registry: name -> (function, param name, default)
FILTERS = {
    'ma': (moving_average, 'window', 5),
    'ema': (exponential_smoothing, 'alpha', 0.2),
    'lpf': (low_pass, 'cutoff', 5.0),
    'kf': (kalman_filter, 'meas_var', 50.0),        # original (over-smooths turns)
    'kf2': (tuned_kalman_filter, 'q', 5.0),         # retuned (reaches turn extremes)
}

# filters that consume the per-sample timestamps for real dt
_TIME_AWARE = ('lpf', 'kf', 'kf2')


class Plotter:

    def __init__(self, file):
        self.file = file
        self.df = load_log(file)

    def _apply(self, columns, kind=None, param=None):
        """Return a copy of df with the given columns filtered.

        kind: one of FILTERS keys, or None for raw.
        param: override the filter's parameter (window / alpha / cutoff / ...).
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

    def _plot_lr(self, xcol, ycol):
        """Scatter the raw points and overlay the best-fit regression line."""
        m, b, r2, model = fit_line(self.df, xcol, ycol)
        ax = self.df.plot.scatter(x=xcol, y=ycol, color='red', alpha=0.3,
                                  s=6, label=f'{ycol} (raw)')
        xs = np.linspace(self.df[xcol].min(), self.df[xcol].max(), 200)
        ys = model.predict(xs.reshape(-1, 1))
        ax.plot(xs, ys, color='black', lw=2,
                label=f'fit: y = {m:.3f}x + {b:.1f}   R² = {r2:.3f}')
        ax.legend()
        plot.show()

    def _plot_rf(self, xcol, ycol):
        """Scatter the raw points and overlay the random-forest prediction."""
        r2, model = fit_forest(self.df, xcol, ycol)
        ax = self.df.plot.scatter(x=xcol, y=ycol, color='red', alpha=0.3,
                                  s=6, label=f'{ycol} (raw)')
        xs = np.linspace(self.df[xcol].min(), self.df[xcol].max(), 400)
        ys = model.predict(xs.reshape(-1, 1))
        ax.plot(xs, ys, color='black', lw=2,
                label=f'random forest   R² = {r2:.3f}')
        ax.legend()
        plot.show()

    def _plot_param(self, xcol, ycol, kind):
        """Scatter raw points and overlay the parametric (index-based) fit."""
        xg, yg, r2x, r2y = fit_parametric(self.df, xcol, ycol, kind)
        ax = self.df.plot.scatter(x=xcol, y=ycol, color='red', alpha=0.3,
                                  s=6, label=f'{ycol} (raw)')
        name = 'linear' if kind == 'lr' else 'random forest'
        ax.plot(xg, yg, color='black', lw=2,
                label=f'{name} param(t)   R²x={r2x:.3f}  R²y={r2y:.3f}')
        ax.legend()
        plot.show()

    def _plot_features(self, xcol, ycol, kind):
        """Scatter raw points and overlay the feature-based predicted path."""
        xg, yg, r2x, r2y, cols = fit_features(self.df, xcol, ycol, kind)
        ax = self.df.plot.scatter(x=xcol, y=ycol, color='red', alpha=0.3,
                                  s=6, label=f'{ycol} (raw)')
        name = 'linear' if kind == 'lr' else 'random forest'
        ax.plot(xg, yg, color='black', lw=1.5,
                label=f"{name} [{'+'.join(cols)}]  R²x={r2x:.3f} R²y={r2y:.3f}")
        ax.legend()
        plot.show()

    def load_pos(self, kind=None, param=None):
        self._plot('x', 'y', kind, param)

    def load_mag(self, kind=None, param=None):
        self._plot('Magnitude', 'Direction', kind, param)
