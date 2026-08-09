"""Parametric trajectory fitting: x and y each as a function of sample index.

A trajectory is parametric: x = f(t), y = g(t). Regressing against the index t
(instead of y on x) gives one (x, y) per t, so multivalued paths like a circle
or crisscrossing flicks fit correctly.
"""
import numpy as np

from analysis.feature_extraction import time_index


def _make_estimator(kind):
    if kind == 'lr':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, max_depth=8,
                                 min_samples_leaf=5, random_state=0)


def fit_parametric(df, xcol, ycol, kind='rf'):
    """Fit xcol and ycol each as a function of sample index (parametric).

    kind: 'lr' (straight-line param) or 'rf' (nonlinear, traces curves).
    Returns (x_pred, y_pred, r2x, r2y) over a dense t grid.
    """
    n = len(df)
    t = time_index(df)
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    mx, my = _make_estimator(kind), _make_estimator(kind)
    mx.fit(t, x)
    my.fit(t, y)
    tg = np.linspace(0, n - 1, min(2000, 2 * n)).reshape(-1, 1)
    return (mx.predict(tg), my.predict(tg),
            float(mx.score(t, x)), float(my.score(t, y)))
