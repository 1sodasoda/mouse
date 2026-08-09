"""Predict position from time plus the logged motion features.

Feature matrix = [t, Magnitude, Direction, ...]. Predicts on the real feature
rows (not a synthetic grid, since the extra features are only defined at
samples), so the fit is the per-sample predicted trajectory in order.
"""
from analysis.feature_extraction import feature_matrix
from ml.trajectory_fitting import _make_estimator


def fit_features(df, xcol, ycol, kind='rf', feature_cols=None):
    """Predict xcol, ycol from sample index t plus extra logged columns.

    Returns (x_pred, y_pred, r2x, r2y, used_cols).
    """
    X, cols = feature_matrix(df, feature_cols)
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)

    mx, my = _make_estimator(kind), _make_estimator(kind)
    mx.fit(X, x)
    my.fit(X, y)
    return (mx.predict(X), my.predict(X),
            float(mx.score(X, x)), float(my.score(X, y)), cols)
