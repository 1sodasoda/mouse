"""Least-squares linear regression of one column on another."""


def fit_line(df, xcol, ycol):
    """Least-squares linear fit of ycol on xcol via scikit-learn.

    Returns (slope, intercept, r2, model).
    """
    from sklearn.linear_model import LinearRegression

    x = df[[xcol]].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    model = LinearRegression().fit(x, y)
    return (float(model.coef_[0]), float(model.intercept_),
            float(model.score(x, y)), model)
