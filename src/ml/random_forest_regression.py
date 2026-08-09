"""Random-forest regression of one column on another."""


def fit_forest(df, xcol, ycol, n_estimators=100, max_depth=5,
               min_samples_leaf=20):
    """Random-forest regression of ycol on xcol via scikit-learn.

    Nonlinear, so it can follow curved paths a straight line can't.
    max_depth / min_samples_leaf are regularizers: with the defaults each
    leaf averages >=20 points and trees stay shallow, so the fit is smooth
    instead of memorizing the training points (unlimited depth overfits
    badly on this data). Returns (r2, model); random_state fixed.
    """
    from sklearn.ensemble import RandomForestRegressor

    x = df[[xcol]].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=0).fit(x, y)
    return float(model.score(x, y)), model
