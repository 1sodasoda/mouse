"""Exponential smoothing filter."""


def exponential_smoothing(series, alpha=0.2):
    """Exponential smoothing: y[n] = alpha*x[n] + (1-alpha)*y[n-1].

    alpha in (0, 1]. Smaller alpha -> smoother, more lag.
    """
    return series.ewm(alpha=alpha, adjust=False).mean()
