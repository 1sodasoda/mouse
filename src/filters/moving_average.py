"""Simple moving-average filter."""


def moving_average(series, window=5):
    """Simple moving average over a fixed window (centered)."""
    return series.rolling(window=window, min_periods=1, center=True).mean()
