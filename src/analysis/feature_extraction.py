"""Feature-matrix construction for the ML models."""
import numpy as np

# Extra logged columns fed to the feature-based fits, when present.
EXTRA_FEATURES = ('Magnitude', 'Direction')


def time_index(df):
    """Sample-index parameter t = 0..n-1 as a column vector."""
    return np.arange(len(df), dtype=float).reshape(-1, 1)


def feature_matrix(df, feature_cols=None):
    """Build [t, <extra features>] for predicting position.

    feature_cols defaults to whichever of EXTRA_FEATURES exist in df.
    Returns (X, used_cols) where used_cols[0] == 't'.
    """
    if feature_cols is None:
        feature_cols = [c for c in EXTRA_FEATURES if c in df.columns]
    t = np.arange(len(df), dtype=float)
    X = np.column_stack(
        [t] + [df[c].to_numpy(dtype=float) for c in feature_cols])
    return X, ['t'] + list(feature_cols)
