"""Turn logged sessions into a supervised next-movement prediction problem.

Why not "predict the next row": the raw logs are sampled as fast as the
pointer reports, so the next sample sits ~1 px away and repeating the previous
row already solves it. The target is therefore the displacement a fixed number
of steps ahead, over a window of recent motion.

**Mixed polling rates.** Recordings in ``data/raw/`` do not share a sample
rate: the early logs are ~1000 Hz (median step ~1-5 px) while sessions from
``data_collector.py`` come in at ~40-160 Hz (median step ~20-35 px). Striding
by row index would make one step mean 1 ms in one session and 25 ms in
another, so ``window`` and ``horizon`` would denote a different amount of real
time per session and the learning problem would not be well defined.

Every session is instead resampled onto a uniform **time** grid of
``STEP_MS`` milliseconds taken from the log's own timestamps, so one step is
the same duration everywhere and ``horizon`` steps is always
``horizon * STEP_MS`` ms ahead.

Resampling alone would hide how the data was captured: a 100 Hz log
interpolated onto a 4 ms grid looks like fine-grained motion that was never
measured. Each step therefore carries a third channel, ``log2(native_dt /
STEP_MS)`` -- negative when the session was sampled finer than the grid (real
detail, downsampled), positive when it was coarser (detail interpolated in).
The model sees the polling rate its input came from and can treat the two
regimes differently.

Inputs are **deltas**, never absolute positions, so a model cannot key on
where on the screen a gesture happened -- it has to use the shape of the
recent motion. This is the missing ingredient identified in
``docs/ml_analysis.md``: the earlier regressors were memoryless, these windows
hand the model ``WINDOW`` steps of history.
"""
import numpy as np

from commons.mouse_logger import POS_COLS, TIME_COL, dt_from_times

STEP_MS = 4.0       # one step of the resampling grid (ms of real time)
WINDOW = 20         # steps of history fed to the model (~80 ms at STEP_MS=4)
HORIZON = 5         # predict displacement this many steps ahead (~20 ms)
MAX_GAP_MS = 50.0   # drop windows spanning a native gap longer than this
MIN_MOVE_PX = 2.0   # a window whose history moved less than this is "still"

# X channels: 0,1 are the dx/dy deltas; 2 is the polling-rate channel.
DELTAS = slice(0, 2)
RATE = 2


def session_times(df):
    """Seconds since the session start, from the log's own timestamps."""
    dt = dt_from_times(df[TIME_COL], len(df))
    t = np.cumsum(dt)
    return t - t[0]


def native_dt(df):
    """Median time between samples (seconds) -- the session's polling period."""
    dt = dt_from_times(df[TIME_COL], len(df))
    return float(np.median(dt[1:])) if len(dt) > 1 else float(dt[0])


def native_hz(df):
    """Session polling rate in Hz, for reporting."""
    d = native_dt(df)
    return 1.0 / d if d > 0 else float("nan")


def resample(df, step_ms=STEP_MS):
    """Positions on a uniform time grid, plus the native gap at each point.

    Returns (pos, gap_ms) where pos has shape (n, 2) sampled every step_ms of
    real time, and gap_ms[i] is the length of the native sampling interval the
    grid point fell inside. gap_ms exposes pauses: a grid point landing in a
    1.2 s pause is interpolated across motion that was never recorded, so
    ``session_windows`` can drop it.
    """
    t = session_times(df)
    pos_native = df[list(POS_COLS)].to_numpy(dtype=float)
    step_s = step_ms / 1000.0
    if t[-1] < step_s:
        return np.empty((0, 2)), np.empty(0)

    grid = np.arange(0.0, t[-1], step_s)
    pos = np.column_stack([np.interp(grid, t, pos_native[:, 0]),
                           np.interp(grid, t, pos_native[:, 1])])

    # native interval containing each grid point
    idx = np.clip(np.searchsorted(t, grid, side="right"), 1, len(t) - 1)
    gap_ms = (t[idx] - t[idx - 1]) * 1000.0
    return pos, gap_ms


def session_windows(df, window=WINDOW, horizon=HORIZON, step_ms=STEP_MS,
                    max_gap_ms=MAX_GAP_MS):
    """Sliding windows for one session, on the uniform time grid.

    Returns (X, Y) with X shape (m, window, 3) -- the last ``window`` deltas
    plus the polling-rate channel -- and Y shape (m, 2), the displacement from
    the window's last position to the position ``horizon`` steps later.
    Windows spanning a native gap longer than ``max_gap_ms`` are dropped.
    """
    pos, gap_ms = resample(df, step_ms)
    n = len(pos)
    empty = (np.empty((0, window, 3)), np.empty((0, 2)))
    if n < window + horizon + 1:
        return empty

    deltas = np.diff(pos, axis=0)
    rate = np.log2(max(native_dt(df) * 1000.0, 1e-3) / step_ms)

    xs, ys = [], []
    for t_end in range(window, n - horizon):
        # grid points touched by this window's history and its target
        if gap_ms[t_end - window:t_end + horizon + 1].max() > max_gap_ms:
            continue
        hist = deltas[t_end - window:t_end]
        xs.append(np.column_stack([hist, np.full(window, rate)]))
        ys.append(pos[t_end + horizon] - pos[t_end])

    if not xs:
        return empty
    return np.stack(xs), np.stack(ys)


def build(sessions, window=WINDOW, horizon=HORIZON, step_ms=STEP_MS,
          max_gap_ms=MAX_GAP_MS):
    """Stack windows from many sessions.

    Returns (X, Y, groups) where groups[i] is the index of the session window
    i came from -- kept so per-session or per-gesture error can be reported.
    """
    xs, ys, groups = [], [], []
    for i, s in enumerate(sessions):
        X, Y = session_windows(s.df, window, horizon, step_ms, max_gap_ms)
        if len(X) == 0:
            continue
        xs.append(X)
        ys.append(Y)
        groups.append(np.full(len(X), i))
    if not xs:
        return (np.empty((0, window, 3)), np.empty((0, 2)),
                np.empty(0, dtype=int))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def window_motion(X):
    """Distance the pointer travelled during each window's history, in px.

    Measured from the input deltas only -- never from the target. Selecting
    windows on the target would be choosing test cases by their answer; the
    history is information a live predictor also has, so filtering on it is a
    decision that can actually be made at inference time ("only predict while
    the pointer is moving").
    """
    d = np.asarray(X, dtype=float)[:, :, DELTAS]
    return np.hypot(d[:, :, 0], d[:, :, 1]).sum(axis=1)


def moving_mask(X, min_move_px=MIN_MOVE_PX):
    """True where the window's history shows real motion, not idle jitter."""
    return window_motion(X) >= min_move_px


def fit_scale(X):
    """Pixel scale used to normalise the delta channels and the targets.

    One scalar (the RMS delta of the training windows) rather than per-axis
    stats, so x and y stay on the same footing and a predicted delta can be
    turned straight back into pixels. The rate channel is excluded -- it is
    already in log units. Fitted on **train only**; passing val or test data
    in here would leak.
    """
    d = np.asarray(X, dtype=float)[:, :, DELTAS]
    scale = float(np.sqrt((d ** 2).mean()))
    return scale if scale > 0 else 1.0


def normalise(X, scale):
    """Divide the delta channels by ``scale``, leaving the rate channel as-is."""
    out = np.asarray(X, dtype=float).copy()
    out[:, :, DELTAS] /= scale
    return out
