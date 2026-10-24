"""A small GRU that predicts the next displacement from a window of deltas.

The project's earlier models (``linear_regression``, ``random_forest_regression``,
``trajectory_fitting``, ``temporal_prediction``) fit a static function to
independent rows, so they could only reconstruct a recorded path. A recurrent
net carries state across the window, which is what ``docs/ml_analysis.md``
concluded was needed for actual prediction.

Deliberately small (one GRU layer, 32 hidden units): the dataset is a handful
of short sessions, so a bigger net would only memorise them.
"""
import torch
from torch import nn


class DeltaGRU(nn.Module):
    """Window of per-step deltas -> displacement over the next horizon.

    ``in_dim`` is 3: the dx/dy deltas plus the polling-rate channel described
    in ``ml/windowing.py``, so the net can tell real fine-grained motion from
    motion interpolated up from a slower recording.
    """

    def __init__(self, in_dim=3, hidden=32, layers=1, out_dim=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.gru(x)        # (B, window, hidden)
        return self.head(out[:, -1])  # last step's state -> (B, 2)


def pick_device():
    """MPS on Apple silicon when available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeltaLSTM(nn.Module):
    """Same shape as DeltaGRU but with LSTM cells (separate cell state)."""

    def __init__(self, in_dim=3, hidden=32, layers=1, out_dim=2):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class DeltaMLP(nn.Module):
    """Flatten the window and regress -- no recurrence, but it sees all steps.

    Worth comparing: if an MLP matches the RNNs, the task does not actually
    need sequential state, only the whole window at once.
    """

    def __init__(self, in_dim=3, hidden=64, window=20, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeltaCNN(nn.Module):
    """Temporal 1-D convolutions over the window, then take the last step."""

    def __init__(self, in_dim=3, hidden=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))     # (B, hidden, window)
        return self.head(h[:, :, -1])


class DeltaLinear(nn.Module):
    """One linear layer on the flattened window.

    The reference point that matters: constant velocity is itself a linear
    function of the last delta, so this model can represent the baseline
    exactly. Anything that cannot beat it is not learning structure.
    """

    def __init__(self, in_dim=3, window=20, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(window * in_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class ResidualWrapper(nn.Module):
    """Predict the *correction* to constant velocity instead of the motion.

    The wrapped model starts from the constant-velocity prediction (last
    observed delta carried forward ``horizon`` steps) and only has to learn
    how the real motion departs from it. Inputs are normalised by the same
    pixel scale as the targets, so the baseline can be computed inside the
    network from the delta channels.
    """

    def __init__(self, inner, horizon):
        super().__init__()
        self.inner = inner
        self.horizon = horizon

    def forward(self, x):
        const_vel = x[:, -1, 0:2] * self.horizon
        return const_vel + self.inner(x)


ARCHITECTURES = ('linear', 'mlp', 'cnn', 'lstm', 'gru', 'gru_res', 'mlp_res')


def make_model(name, hidden=32, window=20, horizon=5, in_dim=3):
    """Build one of ARCHITECTURES by name."""
    base = name[:-4] if name.endswith('_res') else name
    if base == 'gru':
        inner = DeltaGRU(in_dim=in_dim, hidden=hidden)
    elif base == 'lstm':
        inner = DeltaLSTM(in_dim=in_dim, hidden=hidden)
    elif base == 'mlp':
        inner = DeltaMLP(in_dim=in_dim, hidden=max(hidden, 64), window=window)
    elif base == 'cnn':
        inner = DeltaCNN(in_dim=in_dim, hidden=hidden)
    elif base == 'linear':
        inner = DeltaLinear(in_dim=in_dim, window=window)
    else:
        raise ValueError(f"unknown architecture '{name}', "
                         f"pick from {list(ARCHITECTURES)}")
    if name.endswith('_res'):
        return ResidualWrapper(inner, horizon)
    return inner


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
