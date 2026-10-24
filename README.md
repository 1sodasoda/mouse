# Mouse Movement Analysis

Records mouse motion, filters/models it, and analyzes aim behavior.

## Installation

Pick one environment — Nix or conda (both pull the same deps: pandas,
matplotlib, scikit-learn, pynput).

**Nix:**
```
nix develop
```

**conda:**
```
conda env create -f environment.yml
conda activate mouse
```

## Usage

```
cd src
python testmain1.py
```

Then follow the prompts: **log** to record a session (saved to
`data/raw/free/`), or **graph** to pick a log and view it with any
filter/model. macOS needs Accessibility permission for the terminal so pynput
can read the pointer.

### Collecting labelled data

```
cd src
python data_collector.py
```

It names a gesture, you perform it, press Enter, and the sample is saved to
`data/raw/<gesture>/`. Recording is quiet by default; pass `verbose=True` to
`Logger` to echo every sample while debugging.

Holding the pointer still auto-pauses: after the idle split (300 ms by default,
set at the prompt, 0 to disable) the current segment closes and the next
movement starts `..._part2.csv`. Each saved file is then one stretch of
continuous motion, with no pause baked into it and no velocity measured across
the pause.

### Training the sequence model

```
cd src
python train.py
```

Loads every usable log, splits **by session file** (so no test window shares a
recording with a training window), trains a small GRU to predict the next
displacement from a window of recent deltas, and reports held-out error in
pixels next to trivial baselines.

Recordings do not share a polling rate (~1000 Hz for the early logs, ~40-160 Hz
from the collector), so every session is resampled onto a fixed real-time grid
and its native rate is fed to the model as a feature. The horizon therefore
means the same number of milliseconds for every session.

Options: `--step-ms --max-gap-ms --window --horizon --epochs --hidden --seed
--patience --no-rate`.

## Reflections

Weekly notes live under `docs/`:

- Week 0 — [Initial thoughts](docs/week0/initial_thoughts.md): the rapid-trigger idea and aim/pivot observations.
- Week 1 — [First logger & observations](docs/week1/observations.md): primitive logger/grapher; data cleaner than expected; circle shakier than flicks.
- Week 2 — [Dedicated logger](docs/week2/dedicated_logger.md): decision to build a dedicated logger.
- Week 3 — [Report](docs/week3/2026-08-09.md): filters, ML experiments, Kalman retuning, and the package refactor.
- Week 4 — [Report](docs/week4/2026-10-18.md): PyTorch, sequence models, the dataset/windowing pipeline, and the first model to beat constant velocity on held-out recordings.

## Documentation

- [Technical Methodology](docs/methodology.md)
- [Experimental Results](results/experiment_summary.md)
- [Development Log](docs/development_log.md)
