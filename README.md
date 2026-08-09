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

## Reflections

Weekly notes live under `docs/`:

- Week 0 — [Initial thoughts](docs/week0/initial_thoughts.md): the rapid-trigger idea and aim/pivot observations.
- Week 1 — [First logger & observations](docs/week1/observations.md): primitive logger/grapher; data cleaner than expected; circle shakier than flicks.
- Week 2 — [Dedicated logger](docs/week2/dedicated_logger.md): decision to build a dedicated logger.
- Week 3 — [Report](docs/week3/2026-08-09.md): filters, ML experiments, Kalman retuning, and the package refactor.

## Documentation

- [Technical Methodology](docs/methodology.md)
- [Experimental Results](results/experiment_summary.md)
- [Development Log](docs/development_log.md)
