# Technical Methodology

How mouse motion is recorded, filtered, modeled, and analyzed.

## Data capture
- Mouse motion recorded via `src/logger/mouse_logger.py` (pynput listener).
- Each sample logs position (`x`, `y`), velocity (`Magnitude`, `Direction`),
  a force-direction discrepancy, and a timestamp.
- Raw logs live under `data/raw/<gesture>/` (line, circle, zigzag, free).
- Shared read/timing helpers: `src/commons/mouse_logger.py`
  (`load_log`, `dt_from_times`).

## Processing
- **Filters** (`src/filters/`): moving average, exponential smoothing, 1-pole
  low-pass. Time-aware filters use the real per-sample dt from timestamps.
- **Kalman** (`src/kalman/`): `kalman_filter` (original) and
  `tuned_kalman_filter` (retuned — higher process noise + velocity seeding).
- **ML fits** (`src/ml/`): linear regression, random forest, parametric
  trajectory fitting (`x(t)`, `y(t)`), and temporal/feature-based prediction.
- **Metrics** (`src/analysis/`): velocity (speed/heading), motion metrics
  (max reach, reach retention, jitter), and feature-matrix extraction.

## Visualization
- `src/etc/grapher.py` (`Plotter`) applies any filter/model and plots it,
  including the `all` overlay comparison. Plots are saved under `plots/`.

## Reproducibility
- Environments: Nix (`flake.nix`, `nix develop`) or conda
  (`environment.yml`). Run the app with `cd src && python testmain1.py`.

## Related
- [Experimental Results](../results/experiment_summary.md)
- [Development Log](development_log.md)
