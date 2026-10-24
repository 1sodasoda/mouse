# Development Log

Chronological record of development. Weekly reflections (in the author's voice)
are linked per entry.

## Week 1 — First tools
- Built a primitive logger and grapher.
- Observation: data cleaner than expected; on_move fires ~every 0.001 s
  (~1000 Hz polling); circle test shakier than flicks.
- Reflection: [week1/observations.md](week1/observations.md).

## Week 2 — Dedicated logger
- Decided to build a dedicated logger.
- Reflection: [week2/dedicated_logger.md](week2/dedicated_logger.md).

## Week 3 — Filters, ML, Kalman, refactor
- **Filters** added: moving average, exponential smoothing, 1-pole low-pass
  (time-aware via real dt).
- **Comparison views**: raw overlay, then an `all` view layering every filter
  with distinct widths/styles so overlapping curves stay visible.
- **Kalman**: added `kf`; found it undershot sharp turns; retuned to `kf2`
  (process noise q 1→5, initial-velocity seeding) which reaches turn extremes.
- **ML**: linear regression and random forest, in three framings — `y~x`,
  parametric on sample index `t`, and `t` + logged motion features. Regularized
  the forest (max_depth, min_samples_leaf) to stop overfitting.
- **Analysis**: reach/jitter metrics; concluded these models reconstruct but do
  not predict mouse movement.
- **Refactor**: split the monolithic grapher into packages (`filters/`,
  `kalman/`, `ml/`, `analysis/`, `commons/`, `logger/`, `etc/`); reorganized
  into `data/raw/<gesture>/`, `plots/`, `results/`, `docs/week*`.
- **Environments**: added conda (`environment.yml`) alongside Nix.
- Report: [week3/2026-08-09.md](week3/2026-08-09.md).

## Week 4 — Deep learning: sequence models and an honest experiment
- **Environment**: torch 2.12.0 under Nix (MPS available); `environment.yml`
  still missing torch.
- **Dataset pipeline** (`commons/dataset.py`): discovers every log under
  `data/raw/`, tags gesture, rejects unusable logs with a reason, and splits
  whole **session files** into train/val/test (a row-level split leaks, since
  neighbouring rows are ~1 ms apart).
- **Windowing** (`ml/windowing.py`): resamples each session onto a fixed
  real-time grid so one step means the same milliseconds regardless of polling
  rate (recordings range 83–1133 Hz), feeds deltas plus a polling-rate channel,
  targets displacement a fixed time ahead, and drops windows spanning pauses.
- **Models** (`ml/sequence_model.py`): GRU, LSTM, MLP, CNN, linear, plus a
  residual-from-constant-velocity wrapper. Baselines in `ml/baselines.py`.
- **Runners**: `train.py` (single run, per-session breakdown) and
  `compare_models.py` (all architectures on identical tensors across seeds).
- **Result**: a window of recent motion beats constant velocity on 3/3 splits
  — GRU 5.36 px vs 7.02 px mean error, 20 ms ahead. GRU and LSTM are
  indistinguishable; the ranking flips with batch size.
- **Collector**: recording is quiet by default (`verbose` flag) and auto-pauses
  at idle, splitting each take into continuous-motion segments.
- **Bugs found at scale**: `str(datetime.now())` omits microseconds when they
  are exactly zero (broke parsing at 2M rows); the first sample after a pause
  carried a velocity measured across the whole pause.
- **Biggest lesson**: 54% of recorded sessions were motionless, which made
  "predict zero" score a median error of 0.00 px and rendered the metric
  unable to distinguish models. Data and target framing decided the outcome;
  architecture was worth ~9%.
- Report: [week4/2026-10-18.md](week4/2026-10-18.md).

## See also
- [Technical Methodology](methodology.md)
- [Experimental Results](../results/experiment_summary.md)
