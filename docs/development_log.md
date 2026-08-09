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

## See also
- [Technical Methodology](methodology.md)
- [Experimental Results](../results/experiment_summary.md)
