# Quick Analysis: Why Linear Regression and Random Forest Fail to Predict Mouse Movement

Both `LinearRegression` and `RandomForestRegressor` were tried on the `circle`,
`uturn2`, and `flicktopositions` logs. Neither reliably predicts mouse
movement. Below is why — the failures are structural, not a tuning problem.

## 1. Predicting `y` from `x` is ill-posed

The first framing regressed `y` on `x`. A regressor must return **one** `y`
per `x`, but a mouse path is not a function of `x`:

| Log | linear `y~x` R² | random forest `y~x` R² |
|-----|-----------------|------------------------|
| circle | 0.037 | 0.134 |
| uturn2 | 0.899 | 0.904 |
| flicktopositions | 0.002 | 0.301 |

- **circle / flick** revisit the same `x` with many different `y` values, so
  the model averages them into a meaningless line through the middle.
- **uturn2** only scores well because it happens to be nearly monotonic in `x`
  — an accident of that shape, not real predictive skill.

## 2. Parametric fitting works, but it is interpolation, not prediction

Reframing as `x = f(t)`, `y = g(t)` (index-based) removes the multivalued
problem and the random forest traces every path (R² ≈ 1.0). But this is
**memorizing the recorded curve**, not predicting movement:

- R² is measured on the same samples the model trained on — no held-out data.
- A tree keyed on the sample index cannot say anything about a `t` it never
  saw; it only interpolates between known points.
- It reconstructs a path that already happened. It does not forecast where the
  mouse goes next, which is what "predicting mouse movement" actually means.

## 3. The logged motion features do not help

Adding `Magnitude` and `Direction` as inputs (`[t, Magnitude, Direction]`)
did not improve the fit and made `flick` worse:

| Log | `t` only R²x | `[t, Mag, Dir]` R²x |
|-----|--------------|---------------------|
| circle | 1.000 | 1.000 |
| uturn2 | 1.000 | 1.000 |
| flicktopositions | 0.838 | **0.690** |

`Magnitude` and `Direction` are the path's *derivative* (per-step velocity in
polar form); integrating them reconstructs `x,y` to within ~1.6 px. But a
memoryless regressor sees one row at a time and cannot integrate, so the same
speed and heading occur at many different positions. As inputs they are noise
for absolute-position prediction, so the forest wastes splits on them.

## Root cause

Both algorithms fit a **static function** to independent rows. Mouse movement
is a **temporal, stateful, human-driven** process:

- **No memory.** Position depends on the accumulated history of motion; these
  models have no state to accumulate.
- **Not deterministic.** Human input is noisy and intention-driven; the same
  local conditions lead to different next moves.
- **Fitting ≠ forecasting.** At best these models interpolate an
  already-recorded trajectory. They cannot extrapolate future movement.

## Takeaway

The two algorithms can *smooth or reconstruct* a known path, but they cannot
*predict* mouse movement. Doing that needs a model with memory and a temporal
target — e.g. predicting the next step from a window of recent steps, with a
sequence model (RNN/LSTM) or an explicit state-space filter (Kalman) — plus an
honest train/test split to measure real generalization.
