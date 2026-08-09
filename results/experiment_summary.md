# Experimental Results

Summary of the filter and machine-learning experiments. Full write-ups are in
`docs/`; the raw comparison tables sit next to this file.

## Data tables
- [model_comparison.csv](model_comparison.csv) — R² per model × log.
- [movement_comparison.csv](movement_comparison.csv) — reach % and jitter %
  (of raw) per filter × log.

## Filter reach at turns (from movement_comparison.csv)

Whether each filter still reaches the extreme turn positions (100% = preserved):

| Log | ma | ema | lpf | kf | kf2 |
|-----|----|-----|-----|----|-----|
| circle | 99.9% | 99.9% | 97.7% | 87.1% | 104.3% |
| uturn2 | 99.9% | 100% | 100% | 100% | 100% |
| flick | 99.9% | 99.7% | 98.6% | 81.1% | 101.1% |

Retuning the Kalman filter (`kf` → `kf2`) fixed the turn undershoot.

## ML fit quality (from model_comparison.csv)

R² on `x` (parametric/feature models also fit `y`):

| Model | circle | uturn2 | flick |
|-------|--------|--------|-------|
| linear `y~x` | 0.04 | 0.90 | 0.00 |
| random forest `y~x` | 0.13 | 0.90 | 0.30 |
| rf parametric `t` | 1.00 | 1.00 | 0.84 |
| rf + features | 1.00 | 1.00 | 0.69 |

## Deep-dive write-ups
- [Filter comparison](../docs/results.md)
- [ML regression results](../docs/ml_results.md)
- [ML analysis — why prediction fails](../docs/ml_analysis.md)
- [Kalman kf1 vs kf2](../docs/kf1_vs_kf2.md)
