# Evaluator-test results (test only)

The 7 pre-trained models (trained on the 80 train trips only) were scored once on the 10 evaluator trips.
Nothing was trained or tuned on this data. The `.route` files were used only to make reference labels for scoring.
Reproduce with: `PYTHONPATH=src python -m pathsnap.evaluate_holdout <evaluator_test folder>`

**Data:** 10 trips, 19,674 GPS points, 146,494 candidate rows. 9.0% of points are ambiguous. The true road is among the candidates for 99.9% of evaluable points.
**Checks:** no data problems, apart from 2 self-loop arcs in trip 00000087. None of these trip ids appear in train or student_test.

## Models (sorted by Top-1)

| Model | Top-1 | Macro-F1 | Log loss | Continuity | Worst trip |
|---|---|---|---|---|---|
| **HistGradientBoosting** | **0.9483** | **0.9665** | **0.0495** | 0.9705 | 0.857 |
| Random Forest | 0.9479 | 0.9657 | 0.0513 | 0.9668 | 0.867 |
| MLP | 0.9466 | 0.9611 | 0.0517 | 0.9712 | 0.868 |
| Extra Trees | 0.9354 | 0.9534 | 0.0721 | 0.9688 | 0.862 |
| Logistic Regression | 0.9281 | 0.9423 | 0.0849 | 0.9575 | 0.847 |
| KNN | 0.9173 | 0.9419 | 0.0840 | 0.9721 | 0.830 |
| Gaussian NB | 0.8952 | 0.8639 | 0.6133 | 0.9735 | 0.773 |
| *Rule: nearest road* | 0.5080 | – | – | 0.9402 | 0.381 |
| *Rule: nearest road in the same direction* | 0.8636 | – | – | 0.9543 | 0.717 |

## Comparison with earlier stages (Top-1)

| Model | CV (train) | Student-test | Evaluator-test |
|---|---|---|---|
| HistGB | 0.930 | 0.933 | 0.948 |
| Random Forest | 0.930 | 0.932 | 0.948 |
| MLP | 0.931 | 0.927 | 0.947 |

The ranking is the same at all three stages, and HistGB, the model chosen before this test, is still best. Evaluator scores are a bit higher, probably because these trips are easier: fewer ambiguous points (9% here against 11% on student-test and 18% on train).

## HistGB per trip

| Trip | Evaluable points | Top-1 | Continuity |
|---|---|---|---|
| 00000015 | 6131 | 0.901 | 0.990 |
| 00000017 | 1291 | 0.964 | 0.873 |
| 00000023 | 448 | 0.960 | 0.974 |
| 00000030 | 607 | 0.857 | 0.879 |
| 00000049 | 2625 | 0.992 | 0.992 |
| 00000059 | 1002 | 0.978 | 0.919 |
| 00000067 | 711 | 0.987 | 0.987 |
| 00000073 | 1127 | 0.990 | 0.986 |
| 00000087 | 1386 | 0.958 | 0.964 |
| 00000094 | 2556 | 0.981 | 0.979 |

Trips 30 and 15 are the weakest. Trips 17 and 30 have the lowest continuity, which is where the HMM should help most.

## Files
- `summary.csv`: all metrics for each model
- `baselines.csv`: the two rule-based baselines
- `per_trip_<model>.csv`, `per_trip_all_models.csv`: results per trip
- `predictions/<model>/<trip>.csv`: the chosen arc and score for each point, and whether it was correct
- `data_checks.csv`, `labels_candidates.csv`: data checks, and label and candidate statistics
- `run_info.json`: configuration, environment and timestamp

Caveat: Top-1 counts only evaluable points (not ambiguous, with the true road among the candidates), which is 90.9% of points.
