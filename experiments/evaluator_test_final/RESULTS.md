# Evaluator-test: final pipeline (test only)

The final pipeline (tuned HistGB + HMM/Viterbi, `artifacts/final/`) was run **once** on the 10 evaluator trips.
- Nothing was fitted, tuned or chosen on this data. The model and decoder settings were fixed beforehand, on train and student-test only.
- `.route` was read only after prediction, for scoring.
- The untuned model was run through the same pipeline **only for reporting**. The final model was not changed afterwards.

**Data:** 10 trips, 19,674 points, of which 17,899 are evaluable. Top-1 is point-weighted over evaluable points; route metrics are trip-weighted means.

## Summary

| Pipeline | Top-1 (classifier only) | Top-1 (with HMM) | Route F1 (with HMM) |
|---|---|---|---|
| **Final: tuned HistGB + HMM** | 0.9279 | **0.9616** | **0.9530** |
| Reference: untuned HistGB + HMM | 0.9475 | 0.9616 | 0.9411 |

Other final-pipeline metrics:

| Metric | Value |
|---|---|
| Route continuity | 0.967 |
| Route precision | 0.921 |
| Route recall | 0.994 |
| Normalised edit distance | 0.122 |
| Mean prediction time per trip (features + model + HMM) | 0.35 s |
| Trips where the HMM improved Top-1 | 7 of 10 |
| Trips where it was equal | 2 |
| Trips where it was worse | 1 (00000094: 0.984 → 0.977) |

## Per trip (final pipeline)

| Trip | Evaluable | Top-1 classifier | Top-1 HMM | Continuity | Route F1 | Norm. edit |
|---|---|---|---|---|---|---|
| 00000015 | 6131 | 0.838 | 0.921 | 0.999 | 0.766 | 0.612 |
| 00000017 | 1301 | 0.957 | 0.972 | 0.887 | 0.953 | 0.087 |
| 00000023 | 449 | 0.953 | 0.982 | 0.991 | 0.922 | 0.169 |
| 00000030 | 610 | 0.856 | 0.895 | 0.916 | 0.954 | 0.091 |
| 00000049 | 2625 | 0.997 | 0.999 | 0.994 | 0.999 | 0.002 |
| 00000059 | 1003 | 0.979 | 0.989 | 0.931 | 0.971 | 0.083 |
| 00000067 | 711 | 0.990 | 0.990 | 0.991 | 0.988 | 0.012 |
| 00000073 | 1127 | 0.991 | 0.991 | 0.991 | 0.996 | 0.036 |
| 00000087 | 1386 | 0.968 | 0.998 | 0.989 | 0.999 | 0.002 |
| 00000094 | 2556 | 0.984 | 0.977 | 0.984 | 0.983 | 0.130 |

## Findings
1. **The HMM is the main gain on unseen data:** Top-1 0.928 → 0.962, matching what we saw on student-test.
2. **The tuning gain did not carry over to the classifier alone.** Tuned 0.928 vs untuned 0.948 here, although tuning improved CV (0.930 → 0.937) and student-test (0.933 → 0.937).
   - Almost all of the gap comes from trip 00000015 (6,131 evaluable points, 34 % of the total), where the tuned classifier falls from 0.901 to 0.838.
   - The tuned model is larger: 239 leaves, minimum leaf size 14. It probably fits the train trips' conditions more closely, so it generalises less well to an unusual trip.
3. **With the HMM both models reach the same Top-1 (0.9616)**, and the tuned pipeline has the better route F1 (0.953 vs 0.941). The decoder absorbs most of the classifier's per-point errors.
4. **Trip 00000015 is still the hardest trip for route reconstruction** (route F1 0.77, edit distance 0.61) even though Top-1 is 0.92. *[Inspect it on a map: the spurious route segments are likely detours inserted between consecutive chosen arcs.]*

Files: `per_trip.csv` (final), `per_trip_untuned_reference.csv`, `predictions_<trip>.csv` (HMM and classifier arc per point), `run_info.json`.
