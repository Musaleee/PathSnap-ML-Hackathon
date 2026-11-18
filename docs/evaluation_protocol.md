# Evaluation protocol

Implemented in `src/pathsnap/evaluation.py`.

| Level | Unit | Metric | Denominator / notes |
|---|---|---|---|
| Candidate | (point, candidate) pair | precision, recall, F1 of the positive class; **macro-F1 = mean F1 over the two classes {correct, incorrect}**; log loss | Non-ambiguous points only. Threshold 0.5. This is pair-level, not arc-level accuracy |
| Observation | GPS point | **Top-1** = highest-scoring candidate equals the aligned reference arc | `top1_evaluable`: non-ambiguous points whose reference arc is among the candidates (main metric). `top1_non_ambiguous`: all non-ambiguous points (misses count as wrong). Per-trip mean and minimum also logged |
| Candidate miss rate | GPS point | reference arc not in candidate set | Reported per trip in `experiments/results/labels_candidates_*.csv` |
| Route | trip | **continuity** = share of consecutive observations whose chosen arcs are the same arc or share a node | Point pairs across a missing observation are skipped. Sequence-overlap metric to be added with the HMM |
| Cost | run | fit time, inference time per trip | Environment stored in each log |

Aggregation: Top-1 and continuity are point-weighted over all trips; per-trip values are also kept. CV values are the mean ± std over 5 trip-grouped folds.

Arc ids are trip-local, so no global multiclass metric over raw arc ids is used.
