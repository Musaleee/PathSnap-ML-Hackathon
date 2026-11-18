# Decision log

Format: id · status (DECIDED / OPEN / REVISIT) · decision · alternatives · reason · evidence.
**The team must review every DECIDED item, change what it disagrees with, and add who signed it off.**
Initial entries were drafted with AI assistance (Claude) on 2026-11-18.

| ID | Status | Decision | Alternatives | Reason / evidence |
|---|---|---|---|---|
| D-01 | DECIDED | Loaders split on any whitespace, no header; ids = 0-based row index | Tab-only | Dataset Exploration §3 says whitespace-separated |
| D-02 | DECIDED | **Per-trip UTM projection** chosen from the trip's median GPS position | One fixed CRS (originally planned EPSG:32632); haversine | Trips are **worldwide** (Europe, Americas, Asia, Africa — `experiments/results/inventory.csv`); trip 00000057 (Ecuador) cannot be projected into zone 32N |
| D-03 | DECIDED | Reference labels from **order-preserving (monotonic) DP alignment** of GPS points to the route sequence | Nearest route arc (baseline, kept for comparison) | Nearest ignores time order (fails at U-turns/revisits). Agreement between the two: train 0.934, student-test 0.897 |
| D-04 | DECIDED | A point is **ambiguous** if its aligned arc is > 30 m away **or** the two strategies disagree | Distance only | Train 17.9 % ambiguous, student-test 10.8 %. Threshold 30 m is a first guess — **REVISIT** with a sensitivity check |
| D-05 | DECIDED | Train only on non-ambiguous points; Top-1 reported on evaluable points (non-ambiguous, reference among candidates) plus all non-ambiguous points | Train on everything | Avoids learning from labels we do not trust |
| D-06 | DECIDED | Candidates: arcs within **50 m**, max **12** per point, KD-tree over arcs sampled every 10 m | Larger radius | Candidate miss rate on evaluable points: train 0.001, student-test 0.000. **REVISIT** with a radius/k vs. miss-rate plot |
| D-07 | DECIDED | Formulation: binary classifier on (point, candidate) pairs; argmax per point | Ranking model | Probabilities feed directly into HMM emissions |
| D-08 | DECIDED | Primary model-selection metric: **CV Top-1 (evaluable points)**; tie-break: CV candidate log loss | Macro-F1 | Declared before the comparison; log loss matters because probabilities feed the HMM |
| D-09 | DECIDED | CV: `GroupKFold(5)` over trip ids (scikit-learn 1.6.1, deterministic, no seed). Fold membership stored in each experiment log | Random split | Trip-level grouping prevents leakage |
| D-10 | DECIDED | KNN trained on a 150k-row subsample, MLP on 400k (seed 42) | Full data | Run time. Stated in logs and must be stated in the report |
| D-11 | DECIDED | LightGBM/XGBoost not used: they need the `libomp` system library, not installed on this machine. sklearn `HistGradientBoostingClassifier` used for boosting | Install libomp via Homebrew | Environment limitation |
| O-01 | RESOLVED | Trip 00000054 (Amsterdam area): after point 2710 the GPS lies 80–110 m from every road (median 84 m) and about 140 m from the reference route, which looks like a systematic GPS offset or travel off the mapped network | — | These points are labelled ambiguous (> 30 m) and excluded from training and Top-1. At inference they have no candidates, so they get no state (D-15). Kept in the data, not deleted |
| O-02 | OPEN | Trips 00000011, 00000061 (train) and 00000065, 00000077 (student-test): strategies agree only 57–78 % | — | Inspect on a map; may be reverse-direction arcs or loops |
| O-03 | OPEN | 9 trips contain self-loop arcs (2–8 each); 14 of 34,723 consecutive route steps are not connected | — | Decide handling in HMM transitions |
| O-04 | RESOLVED | Large time gaps (> 10 s) in 42 trips | — | Handled in the HMM by D-15 (restart after > 60 s; shorter gaps widen the route search bound) |
| D-12 | DECIDED (team to sign off) | **Model to tune with Bayesian optimisation: HistGradientBoosting.** By the declared rule (D-08) MLP leads CV Top-1 (0.9310), but HistGB (0.9301) and Random Forest (0.9295) are within one fold std (~0.01–0.015). Tie-break on CV log loss: RF 0.0573 < HistGB 0.0589 < MLP 0.0616 | MLP / HistGB / RF | `experiments/results/model_comparison.csv`. Top three tied within fold std. Chosen for best student-test Top-1 (0.9331), near-best log loss, 4 s fit (cheap Optuna trials), native NaN handling. Was O-05 |
| D-13 | DECIDED | Optuna TPE (seed 42, 10 random start-up trials, median pruner), objective = mean GroupKFold(5) CV Top-1 on train; trial 0 = untuned config | Grid / random search | Bayesian search required (Models §7). Evaluator data never used. `experiments/optuna/` |
| D-14 | DECIDED | HMM: states = candidate arcs; emission = w·log(normalised classifier prob); transition = −\|route dist − GPS dist\|/β with bounded Dijkstra; unreachable = impossible | Distance-only emission; connectivity-only transitions | Uses the selected model's probabilities consistently (Models §8) |
| D-15 | DECIDED | HMM chain restarts after a time gap > 60 s or when no transition is possible; points without candidates get no state | Interpolate across gaps | Never invents connectivity (plan §4 Phase 5). Resolves O-04 |
| D-16 | DECIDED | Decoder parameters β ∈ {2,5,10,20,50}, w ∈ {0.5,1,2} chosen by grid on **train out-of-fold** Top-1; student-test consulted once | Tune on student-test | Models §9 allows train + student-test; we keep student-test for the comparison |
| D-17 | DECIDED | Route reconstruction joins consecutive chosen arcs with the shortest directed path; if none within the bound a gap (-1) is inserted | Straight lines between arcs | Route geometry uses real arcs only (Gate D). Self-loops (O-03) are ordinary arcs for the decoder |
| D-18 | DECIDED | API: FastAPI + uvicorn; `/health`, `/trips`, `/predict {trip_id}`; data folder from `PATHSNAP_DATA_DIR`; `.route` read only after prediction, metrics returned as aggregates | Flask | `docs/api_contract.md`; leak test `tests/test_api.py` |
| D-19 | DECIDED | UI: single static page (Leaflet) served with `python -m http.server`; API URL configurable | Streamlit / React | No extra dependencies on Laptop 2 |
| D-20 | NOTE | Evaluator-test was run once (pre-tuning models, 2026-11-18 15:19) at the organisers' supply of the data; results in `experiments/evaluator_test/`. They were **not** used for any choice afterwards | — | Models §7 rule |
| D-21 | DECIDED | Tuning stopped at 20 trials | 40 trials | Best CV Top-1 changed by 0.001 over the last 15 trials (< CV std). Tuned: CV 0.9369, student-test 0.9373 (untuned 0.9301 / 0.9331) |
| D-22 | DECIDED | HMM decoder β = 5 m, emission weight 0.5 (train OOF grid) | β 2–50, w 0.5–2 | Student-test: Top-1 0.937 → 0.952, route F1 0.82 → 0.97. Known failure: trip 00000007 fork (Top-1 0.983 → 0.901); not retuned on student-test |

| D-23 | NOTE | Final pipeline run once on evaluator-test (2026-11-18): Top-1 0.9616 with the HMM (classifier only 0.9279), route F1 0.953. The untuned model was run through the same pipeline for reporting only (classifier only 0.9475, HMM 0.9616). No choice was changed afterwards | — | `experiments/evaluator_test_final/RESULTS.md`. The tuned classifier generalised worse on trip 00000015; the HMM evens it out |
