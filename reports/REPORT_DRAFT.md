# PathSnap: Technical Report (DRAFT)

> **Read this first (integrity).** This draft collects facts, numbers and figures from our own logged runs, so the team can write the final PDF. Deliverables §8 says the final text must be the team's own work and understanding, and that AI assistance must be disclosed. Every section owner must **rewrite the explanations in their own words**, check every number against `reports/report_tables.md` and `experiments/`, and be ready to explain it in the viva. Parts in *[brackets]* are notes to the team.
>
> All tables come from `python -m pathsnap.report_tables` and all figures from `pathsnap.report_figures`, `pathsnap.plots` and `pathsnap.viz`.

Team: *[name, members, roles]* · Repository: *[GitHub URL]*

---

## 2.2 Dataset understanding and exploratory data analysis

**Partitions used.**
- *Train* (80 trips, 208,417 GPS points): EDA, model fitting, CV, tuning, HMM decoder selection.
- *Student-test* (10 trips, 19,160 points): development comparisons only.
- *Evaluator-test*: used once for a test-only run of the Phase-1 models (`experiments/evaluator_test/`). Never used for any choice.

**Files.** Each trip has four files, all whitespace-separated with no header:
- `.track`: lon, lat, t
- `.nodes`: lon, lat; row index = node id
- `.arcs`: directed u → v; row index = arc id
- `.route`: ordered reference arc ids

Ids are trip-local.

### 2.2.1 Preprocessing
1. **Loading and invariant checks on all 90 trips** (`data_io.py`, `eda.py`):
   - finite coordinates in range
   - timestamp order and duplicates
   - arc endpoints are valid nodes
   - route ids are valid arcs
   - self-loops

   Nothing is deleted silently. Issues are logged in `experiments/results/data_quality_summary.txt`.
2. **Projection to metres.** A per-trip UTM zone is chosen from the median GPS position (`geo.py`). The trips are worldwide: 58 in Europe/Africa, 11 in the Americas, 21 further east. A single fixed zone failed for trip 00000057 (Ecuador) (D-02).
3. **Reference labels per GPS point.** `.route` is a sequence of arcs, not a label per point, so we aligned points to the route.
   - We compared two strategies: nearest route arc, and an order-preserving dynamic-programming alignment (`labels.py`). The DP alignment was chosen (D-03).
   - They agree on 93.4 % of train points and 89.7 % of student-test points.
   - Points whose aligned arc is more than 30 m away, or where the strategies disagree, are flagged *ambiguous*: 17.9 % of train, 10.8 % of student-test. These are excluded from training and from Top-1 (D-04, D-05).
4. **Candidate generation.** A KD-tree over arcs densified every 10 m returns arcs within 50 m, at most 12 per point (`candidates.py`).
   - Train: 1,342,864 candidate rows, about 6.4 per point.
   - Candidate miss rate on evaluable points: 0.001 for train, 0.000 for student-test.

### 2.2.2 Missing values and patterns
*Figure: `reports/figures/eda_missing_values_and_gaps.png`. Table: `experiments/results/missing_value_analysis.csv`.*

| | Train | Student-test |
|---|---|---|
| Missing / non-finite lon, lat, t | 0 | 0 |
| Duplicate / backwards timestamps | 0 / 0 | 0 / 0 |
| Gaps > 10 s | 590 (in 42 trips across both partitions) | 27 |
| Share of 1-s steps (point-weighted, all trips) | 0.967 | |
| Self-loop arcs | 8 trips | 1 trip |
| Disconnected consecutive route steps | 14 of 34,723 (all trips) | |
| Reciprocal arcs (median per trip) | 0.951 | |

*Figures:*
- `eda_gps_trajectories.png`: raw GPS for 8 train trips
- `eda_road_network.png`: road network, GPS and reference route, train 00000001
- `eda_network_and_trip_sizes.png`
- Folium: `maps/eda_train_00000001.html`

**Patterns to discuss:**
- About 95 % of roads exist as two opposite arcs, so direction is essential.
- GPS is dense (1 Hz) but has occasional long gaps.
- Trip 00000054 has a systematic 80–110 m offset from every road after point 2710 (O-01).
- A few trips have low agreement between the two labelling strategies, because of U-turns and revisits (O-02).

### 2.2.3 Features (24) and why
See `docs/features.md`. There are six groups:
- distance (absolute and relative to the other candidates)
- position along the arc
- direction (vehicle heading vs. directed arc bearing; NaN when slower than 1 m/s)
- kinematics
- a ±3-point temporal window
- continuity and network structure

None uses `.route`.

Permutation importance (HistGB, student-test, drop in Top-1 from 0.933):

| Feature | Drop in Top-1 |
|---|---|
| `win_mean_heading_diff` | 0.352 |
| `frac_along` | 0.092 |
| `heading_diff` | 0.037 |
| `dist_minus_min` | 0.021 |
| `dist` | 0.020 |

Rule baselines on student-test: nearest road 0.572, nearest road in the same direction 0.890, HistGB 0.933. *[Explain in your own words why direction plus context dominates.]*

---

## 3. Model experimentation and comparison

**Setup.**
- Binary classification of (point, candidate arc) pairs; the highest-probability candidate per point is the prediction.
- GroupKFold(5) by trip on the 80 train trips; imputation and scaling are fitted inside each fold.
- Primary metric, declared before the comparison: CV Top-1 on evaluable points. Tie-break: log loss (D-08).
- Final fit on all 80 trips, then one student-test evaluation.
- Seeds and folds are in `experiments/logs/*.json`.

### 3.1 Overall comparison
→ Paste table 3.1 from `reports/report_tables.md`.

*Figures:*
- `cv_comparison.png`: CV Top-1, CV macro-F1, CV log loss
- `student_test_comparison.png`
- `logloss_and_inference_time.png`
- `confusion_matrices_student_test.png`
- `history_hist_gb.png`, `history_mlp.png`: the only two models with real iteration histories

N/A notes: KNN, RF, Extra Trees, Logistic Regression and Naive Bayes have no epoch-wise loss curves, so none are shown.

### 3.2 Per-model answers
*[Owner of each model: rewrite A in your own words. B and C are facts from the logs.]*

#### Logistic Regression (linear)
- **A.** Simple and calibrated-ish baseline that tests whether a linear boundary in feature space is enough. Limitation: it cannot express interactions such as "close AND same direction AND consistent with neighbours".
- **B.** C = 1.0 (L2), lbfgs, max_iter 500. Loss: log loss plus L2 penalty.
- **C.** Student-test Top-1 0.9073, macro-F1 0.9296, log loss 0.0864.
  - Fit: CV Top-1 0.9184 vs student-test 0.9073, so there is no overfitting.
  - The model **underfits** relative to the tree models: about 1.3 points below them on both CV and student-test.

#### Gaussian Naive Bayes (probabilistic)
- **A.** Fast generative baseline. It assumes features are independent given the class, which is violated here (e.g. `dist`, `win_mean_dist`, `dist_rank`).
- **B.** var_smoothing 1e-9. Objective: maximum likelihood of per-class Gaussians.
- **C.** Student-test Top-1 0.8981, macro-F1 0.8552, log loss 0.7092.
  - Ranking is OK (Top-1 near 0.90), but probabilities are badly calibrated: log loss is over 10× that of the trees.
  - Underfits; no overfitting (CV 0.8987 ≈ student-test 0.8981).

#### KNN (instance-based)
- **A.** Non-parametric; neighbours in scaled feature space. Sensitive to scaling and slow at prediction time.
- **B.** k = 25, distance-weighted, Euclidean, 150k-row subsample (D-10). No loss function (lazy learner).
- **C.** Student-test Top-1 0.9073, macro-F1 0.9357, log loss 0.0955. Inference 1.58 s per trip, about 200× slower than the trees.
  - CV 0.9114 vs student-test 0.9073 gives a small gap.
  - Limited by the subsample and by the curse of dimensionality (24 or more dimensions).

#### Random Forest (bagging)
- **A.** Non-linear rules, robust to scale, good probabilities. Limitations: large model (83 MB) and slower fit (26 s).
- **B.** 150 trees, max_depth 18, min_samples_leaf 5, max_samples 0.3. Objective: Gini impurity.
- **C.** Student-test Top-1 0.9319, macro-F1 0.9559, log loss 0.0500 (best).
  - CV 0.9295 vs student-test 0.9319, so no overfitting.
  - Depth and leaf limits regularise the trees.

#### Extra Trees (randomised trees)
- **A.** Like RF but with random split thresholds, so lower variance and faster. The extra randomness costs accuracy here.
- **B.** As RF, with bootstrap. Objective: Gini.
- **C.** Student-test Top-1 0.9209, macro-F1 0.9472, log loss 0.0674. CV 0.9248, so no overfitting; it slightly underfits relative to RF.

#### HistGradientBoosting (boosting): selected
- **A.** Sequential error correction on binned features. Handles NaN headings natively, fits fast (4 s), gives good probabilities for HMM emissions. Limitation: more hyperparameters to tune.
- **B.** max_iter 300, learning_rate 0.1, 31 leaves, min_samples_leaf 20, early stopping on a 10 % validation split. Loss: binary log loss.
- **C.** Student-test Top-1 0.9331 (best), macro-F1 0.9555, log loss 0.0523.
  - CV 0.9301 ± 0.015 vs student-test 0.9331.
  - `history_hist_gb.png` shows the validation loss flattening and early stopping, so no overfitting.

#### MLP (neural network)
- **A.** Learns feature interactions. Needs scaling and imputation; training is less stable; 400k-row subsample (D-10).
- **B.** Layers 64-32, ReLU, Adam, alpha 1e-4, batch 1024, early stopping (patience 5). Loss: log loss.
- **C.** Student-test Top-1 0.9274, macro-F1 0.9509, log loss 0.0555.
  - Best CV Top-1 (0.9310) but lower on student-test, the largest negative gap of the top three: some variance.
  - `history_mlp.png` shows the training loss and validation score.

**Selection (D-12).**
- The top three are within one fold standard deviation of each other.
- HistGB was chosen for:
  - best student-test Top-1
  - near-best log loss
  - fastest fit, which makes Bayesian tuning affordable
  - native NaN handling
  - direct use as HMM emissions

---

## 4. Bayesian hyperparameter optimisation (HistGB)
*Code: `src/pathsnap/tuning.py`. Records: `experiments/optuna/`. Table: section 4 of `report_tables.md`. Figures: `tuning_history.png`, `tuning_untuned_vs_tuned.png`.*

1. **Search space.**

   | Hyperparameter | Range |
   |---|---|
   | learning_rate | 0.02–0.3, log scale |
   | max_leaf_nodes | 15–255, log scale |
   | max_depth | {None, 6, 10, 16} |
   | min_samples_leaf | 10–1000, log scale |
   | l2_regularization | 1e-4–10, log scale |
   | max_features | 0.5–1.0 |
   | max_iter | 150–800 |

   Ranges are centred on the untuned defaults.
2. **Objective:** mean GroupKFold(5) CV Top-1 on evaluable points (the primary metric). Optuna TPE sampler (seed 42), 10 random start-up trials, median pruner. Trial 0 is the untuned configuration.
3. **Trials:** 20 complete, 0 pruned. A 21st trial was stopped by us and is marked FAIL. We stopped at 20 because the best value changed by only 0.001 over the last 15 trials, below the CV standard deviation (0.015–0.019). Total search time was 37 minutes.
4. **Best values:**

   | Hyperparameter | Value |
   |---|---|
   | learning_rate | 0.097 |
   | max_leaf_nodes | 239 |
   | max_depth | None |
   | min_samples_leaf | 14 |
   | l2_regularization | 0.99 |
   | max_features | 0.63 |
   | max_iter | 500 |
5. **Untuned vs tuned** (same folds and partitions):

   | Metric | Untuned | Tuned |
   |---|---|---|
   | CV Top-1 | 0.9301 | 0.9369 |
   | CV macro-F1 | 0.9545 | 0.9561 |
   | Student-test Top-1 | 0.9331 | 0.9373 |
   | Student-test macro-F1 | 0.9555 | 0.9595 |
   | Student-test log loss | 0.0523 | 0.0510 |
   | CV log loss | 0.0589 | 0.0609 |
   | Inference (s/trip) | 0.010 | 0.061 |
   | Fit time (s) | 5 | 47 |

   The tuned model is better on every accuracy metric. Its CV log loss is slightly worse, and it is slower because of its larger trees.
6. **Computational cost:** 37 minutes of search on a 10-core laptop, about 1.9 minutes per trial (5 fits per trial).

---

## 5. HMM and Viterbi route reconstruction
*Code: `src/pathsnap/hmm.py`, `hmm_experiment.py`. Results: `experiments/results/hmm_*.csv`. Figures: `hmm_vs_classifier_per_trip.png`, `hmm_grid_train.png`.*

1. **Hidden states:** the candidate arcs of each GPS point (the candidate row keeps its arc id).
2. **Emission:** w · log(p / Σp), where p is the selected HistGB probability, normalised over the point's candidates.
3. **Transition:** −|d_route − d_gc| / β.
   - d_route: driving distance along the directed network from the projected position on the previous arc to the position on the next arc, from a bounded Dijkstra (bound 2·d_gc + 200 m, at most 5 km).
   - d_gc: straight-line GPS distance.
   - Same arc: forward movement along it; up to 15 m backwards is accepted as noise.
   - Unreachable: impossible. Decoding is Viterbi in log space.
4. **Missing candidates and gaps:**
   - Points without candidates get no state, and the chain continues across them.
   - Gaps longer than 60 s restart the chain.
   - Shorter gaps widen the search bound through d_gc.
5. **Disconnected segments:** if no transition is possible, the chain restarts and the break is counted. In route reconstruction, consecutive arcs are joined by the shortest directed path; with no path, a gap (-1) is inserted. Connectivity is never invented.
6. **Did it help?** Yes, overall. The decoder was selected on train out-of-fold: β = 5 m, w = 0.5. Student-test results:

   | Metric | Classifier only | Classifier + HMM |
   |---|---|---|
   | Top-1 | 0.9373 | 0.9515 |
   | Route continuity | 0.972 | 0.984 |
   | Route precision | 0.714 | 0.947 |
   | Route F1 | 0.824 | 0.968 |
   | Normalised edit distance | 0.576 | 0.099 |

   Train out-of-fold shows the same pattern: Top-1 0.933 → 0.950, route F1 0.797 → 0.957. Route recall is about 0.99 for both: the classifier-only route contains the right roads, but also many spurious detours.
   - **Where it did not help:** student-test trip 00000007. Top-1 drops from 0.983 to 0.901 (route F1 still improves, 0.972 → 0.980). At a fork, arcs 20→89 and 20→1386 leave the same node. The HMM committed to the wrong branch for about 140 consecutive observations (points 1208–1353), while the classifier picked each point correctly. The transition term (β = 5) is strong enough to keep a whole stretch on the branch whose length matches the GPS distance better. We did **not** retune on student-test to fix this, because that would bias the comparison. Future work: direction or heading in the transition, or a larger emission weight near forks.
   - **Grid:** smaller β (stricter route-length agreement) and lower w work best on train. Every HMM setting beat classifier-only on train Top-1 (`hmm_grid_train.png`).
   - Discuss trips where it did not help (see the per-trip table).
   - Decoder parameters β and w were chosen on train out-of-fold predictions; student-test was consulted once.

---

## 6. Geographical visualisation (Folium)
*Code: `src/pathsnap/viz.py`. Maps: `maps/*.html`.*

Each map has these layers, each with its own colour, plus a legend and a layer control:
- road network near the track (grey)
- raw GPS (blue)
- reference route (green, offline only)
- classifier-only picks (orange)
- HMM route (red)

Screenshots: `reports/figures/folium_student_test_00000077.jpg` (hard trip: HMM 0.871 vs classifier 0.821) and `folium_student_test_00000007.jpg` (the case where the HMM hurt). Interactive maps: `maps/student_test_*.html`, `maps/eda_train_00000001.html`.

---

## 7. API, UI and deployment
*Code: `api/main.py`, `ui/index.html`. Docs: `docs/api_contract.md`, `docs/deployment.md`. Tests: `tests/test_api.py` (17 tests, including the leak test).*

- The API runs on Laptop 1 (FastAPI + uvicorn) and serves `/health`, `/trips` and `/predict`. It logs every request to the console.
- The UI runs on Laptop 2 (a static Leaflet page). It lists the trips, sends the selected id and shows the raw GPS, the predicted route, the metrics computed by the API and the timings.
- Ground truth stays on the server. The leak test checks that responses are whitelisted and that predictions are identical when `.route` is deleted.

UI screenshot: `reports/figures/ui_trip76.jpg` (single-laptop run, student-test data). API console log: `experiments/logs/api_console_final.txt`. *[Take the real two-laptop screenshots of the UI and the API console during the rehearsal.]*

---

## Limitations
- Labels come from our alignment, not from the organisers. The ambiguity threshold (30 m) has not had a sensitivity check yet.
- 9–18 % of points are excluded as ambiguous.
- LightGBM/XGBoost were not run (libomp missing).
- KNN and MLP were trained on subsamples.
- The reference routes are not guaranteed error-free (Dataset §6).

## AI-use disclosure
*[As in README; state which parts were AI-assisted and what the team changed.]*
