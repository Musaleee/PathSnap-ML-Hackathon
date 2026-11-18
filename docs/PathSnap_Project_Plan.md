# PathSnap — Team Project Plan

> Built from the five supporting documents (Main Guidelines, Dataset Exploration, Models & Training Guidelines, Deliverables, Assessment & Plagiarism Policy) and a first look at the provided data.
>
> **Originality note:** This plan is a *work structure*, not a solution. It lists what must be produced and the decisions the team must make. It deliberately leaves the technical choices (features, models, parameters, probability formulas) to the team. Every choice must be made, tested and recorded by you. AI assistance (including this plan) should be disclosed as the policy requires (Deliverables §8, rule 5).

---

## 0. Ground rules (from the policy — read first)

| Rule | Source | What it means for us |
|---|---|---|
| Train and compare **≥ 5 models** from different families | Models §2, §4; Policy §3 | Plan for 6–7, so one failure does not drop us below 5 |
| **Bayesian** hyperparameter optimisation, with the search record kept | Models §7; Policy §3 | Use Optuna (TPE) / scikit-optimize, save the study (SQLite/CSV) |
| HMM + Viterbi, compared with classifier-only | Models §8–9; Deliverables §5 | Both results go in the same table |
| Evaluator-test data untouched; ground truth stays server-side | Policy §7; Models §6 | We never see it; the API must load it only for scoring |
| **No hardcoded predictions**: repos are examined | Policy §7 | Every output must come from the pipeline at request time |
| No sharing code, weights, configs or results between teams | Policy §4; Models §11 | Don't discuss implementation details with other teams; keep the repo private until submission |
| No fabricated metrics, graphs or curves | Policy §5; Deliverables §3.3 | Every number and plot is produced by a script in the repo |
| Every member can explain everything | Policy §5–6 | Cross-teaching sessions are part of this plan (Phase 9) |
| Repo history is compared | Policy §5 (Similarity Review) | Commit often, from each person's own account |
| First-phase evaluation **before lunch break** | Deliverables p.1 | Phases 1–3 must be demo-ready by then (see timeline) |

---

## 1. What we know about the data (verified)

| Item | Fact |
|---|---|
| Partitions | `train/` = 80 trips, `student_test/` = 10 trips (both have `.route`). Evaluator test = 10 trips, not given to us |
| Files per trip | `.track`, `.nodes`, `.arcs`, `.route` |
| `.track` | `lon lat t` (tab-separated, no header). 361 – 19,313 rows per trip (median ≈ 1,500). ~1 Hz |
| `.nodes` | `lon lat`; the row index is the node id. 299 – 155,003 nodes per trip (median ≈ 9,600) |
| `.arcs` | `from_node to_node`; the row index is the arc id |
| `.route` | Ordered sequence of ground-truth **arc ids**. 59 – 2,266 arcs per trip (median ≈ 277) |
| Region | lon ≈ 11.0, lat ≈ 47.6 (southern Bavaria / Tyrol area) — verify |

**Key observations the team must reason about (not answered here):**
1. Each trip has **its own road-network file**. Arc ids are local to a trip, so they are not comparable across trips.
2. `.route` is a sequence of **arcs**, not one label per GPS point. You must design how each GPS observation gets a reference arc (needed for Top-1 accuracy, F1, log loss). This is one of the most important design decisions. Document it and its limitations (Dataset §6 says labels are not error-free).
3. The classification task is therefore *"is this candidate arc the correct one for this GPS point?"* (or a ranking over candidates), not "predict arc id directly". Decide your formulation and justify it.
4. Degrees are not metres: pick a projection (e.g. a UTM zone that fits the coordinates) or a geodesic distance, and justify it (Dataset §5).
5. Check whether arcs are directed (are both `a→b` and `b→a` present?). This matters for HMM transitions.

---

## 2. Team roles (adapt to team size)

Each area has an **owner** (writes most code, commits it) and a **reviewer** (reads it, can explain it in the viva). Everyone owns at least one model.

| Area | Owner | Reviewer |
|---|---|---|
| Data loading, EDA, Folium maps | Member A | Member C |
| Candidate generation + features + label assignment | Member B | Member A |
| Model experiments (≥ 5) + Bayesian optimisation | Member C (+ each member trains ≥ 1 model) | Member D |
| HMM / Viterbi | Member D | Member B |
| FastAPI backend (Laptop 1) | Member B | Member D |
| UI (Laptop 2) | Member A | Member C |
| Report assembly | Member D | Everyone |

Write your actual names in the README "Team & contributions" section.

---

## 3. Repository setup (do this in the first 30 minutes)

```
pathsnap/
├── README.md                  # install, run, deploy, team contributions, AI-use disclosure
├── requirements.txt           # pinned versions
├── .gitignore                 # data/, *.zip, __pycache__, .env, large artifacts
├── configs/                   # YAML/JSON: paths, CRS, candidate radius, seeds, model params
├── data/                      # NOT committed — README explains where to put "Student folder"
├── src/
│   ├── io/                    # readers for .track/.nodes/.arcs/.route
│   ├── geo/                   # projection, point-to-segment distance, spatial index
│   ├── eda/                   # exploration scripts
│   ├── candidates/            # candidate generation
│   ├── labels/                # GPS-point → reference-arc assignment
│   ├── features/              # feature engineering
│   ├── models/                # one file per model + common train/eval harness
│   ├── tuning/                # Bayesian optimisation
│   ├── hmm/                   # emission, transition, Viterbi
│   ├── evaluation/            # metrics, route continuity, plots
│   └── viz/                   # Folium map generation
├── api/                       # FastAPI app (Laptop 1)
├── ui/                        # evaluator UI (Laptop 2)
├── notebooks/                 # exploration notebooks (with outputs kept)
├── experiments/
│   ├── logs/                  # one JSON/CSV per run: config, seed, metrics, timings
│   ├── optuna/                # study DB / trial CSV
│   └── results/               # comparison tables (CSV)
├── artifacts/                 # final model, scaler, HMM config (Git LFS if large)
├── reports/figures/           # every figure used in the PDF, generated by scripts
└── maps/                      # exported Folium HTML
```

**Git practice (originality evidence):**
- Each member clones and commits from **their own account**, with `git config user.name/email` set correctly.
- Small, frequent commits with meaningful messages (`feat(candidates): radius search with STRtree`), not one bulk upload.
- Use branches per area (`eda`, `features`, `models-xgb`, `hmm`, `api`, `ui`), merged via pull requests that the reviewer approves. PR reviews also show understanding.
- Never commit the raw dataset, secrets, or evaluator data.
- Make the repo public only at submission time (Deliverables: "made public").

---

## 4. Phase-by-phase plan

The timings assume a one-day hackathon with a first-phase review before lunch. Adjust to your actual schedule.

### Phase 1 — Understand & explore (start → ~1.5 h) · Owner: A

**Tasks**
1. Write readers for the four file types. Validate column counts and dtypes, and that arc endpoints are valid node indices.
2. Build the dataset inventory (Dataset §4, §8):
   - trips per partition, file sizes, rows per file
   - missing/invalid GPS values (NaN, out-of-range coords, duplicates)
   - timestamp ordering, gap distribution (the doc says ~97% are 1 s — confirm), trips with large gaps
   - geographic extent per trip
   - road network: #nodes, #arcs, connected components, isolated nodes, directedness
   - route: length, whether consecutive route arcs are actually connected in `.arcs`
3. Plots: sampling-interval histogram, track-length distribution, GPS scatter, network-size table.
4. **Folium map** for 2–3 trips: raw GPS points + road network (clipped to the trip bbox) + ground-truth route, with a layer control.
5. Write `experiments/results/data_quality_log.md`: issues found, assumptions, decisions.

**Decision points (team decides, records the reason):** projection/CRS, how to treat duplicate or out-of-order timestamps, what counts as a "large gap".

**Done when:** inventory CSV + plots + ≥ 1 Folium map are committed. ✅ *Ready for the first-phase review.*

---

### Phase 2 — Candidates, labels, features (~1.5 h → ~3.5 h) · Owner: B

**Tasks**
1. **Candidate generation:** for each GPS point, find nearby arcs (spatial index, e.g. Shapely STRtree or a KD-tree on segment samples). Choose the radius or k by measuring the **Candidate Miss Rate** (Models §5) on training trips. Plot miss rate against radius/k and pick the value from that plot.
2. **Reference labels:** implement your method for mapping each GPS point to its ground-truth arc using the `.route` sequence. Measure how often it is ambiguous and log those cases.
3. **Features per (GPS point, candidate arc) pair:** brainstorm as a team and write down *why* each feature should help. Feature categories to consider (choose your own):
   - geometric (distance and position relative to the segment)
   - directional (heading of the vehicle vs. segment bearing)
   - kinematic (speed and acceleration from timestamps)
   - contextual (neighbouring points, candidate rank among candidates, number of candidates)
   - network (segment length, node degree)
4. Build one training table: `trip_id, point_idx, arc_id, features…, label`.
5. Fit any scaler or encoder **on training trips only**, inside the CV folds.

**Decision points:** classification formulation (binary per candidate vs. ranking), class imbalance handling, feature list.

**Done when:** the feature table can be rebuilt with one command, the miss-rate plot is saved, and the feature rationale is written in `docs/features.md` (in your own words — the report needs it).

---

### Phase 3 — Model experiments (~3.5 h → ~6 h) · Owner: C, every member trains ≥ 1 model

**Common harness (write once, used by all models):**
- **GroupKFold by `trip_id`** on the 80 training trips (Models §7.3), with a fixed seed.
- Logs per run: model, family, config, seed, CV macro-F1 (mean ± std per fold), training time, inference time.
- Student-test evaluation on the 10 trips: accuracy, precision, recall, F1, macro-F1, log loss, observation-level **Top-1 accuracy**, candidate miss rate, route continuity, inference time.
- Appends results to `experiments/results/model_comparison.csv`.

**Pick ≥ 5 from different families (Models §3).** Example spread, choose your own:

| Family | Candidate |
|---|---|
| Linear / probabilistic | Logistic Regression or Gaussian NB |
| Instance-based | KNN (scaled features) |
| Tree / ensemble | Random Forest or Extra Trees |
| Boosting | HistGradientBoosting / LightGBM / XGBoost |
| Kernel | SVM (with calibrated probabilities) |
| Neural | MLP (keep the epoch-wise loss history → loss plot) |

Notes:
- Models need **calibrated probabilities** for HMM emissions. Note which ones need calibration (e.g. SVM, Ridge) and how you did it.
- Training/validation loss curves are drawn **only** for models that really have epochs (MLP, boosting with eval sets). No artificial curves (Deliverables §3.3).
- Each model owner writes their model's report subsection (A: why suitable, strengths/limits; B: key hyperparameters, loss; C: student-test results, over- or underfitting evidence).

**Done when:** `model_comparison.csv` has ≥ 5 rows with all metrics, and the comparison plots are generated by a script.

---

### Phase 4 — Model selection + Bayesian optimisation (~6 h → ~7 h) · Owner: C

1. Select the model using a **primary metric you justify** (Models §5 — e.g. CV macro-F1 or Top-1), and also consider fold stability, probability quality (log loss), speed and HMM compatibility. Record the reasoning.
2. Define a justified search space (ranges + why).
3. Run Optuna (TPE) with group-aware CV on the 80 trips as the objective. Record the number of trials and wall-clock cost.
4. Save the study and plot the optimisation history + parameter importance.
5. Refit with the best params on all 80 trips. Evaluate on student-test.
6. Make an **untuned vs. tuned** table: CV macro-F1, student-test Top-1, student-test macro-F1, log loss, inference time.

**Done when:** the study file, plots and comparison table are committed. Export the final model + preprocessing to `artifacts/` (joblib), with a version/config file.

---

### Phase 5 — HMM + Viterbi (can start in parallel at ~4 h) · Owner: D

1. **Hidden states:** the candidate arcs per GPS point (keep the candidate ↔ arc mapping exact).
2. **Emission probability:** derived from the selected model's output. Decide and justify how (raw probability, log-probability, combined with distance, etc.).
3. **Transition probability:** based on road-network connectivity. Decide and justify the approach (e.g. network-distance vs. straight-line-distance comparison, same/adjacent arc, penalties for unconnected arcs). Precompute graph structures (NetworkX / SciPy sparse) for speed.
4. **Viterbi** in log-space. Handle:
   - points with no candidates
   - all-zero transitions (break and restart the chain)
   - large GPS gaps (decide: split the sequence, widen the search, or loosen transitions)
5. Tune decoder parameters on **training + student-test only** (Models §9).
6. Compare **classifier-only vs. classifier + HMM**: Top-1, macro-F1, route continuity, sequence-level metric. Include per-trip results and at least one case where HMM did **not** help, with an explanation.

**Done when:** the comparison table and two Folium maps (one improved, one not improved) are committed.

---

### Phase 6 — API (Laptop 1) · Owner: B

FastAPI app that:
- Loads model + preprocessing + HMM config at startup.
- `GET /trips` → the list of available trip ids.
- `POST /predict` (`{trip_id}`) → runs the **full pipeline at request time** (read track → candidates → features → model → Viterbi) → returns the predicted arc sequence, route geometry (GeoJSON) and metrics.
- Computes accuracy using ground truth that is **loaded only on the server** and never returned before prediction, and never fed into the pipeline.
- Logs every request (trip id, timings, metrics) to the console — the evaluator will check these logs (Models §6).
- Runs with `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Write down the firewall/LAN steps in the README.
- Reads its data directory from config, so evaluators can drop in their 10 trips.

**Test:** run it against all 10 student-test trips; the system must work for **every** trip, not one demo.

---

### Phase 7 — UI (Laptop 2) · Owner: A

- Shows the list of evaluation trips (from `/trips`), and lets the evaluator pick one.
- Sends the trip id to the API (API URL is configurable — the IP of Laptop 1).
- Displays a map: raw GPS, predicted route, reference route (from the API's server-side scoring), with a legend and layer toggle. Also shows accuracy/metrics and latency.
- Handles API errors and timeouts clearly.
- Tech is the team's choice (Streamlit, plain HTML + Leaflet, React…). Justify it in one line in the report.

**Two-laptop rehearsal:** at least once before evaluation, on the real network, with screenshots of the UI and the API console (both are needed in the report, Deliverables §7).

---

### Phase 8 — Report (written progressively, finalised at the end) · Owner: D, all contribute

Follow the exact structure in Deliverables §2–7:

| Section | Content | Evidence source |
|---|---|---|
| 2.2 EDA | Preprocessing, patterns, features + why, missing-value analysis, GPS plots, network plots, ≥ 1 Folium map, **partition used for each analysis** | Phase 1–2 outputs |
| 3.1 Comparison table | Model, family, key config, CV macro-F1, Top-1, route continuity, student-test macro-F1, log loss, inference time (N/A + reason where needed) | `model_comparison.csv` |
| 3.2 Per-model subsections | A. why suitable, strengths/limits · B. hyperparameters, loss · C. student-test results, Top-1/F1/log loss, overfitting evidence | each model owner |
| 3.3 Visualisations | CV macro-F1 across models, student-test Top-1 & F1, log loss, (optional) confusion matrices, real loss curves only | `reports/figures/` |
| 4 Bayesian optimisation | Params, ranges, objective, #trials, best values, untuned vs. tuned table | Optuna study |
| 5 HMM/Viterbi | States, emissions, transitions, gaps, disconnected segments, did it help (with numbers + failure case) | Phase 5 |
| 6 Folium | GPS, network, reference, predicted, legend, layer control | `maps/` |
| 7 API/UI | UI screenshots, API responses on the CMD of the receiving laptop | rehearsal |

**Writing rules (originality):**
- Write in your own words, from your own results. Every claim points to a file in the repo.
- If AI tools help with wording, check every sentence against your actual results, and disclose the use.
- No textbook-only explanations: always connect to "in our experiments, we observed…".
- Keep a `docs/decisions.md` log during the day (date, decision, alternatives, why, who). It feeds the report and the viva directly.

---

### Phase 9 — Viva preparation (last hour) · Everyone

Each member should be able to answer, without notes, for **every** area (Policy §6):
- How is the dataset organised? What does each file mean? How do they relate?
- How did we assign a reference arc to each GPS point, and what are the limitations?
- Why these features? Which helped most (show feature importance / ablation)?
- Why did we pick model X over the others? What is the primary metric and why?
- What was the Bayesian search space and objective? How many trials? Did tuning help?
- How are emissions and transitions computed? What happens on a large gap or a disconnected candidate?
- How does the API avoid leaking ground truth? Show the request logs.
- How would you reproduce result Y from scratch? (Run the command live.)

Do a 20-minute round where each member explains a part they **did not** write.

---

## 5. Suggested timeline (one-day example)

| Time | A | B | C | D |
|---|---|---|---|---|
| 0:00–0:30 | Repo, README skeleton | Readers | Env, requirements | Decision log, report skeleton |
| 0:30–1:30 | EDA + Folium | Geo utils, projection | EDA stats | Network connectivity analysis |
| **— first-phase review (before lunch) —** | EDA + maps | Candidate generation started | Harness ready | HMM design notes |
| 1:30–3:30 | UI skeleton | Candidates, labels, features | Harness + model 1 | Transitions/graph precompute |
| 3:30–6:00 | Model 4 + UI | API skeleton + model 5 | Models 2–3, comparison | Viterbi + model 6 |
| 6:00–7:00 | UI ↔ API integration | API full pipeline | Bayesian optimisation | HMM vs. classifier experiments |
| 7:00–8:00 | Two-laptop rehearsal, screenshots | Fix bugs | Final plots | Report assembly |
| Last hour | Viva practice — everyone | | | |

---

## 6. Originality & integrity checklist

- [ ] Every member has commits under their own name, spread over the whole day.
- [ ] No code, weights, configs or results copied from or shared with other teams.
- [ ] All design decisions are in `docs/decisions.md` with reasons.
- [ ] Every number in the report is produced by a script and stored in `experiments/`.
- [ ] Every figure is generated by a script in `src/evaluation` or `src/viz` — no hand-made or edited plots.
- [ ] No loss curves for models without epochs.
- [ ] Student-test was used for development comparison only; evaluator-test never seen or used.
- [ ] Scaler/encoder fitted on training folds only.
- [ ] No hardcoded trip ids, answers or routes anywhere (`grep` the repo for trip ids before submitting).
- [ ] Ground truth never returned before prediction and never used as model input in the API.
- [ ] Random seeds fixed and recorded.
- [ ] AI-tool use disclosed in the README and the report.
- [ ] Every member has done the cross-explanation round.

## 7. Final submission checklist

- [ ] Repo public and accessible; README with install / run / deploy / two-laptop setup.
- [ ] Code for data processing, features, all ≥ 5 models, evaluation, Bayesian optimisation, HMM, API, UI, Folium.
- [ ] Final model + config in `artifacts/` (or regeneration instructions if too large).
- [ ] Experiment logs, Optuna study, comparison CSVs committed.
- [ ] Folium HTML maps or their generation code committed.
- [ ] No secrets, no raw data, no evaluator data in the repo.
- [ ] One consolidated PDF report, following Deliverables §2–7 exactly.
- [ ] Code in the repo matches what is demonstrated.
- [ ] API tested on all 10 student-test trips; two-laptop setup rehearsed.
