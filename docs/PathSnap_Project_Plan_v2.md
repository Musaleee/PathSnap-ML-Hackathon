# PathSnap — Team Project Plan (v2, final)

> v2 revises `PathSnap_Project_Plan.md` using the review in `PathSnap_Plan_Review_Corrections.md`. The structure is the same; corrections and decision gates are added. The change log is at the end.
>
> **Labels used in this plan:** **[OFFICIAL]** = stated in the five PathSnap PDFs. **[TEAM]** = our own choice or suggestion (can be changed). **[UNCONFIRMED]** = assumption about the event to confirm with organisers.
>
> **Originality:** The plan and the starter code were produced with AI assistance (Claude). This must be disclosed accurately in the README and report [OFFICIAL: Deliverables §8 rule 5]. Every team member must read, run and be able to explain all code, and must own the decisions in `docs/decisions.md` — change anything you disagree with, and record why.

---

## 0. Ground rules

| Rule | Source | What it means for us |
|---|---|---|
| Train and evaluate **at least five distinct models**; experimenting across algorithm families is strongly encouraged | [OFFICIAL] Models §2, §4 | 5 is the requirement. We aim for 6–7 across several families [TEAM], but never at the expense of HMM, deployment or report |
| Bayesian hyperparameter optimisation on the chosen model, with the search record kept | [OFFICIAL] Models §7, Policy §3 | Optuna (TPE), study saved to `experiments/optuna/` |
| HMM + Viterbi, compared with classifier-only | [OFFICIAL] Models §8–9, Deliverables §5 | Same metrics, same trips |
| Evaluator-test data untouched; ground truth never enters the prediction pipeline **or the UI** | [OFFICIAL] Policy §7, Models §6 | See §3 "Data isolation" |
| No hardcoded predictions; repos are examined | [OFFICIAL] Policy §7 | All outputs computed at request time |
| No sharing of code, weights, configs or results between teams | [OFFICIAL] Policy §4, Models §11 | |
| No fabricated metrics, graphs, curves or citations | [OFFICIAL] Policy §5, Deliverables §3.3 | Every number and plot comes from a logged run |
| Every member must be able to explain everything | [OFFICIAL] Policy §5–6 | Cross-explanation round (Phase 9) |
| Repository history is compared across teams | [OFFICIAL] Policy §5 | Honest, frequent commits from each member's own account. Never fake history or contributions |
| A first-phase evaluation happens before lunch | [OFFICIAL] Deliverables p.1 | What is assessed is **not specified** → see §4 Phase 1 readiness package [UNCONFIRMED] |

---

## 1. Data facts (verified on the provided files)

| Item | Fact |
|---|---|
| Partitions | `train/` 80 trips, `student_test/` 10 trips (both include `.route`). Evaluator test = 10 trips, not given to us |
| Format | All four file types are **whitespace-separated, no header** [OFFICIAL Dataset §3]. (The files we have use tabs, but loaders must accept any whitespace) |
| `.track` | `lon lat t` (seconds). 361 – 19,313 rows per trip, 208,417 rows in train in total, 19,160 in student-test |
| `.nodes` | `lon lat`; 0-based row index = node id. 299 – 155,003 nodes per trip |
| `.arcs` | `from_node to_node` (integers); 0-based row index = arc id. In the sample checked, almost every road has both directions as separate arcs |
| `.route` | Ordered sequence of ground-truth arc ids, 59 – 2,266 per trip |
| Region | lon ≈ 11.0°, lat ≈ 47.6° → UTM zone 32N (EPSG:32632) is a suitable metric projection [TEAM, to verify] |

**Invariants every loader/check must enforce (P1-4):**
- Node and arc ids are **trip-local**. Never join ids across trips. Every cache key includes `trip_id`.
- `arc_id == row index`; every arc endpoint is a valid node index; route arc ids are valid arc indices.
- Arcs are **directed**. Check reciprocal arcs, self-loops, duplicates, and connected components.
- Coordinate order is `lon, lat`. Check plausibility.
- Processing of one trip is deterministic for a given config.

**Semantics to keep in mind:**
- `.route` is a sequence of arcs, **not one label per GPS point**. Row *i* of `.track` has nothing to do with row *i* of `.route` [OFFICIAL Dataset §3].
- Reference routes are **not guaranteed error-free** [OFFICIAL Dataset §6]. We report this limitation.

---

## 2. Dependencies and approval gates

| Gate | Condition to pass | Unlocks |
|---|---|---|
| **A — Data semantics** | Loaders pass invariant checks on all 90 trips; inventory and data-quality log done | Candidate generation, labels |
| **B — Labels & candidates valid** | Alignment method chosen from ≥ 2 strategies, ambiguous cases flagged and inspected on real trips; candidate miss rate measured | Supervised training |
| **C — Comparable models** | ≥ 5 genuine model runs, same folds, same metric definitions (§5) | Model selection, Bayesian tuning |
| **D — Decoding & isolation** | HMM output is valid; route geometry uses real arcs only; negative test proves no ground truth in evaluator API responses | API/UI deployment |
| **E — Submission ready** | Two-laptop rehearsal done; all reported numbers reproducible from logs; report complete | Submit |

Unresolved decisions are listed in `docs/decisions.md` (single canonical path) as **OPEN** until settled. No later phase may silently assume an answer.

---

## 3. Data isolation and evaluator privacy (P0-1, P1-7)

| Mode | May show reference route? | Used for |
|---|---|---|
| **Offline / development** (train + student-test, scripts and notebooks) | Yes — we legitimately have these labels | EDA, report figures, Folium maps, error analysis |
| **Evaluator mode** (API + UI during final evaluation) | **No** | Live demonstration |

API rules in evaluator mode:
1. Evaluation data is loaded server-side only.
2. Inference never reads `.route`.
3. Scoring may read `.route` **only after** the prediction is complete, inside the server.
4. Responses contain: the predicted arcs and their geometry, the raw GPS, permitted **aggregate** metrics, and timings. They never contain reference arc ids or reference geometry.
5. An automated negative test asserts that no reference fields appear in any evaluator API response.
6. Tested on every student-test trip and on trip ids it has never seen; no special-case logic per trip.

---

## 4. Phases

Each phase lists: **Required**, **Optional**, **Depends on**, **Acceptance test**, **Open decisions**. Owners/reviewers are in §7 (editable).

### Phase 1 — Understand & explore → Gate A

- **Required:** robust loaders; invariant checks on all 90 trips; inventory (trips, file sizes, rows); missing/non-finite values; timestamp order, duplicates and gaps; geographic extent; network size, directedness, components, isolated nodes; whether consecutive route arcs are connected; at least one Folium map (GPS + network + reference route, offline mode); `experiments/results/data_quality_log.md`.
- **Never silently delete data.** Flag non-finite coordinates, duplicate timestamps and large gaps, and record how each is handled.
- **Optional:** more maps, per-trip anomaly ranking.
- **Acceptance test:** one command regenerates the inventory and checks; checks pass or failures are documented.
- **Open decisions:** projection (proposed EPSG:32632), gap threshold, duplicate-timestamp policy.
- **First-phase readiness package [UNCONFIRMED what is assessed]:** working loaders, real EDA statistics, ≥ 1 map, documented data issues, reproducible commands, and — **only if genuinely ready** — a baseline result. **Ask organisers** for the exact rubric and timing.

### Phase 2 — Labels, candidates, features → Gate B

- **Reference label alignment (P1-1, hard gate):**
  - Strategy 1 (baseline): each GPS point → nearest arc among the route's arcs (ignores order).
  - Strategy 2: **order-preserving alignment** — each GPS point is assigned a position in the route sequence that never goes backwards in time, minimising total point-to-arc distance (dynamic programming).
  - Compare the two; inspect ≥ 2 real trips including ambiguous cases (parallel roads, U-turns, revisits).
  - Each point gets an **ambiguity flag** (e.g. far from its aligned arc, or strategies disagree). Ambiguous points are reported separately and the rule for excluding them from metrics is documented.
  - Do not claim verified observation-level accuracy before this gate passes.
- **Candidates:** arcs within radius *R* of each point, capped at *k*, using a spatial index — **no brute-force N_GPS × N_arcs scan** (P2-1). Choose *R*, *k* from a miss-rate vs. size plot on training trips.
- **Features:** per (point, candidate). The team writes the rationale for each in `docs/features.md`. Distinguish straight-line distance from network distance (P2-6). Decide whether heading/speed are trustworthy across gaps and stationary periods.
- **Formulation [TEAM, initial]:** binary classification "is this candidate the reference arc?"; per point, pick the highest-scoring candidate.
- **Acceptance test:** feature table rebuilt by one command; miss-rate plot; alignment comparison table; ambiguous-case examples.
- **Open decisions:** *R*, *k*, alignment method, ambiguity threshold, class-imbalance handling.

### Phase 3 — Model experiments → Gate C

- **Required:** ≥ 5 distinct models, same folds, same features, same metrics (§5). Each run logged with a stable experiment id (P2-5).
- **CV (P1-5, P1-6):** group-aware CV by **trip** on the 80 training trips (scikit-learn `GroupKFold`). Record the actual fold membership and the pinned scikit-learn version; use a seed only where there is real randomness (models, subsampling).
- Scalers, imputers and calibration are fitted **inside each training fold** only (sklearn `Pipeline`).
- **Primary selection metric chosen before comparing results [TEAM, proposed]:** observation-level Top-1 accuracy in CV, with candidate-level log loss as the tie-breaker (probabilities feed the HMM).
- Student-test is used sparingly for documented comparisons; every use is logged.
- Large models/slow models (KNN, SVM) may be trained on a documented subsample; this is stated in the report.
- Loss curves **only** for models with real epoch histories.
- **Optional:** feature ablations, more models.

### Phase 4 — Selection and Bayesian optimisation

- Select by the pre-declared primary metric plus fold stability, probability quality, speed, HMM compatibility.
- Optuna TPE, objective = group-aware CV on training trips; search space justified; trials and wall-clock cost recorded.
- Untuned vs. tuned table: CV score, student-test Top-1, student-test macro-F1 (definition in §5), log loss, inference time.
- Export final model + preprocessing + config to `artifacts/`.

### Phase 5 — HMM + Viterbi → Gate D (part)

- Hidden states = candidate arcs per point (candidate ↔ arc mapping preserved). Emissions from the selected model; transitions from directed network connectivity. Viterbi in log-space.
- Handle: no candidates, all transitions impossible (break and restart the chain), large time gaps, disconnected components.
- **Two outputs, kept separate (P1-3):**
  1. the selected arc per observation (state sequence);
  2. the **reconstructed route**, expanded with real network paths between selected arcs, respecting direction. If no valid path exists, the route is marked as broken there — **connectivity is never invented**.
- Bounded shortest-path searches only (P2-1).
- Decoder parameters tuned mainly with training-trip CV; student-test only for documented comparison [OFFICIAL Models §9 allows train + student-test].
- Compare classifier-only vs. classifier + HMM per trip and aggregated; include a case where HMM did not help.

### Phase 6 — API (Laptop 1)

- **Interface contract written first (P1-7):** trip id format (8-digit string), `GET /health`, `GET /trips`, `POST /predict {trip_id}` → predicted arc ids, route GeoJSON (WGS84 lon/lat), metrics, timings; error payloads; data root from config.
- Follows §3 isolation rules; prints request logs to the console [OFFICIAL Models §6].
- **Acceptance test:** works on all 10 student-test trips; negative leak test passes; latency and cold-start measured.

### Phase 7 — UI (Laptop 2)

- Lists trips from `/trips`, sends the selected trip id, shows **raw GPS and the predicted route** (+ network if helpful), aggregate metrics and latency. **No reference route in evaluator mode.**
- Configurable API URL; clear error messages.
- Two-laptop rehearsal with screenshots of the UI and API console [OFFICIAL Deliverables §7].

### Phase 8 — Report

Follows Deliverables §2–7 exactly (table in v1 still applies). Reference-route maps come from **offline mode** only, with the partition stated. Report failures and limitations, including label-alignment uncertainty.

### Phase 9 — Viva preparation

Same as v1: each member explains a part they did not write; everyone can rerun a result live.

---

## 5. Evaluation protocol (P1-2) — `docs/evaluation_protocol.md`

| Level | Unit | Metrics | Notes |
|---|---|---|---|
| **Candidate** | (GPS point, candidate arc) pair | precision, recall, F1 for the positive class; **macro-F1 over the two classes {correct, incorrect}**; log loss | Clearly labelled as pair-level, never presented as arc-level accuracy |
| **Observation** | GPS point | **Top-1 accuracy** = chosen arc equals aligned reference arc | Denominator = evaluable points (non-ambiguous, reference arc in candidate set). Also report: all-points Top-1, # ambiguous, # zero-candidate points |
| **Candidate miss rate** | GPS point | fraction of evaluable points whose reference arc is not among candidates | Coverage reported separately |
| **Route** | Trip | route continuity = fraction of consecutive selected-arc transitions that are the same arc or connected; sequence overlap with reference (e.g. precision/recall of route arcs, edit distance with repeats handled) | Legitimate gaps (large time gaps) defined and reported |
| **Cost** | Trip / run | training time, inference time per trip | Environment (OS, CPU, RAM) recorded |

Metrics are computed per trip **and** aggregated (state whether point-weighted or trip-weighted). Fold-to-fold variation is reported for CV.

Arc ids are trip-local, so a global multiclass macro-F1 over raw arc ids is **not** used.

---

## 6. Repository layout (P2-7)

```
pathsnap/
├── README.md               # install, run, deploy, two-laptop setup, contributions, AI-use disclosure
├── requirements.txt        # pinned
├── .gitignore              # data, zips, .venv, caches, secrets
├── configs/default.yaml    # data root, CRS, R, k, seeds, model params
├── src/pathsnap/
│   ├── data_io.py          # loaders + invariant checks
│   ├── geo.py              # projection, point-to-segment geometry
│   ├── eda.py              # inventory & data-quality checks
│   ├── labels.py           # reference alignment strategies
│   ├── candidates.py       # spatial index + candidate search
│   ├── features.py         # feature engineering
│   ├── dataset.py          # builds the training table for a partition
│   ├── models.py           # model zoo (pipelines)
│   ├── evaluation.py       # metric definitions (§5)
│   ├── train.py            # CV + student-test experiment runner
│   ├── tuning.py           # Bayesian optimisation
│   ├── hmm.py              # emissions, transitions, Viterbi, route expansion
│   └── viz.py              # Folium maps
├── api/  ui/  tests/  docs/  experiments/  artifacts/  reports/figures/  maps/
```

Experiment manifest per run (P2-5): experiment id, partition, fold membership, code revision, seed, feature config, label method/version, model params, calibration, environment, timestamps, metrics, train/inference time, artifact path.

---

## 7. Team roles (editable — P2-3)

Owner/reviewer pairs as in v1 (A: EDA/maps/UI, B: candidates/features/API, C: models/tuning, D: HMM/report). These are **our assignments, not organiser requirements**. Candidate generation, labels and features are tightly coupled: one owner, with a shared interface (`dataset.py`) used by everyone else. Everyone must understand the whole system.

---

## 8. Timeline — milestones, not promises (P2-4)

Today's working target [TEAM]: **by 12:00 — preprocessing, feature extraction and the first five model runs done.**

| Milestone | Baseline (must have) | Target | Fallback if late |
|---|---|---|---|
| **M1 ~11:15** Gate A | Loaders + checks + inventory | + Folium map | Map later, inventory first |
| **M2 ~11:35** Gate B | Alignment (2 strategies) + candidates + features for train & student-test | + miss-rate plot | Fixed *R*/*k*, document as initial |
| **M3 ~12:00** Gate C | ≥ 5 models with CV + student-test metrics logged | 6–7 models | Subsample slow models (documented) |
| M4 | Bayesian tuning of selected model | + importance plots | Fewer trials, documented |
| M5 | HMM + route expansion + comparison | per-trip analysis | Simple transitions first |
| M6 | API + UI + leak test | rehearsal on two laptops | Single-laptop dry run first |
| M7 | Report + viva prep | polish | — |

**Critical path:** loaders → labels → candidates → features → 5 models → tuning → HMM → API → UI → report. Parallel work starts only after `dataset.py`'s output format is fixed. Keep a testing buffer before submission.

---

## 9. Checklists

**Originality & integrity:** v1 checklist plus —
- [ ] No reference route in evaluator UI or evaluator API responses (negative test passes).
- [ ] Primary metric declared before comparing models.
- [ ] Fold membership, seeds and every student-test consultation logged.
- [ ] Negative results and abandoned ideas recorded in `docs/decisions.md`.
- [ ] AI-use disclosure is accurate and specific (which parts, which tool).

**Submission:** unchanged from v1.

---

## Change log (v1 → v2)

| Review item | What changed | Where |
|---|---|---|
| P0-1 Ground truth in UI | UI shows predicted route only in evaluator mode; offline vs. evaluator modes; negative leak test | §3, Phase 6–7 |
| P0-2 First-phase claim | Removed "Phases 1–3 before lunch"; added readiness package and "confirm with organisers" | §0, Phase 1, §8 |
| P0-3 Five-model rule | "≥ 5 distinct models, families encouraged" | §0, Phase 3 |
| P0-4 Delimiter | Whitespace-separated for all files | §1 |
| P1-1 Label alignment | Hard gate, 2 strategies, ambiguity flag | §2 Gate B, Phase 2 |
| P1-2 Metrics | Evaluation protocol by level, denominators, macro-F1 defined | §5 |
| P1-3 Route reconstruction | State sequence vs. expanded route; no invented connectivity | Phase 5 |
| P1-4 Trip-local ids | Invariants list | §1 |
| P1-5 Partition discipline | Trip-level folds, in-fold preprocessing, pre-declared metric, logged student-test use | Phase 3–5 |
| P1-6 CV seed | GroupKFold described correctly; fold membership recorded | Phase 3 |
| P1-7 I/O contract | Contract + isolation rules | §3, Phase 6 |
| P2-1 Performance | No brute-force scans; bounded path search; timings | Phase 2, 5 |
| P2-2 Originality | Hypotheses tested via ablations, negative results recorded | §9 |
| P2-3 Roles | Marked editable; coupled modules have one owner | §7 |
| P2-4 Timeline | Milestones with baseline/target/fallback | §8 |
| P2-5 Artefacts | Experiment manifest | §6 |
| P2-6 Geometry/gaps | Never silently delete; heading/speed across gaps | Phase 1–2 |
| P2-7 Repo | `src/pathsnap/` package, tests | §6 |
| P2-8 Stage 1 gate | Gate A + B | §2 |
| Not adopted | The review's "stop after the plan" instruction — the team asked to start building now | — |
