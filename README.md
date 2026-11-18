# PathSnap: Map-Matching and Route Reconstruction

PathSnap matches noisy GPS trajectories to a directed road network. The pipeline is:
1. Candidate road segments for each GPS point (KD-tree)
2. A classifier that scores each (point, candidate) pair (tuned HistGradientBoosting)
3. HMM/Viterbi decoding over the candidates
4. Route reconstruction with real network paths
5. A FastAPI backend (Laptop 1) and a web UI (Laptop 2)

| Guide | File |
|---|---|
| Run in VS Code, step by step | `RUN_LOCALLY.md` |
| Two-laptop deployment | `docs/deployment.md` |
| API fields | `docs/api_contract.md` |
| Decisions | `docs/decisions.md` |
| Metrics | `docs/evaluation_protocol.md` |
| Features | `docs/features.md` |
| Report tables (generated) | `reports/report_tables.md` |
| Report draft | `reports/REPORT_DRAFT.md` |

## Results (student-test, 10 trips)

| Stage | Top-1 | Macro-F1 | Route F1 | Route continuity |
|---|---|---|---|---|
| Best untuned model (HistGB) | 0.9331 | 0.9555 | – | 0.968 |
| Tuned HistGB, classifier only | 0.9373 | 0.9595 | 0.824 | 0.972 |
| **Tuned HistGB + HMM/Viterbi (final)** | **0.9515** | – | **0.968** | **0.984** |

- 7 models were compared with GroupKFold(5) CV over the 80 train trips. Table: `experiments/results/model_comparison.csv`.
- Bayesian tuning used Optuna TPE with 20 trials: `experiments/optuna/`.
- The HMM comparison is in `experiments/results/hmm_comparison.csv`.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
Data is **not** in the repo. Put the provided `Student folder` (with `train/` and `student_test/`) next to this repo, or change `data_root` in `configs/default.yaml`.

## Reproduce (in order, from the repo root, with `PYTHONPATH=src`)
```bash
python -m pathsnap.eda               # inventory + data checks
python -m pathsnap.dataset           # labels, candidates, features (cache)
python -m pathsnap.train             # 7 models: CV + student-test
python -m pathsnap.plots             # comparison figures
python -m pathsnap.tuning 20         # Bayesian optimisation of HistGB -> artifacts/final/
python -m pathsnap.hmm_experiment    # classifier-only vs HMM -> artifacts/final/hmm_config.json
python -m pathsnap.viz               # Folium maps -> maps/
python -m pathsnap.report_figures    # EDA, confusion matrices, tuning and HMM figures
python -m pathsnap.report_tables     # reports/report_tables.md
python -m pytest -q tests            # API tests, including the ground-truth leak test
```

## Deploy (details in `docs/deployment.md`)
- **Laptop 1 (API):**
  ```bash
  PATHSNAP_DATA_DIR=/path/to/evaluator_test .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```
- **Laptop 2 (UI):**
  ```bash
  python3 -m http.server 5500 --directory ui
  ```
  Then open http://localhost:5500 and enter `http://<laptop-1-ip>:8000`.

## Layout
| Path | Contents |
|---|---|
| `src/pathsnap/data_io.py`, `eda.py` | Readers, invariant checks, inventory |
| `src/pathsnap/geo.py` | Per-trip UTM projection, point-to-segment geometry |
| `src/pathsnap/labels.py` | Reference-label alignment (monotonic DP vs nearest baseline) |
| `src/pathsnap/candidates.py`, `features.py`, `dataset.py` | Candidate search, 24 features, training tables |
| `src/pathsnap/models.py`, `train.py`, `evaluation.py`, `plots.py` | Model zoo, experiment runner, metrics, figures |
| `src/pathsnap/tuning.py` | Optuna TPE search for HistGB |
| `src/pathsnap/hmm.py`, `hmm_experiment.py` | Viterbi decoder, route expansion, sequence metrics, comparison |
| `src/pathsnap/pipeline.py` | End-to-end inference for one trip (used by the API and the maps) |
| `src/pathsnap/viz.py` | Folium maps |
| `src/pathsnap/evaluate_holdout.py` | Test-only scoring of saved models on a held-out folder |
| `src/pathsnap/report_figures.py`, `report_tables.py` | Report figures and tables from logged results |
| `api/main.py` | FastAPI backend |
| `ui/index.html` | Evaluator UI (Leaflet) |
| `tests/` | API and leak tests |
| `artifacts/final/` | Final tuned model, model config, HMM config |
| `artifacts/candidates/` | The 7 compared models (large files are not committed) |
| `experiments/` | Logs (one JSON per run), results CSVs, Optuna study, evaluator-test run |
| `maps/`, `reports/figures/` | Interactive maps, figures, screenshots |

## Data isolation
- Evaluator data was used only for one test-only run of the Phase-1 models (`experiments/evaluator_test/`), never for any choice.
- At inference the API never reads `.route`. It scores server-side after prediction and returns aggregate metrics only. `tests/test_api.py` checks this.

## Team and contributions
_Fill in: member, areas owned, areas reviewed._

## AI-use disclosure
The following were drafted with an AI assistant (Claude, Anthropic):
- the project plan
- the code for data loading, EDA, label alignment, candidate generation, feature extraction, model training, Bayesian tuning, the HMM decoder, the API, the UI, the maps and the report figures and tables
- the report draft

The team reviewed, ran and is responsible for all of it. Changes made afterwards are in the commit history and `docs/decisions.md`. The final report text is written by the team.
