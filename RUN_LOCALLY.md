# Running PathSnap locally in VS Code

## 1. What you need

- **Python 3.9 or newer** (tested with 3.9.6). Check with `python3 --version` (macOS/Linux) or `python --version` (Windows).
- **VS Code** with the **Python** extension (Microsoft).
- The dataset folder **`Student folder`** (contains `train/` and `student_test/`).

## 2. Folder layout

Unzip `PathSnap_Phase1.zip` and put the dataset **next to** the `pathsnap` folder:

```
ML_Hackathon/
├── pathsnap/              ← from the zip (open THIS folder in VS Code)
└── Student folder/
    ├── train/
    └── student_test/
```

If your data is somewhere else, edit `data_root` in `pathsnap/configs/default.yaml` (relative to `pathsnap/` or an absolute path).

## 3. Open in VS Code

1. **File → Open Folder…** → choose `pathsnap`.
2. Open the terminal: **Terminal → New Terminal**. It opens inside `pathsnap/`.

## 4. Create the virtual environment and install packages

macOS / Linux:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

Then select the interpreter: **Cmd/Ctrl + Shift + P → "Python: Select Interpreter"** → choose the one in `.venv`.

## 5. Tell Python where the code is

The code is in `src/pathsnap`, so `src` must be on `PYTHONPATH`.

macOS / Linux (in the same terminal, every new terminal):

```bash
export PYTHONPATH=src
```

Windows (PowerShell):

```powershell
$env:PYTHONPATH = "src"
```

Optional, so VS Code always knows it: create a file `.env` in `pathsnap/` with this one line:

```
PYTHONPATH=src
```

## 6. Run the pipeline (in this order)

| Step | Command | Time* | Output |
|---|---|---|---|
| 1. Data inventory & checks | `python -m pathsnap.eda` | ~5 s | `experiments/results/inventory.csv`, `data_quality_summary.txt` |
| 2. Labels, candidates, features | `python -m pathsnap.dataset` | ~15 s | `experiments/cache/*.pkl`, `experiments/results/labels_candidates_*.csv` |
| 3. Train all 7 models | `python -m pathsnap.train` | ~8 min | `experiments/logs/*.json`, `experiments/results/model_comparison.csv`, `artifacts/candidates/*.joblib` |
| 4. Figures | `python -m pathsnap.plots` | ~5 s | `reports/figures/*.png` |
| 5. Bayesian tuning (HistGB) | `python -m pathsnap.tuning 20` | ~45 min | `experiments/optuna/`, `experiments/results/tuning_comparison.csv`, `artifacts/final/hist_gb_tuned.joblib` |
| 6. HMM vs classifier | `python -m pathsnap.hmm_experiment` | ~5 min | `experiments/results/hmm_*.csv`, `artifacts/final/hmm_config.json` |
| 7. Folium maps | `python -m pathsnap.viz` | ~10 s | `maps/*.html` |
| 8. Report figures | `python -m pathsnap.report_figures` | ~1 min | `reports/figures/*.png` |
| 9. Tests (API leak test) | `python -m pytest -q tests` | ~10 s | – |
| 10. API | `.venv/bin/uvicorn api.main:app --port 8000` | – | http://localhost:8000/health |
| 11. UI | `python3 -m http.server 5500 --directory ui` | – | http://localhost:5500 |

Two-laptop setup for the final evaluation: `docs/deployment.md`. API fields: `docs/api_contract.md`.

\* measured on a 10-core Mac with 16 GB RAM. Step 2 needs about 2 GB RAM; step 3 more for Random Forest.

Train only some models (faster):

```bash
python -m pathsnap.train hist_gb mlp random_forest
```

Model names: `logreg`, `gaussian_nb`, `knn`, `random_forest`, `extra_trees`, `hist_gb`, `mlp`.

> **Note:** `train` **appends** one row per run to `experiments/results/model_comparison.csv`. `plots` uses the latest row per model. To start fresh, delete that CSV first.

## 7. Run from VS Code's Run button (optional)

Create `.vscode/launch.json` in `pathsnap/`:

```json
{
  "version": "0.2.0",
  "configurations": [
    { "name": "1 EDA",      "type": "debugpy", "request": "launch", "module": "pathsnap.eda",     "cwd": "${workspaceFolder}", "env": { "PYTHONPATH": "src" } },
    { "name": "2 Dataset",  "type": "debugpy", "request": "launch", "module": "pathsnap.dataset", "cwd": "${workspaceFolder}", "env": { "PYTHONPATH": "src" } },
    { "name": "3 Train",    "type": "debugpy", "request": "launch", "module": "pathsnap.train",   "cwd": "${workspaceFolder}", "env": { "PYTHONPATH": "src" } },
    { "name": "4 Plots",    "type": "debugpy", "request": "launch", "module": "pathsnap.plots",   "cwd": "${workspaceFolder}", "env": { "PYTHONPATH": "src" } }
  ]
}
```

Then pick a configuration in the **Run and Debug** panel (Cmd/Ctrl + Shift + D) and press ▶. You can set breakpoints in any file under `src/pathsnap/`.

## 8. Where to look

| What | File |
|---|---|
| Model comparison table | `experiments/results/model_comparison.csv` |
| Per-run details (params, folds, per-fold metrics) | `experiments/logs/<id>.json` |
| Data-quality summary | `experiments/results/data_quality_summary.txt` |
| Label / candidate stats per trip | `experiments/results/labels_candidates_train.csv` |
| Charts | `reports/figures/` |
| Decisions & open questions | `docs/decisions.md` |
| Feature explanations | `docs/features.md` |
| Metric definitions | `docs/evaluation_protocol.md` |

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pathsnap'` | `PYTHONPATH` not set (step 5), or the terminal is not in `pathsnap/` |
| `FileNotFoundError` for `Student folder` | Check the layout in step 2 or `data_root` in `configs/default.yaml` |
| `FileNotFoundError ... experiments/cache/train.pkl` when training | Run step 2 (`pathsnap.dataset`) first |
| Many `RuntimeWarning: ... encountered in matmul` lines on macOS | Harmless numpy/Accelerate warnings. Hide them with `export PYTHONWARNINGS=ignore` |
| `pip install` fails on a pinned version | Your Python is too new/old for that pin. Use Python 3.9–3.12, or remove the `==version` for that package |
| Training too slow / out of memory | Train fewer models (step 6), or close other apps. KNN and MLP already use subsamples |
| Want LightGBM / XGBoost | macOS needs `brew install libomp` first; then `pip install lightgbm` |
