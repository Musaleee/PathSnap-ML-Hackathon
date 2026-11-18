# Demo on one laptop (macOS)

Folder layout on this Mac:
```
ML_Hackathon/
├── pathsnap/                      ← this repo (with .venv)
└── evaluator_data/evaluator_test/ ← the 10 evaluation trips (not in git)
```

## Start (two terminals, about 10 seconds)

**Terminal 1: API.** Keep it on screen, because the evaluator must see the request logs.
```bash
cd ~/Documents/ML_Hackathon/pathsnap && scripts/demo_api.sh
```
Wait for `Uvicorn running on http://127.0.0.1:8000`.

**Terminal 2: UI.** This also opens http://localhost:5500 in the browser.
```bash
cd ~/Documents/ML_Hackathon/pathsnap && scripts/demo_ui.sh
```

Another trips folder: `scripts/demo_api.sh /path/to/folder`.
To use a second laptop for the UI: `HOST=0.0.0.0 scripts/demo_api.sh`.

## What to show (about 5 minutes)

1. **UI:** the API box says `http://localhost:8000`. Click **Connect**. The status shows the model `hist_gb_tuned.joblib` and 10 trips.
2. Pick any trip and click **Predict selected trip**.
   - The map shows the raw GPS (blue dots) and the predicted route (red line). Use the layer switch to toggle them.
   - The panel shows the metrics computed by the API: Top-1 with HMM and with the classifier alone, route continuity, precision, recall, F1 and edit distance, plus timings.
3. **Terminal 1:** point to the log lines for that request:
   ```
   POST /predict from 127.0.0.1 -> 200 in … ms
   trip 000000xx: … points, … route arcs, metrics {...}
   ```
4. Repeat with 2–3 more trips. All 10 work, each in under 1 second.
5. **Ground truth stays on the server.**
   - Inference never reads `.route`. The server reads it only afterwards, to score.
   - The response has no reference route. If asked, show this live:
     ```bash
     curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"trip_id":"00000049"}' | python3 -m json.tool | head -40
     ```
   - The automated proof is `PYTHONPATH=src .venv/bin/python -m pytest -q tests` (17 tests, including the leak test).

Expected results (evaluator run, `experiments/evaluator_test_final/RESULTS.md`):

| Trip | Top-1 (HMM) | Route F1 |
|---|---|---|
| 15 | 0.921 | 0.766 |
| 17 | 0.972 | 0.953 |
| 23 | 0.982 | 0.922 |
| 30 | 0.895 | 0.954 |
| 49 | 0.999 | 0.999 |
| 59 | 0.989 | 0.971 |
| 67 | 0.990 | 0.988 |
| 73 | 0.991 | 0.996 |
| 87 | 0.998 | 0.999 |
| 94 | 0.977 | 0.983 |

Overall: Top-1 0.962 (classifier alone 0.928), route F1 0.953.

## Extra things to have open
- Folium maps with the reference route (offline mode only): `maps/student_test_00000077.html`, `maps/eda_train_00000001.html`
- Results: `reports/report_tables.md`
- Decisions: `docs/decisions.md`

## If something goes wrong

| Problem | Fix |
|---|---|
| UI shows "Cannot reach the API" | Check that terminal 1 is running and the API box says `http://localhost:8000`, then click Connect again |
| `Address already in use` | Something is already on the port. Run `lsof -ti:8000 \| xargs kill` (or `:5500`), then start again |
| Map is grey | No internet, so map tiles can't load. The route still draws, and the metrics still show |
| `Trips folder not found` | Pass the folder: `scripts/demo_api.sh ../evaluator_data/evaluator_test` |
| `No module named …` | Run `.venv/bin/pip install -r requirements.txt` |

To stop: press Ctrl+C in both terminals.
