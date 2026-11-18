# Two-laptop deployment

Both laptops must be on the same Wi-Fi network. Laptop 2 needs internet access for map tiles and the Leaflet library.

## Laptop 1: API
1. Clone the repository, create the venv and install dependencies (see `RUN_LOCALLY.md`).
2. Make sure `artifacts/final/hist_gb_tuned.joblib` and `artifacts/final/hmm_config.json` exist. They are committed; to rebuild them run `pathsnap.dataset`, `pathsnap.tuning` and `pathsnap.hmm_experiment`.
3. Put the evaluation trips in a folder, for example `~/pathsnap_eval/evaluator_test/00000015/...`.
4. Start the API, which listens on all interfaces:
   ```bash
   PATHSNAP_DATA_DIR=~/pathsnap_eval/evaluator_test PYTHONWARNINGS=ignore .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   On Windows (PowerShell):
   ```powershell
   $env:PATHSNAP_DATA_DIR="C:\pathsnap_eval\evaluator_test"; .venv\Scripts\uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
5. Find the IP address: `ipconfig getifaddr en0` on macOS, `ipconfig` on Windows. Allow port 8000 through the firewall if you are asked.
6. Keep this terminal visible. It prints one log line per request, which the evaluator needs to see.

## Laptop 2: UI
1. Copy the `ui/` folder; nothing else is needed.
2. Serve it with `python3 -m http.server 5500 --directory ui`, then open `http://localhost:5500`. Opening `ui/index.html` directly also works.
3. Enter `http://<laptop-1-ip>:8000` in the API box and click **Connect**. The 10 trips are listed.
4. Select a trip and click **Predict selected trip**. The map shows the raw GPS and the predicted route. The panel shows the metrics computed by the API and the timings.

## Checks before the evaluation
- `curl http://<laptop-1-ip>:8000/health` from Laptop 2 returns `"status":"ok"`.
- Run `PYTHONPATH=src .venv/bin/python -m pytest -q tests` on Laptop 1. All tests should pass.
- Run all 10 trips once and take screenshots of the UI and of the API console for the report (Deliverables §7).
