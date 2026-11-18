# Two-laptop demo

| | Laptop 1: backend (this Mac) | Laptop 2: user interface |
|---|---|---|
| Runs | FastAPI + model + HMM + evaluator trips | The UI web page only |
| Terminal shows | One log line per request (method, client IP, status, latency, trip metrics) | The page requests from the browser |
| Needs | The repo with `.venv`, and `../evaluator_data/evaluator_test/` | Python 3 (any) and a browser. Internet is only needed for map tiles |

Both laptops must be on the **same Wi-Fi**. If the venue Wi-Fi blocks laptops from talking to each other, connect both to one phone hotspot.

Rules from the organisers: everything runs in the terminal, and you must not kill any process during the evaluation. So **start both servers before the evaluator arrives and leave them running**. Close nothing until the evaluation is over.

---

## Laptop 1: backend

1. Open a terminal:
   ```bash
   cd ~/Documents/ML_Hackathon/pathsnap && HOST=0.0.0.0 scripts/demo_api.sh
   ```
   Wait for `Uvicorn running on http://0.0.0.0:8000`.
2. Find this laptop's IP address in a **second** terminal. Leave the first one alone.
   ```bash
   ipconfig getifaddr en0
   ```
   On 18 Nov at the hackathon venue it was `10.20.201.72`. **It can change** when the network changes, so check it on the day.
3. Check from Laptop 1: `curl http://<IP>:8000/health` should print `"status":"ok"` with `"n_trips":10`.
4. Leave terminal 1 on screen. This is where the evaluator sees the API logs.

## Laptop 2: user interface

1. Get the code. Only the `ui/` folder is needed:
   ```bash
   git clone https://github.com/Musaleee/PathSnap-ML-Hackathon.git
   cd PathSnap-ML-Hackathon
   ```
   Or copy the `ui/` folder over on a USB stick.
2. Start the UI in a terminal, using Laptop 1's IP:
   - macOS / Linux:
     ```bash
     scripts/demo_ui.sh http://<IP>:8000
     ```
   - Windows:
     ```bat
     scripts\demo_ui.bat http://<IP>:8000
     ```
   The browser opens with the API address already filled in. If not, open `http://localhost:5500` and type `http://<IP>:8000` in the API box.
3. Leave this terminal running too.

## What to show the evaluator

1. On Laptop 2, click **Connect**. The status shows `connected · model hist_gb_tuned.joblib · 10 trips`.
2. The evaluator picks any trip and clicks **Predict selected trip**.
   - The map shows the raw GPS (blue) and the predicted route (red). Use the layer switch at the top right to toggle them.
   - The panel shows the metrics computed **by the API**: Top-1 with HMM and with the classifier alone, route continuity, precision, recall, F1, edit distance and timings.
3. On Laptop 1, point to the new log lines. They show Laptop 2's IP as the client:
   ```
   POST /predict from <laptop-2-ip> -> 200 in … ms
   trip 000000xx: … points, … route arcs, metrics {...}
   ```
4. Repeat with any other trips. All 10 work, each in under 1 second.
5. **Ground truth never leaves Laptop 1.**
   - Inference does not read `.route`. The server reads it only after predicting, to score.
   - The response contains no reference route.
   - If asked, prove it in a **new** terminal on Laptop 1, without touching the running ones:
     ```bash
     cd ~/Documents/ML_Hackathon/pathsnap && PYTHONPATH=src .venv/bin/python -m pytest -q tests
     ```
     All 17 tests pass, including the leak test. The tests use their own in-process copy of the API, so they do not disturb the running server.

Expected results (`experiments/evaluator_test_final/RESULTS.md`):

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

## Troubleshooting (without stopping anything)

| Problem | Fix |
|---|---|
| UI: "Cannot reach the API" | Check the IP in the API box and click Connect again. From Laptop 2, `curl http://<IP>:8000/health` should answer. If it doesn't, the Wi-Fi is isolating devices, so use a phone hotspot for both laptops |
| macOS asks "accept incoming connections?" for Python | Click **Allow** |
| Windows firewall prompt on Laptop 2 | Allow, on private networks |
| Map is grey | No internet on Laptop 2, so tiles can't load. The route line and metrics still show |
| Port 8000 / 5500 already in use (only at start-up) | Use another port: `PORT=8001 HOST=0.0.0.0 scripts/demo_api.sh` and put `:8001` in the UI |

Single-laptop rehearsal: run `scripts/demo_api.sh` and `scripts/demo_ui.sh` on one machine (API box `http://localhost:8000`).
