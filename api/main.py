"""PathSnap evaluation API (runs on Laptop 1).

Run (from the repo root):
    PATHSNAP_DATA_DIR="/path/to/evaluator_test" .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            status, model and HMM configuration, data folder
    GET  /trips             trip ids available on the server
    POST /predict           {"trip_id": "00000015"} -> predicted arcs, route GeoJSON, GPS, metrics, timings

Isolation rules (docs/api_contract.md): inference never reads .route; the server reads .route only after
prediction to compute aggregate metrics; responses never contain reference arc ids or geometry.
"""
import logging
import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import load_config, partition_dir  # noqa: E402
from pathsnap.pipeline import Predictor, route_lines      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

CFG = load_config()
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR") or partition_dir(CFG, "student_test")).resolve()
TRIP_RE = re.compile(r"^\d{8}$")

t0 = time.time()
PREDICTOR = Predictor(cfg=CFG)
COLD_START_S = time.time() - t0
log.info("model loaded from %s in %.2f s; HMM %s; data folder %s",
         PREDICTOR.model_path.name, COLD_START_S, PREDICTOR.hmm, DATA_DIR)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t = time.time()
    resp = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info("%s %s from %s -> %d in %.0f ms", request.method, request.url.path, client,
             resp.status_code, (time.time() - t) * 1000)
    return resp


def _trips():
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and TRIP_RE.match(p.name))


@app.get("/health")
def health():
    return dict(status="ok", model=PREDICTOR.model_path.name, hmm=PREDICTOR.hmm.__dict__,
                data_folder=DATA_DIR.name, n_trips=len(_trips()), cold_start_s=round(COLD_START_S, 3))


@app.get("/trips")
def trips():
    return dict(trips=_trips())


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if not TRIP_RE.match(tid):
        raise HTTPException(422, detail="trip_id must be an 8-digit string, e.g. '00000015'")
    d = DATA_DIR / tid
    if not d.is_dir():
        raise HTTPException(404, detail=f"trip {tid} not found on the server")
    t = time.time()
    try:
        pred = PREDICTOR.predict(d)          # never reads .route
    except Exception as e:                   # noqa: BLE001
        log.exception("prediction failed for %s", tid)
        raise HTTPException(500, detail=f"prediction failed: {e}")
    predict_s = time.time() - t
    t = time.time()
    metrics = PREDICTOR.score(d, pred)       # server-side, after prediction; aggregates only
    score_s = time.time() - t

    trip = pred["trip"]
    lon, lat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    lines = route_lines(pred["route_arcs"], trip.arcs.to_numpy(), lon, lat)
    route_geojson = dict(type="FeatureCollection", features=[
        dict(type="Feature", properties=dict(layer="predicted_route", part=i),
             geometry=dict(type="LineString", coordinates=ln)) for i, ln in enumerate(lines)])
    timings = {k: round(v, 3) for k, v in pred["timings"].items()} | dict(
        predict_total_s=round(predict_s, 3), scoring_s=round(score_s, 3))
    log.info("trip %s: %d points, %d route arcs, metrics %s, predict %.2f s",
             tid, pred["n_points"], sum(a >= 0 for a in pred["route_arcs"]), metrics, predict_s)
    return dict(
        trip_id=tid,
        n_points=pred["n_points"],
        n_points_without_candidates=pred["n_no_candidates"],
        gps=[[float(x), float(y)] for x, y in zip(trip.track.lon, trip.track.lat)],
        predicted_point_arcs=[int(a) for a in pred["point_arcs"]],
        predicted_route_arcs=[int(a) for a in pred["route_arcs"]],
        predicted_route_geojson=route_geojson,
        chain_breaks=pred["chain_breaks"],
        route_gaps=pred["route_gaps"],
        metrics=metrics,
        timings=timings,
    )
