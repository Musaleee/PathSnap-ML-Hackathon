"""API tests, including the negative leak test (Gate D).

Run: PYTHONPATH=src .venv/bin/python -m pytest -q tests
Uses the student-test partition (the evaluator data is not needed and not used).
"""
import json
import shutil

import pytest
from fastapi.testclient import TestClient

import api.main as api
from pathsnap.data_io import load_trip

client = TestClient(api.app)
TRIPS = api._trips()
ALLOWED = {"trip_id", "n_points", "n_points_without_candidates", "gps", "predicted_point_arcs",
           "predicted_route_arcs", "predicted_route_geojson", "chain_breaks", "route_gaps", "metrics", "timings"}
METRIC_KEYS = {"n_evaluable", "top1_hmm", "top1_classifier_only", "route_continuity", "route_precision",
               "route_recall", "route_f1", "route_norm_edit"}


def test_health_and_trips():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/trips").json()["trips"] == TRIPS and len(TRIPS) == 10


@pytest.mark.parametrize("bad", ["15", "abcdefgh", "../../etc", "999999999"])
def test_bad_ids(bad):
    assert client.post("/predict", json={"trip_id": bad}).status_code in (404, 422)


def test_unknown_trip():
    assert client.post("/predict", json={"trip_id": "12345678"}).status_code == 404


@pytest.mark.parametrize("tid", TRIPS)
def test_predict_no_reference_leak(tid):
    r = client.post("/predict", json={"trip_id": tid})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == ALLOWED                                  # no extra (reference) fields
    assert set(body["metrics"]) == METRIC_KEYS                    # aggregates only
    text = json.dumps(body).lower()
    for word in ("ref_arc", "reference", "ground_truth", "\"route\"", "label"):
        assert word not in text
    ref = load_trip(api.DATA_DIR / tid).route.tolist()
    assert body["predicted_route_arcs"] != ref or len(ref) < 3   # the reference list is never echoed
    assert len(body["predicted_point_arcs"]) == body["n_points"]


def test_prediction_does_not_depend_on_route(tmp_path):
    """Same prediction when the .route file is removed: inference never reads the reference."""
    tid = TRIPS[0]
    src = api.DATA_DIR / tid
    dst = tmp_path / tid
    shutil.copytree(src, dst)
    (dst / f"{tid}.route").unlink()
    a = api.PREDICTOR.predict(src)
    b = api.PREDICTOR.predict(dst)
    assert (a["point_arcs"] == b["point_arcs"]).all() and a["route_arcs"] == b["route_arcs"]
    assert api.PREDICTOR.score(dst, b) is None                   # no reference -> no metrics
