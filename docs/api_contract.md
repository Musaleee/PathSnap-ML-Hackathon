# API contract (Laptop 1)

Base URL: `http://<laptop-1-ip>:8000`. JSON only. CORS is open so the UI on Laptop 2 can call it.
Data folder: environment variable `PATHSNAP_DATA_DIR`, a folder of 8-digit trip folders (each with `.track`, `.nodes`, `.arcs` and, server-side only, `.route`). Default: `student_test` from `configs/default.yaml`.

| Method | Path | Request | Response |
|---|---|---|---|
| GET | `/health` | – | `status`, `model` (file name), `hmm` (decoder parameters), `data_folder`, `n_trips`, `cold_start_s` |
| GET | `/trips` | – | `{"trips": ["00000015", ...]}` |
| POST | `/predict` | `{"trip_id": "00000015"}` | see below |

`/predict` response fields (no other fields are ever returned):

| Field | Meaning |
|---|---|
| `trip_id` | echo |
| `n_points`, `n_points_without_candidates` | counts |
| `gps` | raw GPS `[lon, lat]` per observation |
| `predicted_point_arcs` | predicted arc id per GPS observation (-1 = no candidate) |
| `predicted_route_arcs` | reconstructed route, consecutive real arcs; -1 marks a gap |
| `predicted_route_geojson` | FeatureCollection of LineStrings (WGS84 lon/lat) of the predicted route |
| `chain_breaks`, `route_gaps` | HMM restarts (time gaps / impossible transitions) and gaps in the route |
| `metrics` | aggregates only: `n_evaluable`, `top1_hmm`, `top1_classifier_only`, `route_continuity`, `route_precision`, `route_recall`, `route_f1`, `route_norm_edit`; `null` if the server has no `.route` |
| `timings` | `load_s`, `features_s`, `model_s`, `hmm_s`, `predict_total_s`, `scoring_s` |

Errors: `422` invalid id (must match `^\d{8}$`), `404` unknown trip, `500` prediction failure, each with `{"detail": "..."}`.

## Isolation rules
1. Inference loads the trip with `load_route=False`; the `.route` file is not opened before the prediction is complete.
2. Scoring reads `.route` inside the server afterwards and returns aggregate metrics only.
3. Responses never contain reference arc ids, reference geometry or per-point labels.
4. Checked by `tests/test_api.py`: response fields are whitelisted, forbidden words are absent, and predictions are identical when the `.route` file is deleted.
5. Every request is logged to the console (method, path, client IP, status, latency, trip metrics), so the evaluator can see the logs on Laptop 1.
