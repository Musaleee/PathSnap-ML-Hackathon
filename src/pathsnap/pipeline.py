"""End-to-end inference for one trip folder (used by the API and the map scripts).

predict() never reads the .route file (load_trip(load_route=False)). score() is a separate, later
step that reads .route server-side and returns aggregate metrics only.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .config import REPO_ROOT, load_config
from .data_io import load_trip
from .dataset import prepare_trip
from .evaluation import route_continuity
from .features import FEATURES
from .hmm import Graph, HMMParams, expand_route, precompute, route_metrics, viterbi
from .labels import align

FINAL = REPO_ROOT / "artifacts" / "final"


class Predictor:
    def __init__(self, model_path=None, hmm_path=None, cfg=None):
        self.cfg = cfg or load_config()
        model_path = Path(model_path or FINAL / "hist_gb_tuned.joblib")
        if not model_path.exists():                      # fall back to the untuned selected model
            model_path = REPO_ROOT / "artifacts" / "candidates" / "hist_gb.joblib"
        self.model_path = model_path
        self.model = joblib.load(model_path)
        hmm_path = Path(hmm_path or FINAL / "hmm_config.json")
        self.hmm = HMMParams(**json.loads(hmm_path.read_text())["hmm"]) if hmm_path.exists() else HMMParams()

    def predict(self, trip_dir):
        tm = {}
        t0 = time.time()
        trip = load_trip(trip_dir, load_route=False)
        assert trip.route is None
        tm["load_s"] = time.time() - t0

        t0 = time.time()
        feats, (gx, gy, nx, ny, idx) = prepare_trip(trip, self.cfg)
        tm["features_s"] = time.time() - t0

        t0 = time.time()
        prob = self.model.predict_proba(feats[FEATURES])[:, 1] if len(feats) else np.empty(0)
        tm["model_s"] = time.time() - t0

        t0 = time.time()
        a = trip.arcs.to_numpy()
        g = Graph(a, idx.length)
        pts, arc_c = feats.point.to_numpy(), feats.arc.to_numpy()
        pre = precompute(pts, arc_c, feats.frac_along.to_numpy(), trip.track.t.to_numpy(), gx, gy, g, self.hmm)
        rows, brk = viterbi(pre, prob, self.hmm)
        hmm_pts, hmm_arcs = pre["upts"], arc_c[rows]
        seq = hmm_arcs.copy()
        if len(seq):
            parts = np.split(seq, np.flatnonzero(brk)[1:])
            seq = np.concatenate([np.append(p, -1) for p in parts])[:-1]
        route, gaps = expand_route(seq, g, self.hmm)
        tm["hmm_s"] = time.time() - t0

        # classifier-only choice, kept for comparison
        best = feats.assign(p=prob).reset_index(drop=True).groupby("point", sort=True).p.idxmax().to_numpy()
        n = len(trip.track)
        pred = np.full(n, -1)
        pred[hmm_pts] = hmm_arcs
        clf = np.full(n, -1)
        clf[feats.point.to_numpy()[best]] = arc_c[best]
        return dict(trip=trip, n_points=n, point_arcs=pred, clf_point_arcs=clf, route_arcs=route,
                    route_gaps=gaps, chain_breaks=int(brk.sum()) - 1 if len(brk) else 0,
                    n_no_candidates=int(n - len(hmm_pts)), timings=tm,
                    _scoring=(gx, gy, nx, ny))

    def score(self, trip_dir, pred):
        """Server-side scoring after prediction. Returns aggregate metrics only."""
        ref_trip = load_trip(trip_dir, load_route=True)
        if ref_trip.route is None:
            return None
        gx, gy, nx, ny = pred["_scoring"]
        lab = align(gx, gy, ref_trip, nx, ny, self.cfg["labels"]["ambiguous_distance_m"])
        ref = lab["ref_arc"] if self.cfg["labels"]["method"] == "monotonic" else lab["ref_arc_nearest"]
        ev = ~lab["ambiguous"] & (pred["point_arcs"] >= 0)
        out = dict(n_evaluable=int(ev.sum()),
                   top1_hmm=float(np.mean(pred["point_arcs"][ev] == ref[ev])) if ev.any() else None,
                   top1_classifier_only=float(np.mean(pred["clf_point_arcs"][ev] == ref[ev])) if ev.any() else None)
        import pandas as pd
        has = pred["point_arcs"] >= 0
        ch = pd.DataFrame(dict(trip_id="x", point=np.flatnonzero(has), pred_arc=pred["point_arcs"][has]))
        out["route_continuity"] = float(route_continuity(ch, {"x": ref_trip.arcs.to_numpy()})["route_continuity"])
        out |= route_metrics(pred["route_arcs"], ref_trip.route)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def route_lines(route_arcs, arcs, nodes_lon, nodes_lat):
    """Route arcs -> list of polylines [[lon, lat], ...]; -1 entries split the route."""
    lines, cur, last_v = [], [], None
    for e in route_arcs:
        if e < 0:
            if len(cur) > 1:
                lines.append(cur)
            cur, last_v = [], None
            continue
        u, v = arcs[e]
        if last_v is None or last_v != u:
            if len(cur) > 1:
                lines.append(cur)
            cur = [[float(nodes_lon[u]), float(nodes_lat[u])]]
        cur.append([float(nodes_lon[v]), float(nodes_lat[v])])
        last_v = v
    if len(cur) > 1:
        lines.append(cur)
    return lines
