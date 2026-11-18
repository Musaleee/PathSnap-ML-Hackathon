"""Test-only evaluation of the already-trained models on a held-out folder of trips.

Nothing is fitted here: models are loaded from artifacts/candidates/*.joblib (trained on the 80 train
trips only). The .route files are used only to build reference labels for scoring, never as features.

Usage:  python -m pathsnap.evaluate_holdout <folder_with_trip_dirs> [out_dir]
Output: <out_dir>/ summary.csv, per_trip_<model>.csv, per_trip_all_models.csv, baselines.csv,
        data_checks.csv, labels_candidates.csv, predictions/<model>/<trip>.csv, run_info.json
"""
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import REPO_ROOT, load_config
from .data_io import check_invariants, load_trip
from .dataset import build_trip
from .evaluation import candidate_metrics, observation_metrics, route_continuity, top1
from .features import FEATURES
from .models import ALL_MODELS

MODELS_DIR = REPO_ROOT / "artifacts" / "candidates"


def score_table(d, arcs_by_trip, n_points_total):
    na = ~d.ambiguous
    m = candidate_metrics(d.label[na].to_numpy(), d.score[na].to_numpy())
    chosen = top1(d)
    m |= observation_metrics(chosen, n_points_total)
    m |= route_continuity(chosen, arcs_by_trip)
    return m, chosen


def main(folder, out_dir=None):
    folder = Path(folder)
    out = Path(out_dir) if out_dir else REPO_ROOT / "experiments" / "evaluator_test"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    trips = sorted(p.name for p in folder.iterdir() if p.is_dir())

    feats, stats, checks, arcs_by_trip = [], [], [], {}
    for tid in trips:
        t0 = time.time()
        trip = load_trip(folder / tid)
        issues = check_invariants(trip)
        checks.append(dict(trip_id=tid, n_points=len(trip.track), n_nodes=len(trip.nodes), n_arcs=len(trip.arcs),
                           route_len=len(trip.route), issues="; ".join(issues) if issues else "none"))
        f, _, s = build_trip(trip, cfg)
        s["seconds"] = round(time.time() - t0, 2)
        feats.append(f), stats.append(s)
        arcs_by_trip[tid] = trip.arcs.to_numpy()
        print(f"{tid}: {s['n_points']} pts, ambiguous {s['frac_ambiguous']:.3f}, "
              f"miss {s['miss_rate_evaluable']:.3f}, {s['seconds']} s", flush=True)
    df = pd.concat(feats, ignore_index=True)
    st = pd.DataFrame(stats)
    n_total = int(st.n_points.sum())
    pd.DataFrame(checks).to_csv(out / "data_checks.csv", index=False)
    st.to_csv(out / "labels_candidates.csv", index=False)
    base = df[["trip_id", "point", "arc", "label", "ambiguous"]].copy()

    # rule-based baselines (no learning)
    bl = []
    for name, score in [("baseline_nearest_road", -df["dist"].to_numpy()),
                        ("baseline_nearest_same_direction",
                         np.where(df["heading_cos"].fillna(-1).to_numpy() > 0, 0, -1000) - df["dist"].to_numpy())]:
        d = base.assign(score=score)
        chosen = top1(d)
        r = dict(model=name) | observation_metrics(chosen, n_total) | route_continuity(chosen, arcs_by_trip)
        bl.append(r)
    pd.DataFrame(bl).to_csv(out / "baselines.csv", index=False)

    rows, per_trip_all = [], []
    for name in ALL_MODELS:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"skip {name}: no saved model", flush=True)
            continue
        model = joblib.load(path)
        t0 = time.time()
        p = model.predict_proba(df[FEATURES])[:, 1]
        infer_s = time.time() - t0
        d = base.assign(score=p)
        m, chosen = score_table(d, arcs_by_trip, n_total)
        m.update(model=name, inference_s=infer_s, inference_s_per_trip=infer_s / len(trips))
        rows.append(m)

        pt = []
        for tid, g in chosen.groupby("trip_id"):
            ev = g[(~g.ambiguous) & (g.has_ref == 1)]
            pt.append(dict(model=name, trip_id=tid, n_points=len(g), n_evaluable=len(ev),
                           top1_evaluable=ev.label.mean() if len(ev) else np.nan,
                           route_continuity=route_continuity(g, arcs_by_trip)["route_continuity"]))
        pt = pd.DataFrame(pt)
        pt.to_csv(out / f"per_trip_{name}.csv", index=False)
        per_trip_all.append(pt)

        pdir = out / "predictions" / name
        pdir.mkdir(parents=True, exist_ok=True)
        for tid, g in d.assign(rank=d.groupby(["trip_id", "point"]).score.rank(ascending=False, method="first")) \
                       .query("rank == 1").groupby("trip_id"):
            g.sort_values("point")[["point", "arc", "score", "label", "ambiguous"]] \
             .rename(columns={"arc": "pred_arc", "label": "correct"}).to_csv(pdir / f"{tid}.csv", index=False)
        print(f"{name:14s} top1 {m['top1_evaluable']:.4f}  macroF1 {m['cand_macro_f1']:.4f}  "
              f"logloss {m['cand_log_loss']:.4f}  continuity {m['route_continuity']:.4f}  "
              f"worst trip {m['top1_trip_min']:.4f}", flush=True)

    cols = ["model", "top1_evaluable", "top1_non_ambiguous", "top1_trip_mean", "top1_trip_min", "cand_macro_f1",
            "cand_f1", "cand_precision", "cand_recall", "cand_log_loss", "route_continuity", "n_evaluable",
            "frac_points_evaluable", "inference_s_per_trip"]
    summary = pd.DataFrame(rows)[cols].sort_values("top1_evaluable", ascending=False)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(per_trip_all).to_csv(out / "per_trip_all_models.csv", index=False)
    w = st.n_points / st.n_points.sum()
    info = dict(
        purpose="test-only evaluation of pre-trained models; no fitting, no tuning on this data",
        data_folder_name=folder.name, trips=trips, n_points=n_total, n_candidate_rows=len(df),
        ambiguous_frac=float(np.sum(w * st.frac_ambiguous)),
        miss_rate_evaluable=float(np.nansum(w * st.miss_rate_evaluable)),
        models_dir=str(MODELS_DIR.relative_to(REPO_ROOT)), config=dict(labels=cfg["labels"], candidates=cfg["candidates"]),
        environment=dict(python=platform.python_version(), sklearn=sklearn.__version__, machine=platform.machine()),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
