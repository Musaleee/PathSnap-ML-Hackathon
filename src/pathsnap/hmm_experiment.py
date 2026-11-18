"""Classifier-only vs classifier + HMM/Viterbi.

1. Out-of-fold probabilities on the 80 train trips (same GroupKFold as model selection) from the
   selected HistGB configuration; student-test probabilities from the final model (fit on all train).
2. HMM decoder parameters (beta, emission weight) chosen by grid search on the train OOF predictions.
3. One comparison on student-test: pointwise Top-1, route continuity, sequence-level route metrics.

Usage:  python -m pathsnap.hmm_experiment
Output: experiments/results/hmm_grid_train.csv, hmm_comparison.csv, hmm_per_trip_student_test.csv;
        artifacts/final/hmm_config.json; experiments/logs/<exp_id>.json
"""
import json
import time
from dataclasses import asdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import REPO_ROOT, list_trips, load_config, partition_dir
from .data_io import load_trip
from .evaluation import route_continuity
from .features import FEATURES
from .geo import project, utm_crs
from .hmm import Graph, HMMParams, expand_route, precompute, route_metrics, viterbi
from .models import get_model
from .train import load_split

RES = REPO_ROOT / "experiments" / "results"
FINAL = REPO_ROOT / "artifacts" / "final"


def best_params():
    p = REPO_ROOT / "experiments" / "optuna" / "best_params.json"
    return json.loads(p.read_text())["best_params"] if p.exists() else {}


def trip_geometry(trip):
    crs = utm_crs(trip.track.lon.to_numpy(), trip.track.lat.to_numpy())
    px, py = project(trip.track.lon.to_numpy(), trip.track.lat.to_numpy(), crs)
    nx, ny = project(trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy(), crs)
    a = trip.arcs.to_numpy()
    length = np.hypot(nx[a[:, 1]] - nx[a[:, 0]], ny[a[:, 1]] - ny[a[:, 0]])
    return px, py, Graph(a, length)


def prepare(df, cfg, partition, prm):
    """Per trip: candidate rows, geometry, HMM precomputation, reference route (offline scoring only)."""
    out = {}
    for tid in list_trips(cfg, partition):
        trip = load_trip(partition_dir(cfg, partition) / tid)
        px, py, g = trip_geometry(trip)
        d = df[df.trip_id == tid]
        t0 = time.time()
        pre = precompute(d.point.to_numpy(), d.arc.to_numpy(), d.frac_along.to_numpy(),
                         trip.track.t.to_numpy(), px, py, g, prm)
        out[tid] = dict(df=d, pre=pre, graph=g, arcs=trip.arcs.to_numpy(), route=trip.route,
                        t=trip.track.t.to_numpy(), px=px, py=py, pre_s=time.time() - t0)
    return out


def evaluate(trips, prm, use_hmm, with_routes=False):
    chosen_all, per_trip = [], []
    for tid, T in trips.items():
        d = T["df"]
        prob = d.score.to_numpy()
        if use_hmm:
            t0 = time.time()
            rows, brk = viterbi(T["pre"], prob, prm)
            dec_s = time.time() - t0
            n_brk = int(brk.sum()) - 1
        else:
            t0 = time.time()
            rows = d.reset_index(drop=True).groupby("point", sort=True).score.idxmax().to_numpy()
            dec_s, n_brk = time.time() - t0, 0
        c = d.iloc[rows]
        has_ref = d.groupby("point", sort=True).label.max().to_numpy()
        ch = pd.DataFrame(dict(trip_id=tid, point=c.point.to_numpy(), pred_arc=c.arc.to_numpy(),
                               label=c.label.to_numpy(), ambiguous=c.ambiguous.to_numpy(), has_ref=has_ref))
        chosen_all.append(ch)
        ev = ch[~ch.ambiguous & (ch.has_ref == 1)]
        r = dict(trip_id=tid, n_points=len(ch), n_evaluable=len(ev), top1=ev.label.mean(),
                 continuity=route_continuity(ch, {tid: T["arcs"]})["route_continuity"],
                 chain_breaks=n_brk, decode_s=dec_s)
        if with_routes:
            seq = ch.pred_arc.to_numpy().copy()
            if use_hmm:
                parts = np.split(seq, np.flatnonzero(brk)[1:])
                seq = np.concatenate([np.append(p, -1) for p in parts])[:-1]
            route, n_route_breaks = expand_route(seq, T["graph"], prm)
            r |= route_metrics(route, T["route"]) | dict(route_len=sum(a >= 0 for a in route),
                                                        route_gaps=n_route_breaks)
        per_trip.append(r)
    ch = pd.concat(chosen_all)
    ev = ch[~ch.ambiguous & (ch.has_ref == 1)]
    pt = pd.DataFrame(per_trip)
    agg = dict(top1=ev.label.mean(), continuity=route_continuity(ch, {k: v["arcs"] for k, v in trips.items()})["route_continuity"],
               chain_breaks=int(pt.chain_breaks.sum()), decode_s_per_trip=pt.decode_s.mean())
    if with_routes:
        for k in ["route_precision", "route_recall", "route_f1", "route_norm_edit"]:
            agg[k] = pt[k].mean()                                   # trip-weighted
        agg["route_gaps"] = int(pt.route_gaps.sum())
    return agg, pt


def main():
    cfg = load_config()
    seed = cfg["seed"]
    params = best_params()
    print("model params:", params or "untuned defaults", flush=True)
    train, test = load_split("train"), load_split("student_test")

    # 1. out-of-fold train probabilities + student-test probabilities
    trips = np.array(sorted(train.trip_id.unique()))
    train["score"] = np.nan
    for tr_t, va_t in [(trips[a], trips[b]) for a, b in GroupKFold(cfg["cv_folds"]).split(trips, groups=trips)]:
        tr = train[train.trip_id.isin(tr_t) & ~train.ambiguous]
        m, _, _ = get_model("hist_gb", seed, **params)
        m.fit(tr[FEATURES], tr.label)
        va = train.trip_id.isin(va_t)
        train.loc[va, "score"] = m.predict_proba(train.loc[va, FEATURES])[:, 1]
    final_path = FINAL / "hist_gb_tuned.joblib"
    if final_path.exists():
        final = joblib.load(final_path)
    else:
        final, _, _ = get_model("hist_gb", seed, **params)
        final.fit(train.loc[~train.ambiguous, FEATURES], train.loc[~train.ambiguous, "label"])
    test["score"] = final.predict_proba(test[FEATURES])[:, 1]

    prm0 = HMMParams()
    t0 = time.time()
    tr_trips = prepare(train, cfg, "train", prm0)
    st_trips = prepare(test, cfg, "student_test", prm0)
    print(f"precompute {time.time() - t0:.0f} s", flush=True)

    # 2. decoder grid on train OOF
    grid = []
    base, _ = evaluate(tr_trips, prm0, use_hmm=False)
    grid.append(dict(beta=None, emission_weight=None, decoder="classifier_only") | base)
    for beta in [2, 5, 10, 20, 50]:
        for w in [0.5, 1.0, 2.0]:
            prm = HMMParams(beta=beta, emission_weight=w)
            agg, _ = evaluate(tr_trips, prm, use_hmm=True)
            grid.append(dict(beta=beta, emission_weight=w, decoder="hmm") | agg)
            print(f"beta {beta} w {w}: top1 {agg['top1']:.4f} continuity {agg['continuity']:.4f}", flush=True)
    g = pd.DataFrame(grid)
    g.to_csv(RES / "hmm_grid_train.csv", index=False)
    h = g[g.decoder == "hmm"].sort_values(["top1", "continuity"], ascending=False).iloc[0]
    prm = HMMParams(beta=float(h.beta), emission_weight=float(h.emission_weight))
    print("selected", asdict(prm), flush=True)

    # 3. student-test comparison (one consultation)
    rows, per = [], []
    for name, use in [("classifier_only", False), ("classifier_hmm", True)]:
        agg, pt = evaluate(st_trips, prm, use_hmm=use, with_routes=True)
        rows.append(dict(decoder=name, partition="student_test") | agg)
        per.append(pt.assign(decoder=name))
        agg_tr, _ = evaluate(tr_trips, prm, use_hmm=use, with_routes=True)
        rows.append(dict(decoder=name, partition="train_oof") | agg_tr)
        print(name, rows[-2], flush=True)
    comp = pd.DataFrame(rows)
    comp.to_csv(RES / "hmm_comparison.csv", index=False)
    pd.concat(per).to_csv(RES / "hmm_per_trip_student_test.csv", index=False)
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "hmm_config.json").write_text(json.dumps(dict(hmm=asdict(prm), selected_on="train out-of-fold Top-1 (grid)",
                                                            grid=[dict(beta=b, w=w) for b in [2, 5, 10, 20, 50] for w in [0.5, 1, 2]]), indent=1))
    exp_id = f"{datetime.now():%Y%m%d-%H%M%S}-hmm"
    (REPO_ROOT / "experiments" / "logs" / f"{exp_id}.json").write_text(json.dumps(dict(
        exp_id=exp_id, model_params=params, hmm=asdict(prm), comparison=rows), indent=1, default=str))


if __name__ == "__main__":
    main()
