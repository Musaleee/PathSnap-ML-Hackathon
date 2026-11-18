"""Experiment runner: group-aware CV on the 80 training trips + evaluation on the 10 student-test trips.

Usage:  python -m pathsnap.train logreg hist_gb ...      (default: all models)

Training rows: non-ambiguous observations only (see docs/decisions.md D-05).
Evaluation: metrics from pathsnap.evaluation. Each run writes experiments/logs/<exp_id>.json
and appends one row to experiments/results/model_comparison.csv.
"""
import json
import platform
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from .config import REPO_ROOT, list_trips, load_config, partition_dir
from .data_io import load_trip
from .dataset import CACHE
from .evaluation import candidate_metrics, observation_metrics, route_continuity, top1
from .features import FEATURES
from .models import ALL_MODELS, get_model

LOGS = REPO_ROOT / "experiments" / "logs"
RESULTS = REPO_ROOT / "experiments" / "results"


def load_split(name):
    return pd.read_pickle(CACHE / f"{name}.pkl")


def arcs_for(cfg, partition):
    return {tid: load_trip(partition_dir(cfg, partition) / tid, load_route=False).arcs.to_numpy()
            for tid in list_trips(cfg, partition)}


def subsample(df, max_rows, seed):
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def evaluate(model, df, arcs_by_trip):
    t0 = time.time()
    p = model.predict_proba(df[FEATURES])[:, 1]
    infer_s = time.time() - t0
    d = df[["trip_id", "point", "arc", "label", "ambiguous"]].copy()
    d["score"] = p
    na = ~d.ambiguous
    m = candidate_metrics(d.label[na].to_numpy(), p[na.to_numpy()])
    chosen = top1(d)
    m |= observation_metrics(chosen)
    m |= route_continuity(chosen, arcs_by_trip)
    m["inference_s"] = infer_s
    m["inference_s_per_trip"] = infer_s / d.trip_id.nunique()
    return m


def run(name, train, test, cfg, arcs_train, arcs_test, folds):
    seed = cfg["seed"]
    model, family, max_rows = get_model(name, seed)
    exp_id = f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    print(f"\n=== {exp_id} ({family}) ===", flush=True)
    fold_rows = []
    for k, (tr_trips, va_trips) in enumerate(folds):
        tr = train[train.trip_id.isin(tr_trips) & ~train.ambiguous]
        va = train[train.trip_id.isin(va_trips)]
        tr = subsample(tr, max_rows, seed)
        model, _, _ = get_model(name, seed)
        t0 = time.time()
        model.fit(tr[FEATURES], tr.label)
        fit_s = time.time() - t0
        m = evaluate(model, va, arcs_train)
        m.update(fold=k, fit_s=fit_s, n_train_rows=len(tr))
        fold_rows.append(m)
        print(f"  fold {k}: top1 {m['top1_evaluable']:.4f} macroF1 {m['cand_macro_f1']:.4f} "
              f"logloss {m['cand_log_loss']:.4f} fit {fit_s:.1f}s", flush=True)
    cv = pd.DataFrame(fold_rows)

    # final fit on all 80 training trips, evaluated once on student-test
    tr = subsample(train[~train.ambiguous], max_rows, seed)
    model, _, _ = get_model(name, seed)
    t0 = time.time()
    model.fit(tr[FEATURES], tr.label)
    fit_s = time.time() - t0
    st = evaluate(model, test, arcs_test)
    print(f"  student-test: top1 {st['top1_evaluable']:.4f} macroF1 {st['cand_macro_f1']:.4f} "
          f"logloss {st['cand_log_loss']:.4f} continuity {st['route_continuity']:.4f}", flush=True)

    history = None
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):
        history = dict(train_loss=list(map(float, est.loss_curve_)),
                       val_score=list(map(float, getattr(est, "validation_scores_", []) or [])))
    elif hasattr(est, "train_score_") and getattr(est, "validation_score_", None) is not None:
        history = dict(train_loss=list(map(float, -est.train_score_)), val_loss=list(map(float, -est.validation_score_)))

    log = dict(
        exp_id=exp_id, model=name, family=family, params=est.get_params(), features=FEATURES,
        label_method=cfg["labels"]["method"], candidates=cfg["candidates"], seed=seed,
        max_train_rows=max_rows, folds=[dict(train=list(a), val=list(b)) for a, b in folds],
        cv_folds=fold_rows, cv_mean=cv.mean(numeric_only=True).to_dict(), cv_std=cv.std(numeric_only=True).to_dict(),
        student_test=st, final_fit_s=fit_s, history=history,
        environment=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                         machine=platform.machine(), system=platform.system()),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{exp_id}.json").write_text(json.dumps(log, indent=1, default=str))
    (REPO_ROOT / "artifacts" / "candidates").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, REPO_ROOT / "artifacts" / "candidates" / f"{name}.joblib")

    row = dict(exp_id=exp_id, model=name, family=family, max_train_rows=max_rows,
               cv_top1=cv.top1_evaluable.mean(), cv_top1_std=cv.top1_evaluable.std(),
               cv_macro_f1=cv.cand_macro_f1.mean(), cv_macro_f1_std=cv.cand_macro_f1.std(),
               cv_log_loss=cv.cand_log_loss.mean(), cv_continuity=cv.route_continuity.mean(),
               st_top1=st["top1_evaluable"], st_macro_f1=st["cand_macro_f1"], st_f1_pos=st["cand_f1"],
               st_precision=st["cand_precision"], st_recall=st["cand_recall"], st_log_loss=st["cand_log_loss"],
               st_continuity=st["route_continuity"], train_s=fit_s, infer_s_per_trip=st["inference_s_per_trip"])
    out = RESULTS / "model_comparison.csv"
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
    return row


def main(names):
    cfg = load_config()
    train, test = load_split("train"), load_split("student_test")
    print(f"train rows {len(train):,}  student-test rows {len(test):,}", flush=True)
    trips = np.array(sorted(train.trip_id.unique()))
    gkf = GroupKFold(n_splits=cfg["cv_folds"])
    folds = [(trips[a], trips[b]) for a, b in gkf.split(trips, groups=trips)]
    arcs_train, arcs_test = arcs_for(cfg, "train"), arcs_for(cfg, "student_test")
    for name in names:
        run(name, train, test, cfg, arcs_train, arcs_test, folds)


if __name__ == "__main__":
    main(sys.argv[1:] or ALL_MODELS)
