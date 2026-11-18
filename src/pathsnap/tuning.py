"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (HistGradientBoosting).

Objective: mean group-aware 5-fold CV Top-1 accuracy on evaluable points of the 80 training trips
(the primary metric declared in docs/evaluation_protocol.md). Student-test is used once, after the
search, to compare untuned vs tuned. The evaluator partition is never used.

Usage:  python -m pathsnap.tuning [n_trials]
Output: experiments/optuna/study.db, trials.csv, best_params.json; experiments/logs/<exp_id>.json;
        experiments/results/tuning_comparison.csv; artifacts/final/hist_gb_tuned.joblib
"""
import json
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import REPO_ROOT, load_config
from .features import FEATURES
from .models import get_model
from .train import arcs_for, evaluate, load_split

OPT = REPO_ROOT / "experiments" / "optuna"
FINAL = REPO_ROOT / "artifacts" / "final"


def search_space(trial):
    # Ranges chosen around the untuned defaults (lr 0.1, 31 leaves, min leaf 20, no L2)
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        max_depth=trial.suggest_categorical("max_depth", [None, 6, 10, 16]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 1000, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        max_features=trial.suggest_float("max_features", 0.5, 1.0),
        max_iter=trial.suggest_int("max_iter", 150, 800, step=50),
    )


def main(n_trials=40):
    cfg = load_config()
    seed = cfg["seed"]
    train, test = load_split("train"), load_split("student_test")
    trips = np.array(sorted(train.trip_id.unique()))
    folds = [(trips[a], trips[b]) for a, b in GroupKFold(cfg["cv_folds"]).split(trips, groups=trips)]
    arcs_train, arcs_test = arcs_for(cfg, "train"), arcs_for(cfg, "student_test")

    def cv(params, trial=None):
        rows = []
        for k, (tr_t, va_t) in enumerate(folds):
            tr = train[train.trip_id.isin(tr_t) & ~train.ambiguous]
            va = train[train.trip_id.isin(va_t)]
            model, _, _ = get_model("hist_gb", seed, **params)
            t0 = time.time()
            model.fit(tr[FEATURES], tr.label)
            m = evaluate(model, va, arcs_train)
            m["fit_s"] = time.time() - t0
            rows.append(m)
            if trial is not None:
                trial.report(float(np.mean([r["top1_evaluable"] for r in rows])), k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return pd.DataFrame(rows)

    def objective(trial):
        params = search_space(trial)
        r = cv(params, trial)
        trial.set_user_attr("cv_macro_f1", float(r.cand_macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(r.cand_log_loss.mean()))
        trial.set_user_attr("cv_top1_std", float(r.top1_evaluable.std()))
        trial.set_user_attr("cv_continuity", float(r.route_continuity.mean()))
        trial.set_user_attr("fit_s_mean", float(r.fit_s.mean()))
        return float(r.top1_evaluable.mean())

    OPT.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="hist_gb_top1", direction="maximize", storage=f"sqlite:///{OPT / 'study.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    # trial 0 = the untuned configuration, so the search starts from the baseline
    if len(study.trials) == 0:
        study.enqueue_trial(dict(learning_rate=0.1, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,
                                 l2_regularization=1e-4, max_features=1.0, max_iter=300))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.state.name} value={t.value} "
                                                 f"best={s.best_value:.4f}", flush=True)])
    # search time from the stored trials, so a resumed / comparison-only run reports the real cost
    fin = [t for t in study.trials if t.datetime_complete is not None]
    search_s = sum((t.datetime_complete - t.datetime_start).total_seconds() for t in fin) or (time.time() - t0)
    study.trials_dataframe().to_csv(OPT / "trials.csv", index=False)
    best = study.best_params
    (OPT / "best_params.json").write_text(json.dumps(dict(best_params=best, best_cv_top1=study.best_value,
                                                         n_trials=len(study.trials), search_seconds=search_s), indent=1))

    # untuned vs tuned: same folds, same metrics, same partitions
    out = []
    for tag, params in [("untuned", {}), ("tuned", best)]:
        r = cv(params)
        tr = train[~train.ambiguous]
        model, _, _ = get_model("hist_gb", seed, **params)
        t1 = time.time()
        model.fit(tr[FEATURES], tr.label)
        fit_s = time.time() - t1
        st = evaluate(model, test, arcs_test)
        out.append(dict(model=f"hist_gb_{tag}", cv_top1=r.top1_evaluable.mean(), cv_top1_std=r.top1_evaluable.std(),
                        cv_macro_f1=r.cand_macro_f1.mean(), cv_log_loss=r.cand_log_loss.mean(),
                        st_top1=st["top1_evaluable"], st_macro_f1=st["cand_macro_f1"], st_log_loss=st["cand_log_loss"],
                        st_continuity=st["route_continuity"], train_s=fit_s,
                        infer_s_per_trip=st["inference_s_per_trip"], params=json.dumps(params)))
        print(out[-1], flush=True)
        if tag == "tuned":
            FINAL.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, FINAL / "hist_gb_tuned.joblib")
            (FINAL / "model_config.json").write_text(json.dumps(dict(
                model="HistGradientBoostingClassifier", params=model.get_params(), features=FEATURES,
                candidates=cfg["candidates"], labels=cfg["labels"], seed=seed,
                trained_on="80 train trips, non-ambiguous observations"), indent=1, default=str))
    pd.DataFrame(out).to_csv(REPO_ROOT / "experiments" / "results" / "tuning_comparison.csv", index=False)
    exp_id = f"{datetime.now():%Y%m%d-%H%M%S}-tuning-hist_gb"
    (REPO_ROOT / "experiments" / "logs" / f"{exp_id}.json").write_text(json.dumps(dict(
        exp_id=exp_id, sampler="TPE", pruner="Median", seed=seed, n_trials=len(study.trials),
        objective="mean GroupKFold(5) Top-1 on evaluable points, 80 train trips",
        best_params=best, best_value=study.best_value, search_seconds=search_s, comparison=out), indent=1, default=str))
    print(f"done in {search_s:.0f} s; best {study.best_value:.4f} {best}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
