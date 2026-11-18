"""Builds reports/report_tables.md from the logged results, so every number in the report is traceable.

Usage:  python -m pathsnap.report_tables
"""
import json

import pandas as pd

from .config import REPO_ROOT

RES = REPO_ROOT / "experiments" / "results"
OUT = REPO_ROOT / "reports" / "report_tables.md"

CONFIG = {
    "logreg": "C=1.0, L2, lbfgs, max_iter=500; median impute + scaling",
    "gaussian_nb": "default var_smoothing=1e-9; median impute + scaling",
    "knn": "k=25, distance-weighted, Euclidean on scaled features; 150k-row subsample",
    "random_forest": "150 trees, max_depth=18, min_samples_leaf=5, max_samples=0.3",
    "extra_trees": "150 trees, max_depth=18, min_samples_leaf=5, bootstrap, max_samples=0.3",
    "hist_gb": "max_iter=300, lr=0.1, 31 leaves, early stopping",
    "mlp": "64-32 ReLU, Adam, alpha=1e-4, batch 1024, early stopping; 400k-row subsample",
}


def f(x, n=4):
    return "N/A" if pd.isna(x) else f"{x:.{n}f}"


def main():
    L = []
    d = pd.read_csv(RES / "model_comparison.csv").drop_duplicates("model", keep="last").sort_values("cv_top1", ascending=False)
    L += ["## 3.1 Overall model comparison", "",
          "CV = GroupKFold(5) over the 80 train trips (mean ± std). Top-1 = observation-level, evaluable points. "
          "Macro-F1 = candidate-pair level over the two classes. Inference time = seconds per student-test trip (model only).", "",
          "| Model | Family | Key configuration | CV macro-F1 | CV Top-1 | CV route continuity | Student-test Top-1 | Student-test macro-F1 | Student-test log loss | Train time (s) | Inference (s/trip) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        L.append(f"| {r.model} | {r.family} | {CONFIG.get(r.model, '')} | {f(r.cv_macro_f1)} ± {f(r.cv_macro_f1_std, 3)} | "
                 f"{f(r.cv_top1)} ± {f(r.cv_top1_std, 3)} | {f(r.cv_continuity)} | {f(r.st_top1)} | {f(r.st_macro_f1)} | "
                 f"{f(r.st_log_loss)} | {f(r.train_s, 1)} | {f(r.infer_s_per_trip, 3)} |")
    L += ["", "### Student-test precision / recall / F1 (positive class, threshold 0.5)", "",
          "| Model | Precision | Recall | F1 | Route continuity |", "|---|---|---|---|---|"]
    for _, r in d.iterrows():
        L.append(f"| {r.model} | {f(r.st_precision)} | {f(r.st_recall)} | {f(r.st_f1_pos)} | {f(r.st_continuity)} |")

    L += ["", "### Overfitting check (final fit, CV vs student-test Top-1)", "", "| Model | CV Top-1 | Student-test Top-1 | Difference |", "|---|---|---|---|"]
    for _, r in d.iterrows():
        L.append(f"| {r.model} | {f(r.cv_top1)} | {f(r.st_top1)} | {r.st_top1 - r.cv_top1:+.4f} |")

    p = RES / "tuning_comparison.csv"
    if p.exists():
        t = pd.read_csv(p)
        bp = json.loads((REPO_ROOT / "experiments" / "optuna" / "best_params.json").read_text())
        tr = pd.read_csv(REPO_ROOT / "experiments" / "optuna" / "trials.csv")
        L += ["", "## 4. Bayesian optimisation (HistGB)", "",
              f"Trials: {len(tr)} ({(tr.state == 'COMPLETE').sum()} complete, {(tr.state == 'PRUNED').sum()} pruned); "
              f"search time {bp['search_seconds'] / 60:.1f} min; best CV Top-1 {bp['best_cv_top1']:.4f}.", "",
              "Best parameters: `" + json.dumps(bp["best_params"]) + "`", "",
              "| Model | CV Top-1 | CV macro-F1 | CV log loss | Student-test Top-1 | Student-test macro-F1 | Student-test log loss | Train (s) | Inference (s/trip) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r.model} | {f(r.cv_top1)} ± {f(r.cv_top1_std, 3)} | {f(r.cv_macro_f1)} | {f(r.cv_log_loss)} | {f(r.st_top1)} | "
                     f"{f(r.st_macro_f1)} | {f(r.st_log_loss)} | {f(r.train_s, 1)} | {f(r.infer_s_per_trip, 3)} |")

    p = RES / "hmm_comparison.csv"
    if p.exists():
        h = pd.read_csv(p)
        L += ["", "## 5. Classifier-only vs classifier + HMM", "",
              "Route metrics compare the reconstructed route with the reference route (collapsed arc sequences; trip-weighted mean). "
              "Lower normalised edit distance is better.", "",
              "| Partition | Decoder | Top-1 | Route continuity | Route precision | Route recall | Route F1 | Norm. edit distance | Chain breaks | Decode (s/trip) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in h.sort_values(["partition", "decoder"]).iterrows():
            L.append(f"| {r.partition} | {r.decoder} | {f(r.top1)} | {f(r.continuity)} | {f(r.route_precision)} | {f(r.route_recall)} | "
                     f"{f(r.route_f1)} | {f(r.route_norm_edit)} | {int(r.chain_breaks)} | {f(r.decode_s_per_trip, 3)} |")
        pt = pd.read_csv(RES / "hmm_per_trip_student_test.csv", dtype={"trip_id": str})
        w = pt.pivot(index="trip_id", columns="decoder", values=["top1", "continuity", "route_f1"])
        L += ["", "Per student-test trip:", "", "| Trip | Top-1 clf | Top-1 HMM | Continuity clf | Continuity HMM | Route F1 clf | Route F1 HMM |", "|---|---|---|---|---|---|---|"]
        for tid, r in w.iterrows():
            L.append(f"| {tid} | {f(r[('top1', 'classifier_only')])} | {f(r[('top1', 'classifier_hmm')])} | "
                     f"{f(r[('continuity', 'classifier_only')])} | {f(r[('continuity', 'classifier_hmm')])} | "
                     f"{f(r[('route_f1', 'classifier_only')])} | {f(r[('route_f1', 'classifier_hmm')])} |")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("# Report tables (generated by `python -m pathsnap.report_tables`; do not edit by hand)\n\n" + "\n".join(L) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
