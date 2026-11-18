"""Comparison figures generated only from experiments/results/model_comparison.csv and logs.

Usage:  python -m pathsnap.plots
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import REPO_ROOT

FIG = REPO_ROOT / "reports" / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(REPO_ROOT / "experiments" / "results" / "model_comparison.csv")
    d = d.drop_duplicates("model", keep="last").sort_values("cv_top1")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].barh(d.model, d.cv_top1, xerr=d.cv_top1_std, color="#4C78A8")
    ax[0].set_title("CV Top-1 accuracy (evaluable points)"); ax[0].set_xlim(0.8, 1)
    ax[1].barh(d.model, d.cv_macro_f1, xerr=d.cv_macro_f1_std, color="#4C78A8")
    ax[1].set_title("CV candidate macro-F1"); ax[1].set_xlim(0.8, 1)
    ax[2].barh(d.model, d.cv_log_loss, color="#E45756")
    ax[2].set_title("CV candidate log loss (lower is better)")
    fig.tight_layout(); fig.savefig(FIG / "cv_comparison.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.4
    y = range(len(d))
    ax.barh([i - w / 2 for i in y], d.st_top1, w, label="Top-1", color="#4C78A8")
    ax.barh([i + w / 2 for i in y], d.st_macro_f1, w, label="Macro-F1", color="#F58518")
    ax.set_yticks(list(y)); ax.set_yticklabels(d.model); ax.set_xlim(0.8, 1); ax.legend()
    ax.set_title("Student-test (10 trips)")
    fig.tight_layout(); fig.savefig(FIG / "student_test_comparison.png", dpi=150)

    # real training histories only (models that produce them)
    for log in sorted((REPO_ROOT / "experiments" / "logs").glob("*.json")):
        L = json.loads(log.read_text())
        h = L.get("history")
        if not h:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["train_loss"], label="training loss")
        if h.get("val_loss"):
            ax.plot(h["val_loss"], label="validation loss")
        if h.get("val_score"):
            ax2 = ax.twinx(); ax2.plot(h["val_score"], color="C2", label="validation accuracy"); ax2.legend(loc="center right")
        ax.set_xlabel("iteration / epoch"); ax.set_title(f"{L['model']} training history (final fit)"); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / f"history_{L['model']}.png", dpi=150)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
