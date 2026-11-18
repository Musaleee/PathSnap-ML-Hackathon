"""Report figures generated from the data and from logged results only.

Usage:  python -m pathsnap.report_figures [eda|models|tuning|hmm|all]
Output: reports/figures/*.png
"""
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import REPO_ROOT, list_trips, load_config, partition_dir
from .data_io import load_trip
from .features import FEATURES
from .models import ALL_MODELS

FIG = REPO_ROOT / "reports" / "figures"
RES = REPO_ROOT / "experiments" / "results"


def eda(cfg):
    rows = []
    for part in ("train", "student_test"):
        for tid in list_trips(cfg, part):
            t = load_trip(partition_dir(cfg, part) / tid)
            tr = t.track
            dt = np.diff(tr.t.to_numpy())
            rows.append(dict(partition=part, trip=tid, n=len(tr),
                             nan_lon=int(tr.lon.isna().sum()), nan_lat=int(tr.lat.isna().sum()), nan_t=int(tr.t.isna().sum()),
                             nonfinite=int((~np.isfinite(tr.to_numpy())).sum()), dup_t=int((dt == 0).sum()),
                             back_t=int((dt < 0).sum()), gaps_10s=int((dt > 10).sum()), max_gap=float(dt.max()),
                             frac_1s=float(np.mean(dt == 1))))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "missing_value_analysis.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cols = ["nan_lon", "nan_lat", "nan_t", "nonfinite", "dup_t", "back_t"]
    s = d.groupby("partition")[cols].sum().T
    s.plot.bar(ax=ax[0], color=["#4C78A8", "#F58518"])
    ax[0].set_title("Missing / invalid values per column (all trips)"); ax[0].set_ylabel("count")
    ax[0].set_ylim(0, 1); ax[0].text(0.5, 0.5, "all counts are 0", transform=ax[0].transAxes, ha="center")
    ax[1].hist(d[d.partition == "train"].frac_1s, bins=20, color="#4C78A8")
    ax[1].set_title("Share of 1-second steps per trip (train)"); ax[1].set_xlabel("fraction of steps with dt = 1 s")
    ax[2].hist(np.log10(d[d.partition == "train"].max_gap), bins=20, color="#E45756")
    ax[2].set_title("Largest time gap per trip (train)"); ax[2].set_xlabel("log10(seconds)")
    fig.tight_layout(); fig.savefig(FIG / "eda_missing_values_and_gaps.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, tid in zip(axs.ravel(), list_trips(cfg, "train")[::10]):
        t = load_trip(partition_dir(cfg, "train") / tid)
        sc = ax.scatter(t.track.lon, t.track.lat, c=t.track.t - t.track.t.min(), s=1, cmap="viridis")
        ax.set_title(f"train {tid} ({len(t.track)} pts)", fontsize=10); ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    fig.suptitle("Raw GPS trajectories (colour = time since start)"); fig.tight_layout()
    fig.savefig(FIG / "eda_gps_trajectories.png", dpi=150); plt.close(fig)

    t = load_trip(partition_dir(cfg, "train") / "00000001")
    a = t.arcs.to_numpy(); lon, lat = t.nodes.lon.to_numpy(), t.nodes.lat.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.stack([np.c_[lon[a[:, 0]], lat[a[:, 0]]], np.c_[lon[a[:, 1]], lat[a[:, 1]]]], 1),
                                     colors="#B0B0B0", linewidths=0.4, label="road network"))
    r = t.route
    ax.add_collection(LineCollection(np.stack([np.c_[lon[a[r, 0]], lat[a[r, 0]]], np.c_[lon[a[r, 1]], lat[a[r, 1]]]], 1),
                                     colors="#2CA02C", linewidths=3, label="reference route"))
    ax.scatter(t.track.lon, t.track.lat, s=1, c="#1F77B4", label="raw GPS")
    pad = 0.003
    ax.set_xlim(t.track.lon.min() - pad, t.track.lon.max() + pad); ax.set_ylim(t.track.lat.min() - pad, t.track.lat.max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(t.track.lat.mean())))
    ax.legend(loc="upper right"); ax.set_title(f"train 00000001: road network ({len(a)} arcs), GPS and reference route")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.tight_layout(); fig.savefig(FIG / "eda_road_network.png", dpi=150); plt.close(fig)

    deg = []
    for part in ("train",):
        for tid in list_trips(cfg, part):
            t = load_trip(partition_dir(cfg, part) / tid, load_route=False)
            deg.append(dict(trip=tid, nodes=len(t.nodes), arcs=len(t.arcs), points=len(t.track)))
    g = pd.DataFrame(deg)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(g.nodes, g.arcs, s=12); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs"); ax[0].set_title("Road-network size per trip (train)")
    ax[1].hist(g.points, bins=25, color="#4C78A8"); ax[1].set_xlabel("GPS points per trip"); ax[1].set_title("Trip length (train)")
    fig.tight_layout(); fig.savefig(FIG / "eda_network_and_trip_sizes.png", dpi=150); plt.close(fig)


def models():
    test = pd.read_pickle(REPO_ROOT / "experiments" / "cache" / "student_test.pkl")
    test = test[~test.ambiguous]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, name in zip(axs.ravel(), ALL_MODELS):
        m = joblib.load(REPO_ROOT / "artifacts" / "candidates" / f"{name}.joblib")
        yhat = (m.predict_proba(test[FEATURES])[:, 1] >= 0.5).astype(int)
        cm = confusion_matrix(test.label, yhat, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
        ax.set_xticks([0, 1], ["pred wrong", "pred correct"]); ax.set_yticks([0, 1], ["wrong arc", "correct arc"])
        ax.set_title(name)
    axs.ravel()[-1].axis("off")
    fig.suptitle("Candidate-level confusion matrices, student-test, threshold 0.5 (row-normalised)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices_student_test.png", dpi=150); plt.close(fig)

    d = pd.read_csv(RES / "model_comparison.csv").drop_duplicates("model", keep="last").sort_values("st_log_loss")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].barh(d.model, d.cv_log_loss, color="#E45756"); ax[0].set_title("CV log loss (lower is better)"); ax[0].set_xscale("log")
    ax[1].barh(d.model, d.infer_s_per_trip, color="#72B7B2"); ax[1].set_title("Inference time per trip (s, student-test)"); ax[1].set_xscale("log")
    fig.tight_layout(); fig.savefig(FIG / "logloss_and_inference_time.png", dpi=150); plt.close(fig)


def tuning():
    tr = pd.read_csv(REPO_ROOT / "experiments" / "optuna" / "trials.csv")
    done = tr[tr.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(tr.number, tr.value, c=np.where(tr.state == "COMPLETE", "#4C78A8", "#BBBBBB"), s=20, label="trial (grey = pruned)")
    ax.plot(done.number, done.value.cummax(), color="#E45756", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1 (5 folds, train)"); ax.set_title("Optuna TPE search, HistGB"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "tuning_history.png", dpi=150); plt.close(fig)
    c = pd.read_csv(RES / "tuning_comparison.csv")
    met = ["cv_top1", "cv_macro_f1", "st_top1", "st_macro_f1"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(met)); w = .38
    for i, r in c.iterrows():
        ax[0].bar(x + (i - .5) * w, r[met], w, label=r.model)
    ax[0].set_xticks(x, met); ax[0].set_ylim(.9, .98); ax[0].legend(); ax[0].set_title("Untuned vs tuned")
    ax[1].bar(c.model, c.st_log_loss, color=["#4C78A8", "#F58518"]); ax[1].set_title("Student-test log loss")
    fig.tight_layout(); fig.savefig(FIG / "tuning_untuned_vs_tuned.png", dpi=150); plt.close(fig)


def hmm():
    p = pd.read_csv(RES / "hmm_per_trip_student_test.csv", dtype={"trip_id": str})
    w = p.pivot(index="trip_id", columns="decoder", values=["top1", "continuity", "route_f1"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, m in zip(axs, ["top1", "continuity", "route_f1"]):
        x = np.arange(len(w)); bw = .4
        ax.bar(x - bw / 2, w[(m, "classifier_only")], bw, label="classifier only", color="#FF7F0E")
        ax.bar(x + bw / 2, w[(m, "classifier_hmm")], bw, label="classifier + HMM", color="#D62728")
        ax.set_xticks(x, [t[-2:] for t in w.index]); ax.set_xlabel("student-test trip (last 2 digits)")
        ax.set_title(m); ax.set_ylim(max(0, w[m].min().min() - .1), 1.01); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "hmm_vs_classifier_per_trip.png", dpi=150); plt.close(fig)
    g = pd.read_csv(RES / "hmm_grid_train.csv").dropna(subset=["beta"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for wgt, s in g.groupby("emission_weight"):
        ax.plot(s.beta, s.top1, marker="o", label=f"emission weight {wgt}")
    base = pd.read_csv(RES / "hmm_grid_train.csv").query("decoder == 'classifier_only'").top1.iloc[0]
    ax.axhline(base, color="grey", ls="--", label="classifier only")
    ax.set_xscale("log"); ax.set_xlabel("beta (m)"); ax.set_ylabel("Top-1, train out-of-fold"); ax.legend()
    ax.set_title("HMM decoder grid (train OOF)")
    fig.tight_layout(); fig.savefig(FIG / "hmm_grid_train.png", dpi=150); plt.close(fig)


def evaluator():
    """Evaluator-test (one test-only run): untuned vs tuned, classifier only vs + HMM, per trip."""
    d = REPO_ROOT / "experiments" / "evaluator_test_final"
    t = pd.read_csv(d / "per_trip.csv", dtype={"trip_id": str})
    u = pd.read_csv(d / "per_trip_untuned_reference.csv", dtype={"trip_id": str})
    x = np.arange(len(t)); bw = .2
    fig, ax = plt.subplots(1, 2, figsize=(17, 4.8))
    ax[0].bar(x - 1.5 * bw, u.top1_classifier_only, bw, label="untuned, classifier only", color="#9ECAE9")
    ax[0].bar(x - .5 * bw, t.top1_classifier_only, bw, label="tuned, classifier only", color="#4C78A8")
    ax[0].bar(x + .5 * bw, u.top1_hmm, bw, label="untuned + HMM", color="#FF9DA6")
    ax[0].bar(x + 1.5 * bw, t.top1_hmm, bw, label="tuned + HMM (final)", color="#D62728")
    ax[0].set_ylim(.8, 1.06); ax[0].set_title("Evaluator-test Top-1 per trip"); ax[0].legend(fontsize=8, loc="upper left", ncol=4)
    ax[1].bar(x - bw / 2, u.route_f1, bw, label="untuned + HMM", color="#FF9DA6")
    ax[1].bar(x + bw / 2, t.route_f1, bw, label="tuned + HMM (final)", color="#D62728")
    ax[1].set_ylim(.7, 1.06); ax[1].set_title("Evaluator-test route F1 per trip"); ax[1].legend(fontsize=8, loc="upper left", ncol=2)
    for a in ax:
        a.set_xticks(x, [s[-2:] for s in t.trip_id]); a.set_xlabel("evaluator trip (last 2 digits)")
    fig.tight_layout(); fig.savefig(FIG / "evaluator_test_final.png", dpi=150); plt.close(fig)


def main(what="all"):
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    if what in ("evaluator", "all"):
        evaluator()
    if what in ("eda", "all"):
        eda(cfg)
    if what in ("models", "all"):
        models()
    if what in ("tuning", "all"):
        tuning()
    if what in ("hmm", "all"):
        hmm()
    print("figures in", FIG)


if __name__ == "__main__":
    main(*sys.argv[1:])
