"""Metric definitions (plan v2 §5 / docs/evaluation_protocol.md).

* Candidate level  – (point, candidate) pairs: precision/recall/F1 of the positive class,
                     macro-F1 over the two classes {correct, incorrect}, log loss.
* Observation level – Top-1: the highest-scoring candidate equals the aligned reference arc.
                     Main denominator = evaluable points (not ambiguous, reference arc among candidates).
* Route level      – continuity: fraction of consecutive observations whose chosen arcs are the
                     same arc or share a node.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(
        cand_precision=precision_score(y, yhat, zero_division=0),
        cand_recall=recall_score(y, yhat, zero_division=0),
        cand_f1=f1_score(y, yhat, zero_division=0),
        cand_macro_f1=f1_score(y, yhat, average="macro", zero_division=0),
        cand_log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    )


def top1(df, score_col="score"):
    """df has trip_id, point, arc, label, ambiguous, score. Returns per-point table of chosen arcs."""
    i = df.groupby(["trip_id", "point"], sort=False)[score_col].idxmax()
    chosen = df.loc[i, ["trip_id", "point", "arc", "label", "ambiguous"]].rename(columns={"arc": "pred_arc"})
    chosen["has_ref"] = df.groupby(["trip_id", "point"], sort=False)["label"].max().to_numpy()
    return chosen.reset_index(drop=True)


def observation_metrics(chosen, n_points_total=None):
    ev = chosen[(~chosen.ambiguous) & (chosen.has_ref == 1)]
    na = chosen[~chosen.ambiguous]
    per_trip = ev.groupby("trip_id").label.mean()
    out = dict(
        top1_evaluable=float(ev.label.mean()),
        top1_non_ambiguous=float(na.label.mean()),
        top1_trip_mean=float(per_trip.mean()),
        top1_trip_min=float(per_trip.min()),
        n_evaluable=int(len(ev)),
    )
    if n_points_total is not None:
        out["frac_points_evaluable"] = len(ev) / n_points_total
    return out


def route_continuity(chosen, arcs_by_trip):
    """Fraction of consecutive chosen arcs that are identical or share a node (per trip, point-weighted)."""
    ok, total = 0, 0
    for tid, g in chosen.sort_values(["trip_id", "point"]).groupby("trip_id"):
        a = arcs_by_trip[tid]
        p = g.pred_arc.to_numpy()
        consecutive = np.diff(g.point.to_numpy()) == 1
        p0, p1 = p[:-1][consecutive], p[1:][consecutive]
        u0, v0, u1, v1 = a[p0, 0], a[p0, 1], a[p1, 0], a[p1, 1]
        conn = (p0 == p1) | (v0 == u1) | (u0 == u1) | (v0 == v1) | (u0 == v1)
        ok += conn.sum()
        total += len(conn)
    return dict(route_continuity=ok / max(total, 1))
