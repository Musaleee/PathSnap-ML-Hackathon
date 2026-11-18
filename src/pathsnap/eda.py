"""Dataset inventory and data-quality checks for all trips (Gate A).

Usage:  python -m pathsnap.eda
Writes experiments/results/inventory.csv and data_quality_summary.txt
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import REPO_ROOT, list_trips, load_config, partition_dir
from .data_io import load_trip


def trip_stats(trip, partition, large_gap_s):
    tr, a = trip.track, trip.arcs.to_numpy()
    n = len(trip.nodes)
    dt = np.diff(tr.t.to_numpy())
    g = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n, n))
    n_comp_weak, _ = connected_components(g, directed=True, connection="weak")
    n_comp_strong, _ = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(a.ravel(), minlength=n)
    pairs = set(map(tuple, a))
    reciprocal = sum((v, u) in pairs for u, v in pairs) / max(len(pairs), 1)
    row = dict(
        partition=partition, trip_id=trip.trip_id,
        n_gps=len(tr), duration_s=float(tr.t.iloc[-1] - tr.t.iloc[0]),
        dt_median=float(np.median(dt)), frac_dt_1s=float(np.mean(dt == 1)),
        n_dup_t=int((dt == 0).sum()), n_back_t=int((dt < 0).sum()),
        n_large_gaps=int((dt > large_gap_s).sum()), max_gap_s=float(dt.max()),
        n_nan=int(tr.isna().sum().sum()),
        lon_min=tr.lon.min(), lon_max=tr.lon.max(), lat_min=tr.lat.min(), lat_max=tr.lat.max(),
        n_nodes=n, n_arcs=len(a), isolated_nodes=int((deg == 0).sum()),
        weak_components=n_comp_weak, strong_components=n_comp_strong,
        frac_reciprocal_arcs=reciprocal, duplicate_arcs=len(a) - len(pairs),
        issues="; ".join(trip.issues),
    )
    if trip.route is not None:
        r = trip.route
        heads, tails = a[r[:-1], 1], a[r[1:], 0]
        row.update(route_len=len(r), route_unique=len(np.unique(r)),
                   route_disconnected_steps=int((heads != tails).sum()),
                   route_repeated_consecutive=int((r[:-1] == r[1:]).sum()))
    return row


def main():
    cfg = load_config()
    rows = []
    for part in ("train", "student_test"):
        for tid in list_trips(cfg, part):
            trip = load_trip(partition_dir(cfg, part) / tid)
            rows.append(trip_stats(trip, part, cfg["gaps"]["large_gap_s"]))
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "experiments" / "results"
    df.to_csv(out / "inventory.csv", index=False)
    lines = [
        df.groupby("partition").agg(trips=("trip_id", "count"), gps=("n_gps", "sum"),
                                    nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median")).to_string(),
        "",
        f"Trips with invariant issues: {(df.issues != '').sum()}",
        f"Trips with large gaps (> {cfg['gaps']['large_gap_s']} s): {(df.n_large_gaps > 0).sum()}",
        f"Overall fraction of 1 s steps (point-weighted): {np.average(df.frac_dt_1s, weights=df.n_gps):.3f}",
        f"Trips with duplicate timestamps: {(df.n_dup_t > 0).sum()}, backwards timestamps: {(df.n_back_t > 0).sum()}",
        f"Median fraction of reciprocal arcs: {df.frac_reciprocal_arcs.median():.3f}",
        f"Trips whose route has disconnected consecutive arcs: {(df.route_disconnected_steps > 0).sum()}",
        f"Total disconnected route steps: {int(df.route_disconnected_steps.sum())} of {int(df.route_len.sum() - len(df))}",
        f"Weak components (median/max): {df.weak_components.median()}/{df.weak_components.max()}",
    ]
    issues = df[df.issues != ""][["partition", "trip_id", "issues"]]
    if len(issues):
        lines += ["", "Issues:", issues.to_string(index=False)]
    (out / "data_quality_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
