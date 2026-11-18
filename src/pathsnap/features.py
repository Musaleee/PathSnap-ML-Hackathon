"""Features for each (GPS observation, candidate arc) pair.

Only uses the GPS track and the road network — never `.route` — so it is identical at
training and inference time. Rationale for each feature: docs/features.md.
"""
import numpy as np
import pandas as pd

from .geo import angle_diff, bearing, point_segment

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cand", "frac_along", "seg_len",
    "speed", "dt_prev", "dt_next", "moving",
    "heading_diff", "heading_cos", "heading_diff_rev",
    "win_mean_dist", "win_max_dist", "win_mean_heading_diff",
    "in_prev_cands", "in_next_cands", "connected_prev_best", "connected_next_best",
    "out_deg_v", "in_deg_u", "has_reverse",
]

WINDOW = 3          # neighbouring observations on each side used for context features
MIN_SPEED = 1.0     # m/s; below this the GPS heading is treated as unreliable (NaN)


def _vehicle_heading(px, py, t):
    """Centred heading/speed from points t-2 .. t+2 (clipped at the ends)."""
    n = len(px)
    i0, i1 = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
    dx, dy, dt = px[i1] - px[i0], py[i1] - py[i0], t[i1] - t[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(dt > 0, np.hypot(dx, dy) / dt, 0.0)
    head = bearing(dx, dy)
    return head, speed


def build_features(px, py, t, arcs, idx, cand):
    """cand = (point_idx, arc_id, dist, frac) from ArcIndex.query. Returns a DataFrame."""
    pt, arc, dist, frac = cand
    n = len(px)
    a = arcs.to_numpy()
    df = pd.DataFrame({"point": pt, "arc": arc, "dist": dist, "frac_along": frac})

    # distance features relative to the other candidates of the same point
    first = np.searchsorted(pt, pt)
    df["dist_rank"] = np.arange(len(pt)) - first
    n_cand = np.bincount(pt, minlength=n)
    df["n_cand"] = n_cand[pt]
    min_d = dist[first]
    df["dist_minus_min"] = dist - min_d
    df["dist_ratio_min"] = (dist + 1.0) / (min_d + 1.0)
    df["seg_len"] = idx.length[arc]

    # kinematics
    head, speed = _vehicle_heading(px, py, t)
    dt = np.diff(t)
    dt_prev = np.concatenate([[np.nan], dt])
    dt_next = np.concatenate([dt, [np.nan]])
    moving = speed >= MIN_SPEED
    df["speed"], df["dt_prev"], df["dt_next"] = speed[pt], dt_prev[pt], dt_next[pt]
    df["moving"] = moving[pt].astype(float)

    # heading vs. directed arc bearing
    arc_b = bearing(idx.bx - idx.ax, idx.by - idx.ay)
    hd = angle_diff(head[pt], arc_b[arc])
    hd = np.where(moving[pt], hd, np.nan)
    df["heading_diff"] = hd
    df["heading_cos"] = np.cos(hd)
    df["heading_diff_rev"] = np.minimum(hd, np.pi - hd)   # undirected version

    # temporal window: how well does this arc fit the neighbouring observations?
    win_d, win_h = [], []
    for off in range(-WINDOW, WINDOW + 1):
        if off == 0:
            continue
        q = np.clip(pt + off, 0, n - 1)
        d, _ = point_segment(px[q], py[q], idx.ax[arc], idx.ay[arc], idx.bx[arc], idx.by[arc])
        win_d.append(d)
        h = angle_diff(head[q], arc_b[arc])
        win_h.append(np.where(moving[q], h, np.nan))
    win_d, win_h = np.vstack(win_d), np.vstack(win_h)
    df["win_mean_dist"] = win_d.mean(0)
    df["win_max_dist"] = win_d.max(0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df["win_mean_heading_diff"] = np.nanmean(win_h, axis=0)

    # continuity with neighbouring observations' candidates
    n_arcs = len(a)
    keys = set((pt * n_arcs + arc).tolist())
    prev_key, next_key = (pt - 1) * n_arcs + arc, (pt + 1) * n_arcs + arc
    df["in_prev_cands"] = np.fromiter((k in keys for k in prev_key), bool, len(pt)).astype(float)
    df["in_next_cands"] = np.fromiter((k in keys for k in next_key), bool, len(pt)).astype(float)

    best_arc = np.full(n, -1)
    best_arc[pt[df["dist_rank"].to_numpy() == 0]] = arc[df["dist_rank"].to_numpy() == 0]
    u, v = a[arc, 0], a[arc, 1]

    def connected(nb):
        ok = nb >= 0
        nb_ = np.where(ok, nb, 0)
        same = nb_ == arc
        touch = (a[nb_, 1] == u) | (a[nb_, 0] == v) | (a[nb_, 0] == u) | (a[nb_, 1] == v)
        return np.where(ok, (same | touch).astype(float), np.nan)

    prev_best = np.where(pt > 0, best_arc[np.clip(pt - 1, 0, n - 1)], -1)
    next_best = np.where(pt < n - 1, best_arc[np.clip(pt + 1, 0, n - 1)], -1)
    df["connected_prev_best"] = connected(prev_best)
    df["connected_next_best"] = connected(next_best)

    # network structure
    out_deg = np.bincount(a[:, 0], minlength=a.max() + 1)
    in_deg = np.bincount(a[:, 1], minlength=a.max() + 1)
    df["out_deg_v"] = out_deg[v]
    df["in_deg_u"] = in_deg[u]
    pair_key = a[:, 0].astype(np.int64) * (a.max() + 1) + a[:, 1]
    rev_key = a[:, 1].astype(np.int64) * (a.max() + 1) + a[:, 0]
    df["has_reverse"] = np.isin(rev_key, pair_key)[arc].astype(float)
    return df
