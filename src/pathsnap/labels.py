"""Reference-label alignment: give each GPS observation a reference arc from `.route`.

`.route` is an ordered arc sequence, not one label per GPS row, so labels must be derived.
Two strategies are implemented and compared (plan v2, Gate B):

* nearest   – baseline: nearest arc among the route's arcs, ignoring order.
* monotonic – order-preserving: each point gets a position in the route sequence that never
              decreases over time; total point-to-arc distance is minimised by dynamic programming.

Used for training/offline evaluation only. The inference pipeline never calls this module.
"""
import numpy as np

from .geo import point_segment


def _route_geometry(trip, x_nodes, y_nodes):
    a = trip.arcs.to_numpy()[trip.route]
    return x_nodes[a[:, 0]], y_nodes[a[:, 0]], x_nodes[a[:, 1]], y_nodes[a[:, 1]]


def align_nearest(px, py, trip, x_nodes, y_nodes):
    ax, ay, bx, by = _route_geometry(trip, x_nodes, y_nodes)
    best_pos = np.empty(len(px), dtype=np.int64)
    best_d = np.empty(len(px))
    for t in range(len(px)):
        d, _ = point_segment(px[t], py[t], ax, ay, bx, by)
        best_pos[t] = d.argmin()
        best_d[t] = d[best_pos[t]]
    return trip.route[best_pos], best_d, best_pos


def align_monotonic(px, py, trip, x_nodes, y_nodes):
    """DP: D[t, i] = dist(p_t, route_i) + min_{j <= i} D[t-1, j]; backtrack the best path."""
    ax, ay, bx, by = _route_geometry(trip, x_nodes, y_nodes)
    n, m = len(px), len(ax)
    idx = np.arange(m)
    ptr = np.empty((n, m), dtype=np.int32)
    prev, _ = point_segment(px[0], py[0], ax, ay, bx, by)
    ptr[0] = idx
    for t in range(1, n):
        run_min = np.minimum.accumulate(prev)
        # last index j <= i that attains the running minimum
        ptr[t] = np.maximum.accumulate(np.where(prev == run_min, idx, 0))
        d, _ = point_segment(px[t], py[t], ax, ay, bx, by)
        prev = d + run_min
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(prev.argmin())
    for t in range(n - 1, 0, -1):
        pos[t - 1] = ptr[t, pos[t]]
    d, _ = point_segment(px, py, ax[pos], ay[pos], bx[pos], by[pos])
    return trip.route[pos], d, pos


def align(px, py, trip, x_nodes, y_nodes, ambiguous_distance_m):
    """Return dict with the chosen (monotonic) label, the baseline label and an ambiguity flag."""
    arc_m, d_m, pos_m = align_monotonic(px, py, trip, x_nodes, y_nodes)
    arc_n, d_n, _ = align_nearest(px, py, trip, x_nodes, y_nodes)
    ambiguous = (d_m > ambiguous_distance_m) | (arc_m != arc_n)
    return dict(ref_arc=arc_m, ref_dist=d_m, ref_pos=pos_m, ref_arc_nearest=arc_n,
                ref_dist_nearest=d_n, ambiguous=ambiguous)
