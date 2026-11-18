"""Candidate road segments for each GPS observation, using a KD-tree over densified arcs.

Avoids a brute-force N_gps x N_arcs scan: arcs are sampled every `step` metres, a KD-tree
finds samples near each GPS point, and exact point-to-segment distance is computed only for
those arcs.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geo import point_segment


class ArcIndex:
    def __init__(self, arcs, x_nodes, y_nodes, step):
        a = arcs.to_numpy()
        self.ax, self.ay = x_nodes[a[:, 0]], y_nodes[a[:, 0]]
        self.bx, self.by = x_nodes[a[:, 1]], y_nodes[a[:, 1]]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.step = step
        n_s = np.maximum(np.ceil(self.length / step).astype(int), 1) + 1
        arc_of_sample = np.repeat(np.arange(len(a)), n_s)
        start = np.repeat(np.cumsum(n_s) - n_s, n_s)
        frac = (np.arange(n_s.sum()) - start) / np.repeat(n_s - 1, n_s)
        sx = self.ax[arc_of_sample] + frac * (self.bx - self.ax)[arc_of_sample]
        sy = self.ay[arc_of_sample] + frac * (self.by - self.ay)[arc_of_sample]
        self.arc_of_sample = arc_of_sample
        self.tree = cKDTree(np.column_stack([sx, sy]))

    def query(self, px, py, radius, max_k):
        """Return arrays (point_idx, arc_id, dist, frac) of candidates within `radius`, at most max_k per point."""
        n_arcs = len(self.ax)
        hits = self.tree.query_ball_point(np.column_stack([px, py]), r=radius + self.step / 2, workers=-1)
        lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if lens.sum() == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, np.empty(0), np.empty(0)
        pt = np.repeat(np.arange(len(hits)), lens)
        arc = self.arc_of_sample[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if h])]
        key = np.unique(pt * n_arcs + arc)
        pt, arc = key // n_arcs, key % n_arcs
        d, f = point_segment(px[pt], py[pt], self.ax[arc], self.ay[arc], self.bx[arc], self.by[arc])
        keep = d <= radius
        pt, arc, d, f = pt[keep], arc[keep], d[keep], f[keep]
        order = np.lexsort((d, pt))                      # by point, then distance
        pt, arc, d, f = pt[order], arc[order], d[order], f[order]
        rank = np.arange(len(pt)) - np.searchsorted(pt, pt)  # 0-based rank within point
        keep = rank < max_k
        return pt[keep], arc[keep], d[keep], f[keep]
