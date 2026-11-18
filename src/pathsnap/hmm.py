"""HMM + Viterbi decoding over candidate arcs, and route reconstruction.

Hidden state at observation t  = one of the candidate arcs of that GPS point (the candidate row keeps
                                 the arc id, so the state <-> road-segment mapping is preserved).
Emission  log P(z_t | s)       = w * log( p_model(s) / sum_candidates p_model )   (selected classifier)
Transition log P(s_t | s_t-1)  = -| d_route(s_t-1 -> s_t) - d_gc(z_t-1, z_t) | / beta
    d_route: driving distance along the directed network from the projected position on the previous
             arc to the projected position on the next arc (bounded Dijkstra); d_gc: straight-line
             distance between the two GPS points. Unreachable within the search bound -> impossible.
Chain breaks: time gap > gap_break_s, or no possible transition at all -> decoding restarts from the
emissions at that point (the break is reported; connectivity is never invented).
Points without candidates have no state and get no prediction (they are skipped, the chain continues).
"""
import heapq
from dataclasses import dataclass

import numpy as np

INF = np.inf


@dataclass
class HMMParams:
    beta: float = 10.0            # metres; tolerance of route-vs-GPS distance mismatch
    emission_weight: float = 1.0
    gap_break_s: float = 60.0     # restart the chain after a longer time gap
    route_factor: float = 2.0     # search bound: route_factor * d_gc + slack_m
    slack_m: float = 200.0
    max_search_m: float = 5000.0
    back_tol_m: float = 15.0      # backwards movement on the same arc accepted as GPS noise


class Graph:
    """Directed road graph with bounded, cached Dijkstra from nodes."""

    def __init__(self, arcs, length):
        a = np.asarray(arcs)
        self.u, self.v, self.length = a[:, 0], a[:, 1], np.asarray(length, float)
        order = np.argsort(self.u, kind="stable")
        n_nodes = int(a.max()) + 1
        self.indptr = np.searchsorted(self.u[order], np.arange(n_nodes + 1))
        self.out_arc = order
        self._cache = {}

    def sssp(self, src, cutoff):
        """Distances (dict node->m) and predecessor arcs (dict node->arc) from src, up to cutoff."""
        c = self._cache.get(src)
        if c is not None and c[0] >= cutoff:
            return c[1], c[2]
        dist, pred = {src: 0.0}, {}
        heap = [(0.0, src)]
        indptr, out_arc, v, L = self.indptr, self.out_arc, self.v, self.length
        while heap:
            d, x = heapq.heappop(heap)
            if d > dist.get(x, INF) or d > cutoff:
                continue
            for k in range(indptr[x], indptr[x + 1]):
                e = out_arc[k]
                nd = d + L[e]
                y = v[e]
                if nd < dist.get(y, INF) and nd <= cutoff:
                    dist[y] = nd
                    pred[y] = e
                    heapq.heappush(heap, (nd, y))
        self._cache[src] = (cutoff, dist, pred)
        return dist, pred

    def path_arcs(self, src, dst, cutoff):
        dist, pred = self.sssp(src, cutoff)
        if dst not in dist:
            return None
        out, x = [], dst
        while x != src:
            e = pred[x]
            out.append(int(e))
            x = self.u[e]
        return out[::-1]


def precompute(points, arcs_c, frac, t, px, py, graph, prm):
    """Route-distance matrices between consecutive candidate sets (independent of beta / weight).

    points/arcs_c/frac: candidate rows sorted by point. Returns dict used by viterbi()."""
    upts, start = np.unique(points, return_index=True)
    end = np.append(start[1:], len(points))
    L = graph.length
    steps = [None]
    for k in range(1, len(upts)):
        i, j = upts[k - 1], upts[k]
        dt = t[j] - t[i]
        gc = float(np.hypot(px[j] - px[i], py[j] - py[i]))
        if dt > prm.gap_break_s:
            steps.append(("gap", gc, None))
            continue
        cutoff = min(prm.route_factor * gc + prm.slack_m, prm.max_search_m)
        A, fa = arcs_c[start[k - 1]:end[k - 1]], frac[start[k - 1]:end[k - 1]]
        B, fb = arcs_c[start[k]:end[k]], frac[start[k]:end[k]]
        D = np.full((len(A), len(B)), INF)
        for r, (a, f_a) in enumerate(zip(A, fa)):
            rest = (1.0 - f_a) * L[a]
            dist, _ = graph.sssp(graph.v[a], cutoff)
            for c, (b, f_b) in enumerate(zip(B, fb)):
                if a == b:
                    d = (f_b - f_a) * L[a]
                    if d >= -prm.back_tol_m:
                        D[r, c] = max(d, 0.0)
                        continue
                du = dist.get(graph.u[b])
                if du is not None:
                    d = rest + du + f_b * L[b]
                    if d <= cutoff:
                        D[r, c] = min(D[r, c], d)
        steps.append(("ok", gc, D))
    return dict(upts=upts, start=start, end=end, steps=steps)


def viterbi(pre, prob, prm):
    """Returns (chosen candidate-row index per state point, chain-break flag per state point)."""
    upts, start, end, steps = pre["upts"], pre["start"], pre["end"], pre["steps"]
    n = len(upts)
    if n == 0:
        return np.empty(0, int), np.empty(0, bool)
    emis = []
    for k in range(n):
        p = np.clip(prob[start[k]:end[k]], 1e-9, None)
        emis.append(prm.emission_weight * np.log(p / p.sum()))
    back = [None] * n
    seg_end = {}                                   # k-1 -> final scores of the segment ending there
    brk = np.zeros(n, bool)
    brk[0] = True
    delta = emis[0]
    for k in range(1, n):
        kind, gc, D = steps[k]
        if kind == "ok":
            tot = delta[:, None] - np.abs(D - gc) / prm.beta     # -inf where unreachable
            bp = np.argmax(tot, axis=0)
            best = tot[bp, np.arange(tot.shape[1])]
            if np.isfinite(best).any():
                back[k] = bp
                delta = best + emis[k]
                continue
        brk[k] = True                                  # time gap or no possible transition
        seg_end[k - 1] = delta
        delta = emis[k]
    choice = np.empty(n, int)
    s = int(np.argmax(delta))
    for k in range(n - 1, -1, -1):
        choice[k] = s
        if k == 0:
            break
        s = int(np.argmax(seg_end[k - 1])) if brk[k] else int(back[k][s])
    return start + choice, brk


def expand_route(arc_seq, graph, prm, gc_hint=500.0):
    """Turn per-observation arcs into a route of real arcs. Returns (route_arcs, n_breaks).

    Consecutive different arcs are joined by the shortest directed path; if there is none within the
    bound, a break (-1) is inserted rather than inventing a connection."""
    route, breaks = [], 0
    prev = None
    for a in arc_seq:
        a = int(a)
        if a < 0:                      # explicit chain break
            if route and route[-1] != -1:
                route.append(-1)
            prev = None
            continue
        if prev is None:
            route.append(a)
        elif a != prev:
            if graph.v[prev] == graph.u[a]:
                route.append(a)
            else:
                p = graph.path_arcs(graph.v[prev], graph.u[a], max(gc_hint, prm.slack_m) * 4)
                if p is None:
                    route.append(-1)
                    breaks += 1
                    route.append(a)
                else:
                    route.extend(p + [a])
        prev = a
    return route, breaks


def edit_distance(a, b):
    """Levenshtein distance between two integer sequences (vectorised row update)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return max(len(a), len(b))
    prev = np.arange(len(b) + 1, dtype=np.int64)
    j = np.arange(len(b) + 1)
    for x in a:
        cur = np.empty_like(prev)
        cur[0] = prev[0] + 1
        cur[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b != x))
        cur = np.minimum.accumulate(cur - j) + j       # insertion chain
        prev = cur
    return int(prev[-1])


def route_metrics(pred_route, ref_route):
    """Sequence-level comparison of a reconstructed route with the reference route (offline only)."""
    p = [a for a in pred_route if a >= 0]
    collapse = lambda s: [x for i, x in enumerate(s) if i == 0 or x != s[i - 1]]
    p, r = collapse(p), collapse(list(map(int, ref_route)))
    ps, rs = set(p), set(r)
    inter = len(ps & rs)
    prec = inter / len(ps) if ps else 0.0
    rec = inter / len(rs) if rs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ed = edit_distance(p, r)
    return dict(route_precision=prec, route_recall=rec, route_f1=f1,
                route_norm_edit=ed / max(len(p), len(r), 1))
