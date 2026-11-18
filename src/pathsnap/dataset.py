"""Build the (observation, candidate) table for a partition.

Usage:  python -m pathsnap.dataset            # builds train + student_test, writes cache + reports
Output: experiments/cache/<partition>.pkl (not committed) and
        experiments/results/labels_candidates_<partition>.csv (per-trip alignment & miss-rate stats)
"""
import time

import numpy as np
import pandas as pd

from .candidates import ArcIndex
from .config import REPO_ROOT, list_trips, load_config, partition_dir
from .data_io import load_trip
from .features import build_features
from .geo import project, utm_crs
from .labels import align

CACHE = REPO_ROOT / "experiments" / "cache"


def prepare_trip(trip, cfg):
    """Projection + spatial index + candidates + features. Never reads trip.route."""
    crs = utm_crs(trip.track.lon.to_numpy(), trip.track.lat.to_numpy())
    gx, gy = project(trip.track.lon.to_numpy(), trip.track.lat.to_numpy(), crs)
    nx, ny = project(trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy(), crs)
    c = cfg["candidates"]
    idx = ArcIndex(trip.arcs, nx, ny, c["sample_step_m"])
    cand = idx.query(gx, gy, c["radius_m"], c["max_k"])
    feats = build_features(gx, gy, trip.track.t.to_numpy(), trip.arcs, idx, cand)
    return feats, (gx, gy, nx, ny, idx)


def build_trip(trip, cfg):
    feats, (gx, gy, nx, ny, _) = prepare_trip(trip, cfg)
    lab = align(gx, gy, trip, nx, ny, cfg["labels"]["ambiguous_distance_m"])
    ref = lab["ref_arc"] if cfg["labels"]["method"] == "monotonic" else lab["ref_arc_nearest"]
    feats["label"] = (feats["arc"].to_numpy() == ref[feats["point"].to_numpy()]).astype(np.int8)
    feats["ambiguous"] = lab["ambiguous"][feats["point"].to_numpy()]
    feats.insert(0, "trip_id", trip.trip_id)

    n = len(gx)
    has_cand = np.bincount(feats.point, minlength=n) > 0
    hit = np.zeros(n, bool)
    hit[feats.point[feats.label == 1].to_numpy()] = True
    evaluable = ~lab["ambiguous"]
    stats = dict(
        trip_id=trip.trip_id, n_points=n,
        strategies_agree=float(np.mean(lab["ref_arc"] == lab["ref_arc_nearest"])),
        ref_dist_median=float(np.median(lab["ref_dist"])), ref_dist_p95=float(np.percentile(lab["ref_dist"], 95)),
        frac_ambiguous=float(np.mean(lab["ambiguous"])),
        frac_zero_candidates=float(np.mean(~has_cand)),
        miss_rate_all=float(np.mean(~hit)),
        miss_rate_evaluable=float(np.mean(~hit[evaluable])) if evaluable.any() else np.nan,
        cands_per_point=float(len(feats) / n),
    )
    # per-point label table, kept for evaluation (offline only)
    points = pd.DataFrame({"trip_id": trip.trip_id, "point": np.arange(n), "ref_arc": ref,
                           "ambiguous": lab["ambiguous"], "ref_in_cands": hit, "t": trip.track.t.to_numpy()})
    return feats, points, stats


def build_partition(partition, cfg):
    feats, points, stats = [], [], []
    for tid in list_trips(cfg, partition):
        t0 = time.time()
        trip = load_trip(partition_dir(cfg, partition) / tid)
        f, p, s = build_trip(trip, cfg)
        s["seconds"] = round(time.time() - t0, 2)
        feats.append(f), points.append(p), stats.append(s)
        print(f"{partition} {tid}: {s['n_points']} pts, {s['cands_per_point']:.1f} cand/pt, "
              f"miss {s['miss_rate_evaluable']:.3f}, ambiguous {s['frac_ambiguous']:.3f}, {s['seconds']} s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat(feats, ignore_index=True).to_pickle(CACHE / f"{partition}.pkl")
    pd.concat(points, ignore_index=True).to_pickle(CACHE / f"{partition}_points.pkl")
    st = pd.DataFrame(stats)
    st.to_csv(REPO_ROOT / "experiments" / "results" / f"labels_candidates_{partition}.csv", index=False)
    w = st.n_points / st.n_points.sum()
    print(f"[{partition}] strategies agree {np.sum(w * st.strategies_agree):.3f} | ambiguous {np.sum(w * st.frac_ambiguous):.3f} | "
          f"miss (evaluable) {np.nansum(w * st.miss_rate_evaluable):.3f} | zero-cand {np.sum(w * st.frac_zero_candidates):.4f}")


def main():
    cfg = load_config()
    for part in ("train", "student_test"):
        build_partition(part, cfg)


if __name__ == "__main__":
    main()
