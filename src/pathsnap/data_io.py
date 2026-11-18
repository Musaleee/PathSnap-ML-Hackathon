"""Readers for the raw PathSnap files and invariant checks.

All files are whitespace-separated without headers (Dataset Exploration §3).
Node and arc ids are the 0-based row index and are local to one trip.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read(path, names, dtypes):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="c")
    return df.astype(dtypes)


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame            # lon, lat, t
    nodes: pd.DataFrame            # lon, lat  (index = node_id)
    arcs: pd.DataFrame             # u, v      (index = arc_id)
    route: Optional[np.ndarray]    # arc ids, ordered; None when not available
    issues: list = field(default_factory=list)


def load_trip(trip_dir, load_route=True):
    """Load one trip. `load_route=False` must be used by the inference pipeline."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"], {"lon": float, "lat": float, "t": float})
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"], {"lon": float, "lat": float})
    arcs = _read(trip_dir / f"{tid}.arcs", ["u", "v"], {"u": np.int64, "v": np.int64})
    route = None
    rp = trip_dir / f"{tid}.route"
    if load_route and rp.exists():
        route = _read(rp, ["arc"], {"arc": np.int64})["arc"].to_numpy()
    trip = Trip(tid, track, nodes, arcs, route)
    trip.issues = check_invariants(trip)
    return trip


def check_invariants(trip):
    """Return a list of human-readable problems (empty list = all good). Never modifies data."""
    issues = []
    n_nodes, n_arcs = len(trip.nodes), len(trip.arcs)
    tr = trip.track
    if not np.isfinite(tr[["lon", "lat", "t"]].to_numpy()).all():
        issues.append("track: non-finite values")
    if ((tr.lon.abs() > 180) | (tr.lat.abs() > 90)).any():
        issues.append("track: coordinates out of range")
    dt = np.diff(tr.t.to_numpy())
    if (dt < 0).any():
        issues.append(f"track: {(dt < 0).sum()} backwards timestamps")
    if (dt == 0).any():
        issues.append(f"track: {(dt == 0).sum()} duplicate timestamps")
    if not np.isfinite(trip.nodes.to_numpy()).all():
        issues.append("nodes: non-finite values")
    a = trip.arcs.to_numpy()
    if a.min() < 0 or a.max() >= n_nodes:
        issues.append("arcs: endpoint outside node range")
    if (a[:, 0] == a[:, 1]).any():
        issues.append(f"arcs: {(a[:, 0] == a[:, 1]).sum()} self-loops")
    if trip.route is not None:
        r = trip.route
        if r.min() < 0 or r.max() >= n_arcs:
            issues.append("route: arc id outside arc range")
    return issues
