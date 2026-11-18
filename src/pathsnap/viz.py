"""Folium maps (offline/development mode only: these maps may show the reference route).

Layers: road network near the track, raw GPS, reference route, classifier-only picks, HMM route.
Usage:  python -m pathsnap.viz student_test 00000077 00000052 ...   (default: 3 student-test trips)
Output: maps/<partition>_<trip>.html
"""
import sys

import folium
import numpy as np
from scipy.spatial import cKDTree

from .config import REPO_ROOT, load_config, partition_dir
from .data_io import load_trip
from .pipeline import Predictor, route_lines

MAPS = REPO_ROOT / "maps"
COL = dict(network="#9AA0A6", gps="#1F77B4", ref="#2CA02C", clf="#FF7F0E", hmm="#D62728")


def _legend(items):
    rows = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{t}</div>' for t, c in items)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #888;border-radius:6px;font:13px Arial">{rows}</div>')
    return folium.Element(html)


def network_near_track(trip, radius_deg=0.0015):
    lon, lat = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy()
    tree = cKDTree(np.column_stack([trip.track.lon, trip.track.lat]))
    near = tree.query(np.column_stack([lon, lat]), distance_upper_bound=radius_deg)[0] < np.inf
    a = trip.arcs.to_numpy()
    keep = near[a[:, 0]] & near[a[:, 1]]
    return [[[lat[u], lon[u]], [lat[v], lon[v]]] for u, v in a[keep]]


def make_map(trip, pred=None, ref_route=None, title=""):
    lat, lon = trip.track.lat.to_numpy(), trip.track.lon.to_numpy()
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    nodes_lon, nodes_lat, arcs = trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy(), trip.arcs.to_numpy()

    fg = folium.FeatureGroup("Road network (near track)")
    folium.PolyLine(network_near_track(trip), color=COL["network"], weight=1, opacity=0.6).add_to(fg)
    fg.add_to(m)

    fg = folium.FeatureGroup("Raw GPS observations")
    step = max(1, len(lat) // 2500)
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color=COL["gps"], fill=True, weight=0,
                            fill_opacity=0.7, tooltip=f"point {i}, t={trip.track.t.iloc[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    items = [("Road network", COL["network"]), ("Raw GPS", COL["gps"])]

    if ref_route is not None:
        fg = folium.FeatureGroup("Reference route (offline only)")
        for ln in route_lines(ref_route, arcs, nodes_lon, nodes_lat):
            folium.PolyLine([[y, x] for x, y in ln], color=COL["ref"], weight=7, opacity=0.5).add_to(fg)
        fg.add_to(m)
        items.append(("Reference route", COL["ref"]))
    if pred is not None:
        fg = folium.FeatureGroup("Classifier-only picks", show=False)
        ca = pred["clf_point_arcs"]
        for i in range(0, len(ca), step):
            if ca[i] >= 0:
                u, v = arcs[ca[i]]
                folium.PolyLine([[nodes_lat[u], nodes_lon[u]], [nodes_lat[v], nodes_lon[v]]],
                                color=COL["clf"], weight=3, opacity=0.8).add_to(fg)
        fg.add_to(m)
        fg = folium.FeatureGroup("Predicted route (classifier + HMM)")
        for ln in route_lines(pred["route_arcs"], arcs, nodes_lon, nodes_lat):
            folium.PolyLine([[y, x] for x, y in ln], color=COL["hmm"], weight=3, opacity=0.9).add_to(fg)
        fg.add_to(m)
        items += [("Classifier-only picks", COL["clf"]), ("Predicted route (HMM)", COL["hmm"])]
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(items))
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:4px 8px;'
            f'border:1px solid #888;border-radius:4px;font:14px Arial"><b>{title}</b></div>'))
    return m


def main(partition="student_test", *trips):
    cfg = load_config()
    trips = trips or ("00000052", "00000040", "00000077")
    MAPS.mkdir(exist_ok=True)
    P = Predictor(cfg=cfg)
    for tid in trips:
        d = partition_dir(cfg, partition) / tid
        pred = P.predict(d)
        ref = load_trip(d).route
        sc = P.score(d, pred)
        title = (f"{partition} {tid} | Top-1 HMM {sc['top1_hmm']:.3f} vs classifier {sc['top1_classifier_only']:.3f} "
                 f"| route F1 {sc['route_f1']:.3f}")
        make_map(pred["trip"], pred, ref, title).save(MAPS / f"{partition}_{tid}.html")
        print(title, flush=True)
    # EDA map (raw data only): GPS + network + reference route
    t = load_trip(partition_dir(cfg, "train") / "00000001")
    make_map(t, None, t.route, "train 00000001 | raw GPS, road network, reference route").save(MAPS / "eda_train_00000001.html")


if __name__ == "__main__":
    main(*sys.argv[1:])
