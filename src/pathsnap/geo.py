"""Projection and point-to-segment geometry (all distances in metres)."""
import numpy as np
from pyproj import Transformer

_TRANSFORMERS = {}


def utm_crs(lon, lat):
    """UTM zone EPSG code for a trip, from the median GPS position.

    Trips are spread worldwide (Europe, Americas, Asia, Africa), so one fixed projection is not
    valid; each trip is projected into its own UTM zone (distortion < 0.1% inside a zone).
    """
    lon0, lat0 = float(np.median(lon)), float(np.median(lat))
    zone = int((lon0 + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat0 >= 0 else 32700 + zone}"


def project(lon, lat, crs):
    """WGS84 lon/lat -> projected x/y in metres."""
    if crs not in _TRANSFORMERS:
        _TRANSFORMERS[crs] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = _TRANSFORMERS[crs].transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised projection of points P onto segments AB.

    Returns (distance, fraction along segment in [0, 1]).
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        f = ((px - ax) * dx + (py - ay) * dy) / L2
    f = np.where(L2 > 0, np.clip(f, 0.0, 1.0), 0.0)
    cx, cy = ax + f * dx, ay + f * dy
    return np.hypot(px - cx, py - cy), f


def bearing(dx, dy):
    """Bearing in radians, measured from the +y (north) axis, clockwise."""
    return np.arctan2(dx, dy)


def angle_diff(a, b):
    """Absolute angular difference in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)
