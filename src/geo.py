"""Geometry helpers. Each trip is projected to its own UTM zone (metres), since
trips are scattered worldwide and degrees are not distances."""
import numpy as np
from pyproj import Transformer


def make_projector(lon, lat):
    """UTM transformer for the trip centroid (lon/lat in degrees -> x/y metres)."""
    zone = int((np.mean(lon) + 180) // 6) % 60 + 1
    epsg = (32600 if np.mean(lat) >= 0 else 32700) + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def project(tf, lon, lat):
    x, y = tf.transform(np.asarray(lon), np.asarray(lat))
    return np.column_stack([x, y])


def seg_dist(P, A, B):
    """Distance from points P to segments AB (all [n,2]); also clamped projection fraction t."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((P - A) * AB).sum(1) / L2, 0.0, 1.0)
    proj = A + t[:, None] * AB
    return np.sqrt(((P - proj) ** 2).sum(1)), t


def bearing(dxy):
    """Compass bearing in radians (0 = north, clockwise) of displacement vectors [n,2]."""
    return np.arctan2(dxy[:, 0], dxy[:, 1])
