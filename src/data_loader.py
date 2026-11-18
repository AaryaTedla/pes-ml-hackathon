"""Load one PathSnap trip from its four raw files."""
import os
import numpy as np
import pandas as pd


def _read(path, dtype=float, ncols=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c")
    return df.to_numpy()


def load_trip(trip_dir, with_route=True):
    """Return dict(trip_id, track[N,3]=lon,lat,t ; nodes[M,2]=lon,lat ; arcs[K,2]=from,to ; route or None)."""
    trip_id = os.path.basename(os.path.normpath(trip_dir))
    p = os.path.join(trip_dir, trip_id)
    track = _read(p + ".track", float)
    nodes = _read(p + ".nodes", float)
    arcs = _read(p + ".arcs", np.int64)
    route = None
    if with_route:
        r = _read(p + ".route", np.int64)
        route = None if r is None else r.reshape(-1)
    return dict(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)


def list_trips(root, part):
    d = os.path.join(root, part)
    return sorted(os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
