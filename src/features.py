"""Candidate generation + feature extraction for one trip.

One row per (GPS point, candidate arc). Works WITHOUT a route (inference): the label
columns are only added when trip["route"] is given. Features never use the route.
"""
import numpy as np
import pandas as pd
from candidates import RoadNet
from geo import seg_dist, bearing
from labels import assign_route_arcs

RADIUS, K = 60.0, 30

FEATURE_COLS = [
    "dist_m", "dist_rel", "cand_rank", "n_cand",
    "proj_frac", "arc_len_m",
    "cos_head", "abs_head_diff_deg", "speed_mps", "moving",
    "dist_prev1", "dist_next1", "dist_prev2", "dist_next2", "dist_win_mean",
    "from_out_deg", "to_out_deg", "dt_prev_s",
]
ID_COLS = ["trip_id", "point_idx", "arc_id", "label"]


def trip_features(trip, radius=RADIUS, k=K):
    """Return (feat_df, point_df). point_df has per-point info (+ reference arc if route given)."""
    net = RoadNet(trip)
    tr = trip["track"]
    N = len(tr)
    P = net.project_points(tr[:, 0], tr[:, 1])
    t = tr[:, 2]
    pi, ai = net.candidates(P, radius, k)

    # --- GPS course/speed from a +-2 point window (NaN-safe when time gap is large) ---
    lo, hi = np.clip(np.arange(N) - 2, 0, N - 1), np.clip(np.arange(N) + 2, 0, N - 1)
    disp = P[hi] - P[lo]
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    dist_w = np.sqrt((disp ** 2).sum(1))
    speed = dist_w / dt
    moving = (dist_w > 3.0) & (speed > 1.0) & (dt <= 10.0)
    head = bearing(disp)
    dt_prev = np.r_[0.0, np.diff(t)]

    # --- candidate geometry ---
    d, frac = seg_dist(P[pi], net.A[ai], net.B[ai])
    dmin = np.full(N, np.inf)
    np.minimum.at(dmin, pi, d)
    ncand = np.bincount(pi, minlength=N)
    start = np.searchsorted(pi, np.arange(N))
    # rank by distance within point
    o = np.lexsort((d, pi))
    rank = np.empty(len(pi), np.int64)
    rank[o] = np.arange(len(pi)) - start[pi[o]]

    dh = head[pi] - net.bearing[ai]
    cos_h = np.where(moving[pi], np.cos(dh), 0.0)
    abs_dh = np.where(moving[pi], np.degrees(np.abs(np.arctan2(np.sin(dh), np.cos(dh)))), 90.0)

    def d_off(off):
        j = np.clip(pi + off, 0, N - 1)
        return seg_dist(P[j], net.A[ai], net.B[ai])[0]

    dp1, dn1, dp2, dn2 = d_off(-1), d_off(1), d_off(-2), d_off(2)
    f = pd.DataFrame({
        "trip_id": trip["trip_id"], "point_idx": pi.astype(np.int32), "arc_id": ai.astype(np.int32),
        "dist_m": d, "dist_rel": d - dmin[pi], "cand_rank": rank, "n_cand": ncand[pi],
        "proj_frac": frac, "arc_len_m": net.length[ai],
        "cos_head": cos_h, "abs_head_diff_deg": abs_dh, "speed_mps": speed[pi], "moving": moving[pi].astype(np.int8),
        "dist_prev1": dp1, "dist_next1": dn1, "dist_prev2": dp2, "dist_next2": dn2,
        "dist_win_mean": (d + dp1 + dn1 + dp2 + dn2) / 5.0,
        "from_out_deg": net.out_deg[net.arcs[ai, 0]], "to_out_deg": net.out_deg[net.arcs[ai, 1]],
        "dt_prev_s": dt_prev[pi],
    })
    pts = pd.DataFrame({"trip_id": trip["trip_id"], "point_idx": np.arange(N, dtype=np.int32),
                        "lon": tr[:, 0], "lat": tr[:, 1], "t": t, "n_cand": ncand})

    if trip.get("route") is not None:
        true_arc, ld = assign_route_arcs(P, t, trip["route"], net)
        pts["true_arc"] = true_arc.astype(np.int32)
        pts["label_dist_m"] = ld
        f["label"] = (f["arc_id"].to_numpy() == true_arc[pi]).astype(np.int8)
        has_pos = np.zeros(N, bool)
        has_pos[pi[f["label"].to_numpy() == 1]] = True
        pts["has_positive"] = has_pos
        f["pt_has_pos"] = has_pos[pi].astype(np.int8)
    f = f.astype({c: "float32" for c in f.columns if f[c].dtype == np.float64})
    return f, pts
