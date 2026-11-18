"""Point-level labels from the reference route.

The raw data has NO per-GPS-point labels, only the ordered arc sequence the vehicle
traversed. We align the N GPS points to the M route arcs with a monotone dynamic
programme (like dynamic time warping with a forward-only constraint):

  state j in 0..M-1 = "point i lies on route arc route[j]"
  cost(i, j)  = min(distance(point i, arc route[j]), CLIP)   [metres]
  dp[i, j]    = cost(i, j) + min_{k<=j} ( dp[i-1, k] + lam * (j - k) )

so the route position never goes backwards and skipping arcs costs lam per arc. After a
long time gap (> GAP s) the skip penalty is relaxed since the vehicle may have moved far.
The assigned arc id is route[j*]. This is a reference label, not verified physical truth.
"""
import numpy as np
from geo import seg_dist

CLIP = 200.0   # cap on per-point cost so GPS dropouts don't dominate
LAM = 2.0      # cost (metres-equivalent) per route arc skipped between consecutive points
GAP = 5.0      # seconds; larger gaps use a reduced skip penalty


def assign_route_arcs(P, t, route, net, lam=LAM):
    """Return (true_arc[N], label_dist[N]) for the points P (projected metres)."""
    N, M = len(P), len(route)
    A, B = net.A[route], net.B[route]
    dp = np.empty((N, M), np.float32)
    idx = np.arange(M, dtype=np.float32)
    prev = None
    for s in range(0, N, 512):
        e = min(s + 512, N)
        PP = np.repeat(P[s:e], M, axis=0)
        d, _ = seg_dist(PP, np.tile(A, (e - s, 1)), np.tile(B, (e - s, 1)))
        cost = np.minimum(d.reshape(e - s, M), CLIP).astype(np.float32)
        for i in range(s, e):
            if i == 0:
                row = cost[0]
            else:
                l = lam if (t[i] - t[i - 1]) <= GAP else lam * 0.1
                m = np.minimum.accumulate(prev - l * idx) + l * idx
                row = cost[i - s] + m
            dp[i] = row
            prev = row
    j = np.empty(N, np.int64)
    j[-1] = int(np.argmin(dp[-1]))
    for i in range(N - 2, -1, -1):
        j[i] = int(np.argmin(dp[i, : j[i + 1] + 1]))
    true_arc = route[j]
    dist, _ = seg_dist(P, net.A[true_arc], net.B[true_arc])
    return true_arc, dist
