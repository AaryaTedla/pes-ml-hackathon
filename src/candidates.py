"""Per-trip road network + candidate-arc generation (STRtree spatial index)."""
import numpy as np
import shapely
from shapely import STRtree
from geo import make_projector, project, seg_dist, bearing


class RoadNet:
    def __init__(self, trip):
        nodes, arcs, track = trip["nodes"], trip["arcs"], trip["track"]
        self.tf = make_projector(track[:, 0], track[:, 1])
        self.node_xy = project(self.tf, nodes[:, 0], nodes[:, 1])
        self.arcs = arcs
        self.A = self.node_xy[arcs[:, 0]]
        self.B = self.node_xy[arcs[:, 1]]
        d = self.B - self.A
        self.length = np.sqrt((d ** 2).sum(1))
        self.bearing = bearing(d)
        self.out_deg = np.bincount(arcs[:, 0], minlength=len(nodes))
        self.in_deg = np.bincount(arcs[:, 1], minlength=len(nodes))
        lines = shapely.linestrings(np.stack([self.A, self.B], axis=1))
        self.tree = STRtree(lines)

    def project_points(self, lon, lat):
        return project(self.tf, lon, lat)

    def candidates(self, P, radius=60.0, k=30, fallback=300.0):
        """Return (pt_idx, arc_idx) pairs: arcs within `radius` m of each point, at most `k`
        nearest per point. Points with none in radius get their single nearest arc within
        `fallback` m (so almost every point has >=1 candidate). Labels are never used here."""
        pts = shapely.points(P)
        pi, ai = self.tree.query(pts, predicate="dwithin", distance=radius)
        d, _ = seg_dist(P[pi], self.A[ai], self.B[ai])
        order = np.lexsort((d, pi))
        pi, ai, d = pi[order], ai[order], d[order]
        start = np.searchsorted(pi, np.arange(len(P)))
        rank = np.arange(len(pi)) - start[pi]
        keep = rank < k
        pi, ai = pi[keep], ai[keep]
        have = np.zeros(len(P), bool)
        have[pi] = True
        miss = np.where(~have)[0]
        if len(miss):
            qi, qa = self.tree.query_nearest(pts[miss], max_distance=fallback, all_matches=False)
            pi = np.concatenate([pi, miss[qi]])
            ai = np.concatenate([ai, qa])
            o = np.argsort(pi, kind="stable")
            pi, ai = pi[o], ai[o]
        return pi, ai
