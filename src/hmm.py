"""HMM / Viterbi route reconstruction.

Hidden state at GPS step i  = the road arc (one of the candidate arcs of point i).
Emission                    = log P(arc is correct | features) from the classifier  (`prob` column).
Transition a -> b           = -penalty, where
    same arc                      0
    b reachable from a in h hops  hop_penalty * h        (h = number of arcs travelled, BFS on the road graph)
    b is the reverse twin of a    uturn_penalty          (instead of hop_penalty when h == 1)
    not reachable within depth    jump_penalty           (finite, so the decoder can recover from bad candidates)
BFS depth = max_hops + floor(min(dt, gap_s) / gap_hop_s): longer time steps allow more arcs to be travelled.
GPS gaps: if dt > gap_s the chain is *relaxed* (no transition penalty): the vehicle could be anywhere
reachable, so only the emissions decide, then constraints resume.
Points with no candidate arc are skipped (output -1) and the chain continues across them.
Viterbi recurrence (log space):  delta_i(b) = e_i(b) + max_a [ delta_{i-1}(a) - pen(a,b) ].
"""
import numpy as np

DEFAULT_CFG = dict(hop_penalty=1.0, jump_penalty=12.0, uturn_penalty=4.0,
                   max_hops=3, gap_s=30.0, gap_hop_s=5.0, eps=1e-6)


class _Graph:
    """Arc adjacency with memoised bounded BFS (shared across calls on the same trip)."""

    def __init__(self, arcs, n_nodes):
        self.arcs = arcs
        self.order = np.argsort(arcs[:, 0], kind="stable")
        self.start = np.searchsorted(arcs[self.order, 0], np.arange(n_nodes + 1))
        self.memo = {}

    def succ(self, a):
        n = self.arcs[a, 1]
        return self.order[self.start[n]: self.start[n + 1]]

    def reach(self, a, depth):
        key = (a, depth)
        r = self.memo.get(key)
        if r is None:
            r = {a: 0}
            frontier = [a]
            for h in range(1, depth + 1):
                nxt = []
                for x in frontier:
                    for y in self.succ(x).tolist():
                        if y not in r:
                            r[y] = h
                            nxt.append(y)
                frontier = nxt
                if not frontier:
                    break
            self.memo[key] = r
        return r

    def is_twin(self, a, b):
        return self.arcs[a, 0] == self.arcs[b, 1] and self.arcs[a, 1] == self.arcs[b, 0]


def _graph(trip):
    g = trip.get("_hmm_graph")
    if g is None:
        g = trip["_hmm_graph"] = _Graph(trip["arcs"], len(trip["nodes"]))
    return g


def decode(trip, feats, cfg=None):
    """feats: DataFrame with point_idx, arc_id, prob (one row per candidate). Returns int64[N] arc per point, -1 if none."""
    c = {**DEFAULT_CFG, **(cfg or {})}
    g = _graph(trip)
    N = len(trip["track"])
    t = trip["track"][:, 2]
    pi = feats["point_idx"].to_numpy()
    o = np.argsort(pi, kind="stable")
    pi = pi[o]
    ai = feats["arc_id"].to_numpy()[o]
    lp = np.log(np.clip(feats["prob"].to_numpy()[o], c["eps"], 1.0))
    starts = np.searchsorted(pi, np.arange(N + 1))
    out = np.full(N, -1, np.int64)

    steps, cand, back = [], [], []
    score = None
    prev_i = None
    for i in range(N):
        s0, s1 = starts[i], starts[i + 1]
        if s1 == s0:
            continue
        B = ai[s0:s1]
        e = lp[s0:s1]
        if score is None:
            score = e.copy()
            bp = np.zeros(len(B), np.int64)
        else:
            A = cand[-1]
            dt = t[i] - t[prev_i]
            if dt <= 0:
                dt = 1.0
            if dt > c["gap_s"]:
                pen = np.zeros((len(A), len(B)))                      # gap: relax the chain
            else:
                depth = int(c["max_hops"] + min(dt, c["gap_s"]) // c["gap_hop_s"])
                pen = np.full((len(A), len(B)), c["jump_penalty"])
                bl = B.tolist()
                for ia, a in enumerate(A.tolist()):
                    d = g.reach(a, depth)
                    for ib, b in enumerate(bl):
                        h = d.get(b)
                        if h is None:
                            continue
                        if h == 0:
                            pen[ia, ib] = 0.0
                        elif h == 1 and g.is_twin(a, b):
                            pen[ia, ib] = c["uturn_penalty"]
                        else:
                            pen[ia, ib] = c["hop_penalty"] * h
            tot = score[:, None] - pen
            bp = tot.argmax(0)
            score = e + tot[bp, np.arange(len(B))]
        steps.append(i)
        cand.append(B)
        back.append(bp)
        prev_i = i
    if not steps:
        return out
    k = int(score.argmax())
    for j in range(len(steps) - 1, -1, -1):
        out[steps[j]] = cand[j][k]
        k = int(back[j][k])
    return out
