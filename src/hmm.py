"""Directed-road HMM. Inference never reads reference routes or labels.

States are candidate arcs; emissions are classifier probabilities (distance-based
in the demo). Transitions penalise |directed network distance - GPS displacement|.
Large/nonpositive timestamp gaps start independent Viterbi sequences.
"""
from dataclasses import dataclass
import heapq
import numpy as np
from candidates import RoadNet
from geo import seg_dist


@dataclass
class HMMConfig:
    sigma_m: float = 20.0
    heading_weight: float = 2.0
    beta_m: float = 30.0
    gap_s: float = 30.0
    max_speed_mps: float = 55.0
    search_slack_m: float = 100.0
    unreachable_penalty: float = 30.0
    radius_m: float = 60.0
    max_candidates: int = 12


def mock_probabilities(feats, config=None):
    """Softmax(-dist_m / sigma_m + heading_weight * cos_head) per point."""
    cfg = config or HMMConfig()
    result = feats.copy()
    logits = -result['dist_m'] / cfg.sigma_m + cfg.heading_weight * result['cos_head']
    weights = np.exp(logits - logits.groupby(result['point_idx']).transform('max'))
    result['probability'] = weights / weights.groupby(result['point_idx']).transform('sum')
    return result


def mock_candidates(trip, net=None, config=None):
    """Return point_idx, arc_id, probability rows without using ground truth."""
    cfg = config or HMMConfig()
    net = net or RoadNet(trip)
    p = net.project_points(*trip['track'][:, :2].T)
    pi, ai = net.candidates(p, radius=cfg.radius_m, k=cfg.max_candidates)
    # Unbounded nearest fallback ensures one output for every observation.
    missing = np.setdiff1d(np.arange(len(p)), pi)
    if len(missing):
        import shapely
        ai = np.concatenate((ai, net.tree.nearest(shapely.points(p[missing]))))
        pi = np.concatenate((pi, missing))
    d, _ = seg_dist(p[pi], net.A[ai], net.B[ai])
    logp = -0.5 * (d / cfg.sigma_m) ** 2
    probs = np.empty(len(pi))
    for i in range(len(p)):
        ix = np.flatnonzero(pi == i)
        weights = np.exp(logp[ix] - logp[ix].max())
        probs[ix] = weights / weights.sum()
    return dict(point_idx=pi, arc_id=ai, probability=probs)


def _distances(adj, source, targets, limit):
    """Bounded directed Dijkstra, stopping when all target nodes are reached."""
    todo = set(targets)
    found, best, queue = {}, {source: 0.0}, [(0.0, source)]
    while queue and todo:
        dist, u = heapq.heappop(queue)
        if dist != best[u]:
            continue
        if u in todo:
            found[u] = dist
            todo.remove(u)
        for v, length in adj[u]:
            nd = dist + length
            if nd <= limit and nd < best.get(v, float('inf')):
                best[v] = nd
                heapq.heappush(queue, (nd, v))
    return found


def decode(trip, candidates_with_probs, config=None, net=None):
    """Return one arc id per point. Rows require point_idx, arc_id, probability.

    Accepts a dict of arrays or a pandas DataFrame. Candidate probabilities must
    be finite in [0,1]. Missing point rows receive distance-scored nearest arcs.
    """
    cfg = config or HMMConfig()
    if min(cfg.sigma_m, cfg.beta_m, cfg.gap_s, cfg.max_speed_mps) <= 0:
        raise ValueError('Distance scales, speed and gap must be positive')
    net = net or RoadNet(trip)
    p = net.project_points(*trip['track'][:, :2].T)
    n = len(p)
    if not n:
        return []
    pi = np.asarray(candidates_with_probs['point_idx'], dtype=int)
    ai = np.asarray(candidates_with_probs['arc_id'], dtype=int)
    prob = np.asarray(candidates_with_probs['probability'], dtype=float)
    if not (len(pi) == len(ai) == len(prob)) or np.any(~np.isfinite(prob)) or np.any((prob < 0) | (prob > 1)):
        raise ValueError('Invalid candidate probabilities or row lengths')
    if np.any((pi < 0) | (pi >= n)) or np.any((ai < 0) | (ai >= len(net.arcs))):
        raise ValueError('Candidate index out of range')
    states, emissions, fractions = [], [], []
    order = np.argsort(pi, kind='stable')
    boundaries = np.searchsorted(pi[order], np.arange(n + 1))
    for i in range(n):
        ix = order[boundaries[i]:boundaries[i+1]]
        if len(ix):
            arcs, q = ai[ix], prob[ix]
        else:
            import shapely
            arcs = np.atleast_1d(net.tree.nearest(shapely.points(p[i:i+1])))
            q = np.ones(len(arcs))
        _, frac = seg_dist(np.repeat(p[i:i+1], len(arcs), axis=0), net.A[arcs], net.B[arcs])
        states.append(arcs)
        emissions.append(np.log(np.maximum(q, 1e-300)))
        fractions.append(frac)
    adj = [[] for _ in net.node_xy]
    for (u, v), length in zip(net.arcs, net.length):
        adj[u].append((int(v), float(length)))
    output = np.empty(n, dtype=int)
    times = trip['track'][:, 2]
    breaks = np.flatnonzero((np.diff(times) > cfg.gap_s) | (np.diff(times) <= 0)) + 1
    for start, end in zip(np.r_[0, breaks], np.r_[breaks, n]):
        score, back = emissions[start].copy(), []
        for t in range(start + 1, end):
            old, new = states[t-1], states[t]
            displacement = float(np.linalg.norm(p[t] - p[t-1]))
            limit = cfg.max_speed_mps * (times[t] - times[t-1]) + cfg.search_slack_m
            sources = net.arcs[old, 1]
            targets = net.arcs[new, 0]
            paths = {int(u): _distances(adj, int(u), targets, limit) for u in np.unique(sources)}
            transition = np.full((len(old), len(new)), -cfg.unreachable_penalty)
            for i, a in enumerate(old):
                for j, b in enumerate(new):
                    distance = float('inf')
                    if a == b and fractions[t][j] >= fractions[t-1][i]:
                        distance = (fractions[t][j] - fractions[t-1][i]) * net.length[a]
                    middle = paths[int(sources[i])].get(int(targets[j]), float('inf'))
                    distance = min(distance, (1-fractions[t-1][i])*net.length[a] + middle + fractions[t][j]*net.length[b])
                    if distance <= limit:
                        transition[i, j] = -abs(distance - displacement) / cfg.beta_m
            values = score[:, None] + transition
            parent = np.argmax(values, axis=0)
            score = values[parent, np.arange(len(new))] + emissions[t]
            score -= score.max()
            back.append(parent)
        j = int(np.argmax(score))
        for t in range(end-1, start-1, -1):
            output[t] = states[t][j]
            if t > start:
                j = int(back[t-start-1][j])
    return output.tolist()
