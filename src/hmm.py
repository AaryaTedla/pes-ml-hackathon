"""Fast candidate-arc Viterbi. Ground-truth routes are never read here."""
import numpy as np
import heapq
from functools import lru_cache

DEFAULT_CONFIG = dict(distance_scale_m=10., heading_weight=3., gap_s=30.,
                      stay_log=0., connected_log=-0.2, disconnected_log=-8.,
                      mode='fixed')


def _network_transitions(trip, feats, config):
    """Build bounded, cached directed-distance potentials; geometry uses metres."""
    from candidates import RoadNet
    from geo import seg_dist
    net = RoadNet(trip)
    xy = net.project_points(*trip['track'][:, :2].T)
    pi, ai = feats.point_idx.to_numpy(dtype=int), feats.arc_id.to_numpy(dtype=int)
    _, fractions = seg_dist(xy[pi], net.A[ai], net.B[ai])
    adj = [[] for _ in net.node_xy]
    for (u, v), length in zip(net.arcs, net.length):
        adj[u].append((int(v), float(length)))
    cutoff = config.get('network_cutoff_m', 250.)
    max_nodes = int(config.get('max_search_nodes', 128))
    beta = config.get('beta_m', 30.)
    if cutoff <= 0 or max_nodes < 1 or beta <= 0:
        raise ValueError('Network cutoff, node budget, and beta must be positive')

    @lru_cache(maxsize=2048)
    def distances(source):
        best, found, queue = {source: 0.}, {}, [(0., source)]
        while queue and len(found) < max_nodes:
            dist, u = heapq.heappop(queue)
            if dist != best[u]:
                continue
            found[u] = dist
            for v, length in adj[u]:
                nd = dist + length
                if nd <= cutoff and nd < best.get(v, float('inf')):
                    best[v] = nd
                    heapq.heappush(queue, (nd, v))
        return found

    def transition(old, new, old_rows, new_rows, t):
        matrix = np.full((len(old), len(new)), config['disconnected_log'], dtype=float)
        displacement = np.linalg.norm(xy[t] - xy[t-1])
        for i, a in enumerate(old):
            paths = distances(int(net.arcs[a, 1]))
            for j, b in enumerate(new):
                distance = ((1-fractions[old_rows[i]]) * net.length[a]
                            + paths.get(int(net.arcs[b, 0]), float('inf'))
                            + fractions[new_rows[j]] * net.length[b])
                delta = fractions[new_rows[j]] - fractions[old_rows[i]]
                if a == b and delta >= 0:
                    distance = min(distance, delta * net.length[a])
                if distance <= cutoff:
                    matrix[i, j] = -abs(distance-displacement) / beta
        return matrix
    return transition


def mock_probabilities(feats):
    """Stand-in softmax(-dist_m/10 + 3*cos_head), separately for each GPS point."""
    result = feats.copy()
    logits = -result.dist_m / 10. + 3. * result.cos_head
    weights = np.exp(logits - logits.groupby(result.point_idx).transform('max'))
    result['prob'] = weights / weights.groupby(result.point_idx).transform('sum')
    return result


def classifier_only(trip, feats):
    output = np.full(len(trip['track']), -1, dtype=np.int64)
    if len(feats):
        rows = feats.loc[feats.groupby('point_idx')['prob'].idxmax()]
        output[rows.point_idx.to_numpy(dtype=int)] = rows.arc_id.to_numpy(dtype=int)
    return output


def decode(trip, feats, cfg):
    """decode(trip, feats, cfg) -> int64 ndarray, one arc per point (-1 if none).

    Required feature columns: point_idx, arc_id, prob. cfg is a JSON-compatible
    dict. Fixed transitions favour staying/adjacency. Missing candidates and long
    or nonpositive time gaps break sequences. No shortest-path searches in this
    fixed mode. Optional network mode caps Dijkstra distance and settled nodes,
    then penalises network-distance/GPS-distance disagreement. Probabilities must
    be finite and in [0,1].
    """
    config = {**DEFAULT_CONFIG, **cfg}
    if config['gap_s'] <= 0:
        raise ValueError('gap_s must be positive')
    if config['mode'] not in ('fixed', 'network'):
        raise ValueError('mode must be fixed or network')
    n = len(trip['track'])
    output = np.full(n, -1, dtype=np.int64)
    if not n or not len(feats):
        return output
    pi = feats['point_idx'].to_numpy(dtype=np.int64)
    ai = feats['arc_id'].to_numpy(dtype=np.int64)
    prob = feats['prob'].to_numpy(dtype=float)
    arcs = np.asarray(trip['arcs'])
    if np.any((pi < 0) | (pi >= n)) or np.any((ai < 0) | (ai >= len(arcs))):
        raise ValueError('Candidate indices out of range')
    if np.any(~np.isfinite(prob)) or np.any((prob < 0) | (prob > 1)):
        raise ValueError('prob must contain finite probabilities in [0,1]')
    network = _network_transitions(trip, feats, config) if config['mode'] == 'network' else None
    order = np.argsort(pi, kind='stable')
    boundaries = np.searchsorted(pi[order], np.arange(n + 1))
    states = [ai[order[boundaries[i]:boundaries[i+1]]] for i in range(n)]
    emissions = [np.log(np.maximum(prob[order[boundaries[i]:boundaries[i+1]]], 1e-300)) for i in range(n)]
    times = trip['track'][:, 2]
    start = 0
    while start < n:
        if not len(states[start]):
            start += 1
            continue
        score = emissions[start].copy()
        back = []
        end = start + 1
        while end < n and len(states[end]):
            dt = times[end] - times[end-1]
            if not np.isfinite(dt) or dt <= 0 or dt > config['gap_s']:
                break
            old, new = states[end-1], states[end]
            transition = np.full((len(old), len(new)), config['disconnected_log'], dtype=float)
            connected = arcs[old, 1, None] == arcs[new, 0][None, :]
            transition[connected] = config['connected_log']
            transition[old[:, None] == new[None, :]] = config['stay_log']
            if network is not None:
                transition = network(old, new, order[boundaries[end-1]:boundaries[end]],
                                     order[boundaries[end]:boundaries[end+1]], end)
            values = score[:, None] + transition
            parent = np.argmax(values, axis=0)
            score = values[parent, np.arange(len(new))] + emissions[end]
            score -= score.max()
            back.append(parent)
            end += 1
        j = int(np.argmax(score))
        for t in range(end-1, start-1, -1):
            output[t] = states[t][j]
            if t > start:
                j = int(back[t-start-1][j])
        start = end
    return output
