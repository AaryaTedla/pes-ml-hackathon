"""Fast candidate-arc Viterbi. Ground-truth routes are never read here."""
import numpy as np

DEFAULT_CONFIG = dict(distance_scale_m=10., heading_weight=3., gap_s=30.,
                      stay_log=0., connected_log=-0.2, disconnected_log=-8.,
                      mode='fixed')


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
    initial API implementation. Probabilities must be finite and in [0,1].
    """
    config = {**DEFAULT_CONFIG, **cfg}
    if config['gap_s'] <= 0:
        raise ValueError('gap_s must be positive')
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
