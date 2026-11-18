"""Route-level metrics shared by the API, the HMM comparison and the report."""
import numpy as np


def pointwise_accuracy(pred, true, mask=None):
    """Fraction of GPS points whose predicted arc equals the reference arc."""
    pred, true = np.asarray(pred), np.asarray(true)
    if mask is not None:
        pred, true = pred[mask], true[mask]
    return float((pred == true).mean()) if len(pred) else float("nan")


def route_continuity(pred, arcs):
    """Fraction of consecutive predicted arcs (that differ) which are connected in the road
    network, i.e. arc a ends at the node where arc b starts. Staying on the same arc counts as connected."""
    pred = np.asarray(pred)
    a, b = pred[:-1], pred[1:]
    diff = a != b
    if not diff.any():
        return 1.0
    ok = arcs[a[diff], 1] == arcs[b[diff], 0]
    return float(ok.mean())


def sequence_exact_rate(pred, true, window=50):
    """Sequence-level view: fraction of non-overlapping windows of `window` points predicted fully correctly."""
    pred, true = np.asarray(pred), np.asarray(true)
    n = len(pred) // window
    if n == 0:
        return float("nan")
    ok = (pred[: n * window] == true[: n * window]).reshape(n, window).all(1)
    return float(ok.mean())


def ordered_unique(arcs):
    """Predicted arc per point -> route as an ordered list without consecutive repeats."""
    arcs = np.asarray(arcs)
    keep = np.r_[True, arcs[1:] != arcs[:-1]]
    return arcs[keep]
