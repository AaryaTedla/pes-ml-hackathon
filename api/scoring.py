"""Server-only scoring, invoked strictly after reference-free inference finishes."""
import numpy as np

from data_loader import load_trip
from candidates import RoadNet
from labels import assign_route_arcs
from evaluate import pointwise_accuracy, sequence_exact_rate

COVERED_M = 60.0


def _finite(value):
    return float(value) if np.isfinite(value) else None


def continuity(pred, arcs):
    # Do not connect the arcs on either side of an unmatched (-1) observation.
    a, b = pred[:-1], pred[1:]
    changes = a != b
    if not changes.any():
        return 1.0 if len(pred) and (pred >= 0).all() else None
    valid = (a >= 0) & (b >= 0) & changes
    connected = np.zeros(len(a), dtype=bool)
    connected[valid] = arcs[a[valid], 1] == arcs[b[valid], 0]
    return float(connected[changes].mean())


def score(trip_dir, pred, clf_pred, arcs):
    ref = load_trip(str(trip_dir), with_route=True)
    if ref["route"] is None:
        return None
    if (ref["route"] < 0).any() or (ref["route"] >= len(arcs)).any():
        raise ValueError("Reference route contains invalid arc ids")
    net = RoadNet(ref)
    P = net.project_points(ref["track"][:, 0], ref["track"][:, 1])
    true_arc, distance = assign_route_arcs(P, ref["track"][:, 2], ref["route"], net)
    covered = distance <= COVERED_M
    return {
        "pointwise_accuracy": _finite(pointwise_accuracy(pred, true_arc)),
        "pointwise_accuracy_classifier_only": _finite(pointwise_accuracy(clf_pred, true_arc)),
        "pointwise_accuracy_reference_covered": _finite(pointwise_accuracy(pred, true_arc, covered)),
        "frac_points_reference_covered": float(covered.mean()),
        "route_continuity": continuity(pred, arcs),
        "sequence_window50_accuracy": _finite(sequence_exact_rate(pred, true_arc, 50)),
        "n_points_scored": int(len(pred)),
    }
