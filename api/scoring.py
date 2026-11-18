"""SERVER-SIDE scoring against the reference route. This module is only called AFTER a prediction
has been made. Nothing it reads (the .route file, true arcs) is ever passed to the predictor or
returned to the UI; only aggregate metrics leave the server."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import load_trip
from candidates import RoadNet
from labels import assign_route_arcs
from evaluate import pointwise_accuracy, route_continuity, sequence_exact_rate

COVERED_M = 60.0   # a point counts as "reference-covered" if its reference arc is within this distance


def score(trip_dir, pred, clf_pred, arcs):
    ref = load_trip(trip_dir, with_route=True)
    if ref["route"] is None:
        return None                                  # no reference available -> nothing to score
    net = RoadNet(ref)
    P = net.project_points(ref["track"][:, 0], ref["track"][:, 1])
    true_arc, d = assign_route_arcs(P, ref["track"][:, 2], ref["route"], net)
    covered = d <= COVERED_M
    valid = pred >= 0
    return {
        "pointwise_accuracy": pointwise_accuracy(pred, true_arc),
        "pointwise_accuracy_classifier_only": pointwise_accuracy(clf_pred, true_arc),
        "pointwise_accuracy_reference_covered": pointwise_accuracy(pred, true_arc, covered),
        "frac_points_reference_covered": float(covered.mean()),
        "route_continuity": route_continuity(pred[valid], arcs),
        "route_continuity_reference": route_continuity(true_arc, arcs),
        "sequence_window50_accuracy": sequence_exact_rate(pred, true_arc, 50),
        "n_points_scored": int(len(pred)),
    }
