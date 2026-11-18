"""Inference entry point. Prepares one trip for prediction WITHOUT reading its route file,
so ground truth can stay server-side. Used by the API (and anything else that predicts).

    trip, feats, points = prepare_trip(trip_dir)
    feats  : one row per (GPS point, candidate arc); columns ID_COLS[:3] + FEATURE_COLS (no label)
    points : one row per GPS point (lon, lat, t, n_cand)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_trip
from features import trip_features, FEATURE_COLS


def prepare_trip(trip_dir):
    trip = load_trip(trip_dir, with_route=False)   # .route is never opened here
    feats, points = trip_features(trip)
    return trip, feats[["trip_id", "point_idx", "arc_id"] + FEATURE_COLS], points
