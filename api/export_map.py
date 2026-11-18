"""Offline Folium exports. Reference layers are opt-in and never served by the API."""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import Settings
from .main import trip_directory
from .predictor import Predictor, validate_arcs
from data_loader import load_trip
from viz import make_trip_map


def export_map(trip, out_html, pred_arcs, *, include_reference=False):
    safe_trip = dict(trip)
    if not include_reference:
        safe_trip["route"] = None
    pred = validate_arcs(pred_arcs, len(trip["track"]), len(trip["arcs"]))
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    matched = pred[pred >= 0]
    label = 'offline reference comparison' if include_reference else 'prediction'
    if not len(matched):
        label += ' | no matched arcs'
    return make_trip_map(safe_trip, str(out_html), pred_arcs=matched if len(matched) else None,
                         title=f"Trip {trip['trip_id']} | {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_id")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prediction", type=Path, help="Saved /predict JSON or a JSON array of arc ids")
    parser.add_argument("--include-reference", action="store_true", help="Offline report use only")
    args = parser.parse_args()
    settings = Settings.from_env()
    directory = trip_directory(settings.trips_dir, args.trip_id)
    if args.prediction:
        data = json.loads(args.prediction.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("trip_id") != args.trip_id:
            parser.error("Prediction trip_id does not match the requested trip")
        arcs = data["predicted_arcs"] if isinstance(data, dict) else data
        trip = load_trip(str(directory), with_route=False)
    else:
        result = Predictor(settings).predict_trip(directory)
        trip, arcs = result.trip, result.arcs
    if args.include_reference:
        trip = load_trip(str(directory), with_route=True)
    export_map(trip, args.out, np.asarray(arcs), include_reference=args.include_reference)
    print(f"Map saved: {args.out.resolve()} | reference={args.include_reference}")


if __name__ == "__main__":
    main()
