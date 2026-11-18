"""Run from the repository root: python src/demo_hmm.py --trip <trip directory>."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from pipeline import prepare_trip
from candidates import RoadNet
from hmm import HMMConfig, mock_probabilities, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trip', required=True)
    parser.add_argument('--beta-m', type=float, default=30.)
    parser.add_argument('--output', default='results/hmm_demo.json')
    args = parser.parse_args()
    trip, feats, points = prepare_trip(args.trip)
    cfg = HMMConfig(beta_m=args.beta_m)
    net = RoadNet(trip)
    begin = time.perf_counter()
    rows = mock_probabilities(feats, cfg)
    baseline = np.full(len(trip['track']), -1, dtype=int)
    winners = rows.loc[rows.groupby('point_idx').probability.idxmax()]
    baseline[winners.point_idx.to_numpy(dtype=int)] = winners.arc_id.to_numpy(dtype=int)
    missing = np.flatnonzero(baseline < 0)
    if len(missing):
        import shapely
        xy = net.project_points(*trip['track'][missing, :2].T)
        baseline[missing] = net.tree.nearest(shapely.points(xy))
    baseline_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    predicted = np.array(decode(trip, rows, cfg, net))
    decode_seconds = time.perf_counter() - begin
    # Reference data is read only after predictions are fixed.
    labels = pd.read_parquet('results/points_train.parquet')
    labels = labels[labels.trip_id.astype(str).str.zfill(8) == trip['trip_id']].sort_values('point_idx')
    if len(labels) != len(predicted) or not np.array_equal(labels.point_idx, np.arange(len(predicted))):
        raise ValueError('Training reference rows do not align with trip')
    truth = labels.true_arc.to_numpy()
    positive = labels.has_positive.to_numpy(dtype=bool)
    valid_steps = (np.diff(trip['track'][:, 2]) > 0) & (np.diff(trip['track'][:, 2]) <= cfg.gap_s)

    def metrics(arcs):
        connected = (arcs[:-1] == arcs[1:]) | (net.arcs[arcs[:-1], 1] == net.arcs[arcs[1:], 0])
        chosen = set(arcs.tolist())
        reference = set(truth.tolist())
        return dict(point_accuracy_on_has_positive=float(np.mean(arcs[positive] == truth[positive])) if positive.any() else None,
                    point_accuracy_all=float(np.mean(arcs == truth)),
                    adjacent_or_same_arc_fraction=float(np.mean(connected[valid_steps])) if valid_steps.any() else None,
                    unique_arc_jaccard=len(chosen & reference) / len(chosen | reference),
                    exact_sequence_match=bool(np.array_equal(arcs, truth)))
    result = dict(trip_id=trip['trip_id'], points=len(predicted), config=asdict(cfg),
                  classifier_only=metrics(baseline), hmm=metrics(predicted),
                  evaluated_positive_points=int(positive.sum()),
                  candidate_and_baseline_seconds=baseline_seconds, decode_seconds=decode_seconds,
                  excluded_gap_steps=int((~valid_steps).sum()),
                  note='One train-trip smoke run; parameters untuned; reference labels are derived, not physical truth. Continuity measures direct adjacency, not full network reachability.')
    Path('results').mkdir(exist_ok=True)
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/hmm_config.json').write_text(json.dumps(asdict(cfg), indent=2))
    Path(args.output).write_text(json.dumps(result, indent=2))
    pd.DataFrame(dict(point_idx=np.arange(len(predicted)), classifier_arc=baseline, hmm_arc=predicted)).to_csv('results/hmm_predictions.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
