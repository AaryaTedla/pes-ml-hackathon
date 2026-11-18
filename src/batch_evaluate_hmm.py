"""Whole-trip tuning/validation; only train and student_test are accepted."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from pipeline import prepare_trip
from hmm import decode, mock_probabilities, classifier_only
from evaluate import pointwise_accuracy, route_continuity, sequence_exact_rate

TUNE = ['00000090', '00000069', '00000042', '00000003', '00000005']
CHECK = ['00000078', '00000021', '00000081', '00000010']


def measure(trip, feats, truth, cfg):
    start = time.perf_counter()
    pred = decode(trip, feats, cfg)
    elapsed = time.perf_counter() - start
    mask = truth.has_positive.to_numpy(dtype=bool)
    return dict(accuracy=pointwise_accuracy(pred, truth.true_arc.to_numpy(), mask),
                continuity=route_continuity(pred, trip['arcs']),
                sequence_exact_50=sequence_exact_rate(pred, truth.true_arc.to_numpy()),
                decode_s=elapsed, missing_predictions=int((pred < 0).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--partition', choices=['train', 'student_test'], default='train')
    parser.add_argument('--trips', nargs='+')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--mode', choices=['fixed', 'network'], default='fixed')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'artifacts/hmm_config.json').read_text())
    cfg['mode'] = args.mode
    if args.mode == 'network':
        cfg.update(network_cutoff_m=250., max_search_nodes=128, beta_m=30.)
    labels = pd.read_parquet(root/f'results/points_{args.partition}.parquet')
    requested = args.trips or (TUNE + CHECK if args.partition == 'train' else sorted(labels.trip_id.astype(str).str.zfill(8).unique()))
    tune_ids = set(TUNE) if args.partition == 'train' else set()
    trials = [cfg]
    if args.tune:
        if args.partition != 'train' or args.trips:
            parser.error('--tune uses only the predefined whole-train-trip split')
        trials = [{**cfg, 'disconnected_log': p, 'gap_s': gap} for p in [-4., -8., -12.] for gap in [5., 30.]]
        if args.mode == 'network':
            trials = [{**cfg, 'beta_m': beta, 'gap_s': gap} for beta in [10., 30., 60.] for gap in [5., 30.]]
    # Candidate extraction is done once per trip. Validation labels never select parameters.
    cache, prep_times = {}, {}
    for tid in requested:
        start = time.perf_counter()
        trip, feats, _ = prepare_trip(Path(args.data_root)/args.partition/tid)
        feats = mock_probabilities(feats)
        prep_times[tid] = time.perf_counter() - start
        truth = labels[labels.trip_id.astype(str).str.zfill(8) == tid].sort_values('point_idx')
        if len(truth) != len(trip['track']) or not np.array_equal(truth.point_idx, np.arange(len(truth))):
            raise ValueError(f'Reference rows not aligned for {tid}')
        cache[tid] = (trip, feats, truth)
    trial_rows = []
    if args.tune:
        for number, candidate in enumerate(trials):
            scores = []
            for tid in TUNE:
                result = measure(*cache[tid], candidate)
                scores.append(result['accuracy'])
                trial_rows.append(dict(trial=number, trip_id=tid, **candidate, **result))
            print(f'trial {number}: train-trip macro accuracy {np.mean(scores):.6f}', flush=True)
        macro = pd.DataFrame(trial_rows).groupby('trial').accuracy.mean()
        cfg = trials[int(macro.idxmax())]
        # Network is opt-in until its held-out performance and latency are reviewed.
        target = 'hmm_config.json' if args.mode == 'fixed' else 'hmm_network_config.json'
        (root/'artifacts'/target).write_text(json.dumps(cfg, indent=2))
        pd.DataFrame(trial_rows).to_csv(root/f'results/hmm_{args.mode}_tuning.csv', index=False)
    rows = []
    for tid in requested:
        trip, feats, truth = cache[tid]
        mask = truth.has_positive.to_numpy(dtype=bool)
        baseline = classifier_only(trip, feats)
        rows.append(dict(trip_id=tid, partition=args.partition,
                         split='tune' if tid in tune_ids else 'check', points=len(trip['track']),
                         scored_points=int(mask.sum()), prepare_s=prep_times[tid],
                         max_gap_s=float(np.diff(trip['track'][:, 2]).max()),
                         classifier_accuracy=pointwise_accuracy(baseline, truth.true_arc.to_numpy(), mask),
                         classifier_continuity=route_continuity(baseline, trip['arcs']),
                         classifier_sequence_exact_50=sequence_exact_rate(baseline, truth.true_arc.to_numpy()),
                         **measure(trip, feats, truth, cfg)))
    result = pd.DataFrame(rows)
    result.to_csv(root/f'results/hmm_{args.mode}_batch.csv', index=False)
    (root/f'results/hmm_{args.mode}_run.json').write_text(json.dumps(dict(config=cfg, python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, partition=args.partition, tune_trips=TUNE,
        check_trips=CHECK, seed=None, note='Deterministic mock; accuracy uses has_positive; shared continuity counts changed arcs only; preparation timing includes mock probabilities.'), indent=2))
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
