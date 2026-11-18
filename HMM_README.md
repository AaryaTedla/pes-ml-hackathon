# SRN 549: HMM prototype

The local system Python lacks its standard library (`encodings`); a workspace-local
portable interpreter is available at `.runtime/python/python.exe` (gitignored).
Models and Training Guidelines.pdf was not found, so sections 8–9 remain unchecked.

With a working Python 3.10+ installation, from this repository root:

```powershell
python -m pip install numpy pandas shapely pyproj pyarrow
python src/demo_hmm.py --trip "../Student folder/train/00000001"
```

Outputs: `results/hmm_demo.json`, `results/hmm_predictions.csv`, and
`artifacts/hmm_config.json`. The checked-in configuration contains untuned defaults.
The demo reads Track A's `results/points_train.parquet` only after inference.
No hidden evaluator data is used. There is no stochastic training in this demo.

## Model definition

- State: a directed candidate arc at each GPS observation.
- Demo mock emission: softmax(-dist_m / sigma_m + heading_weight * cos_head),
  normalised per point. Heading distinguishes reverse arcs at identical distances.
  `mock_candidates` also retains the earlier distance-only Gaussian baseline helper.
- Transition log potential: -abs(network_distance - GPS_displacement) / beta.
  Network distance includes the remainder of the previous arc, a directed shortest
  path between arc endpoints, and the fraction of the next arc travelled. Remaining
  on the same arc allows forward progress directly. Bounded Dijkstra limits cost.
- Unreachable or over-budget transitions receive a finite negative penalty so the
  decoder can recover from bad candidates; this does not guarantee a connected route.
- Gaps above 30 seconds or nonpositive timestamp differences start independent
  sequences. Empty candidate sets fall back to the nearest network arc.
- Viterbi adds log emissions and transition potentials, stores predecessor indices,
  rescales each score vector by its maximum, and backtracks within each segment.
  Transition potentials are unnormalised compatibility scores, not calibrated probabilities.

## Integration

`decode(trip, rows, config=None, net=None)` returns one arc id per point.
`rows` is a dict or DataFrame containing `point_idx`, `arc_id`, `probability`.
Track B should supply its positive-class probabilities in the last column.
Pass only track, nodes and arcs to inference; do not supply route-derived features.
Imports follow the existing repository convention: run scripts from `src/` by path.

## Evaluation limits

The demo calls `prepare_trip`, then compares mock classifier-only and HMM on identical
candidates. It reports point accuracy on `has_positive` points and all points against
Track A's derived labels, unique-arc Jaccard, exact sequence match,
and same-or-directly-adjacent continuity excluding gap boundaries. Direct adjacency
does not measure full network reachability: skipped intermediate arcs can lower it.
Candidate generation plus baseline timing and decoder timing are reported separately;
road-index construction and label loading are excluded. One training trip is a smoke
test, not evidence of generalisation. Parameters must be validated across whole trips.
The prototype repeatedly scans candidate rows per observation and runs bounded graph
searches; profile before scaling to all trips. It outputs point assignments, not an
expanded route containing every intermediate connecting arc.

## Verified run (2026-11-18)

Training trip 00000001: 1,315 points, all `has_positive`. Mock classifier accuracy
89.58%, HMM accuracy 92.93%. Direct continuity: 91.55% versus 95.97%. HMM decode
time: 1.33 s. Exact sequences do not match the reference. These are untuned smoke
results; see `results/hmm_demo.json` for full precision and settings.
`python -m unittest discover -s tests -v` passes all three HMM tests. The pre-existing
leakage script has a `main()` entry point and is not exercised by unittest discovery.

The demo's candidates come from `prepare_trip` (60 m, up to 30 candidates); config
`max_candidates` applies only to the separate distance-only `mock_candidates` helper.
Use `--beta-m` and `--output` to record further whole-training-trip experiments.
No decoder parameter tuning or evaluator-trip access has been performed.
AI assistance was used to author this prototype; include the event-required disclosure.
