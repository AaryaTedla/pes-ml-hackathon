# Track D handoff agreement

Owner: PES1UG24CS548. Confirm these interfaces with the three owners before switching away from `nearest`.

## 497 / data and features

- [ ] Confirm `src/pipeline.py:prepare_trip(trip_dir)` and existing loader/candidate code.
- [ ] Confirm feature order below, candidate radius/fallback policy, missing-point handling, and package versions.
- [ ] Keep `.route` access out of prediction. The API uses this entry point exclusively.

Feature order from the checked-out `src/features.py:FEATURE_COLS`:

```json
["dist_m", "dist_rel", "cand_rank", "n_cand", "proj_frac", "arc_len_m", "cos_head", "abs_head_diff_deg", "speed_mps", "moving", "dist_prev1", "dist_next1", "dist_prev2", "dist_next2", "dist_win_mean", "from_out_deg", "to_out_deg", "dt_prev_s"]
```

IDs: `trip_id`, `point_idx`, `arc_id`. These are not model input features. Labels and label-derived columns are not model input features.

## 496 / trained classifier

- [ ] Deliver `artifacts/model.joblib`, either the estimator or a full preprocessing Pipeline.
- [ ] State whether a separate `artifacts/scaler.joblib` is needed. Do not provide it if the Pipeline already scales.
- [ ] Deliver the exact ordered JSON array above as `artifacts/feature_columns.json`, or document and resolve any difference with 497.
- [ ] Deliver Python, sklearn and any model-library versions; pin the serving environment to the export environment.
- [ ] Confirm `classes_` contains positive class `1` and `predict_proba(X)` returns one row per candidate.
- [ ] Confirm the final artifact comes from the approved train-only preprocessing/trip-grouped selection process.

Track D checks feature metadata where present, selects the positive class using `classes_`, validates probability bounds, then takes the maximum probability per point for classifier-only output. A point with no candidates remains `-1`.

## 549 / HMM decoder

- [ ] Deliver `src/hmm.py` and `artifacts/hmm_config.json`.
- [ ] Agree on `decode(trip, candidates_with_probs)` or `decode(trip, candidates_with_probs, config)`; both forms are supported by the adapter.
- [ ] Candidate table adds `prob = P(candidate correct)` in the same row order.
- [ ] `trip['route']` is `None`; decoder must use only track, nodes, arcs and candidates.
- [ ] Return a one-dimensional integer array/list of length `len(trip['track'])`, with IDs in `[0, len(arcs))`, or `-1` for unmatched points.
- [ ] Document gap handling, disconnected transitions, no-candidate points and required packages.

The decoder may return road-graph connector arcs beyond the original candidate set. All IDs are range-checked before use. Any required mapping from decoder states to local arc row indices belongs in the adapter, with a matching contract test.

## Integration acceptance

- [ ] Start `PATHSNAP_MODE=classifier`; verify all 10 student-test trips.
- [ ] Start `PATHSNAP_MODE=hmm`; verify all 10 again and save classifier-only/HMM comparisons from real experiments.
- [ ] Confirm predictions are unchanged by removal/replacement of the reference route.
- [ ] Confirm no ground truth reaches the browser or exported API result.
- [ ] Run `api/test_lan.ps1` on laptop 2 and inspect the live log on laptop 1.
- [ ] Capture UI, API-terminal and offline Folium screenshots for the report.

No real classifier/HMM integration is claimed until these artifacts and runs exist.
