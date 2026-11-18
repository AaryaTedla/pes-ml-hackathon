# PathSnap feature tables (Track A hand-off)

Files (in this folder):
- `features_train.parquet`: 80 trips, 2,118,916 rows (candidate rows)
- `features_student_test.parquet`: 10 trips, 244,176 rows
- `points_train.parquet` / `points_student_test.parquet`: one row per GPS point, with the reference arc (`true_arc`), `label_dist_m`, `has_positive` and lon/lat/t. Used for the HMM, maps and metrics.

One row = one (GPS point, candidate road arc). Always group by `trip_id`. Never split a trip across folds, so use GroupKFold(groups=trip_id).

## Columns, in file order

ID columns: `trip_id` (str), `point_idx` (int, 0-based row in the .track file), `arc_id` (int, row in that trip's .arcs), `label` (0/1)

Feature columns (18), in this exact order (`FEATURE_COLS` in `src/features.py`):
```
dist_m, dist_rel, cand_rank, n_cand,
proj_frac, arc_len_m,
cos_head, abs_head_diff_deg, speed_mps, moving,
dist_prev1, dist_next1, dist_prev2, dist_next2, dist_win_mean,
from_out_deg, to_out_deg, dt_prev_s
```
Extra column, not a feature: `pt_has_pos` (1 if this point's reference arc is among its candidates).
`trip_id`, `point_idx` and `arc_id` must NOT be used as model inputs.

| Feature | Meaning |
|---|---|
| dist_m | metres from the GPS point to the arc segment |
| dist_rel | dist_m minus the nearest candidate's distance for this point |
| cand_rank / n_cand | distance rank of the candidate within the point / candidates for the point |
| proj_frac | where the point projects along the arc (0 = start, 1 = end) |
| arc_len_m | arc length (m) |
| cos_head | cos(GPS course minus arc bearing). +1 = same direction, -1 = opposite. 0 if not moving. |
| abs_head_diff_deg | absolute angle between the GPS course and the arc (90 if not moving) |
| speed_mps, moving | speed from a +-2 point window; moving = displacement > 3 m, speed > 1 m/s, window <= 10 s |
| dist_prev1/2, dist_next1/2 | distance of the points 1 and 2 steps before and after to the same arc |
| dist_win_mean | mean of dist_m and those four |
| from_out_deg, to_out_deg | number of outgoing arcs at the arc's start and end node (junction-ness) |
| dt_prev_s | seconds since the previous GPS point (gap indicator) |

Why the heading features matter: about 93 to 95% of arcs have a reverse twin (per-trip median), so most roads have 2 arcs
at the same distance. The distance alone gives a coin flip (nearest-arc Top-1 is about 51%).
The direction of travel is what separates them.

## How the labels were derived

The raw data has no per-point labels. It only has `.route`, the ordered arc sequence the vehicle traversed.
1. Project everything to the trip's UTM zone (metres).
2. For each GPS point, generate candidates: all arcs within 60 m, at most the 30 nearest. If none is
   within 60 m, take the single nearest arc within 300 m. This step never uses the route.
3. Assign each GPS point its reference arc with a **monotone dynamic programme** over the route
   (`src/labels.py`). State j = "point i is on route arc route[j]". Cost = distance from the point to
   that arc (capped at 200 m). The route index may never go backwards, and skipping arcs costs 2 m-equivalent per arc (reduced after time gaps > 5 s).
   The result is `true_arc` for each point.
4. `label = 1` if the candidate's arc_id equals the point's `true_arc`, else 0.

This is a reference label, not verified physical truth.

## Candidate miss rate and points with no positive

Miss = the point's reference arc is not in its candidate set (so the point has no positive row).

| Partition | Points | No-positive points | Candidate miss rate |
|---|---|---|---|
| train (80 trips) | 208,417 | 22,362 | 10.73% |
| student_test (10 trips) | 19,160 | 25 | 0.13% |

All train points have at least 1 candidate. 59 of 80 train trips have zero misses. 9 of 10 student_test trips have zero misses (the 25 misses are all in trip 00000056).

**The train number is dominated by two trips whose reference route does not match the GPS track (it runs along the same corridor but a median of about 140 m away on trip 54):**
- `00000054`: 17,233 of 19,313 points have no positive. The route is 31 km against a 170 km GPS path, and 86% of points are more than 60 m from *any* route arc even when matching without ordering. This is a reference-route limitation, not a candidate bug.
- `00000033`: 3,209 of 5,844 points. Same pattern.
- `00000002`: 837 points, in two stretches (the last 541 points and 284 points mid-trip) where points are 100 to 280 m from the reference route.

Without trips 54 and 33 the train miss rate is about 1.05%.

## Recommendation for training (Kaggle)

- Train and cross-validate on rows with `pt_has_pos == 1`. Points with no positive teach the model that all nearby arcs are wrong, which is noise from the reference.
- Report the candidate miss rate separately, with and without trips 54 and 33, as a data-quality finding.
- Group-aware CV: `GroupKFold(n_splits=5)` with `groups=trip_id`.
- Class balance is about 8.8% positive (train), so use F1, log loss and per-point Top-1. Plain accuracy is misleading.
- **Per-point Top-1** = for each `(trip_id, point_idx)`, take the row with the highest predicted probability and check whether `label == 1`. Baseline to beat: nearest-arc (`cand_rank == 0`) gets 0.516 on train points that have a positive and 0.509 on student_test.
- Student_test is for model comparison only. Never tune on the evaluator trips.

## Reproduce
```
cd pathsnap/src
python build_features.py --jobs 8     # about 12 s for all 90 trips
```
