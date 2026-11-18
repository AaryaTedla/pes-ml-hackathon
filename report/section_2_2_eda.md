# 2.2 Dataset Understanding and Exploratory Data Analysis (DRAFT)

> Draft from real outputs in `results/`. Every number is in `results/eda/trip_inventory.csv`,
> `partition_counts.csv` or `docs/features.md`, and the code is `src/eda.py`.
> Rewrite it in your own words and check each number before it goes in the PDF.
> Partition used is named in each paragraph, as the guide asks.

## How the data is organised
The data comes in trips. Each trip has its own folder with four headerless files: `.track` (lon, lat, time in seconds),
`.nodes` (lon, lat), `.arcs` (from-node, to-node) and `.route` (the reference arc sequence). A row in `.track` does not correspond to a row in any
other file. Arcs reference nodes by row index, and the route references arcs by row index. **Each trip has its own road network**, so there is no shared global graph.
We analysed the **train partition (80 trips, 208,417 GPS points)** and the **student_test partition (10 trips, 19,160 points)**. The 10 evaluator
trips were never opened.

## Preprocessing
1. **Projection.** Trips are scattered across the world (train centroids: longitude -100.3 to 137.2, latitude -34.6 to 61.2), so degrees cannot be used as distances.
   Each trip is projected to its own UTM zone (metres), chosen from the trip's mean longitude and latitude.
2. **Validation.** For all 90 trips we checked: invalid or non-finite coordinates (0), coordinates outside the valid range (0), non-increasing timestamps (0),
   route ids outside the arc range (0). We found 6 exactly repeated GPS positions in train and none in student_test.
3. **Candidate generation.** For each GPS point, all road arcs within 60 m (at most the 30 nearest), found with a spatial index (STRtree).
   If none is within 60 m, the nearest arc within 300 m is used. This step never reads the route.
4. **Labels.** The data has no per-point labels, only the route. We align points to route arcs with a forward-only dynamic programme (see `src/labels.py`)
   and label a candidate positive when it is the aligned arc. These are reference labels, not verified ground truth.

## What the exploration showed
(Figures `01` to `06` in `results/figures/`, interactive maps in `results/maps/`. Train unless stated.)

- **Size varies a lot.** GPS points per trip: median 1,529, range 361 to 19,313. Road arcs per trip: median 19,500, range 649 to 307,392 (Fig. 01, 02).
  Student_test is smaller: median 1,420 points, maximum 3,958.
- **Sampling is nominally 1 Hz but has gaps.** 6.2% of time steps in the average train trip are not exactly 1 s (1.8% in the average student_test trip). 1,216 steps
  in train exceed 5 s, and 22 of 80 train trips contain a gap over 60 s. The largest gaps are 52,810 s (trip 78) and 153,064 s (trip 90). Student_test is much cleaner: 44 gaps over 5 s, with a
  largest gap of 210 s. A decoder trained on this data must therefore treat long gaps specially (Fig. 03).
- **GPS outliers.** 39 train trips and 3 student_test trips contain step-to-step speeds above 50 m/s, up to 892 such steps in one trip (trip 32). We kept these points
  because the decoder must cope with noise at evaluation time. Median driving speed is about 13 m/s (train) and 15 m/s (student_test).
- **GPS noise is moderate in most trips.** The median distance from a point to its reference arc is about 4.7 m per trip (Fig. 04).
- **The road networks are not fully connected.** 79 of 80 train trips and all 10 student_test trips have more than one weakly connected component
  (median 20 in train, 11 in student_test). The largest component still holds a median of 98% of the nodes (minimum 69%). There are no isolated nodes (Fig. 05).
  About 95% of arcs have a reverse arc, so each physical road is usually two arcs at almost the same position.
- **The reference route has limitations.** 10 train trips and 1 student_test trip have 1 to 3 places where consecutive route arcs are not connected.
  In trips 54 and 33 the route does not match the GPS track: for trip 54 the GPS follows the same ring road but 86% of points are more than 60 m (median 140 m) from any route arc, even without the ordering constraint.
  So the candidate miss rate is 10.73% in train, but 1.05% without these two trips, and 0.13% in student_test.
  We treat the route as a reference, not as independently verified truth (Fig. 06 and `map_train_00000054.html`).
- **Missing values.** There are no NaN or invalid GPS values. The missing information is in the timing (gaps) and in the reference-route coverage described above.

## Features and why we chose them
The 18 features are listed in `docs/features.md`. Examples from the train data, rows whose reference arc is among the candidates:

| Feature group | Why | Evidence |
|---|---|---|
| Distance to the arc (`dist_m`, `dist_rel`, `cand_rank`) | Correct arcs are close | Median distance is 5.0 m for the correct candidate and 31.7 m for wrong ones |
| Heading vs arc bearing (`cos_head`, `abs_head_diff_deg`), speed, `moving` | About 95% of arcs have a reverse twin at the same distance, so distance cannot tell them apart. Direction can. | Median `cos_head` is 0.99 for correct candidates and -0.08 for wrong ones. The nearest arc alone is correct only 51.6% of the time |
| Neighbouring points' distance (`dist_prev1/2`, `dist_next1/2`, `dist_win_mean`) | One noisy point is unreliable. Its neighbours should also be near the right arc. | Used by the classifier as short-range context |
| Position along arc, arc length, node degrees (`proj_frac`, `arc_len_m`, `from_out_deg`, `to_out_deg`) | Describe where on the arc and whether a junction is near | Context for the transition model |
| `dt_prev_s` | Flags gaps where heading and speed are less reliable | 6.2% of steps are not 1 s |

The nearest-arc baseline (top-1 distance choice) is only about 51% correct, which shows why a classifier plus sequence decoding is needed.

## Still to add before submission
- [ ] Screenshot of a Folium map (open `results/maps/map_train_00000001.html` in a browser).
- [ ] Insert Figures 01 to 06.
- [ ] Re-check every number against `results/eda/trip_inventory.csv`.
- [ ] Rewrite in your own words, and disclose AI assistance as the event policy requires.
