# 5. HMM and Viterbi Route Reconstruction

> Draft built from the committed results in `results/`. The team must review it, rewrite it in its own words and be able to
> explain every statement. All numbers below come from `results/hmm/final/hmm_decoder_comparison_model.csv` and its per-trip file unless noted.

## 5.1 States and emission scores

A GPS point can be close to several roads, so choosing a road independently for each point can make the route jump between them.
We tested whether using the neighbouring points' choices resolves this.

At each GPS point the hidden states are its candidate directed road arcs. We keep `point_idx` and `arc_id` with every candidate, so
the decoder returns one arc id per GPS point (or -1 when a point has no candidate). Each trip uses its own road graph. The reference
route is used only for scoring after prediction and never as decoder input.

The emission score of a candidate is the log of the classifier's probability that it is the correct arc. The classifier is the tuned
Random Forest (`artifacts/model.joblib`, trained on 40 training trips; see section 4). Before the trained model was available we
used a heuristic stand-in, softmax(-dist_m/10 + 3*cos_head) per GPS point. Those earlier results are kept in
`results/hmm/stand_in/hmm_decoder_comparison_mock.csv` as a baseline and are not the final results.

## 5.2 Two transition designs

Both decoders are compared on the same trips, with the same probabilities.

**Fixed-transition decoder** (`src/hmm_fixed.py`, config `artifacts/hmm_fixed_config.json`). A log score of 0 for staying on the same
arc, -1.0 for moving to a directly adjacent arc (the previous arc ends where the next starts), and -8.0 for any other movement. It is cheap,
but moving across several short road segments between two samples is scored like an unrelated jump. The sequence restarts after a time
gap above 30 s, a missing-candidate point, or a nonpositive or non-finite time difference.

**Multi-hop decoder** (`src/hmm.py`, config `artifacts/hmm_config.json`, used by the API). It allows travel through intermediate arcs. A
memoised, depth-limited breadth-first search on directed arc connections finds how many arcs apart two candidates are:

```
depth = max_hops + floor(min(dt, gap_s) / gap_hop_s)
```

Staying on an arc costs 0; reaching another arc in h hops costs `hop_penalty * h`; an immediate move to the reverse arc costs
`uturn_penalty`; if the arc is not found within the search depth the decoder applies `jump_penalty`. The tuned values are hop_penalty=2,
jump_penalty=20, uturn_penalty=16, max_hops=5, gap_s=120 s, gap_hop_s=5 s. After a gap above 120 s, transition costs are set to zero so the new
point is independent of the previous choice. Points with no candidate stay -1 and decoding continues across them using the elapsed time.
For nonpositive time differences it substitutes 1 s; it has no explicit check for non-finite timestamps.

These costs are dimensionless scores. A hop is an arc transition, not a distance in metres, and neither active decoder computes metric
shortest paths. Both use finite penalties, so decoding can recover from poor candidate sets, but a disconnected result is still possible.
Neither decoder inserts connector arcs between predictions.

## 5.3 Viterbi recurrence

```
V_t(j) = emission_t(j) + max_i [ V_(t-1)(i) + transition_t(i, j) ]
```

For each candidate j we keep the best score of a sequence ending there and the previous candidate that gave it. After the forward pass we
backtrack from the best final candidate. This differs from picking the largest emission at every point: a slightly weaker local choice can
belong to a better complete sequence. Working with log scores avoids multiplying many small probabilities.

## 5.4 Experiment and metrics

**Trips and tuning.** 54 whole trips: 22 training trips labelled TUNE, 22 labelled VAL and the 10 student_test trips (seed 0).
Training trips with a candidate miss rate above 5%, or with more than 8,000 points, are excluded, so the comparison does not cover every
training trip or the longest trips. **Both decoders were tuned with the same coordinate search on the TUNE trips only, using the
trained model's probabilities.** The search logs are `results/hmm/final/hmm_tuning_multihop_model.csv` and `results/hmm/final/hmm_tuning_fixed_model.csv`.
VAL and student_test were not used to choose any setting. Note that VAL is held out from the decoder tuning but not completely from the whole system: 11 of its 22 trips (and 12 of the 22 TUNE trips) were also among the 40 trips used to train the classifier. The 10 student_test trips do not overlap with the classifier's training trips, so they are the cleaner held-out check for the full pipeline. Evaluator trips were never used. (An earlier comparison on stand-in probabilities tuned the two
decoders on different trips; that limitation no longer applies.)

**Metrics.** Point accuracy counts correct arc ids on points whose derived reference arc is within 60 m of the GPS point. Reference
labels come from aligning points to the route and are not independently verified physical truth. Direct continuity (from `src/evaluate.py`)
is the fraction of changes in the predicted arc where the first arc ends at the next arc's start node; it does not credit a connection through
intermediate arcs, so a valid multi-hop movement lowers it. The sequence metric is the fraction of non-overlapping 50-point windows in which
every prediction matches the reference label; it uses all points, without the 60 m mask. Averages give each trip equal weight.

### TUNE (22 training trips)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity | Trips better / worse than classifier-only |
|---|---:|---:|---:|---:|
| Classifier only (tuned Random Forest) | 85.67% | 22.40% | 72.57% | n/a |
| Fixed-transition HMM | 88.92% | 28.16% | 97.85% | 19 / 2 |
| Multi-hop HMM | 91.99% | 42.66% | 88.89% | 22 / 0 |

### VAL (22 training trips not used for tuning)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity | Trips better / worse than classifier-only |
|---|---:|---:|---:|---:|
| Classifier only | 87.80% | 24.82% | 75.56% | n/a |
| Fixed-transition HMM | 91.42% | 31.42% | 98.66% | 20 / 0 |
| Multi-hop HMM | 94.10% | 44.20% | 92.18% | 22 / 0 |

### STUDENT_TEST (10 trips)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity | Trips better / worse than classifier-only |
|---|---:|---:|---:|---:|
| Classifier only | 86.52% | 26.91% | 74.61% | n/a |
| Fixed-transition HMM | 88.79% | 30.75% | 98.53% | 8 / 1 |
| Multi-hop HMM | 91.78% | 39.53% | 90.62% | 9 / 1 |

"Worse" means more than 0.5 percentage points below classifier-only.

**Live API check** (`results/hmm/final/api_hmm_check_model.csv`): the API in `hmm` mode on all 10 student_test trips gives mean point accuracy 0.865 for the
classifier alone and 0.917 after the multi-hop HMM, with mean continuity 0.906.

**Timing.** Decoding all 22 VAL trips took 16.8 s with the multi-hop decoder and 4.7 s with the fixed decoder in the committed run, excluding feature
preparation and API overhead. Times depend on the machine and on what else is running (an earlier run of the same code was faster), so we
report them only as a rough comparison: multi-hop was about 2 to 4 times slower than fixed transitions, and both are fast enough for the live demo.

## 5.5 What helped and what did not

- **The HMM helps consistently.** The multi-hop decoder improved accuracy on all 22 TUNE and all 22 VAL trips, and on 9 of 10 student_test trips.
  The gain over classifier-only is about 6 points on TUNE and VAL and about 5 points on student_test. It also raises the exact 50-point windows
  from about 22 to 27% to about 40 to 44%.
- **Multi-hop is more accurate; fixed transitions are more continuous.** Multi-hop is higher on accuracy and sequence score in every split, while the fixed
  decoder is higher on direct continuity (about 98% vs 89 to 92%). The continuity metric penalises skipping over a road segment even when the graph
  has a valid connecting path, so continuity alone is not enough to choose a decoder. We use the multi-hop variant in the API because of its measured
  accuracy advantage.
- **Largest improvement, train trip 00000004 (VAL):** accuracy 65.13% (classifier) to 85.62% (fixed) to 91.07% (multi-hop). The fixed decoder
  corrected 239 classifier errors and introduced 21. See section 6 for the maps.
- **Where fixed transitions did not help, train trip 00000078 (TUNE):** classifier 92.16%, fixed 90.75% (it corrected 123 errors but introduced
  164, a net loss of 41 correct points) although continuity rose from 76.7% to 95.3%. The longest time gap in this trip is 52,810 s. The multi-hop decoder
  reached 96.80% on the same trip. This trip is in the tuning set, so it illustrates a failure mode and is not held-out evidence. We have not established
  the physical cause of each error.
- **Where the multi-hop HMM did not help, student_test trip 00000007:** accuracy fell from 86.7% to 85.4%. Higher smoothing can keep a plausible but wrong road
  when the classifier was already right.

## 5.6 Limitations

- Only 54 trips are compared, and the evaluator trips (hidden) may behave differently.
- The reference labels are derived and imperfect, and 10 training trips have route breaks, so point accuracy has a ceiling below 100%.
- Both decoders use the classifier's probabilities as they are. We did not calibrate them, and a different classifier could change which decoder settings are best.
- Direct continuity is a strict metric; it undervalues multi-hop movements.
- AI assistance was used for implementation, verification and drafting of this section. The team must review it before submission.

Reproduce: `python src/compare_decoders.py --prob model` (decoder comparison) and `python src/eval_hmm.py --mode tune --prob model [--decoder fixed] --save-config` (tuning).
