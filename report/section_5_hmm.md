# 5. HMM and Viterbi Route Reconstruction

## 5.1 States and emission scores

A GPS point can be close to several roads. Choosing a road separately for every
point can make the predicted route jump between them. We tested whether using the
previous and next predictions helps resolve this ambiguity.

At each GPS point, the possible hidden states are its candidate directed road arcs.
We keep `point_idx` and `arc_id` with each candidate. The decoder returns one arc ID
per point, or -1 when there are no candidates. Each trip uses its own road graph.
The reference route is used for scoring after prediction, never as decoder input.

Both variants currently use the same mock probabilities:

```
score(a) = -dist_m(a)/10 + 3*cos_head(a)
prob(a) = softmax(score(a))       # separately for each GPS point
emission(a) = log(max(prob(a), epsilon))
```

Distance favours nearby roads. Heading helps distinguish the two travel directions
on the same road. Before softmax, we subtract the largest score for that point to
avoid large exponentials. The fixed decoder uses a probability floor of 1e-300;
the multi-hop decoder uses 1e-6. These scores are a stand-in until the trained
classifier arrives. They should not be reported as trained-model probabilities.

## 5.2 Two transition designs

The fixed decoder, `src/hmm_fixed.py`, gives a log score of 0 for staying on an arc,
-0.2 for moving to a directly adjacent arc, and -12 for other movements. Adjacent
means the first arc ends at the node where the next arc starts. It is cheap to
compute, but crossing several road segments between GPS samples receives the same
penalty as an unrelated jump. Its settings are in `artifacts/hmm_fixed_config.json`.

The multi-hop decoder, `src/hmm.py`, also allows travel through intermediate arcs.
It uses breadth-first search on directed arc connections and caches the results.
The search is limited by a hop depth, which increases for longer sampling intervals:

```
depth = max_hops + floor(min(dt, gap_s)/gap_hop_s)
```

Staying on an arc costs zero. Reaching another arc in h hops costs
`hop_penalty*h`. An immediate move to the reverse arc has its own U-turn cost.
If the search does not find the next arc, the decoder applies a jump cost.
The current mock configuration uses hop_penalty=4, jump_penalty=40,
uturn_penalty=16, max_hops=5, gap_s=120 seconds and gap_hop_s=5 seconds.
It is saved in `artifacts/hmm_config.json` and is the variant used by the API.

These costs are dimensionless scores. A hop is an arc transition, not a distance
in metres. The active multi-hop decoder does not calculate metric shortest paths.
The optional Dijkstra mode in `hmm_fixed.py` is a separate experiment and is not
included in this two-variant comparison.

Both designs use finite penalties for unlikely movements. This lets decoding
continue when the candidate set is poor, but it also allows a disconnected result.
The output assigns arcs to GPS points; neither variant inserts connector arcs.
The scores act as compatibility weights rather than calibrated transition
probabilities.

## 5.3 Viterbi and gaps

We find the best sequence with the log-space Viterbi recurrence:

```
V_t(j) = emission_t(j) + max_i [V_(t-1)(i) + transition_t(i,j)]
```

For each candidate j, we keep the best score of a sequence ending there and the
previous candidate that gave that score. After the forward pass, we backtrack from
the best final candidate. This differs from choosing the largest emission at every
point: a slightly weaker local choice can belong to a better complete sequence.
Adding log scores avoids multiplying many small probabilities. The fixed decoder
also subtracts the maximum score at each step, without changing the winning path.

The fixed decoder restarts after a gap above 5 seconds or a missing-candidate point.
It also restarts for nonpositive or nonfinite time differences. The multi-hop decoder
sets transition costs to zero across gaps above 120 seconds. Missing-candidate
points remain -1, while decoding continues between the surrounding available points
using their elapsed time. For nonpositive time differences it substitutes 1 second;
it has no explicit check for nonfinite timestamps. These are different gap policies.

## 5.4 Experiment and metrics

The comparison covers the same 54 trips: 22 training trips labelled TUNE, 22 labelled
VAL, and 10 student_test trips. The split uses seed 0. Training trips with candidate
miss rate above 5% or more than 8,000 points are excluded from this comparison.
It therefore does not cover every training trip or the longest speed-test trip.
The multi-hop tuning log is `results/hmm_tuning_mock.csv`.

The main comparison is saved in `results/hmm_decoder_comparison_mock.csv` and
`results/hmm_decoder_comparison_mock_per_trip.csv`. The additional run in
`results/hmm_two_variant_same54_mock.csv` supplies sequence scores. Its classifier
baseline and multi-hop accuracy matched the earlier results on all 54 trips.

One limitation is that the two variants were tuned on different trip sets. The fixed
variant was tuned earlier on 90, 69, 42, 03 and 05; 90, 69 and 42 appear in the other
variant's VAL set. We retain the saved split labels below, but VAL is not untouched
validation for both variants. The final comparison needs one common split.

We use the shared functions in `src/evaluate.py`. Point accuracy counts correct arc
IDs on observations whose derived reference arc is within 60 metres of the GPS
point. This matches the 54-trip evaluator; the earlier nine-trip experiment instead
used `has_positive`. Route-derived labels are an approximation of the road travelled.
The reported averages give each trip equal weight.

Continuity counts directly connected changes in arc ID. Repeated IDs are excluded
from the denominator, and a trip with no changes receives 1. Missing predictions
are filtered before scoring, which can join observations across a missing point.
This metric does not credit a connection through intermediate arcs. Sequence score
is the fraction of non-overlapping 50-point windows whose predictions all match the
reference. It uses all points, without the accuracy mask, and drops an incomplete
window at the end.

### TUNE (22 trips)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity |
|---|---:|---:|---:|
| Mock classifier only | 77.86% | 4.70% | 66.07% |
| Fixed HMM | 81.08% | 8.02% | 98.74% |
| Multi-hop HMM | 84.03% | 10.26% | 89.44% |

### VAL (22 trips)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity |
|---|---:|---:|---:|
| Mock classifier only | 80.87% | 6.28% | 68.72% |
| Fixed HMM | 84.35% | 8.12% | 97.01% |
| Multi-hop HMM | 86.95% | 9.94% | 91.88% |

### STUDENT_TEST (10 trips)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity |
|---|---:|---:|---:|
| Mock classifier only | 81.36% | 10.38% | 70.48% |
| Fixed HMM | 83.84% | 11.18% | 99.32% |
| Multi-hop HMM | 85.82% | 12.70% | 89.83% |

Timings depend on the machine and run conditions. In the committed comparison run,
total decoding time across the 22 VAL trips was 3.2968 seconds for multi-hop and
0.8492 seconds for fixed transitions (`results/hmm_decoder_comparison_mock.csv`).
These totals exclude feature preparation and API overhead. We do not use timings
from the separate sequence-score run for this speed comparison.

## 5.5 What helped and what did not

Multi-hop decoding gave higher mean accuracy than fixed transitions on VAL and
student_test. Fixed transitions gave higher direct continuity. This is a useful
trade-off: fixed transitions favour adjacent predictions, while multi-hop decoding
allows roads between two observations to be skipped. The continuity metric penalises
such skips even when the graph has a valid connecting path.

Trip 00000078 shows why continuity alone is not enough. In the earlier nine-trip
results (`results/hmm_fixed_batch.csv`), fixed decoding reduced accuracy from
82.09% to 80.47%, although continuity increased from 76.98% to 95.37%. The largest
gap was 52,810 seconds. The decoder may have favoured a connected but wrong route;
we would need to inspect the map to establish where and why this happened.
Multi-hop decoding reached 86.49% accuracy and 84.86% direct continuity on this
trip in the saved comparison. Trip 78 is in multi-hop TUNE, so that result is not
independent validation. Its reference coverage is complete, so the two scoring
masks do not affect this particular accuracy comparison.

## 5.6 Update after the classifier handoff

Report figures and captions: [HMM map comparisons](section_6_hmm_figures.md).
They show the improvement on training trip 00000010 and the fixed-transition
regression on 00000078, with classifier-only and both HMM variants side by side.

All tables above use mock emissions. Once SRN 496 supplies the selected model,
replace the mock scores with its positive-class probabilities in both decoders.
Retune both variants on the same whole-trip tuning set, then score the separate
validation and student_test trips. Update the configuration values, tables, timing
example and failure analysis from those runs; retain the mock results as a baseline.
The numbers here should not be treated as the final model's performance.

Use a fresh output filename for every new run. The reproduction helper in
`report/evidence/reproduce_hmm_comparison.py` refuses to overwrite an existing file;
it currently reproduces mock results and must be adapted for trained probabilities.
Existing committed results were not changed for this revision. No evaluator trips
were used. AI assistance was used for implementation and drafting; the team must
review the explanation before submission.
