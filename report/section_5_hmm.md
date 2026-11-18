# 5. HMM and Viterbi Route Reconstruction

## 5.1 Why we added a sequential decoder

Selecting the highest-scoring road independently at each GPS point can produce a
route that repeatedly switches between nearby streets or between opposite directions
of the same road. We therefore compared two Viterbi decoders using identical candidate
arcs and identical mock probabilities: a fixed-transition decoder in `src/hmm_fixed.py`
and a multi-hop decoder in `src/hmm.py`. The API uses the multi-hop variant. These
experiments measure the effect of sequential decoding with a heuristic emission;
they are not results from the final trained classifier.

## 5.2 Hidden states and emissions

At observation t, each hidden state is one of that observation's candidate directed
road arcs. We preserve `point_idx` and `arc_id`, so a selected candidate maps directly
to a row of the trip's own `.arcs` file. Both decoders return an integer NumPy array
with one arc ID per GPS point. If no candidate exists, that position receives -1.
Neither decoder uses the reference route as an inference input.

Until the selected classifier is available, we use the following stand-in score:

```
z_t(a) = -dist_m(t,a) / 10 + 3 * cos_head(t,a)
p_t(a) = exp(z_t(a)) / sum_b exp(z_t(b))
e_t(a) = log(max(p_t(a), epsilon))
```

The softmax is calculated separately for each observation, after subtracting its
largest score for numerical stability. Distance is in metres; dividing by 10 makes
the score dimensionless. The heading term favours travel along an arc rather than
against it, which helps distinguish reverse arcs with identical geometry. These
probabilities are heuristic scores, not calibrated outputs of a fitted classifier.
The fixed decoder clips at 1e-300 before taking logs; the multi-hop configuration
uses epsilon = 1e-6. When 496's selected model arrives, its positive-class
probabilities must replace the mock emissions consistently in both experiments.

## 5.3 Transition models

The fixed decoder uses three dimensionless log compatibility scores: 0 for staying
on the same arc, -0.2 for moving to an arc whose start node is the previous arc's end
node, and -12 for other movements. It does not search for intermediate connector
arcs. This makes it fast and strongly favours directly adjacent predictions, but
legitimate movements across several short road segments can receive a jump penalty.
The selected fixed configuration is `artifacts/hmm_fixed_config.json`.

The multi-hop decoder searches directed arc adjacency using a memoised,
depth-limited breadth-first search. Staying on an arc has zero cost. A reachable
arc h hops away costs `hop_penalty * h`; an immediate reverse twin uses a separate
U-turn penalty. An arc not found within the search depth receives a finite jump
penalty. Search depth increases with elapsed time:

```
depth = max_hops + floor(min(dt, gap_s) / gap_hop_s)
```

The committed configuration uses hop_penalty=4, jump_penalty=40,
uturn_penalty=16, max_hops=5, gap_s=120 seconds and gap_hop_s=5 seconds.
These are hop-based costs, not physical distances or travel times. In particular,
the active multi-hop decoder does not run a metric shortest-path calculation or
compare road distance with GPS displacement. The bounded Dijkstra experiment in
`hmm_fixed.py` is a separate optional mode and is not the variant compared here.

Both variants use unnormalised compatibility potentials rather than a fully
calibrated generative HMM. A finite jump cost allows recovery when candidate sets
are poor; it does not guarantee a connected output. Neither decoder inserts
intermediate connector arcs into its per-point output.

## 5.4 Viterbi, gaps and missing observations

For candidate j at observation t, we calculate:

```
V_t(j) = e_t(j) + max_i [V_(t-1)(i) + T_t(i,j)]
```

Here T is the fixed log score or the negative multi-hop cost. For every candidate,
we store the predecessor that maximises the expression. After the forward pass,
we start from the highest-scoring final candidate and follow those predecessors
backwards. This finds the best complete candidate sequence under the chosen model,
rather than making a greedy choice at each observation. Log scores avoid multiplying
many small probabilities. The fixed decoder also subtracts the largest cumulative
score at each step; this leaves the best path unchanged.

The fixed variant starts a new sequence after a gap above 5 seconds, a nonpositive
or nonfinite timestamp difference, or an observation with no candidates. In the
multi-hop variant, gaps above 120 seconds set all cross-gap transition costs to
zero, making the new observation independent of the previous state choice.
Candidate-less observations remain -1, but the multi-hop chain links the surrounding
observations using the elapsed time between them. Its implementation substitutes
1 second for nonpositive elapsed times. These behaviours differ and should not be
described as identical resets. Nonfinite timestamp validation is not explicitly
implemented in the active multi-hop decoder.

## 5.5 Experimental protocol and evidence

The committed multi-hop result file, `results/hmm_eval_mock.csv`, contains 54 trips:
22 TUNE training trips, 22 VAL training trips and all 10 student_test trips. Its
split construction uses seed 0, excludes training trips with candidate miss rate
above 5%, and excludes training trips above 8,000 points. Thus this 54-trip comparison
is not a full evaluation of every training trip or of the longest speed-test trip.
Multi-hop parameters were selected using its TUNE trips; trials are recorded in
`results/hmm_tuning_mock.csv`.

We reproduced the fixed decoder on exactly those trip IDs using its existing
configuration, without selecting parameters on the student_test trips. The mock
classifier baseline matched the committed baseline on all 54 trips. We then ran
both variants on the same inputs to record sequence scores and paired timings in
the new file `results/hmm_two_variant_same54_mock.csv`; original result files were
not overwritten. Each timing measures decoding only, excludes feature generation,
and reflects one run with fixed decoding executed before multi-hop decoding.
The script and run metadata are in `report/evidence/`.

Tejas subsequently committed the same 54-trip comparison in
`results/hmm_decoder_comparison_mock.csv` and its per-trip companion
`results/hmm_decoder_comparison_mock_per_trip.csv`. Their accuracy and continuity
agree with the values below after rounding. That committed run records total
multi-hop versus fixed decoding times of 3.2968 versus 0.8492 seconds for VAL,
and 1.1214 versus 0.3176 seconds for student_test, supporting the approximate
fourfold speed advantage in that run. Our additional paired run supplies sequence
scores and a separate timing measurement; its timings should not be substituted
for or mixed with Tejas's run.

To reproduce from the repository root, choose a fresh output file:

```powershell
python report/evidence/reproduce_hmm_comparison.py --data-root "../Student folder" --output results/hmm_comparison_new_run.csv
```

The helper refuses to overwrite an existing output path. The earlier batch script
still uses fixed result filenames and should not be run over the committed evidence.

There is a split limitation: the fixed decoder was previously tuned on trips
90, 69, 42, 03 and 05. Trips 90, 69 and 42 appear in the multi-hop VAL set. We therefore
report the shared VAL label for traceability but do not claim it is untouched
validation for the fixed decoder. A final fair selection experiment should use one
common tuning/validation split for both decoders with the trained emissions.

Point accuracy in this same-54 comparison uses reference labels within 60 metres
of the observation, matching the committed multi-hop evaluator. This differs from
the earlier nine-trip fixed experiment, which scores `has_positive` points. These
reference labels were derived from ordered route arcs and are not independently
verified physical truth. All tabulated aggregate values are unweighted means over
trips, rather than pooling all GPS points.

Route continuity uses `src/evaluate.py`: among changes in predicted arc ID, it
measures the fraction for which the first arc ends at the second arc's start node.
Repeated arc IDs are excluded from its denominator; a trip with no changes returns
1. Missing IDs are filtered before this metric in the paired comparison, so it can
bridge candidate-less observations. This is direct adjacency, not multi-hop
reachability. A valid multi-hop movement can therefore lower the reported continuity.
Sequence performance is the fraction of non-overlapping 50-point windows with every
prediction equal to its reference label. Sequence scores use all points, without the
60-metre accuracy mask; trailing incomplete windows are omitted.

### TUNE (22 trips)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity | Mean decode time/trip |
|---|---:|---:|---:|---:|
| Mock classifier only | 77.86% | 4.70% | 66.07% | Not measured |
| Fixed HMM | 81.08% | 8.02% | 98.74% | 0.312 s |
| Multi-hop HMM | 84.03% | 10.26% | 89.44% | 0.672 s |

### VAL (22 trips)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity | Mean decode time/trip |
|---|---:|---:|---:|---:|
| Mock classifier only | 80.87% | 6.28% | 68.72% | Not measured |
| Fixed HMM | 84.35% | 8.12% | 97.01% | 0.429 s |
| Multi-hop HMM | 86.95% | 9.94% | 91.88% | 0.928 s |

### STUDENT_TEST (10 trips)

| Pipeline | Point accuracy | Exact 50-point windows | Direct continuity | Mean decode time/trip |
|---|---:|---:|---:|---:|
| Mock classifier only | 81.36% | 10.38% | 70.48% | Not measured |
| Fixed HMM | 83.84% | 11.18% | 99.32% | 0.351 s |
| Multi-hop HMM | 85.82% | 12.70% | 89.83% | 0.716 s |

Source: `results/hmm_two_variant_same54_mock.csv`. Paired-run decode times are descriptive, not a controlled benchmark; the historical fourfold speed claim is not reproduced here. In this run the fixed decoder took about half the multi-hop decoding time on VAL and student_test.

## 5.6 Improvements, failure cases and interpretation

The multi-hop variant has higher mean point accuracy than the fixed variant on both
VAL and student_test. Its ability to cross intermediate road segments is consistent
with that result, although this experiment alone does not establish the cause. The
fixed variant has higher direct-adjacency continuity. That reflects both its stronger
adjacency preference and a metric that does not credit multi-hop connections.
Continuity alone is therefore insufficient for choosing the final decoder.

Trip 00000078 is a useful counterexample from the earlier nine-trip experiment
(`results/hmm_fixed_batch.csv`). Classifier-only accuracy was 82.09%, while fixed HMM
accuracy fell to 80.47%; direct continuity rose from 76.98% to 95.37%. Its maximum
recorded gap is 52,810 seconds. A plausible explanation is that the transition cost
favoured a coherent but incorrect road sequence; locating the precise cause would
require inspecting predictions on the map. Higher continuity did not ensure a more
accurate route. On this trip the committed multi-hop run achieved 86.49% accuracy
and 84.86% direct continuity. Trip 78 belongs to multi-hop TUNE, so that figure is not
an independent validation result. Its reference coverage is complete in the saved
comparison, making its accuracy directly comparable across these two masks.

We retain the multi-hop variant in the API because of its measured accuracy advantage,
while keeping fixed transitions as a faster alternative. Final conclusions depend
on rerunning and retuning both variants with 496's real classifier probabilities.
No evaluator trips were accessed or used in these experiments. AI assistance was
used for implementation, verification and this draft; team members must review the
explanation and be able to defend the code and results in the viva.

## Viva preparation

- A hidden state is a directed candidate road arc, not a GPS coordinate or route index.
- Emissions score observation-to-arc agreement; transitions score successive arc compatibility.
- Viterbi optimises a complete sequence; choosing the best emission independently is the baseline.
- Finite penalties permit recovery but allow disconnected predictions. The output does not expand connector arcs.
- Fixed adjacency scores are cheap; multi-hop BFS recognises intermediate segments, but hop counts are not metres.
- Better continuity can coexist with worse accuracy, as trip 78 demonstrates.
- Mock scores are not trained-model results, and the two historical tuning splits differ.
