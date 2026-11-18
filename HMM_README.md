> **Merge note:** this decoder lives in `src/hmm_fixed.py` with config `artifacts/hmm_fixed_config.json` (tests: `tests/test_hmm.py`, batch script: `src/batch_evaluate_hmm.py`). The multi-hop decoder used by the API is `src/hmm.py` with `artifacts/hmm_config.json`. On the same 54 trips with the same mock probabilities, the multi-hop decoder had higher accuracy (VAL 0.870 vs 0.844, student_test 0.858 vs 0.838) and this fixed decoder had higher continuity (VAL 0.970 vs 0.919, student_test 0.993 vs 0.898) and was about 4x faster.

# HMM handoff for Tejas (SRN 549)

API: `decode(trip, feats, cfg) -> np.ndarray` (`int64`, length = GPS points).
`feats` must have `point_idx`, `arc_id`, `prob` (positive-class probabilities in
[0,1]); `cfg` is the dict loaded from `artifacts/hmm_config.json`.
Points with no candidates return -1 and break the sequence. Input is not mutated.
No route or label is accessed by the decoder.

The initial pushed version uses fixed log transition potentials for same arc,
directly adjacent arcs, and other arcs. No graph search or projection is required.
A gap above `gap_s`, nonpositive dt, or nonfinite dt restarts Viterbi. Finite
penalties let the model recover from invalid transitions; continuity is not guaranteed.
Scores are unnormalised log potentials; Viterbi adds them to log emissions,
rescales each step, and backtracks. Candidate arcs are the hidden states.

Mock probabilities: per-point softmax(-dist_m/10 + 3*cos_head). These are heuristics,
not a trained classifier. The API currently enables HMM only after model.joblib
exists; with a real classifier it supplies predict_proba(X)[:,1] as `prob`.

Run from repository root with a working Python and requirements.txt dependencies:

```powershell
python -m unittest discover -s tests -v
python src/batch_evaluate_hmm.py --data-root '../Student folder' --tune
```

Local portable interpreter: `.runtime/python/python.exe` (gitignored).
The batch evaluator explicitly accepts only train/student_test; it uses
`prepare_trip` for inference and reads reference tables only for scoring.
Accuracy is computed on `has_positive` points. Tuning selects macro trip accuracy
on 90/69/42/03/05, then checks 78/21/81/10 without parameter selection on those trips.
Six combinations of gap threshold and disconnected penalty are tested.
Shared metrics from src/evaluate.py are used. Its continuity metric counts changed
arc pairs only, unlike the original prototype's same-or-adjacent metric. Missing
predictions require care when interpreting continuity; the tested trips report
missing counts. Fifty-point exact-match windows include all reference points.
Outputs: mode-specific batch, tuning CSVs and run JSON files (see below).
Historical hmm_demo.json and hmm_predictions.csv describe the superseded shortest-
path prototype with different mock settings; do not mix them with current results.
No evaluator trips are read. Runs are deterministic. AI assistance was used.


## Whole-trip validation

All nine requested train trips were run. Six fixed-transition configurations were
selected only on train 90/69/42/03/05. The selected configuration resets after 5 s
and uses disconnected log potential -12. On held-out train 78/21/81/10, macro trip
accuracy increased from 74.56% to 81.61%, and shared changed-arc continuity from
64.08% to 97.20%. Trip 78 regressed (82.09% to 80.47%). Trip 21 (15,783 points)
decoded in 0.46 s; this excludes feature preparation (0.18 s locally). No predictions
were missing on these trips. Times depend on the laptop and exclude API overhead.

Optional `mode: network` implements directed endpoint shortest paths, capped at
250 m and 128 settled nodes per source, with a 2048-source LRU cache. It penalises
|network distance - GPS displacement| / beta and supports forward progress within
an arc. Budgets can mark a reachable pair unavailable; these use a finite fallback
penalty. Six network configurations were tried on the same tuning trips. Best
network tuning accuracy was 77.83%, versus 83.94% for fixed transitions; therefore
fixed mode remains selected. The network variant is experimental, not an improvement
in the measured results. Its config is saved separately as hmm_network_config.json.
Use `--mode network --tune` to reproduce that experiment.

Current authoritative outputs are results/hmm_fixed_{batch,tuning}.csv,
results/hmm_network_{batch,tuning}.csv and their *_run.json metadata files.
Four HMM tests pass, including network mode, missing-candidate -1 behaviour,
nonfinite probability rejection, log-space underflow resilience, and gap resets.
The tests are synthetic correctness checks, separate from the nine real-trip runs.

Report explanation: the HMM stabilises ambiguous candidate choices by rewarding
adjacent roads. It can also oversmooth and stay on a plausible but wrong road, as
trip 78 shows. Distance-only shortest paths struggled with this mock emission and
reference-label setup; do not claim network mode improves generalisation.

## Guidelines sections 8-9 (checked)

Read Models and Training Guidelines.pdf from the supplied PathSnap_student_materials
folder. It requires explicit state/candidate mapping, emission/transition choices,
Viterbi recurrence, missing/invalid/gap handling, per-observation predictions, and
classifier-only versus HMM comparisons including point accuracy, sequence metrics,
and continuity. These are covered here and in batch results, subject to the mock
classifier limitation. Final compliance still requires repeating the comparison
with 496's selected model outputs consistently used as emissions.

Recurrence: V_t(j) = log(max(prob_t(j), 1e-300)) +
max_i[V_(t-1)(i) + T(i,j)]. Store each maximizing i and backtrack from the final
maximum. Subtract the maximum of each score vector for numerical stability; this
does not alter the maximizing path. At a reset, begin with the current emissions.
Candidates retain their original arc_id; returned IDs are not candidate positions.

The model is a chain with unnormalised compatibility potentials rather than a
calibrated generative HMM. Fixed potentials explicitly favour road adjacency;
the optional distance model estimates realistic movement but was weaker in these
experiments. Missing observations produce -1, and both sides decode independently.
Invalid graph transitions remain possible under a finite penalty so a poor candidate
set cannot make the entire sequence impossible. Sequence exact-50 scores are the
fraction of non-overlapping 50-point windows entirely matching derived reference
labels, and unlike point accuracy they are not masked by has_positive.

