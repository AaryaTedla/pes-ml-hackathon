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
Outputs: results/hmm_tuning.csv, hmm_batch.csv, hmm_run.json.
Historical hmm_demo.json and hmm_predictions.csv describe the superseded shortest-
path prototype with different mock settings; do not mix them with current results.
No evaluator trips are read. Runs are deterministic. AI assistance was used.

The materials ZIP and guideline PDF were not present at the searched Downloads or
repository paths. Report compliance with sections 8–9 is not yet verified.

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
