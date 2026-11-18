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
