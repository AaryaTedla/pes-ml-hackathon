# Track B training — Aarya PES1UG24CS496

Source: `src/train_models.py`. Runnable, self-contained Kaggle notebooks are in
`notebooks/PathSnap_Deadline_{All,Fast,Slow}.ipynb`. All runs five models; Fast/Slow
split the same protocol across two sessions. Attach the private feature dataset.
The notebook installs the original experiment's exact repository pins and
Optuna 5.1.0, then requires a session restart before imports. Local equivalent:

```bash
python -m pip install -r requirements-training.txt
```

The recorded run used Python 3.13.18. The notebook copies embed the same training
functions as the source module. Set RUN_COMPARISON=True to execute; outputs go
to a new results directory. Do not rerun an existing directory or mix protocols.
Read the notebook's declared selection rule and sampling description first.

## Reproduce locally with the same feature tables

```python
from pathlib import Path
from src.train_models import load_tables, compare, plot_comparison, tune_winner

train, test, selected = load_tables(
    'results/features_train.parquet', 'results/features_student_test.parquet')
out = Path('results/reproduction_01')
comparison = compare(train, test, selected, out, group='all')
plot_comparison(comparison, out / 'plots')
winner = comparison.sort_values(['cv_top1', 'cv_log_loss'], ascending=[False, True]).iloc[0]['model']
tune_winner(train, test, winner, 'results/reproduction_tuning_01', trials=25, timeout=600)
```

Feature schema: 18 inputs in src/features.py order; IDs, label and pt_has_pos
are not inputs. Seed 42 selects 40 of 80 training trips. Only pt_has_pos=1 enters
three-fold GroupKFold; validation trips remain separate. RF/MLP fit caps sample
complete point candidate sets up to 300k rows; KNN uses up to 150k. All test
rows are scored. MLP uses an inner whole-trip early-stopping holdout and refits
for the selected epoch count, not sklearn's random-row early-stopping split.
Imputation and scaling never fit on outer validation or student-test data.

Selection: highest mean CV point Top-1, exact ties broken by lowest CV candidate
log loss. Optuna subsequently minimises CV log loss. Student-test is used for
comparison, not the tuning objective. No evaluator data is used.

The original training parquet is absent from Git. The Kaggle training table was
rebuilt using Track A source. All reference arc IDs agree with the source point
table, but candidate coverage differs at 210 points (22,212 no-positive points,
versus 22,362 documented originally). See
`results/models/training_data_provenance.json` for checksums and the source
commit. Rebuilding with different geometry-library versions may change candidate
membership; use those exact input file hashes to reproduce the recorded run.
Do not overwrite the team's original point table with this rebuilt table.

## Evidence and integration

Recorded comparison and tuning evidence is in `results/models/`; the report
interpretation is `report/section_3_4_models.md`. The deployment model at
`artifacts/model.joblib` matches the tuned Kaggle model byte-for-byte.
`artifacts/model_training_metadata.json` records its tuning settings.

The artifact is a standard sklearn Pipeline, including imputation and optional
scaling. Pass raw ordered features; do not scale twice. Use the predict_proba
column whose entry in classes_ is 1. The RF scaler step is passthrough, so no
external scaler is needed. Keep the API's existing artifacts configuration.

The tuning timeout is checked between trials. Six completed trials, including
the first five TPE startup trials, were observed in the recorded run; not 25.
Timing can change how many trials complete, so exact trial replay should use
the saved parameters/trial table rather than assume identical timeout behaviour.
No automatic claim of probability calibration is made.
