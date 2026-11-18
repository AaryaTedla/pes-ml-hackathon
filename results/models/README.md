# Model experiments (Track B, run on Kaggle by SRN 496)

- `pathsnap_deadline_results.zip` is the full output of the Kaggle runs: the five fitted models, their student_test score tables,
  the Optuna study databases and the tuned Random Forest. Unzip to inspect.
- The small readable files from the zip are also unpacked here:
  - `deadline_all_01/`: the 5-model comparison (`comparison.csv`, `metrics.csv`, `metadata.json`, plots, per-model configs and loss curves)
  - `tuning_deadline_02/`: Optuna trials, tuned-vs-untuned comparison and tuning metadata for the Random Forest
  - `tuning_deadline_01/`: an earlier tuning attempt
- Protocol: 40 whole training trips, 3-fold GroupKFold by trip, seed 42; the Random Forest, MLP and KNN were trained on capped row counts (see `metadata.json`).
  Student_test was used for comparison only. Evaluator trips were never used.
- Known limits: only 6 of 25 requested Optuna trials completed (10 minute timeout), so the search was effectively random sampling;
  tuned and untuned Random Forest are nearly identical on student_test (see `tuning_deadline_02/tuned_vs_untuned.csv`).
- The tuned Random Forest used by the API is `artifacts/model.joblib` (a complete scikit-learn Pipeline: imputer, pass-through scaler, model).
