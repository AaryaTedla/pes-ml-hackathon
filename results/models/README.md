# Model experiments (Track B, run on Kaggle by SRN 496)

- `kaggle_run_outputs.zip` is the full output of the Kaggle runs: the five fitted models, their student_test score tables,
  the Optuna study databases and the tuned Random Forest. Unzip to inspect.
- The small readable files from the zip are also unpacked here:
  - `model_comparison/`: the 5-model comparison (`comparison.csv`, `metrics.csv`, `metadata.json`, plots, per-model configs and loss curves)
  - `rf_tuning/`: Optuna trials, tuned-vs-untuned comparison and tuning metadata for the Random Forest
  - `rf_tuning_first_attempt/`: an earlier tuning attempt
- Protocol: 40 whole training trips, 3-fold GroupKFold by trip, seed 42; the Random Forest, MLP and KNN were trained on capped row counts (see `metadata.json`).
  Student_test was used for comparison only. Evaluator trips were never used.
- Known limits: only 6 of 25 requested Optuna trials completed (10 minute timeout), so the search was effectively random sampling;
  tuned and untuned Random Forest are nearly identical on student_test (see `rf_tuning/tuned_vs_untuned.csv`).
- The tuned Random Forest used by the API is `artifacts/model.joblib` (a complete scikit-learn Pipeline: imputer, pass-through scaler, model).
