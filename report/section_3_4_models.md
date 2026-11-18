# Track B: model comparison and Bayesian optimisation

Owner: Aarya, PES1UG24CS496. All values below come from
`results/models/deadline_all_01/` and `results/models/tuning_deadline_02/`.

## Experimental protocol

The deadline experiment used seed 42 to select 40 complete training trips.
After retaining pt_has_pos=1, 1,103,014 candidate rows were available. The 18
features describe candidate distance, heading, speed, neighbouring GPS evidence,
geometry and endpoint degree. IDs, labels and the filter were excluded.
Three-fold GroupKFold held out entire trips. Preprocessing was fitted inside
each fitting partition. RF/MLP used up to 300,000 fitting rows and KNN up to
150,000; samples retained full candidate groups for each selected observation.
LogReg and LightGBM used all eligible fitting rows. This is a time-budgeted
comparison with differing per-model caps, not an equal-compute benchmark.

Student-test evaluation used all 244,176 candidate rows over 19,160 points in
10 trips, including points without a positive candidate. Reference labels are
derived by route alignment and are not independently verified physical truth.
The rebuilt training table's 210-point coverage discrepancy is documented in
the training provenance file. This run does not use all 80 available training
trips; that limitation should remain explicit in the final report.

## Comparison

The predefined rule selected highest mean fold-level point Top-1, with lower
mean candidate log loss breaking exact ties. Student-test did not choose the
winner. Macro-F1/precision/recall are binary candidate metrics at threshold 0.5,
not multiclass road-ID metrics. Point Top-1 takes the maximum positive-class
probability within each observation's candidate set.

| Model | CV Top-1 | Student Top-1 | CV macro-F1 | Student macro-F1 | CV log loss | Student log loss |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.82690 | 0.85057 | 0.87841 | 0.89946 | 0.12020 | 0.07803 |
| LightGBM | 0.83265 | 0.84880 | 0.90670 | 0.91633 | 0.14471 | 0.09993 |
| Random Forest | **0.83765** | 0.85350 | 0.90664 | **0.91783** | 0.09774 | **0.05753** |
| MLP | 0.83006 | **0.85611** | 0.90122 | 0.91594 | **0.08710** | 0.05937 |
| KNN | 0.80877 | 0.82740 | 0.88474 | 0.90038 | 0.22608 | 0.10875 |

Nearest-arc Top-1 was 0.51233 in mean CV and 0.50809 on student-test. All five
models exceeded these matching baselines. RF's lead over LightGBM was only
0.00500 (0.50 percentage points); no statistical significance is claimed.
MLP achieved the best student-test Top-1, but RF won under the declared CV rule.
Candidate F1 and Top-1 rank models differently because they measure different
decisions. Actual confusion matrices, precision and recall are in metrics.csv.

| Model | Mean CV fit seconds | Mean CV inference seconds | Student inference seconds |
|---|---:|---:|---:|
| Logistic Regression | 4.39 | 0.22 | 0.15 |
| LightGBM | 11.94 | 2.23 | 1.38 |
| Random Forest | 37.71 | 3.35 | 2.22 |
| MLP | 25.14 | 0.51 | 0.33 |
| KNN | 0.47 | 778.20 | 448.41 |

These times cover fitting or candidate scoring, not feature generation or HMM
decoding. KNN's cheap fit hides its expensive neighbour search during prediction.

## Model configurations and interpretation

- Logistic Regression: scaled inputs, max_iter=200, binary log-loss objective.
  It gives a fast linear baseline, but cannot directly represent complex
  nonlinear interactions. Its CV Top-1 was below RF's.
- LightGBM: 300 boosted trees, learning_rate=0.1, num_leaves=63, binary log-loss
  objective. It captures nonlinear interactions efficiently. Its point-ranking
  score was competitive, but its probability log loss was worse than RF's in
  this run; ranking quality and probability quality should not be conflated.
- Random Forest: 100 trees, max_depth=18, min_samples_leaf=20, Gini split
  criterion. Bagged trees capture nonlinear feature combinations; depth/leaf
  limits constrain complexity. It won CV point selection and offered practical
  inference speed. Probability calibration is not assumed from predict_proba.
- MLP: scaled inputs, hidden layers (64,32), batch_size=2048, at most 30 epochs,
  Adam optimisation of binary log loss with regularisation. A whole-trip inner
  holdout chose the epoch count, followed by refitting on capped fitting data.
  It achieved the best student Top-1. Actual monitored and final training
  losses are saved; no curves are invented for tree or neighbour models.
- KNN: scaled inputs, k=25, distance weighting, exact ball-tree search, 150k-row
  cap and 4096-row prediction chunks. It has no iterative optimisation loss;
  neighbour labels determine probabilities. It had the weakest CV Top-1,
  highest CV log loss and largest prediction cost in this comparison.

All models used random seed 42 where applicable. Model-specific parameter JSONs
record remaining defaults. Student-test scores exceed mean CV scores here;
this alone establishes neither absence of overfitting nor proof of underfitting.
The report should not invent training-validation gaps absent logged evidence.
Route continuity and HMM sequence results belong to the decoder experiments.

## Bayesian hyperparameter optimisation

The selected RF was tuned with Optuna TPE, minimising mean three-fold training
CV log loss. The search covered max_depth in [10,22] and min_samples_leaf in
[10,50], retaining 100 trees and the same fit caps. Of 25 requested trials,
six completed before the timeout stopped further trials. The first five are
startup trials; only the sixth uses the subsequent TPE proposal mechanism.
Search time was 676.67 seconds because timeout checks occur between trials.
An earlier interrupted study is preserved separately and is not counted as a
completed 25-trial search. No evaluator data entered optimisation.

Best trial: max_depth=12, min_samples_leaf=16, CV log loss=0.0807695.

| Metric | Untuned RF | Tuned RF |
|---|---:|---:|
| Mean CV Top-1 | **0.83765** | 0.83640 |
| Mean CV log loss | 0.09774 | **0.08077** |
| Student Top-1 | 0.85350 | **0.85402** |
| Student macro-F1 | 0.91783 | **0.91802** |
| Student log loss | **0.05753** | 0.05779 |

Tuning improved its CV objective but produced mixed generalisation results.
Student point selection improved by 10 points out of 19,160, only 0.052
percentage points; student log loss slightly worsened. The tuned artifact was
selected on CV log loss, not by choosing whichever version looked better on
student-test. No broad superiority or guaranteed calibration is claimed.
