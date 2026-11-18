### 3.2 Compulsory questions for every model

Data for all answers: 3-fold cross-validation grouped by trip on 40 training trips (seed 42), then scoring on the 10 student_test trips
(244,176 candidate rows, 19,160 GPS points, of which 19,135 rows are positive). Source files are in `results/models/model_comparison/`
(`comparison.csv`, `metrics.csv`, `<Model>_config.json`). Precision and recall are macro-averaged over the two classes (not a correct/incorrect split of road ids).
Statements marked *reasoning* are explanations we did not test with a separate experiment. Confusion matrices are written [[true negative, false positive], [false negative, true positive]].

#### 3.2.1 Logistic Regression (linear / probabilistic)

- **A1. Why suitable?** It is the simplest probabilistic model for "is this candidate road the correct one", so it shows how far a straight-line decision boundary on the 18 features gets us. It is a baseline, not an expected winner.
- **A2. Strengths and limitations.** Fast (CV fit 4.4 s), stable, and its probabilities are reasonably well behaved (student-test log loss 0.078). A linear boundary cannot capture interactions such as "close to the road *and* heading along it", and it had the lowest CV macro-F1 (0.878).
- **B5. Important hyperparameters (initial values).** L2 penalty, C = 1.0, lbfgs solver, max_iter = 200, inputs standardised.
- **B6. Loss.** Binary cross-entropy (log loss) plus the L2 penalty.
- **C7 and C8. Student-test results.** Top-1 0.8506, macro-F1 0.8995, log loss 0.0780, precision 0.909, recall 0.891; confusion matrix [[222,040, 3,001], [3,933, 15,202]].
- **C9. Over- or underfitting?** No training-set scores were logged, so we cannot measure a train-validation gap. The lower macro-F1 than all non-linear models trained on the same rows is *consistent with underfitting* (too simple), and nothing suggests overfitting.

#### 3.2.2 LightGBM (boosting)

- **A1. Why suitable?** Boosted trees capture non-linear interactions between distance, heading and speed efficiently, and are a standard strong choice for tabular data.
- **A2. Strengths and limitations.** Fast to fit (11.9 s) and strong at ranking candidates (CV macro-F1 0.907), but its probabilities were worse than the random forest's: CV log loss 0.145 (vs 0.098) and a wide fold range (0.091 to 0.215). That matters because the HMM multiplies these probabilities.
- **B5. Important hyperparameters (initial values).** 300 trees, learning_rate 0.1, num_leaves 63, min_child_samples 20, no L1/L2 regularisation (reg_alpha = reg_lambda = 0), all rows used.
- **B6. Loss.** Binary log loss (gradient boosting on the binary cross-entropy).
- **C7 and C8. Student-test results.** Top-1 0.8488, macro-F1 0.9163, log loss 0.0999, precision 0.918, recall 0.915; confusion matrix [[222,172, 2,869], [3,013, 16,122]].
- **C9. Over- or underfitting?** No training-set scores were logged, so a train-validation gap is not measured. The high log loss with good F1 and a wide fold range is *consistent with over-confident probabilities* (many leaves, no regularisation), not with underfitting. It is evidence about probability quality, not proof of overfitting.

#### 3.2.3 Random Forest (bagged trees) - selected model

- **A1. Why suitable?** Averaging many decorrelated trees handles non-linear interactions and noisy inputs without feature scaling, and gives a probability from the fraction of trees voting for the candidate.
- **A2. Strengths and limitations.** Highest CV Top-1 (0.838) and the best student-test log loss (0.0575) among the five; stable. Slower to fit (37.7 s) and to score (2.2 s for 244k rows) than the linear model, and its probabilities are vote fractions, so calibration is not guaranteed.
- **B5. Important hyperparameters (initial values).** 100 trees, max_depth 18, min_samples_leaf 20, max_features = sqrt, bootstrap, trained on up to 300,000 rows. After Optuna: max_depth 12, min_samples_leaf 16 (section 4).
- **B6. Loss.** Splits minimise Gini impurity. There is no single global loss; tuning used cross-validated log loss.
- **C7 and C8. Student-test results.** Top-1 0.8535, macro-F1 0.9178, log loss 0.0575, precision 0.932, recall 0.905; confusion matrix [[222,877, 2,164], [3,453, 15,682]].
- **C9. Over- or underfitting?** Depth and leaf-size limits constrain complexity. No training-set score was logged. CV log loss varies a lot between folds (0.060 to 0.155), which shows sensitivity to *which trips are held out*. Student-test scores (Top-1 0.854, log loss 0.058) are better than the CV means (0.838, 0.098); with only 10 test trips we do not read that as proof of no overfitting.

#### 3.2.4 MLP (neural network)

- **A1. Why suitable?** A small network can learn smooth non-linear combinations of the features and was expected to give well-behaved probabilities.
- **A2. Strengths and limitations.** Best CV log loss (0.0871), best student-test Top-1 (0.8561), and very fast to score (0.33 s). It needs scaled inputs and an epoch count chosen without peeking at the test trips; results depend on the random initialisation (seed fixed at 42).
- **B5. Important hyperparameters (initial values).** Hidden layers (64, 32), ReLU, Adam with learning rate 0.001, L2 alpha 0.0001, batch size 2048, at most 30 epochs, trained on up to 300,000 rows. The final epoch count (15) was chosen on a held-out set of whole trips.
- **B6. Loss.** Binary cross-entropy with L2 regularisation, optimised with Adam.
- **C7 and C8. Student-test results.** Top-1 0.8561, macro-F1 0.9159, log loss 0.0594, precision 0.922, recall 0.910; confusion matrix [[222,436, 2,605], [3,235, 15,900]].
- **C9. Over- or underfitting?** This is the only model with a logged training history (`MLP_grouped_early_stopping.csv`, `MLP_loss_curve.csv`). On the held-out whole trips the validation log loss was lowest at epoch 15 (0.1015) while the training loss at that epoch was 0.0697, a gap of about 0.03. That is a modest generalisation gap, and stopping at the validation minimum limits further overfitting. Training loss was still falling, so we did not see underfitting.

#### 3.2.5 K-Nearest Neighbours (instance-based)

- **A1. Why suitable?** Candidates with similar feature values (distance, heading, speed) should have similar labels, so a neighbour vote is a natural non-parametric baseline from a different algorithm family.
- **A2. Strengths and limitations.** No training loop and almost no tuning, but prediction is very slow (448 s for the student-test rows, versus 0.15 to 2.2 s for the others) and it was the weakest model: CV Top-1 0.809, CV log loss 0.226. It is sensitive to feature scaling and its neighbour-vote probabilities are extreme (*reasoning*), which hurts log loss.
- **B5. Important hyperparameters (initial values).** k = 25, distance-weighted votes, Euclidean (Minkowski p = 2) metric, ball-tree search, inputs standardised, trained on up to 150,000 rows and scored in chunks of 4,096.
- **B6. Loss.** None: KNN has no training objective. Probabilities come from the weighted neighbour labels.
- **C7 and C8. Student-test results.** Top-1 0.8274, macro-F1 0.9004, log loss 0.1088, precision 0.916, recall 0.887; confusion matrix [[222,374, 2,667], [4,116, 15,019]].
- **C9. Over- or underfitting?** There is no training history to inspect and no training score was logged. Its CV and test scores are the weakest of the five on both ranking and log loss (CV Top-1 ranges from 0.80 to 0.83 across the three folds). This is *consistent with* a capped 150,000-row sample generalising poorly, but we cannot separate over- from underfitting here.

**Plots and curves.** Only the MLP produces an epoch-wise training history (`MLP_loss_curve.csv`, `MLP_grouped_early_stopping.csv`); no curves are drawn for the other models. The Top-1, macro-F1 and log-loss comparison plots are in section 3.3.
