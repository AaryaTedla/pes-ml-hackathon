### 3.1 Overall model comparison

Cross-validation: 3-fold GroupKFold by trip on 40 whole training trips (seed 42). Student-test: the 10 student_test trips, all 244,176 candidate rows. Top-1 is per-point (highest-probability candidate vs the derived reference arc). Macro-F1 and log loss are binary candidate metrics at threshold 0.5. Route continuity is the classifier alone (no HMM) on student_test, using the shared metric in `src/evaluate.py`; it is not an HMM result. Source: `results/models/model_comparison/comparison.csv` and `results/models/route_continuity_by_model.csv`.

| Model | Algorithm family | Key configuration | CV macro-F1 | Top-1 (CV / student-test) | Route continuity (student-test, classifier only) | Student-test macro-F1 | Log loss (CV / student-test) | Inference time, student-test (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Linear / probabilistic | StandardScaler, max_iter=200, binary log-loss | 0.8784 | 0.8269 / 0.8506 | 0.7383 | 0.8995 | 0.1202 / 0.0780 | 0.15 |
| LightGBM | Boosting | 300 trees, learning_rate=0.1, num_leaves=63, binary log-loss | 0.9067 | 0.8327 / 0.8488 | 0.6821 | 0.9163 | 0.1447 / 0.0999 | 1.38 |
| Random Forest | Bagged trees (ensemble) | 100 trees, max_depth=18, min_samples_leaf=20, Gini; capped at 300k fitting rows | 0.9066 | 0.8377 / 0.8535 | 0.7449 | 0.9178 | 0.0977 / 0.0575 | 2.22 |
| MLP | Neural network | hidden (64,32), batch 2048, up to 30 epochs, Adam; capped at 300k fitting rows | 0.9012 | 0.8301 / 0.8561 | 0.7757 | 0.9159 | 0.0871 / 0.0594 | 0.33 |
| KNN | Instance-based | k=25, distance-weighted, ball-tree, StandardScaler; capped at 150k fitting rows | 0.8847 | 0.8088 / 0.8274 | 0.7624 | 0.9004 | 0.2261 / 0.1088 | 448.41 |

Nearest-arc baseline Top-1: 0.5123 (CV mean), 0.5081 (student-test). Per-fold Top-1 standard deviation is 0.012 to 0.019, larger than the gaps between the top three models, so the ranking is narrow. Inference time covers candidate scoring only (not feature generation or HMM decoding); KNN's cheap fit hides its slow prediction.
