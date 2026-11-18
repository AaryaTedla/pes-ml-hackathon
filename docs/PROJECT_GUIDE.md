# PathSnap project guide

Read this first. It explains what the project does, how the pieces fit together, and where everything lives.

## 1. The problem in one paragraph

A GPS receiver in a moving car gives noisy points, so on a map the points can land on the wrong street. Given a trip's GPS points and a map of
the roads (a graph), we have to say which road segment the car was on at each point and rebuild a connected route.
We do it in two steps. **Step 1:** for each GPS point, a machine-learning classifier scores the nearby roads. **Step 2:** a Hidden Markov Model
(HMM) with Viterbi decoding chooses the best sequence of roads over the whole trip, so the route stays connected from one road to the next.

## 2. The data

Each trip is a folder with four plain-text files:

| File | Content |
|---|---|
| `.track` | GPS points: longitude, latitude, time in seconds (about 1 per second) |
| `.nodes` | road intersections: longitude, latitude (the row number is the node id) |
| `.arcs` | road segments: from-node, to-node (the row number is the arc id; roads are directed) |
| `.route` | the reference route: the ordered list of arcs the vehicle travelled |

100 trips in total: 80 for training, 10 for student testing, and 10 hidden evaluator trips we never used for development.
Every trip has **its own** road network. There are **no labels per GPS point**, only the route, so we derive the labels ourselves.

## 3. The pipeline, step by step (read in this order)

| # | Step | What it does | Files |
|---|---|---|---|
| 1 | Load and project | read a trip; convert degrees to metres (a UTM zone per trip) | `src/data_loader.py`, `src/geo.py` |
| 2 | Candidates | for each GPS point, find road arcs within 60 m (at most the 30 nearest) using a spatial index | `src/candidates.py` |
| 3 | Labels | align GPS points to the reference route with a forward-only dynamic programme; a candidate is positive if it is the aligned arc | `src/labels.py` |
| 4 | Features | 18 numbers per (point, candidate): distance, heading, speed, neighbours, geometry, gap | `src/features.py`, `src/build_features.py`, `docs/features.md` |
| 5 | EDA | statistics, plots and Folium maps of the raw data | `src/eda.py`, `src/viz.py` -> `results/eda/`, `results/figures/`, `results/maps/` |
| 6 | Train and compare models | five models (logistic regression, LightGBM, random forest, MLP, KNN), 3-fold cross-validation grouped by trip | `notebooks/`, `src/train_models.py`, `docs/training.md` -> `results/models/model_comparison/` |
| 7 | Tune | Optuna search on the winning random forest | `results/models/rf_tuning/` -> `artifacts/model.joblib` |
| 8 | HMM / Viterbi | two decoders: multi-hop (used by the API) and fixed-transition; settings tuned on a set of training trips | `src/hmm.py`, `src/hmm_fixed.py`, `src/eval_hmm.py`, `docs/hmm.md` -> `results/hmm/final/` |
| 9 | Serve | API runs features -> model -> HMM, then scores against the reference on the server only | `api/`, `src/pipeline.py` |
| 10 | Show | browser page lists the trips, calls the API, draws the predicted route and the metrics | `ui/` |

## 4. Why each design choice (the answers you will be asked for)

- **Why project to metres?** Degrees are not distances, and the trips are spread across the world.
- **Why candidates?** Scoring every road in a city is wasteful; the true road is almost always within 60 m of the GPS point (the candidate miss rate on the test trips is 0.13%).
- **Why heading features?** About 93 to 95% of roads have a twin in the opposite direction at the same position, so distance alone cannot tell them apart. Choosing the nearest road is right only about 51% of the time.
- **Why the HMM?** The classifier looks at one point at a time and can jump between roads. The HMM rewards choices that are connected in the road graph and picks the best whole sequence, not the best single point.
- **What is the hidden state?** The candidate road at each GPS point. **Emission** = the classifier's probability for that road. **Transition** = how plausible it is to go from one road to the next (same road, adjacent road, a few roads away, a U-turn, a jump).
- **Why two decoders?** Multi-hop allows travel through several short roads between two points and is more accurate. Fixed-transition only allows adjacent roads and gives more continuous routes. We use multi-hop in the API.
- **How was leakage prevented?** Features never read the route (`tests/test_no_leakage.py`); folds are grouped by trip; the evaluator trips were not used for any choice; the API scores against the reference only after predicting.

## 5. Key results (all from files in `results/`, listed in `report/numbers_sheet.md`)

- Models (3-fold CV, Top-1 per point): random forest 0.838, LightGBM 0.833, MLP 0.830, logistic regression 0.827, KNN 0.809; the nearest-road baseline is about 0.51. The top three are close, so the margin is narrow.
- Random forest was chosen by a rule fixed beforehand (highest CV Top-1). Only 6 of 25 planned Optuna trials ran, and tuning barely changed the result.
- With the HMM: accuracy rises from 86.5% to 91.8% on the 10 student-test trips and from 87.8% to 94.1% on 22 unseen training trips.
- Final evaluation on the 10 hidden evaluator trips (run once, after everything was frozen): mean accuracy 0.891 for the classifier alone and 0.938 with the HMM (better on 9 trips, equal on 1, worse on none); all values are in `results/hmm/final/evaluator_trips_results.csv`.

## 6. Folder map

| Folder | Contents |
|---|---|
| `docs/` | this guide and notes on features, training and the HMM |
| `src/` | all pipeline code (steps 1 to 8), evaluation and report tools |
| `api/` | FastAPI service: `main.py` (endpoints), `predictor.py` (model + HMM), `scoring.py` (server-only scoring), `config.py` |
| `ui/` | the browser page (HTML, CSS, JavaScript, local Leaflet) |
| `notebooks/` | the Kaggle notebooks that trained the models: `train_compare_all_models` (all five), and `train_compare_fast_models` / `train_compare_slow_models` (the same comparison split in two so they could run in parallel) |
| `artifacts/` | `model.joblib` (trained model), `feature_columns.json`, `hmm_config.json` and `hmm_fixed_config.json` (tuned decoder settings) |
| `results/` | `features_*.parquet` and `points_*.parquet` (data tables), `eda/`, `figures/`, `maps/`, `models/`, `hmm/` |
| `results/models/` | `model_comparison/` (the 5-model run), `rf_tuning/` (the Optuna run that produced the model), `rf_tuning_first_attempt/` (an earlier interrupted run), `kaggle_run_outputs.zip` (the raw Kaggle output, including the fitted models) |
| `results/hmm/` | `final/` (decoders with the trained model), `stand_in/` (earlier experiments with a hand-made scoring rule, kept as a baseline) |
| `report/` | report sections, `FULL_REPORT_DRAFT.md/.pdf`, `numbers_sheet.md` |
| `scripts/` | `run_api.bat`, `check_all_trips.py`, `integrate_model.py`, `open_firewall.ps1` |
| `tests/` | automated checks (leakage, HMM, training, API/UI contract) |

## 7. How to run things

```text
# install
pip install -r requirements.txt

# rebuild the feature tables (needs the raw data in data/)
python src/build_features.py

# run the tests
python -m pytest tests -q

# start the API with the trained model and the HMM
set PATHSNAP_MODE=hmm
set PATHSNAP_TRIPS_DIR=<folder with the trips to serve>      (default: data/student_test)
scripts\run_api.bat      (in PowerShell use: $env:PATHSNAP_MODE="hmm"; $env:PATHSNAP_TRIPS_DIR="<folder>")
# then open http://localhost:8000/ui/ and check all trips with:
python scripts/check_all_trips.py

# re-run the decoder comparison with the trained model
python src/compare_decoders.py --prob model

# rebuild the report tables and the combined draft
python src/make_report_tables.py
python src/build_report_draft.py --pdf
```

## 8. Words you will hear

- **Arc**: one directed road segment. **Candidate**: an arc near a GPS point. **Reference route**: the supplied list of arcs the vehicle travelled.
- **Top-1 accuracy**: for each GPS point, is the highest-scored candidate the correct arc?
- **Route continuity**: of the times the predicted arc changes, how often is the new arc directly connected to the previous one?
- **Cross-validation grouped by trip**: folds never split a trip, so the model is always tested on trips it has not seen.
- **TUNE / VAL / student_test**: 22 training trips used to choose the HMM settings, 22 other training trips to check them, and the 10 student-test trips as a final check.

## 9. Naming notes

- **"mock" and "stand_in" mean the same thing:** the hand-made scoring rule (softmax of distance and heading) we used to build and test the HMM before the trained model existed. Files with `mock` in the name, and the folder `results/hmm/stand_in/`, hold those earlier results. The command-line option `--prob mock` selects that rule and `--prob model` selects the trained model.
- **`report/section_3_4_models.md`** covers both section 3 (models) and section 4 (Bayesian optimisation); the report builder splits it.
- **Requirements files:** `requirements.txt` (exact versions to run everything), `requirements-dev.txt` (adds the test tools), `requirements-training.txt` (the Kaggle training environment), `requirements-figures.txt` (optional, only for automatic map screenshots) and `requirements-lock.txt` (the full list of package versions from the API developer's machine, for reference).
- **`results/hmm/final/`** is what the report uses. **`results/hmm/stand_in/`** is the earlier baseline.
