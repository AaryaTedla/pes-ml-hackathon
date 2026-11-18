# PathSnap: Distributed Map Matching and Route Reconstruction

**Consolidated technical report - working skeleton, not a final submission**

Team: PES1UG24CS497, PES1UG24CS496, PES1UG24CS549, PES1UG24CS548
Repository: https://github.com/AaryaTedla/pes-ml-hackathon
Submission date: [fill] | Final commit: [fill] | Data partitions and seeds: [fill]

Replace all bracketed prompts with reviewed evidence from real runs. Export one consolidated PDF after the team completes and reviews it. Never use the dummy baseline as a trained-model result. Preserve the required numbering below from Deliverables.pdf.

## 1. Problem and system overview

[Explain noisy GPS, local directed road networks, candidate classification and route decoding in the team's own words. Identify each contributor and link the code used.]

Architecture: laptop 2 browser -> FastAPI on laptop 1 -> reference-free feature preparation -> classifier -> HMM -> server-only scoring -> GPS, predicted route and aggregate metrics -> browser.

## 2.2 Dataset Understanding and Exploratory Data Analysis

Owner: 497. Starting material: `report/section_2_2_eda.md`. Recheck every number against the actual saved outputs.

- Which partition was used for each analysis? State train, student-test or evaluator-test explicitly.
- How was data loaded, validated, projected and preprocessed? How were missing values and timestamp gaps handled?
- What exploration patterns affected implementation choices?
- Which features were extracted, in what order, and why were they selected?
- Explain point-label derivation from the supplied ordered route, reference limitations and candidate miss rate.

| Required evidence | File / figure / partition |
|---|---|
| Missing-value analysis | [fill] |
| GPS trajectory plots | [fill] |
| Road-network plots | [fill] |
| Folium geographical map | [fill] |
| Sampling gaps, candidate miss rate, reference limitations | [fill] |

**Figure placeholder 2.2-A:** Insert actual trajectory/network/EDA figures with partition and caption.

## 3. Model Experimentation and Comparison

Owner: 496. Include every model actually experimented with, at least five from the team's real experiments.

### 3.1 Overall Model Comparison

| Model | Algorithm family | Key configuration | CV Macro F1 | Student-test Top-1 | Route continuity | Student-test Macro F1 | Log Loss | Inference time |
|---|---|---|---|---|---|---|---|---|
| [Model 1] | [fill] | [fill] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| [Model 2] | [fill] | [fill] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| [Model 3] | [fill] | [fill] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| [Model 4] | [fill] | [fill] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| [Model 5] | [fill] | [fill] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |

[Define metric aggregation, candidate-level versus point-level metrics, grouping, seeds, hardware and timing units. Explain N/A wherever a metric is inapplicable. Link raw comparison CSV and experiment configuration.]

### 3.2 Compulsory Questions for Every Model

Repeat the following complete response block for **each** actual model, including additional models beyond five:

#### 3.2.1 [Actual model name]

**A. Model understanding:** Why suitable for this problem? What were its strengths and limitations in the team's experiments?

**B. Training and configuration:** Important hyperparameters and initial values? Loss function or optimisation objective? Seed, grouped folds and preprocessing fit partition?

**C. Experimental results:** Student-testing performance, observation-level Top-1, Macro F1 and Log Loss (where applicable)? Evidence of overfitting or underfitting? Explain actual observations rather than textbook descriptions.

Evidence: [source configuration, metrics CSV and figures].

#### 3.2.2 [Actual model name]

[Answer A, B and C above and cite real run evidence.]

#### 3.2.3 [Actual model name]

[Answer A, B and C above and cite real run evidence.]

#### 3.2.4 [Actual model name]

[Answer A, B and C above and cite real run evidence.]

#### 3.2.5 [Actual model name]

[Answer A, B and C above and cite real run evidence.]

### 3.3 Model Visualisations

- **Figure placeholder 3.3-A:** Cross-validation Macro F1 across all models.
- **Figure placeholder 3.3-B:** Student-testing Top-1 and Macro F1 comparison.
- **Figure placeholder 3.3-C:** Log Loss where applicable.
- **Figure placeholder 3.3-D:** Optional confusion matrices and per-model diagnostics.
- **Figure placeholder 3.3-E:** Training/validation losses only for models with actual epoch histories.

[Explain selection using actual comparison plots. Give source result files for every graph. Do not invent training curves for tree models or other algorithms without epoch histories.]

## 4. Bayesian Hyperparameter Optimisation

Owner: 496.

1. Which hyperparameters were searched?
2. What ranges or distributions were used?
3. Which objective was optimised, with what grouped validation design?
4. How many trials completed? Which seeds were used?
5. What were the best parameter values?
6. Did tuning improve the model? Support the answer using the same partitions and metrics.

| Configuration | CV Macro F1 | Student-test Top-1 | Student-test Macro F1 | Log Loss | Inference time |
|---|---|---|---|---|---|
| Untuned [name] | [pending] | [pending] | [pending] | [pending] | [pending] |
| Tuned [name] | [pending] | [pending] | [pending] | [pending] | [pending] |

Evidence: [search-space configuration, complete trial CSV, best parameters and plots].

## 5. HMM and Viterbi Route Reconstruction

Owner: 549.

1. Define the hidden states and their relation to candidate arc IDs.
2. Explain emission probabilities from the chosen classifier.
3. Explain graph-aware transition probabilities and their units.
4. Explain missing-candidate handling, timestamp gaps and resets.
5. Explain disconnected observations and any inserted connector arcs.
6. Did decoding improve the route? Include improvements and failure cases using actual trips.

| Pipeline / partition | Point accuracy | Exact 50-point windows | Route continuity | Inference time |
|---|---|---|---|---|
| Classifier only / [partition] | [pending] | [pending] | [pending] | [pending] |
| Classifier + HMM / [same partition] | [pending] | [pending] | [pending] | [pending] |

Evidence: [decoder code, config, actual comparison CSV, selected successful and unsuccessful trips].

## 6. Geographical Visualisation Using Folium

Owners: 497 (generator), 548 (offline export wrapper).

Use `python -m api.export_map <trip_id> --prediction <real-result.json> --include-reference --out <report-map.html>` with the final model/HMM output. See README for commands.

- Original GPS observations, relevant real road-network geometry, reference route where available, and predicted route.
- Distinguishable colours and markers, clear labels/legend, suitable extent and zoom, and layer controls.
- State the trip ID, partition, prediction method and source JSON. Commit HTML or reproducible generation code.

**Figure placeholder 6-A:** Actual offline Folium screenshot with all required layers.
Caption: [trip, partition, method, observed behaviour and source HTML].

Reference-bearing maps are offline report artifacts. The evaluator UI receives GPS, predictions and aggregate metrics only.

## 7. API, UI and Deployment

Owner: 548. Implemented source: `api/`, `ui/`, README and requirements files.

Describe the final demonstrated mode and artifacts, not merely the dummy development mode. `GET /trips` lists configured data; `POST /predict {trip_id}` uses `prepare_trip` without reference access, then computes aggregate scoring on laptop 1. UI trip IDs and all plotted predictions come from the API. Each request logs trip ID, points, elapsed time and accuracy.

| Deployment evidence | Actual value / source |
|---|---|
| Final commit, Python and package versions | [fill] |
| Laptop 1 host / port / prediction mode | [fill] |
| Laptop 2 UI origin / connection method | [fill] |
| Actual second-device TCP and prediction test | [pending laptop 2] |
| All 10 final-mode trips verified | [pending real artifacts] |
| Reference-leakage check | [test command/output] |
| Model and HMM handoffs | [artifact provenance/config] |

**Figure placeholder 7-A:** Actual laptop 2 UI showing a completed trip, GPS, predicted route and metrics.
**Figure placeholder 7-B:** Actual receiving API terminal with the matching request ID, trip, point count, runtime and accuracy.
**Figure placeholder 7-C:** Optional API response and successful network test.

[Explain firewall scope, shared Wi-Fi, configurable evaluation data directory and instructions needed to reproduce the demonstration. Link saved smoke-test evidence. State any incomplete test honestly.]

## 8. Academic Integrity and AI Assistance

Draft disclosure to review: OpenAI Codex assisted PES1UG24CS548 with Track D implementation, interface design, tests, setup documentation and this report skeleton. The team must review and understand the code. No trained-model metrics or experimental analysis are supplied by the skeleton. Add the actual assistance used by all other contributors and the independent checks performed.

[Confirm all reported numbers trace to actual runs, train-only preprocessing, trip-grouped folds, untouched evaluator data during development and no inter-team sharing. Each team member should be able to explain their implementation and results.]

## Submission checklist

- [ ] All source, training/evaluation scripts, final model/config and reviewed evidence committed and pushed.
- [ ] Public repository is accessible to evaluators; no secrets or private credentials.
- [ ] Five or more real models documented, all required graphs and Optuna results included.
- [ ] Final classifier + HMM evaluated and real laptop 2 demo verified.
- [ ] Folium screenshot, UI screenshot and receiving-terminal screenshot inserted.
- [ ] Placeholders replaced, claims reviewed, AI assistance disclosed.
- [ ] One consolidated PDF exported and visually reviewed; submitted alongside repository URL.
