# PathSnap

A workspace for matching GPS recordings to road segments. Track D (PES1UG24CS548) provides the FastAPI service, evaluator UI, offline Folium exports, deployment scripts and report skeleton. Existing Track A files in `src/` are used without modification.

**Current default: a development dummy, selecting `cand_rank == 0` for every GPS point. This is not the trained classifier or HMM.** Predictions use real trip geometry, and any displayed accuracy is computed from the supplied reference on the server after inference. Final system completion requires Tracks B/C and a successful two-laptop demonstration.

## Install

Use Python 3.11 or newer. The development verification environment uses Python 3.14 on Windows; `requirements-lock.txt` records its resolved packages. Match Track B's Python and model-library versions before loading its artifacts.

```powershell
git clone https://github.com/AaryaTedla/pes-ml-hackathon.git
cd pes-ml-hackathon
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For serving only, install `requirements.txt`. No Node build or UI package install is needed. Leaflet 1.9.4 and its license are included under `ui/vendor/leaflet/`; DM Sans and Manrope variable fonts and their OFL licenses are under `ui/vendor/fonts/`. The UI code and fonts work over the LAN without a CDN. Map tiles require internet; GPS/route layers still render if the basemap is unavailable.

## Place the data

Extract the contents of `Student folder/` from the provided archive into `data/`:

```text
data/
  train/00000001/00000001.{track,nodes,arcs,route}
  student_test/00000000/00000000.{track,nodes,arcs,route}
  student_test/00000007/00000007.{track,nodes,arcs,route}
  ...
```

Each trip directory must be named exactly like its files. IDs remain strings, including leading zeroes. `.track` is `lon lat time_s`; `.nodes` is `lon lat`; `.arcs` is `from_node to_node`. `.route` is optional for prediction and is used only by server-side scoring or explicit offline reference exports. Raw data is gitignored.

The served trip root is configurable; it is never limited to hardcoded IDs. Only immediate trip directories with `.track`, `.nodes` and `.arcs` files are listed. The supplied student-test partition has 10 trips. At evaluation, set the root to the evaluator-provided directory and restart; do not train, tune or choose settings using evaluator data.

## Laptop 1: run the API

From the repository root:

```powershell
$env:PATHSNAP_TRIPS_DIR = "data/student_test"
$env:PATHSNAP_MODE = "nearest"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or run `powershell -ExecutionPolicy Bypass -File api/start.ps1 -TripsDir "data/student_test" -Mode nearest`. This prints available IPv4 addresses, runs in the current terminal and leaves the live request logs visible. Stop with Ctrl+C.

Open **http://127.0.0.1:8000/ui/** for the workspace, or **http://127.0.0.1:8000/docs** for interactive API documentation. The UI loads its library from `GET /trips`; nothing is prefilled with predictions or metrics. Select a trip and press **Run prediction**. Toggle GPS/predicted layers, fit the route, inspect timing details or export the returned JSON.

Configuration is read at startup. Set environment variables in PowerShell; `.env.example` documents the names but is not automatically loaded.

| Variable | Default | Purpose |
|---|---|---|
| `PATHSNAP_TRIPS_DIR` | `data/student_test` | Parent directory of trip folders; absolute or repository-relative |
| `PATHSNAP_MODE` | `nearest` | `nearest`, `classifier`, or `hmm` |
| `PATHSNAP_ARTIFACTS_DIR` | `artifacts` | Model, optional scaler, feature order and decoder config |
| `PATHSNAP_CORS_ORIGINS` | `*` | Comma-separated UI origins; no credentialed requests |

## Laptop 2: connect over Wi-Fi

1. Put both laptops on the same Wi-Fi. On laptop 1, run `Get-NetIPAddress -AddressFamily IPv4` or `ipconfig` and use the **Wi-Fi IPv4**, not WSL/VirtualBox addresses and not `0.0.0.0`.
2. On laptop 2, open `http://<laptop-1-Wi-Fi-IPv4>:8000/ui/`. The HTML/JS executes on laptop 2 and calls laptop 1; no Python or dataset is needed on laptop 2 for this option.
3. To serve UI files independently on laptop 2, copy `ui/`, run `python -m http.server 8501 --bind 0.0.0.0 --directory ui`, then open `http://127.0.0.1:8501`. Click the API connection control and enter `http://<laptop-1-Wi-Fi-IPv4>:8000`. The address is saved in that browser. A shareable form is `http://127.0.0.1:8501/?api=http://<laptop-1-Wi-Fi-IPv4>:8000`.
4. From laptop 2, run `Test-NetConnection <laptop-1-Wi-Fi-IPv4> -Port 8000`, then the smoke test below. Keep laptop 1's API terminal visible during the demonstration.

```powershell
powershell -ExecutionPolicy Bypass -File api/test_lan.ps1 -ApiUrl "http://<laptop-1-Wi-Fi-IPv4>:8000"
```

If needed, run this **on laptop 1 in an Administrator PowerShell**, matching `Public` or `Private` to the active Wi-Fi category shown by `Get-NetConnectionProfile`:

```powershell
New-NetFirewallRule -DisplayName "PathSnap demo TCP 8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 -RemoteAddress LocalSubnet -Profile Public
```

Remove the temporary rule after the demo:

```powershell
Remove-NetFirewallRule -DisplayName "PathSnap demo TCP 8000"
```

The firewall stays enabled. Campus Wi-Fi may isolate clients; if the TCP test fails despite a scoped rule and a listening API, use a shared phone hotspot. A local test against this laptop's Wi-Fi address does **not** prove laptop 2 can connect. The second-device test must be recorded separately. For a local-only session use `--host 127.0.0.1`; this service is intended for the evaluation LAN, not internet exposure.

## API contract

`GET /health` returns readiness, mode, pipeline label and trip count. Missing data returns 503 with a useful message. `GET /trips` returns `{"trips": ["00000000", "..."]}`. `POST /predict` accepts exactly:

```json
{"trip_id": "00000000"}
```

The response includes:

| Field | Meaning |
|---|---|
| `trip_id`, `request_id`, `mode`, `pipeline` | Identity and an honest label of the active prediction path |
| `n_points`, `n_matched` | GPS count and points assigned an arc |
| `gps` | All observations as `[lat, lon]` |
| `predicted_arcs` | One local arc row index per point; `-1` means no candidate |
| `predicted_route` | Unique predicted arc geometries, `[[lat,lon],[lat,lon]]` per segment; not an invented line between disconnected arcs |
| `metrics` | Aggregate server-side scoring, or `null` when no reference exists |
| `timing_ms` | Feature, prediction, scoring and server total times |

The per-point arc array retains temporal ordering. The map segment list is geometry, not a claimed connected navigation path. Browser rendering and network time are outside server timings.

Scoring first loads `.route` **after** `Predictor.predict_trip` returns, derives aligned labels with `labels.assign_route_arcs`, then computes accuracy. No reference route, point labels, reference geometry, complete road network or input features are returned. Offline exports and raw data directories are not mounted in the web server.

Accuracy is the fraction of all points whose predicted arc equals the aligned reference arc; unmatched points count as incorrect. The alignment is a reference proxy, not independently verified physical truth. Reference-covered accuracy restricts to aligned distances ≤60 m. Route continuity measures connected changes between successive different arc IDs; unmatched transitions are disconnected, and runs with no matched points are undefined. Exact-window accuracy is the fraction of non-overlapping 50-point windows entirely correct; it is `null` for shorter trips. Undefined metrics serialize as `null`, never nonstandard JSON NaN.

Every API request produces a terminal line with request ID, endpoint, status, trip, point count, elapsed time and accuracy (`n/a` if unavailable). Failed requests are logged too. Retain these real logs for the report.

## Connect Tracks A, B and C

The integration checklist and exact feature order are in [api/HANDOFF.md](api/HANDOFF.md). Confirm that contract with 497, 496 and 549. No teammate messages have been sent by this repository.

- **Track A (497):** `prepare_trip(trip_dir)` is the only feature-building entry. It returns `(trip, candidate_features, point_features)` without a route. Keep `src/` ownership with that track.
- **Track B (496):** place a trusted `model.joblib` (classifier or an entire sklearn Pipeline) and, only when needed, `scaler.joblib` in `artifacts/`. Supply `feature_columns.json`, training package versions and class semantics. Avoid applying a separate scaler to a Pipeline that already preprocesses. The positive class is `1`; its probability column is located using `classes_`, not assumed to be column 1.
- **Track C (549):** provide `src/hmm.py` exposing `decode(trip, candidates_with_probs[, config])` and `artifacts/hmm_config.json`. Candidates have a `prob` column. Output is one integer arc ID per GPS point, with `-1` for unmatched points. Validate the final signature with Track C.

Activate the real classifier with `$env:PATHSNAP_MODE = "classifier"`, or classifier plus decoder with `$env:PATHSNAP_MODE = "hmm"`, then restart the API. Missing requested artifacts or incompatible feature metadata fail startup. There is no automatic fallback from a requested real mode. Adapter tests exercise the contract with test doubles; they do not establish real model quality or real HMM integration.

## Generate mandatory Folium maps

Reuse the existing `src/viz.py` through the Track D offline wrapper. It accepts predicted arcs, uses the trip's real GPS and nearby network, adds layers/legend and fits the extent.

```powershell
# Prediction-only export; generates nearest-arc predictions in the default mode.
.\.venv\Scripts\python.exe -m api.export_map 00000000 --out results/track_d/map_prediction.html

# Explicit reference comparison for offline report use only.
.\.venv\Scripts\python.exe -m api.export_map 00000000 --include-reference --out results/track_d/map_report.html

# Use a JSON result downloaded from the UI or produced by api.verify.
.\.venv\Scripts\python.exe -m api.export_map 00000000 --prediction results/track_d/prediction_00000000.json --include-reference --out results/track_d/map_report.html
```

Open the HTML locally and take a representative screenshot for report section 6. Do not copy reference-bearing report maps into `ui/`. The wrapper strips references by default and filters unmatched `-1` IDs before calling the existing generator. Online tiles and the existing Folium CDN assets need internet; only the main evaluator UI vendors Leaflet locally.

## Verification and evidence

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_track_d.py -q
.\.venv\Scripts\python.exe tests/test_no_leakage.py
node --check ui/app.js

# Optional frontend logic tests (Node 18+; no UI build required).
npm --prefix tests ci
npm --prefix tests test

# With the API running; runs every trip returned by /trips.
.\.venv\Scripts\python.exe -m api.verify --expect-mode nearest
```

`api.verify` saves actual prediction JSON, `smoke_summary.csv` and run metadata under gitignored `results/track_d/`. Repeat with `--expect-mode hmm` after receiving real artifacts, and use `--api-url` from laptop 2 to exercise the network. These smoke tests verify serving, response sizes and absence of reference fields; they do not replace model evaluation or a visual browser check.

The completed development checks and their limits are recorded in [report/track_d_verification.md](report/track_d_verification.md), with the actual ten-trip dummy CSV and matching API log lines under `report/evidence/`.

Use [report/technical_report_skeleton.md](report/technical_report_skeleton.md) for the consolidated report. Record screenshots of the UI, offline Folium comparison and receiving API terminal. All model-result placeholders must be filled from the real experiments. AI assistance is disclosed in the skeleton; the team must review explanations and defend the implementation.

## Repository layout

```text
api/        service, prediction adapter, scoring, Folium wrapper, LAN scripts, HTTP verifier
ui/         buildless HTML/CSS/JS workspace and vendored Leaflet
src/        existing Track A pipeline and visualisation (not changed by Track D)
tests/      existing leakage check + Track D contract tests
artifacts/  Track B/C handoffs when supplied
report/     existing EDA draft + consolidated report skeleton and Track D evidence notes
results/    existing team outputs; local Track D runs in results/track_d/
```

Commit source and reviewed result evidence with your own Git identity. Raw data, local materials, virtual environments and `.env` are ignored. The final public GitHub repository and consolidated report PDF remain team submission tasks; preparing a skeleton is not a claim that final experimental results or the PDF exist.
