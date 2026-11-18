# Track D verification record

Owner: PES1UG24CS548. Development verification: **18 November 2026**.

This records the nearest-arc development implementation. It is not a final trained-model evaluation and is not proof of a two-device LAN demonstration.

## Checks completed

| Check | Observed result |
|---|---|
| `python -m pytest tests/test_track_d.py -q` | 22 passed |
| `npm --prefix tests test` | 5 DOM-level UI tests passed |
| `node --check ui/app.js` | Passed |
| `python tests/test_no_leakage.py` | All three existing train/student-test checks passed |
| `python -m api.verify --expect-mode nearest` | 10/10 configured student-test trips passed; 19,160 GPS observations |
| `api/test_lan.ps1` against the laptop's Wi-Fi IPv4 | Passed from the same laptop; this does not test another device or incoming firewall access |
| UI entry point, redirect, CSS, JS, fonts and Leaflet HTTP checks | All returned HTTP 200 after redirect |
| Offline Folium prediction and reference-comparison exports | Generated from trip 00000000 and actual nearest-arc predictions |
| No-match Folium export | Generated successfully in the contract test |
| `python -m pip check` | No broken requirements |
| Track A ownership | No edits to `src/` |

Backend tests exercise actual temporary raw trip files, the real feature/scoring code, reference removal/replacement, request validation, API reference-field isolation, missing data, CORS, log fields, UI asset serving, invalid decoder outputs, two-/three-argument decoder handoffs, probability class ordering, and a temporary serialized sklearn preprocessing Pipeline. The temporary synthetic model is a test fixture, not a trained PathSnap artifact.

UI tests exercise the shipped application code in jsdom with a map test double and controlled API responses. They verify trip selection, request bodies, result display, zero/null metrics, layer toggles, state reset, network/API errors and text-safe rendering. They are not screenshot or browser rendering tests.

## Saved real run evidence

- `evidence/dummy_http_smoke.csv`: actual per-trip counts, timings, accuracy, continuity and request IDs.
- `evidence/dummy_api_requests.log`: matching live API terminal lines for those ten requests.
- `evidence/dummy_run.json`: run timestamp, mode and local API URL.
- Local `results/track_d/`: full prediction responses and Folium HTML maps; regenerate using README commands. This directory is ignored to keep local run artifacts out of normal commits.

The dummy chooses `cand_rank == 0`, not a learned or heading-weighted score. Its recorded accuracy is a development check of scoring and serving. Do not put these values into the five-trained-model comparison table.

## Verification limits and final handoff

- The computer-use tools reported no available browser. Visual layout, real Leaflet canvas interaction, responsive rendering and screenshots have **not** been verified in a browser. Open `/ui/` and check desktop/mobile layouts, run a trip, pan/zoom, toggle layers, export a result and capture the real UI screenshot before the demonstration.
- Only this laptop was available. Run `api/test_lan.ps1` from laptop 2, inspect the matching log on laptop 1, and record the actual result. The local Wi-Fi profile was Public with the firewall enabled; README documents a port-8000 rule restricted to the local subnet if needed. No firewall rules were changed.
- Real model/preprocessing and HMM artifacts were not available. Confirm `api/HANDOFF.md`, match Track B package versions, restart in `classifier` then `hmm` mode and rerun all trips. No full-model integration or final accuracy is claimed.
- Two dependency notices appeared without test failures: the installed Starlette version deprecates its httpx-based TestClient, and Folium's optional CartoDB alternate layer warns about an API key. The report map's default layer is OpenStreetMap. Review dependencies with the final team environment; neither notice justifies changing another owner's `src/` files.
- Final experimental results, report screenshots, consolidated PDF and public-repository submission remain team completion tasks.
