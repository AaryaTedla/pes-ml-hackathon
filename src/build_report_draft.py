"""Assemble the section files into one DRAFT report (Markdown -> HTML -> optional PDF via Edge/Chrome).

    python src/build_report_draft.py [--pdf]

Reads report/section_*.md, report/section_3_1_table.md and figures under results/. Writes report/FULL_REPORT_DRAFT.{md,html,pdf}.
It only concatenates what the team has written plus figures from real runs; the placeholders it leaves are marked [NEEDED: ...].
"""
import os, sys, re, argparse, subprocess, glob
import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REP = os.path.join(ROOT, "report")


def read(name):
    return open(os.path.join(REP, name), encoding="utf-8").read()


def img(path, caption):
    return f"\n![{caption}](../{path})\n\n*{caption}*\n"


HEADER = """# PathSnap: Distributed Map-Matching and Route Reconstruction

**Technical report (DRAFT for team review, not final)**

Team: PES1UG24CS497, PES1UG24CS496, PES1UG24CS549, PES1UG24CS548
Repository: https://github.com/AaryaTedla/pes-ml-hackathon

> The bracketed entries in section 8 (AI-use disclosure) must be completed by each member.
> Every number comes from a committed result file; see `report/numbers_sheet.md`.

## 1. Problem and system overview

GPS positions are noisy, so a vehicle on one road can appear on a parallel street. Given a trip's GPS points and its own road network, PathSnap
decides which road segment (arc) the vehicle was on at each point and reconstructs a connected route. A classifier scores each candidate arc near a
GPS point using 18 features (distance, heading, speed, neighbouring points, road geometry). A Hidden Markov Model with Viterbi decoding then chooses
the best consistent sequence of arcs using the road graph. The final system is a FastAPI server (laptop 1) that runs the full pipeline and scores the
result against the reference route on the server only, plus a browser UI (laptop 2) that shows the predicted route and metrics.

Architecture: laptop 2 browser -> FastAPI on laptop 1 -> reference-free feature preparation -> classifier (tuned Random Forest) -> HMM (multi-hop Viterbi)
-> server-only scoring -> GPS, predicted route and aggregate metrics -> browser.
"""

SEC7 = """
## 7. API, UI and Deployment

Implemented in `api/` (FastAPI) and `ui/` (browser workspace served by the API at `/ui/`). `GET /trips` lists the configured trips; `POST /predict {trip_id}`
builds features with `prepare_trip` (which never opens the `.route` file), runs the classifier and the HMM, and only then computes aggregate scoring against the
reference on laptop 1. The response carries GPS points, the predicted route and aggregate metrics, never the reference route or labels. Every request is logged on the API
terminal with a request id, trip id, number of points, elapsed time and accuracy. `tests/test_no_leakage.py` shows that the features are identical with and without the
route file, and a fresh clone of the public repository ran all 10 student_test trips in `hmm` mode (pipeline `classifier + HMM/Viterbi`).

| Deployment evidence | Value / source |
|---|---|
| Final commit | see `git log` of the submitted repository |
| Prediction modes | `nearest` (development baseline), `classifier`, `hmm` (final); `hmm` refuses to start without `artifacts/model.joblib` |
| All 10 student_test trips in `hmm` mode | `results/hmm/final/api_hmm_check_model.csv`: mean accuracy 0.865 (classifier alone) to 0.917 (with HMM), mean continuity 0.906 |
| Laptop-to-laptop connection (development mode) | A second laptop on the same Wi-Fi called the API on laptop 1 (port 8000); the requests appear in laptop 1's API log with the second laptop's client IP and matching request ids |
| Final-mode demonstration on laptop 2 | **[NEEDED: repeat with `PATHSNAP_MODE=hmm` and capture the screenshots below]** |

### 7.1 Final evaluation on the 10 hidden evaluator trips

After all models and settings were frozen, the final pipeline (tuned Random Forest + multi-hop HMM, `PATHSNAP_MODE=hmm`) was run once through the live API on the 10 evaluator
trips with `scripts/check_all_trips.py`. Nothing was tuned or changed after seeing these numbers. Values are rounded to three decimals as printed by the script.

| Trip | GPS points | Accuracy, classifier alone | Accuracy, final (with HMM) | Route continuity |
|---|---:|---:|---:|---:|
| 00000015 | 6,854 | 0.671 | 0.897 | 0.837 |
| 00000017 | 1,346 | 0.882 | 0.914 | 0.742 |
| 00000023 | 458 | 0.941 | 0.941 | 0.887 |
| 00000030 | 654 | 0.775 | 0.839 | 0.810 |
| 00000049 | 2,642 | 0.981 | 0.991 | 0.954 |
| 00000059 | 1,031 | 0.919 | 0.942 | 0.813 |
| 00000067 | 916 | 0.938 | 0.957 | 0.962 |
| 00000073 | 1,144 | 0.958 | 0.974 | 0.961 |
| 00000087 | 1,399 | 0.940 | 0.983 | 0.968 |
| 00000094 | 3,230 | 0.907 | 0.945 | 0.920 |
| **Mean over the 10 trips** | | **0.891** | **0.938** | **0.885** |

The HMM improved accuracy on 9 of the 10 trips, left one unchanged (trip 00000023) and made none worse. The largest gain is trip 00000015 (0.671 to 0.897). Each request took 0.14 to 0.70 s.
These results are consistent with the development results (student_test 0.865 to 0.918, unseen training trips 0.878 to 0.941), so we see no sign that we overfitted to the development data.
Accuracy is measured against the reference route aligned to the GPS points, which is a derived reference and not independently verified truth.

**[NEEDED: Figure 7-A]** Laptop 2 UI showing a completed trip (GPS, predicted route, metrics) with the pipeline shown as `classifier + HMM/Viterbi`.

**[NEEDED: Figure 7-B]** Laptop 1 API terminal showing the matching request id, the second laptop's client IP, trip, number of points, time and accuracy.

Reproduce: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` with `PATHSNAP_MODE=hmm`; set `PATHSNAP_TRIPS_DIR` to the folder with the evaluation trips.
A Windows firewall rule for TCP port 8000 and a shared network are required; `scripts/check_all_trips.py --api http://<laptop-1-ip>:8000` checks all trips from the second laptop.
"""

SEC8 = """
## 8. Academic integrity and AI assistance

All reported numbers come from committed result files and can be regenerated with the scripts listed in the repository README. Learned preprocessing was
fitted on training data only, cross-validation folds are grouped by trip, the evaluator trips were never opened during development (they were first used for the
single final evaluation in section 7.1), and the reference route is never used by the prediction path. Reference labels are derived by aligning GPS points to the
supplied route and are not independently verified truth. No work was shared with other teams.

**AI-use disclosure (draft for each member to confirm or correct; the event policy requires it to be accurate).**

- **PES1UG24CS497:** an AI coding assistant (Anthropic Claude, used through Claude Code) helped implement and test the data loader, candidate generation, route-based labelling,
  feature extraction, EDA, the multi-hop HMM decoder and its evaluation and tuning scripts, the API integration, the report-generation scripts, and drafted report sections 2.2, 5 and 6.
  **[497: add what you personally ran, checked and can explain.]**
- **PES1UG24CS548:** per the draft in the report skeleton, an AI assistant (OpenAI Codex) helped with the API, UI, tests, setup documentation and the report skeleton. **[548 to confirm.]**
- **PES1UG24CS496:** **[496 to state the AI tools used, if any, for the training code, experiments and the report text in sections 3 and 4.]**
- **PES1UG24CS549:** per the repository notes, AI assistance was used for the fixed-transition decoder, its evaluation, the figure generator and report text. **[549 to confirm and name the tool.]**

Each member must be able to explain the code, experiments and results of their own part.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    parts = [HEADER, read("section_2_2_eda.md")]
    parts.append("\n### EDA figures (train partition unless stated)\n")
    for f, c in [("01_trip_sizes", "Figure 2.1: trip size distributions (train and student_test)"),
                 ("02_locations_and_network_size", "Figure 2.2: trip locations and road-network size"),
                 ("03_sampling_intervals_and_gaps", "Figure 2.3: sampling intervals and large GPS gaps"),
                 ("04_noise_and_reference_quality", "Figure 2.4: GPS noise relative to the reference route, and candidate miss rate per trip"),
                 ("05_network_connectivity", "Figure 2.5: road-network connectivity"),
                 ("06_example_trajectories", "Figure 2.6: GPS tracks, road network and reference route for three trips"),
                 ("folium_map_train_00000001", "Figure 2.7: Folium map of train trip 00000001 (GPS, road network, reference route)"),
                 ("folium_map_train_00000054", "Figure 2.8: Folium map of train trip 00000054, whose reference route runs about 140 m from the GPS corridor")]:
        parts.append(img(f"results/figures/{f}.png", c))
    models_md = read("section_3_4_models.md")
    cut = models_md.index("## Bayesian hyperparameter optimisation")
    part_a = models_md[models_md.index("## Experimental protocol"):cut]
    part_a = re.sub(r"(?m)^## ", "### ", part_a)
    part_b = models_md[cut:].replace("## Bayesian hyperparameter optimisation", "# 4. Bayesian hyperparameter optimisation", 1)
    parts.append("\n# 3. Model experimentation and comparison\n")
    parts.append(part_a)
    parts.append(read("section_3_1_table.md"))
    parts.append(read("section_3_2_per_model.md"))
    parts.append("\n### 3.3 Model visualisations (student-test and cross-validation results)\n")
    for f, c in [("top1", "Figure 3.1: per-point Top-1 accuracy by model"), ("macro_f1", "Figure 3.2: macro-F1 by model"), ("log_loss", "Figure 3.3: log loss by model")]:
        parts.append(img(f"results/models/model_comparison/plots/{f}.png", c))
    parts.append("\nOnly the MLP produces an epoch-wise training history (`results/models/model_comparison/MLP_loss_curve.csv` and `MLP_grouped_early_stopping.csv`); "
                 "no curves are drawn for the other models. Per-model confusion matrices are given in section 3.2 and in `results/models/model_comparison/metrics.csv`.\n")
    parts.append(part_b)
    parts.append(read("section_5_hmm.md"))
    parts.append(read("section_6_hmm_figures.md"))
    sec7 = SEC7
    ui_a = os.path.join(ROOT, "results", "figures", "demo_ui_trip00000015_metrics_and_map.png")
    ui_b = os.path.join(ROOT, "results", "figures", "demo_ui_trip00000015_map_and_run_notes.png")
    term = os.path.join(ROOT, "results", "figures", "demo_terminal_trip00000015.png")
    if os.path.exists(ui_a) and os.path.exists(ui_b):
        a7 = sec7.index("**[NEEDED: Figure 7-A]**"); b7 = sec7.index("**[NEEDED: Figure 7-B]**")
        sec7 = (sec7[:a7]
                + img("results/figures/demo_ui_trip00000015_metrics_and_map.png", "Figure 7-A (1/2): the UI on the second laptop after running evaluator trip 00000015 in the final mode: metrics cards, GPS points (blue) and predicted route (orange) on an OpenStreetMap basemap")
                + img("results/figures/demo_ui_trip00000015_map_and_run_notes.png", "Figure 7-A (2/2): the same run, with the legend and the Run Notes: prediction method `classifier + HMM/Viterbi` and the Request id that appears in the API terminal log (Figure 7-B)")
                + "\n" + sec7[b7:])
        sec7 = sec7.replace("| Final-mode demonstration on laptop 2 | **[NEEDED: repeat with `PATHSNAP_MODE=hmm` and capture the screenshots below]** |",
                            "| Final-mode demonstration on laptop 2 | Done: a second laptop opened the UI served by laptop 1 and ran the evaluator trips; see Figure 7-A |")
    term_local = os.path.join(ROOT, "results", "figures", "demo_terminal_api_log_evaluator_trips.png")
    sheet = os.path.join(ROOT, "results", "figures", "demo_ui_all_10_trips.png")
    extra = ""
    if os.path.exists(term):
        extra += img("results/figures/demo_terminal_trip00000015.png", "Figure 7-B: laptop 1's API terminal showing the matching `REQUEST id=...` line (trip, points, time, accuracy) and the second laptop's client IP")
    elif os.path.exists(term_local):
        extra += img("results/figures/demo_terminal_api_log_evaluator_trips.png",
                     "Figure 7-B: the API terminal on laptop 1 in the final mode (`READY mode=classifier + HMM/Viterbi trips=10 data=...evaluator_test`) after the 10 evaluator trips were requested. "
                     "Each `REQUEST` line shows the trip, number of points, time and accuracy, followed by `POST /predict ... 200 OK`. In this capture the requests came from laptop 1 itself (client 127.0.0.1); "
                     "when a second laptop calls the API, its address appears on the `INFO:` line in the same format.")
    if os.path.exists(sheet):
        extra += img("results/figures/demo_ui_all_10_trips.png",
                     "Figure 7-C: the UI after running each of the 10 evaluator trips in the final mode, one panel per trip (metrics cards and the map with the predicted route). "
                     "Captured in a browser on laptop 1 against the API; every run reports the pipeline `classifier + HMM/Viterbi`.")
    if extra:
        a7 = sec7.index("**[NEEDED: Figure 7-B]**"); b7 = sec7.index("Reproduce:")
        sec7 = sec7[:a7] + extra + "\n" + sec7[b7:]
    parts.append(sec7)
    parts.append(SEC8)
    md = "\n\n".join(parts)
    md_path = os.path.join(REP, "FULL_REPORT_DRAFT.md")
    open(md_path, "w", encoding="utf-8").write(md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    css = ("body{font:12pt/1.45 Georgia,serif;max-width:900px;margin:24px auto;padding:0 18px;color:#111}"
           "h1{font-size:20pt;margin-top:34px;border-bottom:1px solid #999}h2{font-size:15pt;margin-top:24px}h3{font-size:13pt}"
           "table{border-collapse:collapse;font:9.5pt Arial,sans-serif;margin:10px 0;width:100%}th,td{border:1px solid #aaa;padding:4px 6px;vertical-align:top}"
           "th{background:#eee}img{max-width:100%;height:auto;display:block;margin:10px auto}code{font-size:9.5pt;background:#f2f2f2;padding:0 2px}"
           "pre{background:#f2f2f2;padding:8px;overflow-x:auto;font-size:9pt}blockquote{border-left:3px solid #999;margin:8px 0;padding:2px 12px;color:#333}"
           "em{color:#444}@media print{img{page-break-inside:avoid}h1{page-break-before:always}h1:first-of-type{page-break-before:avoid}}")
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap report (draft)</title><style>{css}</style></head><body>{body}</body></html>"
    html_path = os.path.join(REP, "FULL_REPORT_DRAFT.html")
    open(html_path, "w", encoding="utf-8").write(html)
    print("wrote", md_path, "and", html_path)
    if a.pdf:
        exe = next((p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Google\Chrome\Application\chrome.exe"] if os.path.exists(p)), None)
        if not exe:
            sys.exit("no Edge/Chrome found for PDF export")
        pdf = os.path.join(REP, "FULL_REPORT_DRAFT.pdf")
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                        "file:///" + html_path.replace("\\", "/")], check=False, capture_output=True, timeout=180)
        print("wrote", pdf, "exists:", os.path.exists(pdf))


if __name__ == "__main__":
    main()
