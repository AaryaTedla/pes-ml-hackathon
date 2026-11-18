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

> Items marked **[NEEDED: ...]** require evidence only the team can supply (screenshots from the demo laptops, AI-use statements).
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

**[NEEDED: Figure 7-A]** Laptop 2 UI showing a completed trip (GPS, predicted route, metrics) with the pipeline shown as `classifier + HMM/Viterbi`.

**[NEEDED: Figure 7-B]** Laptop 1 API terminal showing the matching request id, the second laptop's client IP, trip, number of points, time and accuracy.

Reproduce: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` with `PATHSNAP_MODE=hmm`; set `PATHSNAP_TRIPS_DIR` to the folder with the evaluation trips.
A Windows firewall rule for TCP port 8000 and a shared network are required; `scripts/check_all_trips.py --api http://<laptop-1-ip>:8000` checks all trips from the second laptop.
"""

SEC8 = """
## 8. Academic integrity and AI assistance

All reported numbers come from committed result files and can be regenerated with the scripts listed in the repository README. Learned preprocessing was
fitted on training data only, cross-validation folds are grouped by trip, the evaluator trips were never opened during development, and the reference route is
never used by the prediction path.

**[NEEDED: AI-use disclosure.]** The event policy requires each member to state which AI tools helped with which parts of the work (for example code, tests,
report drafting) and what they checked themselves. Each member should write their own entry here and be able to explain their code, experiments and results.
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
    parts.append("\n# 3. Model experimentation and comparison\n")
    parts.append(read("section_3_1_table.md"))
    parts.append(read("section_3_4_models.md"))
    parts.append("\n### 3.3 Model visualisations (student-test and cross-validation results)\n")
    for f, c in [("top1", "Figure 3.1: per-point Top-1 accuracy by model"), ("macro_f1", "Figure 3.2: macro-F1 by model"), ("log_loss", "Figure 3.3: log loss by model")]:
        parts.append(img(f"results/models/model_comparison/plots/{f}.png", c))
    parts.append("\nOnly the MLP produces an epoch-wise training history (`results/models/model_comparison/MLP_loss_curve.csv`); no curves are drawn for the other models. "
                 "**[NEEDED: per-model answers to the compulsory questions A, B and C of section 3.2 (suitability, hyperparameters and loss, results, over- or under-fitting evidence), written by the model owner.]**\n")
    parts.append(read("section_5_hmm.md"))
    parts.append(read("section_6_hmm_figures.md"))
    parts.append(SEC7)
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
