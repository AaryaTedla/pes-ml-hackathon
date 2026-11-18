"""One-command integration of a trained model (run once artifacts/model.joblib exists).

    python scripts/integrate_model.py                # all steps
    python scripts/integrate_model.py --steps validate,api     # only some steps

Steps (each stops the run if it fails):
  validate  model loads, feature order matches, predict_proba gives valid positive-class probabilities
  tune      re-tune the multi-hop decoder on the model's probabilities (TUNE trips only), save artifacts/hmm_config.json
  tunefix   same for the fixed-transition decoder, save artifacts/hmm_fixed_config.json
  compare   classifier-only vs both decoders on TUNE / VAL / student_test -> results/hmm_decoder_comparison_model*.csv
  api       API in `hmm` mode on all 10 student_test trips -> results/api_hmm_check_model.csv
Env overrides for tests only: PATHSNAP_ARTIFACTS, PATHSNAP_RESULTS. Evaluator trips are never used.
"""
import os, sys, json, argparse, subprocess, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from features import FEATURE_COLS
import eval_hmm


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"STEP FAILED (exit {r.returncode}): {' '.join(cmd)}")
    print(f"   done in {time.time() - t0:.0f}s", flush=True)


def validate():
    print("== validate model")
    mp = os.path.join(eval_hmm.ART, "model.joblib")
    if not os.path.exists(mp):
        raise SystemExit(f"{mp} not found")
    m, scaler = eval_hmm.load_model()
    print(f"   model: {type(m).__name__} | {os.path.getsize(mp) / 1e6:.1f} MB | scaler file: {scaler is not None}")
    cj = os.path.join(eval_hmm.ART, "feature_columns.json")
    if os.path.exists(cj):
        cols = json.load(open(cj))
        if cols != FEATURE_COLS:
            raise SystemExit(f"feature_columns.json differs from FEATURE_COLS:\n  {cols}\n  {FEATURE_COLS}")
        print("   feature_columns.json matches the 18 agreed features, in order")
    else:
        print("   WARNING: artifacts/feature_columns.json missing (cannot check feature order)")
    for obj in (m, scaler):
        if obj is not None and hasattr(obj, "feature_names_in_") and list(obj.feature_names_in_) != FEATURE_COLS:
            raise SystemExit("model/scaler feature_names_in_ differ from FEATURE_COLS")
    f = pd.read_parquet(os.path.join(ROOT, "results", "features_student_test.parquet"))
    p = eval_hmm.model_probabilities(m, scaler, f)            # raises on bad classes_/shape/probabilities
    print(f"   predict_proba on {len(f)} student_test rows OK | mean prob {p.mean():.3f} | min {p.min():.3f} max {p.max():.3f}")
    top1 = f.assign(p=p).sort_values(["trip_id", "point_idx", "p"], ascending=[True, True, False]) \
            .drop_duplicates(["trip_id", "point_idx"])
    print(f"   student_test per-point Top-1 (all points): {top1.label.mean():.4f}  [nearest-arc baseline is about 0.51]")


def api_check():
    print("== API in hmm mode, all student_test trips")
    from pathlib import Path
    from fastapi.testclient import TestClient
    sys.path.insert(0, ROOT)
    from api.config import Settings
    from api.main import create_app
    app = create_app(Settings(Path(ROOT) / "data" / "student_test", Path(eval_hmm.ART), "hmm"))
    rows = []
    with TestClient(app) as c:
        h = c.get("/health").json()
        print(f"   pipeline: {h['pipeline']}")
        for t in c.get("/trips").json()["trips"]:
            r = c.post("/predict", json={"trip_id": t})
            r.raise_for_status()
            j = r.json(); m = j["metrics"]
            rows.append(dict(trip_id=t, points=j["n_points"], acc_classifier_only=m["pointwise_accuracy_classifier_only"],
                             acc_final=m["pointwise_accuracy"], continuity=m["route_continuity"],
                             total_ms=j["timing_ms"]["total"], pipeline=j["pipeline"]))
    df = pd.DataFrame(rows)
    print(df.drop(columns="pipeline").round(4).to_string(index=False))
    print(f"   mean acc classifier-only {df.acc_classifier_only.mean():.4f} -> final {df.acc_final.mean():.4f} | "
          f"mean continuity {df.continuity.mean():.4f}")
    df.to_csv(os.path.join(eval_hmm.RES, "api_hmm_check_model.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="validate,tune,tunefix,compare,api")
    steps = ap.parse_args().steps.split(",")
    if "validate" in steps:
        validate()
    if "tune" in steps:
        run(["src/eval_hmm.py", "--mode", "tune", "--prob", "model", "--decoder", "multihop", "--save-config"])
    if "tunefix" in steps:
        run(["src/eval_hmm.py", "--mode", "tune", "--prob", "model", "--decoder", "fixed", "--save-config"])
    if "compare" in steps:
        run(["src/compare_decoders.py", "--prob", "model"])
    if "api" in steps:
        api_check()
    print("\nALL REQUESTED STEPS COMPLETED")


if __name__ == "__main__":
    main()
