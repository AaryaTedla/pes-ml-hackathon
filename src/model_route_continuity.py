"""Classifier-only route continuity and point accuracy per model, from saved student_test score files.

    python src/model_route_continuity.py --scores-dir <folder with <Model>_student_scores.parquet> --out results/models/route_continuity_by_model.csv

For each model: take the highest-probability candidate per GPS point (classifier only, no HMM), then compute
  - route continuity (src/evaluate.py: among changes of predicted arc, fraction that are directly connected in the road graph)
  - point accuracy against the route-derived reference labels (results/points_student_test.parquet), on points whose
    reference arc is within 60 m (same mask as the decoder comparison)
Student_test only. Evaluator trips are never used.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from data_loader import load_trip
from evaluate import route_continuity, pointwise_accuracy

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS = ["LogReg", "LightGBM", "RandomForest", "MLP", "KNN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "models", "route_continuity_by_model.csv"))
    a = ap.parse_args()
    pts = pd.read_parquet(os.path.join(ROOT, "results", "points_student_test.parquet"))
    trips = {t: load_trip(os.path.join(ROOT, "data", "student_test", t), with_route=False) for t in pts.trip_id.unique()}
    rows = []
    for m in MODELS:
        path = os.path.join(a.scores_dir, f"{m}_student_scores.parquet")
        if not os.path.exists(path):
            print("missing", path); continue
        s = pd.read_parquet(path)
        top = s.sort_values(["trip_id", "point_idx", "probability_correct"], ascending=[True, True, False]) \
               .drop_duplicates(["trip_id", "point_idx"])
        conts, accs, w = [], [], []
        for tid, g in top.groupby("trip_id"):
            p = pts[pts.trip_id == tid].sort_values("point_idx")
            pred = np.full(len(p), -1, np.int64)
            pred[g.point_idx.to_numpy()] = g.arc_id.to_numpy()
            ok = pred >= 0
            conts.append(route_continuity(pred[ok], trips[tid]["arcs"]))
            cov = p.label_dist_m.to_numpy() <= 60.0
            accs.append(pointwise_accuracy(pred, p.true_arc.to_numpy(), cov))
        rows.append(dict(model=m, trips=len(conts), mean_route_continuity=np.mean(conts),
                         min_route_continuity=np.min(conts), mean_point_accuracy_covered=np.mean(accs)))
    out = pd.DataFrame(rows).round(4)
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
