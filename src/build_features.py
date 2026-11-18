"""Build features_{partition}.parquet for train and student_test.
Usage: python build_features.py [--limit N]   (run from anywhere)"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from joblib import Parallel, delayed
from data_loader import load_trip, list_trips
from features import trip_features, FEATURE_COLS, ID_COLS

ROOT = os.path.join(os.path.dirname(__file__), "..")


def one(trip_dir):
    t0 = time.time()
    f, p = trip_features(load_trip(trip_dir))
    return f, p, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    for part in ["train", "student_test"]:
        dirs = list_trips(os.path.join(ROOT, "data"), part)[: a.limit]
        res = Parallel(n_jobs=a.jobs)(delayed(one)(d) for d in dirs)
        f = pd.concat([r[0] for r in res], ignore_index=True)
        p = pd.concat([r[1] for r in res], ignore_index=True)
        f = f[ID_COLS + FEATURE_COLS + ["pt_has_pos"]]
        f.to_parquet(os.path.join(out, f"features_{part}.parquet"), index=False)
        p.to_parquet(os.path.join(out, f"points_{part}.parquet"), index=False)
        print(part, len(dirs), "trips", len(f), "rows", len(p), "points", f"{sum(r[2] for r in res):.0f}s cpu")
