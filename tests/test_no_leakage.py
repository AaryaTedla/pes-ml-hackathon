"""Checks that prediction-time features do not depend on the reference route.
Run: python tests/test_no_leakage.py   (from the repo root)"""
import os, sys, shutil, tempfile
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from data_loader import load_trip
from features import trip_features, FEATURE_COLS
from pipeline import prepare_trip

TRIPS = [("train", "00000001"), ("student_test", "00000007"), ("student_test", "00000056")]


def main():
    for part, tid in TRIPS:
        src = os.path.join(ROOT, "data", part, tid)
        # 1) copy the trip WITHOUT its .route file: the pipeline must still run
        tmp = os.path.join(tempfile.mkdtemp(), tid)
        os.makedirs(tmp)
        for e in ("track", "nodes", "arcs"):
            shutil.copy(os.path.join(src, f"{tid}.{e}"), tmp)
        assert not os.path.exists(os.path.join(tmp, f"{tid}.route"))
        _, f_nr, p_nr = prepare_trip(tmp)
        # 2) full pipeline with the route (training-time view)
        f_r, p_r = trip_features(load_trip(src))
        # 3) candidates and every feature column must be identical with and without the route
        assert len(f_nr) == len(f_r), "candidate sets differ"
        assert (f_nr["point_idx"].to_numpy() == f_r["point_idx"].to_numpy()).all()
        assert (f_nr["arc_id"].to_numpy() == f_r["arc_id"].to_numpy()).all()
        for c in FEATURE_COLS:
            assert np.array_equal(f_nr[c].to_numpy(), f_r[c].to_numpy()), f"feature {c} depends on route"
        # 4) no label-derived columns at prediction time
        assert not {"label", "pt_has_pos", "true_arc", "label_dist_m", "has_positive"} & (set(f_nr) | set(p_nr))
        print(f"OK {part}/{tid}: {len(f_nr)} candidate rows, {len(FEATURE_COLS)} features identical with/without route")
        shutil.rmtree(os.path.dirname(tmp))
    print("all leakage checks passed")


if __name__ == "__main__":
    main()
