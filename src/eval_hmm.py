"""Batch evaluation of classifier-only vs classifier+HMM over many WHOLE trips, plus decoder tuning.

    python src/eval_hmm.py --mode eval            # evaluate current/default config on tune, val and student_test trips
    python src/eval_hmm.py --mode tune            # coordinate search on the TUNE trips, report on VAL and student_test
Options: --prob mock|model   (mock = softmax(-dist_m/10 + 3*cos_head); model = artifacts/model.joblib)

Protocol: train trips are split (seeded) into TUNE and VAL sets of whole trips; settings are chosen on TUNE only,
then checked on VAL (unseen train trips) and on the 10 student_test trips. Evaluator trips are never used.
Trips whose reference route does not cover the GPS track (candidate miss > 5%) are excluded from the splits.
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from data_loader import load_trip
from features import FEATURE_COLS
from evaluate import pointwise_accuracy, route_continuity
from hmm import decode, DEFAULT_CFG

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, ART = os.path.join(ROOT, "results"), os.path.join(ROOT, "artifacts")
COVERED_M = 60.0
MAX_PTS = 8000          # keep tuning fast: skip very long trips in the splits

GRID = {
    "hop_penalty": [0.25, 0.5, 1.0, 2.0, 4.0],
    "jump_penalty": [4.0, 8.0, 12.0, 20.0, 40.0],
    "uturn_penalty": [1.0, 4.0, 8.0, 16.0],
    "max_hops": [2, 3, 5],
    "gap_s": [10.0, 30.0, 120.0],
}


def get_prob(f, source):
    if source == "mock":
        s = (-f["dist_m"] / 10.0 + 3.0 * f["cos_head"]).to_numpy(float)
        key = (f["trip_id"].astype(str) + "_" + f["point_idx"].astype(str)).to_numpy()
        s = pd.Series(s)
        e = np.exp(s - s.groupby(key).transform("max"))
        return (e / e.groupby(key).transform("sum")).to_numpy()
    import joblib
    m = joblib.load(os.path.join(ART, "model.joblib"))
    X = f[FEATURE_COLS].to_numpy()
    sp = os.path.join(ART, "scaler.joblib")
    if os.path.exists(sp):
        X = joblib.load(sp).transform(X)
    return m.predict_proba(X)[:, 1]


def load_data(source):
    D = {}
    for part in ("train", "student_test"):
        f = pd.read_parquet(os.path.join(RES, f"features_{part}.parquet")) if part == "student_test" or \
            os.path.exists(os.path.join(RES, "features_train.parquet")) else None
        if f is None:
            raise SystemExit("results/features_train.parquet missing: run src/build_features.py first")
        p = pd.read_parquet(os.path.join(RES, f"points_{part}.parquet"))
        f["prob"] = get_prob(f, source)
        pg = {k: v.sort_values("point_idx").reset_index(drop=True) for k, v in p.groupby("trip_id")}
        for tid, ft in f.groupby("trip_id", sort=True):
            pts = pg[tid]
            D[(part, tid)] = dict(part=part, tid=tid, f=ft, pts=pts, n=len(pts),
                                  miss=1 - float(pts.has_positive.mean()),
                                  maxgap=float(np.diff(pts.t.to_numpy()).max()) if len(pts) > 1 else 0.0)
    return D


def prep(d):
    """Lazy per-trip objects: trip arrays, true arc, covered mask, classifier-only prediction."""
    if "trip" in d:
        return d
    d["trip"] = load_trip(os.path.join(ROOT, "data", d["part"], d["tid"]), with_route=False)
    f = d["f"]
    pi, ai, p = f["point_idx"].to_numpy(), f["arc_id"].to_numpy(), f["prob"].to_numpy()
    o = np.lexsort((-p, pi))
    first = np.r_[True, pi[o][1:] != pi[o][:-1]]
    clf = np.full(d["n"], -1, np.int64)
    clf[pi[o][first]] = ai[o][first]
    d["clf"] = clf
    d["true"] = d["pts"]["true_arc"].to_numpy()
    d["cov"] = d["pts"]["label_dist_m"].to_numpy() <= COVERED_M
    return d


def eval_trip(d, cfg):
    prep(d)
    t0 = time.time()
    h = decode(d["trip"], d["f"], cfg)
    rt = time.time() - t0
    arcs, cov, true = d["trip"]["arcs"], d["cov"], d["true"]
    ok_h, ok_c = h >= 0, d["clf"] >= 0
    return dict(part=d["part"], trip_id=d["tid"], n_points=d["n"], max_gap_s=d["maxgap"],
                frac_covered=float(cov.mean()),
                acc_clf=pointwise_accuracy(d["clf"], true, cov), acc_hmm=pointwise_accuracy(h, true, cov),
                cont_clf=route_continuity(d["clf"][ok_c], arcs), cont_hmm=route_continuity(h[ok_h], arcs),
                cont_ref=route_continuity(true, arcs), hmm_runtime_s=rt)


def eval_set(D, keys, cfg):
    return pd.DataFrame([eval_trip(D[k], cfg) for k in keys])


def summary(df, name):
    return (f"{name:12s} trips={len(df):3d} | acc  clf {df.acc_clf.mean():.4f}  HMM {df.acc_hmm.mean():.4f}  "
            f"(HMM better in {int((df.acc_hmm > df.acc_clf).sum())}, worse in {int((df.acc_hmm < df.acc_clf).sum())}) | "
            f"continuity clf {df.cont_clf.mean():.4f} HMM {df.cont_hmm.mean():.4f} | HMM time {df.hmm_runtime_s.sum():.1f}s")


def make_splits(D, seed=0, n_each=10):
    rng = np.random.default_rng(seed)
    ok = [k for k, d in D.items() if d["part"] == "train" and d["miss"] <= 0.05 and d["n"] <= MAX_PTS]
    gap = sorted(k for k in ok if D[k]["maxgap"] > 60)
    reg = sorted(k for k in ok if D[k]["maxgap"] <= 60)
    rng.shuffle(gap); rng.shuffle(reg)
    gg, rr = max(1, len(gap) // 2), 2 * n_each
    tune = gap[:gg] + reg[:n_each + 4]
    val = gap[gg:] + reg[n_each + 4: n_each + 4 + (len(tune) - (len(gap) - gg))]
    test = sorted(k for k, d in D.items() if d["part"] == "student_test")
    return tune, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval", choices=["eval", "tune"])
    ap.add_argument("--prob", default="mock", choices=["mock", "model"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-config", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True); os.makedirs(ART, exist_ok=True)
    D = load_data(a.prob)
    tune, val, test = make_splits(D, a.seed)
    print(f"prob source={a.prob} | tune={len(tune)} val={len(val)} student_test={len(test)} trips "
          f"(gap>60s trips: tune {sum(D[k]['maxgap'] > 60 for k in tune)}, val {sum(D[k]['maxgap'] > 60 for k in val)})", flush=True)
    cfg = dict(DEFAULT_CFG)
    if a.mode == "tune":
        log = []
        score = lambda c: eval_set(D, tune, c).acc_hmm.mean()
        best = score(cfg)
        print(f"default config: tune acc {best:.4f}", flush=True)
        for rnd in range(2):
            improved = False
            for p, vals in GRID.items():
                for v in vals:
                    if v == cfg[p]:
                        continue
                    c = {**cfg, p: v}
                    s = score(c)
                    log.append(dict(round=rnd, param=p, value=v, tune_acc=s))
                    if s > best + 1e-5:
                        best, cfg, improved = s, c, True
                        print(f"  round {rnd}: {p}={v} -> tune acc {s:.4f}", flush=True)
            if not improved:
                break
        pd.DataFrame(log).to_csv(os.path.join(RES, f"hmm_tuning_{a.prob}.csv"), index=False)
        print("best config:", {k: cfg[k] for k in GRID})
    rows = []
    for name, keys in (("TUNE", tune), ("VAL", val), ("STUDENT_TEST", test)):
        df = eval_set(D, keys, cfg).assign(split=name)
        rows.append(df)
        print(summary(df, name))
        if a.mode == "tune":
            dd = eval_set(D, keys, DEFAULT_CFG)
            print(summary(dd, name + " (default cfg)"))
    out = pd.concat(rows)
    out.to_csv(os.path.join(RES, f"hmm_eval_{a.prob}.csv"), index=False)
    print(out.sort_values("acc_hmm").head(0).to_string())
    worse = out[out.acc_hmm < out.acc_clf - 0.01][["split", "trip_id", "n_points", "max_gap_s", "acc_clf", "acc_hmm"]]
    print("trips where the HMM is worse than classifier-only by >1pt:\n", worse.to_string(index=False) if len(worse) else "none")
    if a.save_config:
        json.dump({**{k: cfg[k] for k in DEFAULT_CFG}, "tuned_on": f"{len(tune)} train trips, prob={a.prob}", "seed": a.seed},
                  open(os.path.join(ART, "hmm_config.json"), "w"), indent=2)
        print("saved artifacts/hmm_config.json")


if __name__ == "__main__":
    main()
