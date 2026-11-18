"""Exploratory data analysis. Writes tables to results/eda, figures to results/figures,
interactive Folium maps to results/maps. Run: python eda.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from data_loader import load_trip, list_trips
from geo import make_projector, project
from viz import make_trip_map

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results")
for d in ("eda", "figures", "maps"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)


def trip_stats(part, d):
    t = load_trip(d)
    tid = t["trip_id"]
    tr, nd, ar, rt = t["track"], t["nodes"], t["arcs"], t["route"]
    n, M, K = len(tr), len(nd), len(ar)
    p = os.path.join(d, tid)
    sizes = {e: os.path.getsize(f"{p}.{e}") for e in ("track", "nodes", "arcs", "route")}
    lon, lat, ts = tr[:, 0], tr[:, 1], tr[:, 2]
    invalid = (~np.isfinite(tr).all(1)) | (np.abs(lon) > 180) | (np.abs(lat) > 90) | ((lon == 0) & (lat == 0))
    dt = np.diff(ts)
    xy = project(make_projector(lon, lat), lon, lat)
    step = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1))
    ok = (dt > 0) & (dt <= 5)
    spd = step[ok] / dt[ok]
    # road network
    A = coo_matrix((np.ones(K), (ar[:, 0], ar[:, 1])), shape=(M, M)).tocsr()
    ncw, lw = connected_components(A, directed=False)
    ncs, ls = connected_components(A, directed=True, connection="strong")
    deg = np.bincount(ar.ravel(), minlength=M)
    S = set(map(tuple, ar.tolist()))
    rev = np.mean([(b, a) in S for a, b in ar.tolist()])
    # reference route
    rvalid = bool(rt.min() >= 0 and rt.max() < K)
    breaks = int((ar[rt[:-1], 1] != ar[rt[1:], 0]).sum()) if rvalid and len(rt) > 1 else -1
    uturn = int(((ar[rt[:-1], 0] == ar[rt[1:], 1]) & (ar[rt[:-1], 1] == ar[rt[1:], 0])).sum()) if rvalid else -1
    return dict(
        partition=part, trip_id=tid, n_gps=n, n_nodes=M, n_arcs=K, n_route_arcs=len(rt),
        track_kB=sizes["track"] / 1e3, nodes_kB=sizes["nodes"] / 1e3, arcs_kB=sizes["arcs"] / 1e3, route_kB=sizes["route"] / 1e3,
        invalid_gps=int(invalid.sum()), dup_points=int((step == 0).sum()),
        t_nonmonotonic=int((dt <= 0).sum()), duration_s=float(ts[-1] - ts[0]),
        frac_dt_not_1s=float((dt != 1).mean()), median_dt=float(np.median(dt)), max_gap_s=float(dt.max()),
        n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt60s=int((dt > 60).sum()),
        extent_x_km=float(np.ptp(xy[:, 0]) / 1e3), extent_y_km=float(np.ptp(xy[:, 1]) / 1e3),
        path_km=float(step.sum() / 1e3), median_speed_mps=float(np.median(spd)) if len(spd) else np.nan,
        p99_speed_mps=float(np.percentile(spd, 99)) if len(spd) else np.nan, n_jumps_gt50mps=int((spd > 50).sum()),
        lon_mean=float(lon.mean()), lat_mean=float(lat.mean()),
        weak_components=int(ncw), largest_weak_frac=float(np.bincount(lw).max() / M),
        strong_components=int(ncs), largest_strong_frac=float(np.bincount(ls).max() / M),
        isolated_nodes=int((deg == 0).sum()), self_loops=int((ar[:, 0] == ar[:, 1]).sum()),
        dup_arcs=int(K - len(S)), reverse_arc_frac=float(rev),
        route_ids_valid=rvalid, route_breaks=breaks, route_uturns=uturn,
    )


def main():
    rows = []
    for part in ("train", "student_test"):
        for d in list_trips(os.path.join(ROOT, "data"), part):
            rows.append(trip_stats(part, d))
            print(part, rows[-1]["trip_id"], rows[-1]["n_gps"], flush=True)
    inv = pd.DataFrame(rows)
    pts = pd.concat([pd.read_parquet(os.path.join(OUT, f"points_{p}.parquet")).assign(partition=p)
                     for p in ("train", "student_test")])
    g = pts.groupby(["partition", "trip_id"]).agg(
        n_cand_mean=("n_cand", "mean"), cand_miss=("has_positive", lambda s: 1 - s.mean()),
        label_dist_median=("label_dist_m", "median"), label_dist_p95=("label_dist_m", lambda s: s.quantile(.95))).reset_index()
    inv = inv.merge(g, on=["partition", "trip_id"])
    inv.to_csv(os.path.join(OUT, "eda", "trip_inventory.csv"), index=False)

    cols = ["n_gps", "n_nodes", "n_arcs", "n_route_arcs", "duration_s", "path_km", "max_gap_s", "frac_dt_not_1s",
            "median_speed_mps", "weak_components", "isolated_nodes", "route_breaks", "cand_miss"]
    summ = inv.groupby("partition")[cols].agg(["min", "median", "mean", "max"]).T
    summ.to_csv(os.path.join(OUT, "eda", "partition_summary.csv"))
    counts = inv.groupby("partition").agg(
        trips=("trip_id", "size"), gps_points=("n_gps", "sum"), total_nodes=("n_nodes", "sum"), total_arcs=("n_arcs", "sum"),
        invalid_gps=("invalid_gps", "sum"), nonmonotonic_t=("t_nonmonotonic", "sum"), dup_points=("dup_points", "sum"),
        trips_with_gap_gt60s=("n_gaps_gt60s", lambda s: int((s > 0).sum())),
        trips_with_route_breaks=("route_breaks", lambda s: int((s > 0).sum())),
        trips_disconnected_net=("weak_components", lambda s: int((s > 1).sum())),
        trips_with_isolated_nodes=("isolated_nodes", lambda s: int((s > 0).sum())),
        trips_with_speed_jumps=("n_jumps_gt50mps", lambda s: int((s > 0).sum())),
        bad_route_ids=("route_ids_valid", lambda s: int((~s).sum()))).T
    counts.to_csv(os.path.join(OUT, "eda", "partition_counts.csv"))
    print(counts.to_string())

    # ------------------------------ figures ------------------------------
    F = lambda n: os.path.join(OUT, "figures", n)
    col = {"train": "#1f77b4", "student_test": "#ff7f0e"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, s in inv.groupby("partition"):
        ax[0].hist(np.log10(s.n_gps), bins=20, alpha=.6, label=p, color=col[p])
        ax[1].hist(np.log10(s.n_arcs), bins=20, alpha=.6, label=p, color=col[p])
        ax[2].hist(s.n_route_arcs, bins=20, alpha=.6, label=p, color=col[p])
    ax[0].set(xlabel="log10(GPS points per trip)", ylabel="trips"); ax[1].set(xlabel="log10(road arcs per trip)")
    ax[2].set(xlabel="reference route length (arcs)"); ax[0].legend()
    fig.suptitle("Trip size distributions (train: 80 trips, student_test: 10 trips)"); fig.tight_layout(); fig.savefig(F("01_trip_sizes.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(inv.lon_mean, inv.lat_mean, c=inv.partition.map(col), s=30)
    ax[0].set(xlabel="longitude", ylabel="latitude", title="Trip locations (one dot per trip)")
    for p in col: ax[0].scatter([], [], c=col[p], label=p)
    ax[0].legend()
    ax[1].scatter(inv.n_nodes, inv.n_arcs, c=inv.partition.map(col), s=30)
    ax[1].set(xlabel="nodes", ylabel="arcs", xscale="log", yscale="log", title="Road network size per trip")
    fig.tight_layout(); fig.savefig(F("02_locations_and_network_size.png"), dpi=130); plt.close(fig)

    # sampling intervals + gaps over all trips (train / student_test separately)
    dts = {p: [] for p in col}
    for part in col:
        for d in list_trips(os.path.join(ROOT, "data"), part):
            dts[part].append(np.diff(load_trip(d, with_route=False)["track"][:, 2]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for p, v in dts.items():
        v = np.concatenate(v)
        ax[0].hist(np.clip(v, 0, 10), bins=np.arange(0, 10.5, .25), alpha=.6, label=f"{p} (exactly 1s: {np.mean(v == 1):.1%})", color=col[p])
        big = v[v > 5]
        ax[1].hist(np.log10(big), bins=30, alpha=.6, label=f"{p} ({len(big)} gaps > 5s)", color=col[p])
    ax[0].set(xlabel="time step (s), clipped at 10", ylabel="count", yscale="log", title="Sampling interval"); ax[0].legend()
    ax[1].set(xlabel="log10(gap seconds), gaps > 5s", ylabel="count", title="Large GPS gaps"); ax[1].legend()
    fig.tight_layout(); fig.savefig(F("03_sampling_intervals_and_gaps.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p, s in pts.groupby("partition"):
        ax[0].hist(np.clip(s.label_dist_m, 0, 100), bins=50, alpha=.6, label=p, density=True, color=col[p])
    ax[0].set(xlabel="distance GPS point -> its reference arc (m), clipped at 100", ylabel="density", title="GPS noise vs reference route"); ax[0].legend()
    t = inv.sort_values("cand_miss", ascending=False).head(10)
    ax[1].barh(t.trip_id + " (" + t.partition + ")", t.cand_miss * 100, color=[col[p] for p in t.partition])
    ax[1].set(xlabel="candidate miss rate (%)", title="Trips with highest miss rate (reference limitation)"); ax[1].invert_yaxis()
    fig.tight_layout(); fig.savefig(F("04_noise_and_reference_quality.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(inv.n_nodes, inv.weak_components, c=inv.partition.map(col), s=30)
    ax[0].set(xlabel="nodes", ylabel="weakly connected components", xscale="log", yscale="log", title="Road-network connectivity")
    ax[1].bar(["bidirectional\narcs (mean)", "largest weak comp\n(mean node frac)", "largest strong comp\n(mean node frac)"],
              [inv.reverse_arc_frac.mean(), inv.largest_weak_frac.mean(), inv.largest_strong_frac.mean()], color="#4c72b0")
    ax[1].set(ylim=(0, 1.05), title="Network structure (all 90 trips)")
    fig.tight_layout(); fig.savefig(F("05_network_connectivity.png"), dpi=130); plt.close(fig)

    # example trajectories: GPS vs reference route (static)
    ex = [("train", "00000001"), ("train", "00000054"), ("student_test", "00000000")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, (part, tid) in zip(ax, ex):
        t = load_trip(os.path.join(ROOT, "data", part, tid))
        tf = make_projector(t["track"][:, 0], t["track"][:, 1])
        N = project(tf, t["nodes"][:, 0], t["nodes"][:, 1]); P = project(tf, t["track"][:, 0], t["track"][:, 1])
        ar, rt = t["arcs"], t["route"]
        lo, hi = P.min(0) - 150, P.max(0) + 150
        near = np.where((N[ar[:, 0], 0] > lo[0]) & (N[ar[:, 0], 0] < hi[0]) & (N[ar[:, 0], 1] > lo[1]) & (N[ar[:, 0], 1] < hi[1]))[0]
        a.plot(np.c_[N[ar[near, 0], 0], N[ar[near, 1], 0]].T, np.c_[N[ar[near, 0], 1], N[ar[near, 1], 1]].T, color="#bbbbbb", lw=.5)
        a.plot(np.c_[N[ar[rt, 0], 0], N[ar[rt, 1], 0]].T, np.c_[N[ar[rt, 0], 1], N[ar[rt, 1], 1]].T, color="#2ca02c", lw=2)
        a.scatter(P[:, 0], P[:, 1], s=1, c="#1f77b4", zorder=3)
        a.set(title=f"{part} trip {tid}: {len(P)} GPS pts, {len(rt)} route arcs", aspect="equal"); a.set_xticks([]); a.set_yticks([])
    ax[0].plot([], [], c="#bbb", label="road network"); ax[0].plot([], [], c="#2ca02c", label="reference route")
    ax[0].scatter([], [], c="#1f77b4", s=8, label="GPS"); ax[0].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(F("06_example_trajectories.png"), dpi=130); plt.close(fig)

    # ------------------------------ Folium maps ------------------------------
    for part, tid, note in [("train", "00000001", "clean trip"), ("train", "00000054", "reference route runs along the GPS corridor but ~140 m offset"),
                            ("student_test", "00000000", "student_test example")]:
        t = load_trip(os.path.join(ROOT, "data", part, tid))
        make_trip_map(t, os.path.join(OUT, "maps", f"map_{part}_{tid}.html"), title=f"{part} trip {tid} ({note})")
    print("done")


if __name__ == "__main__":
    main()
