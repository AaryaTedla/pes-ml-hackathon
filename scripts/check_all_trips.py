"""Rehearsal check: calls the running API for every trip it lists and prints one table.
    python scripts/check_all_trips.py                           # API on this laptop
    python scripts/check_all_trips.py --api http://10.30.203.241:8000   # from the other laptop
Exit code is non-zero if any trip fails, so it can be used as a pre-demo gate."""
import argparse, sys, time
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--expect", type=int, default=10, help="expected number of trips")
    ap.add_argument("--csv", help="also save the per-trip table (aggregate metrics only) to this CSV file")
    a = ap.parse_args()
    try:
        h = requests.get(a.api + "/health", timeout=10).json()
        trips = requests.get(a.api + "/trips", timeout=10).json()["trips"]
    except Exception as e:
        print(f"CANNOT REACH API at {a.api}: {e}\n(check the API is running, same Wi-Fi, firewall rule for port 8000)")
        return 2
    print(f"API {a.api} | mode={h.get('mode')} | pipeline={h['pipeline']} | trips={len(trips)}")
    print(f"{'trip':10s}{'points':>7s}{'acc(clf)':>10s}{'acc(final)':>11s}{'continuity':>11s}{'ms':>7s}  status")
    bad = 0
    rows = []
    for t in trips:
        t0 = time.time()
        try:
            r = requests.post(a.api + "/predict", json={"trip_id": t}, timeout=120)
            r.raise_for_status()
            j = r.json(); m = j.get("metrics") or {}
            f = lambda k: f"{m[k]:.3f}" if m.get(k) is not None else "n/a"
            rows.append(dict(trip_id=t, points=j["n_points"], pipeline=j["pipeline"],
                             accuracy_classifier_only=m.get("pointwise_accuracy_classifier_only"),
                             accuracy_final=m.get("pointwise_accuracy"),
                             accuracy_final_reference_covered=m.get("pointwise_accuracy_reference_covered"),
                             fraction_points_reference_covered=m.get("frac_points_reference_covered"),
                             route_continuity=m.get("route_continuity"),
                             exact_50_point_windows=m.get("sequence_window50_accuracy"),
                             points_scored=m.get("n_points_scored"),
                             request_ms=round((time.time() - t0) * 1000), request_id=j.get("request_id")))
            print(f"{t:10s}{j['n_points']:7d}{f('pointwise_accuracy_classifier_only'):>10s}{f('pointwise_accuracy'):>11s}"
                  f"{f('route_continuity'):>11s}{(time.time()-t0)*1000:7.0f}  OK")
        except Exception as e:
            bad += 1
            print(f"{t:10s}{'':>7s}{'':>10s}{'':>11s}{'':>11s}{'':>7s}  FAILED: {e}")
    if len(trips) != a.expect:
        print(f"WARNING: expected {a.expect} trips, API lists {len(trips)}"); bad += 1
    if a.csv and rows:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"saved {a.csv}")
    print("ALL OK" if not bad else f"{bad} PROBLEM(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
