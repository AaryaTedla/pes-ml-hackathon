"""Run every served trip through the real HTTP API; save reproducible smoke-test evidence."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import build_opener, ProxyHandler, Request

FORBIDDEN = {"route", "reference_route", "true_arc", "true_arcs", "labels", "ground_truth", "nodes", "arcs"}


def check_payload(value):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("API leaked reference or internal network data")
        for nested in value.values():
            check_payload(nested)
    elif isinstance(value, list):
        for nested in value:
            check_payload(nested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out-dir", type=Path, default=Path("results/track_d"))
    parser.add_argument("--expect-mode", choices=["nearest", "classifier", "hmm"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.api_url.rstrip("/")
    opener = build_opener(ProxyHandler({}))  # LAN requests must not go through a system proxy.

    def request(path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        req = Request(base + path, data=payload, headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=120) as response:
            return json.load(response)

    health = request("/health")
    trips = request("/trips")["trips"]
    if health["status"] != "ok" or not trips:
        raise ValueError("API is not ready or trip library is empty")
    rows = []
    for trip in trips:
        result = request("/predict", {"trip_id": trip})
        check_payload(result)
        if result["trip_id"] != trip or not result["n_points"]:
            raise ValueError("Unexpected trip or empty prediction")
        if len(result["gps"]) != result["n_points"] or len(result["predicted_arcs"]) != result["n_points"]:
            raise ValueError("Prediction and GPS lengths do not match")
        if args.expect_mode and result["mode"] != args.expect_mode:
            raise ValueError("API mode does not match the requested verification mode")
        (args.out_dir / f"prediction_{trip}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        metrics = result["metrics"] or {}
        row = {"trip_id": trip, "mode": result["mode"], "n_points": result["n_points"],
               "n_matched": result["n_matched"], "accuracy": metrics.get("pointwise_accuracy"),
               "route_continuity": metrics.get("route_continuity"), "total_ms": result["timing_ms"]["total"],
               "request_id": result["request_id"]}
        rows.append(row)
        print(f"PASS trip={trip} mode={row['mode']} points={row['n_points']} time={row['total_ms']}ms accuracy={row['accuracy']}", flush=True)
    with (args.out_dir / "smoke_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "run.json").write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(), "api_url": base,
        "health": health, "n_trips": len(rows), "purpose": "HTTP smoke test, not final model evaluation"
    }, indent=2), encoding="utf-8")
    print(f"Verified {len(rows)} trips. Evidence: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
