"""PathSnap FastAPI backend (runs on Laptop 1).

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Env vars:
    PATHSNAP_TRIPS_DIR   folder holding the trip folders to serve (default: data/student_test)
"""
import os, sys, time, logging
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import prepare_trip
from evaluate import ordered_unique
from predictor import Predictor
from scoring import score

TRIPS_DIR = os.path.abspath(os.environ.get("PATHSNAP_TRIPS_DIR", os.path.join(ROOT, "data", "student_test")))
log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

app = FastAPI(title="PathSnap map-matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # UI is on another laptop
predictor = Predictor()


class PredictRequest(BaseModel):
    trip_id: str


def _trip_dir(trip_id):
    if not trip_id.isalnum():                      # no path tricks
        raise HTTPException(400, "bad trip id")
    d = os.path.join(TRIPS_DIR, trip_id)
    if not os.path.isdir(d):
        raise HTTPException(404, f"trip {trip_id} not found")
    return d


@app.on_event("startup")
def _startup():
    log.info("PathSnap API up | trips dir: %s | %d trips | pipeline: %s",
             TRIPS_DIR, len(_list_trips()), predictor.pipeline)


def _list_trips():
    return sorted(x for x in os.listdir(TRIPS_DIR) if os.path.isdir(os.path.join(TRIPS_DIR, x)))


@app.get("/health")
def health():
    return {"status": "ok", "pipeline": predictor.pipeline, "n_trips": len(_list_trips())}


@app.get("/trips")
def trips():
    return {"trips": _list_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    d = _trip_dir(req.trip_id)
    log.info("REQUEST  POST /predict trip=%s", req.trip_id)
    trip, feats, points = prepare_trip(d)                    # never opens the .route file
    t1 = time.time()
    n = len(trip["track"])
    pred, pipeline, clf = predictor.predict(trip, feats, n)
    t2 = time.time()
    metrics = score(d, pred, clf, trip["arcs"])              # server-side only, after predicting
    t3 = time.time()

    ok = pred >= 0
    route = ordered_unique(pred[ok])
    nodes, arcs = trip["nodes"], trip["arcs"]
    a, b = nodes[arcs[route, 0]], nodes[arcs[route, 1]]
    seg = [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(a, b)]   # (lat, lon)
    gps = np.round(trip["track"][:, [1, 0]], 6).tolist()
    resp = {
        "trip_id": req.trip_id, "pipeline": pipeline, "n_points": n,
        "gps": gps, "predicted_route": seg, "predicted_arcs": pred.tolist(),
        "metrics": metrics,
        "timing_ms": {"features": round((t1 - t0) * 1e3), "predict": round((t2 - t1) * 1e3),
                      "scoring": round((t3 - t2) * 1e3), "total": round((t3 - t0) * 1e3)},
    }
    log.info("RESPONSE trip=%s points=%d pipeline=%s accuracy=%s continuity=%s time=%dms",
             req.trip_id, n, pipeline,
             f"{metrics['pointwise_accuracy']:.4f}" if metrics else "n/a",
             f"{metrics['route_continuity']:.4f}" if metrics else "n/a", resp["timing_ms"]["total"])
    return resp
