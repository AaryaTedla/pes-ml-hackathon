"""Laptop 1: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import ROOT, Settings
from .predictor import Predictor
from .scoring import score

log = logging.getLogger("pathsnap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    trip_id: str = Field(min_length=1, max_length=64, pattern=ID.pattern)


def trip_directory(root: Path, trip_id: str) -> Path:
    if not ID.fullmatch(trip_id):
        raise HTTPException(400, "Invalid trip id")
    path = (root / trip_id).resolve()
    if path.parent != root.resolve() or not path.is_dir():
        raise HTTPException(404, "Trip not found")
    if not all((path / f"{trip_id}.{ext}").is_file() for ext in ("track", "nodes", "arcs")):
        raise HTTPException(422, "Trip is missing a track, nodes or arcs file")
    return path


def list_trips(root: Path):
    if not root.is_dir():
        raise HTTPException(503, "Trip data is unavailable. Set PATHSNAP_TRIPS_DIR on the API laptop.")
    found = []
    for path in root.iterdir():
        if path.is_dir() and ID.fullmatch(path.name):
            try:
                trip_directory(root, path.name)
            except HTTPException:
                continue
            found.append(path.name)
    return sorted(found)


def create_app(settings: Settings | None = None, predictor: Predictor | None = None):
    cfg = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = predictor or Predictor(cfg)
        try:
            count = len(list_trips(cfg.trips_dir))
        except HTTPException:
            count = 0
            log.warning("DATA UNAVAILABLE | set PATHSNAP_TRIPS_DIR=%s", cfg.trips_dir)
        log.info("READY mode=%s trips=%d data=%s", app.state.predictor.pipeline, count, cfg.trips_dir)
        yield

    app = FastAPI(title="PathSnap", version="1.0.0", lifespan=lifespan)
    app.state.settings = cfg
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.cors_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        start = perf_counter()
        request.state.request_id = uuid4().hex[:12]
        request.state.trip_id = "-"
        request.state.n_points = "-"
        request.state.accuracy = "n/a"
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Request failed id=%s", request.state.request_id)
            response = JSONResponse(status_code=500, content={"detail": "Request failed; see API terminal.",
                                                              "request_id": request.state.request_id})
        response.headers["X-Request-ID"] = request.state.request_id
        if request.url.path in {"/health", "/trips", "/predict"} or response.status_code >= 400:
            log.info("REQUEST id=%s %s %s status=%d trip=%s points=%s time=%.1fms accuracy=%s",
                     request.state.request_id, request.method, request.url.path, response.status_code,
                     request.state.trip_id, request.state.n_points, (perf_counter() - start) * 1000,
                     request.state.accuracy)
        return response

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "pipeline": request.app.state.predictor.pipeline,
                "mode": cfg.mode, "n_trips": len(list_trips(cfg.trips_dir))}

    @app.get("/trips")
    def trips():
        return {"trips": list_trips(cfg.trips_dir)}

    @app.post("/predict")
    def predict(req: PredictRequest, request: Request):
        start = perf_counter()
        request.state.trip_id = req.trip_id
        directory = trip_directory(cfg.trips_dir, req.trip_id)
        result = request.app.state.predictor.predict_trip(directory)
        trip, pred = result.trip, result.arcs
        request.state.n_points = len(pred)
        # Prediction completes before the only reference-aware operation is invoked.
        scoring_start = perf_counter()
        metrics = score(directory, pred, result.classifier_arcs, trip["arcs"])
        scoring_ms = (perf_counter() - scoring_start) * 1000
        if metrics and metrics["pointwise_accuracy"] is not None:
            request.state.accuracy = f"{metrics['pointwise_accuracy']:.4f}"
        arc_ids = np.unique(pred[pred >= 0])
        nodes, arcs = trip["nodes"], trip["arcs"]
        a, b = nodes[arcs[arc_ids, 0]], nodes[arcs[arc_ids, 1]]
        segments = np.stack([a[:, [1, 0]], b[:, [1, 0]]], axis=1).tolist()
        return {
            "trip_id": req.trip_id, "request_id": request.state.request_id,
            "pipeline": result.pipeline, "mode": cfg.mode,
            "n_points": len(pred), "n_matched": int((pred >= 0).sum()),
            "gps": trip["track"][:, [1, 0]].tolist(),
            "predicted_route": segments, "predicted_arcs": pred.tolist(), "metrics": metrics,
            "timing_ms": {"features": round(result.feature_ms, 2), "predict": round(result.predict_ms, 2),
                          "scoring": round(scoring_ms, 2), "total": round((perf_counter() - start) * 1000, 2)},
        }

    ui = ROOT / "ui"
    if ui.is_dir():
        @app.get("/", include_in_schema=False)
        def index():
            return RedirectResponse("/ui/")
        app.mount("/ui", StaticFiles(directory=ui, html=True), name="ui")
    return app


app = create_app()
