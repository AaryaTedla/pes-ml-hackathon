"""Reference-free prediction boundary. Only this module connects model/decoder handoffs."""
from dataclasses import dataclass
import importlib
import inspect
import json
from time import perf_counter

import numpy as np

from .config import Settings
from pipeline import prepare_trip
from features import FEATURE_COLS


@dataclass
class Prediction:
    trip: dict
    arcs: np.ndarray
    classifier_arcs: np.ndarray
    pipeline: str
    feature_ms: float
    predict_ms: float


def validate_arcs(values, n_points, n_arcs):
    """-1 means no candidate. Reject broken decoder output before map indexing."""
    values = np.asarray(values)
    if values.shape != (n_points,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Prediction must contain one integer arc id per GPS point")
    if np.any(values < -1) or np.any(values >= n_arcs):
        raise ValueError("Prediction contains an out-of-range arc id")
    return values.astype(np.int64, copy=False)


class Predictor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self.scaler = self.decode = None
        self.hmm_cfg = {}
        self.pipeline = {"nearest": "dummy / nearest arc", "classifier": "classifier",
                         "hmm": "classifier + HMM/Viterbi"}[settings.mode]
        if settings.mode == "nearest":
            return
        import joblib
        art = settings.artifacts_dir
        # Explicit modes fail if a handoff is missing; never silently present a dummy as ML.
        self.model = joblib.load(art / "model.joblib")
        if (art / "scaler.joblib").exists():
            self.scaler = joblib.load(art / "scaler.joblib")
        if (art / "feature_columns.json").exists():
            columns = json.loads((art / "feature_columns.json").read_text(encoding="utf-8"))
            if columns != FEATURE_COLS:
                raise ValueError("Model feature_columns.json differs from Track A FEATURE_COLS")
        for obj in (self.scaler, self.model):
            if obj is not None and hasattr(obj, "feature_names_in_"):
                if list(obj.feature_names_in_) != FEATURE_COLS:
                    raise ValueError("Artifact feature order differs from Track A FEATURE_COLS")
            if obj is not None and getattr(obj, "n_features_in_", len(FEATURE_COLS)) != len(FEATURE_COLS):
                raise ValueError("Artifact feature count differs from Track A FEATURE_COLS")
        if settings.mode == "hmm":
            self.decode = importlib.import_module("hmm").decode
            self.hmm_cfg = json.loads((art / "hmm_config.json").read_text(encoding="utf-8"))

    def _probabilities(self, feats):
        X = feats[FEATURE_COLS]
        if self.scaler is not None:
            X = self.scaler.transform(X if hasattr(self.scaler, "feature_names_in_") else X.to_numpy())
        elif not hasattr(self.model, "feature_names_in_"):
            X = X.to_numpy()
        if hasattr(self.model, "feature_names_in_") and not hasattr(X, "columns"):
            import pandas as pd
            X = pd.DataFrame(X, columns=FEATURE_COLS, index=feats.index)
        probs = np.asarray(self.model.predict_proba(X))
        classes = list(self.model.classes_)
        if 1 not in classes or probs.shape != (len(feats), len(classes)):
            raise ValueError("Model must expose classes_ including 1 and predict_proba(X)")
        p = probs[:, classes.index(1)]
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Model returned invalid probabilities")
        return p

    def predict(self, trip, feats, n_points):
        """Return final and classifier-only arcs, with no labels or reference route."""
        if trip.get("route") is not None:
            raise ValueError("Reference route must not enter prediction")
        clf = np.full(n_points, -1, dtype=np.int64)
        probs = None
        if self.settings.mode == "nearest":
            best = feats.loc[feats["cand_rank"] == 0]
        elif len(feats):
            probs = self._probabilities(feats)
            best = feats.assign(_prob=probs).sort_values(
                ["point_idx", "_prob", "arc_id"], ascending=[True, False, True]
            ).drop_duplicates("point_idx")
        else:
            best = feats
        if len(best):
            clf[best["point_idx"].to_numpy(dtype=int)] = best["arc_id"].to_numpy(dtype=int)
        pred = clf.copy()
        if self.decode is not None and len(feats):
            candidates = feats.assign(prob=probs)
            # Support the two-argument context contract and the configurable three-argument handoff.
            try:
                inspect.signature(self.decode).bind(trip, candidates, self.hmm_cfg)
            except TypeError:
                pred = self.decode(trip, candidates)
            else:
                pred = self.decode(trip, candidates, self.hmm_cfg)
        return validate_arcs(pred, n_points, len(trip["arcs"])), validate_arcs(clf, n_points, len(trip["arcs"]))

    def predict_trip(self, trip_dir) -> Prediction:
        """The API's sole feature entry: prepare_trip never loads a .route file."""
        start = perf_counter()
        trip, feats, _points = prepare_trip(str(trip_dir))
        for key, width in (("track", 3), ("nodes", 2), ("arcs", 2)):
            arr = trip.get(key)
            if arr is None or arr.ndim != 2 or arr.shape[1] != width or not len(arr):
                raise ValueError(f"Trip has missing or invalid {key} data")
        feature_end = perf_counter()
        pred, clf = self.predict(trip, feats, len(trip["track"]))
        return Prediction(trip, pred, clf, self.pipeline,
                          (feature_end - start) * 1000, (perf_counter() - feature_end) * 1000)
