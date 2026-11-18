"""Prediction back-end used by the API.

Pipeline (what is actually used is reported back in the API response as `pipeline`):
  1. baseline  : heuristic score per candidate (NOT a trained model) -> argmax per point
  2. model     : artifacts/model.joblib (+ optional scaler.joblib) -> predict_proba -> argmax per point
  3. model+hmm : the above probabilities decoded with src/hmm.py `decode` and artifacts/hmm_config.json

The baseline exists so the API and UI can be built and tested before the trained model and
decoder arrive. When the artifacts appear, they are picked up automatically at start-up.
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from features import FEATURE_COLS

ART = os.path.join(ROOT, "artifacts")


class Predictor:
    def __init__(self):
        self.model = self.scaler = self.decode = self.hmm_cfg = None
        mp = os.path.join(ART, "model.joblib")
        if os.path.exists(mp):
            import joblib
            self.model = joblib.load(mp)
            sp = os.path.join(ART, "scaler.joblib")
            self.scaler = joblib.load(sp) if os.path.exists(sp) else None
        hp = os.path.join(ART, "hmm_config.json")
        if os.path.exists(hp):
            try:
                from hmm import decode
                self.decode = decode
                self.hmm_cfg = json.load(open(hp))
            except Exception as e:                  # fall back to no-HMM, but say so
                print(f"[predictor] HMM not loaded: {e!r}", flush=True)
        base = "classifier" if self.model is not None else "baseline heuristic (no trained model)"
        self.pipeline = base + (" + HMM/Viterbi" if self.decode is not None else "")

    def _scores(self, feats):
        if self.model is None:
            # heuristic: close to the arc and heading along it, softmax-normalised per GPS point
            s = pd.Series((-feats["dist_m"].to_numpy() / 10.0 + 3.0 * feats["cos_head"].to_numpy()).astype(float))
            key = feats["point_idx"].to_numpy()
            e = np.exp(s - s.groupby(key).transform("max"))
            return (e / e.groupby(key).transform("sum")).to_numpy()
        X = feats[FEATURE_COLS].to_numpy()
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, trip, feats, n_points):
        """Return (arc id per GPS point [n_points], pipeline name, classifier-only arcs [n_points])."""
        s = self._scores(feats)
        pi = feats["point_idx"].to_numpy()
        ai = feats["arc_id"].to_numpy()
        order = np.lexsort((-s, pi))                       # per point, best score first
        first = np.r_[True, pi[order][1:] != pi[order][:-1]]
        best = order[first]
        clf = np.full(n_points, -1, np.int64)
        clf[pi[best]] = ai[best]
        if self.decode is not None:
            f = feats.assign(prob=s)
            arcs = np.asarray(self.decode(trip, f, self.hmm_cfg))
            return arcs, self.pipeline, clf
        return clf, self.pipeline, clf
