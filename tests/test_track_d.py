"""Track D contracts: reference isolation, dummy correctness, API and artifact boundaries."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.config import Settings
from api.main import create_app
from api.predictor import Predictor, validate_arcs
from api.scoring import continuity


@pytest.fixture
def data(tmp_path):
    trip = tmp_path / "00000001"
    trip.mkdir()
    files = {
        "nodes": "77.0000 12.0000\n77.0010 12.0000\n77.0020 12.0000\n",
        "arcs": "0 1\n1 0\n1 2\n2 1\n",
        "track": "77.0001 12.00001 0\n77.0005 12.00001 1\n77.0012 12.00001 2\n77.0018 12.00001 3\n",
        "route": "0\n2\n",
    }
    for ext, text in files.items():
        (trip / f"00000001.{ext}").write_text(text, encoding="utf-8")
    return Settings(tmp_path, tmp_path / "artifacts")


def test_nearest_uses_rank_not_heading_or_row_order(data):
    f = pd.DataFrame({"point_idx": [0, 0, 1], "arc_id": [1, 0, 2], "cand_rank": [1, 0, 0],
                      "cos_head": [1, -1, 0], "dist_m": [2, 1, 1]})
    trip = {"route": None, "arcs": np.array([[0, 1], [1, 0], [1, 2]])}
    pred, clf = Predictor(data).predict(trip, f, 3)
    assert pred.tolist() == [0, 2, -1]
    assert np.array_equal(pred, clf)


def test_api_real_features_scoring_and_no_reference_payload(data):
    with TestClient(create_app(data)) as client:
        assert client.get("/trips").json() == {"trips": ["00000001"]}
        response = client.post("/predict", json={"trip_id": "00000001"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["mode"] == "nearest" and "dummy" in result["pipeline"]
        assert len(result["gps"]) == len(result["predicted_arcs"]) == result["n_points"] == 4
        assert result["metrics"]["sequence_window50_accuracy"] is None
        assert result["metrics"]["n_points_scored"] == 4
        assert response.headers["x-request-id"] == result["request_id"]
        forbidden = {"route", "reference_route", "true_arc", "true_arcs", "labels", "ground_truth", "nodes", "arcs"}
        def check(value):
            if isinstance(value, dict):
                assert not forbidden.intersection(value)
                for nested in value.values(): check(nested)
            elif isinstance(value, list):
                for nested in value: check(nested)
        check(result)
        assert client.get("/data/00000001/00000001.route").status_code == 404
        assert client.get("/results/maps/report.html").status_code == 404


def test_prediction_unchanged_when_reference_is_removed_or_changed(data):
    predictor = Predictor(data)
    directory = data.trips_dir / "00000001"
    first = predictor.predict_trip(directory)
    route = directory / "00000001.route"
    route.write_text("1\n3\n", encoding="utf-8")
    second = predictor.predict_trip(directory)
    route.unlink()
    third = predictor.predict_trip(directory)
    assert first.trip["route"] is second.trip["route"] is third.trip["route"] is None
    assert np.array_equal(first.arcs, second.arcs) and np.array_equal(first.arcs, third.arcs)
    with TestClient(create_app(data)) as client:
        result = client.post("/predict", json={"trip_id": "00000001"}).json()
        assert result["metrics"] is None
        assert result["predicted_arcs"] == first.arcs.tolist()


def test_scoring_happens_only_after_prediction(data, monkeypatch):
    predictor = Predictor(data)
    original = predictor.predict_trip
    events = []
    def predict(directory):
        result = original(directory)
        events.append("predicted")
        return result
    def score(*args):
        assert events == ["predicted"]
        events.append("scored")
        return None
    monkeypatch.setattr(predictor, "predict_trip", predict)
    monkeypatch.setattr("api.main.score", score)
    with TestClient(create_app(data, predictor)) as client:
        assert client.post("/predict", json={"trip_id": "00000001"}).status_code == 200
    assert events == ["predicted", "scored"]


@pytest.mark.parametrize("body", [{"trip_id": "../00000001"}, {"trip_id": 1}, {},
                                      {"trip_id": "00000001", "route": [0]}])
def test_bad_requests_rejected(data, body):
    with TestClient(create_app(data)) as client:
        assert client.post("/predict", json=body).status_code == 422


def test_missing_trip_data_and_incomplete_directories(data, tmp_path):
    (data.trips_dir / "unrelated").mkdir()
    with TestClient(create_app(data)) as client:
        assert client.get("/trips").json()["trips"] == ["00000001"]
        assert client.post("/predict", json={"trip_id": "absent"}).status_code == 404
        assert client.post("/predict", json={"trip_id": "unrelated"}).status_code == 422
    with TestClient(create_app(Settings(tmp_path / "missing", tmp_path))) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/trips").status_code == 503


def test_cors_and_error_logs(data, caplog):
    with TestClient(create_app(data)) as client:
        response = client.options("/predict", headers={"Origin": "http://laptop2:8501",
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        assert response.headers["access-control-allow-origin"] == "*"
        with caplog.at_level("INFO", logger="pathsnap"):
            client.post("/predict", json={"trip_id": "absent"})
        assert "trip=absent" in caplog.text and "status=404" in caplog.text and "accuracy=n/a" in caplog.text


def test_workspace_and_local_assets_are_served(data):
    with TestClient(create_app(data)) as client:
        response = client.get("/")
        assert response.status_code == 200 and "PathSnap" in response.text
        assert response.url.path == "/ui/"
        assert client.get("/ui/app.js").status_code == 200
        assert client.get("/ui/styles.css").status_code == 200
        assert client.get("/ui/vendor/leaflet/leaflet.js").status_code == 200


def test_requested_model_mode_fails_without_artifacts(data):
    settings = Settings(data.trips_dir, data.artifacts_dir, "classifier")
    with pytest.raises(FileNotFoundError), TestClient(create_app(settings)):
        pass


def test_real_serialized_sklearn_pipeline_and_feature_order_guard(data):
    import json
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from features import FEATURE_COLS
    # Synthetic artifact is temporary test data, never a claimed trained PathSnap model.
    X = pd.DataFrame(np.arange(8 * len(FEATURE_COLS)).reshape(8, -1), columns=FEATURE_COLS)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=548)).fit(X, [0, 0, 0, 0, 1, 1, 1, 1])
    data.artifacts_dir.mkdir()
    joblib.dump(model, data.artifacts_dir / "model.joblib")
    metadata = data.artifacts_dir / "feature_columns.json"
    metadata.write_text(json.dumps(FEATURE_COLS), encoding="utf-8")
    predictor = Predictor(Settings(data.trips_dir, data.artifacts_dir, "classifier"))
    assert np.allclose(predictor._probabilities(X), model.predict_proba(X)[:, 1])
    metadata.write_text(json.dumps(list(reversed(FEATURE_COLS))), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_columns"):
        Predictor(Settings(data.trips_dir, data.artifacts_dir, "classifier"))


@pytest.mark.parametrize("values", [[0, 9], [0, -2], [0.5, 1.0], [[0, 1]]])
def test_invalid_decoder_output_rejected(values):
    with pytest.raises(ValueError): validate_arcs(values, 2, 4)


def test_unmatched_points_do_not_create_connected_shortcuts():
    arcs = np.array([[0, 1], [1, 2]])
    assert continuity(np.array([0, -1, 1]), arcs) == 0
    assert continuity(np.array([-1, -1]), arcs) is None


@pytest.mark.parametrize("three_args", [False, True])
def test_classifier_probability_class_order_and_decoder_contract(data, three_args):
    predictor = Predictor(data)
    from features import FEATURE_COLS
    predictor.settings = Settings(data.trips_dir, data.artifacts_dir, "hmm")
    # Positive class deliberately first: cannot assume probabilities[:, 1].
    predictor.model = SimpleNamespace(classes_=np.array([1, 0]),
        predict_proba=lambda X: np.array([[.9, .1], [.1, .9]]))
    features = pd.DataFrame(np.zeros((2, len(FEATURE_COLS))), columns=FEATURE_COLS)
    features["point_idx"] = [0, 0]; features["arc_id"] = [0, 1]
    def decode(trip, f):
        assert trip["route"] is None and f["prob"].tolist() == [.9, .1]
        assert "label" not in f
        return [1]
    def decode_config(trip, f, cfg):
        assert cfg == {"gap_s": 5}
        return decode(trip, f)
    predictor.decode = decode_config if three_args else decode
    predictor.hmm_cfg = {"gap_s": 5}
    pred, classifier = predictor.predict({"route": None, "arcs": np.array([[0, 1], [1, 0]])}, features, 1)
    assert pred.tolist() == [1] and classifier.tolist() == [0]


def test_offline_map_reference_is_explicit(data, monkeypatch, tmp_path):
    from api.export_map import export_map
    result = Predictor(data).predict_trip(data.trips_dir / "00000001")
    trip = dict(result.trip, route=np.array([0, 2]))
    calls = []
    def capture(trip, *args, **kwargs):
        calls.append(trip["route"])
        assert (kwargs["pred_arcs"] >= 0).all()
    monkeypatch.setattr("api.export_map.make_trip_map", capture)
    export_map(trip, tmp_path / "safe.html", result.arcs)
    export_map(trip, tmp_path / "report.html", result.arcs, include_reference=True)
    assert calls[0] is None and calls[1].tolist() == [0, 2]
    assert trip["route"].tolist() == [0, 2]


def test_offline_map_with_no_matched_arcs(data, tmp_path):
    from api.export_map import export_map
    result = Predictor(data).predict_trip(data.trips_dir / "00000001")
    path = tmp_path / "unmatched.html"
    export_map(result.trip, path, np.full(len(result.arcs), -1))
    assert "no matched arcs" in path.read_text(encoding="utf-8")
