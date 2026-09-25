"""Tests for the Version 0.9 backend API (synthetic data only, no DRIAMS, no saved project model).

`models/` is gitignored, so continuous integration has no real bundle. Every test here therefore fits a
small synthetic model, saves it, and serves that, which also means no test can accidentally depend on a
protected split or on the project model's numbers.

The safety-critical tests are the ones that assert what the API does *not* do: leak an identifier, echo an
uploaded filename, serve a fingerprint, return a traceback, load an uploaded file as code, or emit a
non-finite float.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import PROJECT_MODEL_VERSION, Limits, create_app, default_model_path, default_zones_path
from src.api_schemas import FORBIDDEN_IN_RESPONSES, finite_or_none
from src.predict import BUNDLE_FORMAT, DISCLAIMER, save_bundle
from src.preprocessing import PreprocessingConfig
from src.train import ModelSpec, build_pipeline
from src.uncertainty import RESISTANT, SUSCEPTIBLE, UNCERTAIN, ZoneRule, Zones
from src.utils import load_config, project_path
from tests.test_preprocessing import synthetic_spectrum, write_spectrum

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import predict_spectrum  # noqa: E402

# A DRIAMS-shaped identifier, invented here. Raw DRIAMS files are named after their spectrum UUID and
# repeat it in their '#' comment lines, so this is exactly the string the API must never echo or log.
FAKE_UUID = "1f2e3d4c-5b6a-7908-1234-abcdef012345"


# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------

def make_bundle(n_features: int = 6000) -> dict:
    rng = np.random.default_rng(0)
    X = rng.random((60, 6000)).astype(np.float32) * 1e-4
    y = (rng.random(60) < 0.4).astype(int)
    X[y == 1, :50] += 1e-3
    pcfg = PreprocessingConfig()
    pipe = build_pipeline(ModelSpec("lr", "logistic_regression", {"max_iter": 500}, scale=True), seed=42).fit(X, y)
    return {"format": BUNDLE_FORMAT, "pipeline": pipe, "model_version": "test-lr",
            "species": "Escherichia coli", "antibiotic": "Ciprofloxacin", "threshold": 0.5,
            "n_features": n_features, "preprocessing": pcfg.to_dict(),
            "feature_fingerprint": pcfg.fingerprint(),
            "model": "logistic_regression", "model_kind": "sklearn",
            "params": {"max_iter": 500}, "label_map": {"0": "susceptible (S)", "1": "resistant (R or I)"},
            "threshold_rule": {"rule": "fixed"}, "dataset": {"name": "synthetic"},
            "split": {"train_sites": ["DRIAMS-SYNTH"], "train_size": 60}}


@pytest.fixture()
def spectrum(tmp_path) -> Path:
    """A valid synthetic raw spectrum, named and commented like a real DRIAMS file."""
    mz, intensity = synthetic_spectrum()
    path = write_spectrum(tmp_path / f"{FAKE_UUID}.txt", mz, intensity)
    body = path.read_text(encoding="utf-8")
    path.write_text(f"#  /some/path/{FAKE_UUID}/1SLin/fid\n#  {FAKE_UUID}\n{body}", encoding="utf-8")
    return path


@pytest.fixture()
def zones_file(tmp_path) -> Path:
    """Zones shaped like the committed Version 0.6 file: a susceptible edge and no resistant edge."""
    zones = Zones(threshold=0.5, lower=0.2, upper=None, rule=ZoneRule(), fitted_on="validation", n_fitted=100)
    path = tmp_path / "uncertainty.json"
    path.write_text(json.dumps(zones.to_dict()), encoding="utf-8")
    return path


def build_app(tmp_path, *, bundle: dict | None = None, zones: Path | None = None,
              model_path: Path | None = None, limits: dict | None = None):
    """An app serving a synthetic bundle, with optionally tightened limits so tests stay small and fast."""
    config = copy.deepcopy(load_config())
    if limits:
        config["api"]["limits"].update(limits)
    if model_path is None:
        model_path = save_bundle(bundle or make_bundle(), tmp_path / "models" / "m.joblib")
    return create_app(config, model_path=model_path, zones_path=zones)


def post_spectrum(client, path: Path, *, name: str | None = None, endpoint: str = "/predict"):
    return client.post(endpoint, files={"file": (name or path.name, path.read_bytes(), "text/plain")})


def error_of(response) -> tuple[str, str]:
    body = response.json()
    return body["error"]["type"], body["error"]["message"]


# ------------------------------------------------------------------------------------------------
# Happy paths
# ------------------------------------------------------------------------------------------------

def test_health_reports_ok_when_the_model_loaded(tmp_path, zones_file):
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["zones_loaded"] is True
    assert body["model_version"] == "test-lr" and body["detail"] is None
    assert body["uptime_s"] >= 0


def test_predict_returns_every_documented_field(tmp_path, spectrum, zones_file):
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        r = post_spectrum(client, spectrum)
    assert r.status_code == 200
    body = r.json()
    expected = {"species", "antibiotic", "prediction", "resistance_probability", "threshold",
                "model_version", "preprocessing_ms", "inference_ms", "total_ms", "confidence",
                "explanation", "disclaimer"}
    assert expected <= set(body)
    assert 0.0 <= body["resistance_probability"] <= 1.0
    assert body["prediction"] in ("Resistant", "Susceptible")
    assert body["confidence"] in (SUSCEPTIBLE, UNCERTAIN, RESISTANT)
    assert body["disclaimer"] == DISCLAIMER
    assert all(body[k] >= 0 for k in ("preprocessing_ms", "inference_ms", "total_ms"))


def test_predict_without_zones_reports_no_confidence(tmp_path, spectrum):
    """A missing zone file is not an error: the prediction stands, it carries no three-way label."""
    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        r = post_spectrum(client, spectrum)
        assert client.get("/health").json()["zones_loaded"] is False
    assert r.status_code == 200 and r.json()["confidence"] == "not available"


def test_predict_can_explain_when_asked(tmp_path, spectrum):
    """A linear model cannot give exact contributions, so the explanation stays absent rather than wrong."""
    with TestClient(build_app(tmp_path)) as client:
        r = client.post("/predict?explain=true",
                        files={"file": ("s.txt", spectrum.read_bytes(), "text/plain")})
    assert r.status_code == 200 and r.json()["explanation"] is None


def test_advice_accompanies_an_uncertain_call_only(tmp_path, spectrum, zones_file):
    """`advice` is present exactly when the label is the uncertain one, as in the command-line output."""
    wide = tmp_path / "wide.json"      # a zone covering everything makes every call uncertain
    wide.write_text(json.dumps(Zones(0.5, 0.0, 1.0, ZoneRule(), "validation", 100).to_dict()), encoding="utf-8")
    with TestClient(build_app(tmp_path, zones=wide)) as client:
        body = post_spectrum(client, spectrum).json()
    assert body["confidence"] == UNCERTAIN
    assert "advice" in body and "susceptibility testing" in body["advice"]

    narrow = tmp_path / "narrow.json"
    narrow.write_text(json.dumps(Zones(0.5, None, None, ZoneRule(), "validation", 100).to_dict()),
                      encoding="utf-8")
    with TestClient(build_app(tmp_path, zones=narrow, model_path=tmp_path / "models" / "m.joblib")) as client:
        body = post_spectrum(client, spectrum).json()
    assert body["confidence"] == UNCERTAIN and "advice" in body


def test_batch_predict_returns_one_result_per_file_in_order(tmp_path, spectrum, zones_file):
    payload = spectrum.read_bytes()
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        r = client.post("/batch-predict", files=[("files", (f"s{i}.txt", payload, "text/plain")) for i in range(3)])
    assert r.status_code == 200
    body = r.json()
    assert body["n_submitted"] == 3 and body["n_succeeded"] == 3 and body["n_failed"] == 0
    assert [item["index"] for item in body["results"]] == [0, 1, 2]
    probabilities = {item["prediction"]["resistance_probability"] for item in body["results"]}
    assert len(probabilities) == 1          # the same spectrum three times must give the same answer
    assert body["disclaimer"] == DISCLAIMER


def test_one_bad_file_does_not_fail_the_batch(tmp_path, spectrum):
    with TestClient(build_app(tmp_path)) as client:
        r = client.post("/batch-predict", files=[
            ("files", ("good.txt", spectrum.read_bytes(), "text/plain")),
            ("files", ("wrong.csv", spectrum.read_bytes(), "text/csv")),
            ("files", ("broken.txt", b"this is not a spectrum\n", "text/plain")),
        ])
    assert r.status_code == 200
    body = r.json()
    assert body["n_submitted"] == 3 and body["n_succeeded"] == 1 and body["n_failed"] == 2
    assert body["results"][0]["prediction"]["prediction"] in ("Resistant", "Susceptible")
    assert body["results"][1]["error"]["type"] == "UnsupportedUpload"
    assert body["results"][2]["error"]["type"] == "SpectrumFormatError"


def test_model_info_describes_the_served_model(tmp_path, zones_file):
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "test-lr"
    assert body["algorithm"]["n_features"] == 6000
    assert body["target"]["species"] == "Escherichia coli" and body["target"]["antibiotic"] == "Ciprofloxacin"
    assert body["threshold"] == 0.5
    assert body["limits"]["allowed_suffixes"] == [".txt"]
    assert "does not diagnose" in body["intended_use"]
    assert body["disclaimer"] == DISCLAIMER
    # A synthetic bundle is not the project model, so it must not inherit the project model's results.
    assert body["model_version"] != PROJECT_MODEL_VERSION
    assert body["external_validation"] == [] and body["evaluation"] == []


# ------------------------------------------------------------------------------------------------
# The error list from the specification
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "content", "status", "expected"), [
    ("empty.txt", b"", 422, "empty"),
    ("no_rows.txt", b"# only a comment\n", 422, "no data rows"),
    ("ragged.txt", b"hello world\nnot numbers here\n", 422, "corrupted or not a text table"),
    ("words.txt", b"aa bb\ncc dd\nee ff\n", 422, "non-numeric"),
    ("short.txt", b"2000.0 1\n2001.0 2\n", 422, "only 2 points"),
    ("wrong.csv", b"2000.0 1\n", 415, "Unsupported file type"),
    ("model.joblib", b"\x80\x04\x95fake pickle", 415, "Unsupported file type"),
    ("nosuffix", b"2000.0 1\n", 415, "Unsupported file type"),
])
def test_invalid_uploads_get_a_clean_message(tmp_path, name, content, status, expected):
    with TestClient(build_app(tmp_path)) as client:
        r = client.post("/predict", files={"file": (name, content, "application/octet-stream")})
    assert r.status_code == status
    kind, message = error_of(r)
    assert expected in message
    assert "Traceback" not in message and 'File "' not in message


def test_an_oversized_upload_is_refused(tmp_path, spectrum):
    app = build_app(tmp_path, limits={"max_upload_bytes": 2000})
    with TestClient(app) as client:
        r = post_spectrum(client, spectrum)
    assert r.status_code == 413
    kind, message = error_of(r)
    assert kind == "UploadTooLarge" and "limit" in message


def test_a_batch_over_the_file_count_is_refused(tmp_path, spectrum):
    payload = spectrum.read_bytes()
    app = build_app(tmp_path, limits={"max_batch_files": 2})
    with TestClient(app) as client:
        r = client.post("/batch-predict",
                        files=[("files", (f"s{i}.txt", payload, "text/plain")) for i in range(3)])
    assert r.status_code == 413
    kind, message = error_of(r)
    assert kind == "TooManyFiles" and "at most 2" in message


def test_a_batch_over_the_total_byte_budget_is_refused(tmp_path, spectrum):
    """The per-file limit must not be a way to exceed the batch total by splitting the payload."""
    payload = spectrum.read_bytes()
    app = build_app(tmp_path, limits={"max_batch_bytes": len(payload) + 10})
    with TestClient(app) as client:
        r = client.post("/batch-predict",
                        files=[("files", (f"s{i}.txt", payload, "text/plain")) for i in range(3)])
    body = r.json()
    if r.status_code == 413:
        assert error_of(r)[0] == "UploadTooLarge"
    else:                                   # refused per file by the shared budget instead
        assert body["n_failed"] >= 1
        assert any(i.get("error", {}).get("type") == "UploadTooLarge" for i in body["results"])


def test_a_missing_model_degrades_rather_than_crashes(tmp_path, spectrum):
    app = build_app(tmp_path, model_path=tmp_path / "absent.joblib")
    with TestClient(app) as client:
        health = client.get("/health")
        predicted = post_spectrum(client, spectrum)
        info = client.get("/model-info")
    assert health.status_code == 503
    assert health.json()["status"] == "degraded" and health.json()["model_loaded"] is False
    assert "not found" in health.json()["detail"]
    for response in (predicted, info):
        assert response.status_code == 503
        assert "Traceback" not in error_of(response)[1]


def test_a_feature_count_mismatch_is_reported_as_a_server_problem(tmp_path, spectrum):
    """A spectrum always preprocesses to the configured bin count, so a mismatch is the bundle's fault."""
    app = build_app(tmp_path, bundle=make_bundle(n_features=123))
    with TestClient(app) as client:
        r = post_spectrum(client, spectrum)
    assert r.status_code == 503
    kind, message = error_of(r)
    assert "Wrong number of features" in message and "Traceback" not in message


# ------------------------------------------------------------------------------------------------
# Security
# ------------------------------------------------------------------------------------------------

def test_an_uploaded_model_file_is_never_loaded(tmp_path, monkeypatch):
    """A '.joblib' upload must be refused on its suffix, before anything could deserialise it."""
    import joblib

    calls: list[object] = []
    real_load = joblib.load
    monkeypatch.setattr(joblib, "load", lambda *a, **k: calls.append(a) or real_load(*a, **k))

    app = build_app(tmp_path)
    with TestClient(app) as client:
        calls.clear()                        # the start-up load of the real bundle is not what we are testing
        r = client.post("/predict", files={"file": ("payload.joblib", b"\x80\x04\x95", "application/octet-stream")})
    assert r.status_code == 415
    assert calls == [], "an uploaded file reached joblib.load"


def test_no_endpoint_accepts_a_path_or_a_model_override(tmp_path, spectrum):
    """A client must not be able to point the service at another file on disk."""
    app = build_app(tmp_path)
    with TestClient(app) as client:
        r = client.post("/predict?model=C:/Windows/System32/config",
                        files={"file": ("s.txt", spectrum.read_bytes(), "text/plain")})
        assert r.status_code == 200          # the unknown query parameter is ignored, not honoured
        assert r.json()["model_version"] == "test-lr"
        for path in ("/predict?file=C:/DRIAMS/x.txt", "/model-info?path=/etc/passwd"):
            assert client.get(path).status_code in (200, 405, 422, 503)


def test_no_error_body_contains_a_traceback(tmp_path):
    with TestClient(build_app(tmp_path)) as client:
        responses = [
            client.post("/predict", files={"file": ("x.txt", b"", "text/plain")}),
            client.post("/predict", files={"file": ("x.zip", b"PK", "application/zip")}),
            client.get("/model-info"),
        ]
    for r in responses:
        text = r.text
        assert "Traceback" not in text and 'File "' not in text and "site-packages" not in text


# ------------------------------------------------------------------------------------------------
# Privacy
# ------------------------------------------------------------------------------------------------

class Records(logging.Handler):
    """Captures the API logger, which sets propagate=False and so bypasses pytest's caplog."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_an_uploaded_filename_and_its_comments_never_reach_a_response_or_a_log(tmp_path, spectrum, zones_file):
    """The strongest privacy test: a UUID in the filename AND in the file's comment lines must vanish."""
    assert FAKE_UUID in spectrum.read_text(encoding="utf-8")          # it really is in the payload
    assert FAKE_UUID in spectrum.name                                 # and in the name we submit

    records = Records()
    for name in ("api", "", "uvicorn", "uvicorn.access"):
        logging.getLogger(name).addHandler(records)
    try:
        with TestClient(build_app(tmp_path, zones=zones_file)) as client:
            good = post_spectrum(client, spectrum)
            bad = client.post("/predict", files={"file": (f"{FAKE_UUID}.csv", b"x", "text/csv")})
            batch = client.post("/batch-predict",
                                files=[("files", (f"{FAKE_UUID}.txt", spectrum.read_bytes(), "text/plain"))])
    finally:
        for name in ("api", "", "uvicorn", "uvicorn.access"):
            logging.getLogger(name).removeHandler(records)

    assert good.status_code == 200
    for response in (good, bad, batch):
        assert FAKE_UUID not in response.text, "an identifier reached a response body"
        assert "/some/path/" not in response.text, "a source path reached a response body"
    logged = "\n".join(records.lines)
    assert FAKE_UUID not in logged, "an identifier reached the log"
    assert "/some/path/" not in logged, "a source path reached the log"


def test_error_messages_name_the_generated_file_not_the_uploaded_one(tmp_path):
    """Library messages quote path.name, so the upload must be saved under a generated name."""
    with TestClient(build_app(tmp_path)) as client:
        r = client.post("/predict", files={"file": (f"{FAKE_UUID}.txt", b"nonsense here\n", "text/plain")})
    assert r.status_code == 422
    message = error_of(r)[1]
    assert FAKE_UUID not in message
    assert "upload.txt" in message


# ------------------------------------------------------------------------------------------------
# What must never be served
# ------------------------------------------------------------------------------------------------

def test_no_response_contains_a_forbidden_string(tmp_path, spectrum, zones_file):
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        bodies = [client.get("/health").text, client.get("/model-info").text,
                  post_spectrum(client, spectrum).text]
    for text in bodies:
        for forbidden in FORBIDDEN_IN_RESPONSES:
            assert forbidden not in text, f"{forbidden!r} was served"


def test_the_forbidden_list_would_notice_a_leak():
    """A guard on the guard: the list must contain the fingerprints it is meant to catch."""
    assert "347cbd6d5d956ff9" in FORBIDDEN_IN_RESPONSES        # feature_fingerprint
    assert "151415a4d03dbcc5" in FORBIDDEN_IN_RESPONSES        # dataset / split fingerprint
    assert "git_commit" in FORBIDDEN_IN_RESPONSES
    assert any("DRIAMS" in f for f in FORBIDDEN_IN_RESPONSES)


def test_every_response_survives_strict_json(tmp_path, spectrum, zones_file):
    """starlette renders with allow_nan=False, so a NaN would become a 500 rather than bad JSON."""
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        responses = [client.get("/health"), client.get("/model-info"), post_spectrum(client, spectrum),
                     client.post("/predict", files={"file": ("x.txt", b"", "text/plain")})]
    for r in responses:
        json.dumps(r.json(), allow_nan=False)      # raises ValueError on a non-finite float


def test_finite_or_none_converts_what_a_saved_report_can_hold():
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(float("-inf")) is None
    assert finite_or_none(None) is None
    assert finite_or_none("not a number") is None
    assert finite_or_none(0.25) == 0.25
    assert finite_or_none(3) == 3.0


# ------------------------------------------------------------------------------------------------
# Honesty about the confidence zones
# ------------------------------------------------------------------------------------------------

def test_the_high_confidence_resistant_label_is_unreachable_and_said_to_be(tmp_path, zones_file):
    """The committed Version 0.6 fit has no resistant zone, so /model-info must state it, not imply it."""
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        zones = client.get("/model-info").json()["confidence_zones"]
    assert zones["resistant_edge"] is None
    assert RESISTANT not in zones["possible_labels"]
    assert zones["possible_labels"] == [SUSCEPTIBLE, UNCERTAIN]
    assert "No high-confidence resistant zone exists" in zones["note"]


def test_the_committed_zone_file_really_has_no_resistant_edge():
    """Guards the claim above against the real artifact, which is committed and so present in CI."""
    path = project_path("results/metrics/v0.6") / "ecoli_ciprofloxacin" / "uncertainty.json"
    if not path.is_file():
        pytest.skip("the Version 0.6 zone file is not in this checkout")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["upper"] is None
    assert data["sides"]["resistant"]["exists"] is False


# ------------------------------------------------------------------------------------------------
# Parity with the command-line tool, and non-regression
# ------------------------------------------------------------------------------------------------

def test_the_api_and_the_cli_agree_field_for_field(tmp_path, spectrum, zones_file, capsys):
    """Both build their output with src.predict.prediction_payload, and this proves they still match."""
    model_path = save_bundle(make_bundle(), tmp_path / "models" / "m.joblib")
    app = create_app(copy.deepcopy(load_config()), model_path=model_path, zones_path=zones_file)
    with TestClient(app) as client:
        from_api = post_spectrum(client, spectrum).json()

    assert predict_spectrum.main([str(spectrum), "--model", str(model_path),
                                  "--uncertainty", str(zones_file)]) == 0
    from_cli = json.loads(capsys.readouterr().out)

    timing = {"preprocessing_ms", "inference_ms", "total_ms"}
    assert set(from_api) == set(from_cli)
    for key in set(from_api) - timing:
        assert from_api[key] == from_cli[key], f"{key} differs between the API and the CLI"


def test_the_feature_fingerprint_is_unchanged():
    """The Version 0.9 input limits are keyword parameters, so they must not touch the feature definition.

    A change here would mean every saved bundle stops loading and every cached fit is invalidated.
    """
    assert PreprocessingConfig().fingerprint() == "347cbd6d5d956ff9"


def test_the_configured_limits_are_the_pre_registered_ones():
    """docs/v0.9_api_plan.md fixes these numbers; this test is what stops them drifting quietly."""
    limits = Limits.from_config(load_config())
    assert limits.max_upload_bytes == 4_000_000
    assert limits.max_raw_points == 200_000
    assert limits.max_batch_files == 20
    assert limits.max_batch_bytes == 32_000_000
    assert limits.allowed_suffixes == (".txt",)


def test_the_append_only_log_is_not_touched_by_serving(tmp_path, spectrum, zones_file):
    """The API is a read-only consumer: protocol amendment 5, point 1."""
    log_path = project_path("results/experiments/test_evaluations.csv")
    before = log_path.read_bytes() if log_path.is_file() else None
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        post_spectrum(client, spectrum)
        client.post("/batch-predict", files=[("files", ("s.txt", spectrum.read_bytes(), "text/plain"))])
        client.get("/model-info")
    after = log_path.read_bytes() if log_path.is_file() else None
    assert after == before


def test_the_default_paths_come_from_the_configuration(tmp_path):
    config = load_config()
    assert default_model_path(config).name == "best_random.joblib"
    assert "v0.4" in str(default_model_path(config))
    zones = default_zones_path(config)
    assert zones is not None and zones.name == "uncertainty.json" and "v0.6" in str(zones)
