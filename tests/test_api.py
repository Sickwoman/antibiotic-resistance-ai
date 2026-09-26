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
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import (
    API_VERSION,
    DETAIL_MODEL_UNAVAILABLE,
    DETAIL_ZONES_UNUSABLE,
    PROJECT_MODEL_VERSION,
    Limits,
    check_zone_bounds,
    create_app,
    default_model_path,
    default_zones_path,
)
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

# A Windows drive letter, a UNC prefix, or a POSIX absolute path inside a JSON string. Used to assert that
# no response ever carries a filesystem path, which the audit found a degraded /health was doing.
PATH_LIKE = re.compile(r"[A-Za-z]:[\\/]|\\\\\\\\|(?<![\w.])/(?:home|etc|usr|var|tmp|Users)/")


def spectrum_text(n: int = 200, start: float = 2000.0, step: float = 1.0) -> str:
    """A minimal valid two-column spectrum: strictly increasing positive m/z, non-negative intensities."""
    return "".join(f"{start + i * step:.4f} {100 + i}\n" for i in range(n))


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


def test_the_streaming_counter_refuses_an_upload_with_no_content_length(tmp_path):
    """The second layer of the size guard, exercised on its own.

    The middleware refuses on a declared Content-Length, but a chunked request declares none, so the
    per-chunk counter inside stream_to_file is the only thing between the service and an unbounded body.
    A mutation audit showed that deleting that counter left the entire suite green, because every other
    oversize test is satisfied by the middleware alone. This test fails if the counter is removed.
    """
    crlf = chr(13) + chr(10)
    boundary = "----teststreamboundary"
    head = (f"--{boundary}{crlf}"
            f'Content-Disposition: form-data; name="file"; filename="big.txt"{crlf}'
            f"Content-Type: text/plain{crlf}{crlf}").encode()
    tail = f"{crlf}--{boundary}--{crlf}".encode()
    line, n_lines = b"2000.0 1" + chr(10).encode(), 30_000      # ~270 KB against a 50 KB limit

    def chunked():
        yield head
        for _ in range(n_lines):
            yield line
        yield tail

    app = build_app(tmp_path, limits={"max_upload_bytes": 50_000}, zones=tmp_path / "absent.json")
    with TestClient(app) as client:
        r = client.post("/predict", content=chunked(),
                        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                                 "Transfer-Encoding": "chunked"})
    assert "content-length" not in {k.lower() for k in r.request.headers}, (
        "the request declared a length, so this exercised the middleware rather than the counter")
    assert r.status_code == 413
    assert error_of(r)[0] == "UploadTooLarge"


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
    # The load error names the file, i.e. an absolute path. The public detail must be the fixed string.
    assert health.json()["detail"] == DETAIL_MODEL_UNAVAILABLE
    assert not PATH_LIKE.search(health.text), "a filesystem path reached /health"
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


# ------------------------------------------------------------------------------------------------
# Spectrum content validation, through the API (the rules live in the existing reader)
# ------------------------------------------------------------------------------------------------

def _mutate(lines: list[str], index: int, text: str) -> bytes:
    lines = list(lines)
    lines[index] = text
    return "".join(lines).encode()


@pytest.mark.parametrize(("case", "expected"), [
    ("duplicate_mz", "not strictly increasing"),
    ("unsorted_mz", "not strictly increasing"),
    ("non_positive_mz", "must be positive"),
    ("negative_intensity", "must not be negative"),
    ("not_a_number", "non-numeric"),
    ("infinite", "infinite"),
    ("one_column", "expected 2 columns"),
    ("three_columns", "expected 2 columns"),
])
def test_every_spectrum_rule_is_enforced_through_the_api(tmp_path, case, expected):
    lines = spectrum_text().splitlines(keepends=True)
    if case == "duplicate_mz":
        payload = _mutate(lines, 50, lines[49])
    elif case == "unsorted_mz":
        swapped = list(lines)
        swapped[50], swapped[51] = swapped[51], swapped[50]
        payload = "".join(swapped).encode()
    elif case == "non_positive_mz":
        payload = spectrum_text(start=-50.0).encode()
    elif case == "negative_intensity":
        payload = _mutate(lines, 50, "2050.0000 -7\n")
    elif case == "not_a_number":
        payload = _mutate(lines, 50, "2050.0000 nan\n")
    elif case == "infinite":
        payload = _mutate(lines, 50, "2050.0000 inf\n")
    elif case == "one_column":
        payload = "".join(f"{2000.0 + i:.4f}\n" for i in range(200)).encode()
    else:
        payload = "".join(f"{2000.0 + i:.4f} 1 2\n" for i in range(200)).encode()

    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        r = client.post("/predict", files={"file": ("s.txt", payload, "text/plain")})
    assert r.status_code == 422, f"{case} was not refused: {r.status_code} {r.text[:120]}"
    message = error_of(r)[1]
    assert expected in message, f"{case}: unexpected message {message!r}"
    assert "Traceback" not in message


def test_a_spectrum_with_too_many_points_is_refused(tmp_path):
    """The maximum point count is enforced, not merely configured."""
    payload = spectrum_text(n=400).encode()
    app = build_app(tmp_path, limits={"max_raw_points": 150}, zones=tmp_path / "absent.json")
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("s.txt", payload, "text/plain")})
    assert r.status_code == 422
    message = error_of(r)[1]
    assert "400 points" in message and "limit is 150" in message


def test_predict_requires_exactly_one_file(tmp_path):
    """Two files under one field previously returned 200 having silently used only one of them."""
    payload = spectrum_text().encode()
    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        r = client.post("/predict", files=[("file", ("a.txt", payload)), ("file", ("b.txt", payload))])
        assert r.status_code == 400
        kind, message = error_of(r)
        assert kind == "BadRequest" and "exactly one" in message
        assert client.post("/predict", files={"file": ("a.txt", payload)}).status_code == 200


# ------------------------------------------------------------------------------------------------
# Filenames are never used as paths
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "../../../etc/passwd.txt",
    "..\\..\\..\\Windows\\System32\\config.txt",
    "C:\\Windows\\System32\\drivers\\etc\\hosts.txt",
    "/etc/shadow.txt",
    "....//....//secret.txt",
])
def test_a_hostile_filename_is_ignored_not_resolved(tmp_path, filename):
    """The client's filename supplies a suffix and nothing else, so traversal has nothing to traverse."""
    payload = spectrum_text().encode()
    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        r = client.post("/predict", files={"file": (filename, payload, "text/plain")})
    assert r.status_code == 200, r.text[:200]
    assert "passwd" not in r.text and "System32" not in r.text and "shadow" not in r.text
    assert not PATH_LIKE.search(r.text)


def test_no_response_ever_carries_a_filesystem_path(tmp_path, spectrum, zones_file):
    """The general form of the leak the audit found: no endpoint, success or failure, may show a path."""
    responses = []
    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        responses += [client.get("/health"), client.get("/model-info"), post_spectrum(client, spectrum),
                      client.post("/predict", files={"file": ("x.txt", b"", "text/plain")}),
                      client.post("/predict", files={"file": ("x.zip", b"PK", "application/zip")})]
    broken = build_app(tmp_path, model_path=tmp_path / "gone.joblib")
    with TestClient(broken) as client:
        responses += [client.get("/health"), client.get("/model-info"),
                      post_spectrum(client, spectrum)]
    for r in responses:
        found = PATH_LIKE.search(r.text)
        assert not found, f"{r.request.url.path} leaked {found.group(0)!r}: {r.text[:160]}"


# ------------------------------------------------------------------------------------------------
# Temporary-file lifecycle
# ------------------------------------------------------------------------------------------------

def _recording_tempdir(monkeypatch) -> list[Path]:
    """Patch the module's TemporaryDirectory so a test can see where an upload was written."""
    import src.api as api_module

    created: list[Path] = []
    real = api_module.TemporaryDirectory

    class Recording(real):                                    # type: ignore[misc, valid-type]
        def __enter__(self) -> str:
            name = super().__enter__()
            created.append(Path(name))
            return name

    monkeypatch.setattr(api_module, "TemporaryDirectory", Recording)
    return created


def test_the_temporary_upload_is_removed_after_success(tmp_path, spectrum, monkeypatch):
    created = _recording_tempdir(monkeypatch)
    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        assert post_spectrum(client, spectrum).status_code == 200
    assert created, "the upload was not written through the recorded temporary directory"
    for directory in created:
        assert not directory.exists(), f"{directory} survived the request"


def test_the_temporary_upload_is_removed_after_failure(tmp_path, monkeypatch):
    created = _recording_tempdir(monkeypatch)
    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        r = client.post("/predict", files={"file": ("s.txt", b"nonsense\n", "text/plain")})
    assert r.status_code == 422
    assert created, "the upload was not written through the recorded temporary directory"
    for directory in created:
        assert not directory.exists(), f"{directory} survived a failed request"


def test_uploads_are_never_written_inside_the_repository(tmp_path, spectrum, monkeypatch):
    created = _recording_tempdir(monkeypatch)
    root = project_path(".").resolve()
    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        post_spectrum(client, spectrum)
    for directory in created:
        assert root not in directory.resolve().parents and directory.resolve() != root


# ------------------------------------------------------------------------------------------------
# Start-up failure modes
# ------------------------------------------------------------------------------------------------

def test_a_corrupt_bundle_degrades_rather_than_crashes(tmp_path, spectrum):
    path = save_bundle(make_bundle(), tmp_path / "models" / "m.joblib")
    path.write_bytes(b"this is not a joblib file at all")
    app = create_app(copy.deepcopy(load_config()), model_path=path, zones_path=tmp_path / "absent.json")
    with TestClient(app) as client:
        health = client.get("/health")
        predicted = post_spectrum(client, spectrum)
    assert health.status_code == 503 and health.json()["status"] == "degraded"
    assert health.json()["detail"] == DETAIL_MODEL_UNAVAILABLE
    assert predicted.status_code == 503
    assert not PATH_LIKE.search(health.text) and not PATH_LIKE.search(predicted.text)


def test_a_file_that_is_not_a_bundle_degrades(tmp_path, spectrum):
    import joblib

    path = tmp_path / "notabundle.joblib"
    joblib.dump({"hello": "world"}, path)
    app = create_app(copy.deepcopy(load_config()), model_path=path, zones_path=tmp_path / "absent.json")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/health").json()["detail"] == DETAIL_MODEL_UNAVAILABLE


def test_a_malformed_zone_file_degrades_instead_of_being_ignored(tmp_path, spectrum):
    """load_zones distinguishes absent from unparseable precisely so this cannot pass silently.

    Serving on would report confidence "not available" for a model that does have zones.
    """
    bad = tmp_path / "bad_zones.json"
    bad.write_text(json.dumps({"nonsense": 1}), encoding="utf-8")
    with TestClient(build_app(tmp_path, zones=bad)) as client:
        health = client.get("/health")
        predicted = post_spectrum(client, spectrum)
    assert health.status_code == 503
    assert health.json()["status"] == "degraded" and health.json()["zones_loaded"] is False
    assert health.json()["detail"] == DETAIL_ZONES_UNUSABLE
    assert predicted.status_code == 503
    assert not PATH_LIKE.search(health.text)


def test_an_absent_zone_file_is_not_a_failure(tmp_path, spectrum):
    """Absent is fine and must stay fine: a prediction without a label is still a prediction."""
    with TestClient(build_app(tmp_path, zones=tmp_path / "nothing_here.json")) as client:
        health = client.get("/health")
        predicted = post_spectrum(client, spectrum)
    assert health.status_code == 200 and health.json()["status"] == "ok"
    assert health.json()["zones_loaded"] is False
    assert predicted.status_code == 200 and predicted.json()["confidence"] == "not available"


# ------------------------------------------------------------------------------------------------
# Versioning and historical immutability
# ------------------------------------------------------------------------------------------------

def test_the_api_version_is_its_own_constant(tmp_path):
    """It must not be read from config.yaml -> project.version, which tracks the experiment state.

    That field was stale at "0.7.0" for the whole of Version 0.8, so sourcing a public contract version
    from it would let an unrelated edit change the API's identity.
    """
    assert API_VERSION == "0.9.0"
    config = copy.deepcopy(load_config())
    config["project"]["version"] = "99.99.99-nonsense"
    model_path = save_bundle(make_bundle(), tmp_path / "models" / "m.joblib")
    app = create_app(config, model_path=model_path, zones_path=tmp_path / "absent.json")
    with TestClient(app) as client:
        assert client.get("/health").json()["api_version"] == API_VERSION
        assert client.get("/model-info").json()["api_version"] == API_VERSION


def test_serving_leaves_every_historical_artifact_untouched(tmp_path, spectrum, zones_file):
    """Protocol amendment 5, point 1: the API is a read-only consumer."""
    watched = [project_path("results/experiments/test_evaluations.csv"),
               project_path("results/metrics/v0.6") / "ecoli_ciprofloxacin" / "uncertainty.json",
               project_path("results/metrics/v0.4") / "ecoli_ciprofloxacin" / "inference_timing.json"]
    before = {path: path.read_bytes() for path in watched if path.is_file()}
    assert before, "no historical artifact was found to watch"

    models_dir = project_path("models")
    models_before = sorted((f, f.stat().st_mtime_ns, f.stat().st_size)
                           for f in models_dir.rglob("*") if f.is_file()) if models_dir.is_dir() else []

    with TestClient(build_app(tmp_path, zones=zones_file)) as client:
        post_spectrum(client, spectrum)
        client.post("/batch-predict", files=[("files", ("s.txt", spectrum.read_bytes(), "text/plain"))])
        client.get("/model-info")
        client.get("/health")

    for path, content in before.items():
        assert path.read_bytes() == content, f"{path.name} changed while serving"
    models_after = sorted((f, f.stat().st_mtime_ns, f.stat().st_size)
                          for f in models_dir.rglob("*") if f.is_file()) if models_dir.is_dir() else []
    assert models_after == models_before, "a file under models/ changed while serving"


# ------------------------------------------------------------------------------------------------
# Post-merge audit: zone edges that are syntactically fine but not usable probabilities
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("case", "content"), [
    ("lower_nan", '{"threshold": 0.5, "lower": NaN, "upper": null, "rule": {}}'),
    ("lower_inf", '{"threshold": 0.5, "lower": Infinity, "upper": null, "rule": {}}'),
    ("upper_neg_inf", '{"threshold": 0.5, "lower": null, "upper": -Infinity, "rule": {}}'),
    ("lower_below_zero", '{"threshold": 0.5, "lower": -3.0, "upper": 5.0, "rule": {}}'),
    ("upper_above_one", '{"threshold": 0.5, "lower": 0.2, "upper": 1.5, "rule": {}}'),
    ("not_json", "{{{ this is not json"),
    ("wrong_keys", '{"nonsense": 1}'),
    ("edges_out_of_order", '{"threshold": 0.5, "lower": 0.9, "upper": 0.1, "rule": {}}'),
])
def test_an_unusable_zone_file_makes_the_service_not_ready(tmp_path, spectrum, case, content):
    """A zone file that parses but cannot be compared against must not be served.

    Zones.from_dict checks that two edges are ordered; nothing checked that an edge was finite or inside
    [0, 1]. A NaN edge makes every comparison false, so the zone silently disappears and every spectrum
    returns "Uncertain" from a model that has a zone. An infinite edge makes every comparison true, so
    every spectrum is labelled high-confidence. Both are silent, which is exactly the class of failure the
    malformed-zones fix exists to stop.
    """
    zones = tmp_path / f"zones_{case}.json"
    zones.write_text(content, encoding="utf-8")
    with TestClient(build_app(tmp_path, zones=zones)) as client:
        health = client.get("/health")
        predicted = post_spectrum(client, spectrum)
    assert health.status_code == 503, f"{case} was served as healthy"
    assert health.json()["status"] == "degraded" and health.json()["zones_loaded"] is False
    assert health.json()["detail"] == DETAIL_ZONES_UNUSABLE
    assert predicted.status_code == 503, f"{case} still produced a prediction"
    assert not PATH_LIKE.search(health.text) and not PATH_LIKE.search(predicted.text)


@pytest.mark.parametrize(("lower", "upper"), [(0.2, None), (None, 0.8), (0.2, 0.8), (None, None), (0.0, 1.0)])
def test_usable_zone_bounds_are_accepted(lower, upper):
    """The guard must not reject a fitted outcome: a side that does not exist is None, which is fine."""
    check_zone_bounds(Zones(0.5, lower, upper, ZoneRule(), "validation", 10))


def test_the_committed_zone_file_passes_the_bound_guard():
    """The real Version 0.6 artifact must remain servable; the guard may only reject genuine nonsense."""
    path = project_path("results/metrics/v0.6") / "ecoli_ciprofloxacin" / "uncertainty.json"
    if not path.is_file():
        pytest.skip("the Version 0.6 zone file is not in this checkout")
    from src.predict import load_zones as _load_zones

    zones = _load_zones(path)
    assert zones is not None
    check_zone_bounds(zones)


# ------------------------------------------------------------------------------------------------
# Post-merge audit: an unexpected exception must not leak whatever it happens to carry
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("secret_path", [
    r"C:\Projects\antibiotic-resistance-ai\models\secret.joblib",
    "/home/someone/driams/DRIAMS-A/raw/2018/secret.txt",
    "/Users/someone/models/private.joblib",
    r"\\server\share\models\secret.joblib",
])
def test_an_unexpected_exception_never_leaks_its_message(tmp_path, spectrum, monkeypatch, secret_path):
    """The 500 path is the one place an arbitrary internal message could reach a client.

    Tested by making the inference call raise something carrying a filesystem path, rather than by
    trusting that no such exception exists.
    """
    import src.api as api_module

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"internal failure while reading {secret_path}")

    monkeypatch.setattr(api_module, "predict_spectrum_file", explode)
    app = build_app(tmp_path, zones=tmp_path / "absent.json")
    with TestClient(app, raise_server_exceptions=False) as client:
        r = post_spectrum(client, spectrum)
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["type"] == "InternalError"
    assert secret_path not in r.text
    assert "secret" not in r.text and "private" not in r.text
    assert not PATH_LIKE.search(r.text)
    assert "Traceback" not in r.text and "RuntimeError" not in r.text


# ------------------------------------------------------------------------------------------------
# Post-merge audit: degradation is an invariant over every route, not a per-endpoint accident
# ------------------------------------------------------------------------------------------------

def test_no_route_serves_a_success_while_the_service_is_degraded(tmp_path, spectrum):
    """Checked by enumerating the application's own routes, so a new endpoint cannot quietly opt out."""
    payload = spectrum.read_bytes()
    app = build_app(tmp_path, model_path=tmp_path / "absent.joblib")
    checked = 0
    with TestClient(app) as client:
        for route in app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", set()) or set()
            if not path or path.startswith("/openapi") or path in ("/docs", "/redoc", "/docs/oauth2-redirect"):
                continue
            for method in sorted(methods & {"GET", "POST"}):
                if method == "GET":
                    response = client.get(path)
                elif path == "/batch-predict":
                    response = client.post(path, files=[("files", ("s.txt", payload, "text/plain"))])
                else:
                    response = client.post(path, files={"file": ("s.txt", payload, "text/plain")})
                checked += 1
                assert not 200 <= response.status_code < 300, (
                    f"{method} {path} returned {response.status_code} while the service was degraded")
                assert not PATH_LIKE.search(response.text), f"{method} {path} leaked a path when degraded"
    assert checked >= 4, f"only {checked} routes were exercised; the enumeration is not covering the API"


def test_predict_rejects_a_request_with_no_file(tmp_path):
    """Zero files must be a deterministic client error, not a guess."""
    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        assert client.post("/predict").status_code == 422
        assert client.post("/predict", data={"unrelated": "1"}).status_code == 422

