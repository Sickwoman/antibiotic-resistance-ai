"""The research demo's server, page and runner (demo/): what it shows, what it refuses, and what it never does.

Most checks use a fake runner, so they need no model. Two use the real, unchanged CLI where it fails before any model
is needed (a missing bundle, a timeout). The end-to-end checks with the frozen model are skipped when `models/` is
absent (CI). The production and development logs are guarded for the whole session by tests/conftest.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo import synthetic
from demo.__main__ import main as launch
from demo.app import SECURITY_HEADERS, create_app
from demo.evidence import load_evidence
from demo.inference import DemoInferenceError, _classify, default_model_path, predict
from src.predict import DISCLAIMER

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "demo" / "static"
BUNDLE = ROOT / "models" / "v0.4" / "ecoli_ciprofloxacin" / "best_random.joblib"
needs_model = pytest.mark.skipif(not BUNDLE.is_file(), reason="the frozen model bundle is not in this checkout")
FORBIDDEN = [r"\bresistant\b", r"\bsusceptible\b", r"\bsafe\b", r"\bprescri", r"\brule[ -]?out\b", r"\bconfirmed\b",
             r"\btreat(?!ment recommendation)", r"high-confidence"]


def fake_runner(calls: list | None = None, result: dict | None = None, error: DemoInferenceError | None = None):
    def run(text, *, model=None, timeout=None):
        if calls is not None:
            calls.append({"text": text, "model": model, "timeout": timeout})
        if error is not None:
            raise error
        return result or {"score": 0.3, "threshold": 0.1426, "above_threshold": True, "model_version": "v-test",
                          "preprocessing_ms": 1.0, "inference_ms": 1.0, "total_ms": 2.0, "disclaimer": DISCLAIMER,
                          "cli_wall_s": 2.5, "runner": "scripts/predict_spectrum.py (unchanged)"}
    return run


def client(**kwargs) -> TestClient:
    return TestClient(create_app(**kwargs), base_url="http://127.0.0.1")


# --- what the page may and may not say ---------------------------------------------------------------------------

def visible_text() -> str:
    return " ".join((STATIC / name).read_text(encoding="utf-8") for name in ("index.html", "app.js"))


def test_the_page_loads_nothing_from_another_origin_and_stores_nothing():
    for name in ("index.html", "app.js", "style.css"):
        text = (STATIC / name).read_text(encoding="utf-8")
        external = [u for u in re.findall(r"https?://[^\s\"'<>)]+", text) if u != "http://www.w3.org/2000/svg"]
        assert not external, (name, external)
        for api in ("localStorage", "sessionStorage", "indexedDB", "document.cookie", "innerHTML", "eval("):
            assert api not in text, (name, api)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html and " style=" not in html          # nothing inline: the CSP allows 'self' only


def test_the_page_names_no_label_treatment_or_reassurance():
    text = visible_text().lower()
    for pattern in FORBIDDEN:
        assert not re.search(pattern, text), pattern
    assert "never a treatment recommendation" in text                # the only "treatment", and it is a negation


def test_no_colour_on_the_page_is_green():
    css = re.sub(r"/\*.*?\*/", "", (STATIC / "style.css").read_text(encoding="utf-8"), flags=re.S).lower()
    assert "green" not in css and "lime" not in css
    for hexcode in re.findall(r"#([0-9a-f]{6})\b", css + visible_text().lower()):
        r, g, b = (int(hexcode[i:i + 2], 16) for i in (0, 2, 4))
        assert g - max(r, b) < 25, f"#{hexcode} is greenish"


# --- the server -----------------------------------------------------------------------------------------------------

def test_every_response_carries_the_security_headers_and_no_caching():
    with client(runner=fake_runner()) as c:
        for path in ("/", "/static/app.js", "/static/style.css", "/api/status", "/api/examples", "/api/evidence"):
            response = c.get(path)
            assert response.status_code == 200, path
            for key, value in SECURITY_HEADERS.items():
                assert response.headers[key] == value, (path, key)


def test_only_loopback_host_names_are_served():
    with client(runner=fake_runner()) as c:
        assert c.get("/api/status", headers={"host": "attacker.example"}).status_code == 400
        assert c.get("/api/status", headers={"host": "localhost:8050"}).status_code == 200


def test_the_examples_are_the_three_synthetic_spectra():
    with client(runner=fake_runner()) as c:
        examples = c.get("/api/examples").json()
    assert [e["id"] for e in examples] == ["synthetic-1", "synthetic-2", "synthetic-3"]
    assert all(e["synthetic"] and e["seed"] == 42 and e["n_points"] == synthetic.N_POINTS for e in examples)
    assert not any(k in e for e in examples for k in ("label", "truth", "resistant", "susceptible"))


def test_a_prediction_sends_the_example_to_the_runner_and_forwards_the_allow_list_only():
    calls: list = []
    with client(runner=fake_runner(calls), timeout=12.0) as c:
        response = c.post("/api/predict", json={"example": "synthetic-2"})
    assert response.status_code == 200
    body = response.json()
    assert body["example"] == "synthetic-2" and body["synthetic"] is True and body["above_threshold"] is True
    assert calls == [{"text": synthetic.generate()[1].to_text(), "model": None, "timeout": 12.0}]
    raw = response.text.lower()
    for word in ("\"prediction\"", "confidence", "advice", "resistant", "susceptible\"", "high-confidence"):
        assert word not in raw, word


def test_nothing_but_the_three_examples_can_reach_the_model():
    calls: list = []
    with client(runner=fake_runner(calls)) as c:
        assert c.post("/api/predict", json={"example": "../x"}).json()["error"]["code"] == "unknown_example"
        assert c.post("/api/predict", json={"example": "synthetic-1", "text": "1 2"}).status_code == 422
        upload = c.post("/api/predict", files={"file": ("x.txt", b"2000 1\n2001 2\n", "text/plain")})
        assert upload.status_code == 422
        assert c.get("/static/index.html").status_code == 404 and c.get("/static/../app.py").status_code == 404
    assert calls == []


@pytest.mark.parametrize(("code", "status"), [("model_unavailable", 503), ("model_invalid", 503),
                                              ("inference_failed", 500), ("timeout", 504)])
def test_runner_failures_reach_the_page_as_coded_errors(code, status):
    with client(runner=fake_runner(error=DemoInferenceError(code, "what happened", status))) as c:
        response = c.post("/api/predict", json={"example": "synthetic-1"})
    assert response.status_code == status
    assert response.json() == {"error": {"code": code, "message": "what happened"}}


def test_missing_evaluation_results_are_reported_not_invented(monkeypatch):
    def broken():
        raise FileNotFoundError("results/metrics/v2.0/marisma_evaluation.json")
    monkeypatch.setattr("demo.app.load_evidence", broken)
    with client(runner=fake_runner()) as c:
        response = c.get("/api/evidence")
        assert c.get("/api/status").json()["evidence_available"] is False
    assert response.status_code == 503 and response.json()["error"]["code"] == "evidence_unavailable"


def test_the_launcher_refuses_any_host_but_loopback(capsys):
    assert launch(["--host", "0.0.0.0"]) == 2
    assert "Refusing" in capsys.readouterr().err


# --- the evidence panel ---------------------------------------------------------------------------------------------

def test_the_evidence_equals_the_committed_aggregates():
    report = json.loads((ROOT / "results/metrics/v2.0/marisma_evaluation.json").read_bytes())
    internal = json.loads((ROOT / "results/metrics/v0.4/ecoli_ciprofloxacin/test_intervals.json").read_bytes())
    e = load_evidence()
    m = report["primary"]["metrics"]
    for key in ("roc_auc", "sensitivity", "specificity", "zone_npv", "calibration_intercept"):
        shown = {"roc_auc": e["auroc"], "zone_npv": e["zone"]["npv"]}.get(key, e.get(key))
        assert shown == {k: m[key][k] for k in ("estimate", "low", "high")}, key
    assert (e["population"]["n"], e["population"]["n_ri"], e["population"]["n_s"]) == (1145, 455, 690)
    assert (e["zone"]["n"], e["zone"]["ri"]) == (193, 18)
    assert e["ceftriaxone"] == "unavailable" == report["holm_family"]["conclusions"]["ceftriaxone"]
    assert e["error_control"] == report["holm_family"]["error_control"]
    assert e["threshold"] == 0.14261540693905073
    lightgbm = internal["random"]["intervals"]["tuned_lightgbm"]
    assert e["internal"]["auroc"]["estimate"] == lightgbm["roc_auc"]["estimate"]
    assert round(e["auroc"]["estimate"], 3) == 0.772 and round(e["auroc"]["low"], 3) == 0.744
    assert round(e["auroc"]["high"], 3) == 0.798 and round(e["sensitivity"]["estimate"], 3) == 0.897
    assert round(e["specificity"]["estimate"], 3) == 0.393 and round(e["zone"]["npv"]["estimate"], 3) == 0.907


# --- the runner, with the real CLI ----------------------------------------------------------------------------------

def test_cli_messages_are_classified_and_local_paths_are_not_shown():
    root = str(ROOT)
    missing = _classify(f"Error: Model file not found: {root}\\models\\x.joblib. Train and save a model first.", None)
    assert missing.code == "model_unavailable" and missing.status == 503 and root not in missing.message
    assert "train_baselines" not in missing.message and "Model file not found: <repository>/models/x.joblib." in \
        missing.message
    assert _classify("Error: x does not match its checksum in x.sha256", None).code == "model_invalid"
    assert _classify("Error: m/z values are not strictly increasing.", None).code == "inference_failed"


def test_a_missing_bundle_is_reported_by_the_real_cli(tmp_path):
    with pytest.raises(DemoInferenceError) as info:
        predict(synthetic.generate()[0].to_text(), model=tmp_path / "absent.joblib", timeout=120)
    assert info.value.code == "model_unavailable" and info.value.status == 503


def test_a_run_that_takes_too_long_is_stopped():
    with pytest.raises(DemoInferenceError) as info:
        predict(synthetic.generate()[0].to_text(), timeout=0.05)
    assert info.value.code == "timeout" and info.value.status == 504


@needs_model
def test_a_spectrum_the_reader_refuses_is_an_inference_error():
    with pytest.raises(DemoInferenceError) as info:
        predict("mass intensity\n2000 1\n1999 2\n" * 60, timeout=120)
    assert info.value.code == "inference_failed" and info.value.status == 500


@needs_model
def test_demo_scores_are_the_frozen_models_own(tmp_path):
    from src.predict import load_bundle, predict_spectrum_file
    assert default_model_path() == BUNDLE
    bundle = load_bundle(BUNDLE)
    with client() as c:
        for example in synthetic.generate():
            shown = c.post("/api/predict", json={"example": example.id}).json()
            path = tmp_path / f"{example.id}.txt"
            path.write_text(example.to_text(), encoding="utf-8")
            direct = predict_spectrum_file(bundle, path)
            assert shown["score"] == direct.resistance_probability and shown["threshold"] == direct.threshold
            assert shown["above_threshold"] == (direct.prediction == "Resistant")
            assert shown["model_version"] == direct.model_version


def test_the_start_up_sweep_removes_only_old_demo_folders(tmp_path, monkeypatch):
    import os
    import time

    from demo import inference
    monkeypatch.setattr(inference.tempfile, "gettempdir", lambda: str(tmp_path))
    old, fresh, other = tmp_path / "amr-demo-old", tmp_path / "amr-demo-new", tmp_path / "someone-else"
    for folder in (old, fresh, other):
        folder.mkdir()
        (folder / "synthetic_spectrum.txt").write_text("x", encoding="utf-8")
    past = time.time() - inference.STALE_AFTER_S - 60
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    assert inference.sweep_stale() == 1
    assert not old.exists() and fresh.exists() and other.exists()
