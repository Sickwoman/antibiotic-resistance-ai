"""The complete Version 1.2 report: every pre-specified endpoint, every difference with its operands named, and
nothing refitted to make it (scripts/v12_report.py).

It runs on the committed outputs of the Version 1.2 run, which are in the repository, so this works in CI.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from src.utils import ConfigError, load_config, project_path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
REPORT = project_path("results/metrics/v1.2/ecoli_ceftriaxone")


@pytest.fixture(scope="module")
def module():
    return importlib.import_module("v12_report")


@pytest.fixture(scope="module")
def text(module):
    vc = load_config()["v12_development"]
    saved = module.load(REPORT, project_path(vc["development_log"]))
    return module.build(saved, vc["dataset"], int(vc["primary_partition_seed"]))


def test_every_pre_specified_endpoint_is_tabulated(text):
    for heading in ("## Primary endpoint", "## 1. Brier skill", "5. delivered sensitivity", "## 2. Discrimination",
                    "## 3. Calibration and log loss", "## 4. Operating-point stability", "## 6. Forward in time",
                    "## 7. The diagnostics", "## 8. Partitions 43 and 44", "## How every difference is signed"):
        assert heading in text, heading
    for arm in ("B unchanged Version 1.1 procedure", "C candidate", "D1 diagnostic"):
        assert text.count(arm) >= 3, arm
    assert "carry **no meaning**" in text                       # R0's pooled slope and AUROC are flagged


def test_each_difference_is_the_arithmetic_of_its_named_operands(text):
    pooled = json.loads((REPORT / "pooled_results.json").read_text(encoding="utf-8"))["42"]
    iv = pooled["scores"]["intervals"]
    b_minus_c = iv["B"]["brier"]["estimate"] - iv["C"]["brier"]["estimate"]
    b_minus_d1 = iv["B"]["brier"]["estimate"] - iv["D1"]["brier"]["estimate"]
    c_minus_b_auc = iv["C"]["roc_auc"]["estimate"] - iv["B"]["roc_auc"]["estimate"]
    assert f"Brier(B) minus Brier(C): **{b_minus_c:+.4f}" in text
    assert f"**Brier(B) minus Brier(D1)** = {b_minus_d1:+.4f}" in text and b_minus_d1 > 0     # D1 lower Brier
    assert f"| D1 minus B, Brier | {-b_minus_d1:+.4f}" in text
    assert f"C minus B: AUROC **{c_minus_b_auc:+.4f}" in text
    assert "not of the" in text and "cross-fitted cut-off" in text                             # the attribution


def test_flip_and_reading_follow_the_sign_rules(module):
    d = {"estimate": 0.002, "low": 0.0002, "high": 0.0036}
    assert module.flip(d) == {"estimate": -0.002, "low": -0.0036, "high": -0.0002}
    assert module.reading(d, "up", "down").startswith("up")
    assert module.reading(module.flip(d), "up", "down").startswith("down")
    assert module.reading({"estimate": 0.0, "low": -1.0, "high": 1.0}, "up", "down").startswith("not demonstrated")


def test_the_forward_note_does_not_claim_a_cause(text):
    assert "Neither establishes why sensitivity fell" in text
    assert "would leave sensitivity at a fixed cut-off unchanged" in text
    assert not re.search(r"\bso fewer resistant isolates\b|found why", text)


def test_the_report_fits_nothing_and_loads_no_data(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("load_dataset", "fit_calibrated", "predict_proba", "load_bundle", "np.load("):
        assert forbidden not in source, forbidden


def test_missing_saved_outputs_are_refused(module, tmp_path):
    with pytest.raises(ConfigError, match="missing"):
        module.load(tmp_path, tmp_path / "development_runs.csv")
