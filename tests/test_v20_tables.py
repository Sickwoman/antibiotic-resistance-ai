"""Version 2.0's step 6 tables (scripts/v20_tables.py): generated from committed aggregates, never typed by hand.

The committed `results/metrics/v2.0/tables.md` must be exactly what the script generates from the committed records.
The script must read nothing but those records (no work folder, no sealed file, no prediction, no label), and it
must stop when the records disagree, instead of printing a table that does not reconcile.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v20_tables.py"
ALLOWED_IMPORTS = {"__future__", "argparse", "hashlib", "json", "math", "sys", "pathlib", "typing", "pandas",
                   "src.tables", "src.utils"}


@pytest.fixture(scope="module")
def tables():
    spec = importlib.util.spec_from_file_location("v20_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inputs(tables):
    return tables.load_inputs()


def test_the_committed_tables_are_what_the_committed_records_generate(tables, inputs):
    committed = (ROOT / tables.OUT).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert tables.render(inputs) == committed


def test_the_tables_do_not_depend_on_line_endings(tables, tmp_path):
    # A Windows working copy may hold a record with CRLF (the run wrote the report there), while the repository and
    # every CI checkout hold LF: the tables, including the report's digest, must come out the same either way.
    for rel in tables.INPUTS:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        lf = (ROOT / rel).read_bytes().replace(b"\r\n", b"\n")
        dest.write_bytes(lf.replace(b"\n", b"\r\n") if rel == tables.REPORT else lf)
    committed = (ROOT / tables.OUT).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert tables.render(tables.load_inputs(tmp_path)) == committed


def test_the_script_reads_only_committed_aggregates(tables):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= ALLOWED_IMPORTS, f"unexpected imports: {sorted(imported - ALLOWED_IMPORTS)}"
    assert all(path.startswith("results/") for path in tables.INPUTS)
    text = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("MARISMa_v2.0.0_work", "_sealed", "predictions.npz", "interpretations.csv", "joblib"):
        assert forbidden not in text


def test_the_principal_numbers_and_wording_are_in_the_tables(tables, inputs):
    text = tables.render(inputs)
    for expected in ("| AUROC | 0.772 | [0.744, 0.798] |", "1,145", "455 resistant, 690 susceptible",
                     "| 193 | 175 | 18 | 0.907 | [0.865, 0.944] |", "408 of 455 resistant isolates flagged",
                     "419 of 690 susceptible isolates flagged resistant", "4.4 × 10⁻⁷¹",
                     "Evidence of above-chance ranking on MARISMa under the isolate-independence assumption.",
                     "Missing patient linkage leaves actual error control uncertain.", "| Ceftriaxone | unavailable",
                     "13 of 1,607 isolates (0.81 %)", "7 of the 1,172 matched", "1,031 of 16,975"):
        assert expected in text, expected


def test_a_degenerate_interval_is_labelled_and_never_read_as_certainty(tables):
    text = tables.interval({"estimate": 1.0, "low": 1.0, "high": 1.0})
    assert "degenerate" in text and "does not establish certainty" in text
    assert "degenerate" not in tables.interval({"estimate": 0.9, "low": 0.8, "high": 0.95})


def test_a_derived_count_must_be_a_whole_number(tables):
    assert tables.whole(0.9067357512953368 * 193, "zone") == 175
    with pytest.raises(tables.TableError):
        tables.whole(174.5, "zone")


def test_p_values_are_printed_without_false_precision(tables):
    assert tables.pvalue(2.1988403187573934e-71) == "2.2 × 10⁻⁷¹"
    assert tables.pvalue(1.0) == "1"
    assert tables.pvalue(0.0123) == "0.012"


def test_records_that_disagree_stop_the_script(tables, inputs):
    broken = copy.deepcopy(inputs)
    broken["rows"]["primary"]["tp"] += 1                          # a log row that no longer matches the report
    with pytest.raises(tables.TableError):
        tables.render(broken)
    broken = copy.deepcopy(inputs)
    broken["coverage"]["rules"][tables.AMENDED_RULE][tables.CELL_2024]["kept"] -= 1
    with pytest.raises(tables.TableError):
        tables.render(broken)
    broken = copy.deepcopy(inputs)
    broken["report"]["populations"]["conflicting_interpretations"] += 1
    with pytest.raises(tables.TableError):
        tables.render(broken)
