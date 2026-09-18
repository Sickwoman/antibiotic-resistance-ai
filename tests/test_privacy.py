"""Nothing committed to this repository may carry a patient, case or order identifier.

DRIAMS-A's patient_no / case_no / order_no are 32-character hex hashes. They are dropped when the dataset
is built (src/dataset.PRIVATE_COLUMNS), but a report, a plot caption or a notebook output could still
carry one by accident, so every committed text file outside tests/ is scanned here. This automates the
check that was done by hand at the end of Versions 0.2 to 0.5.

tests/ is excluded on purpose: its fixtures use invented identifiers to prove that the code removes them,
and the last test here proves the patterns still catch a real-looking one.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.dataset import PRIVATE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SCANNED_SUFFIXES = {".md", ".csv", ".json", ".yaml", ".yml", ".txt", ".ipynb", ".py", ".cff", ".toml"}
HASH_LIKE = re.compile(r"\b[0-9a-f]{32}\b")                                    # patient_no / case_no / order_no
UUID_LIKE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / line for line in out.stdout.splitlines() if line]


def public_checksums() -> set[str]:
    """The DRIAMS archive checksums from config.yaml: public values, and the only 32-hex ones allowed.

    They reach the reports because every run records the configuration it used.
    """
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    found = {str(v) for line in text.splitlines() if "checksum" in line for v in HASH_LIKE.findall(line)}
    assert found, "no archive checksums found in config.yaml; the allow-list would be empty"
    assert config["project"]["name"]                      # the file really is the project configuration
    return found


@pytest.fixture(scope="module")
def committed() -> list[tuple[str, str]]:
    files = [p for p in tracked_files()
             if p.suffix.lower() in SCANNED_SUFFIXES and p.is_file() and not p.is_relative_to(ROOT / "tests")]
    assert len(files) > 20, "the repository listing looks wrong; the scan would pass for the wrong reason"
    return [(p.relative_to(ROOT).as_posix(), p.read_text(encoding="utf-8", errors="replace")) for p in files]


def test_no_patient_case_or_order_hash_is_committed(committed):
    """A 32-character hex value is what a DRIAMS-A identifier looks like."""
    allowed = public_checksums()
    offenders = {name: sorted(set(HASH_LIKE.findall(text)) - allowed)[:3] for name, text in committed}
    offenders = {name: found for name, found in offenders.items() if found}
    assert not offenders, f"identifier-like hashes found: {offenders}"


def test_no_uuid_identifier_is_committed(committed):
    """DRIAMS-D spectrum codes are UUIDs; none of them belongs in a committed file either."""
    offenders = {name: UUID_LIKE.findall(text)[:3] for name, text in committed if UUID_LIKE.search(text)}
    assert not offenders, f"UUID-like identifiers found: {offenders}"


def test_no_committed_table_has_a_private_column(committed):
    """Naming a column in prose is fine; a column of values is not, so the CSV headers are checked."""
    offenders = []
    for name, _ in committed:
        if not name.endswith(".csv"):
            continue
        header = pd.read_csv(ROOT / name, nrows=0).columns.str.lower()
        if header.isin(PRIVATE_COLUMNS).any():
            offenders.append(name)
    assert not offenders, f"private columns exported in: {offenders}"


def test_the_scan_would_notice_an_identifier():
    """The patterns are checked against examples, so a passing scan means something."""
    assert HASH_LIKE.search("patient " + "a1b2c3d4" * 4)
    assert UUID_LIKE.search("code 0a1b2c3d-4e5f-6789-abcd-ef0123456789")
    assert not HASH_LIKE.search("a1b2c3d4" * 3)                      # 24 characters: not an identifier
    assert not HASH_LIKE.search("z" + "a1b2c3d4" * 4)                # not hexadecimal
    assert PRIVATE_COLUMNS == ("patient_no", "case_no", "order_no")
