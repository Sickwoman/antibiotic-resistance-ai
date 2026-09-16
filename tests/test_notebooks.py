"""Committed notebooks must be valid and must not show identifier-like hashes in their saved outputs."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
# DRIAMS-A patient_no, case_no and order_no values are 32-character lowercase hex hashes.
HASH_LIKE = re.compile(r"\b[0-9a-f]{32}\b")


def output_texts(nb) -> list[str]:
    """Text parts of all saved outputs (stream text, text/plain, text/html; images are skipped)."""
    texts = []
    for cell in nb.cells:
        for output in cell.get("outputs", []) if cell.cell_type == "code" else []:
            parts = [output.get("text", "")]
            parts += [value for key, value in output.get("data", {}).items() if key.startswith("text/")]
            texts += ["".join(p) if isinstance(p, list) else p for p in parts]
    return texts


def test_both_project_notebooks_are_found():
    assert {"01_data_exploration.ipynb", "02_preprocessing.ipynb"} <= {p.name for p in NOTEBOOKS}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid(path):
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_outputs_have_no_identifier_like_hashes(path):
    nb = nbformat.read(path, as_version=4)
    hits = sum(len(HASH_LIKE.findall(text)) for text in output_texts(nb))
    assert hits == 0, f"{path.name}: {hits} saved output value(s) look like patient/case/order hashes"


def test_hash_pattern_catches_driams_style_values():
    fake = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(outputs=[
        nbformat.v4.new_output("stream", text="patient 0123456789abcdef0123456789abcdef\n")])])
    assert sum(len(HASH_LIKE.findall(t)) for t in output_texts(fake)) == 1
    # spectrum codes are UUIDs with dashes and are not matched
    assert not HASH_LIKE.search("0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0")
