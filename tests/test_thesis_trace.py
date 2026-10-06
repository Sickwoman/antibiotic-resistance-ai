"""The thesis chapter's Version 2.0 section (section 11) is traced to the research report (Appendix A, part 2).

Every number in that section must occur verbatim in the report, and the committed part 2 of the appendix must be
exactly what `scripts/trace_thesis_numbers.py` generates from the committed chapter and report.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracer():
    spec = importlib.util.spec_from_file_location("trace_thesis_numbers", ROOT / "scripts/trace_thesis_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_number_in_section_11_occurs_in_the_report():
    t = tracer()
    rows = t.trace(t.CHAPTER.read_text(encoding="utf-8"), t.REPORT.read_text(encoding="utf-8"))
    assert rows and not [n for n, count, _ in rows if count == 0]
    assert {"0.772", "0.642", "0.393", "0.907", "193", "−0.021"} <= {n for n, _, _ in rows}


def test_appendix_part_2_is_what_the_tracer_generates():
    t = tracer()
    rows = t.trace(t.CHAPTER.read_text(encoding="utf-8"), t.REPORT.read_text(encoding="utf-8"))
    appendix = (ROOT / "docs/manuscript/number_trace.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert appendix.endswith(t.table(rows))
