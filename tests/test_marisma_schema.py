"""The restricted schema reader (amendment A2), on synthetic files with planted outcome, MIC and metadata markers.

Every interpretation, MIC, sample, mechanism and organism value in these fixtures is a unique sentinel string. The
reader's return value, its repr, standard output, standard error, logs and exception messages must contain none of
them. Only antibiotic names, availability counts and match counts may come out.
"""

from __future__ import annotations

import logging

import pytest

from src.marisma_schema import RestrictedReaderError, SchemaSummary, antibiotic_names, schema_summary

HEADER = ["Identifier", "target_position", "Year", "Path", "Microorganism", "Species", "Sample",
          "Individual Resistance Mechanism", "CMI_Ciprofloxacino", "Ciprofloxacino", "CMI_Ceftriaxona", "Ceftriaxona",
          "CMI_Colistina", "Colistina"]
SENTINELS = ["SENT-RES-4711", "SENT-SUS-0815", "SENT-MIC-2718", "SENT-SRC-1414", "SENT-MECH-1732", "SENT-ORG-1618"]


def row(identifier, cip, cro, *, extra_mic="SENT-MIC-2718"):
    return [identifier, "SENT-POS", "2020", f"x/{identifier}", "SENT-ORG-1618", "Escherichia coli", "SENT-SRC-1414",
            "SENT-MECH-1732", extra_mic, cip, extra_mic, cro, extra_mic, "SENT-RES-4711"]


def write_csv(path, rows, header=HEADER, *, layout="old"):
    """"old": MARISMa's older pipeline (";" with "CMI_" and a byte-order mark); "2.0.0": "," with "MIC_", no mark."""
    delimiter, encoding = (";", "utf-8-sig") if layout == "old" else (",", "utf-8")
    if layout != "old":
        header = [c.replace("CMI_", "MIC_") for c in header]
    text = "\n".join(delimiter.join(r) for r in [header, *rows]) + "\n"
    path.write_text(text, encoding=encoding)
    return path


ROWS = [
    row("a1", "SENT-RES-4711", "SENT-SUS-0815"),
    row("a1", "SENT-RES-4711", "SENT-SUS-0815"),       # a duplicate record of the same isolate counts once
    row("a2", "-", "SENT-RES-4711"),
    row("a3", "ND", ""),
    row("a4", " SENT-SUS-0815 ", "NaN"),
    row("zz", "SENT-RES-4711", "SENT-RES-4711"),       # not in the cohort: ignored
]
COLUMNS = {"ciprofloxacin": "Ciprofloxacino", "ceftriaxone": "Ceftriaxona"}


def assert_no_sentinel(*texts):
    for text in texts:
        for s in SENTINELS:
            assert s not in text, f"a planted marker leaked: {s}"


@pytest.mark.parametrize("layout", ["old", "2.0.0"])
def test_only_names_and_counts_come_out(tmp_path, capsys, caplog, layout):
    caplog.set_level(logging.DEBUG)
    path = write_csv(tmp_path / "AMR.csv", ROWS, layout=layout)
    summary = schema_summary(path, ["a1", "a2", "a3", "a4", "a5"], COLUMNS)
    assert summary == SchemaSummary(
        antibiotics=("Ciprofloxacino", "Ceftriaxona", "Colistina"), matched_isolates=4, unmatched_isolates=1,
        non_missing={"ciprofloxacin": 2, "ceftriaxone": 2})
    out = capsys.readouterr()
    assert_no_sentinel(repr(summary), str(summary), out.out, out.err, caplog.text)


def test_the_output_object_has_exactly_the_permitted_fields():
    # amendment A2's permitted outputs, and nothing else
    assert set(SchemaSummary.__dataclass_fields__) == {
        "antibiotics", "matched_isolates", "unmatched_isolates", "non_missing"}


@pytest.mark.parametrize("bad_row", [
    ["a1", "SENT-RES-4711"],                                   # too short
    row("a1", "SENT-RES-4711", "SENT-SUS-0815") + ["SENT-MIC-2718"],   # too long
])
def test_malformed_rows_raise_without_content(tmp_path, bad_row):
    path = write_csv(tmp_path / "AMR.csv", [row("a2", "-", "-"), bad_row])
    with pytest.raises(RestrictedReaderError) as e:
        schema_summary(path, ["a1", "a2"], COLUMNS)
    assert_no_sentinel(str(e.value), repr(e.value))
    # no chained exception that could carry row content into a traceback
    assert e.value.__cause__ is None
    assert e.value.__context__ is None or e.value.__suppress_context__


def test_an_unrequested_or_unpaired_antibiotic_is_refused_not_substituted(tmp_path):
    path = write_csv(tmp_path / "AMR.csv", ROWS)
    with pytest.raises(RestrictedReaderError) as e:
        schema_summary(path, ["a1"], {"ceftriaxone": "Cefotaxima"})   # absent: no fuzzy match, no substitute
    assert "ceftriaxone" in str(e.value)
    with pytest.raises(RestrictedReaderError):
        schema_summary(path, ["a1"], {"x": "Sample"})                 # a non-antibiotic column is never counted


def test_a_missing_identifier_column_is_refused(tmp_path):
    path = write_csv(tmp_path / "AMR.csv", [], header=["ID", "CMI_Ciprofloxacino", "Ciprofloxacino"])
    with pytest.raises(RestrictedReaderError):
        schema_summary(path, ["a1"], {"ciprofloxacin": "Ciprofloxacino"})


def test_antibiotic_names_need_both_columns():
    assert antibiotic_names(["Identifier", "CMI_A", "A", "CMI_B", "C", "Sample"]) == ("A",)


def test_the_header_is_read_without_the_byte_order_mark(tmp_path):
    path = write_csv(tmp_path / "AMR.csv", ROWS)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert schema_summary(path, ["a1"], COLUMNS).matched_isolates == 1


def test_the_delimiter_is_detected_from_the_header_only():
    from src.marisma_schema import RestrictedReaderError, detect_delimiter
    assert detect_delimiter("Identifier,MIC_A,A\n") == ","
    assert detect_delimiter("Identifier;CMI_A;A\n") == ";"
    with pytest.raises(RestrictedReaderError):
        detect_delimiter("Identifier\n")                 # no delimiter at all
    with pytest.raises(RestrictedReaderError):
        detect_delimiter("a,b;c\n")                      # a tie is refused, not guessed


def test_antibiotic_names_accept_both_prefixes_without_duplicates():
    assert antibiotic_names(["Identifier", "MIC_A", "A", "CMI_A", "CMI_B", "B", "MIC_C"]) == ("A", "B")
