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


# --- amendment C4: the Sample field (categories and counts only; per-isolate output is the mapped class only) ---------

def sample_row(identifier, sample):
    return [identifier, "SENT-POS", "2020", f"x/{identifier}", "SENT-ORG-1618", "Escherichia coli", sample,
            "SENT-MECH-1732", "SENT-MIC-2718", "SENT-RES-4711", "SENT-MIC-2718", "SENT-SUS-0815", "SENT-MIC-2718",
            "SENT-RES-4711"]


def sample_fixture(tmp_path, layout="2.0.0"):
    ids = [f"IDSENT{i:04d}" for i in range(200)]
    groups = ([("Urine", 120), ("Blood culture", 60), ("Rectal swab", 10), ("Pus", 2),
               ("Jane SENT-FREE 1985-03-02", 1), ("contact SENT-MAIL@example.org", 1), ("", 1), ("NA", 1), ("-", 1)])
    rows, k = [], 0
    for value, n in groups:
        for _ in range(n):
            rows.append(sample_row(ids[k], value))
            k += 1
    rows.append(sample_row(ids[0], ""))                         # a second record without a value: still "Urine"
    for _ in range(2):                                           # two isolates whose records disagree
        rows += [sample_row(ids[k], "Urine"), sample_row(ids[k], "Blood culture")]
        k += 1
    rows.append(sample_row("OUTSIDE-COHORT", "SENT-SAMPLE-OUTSIDE"))
    cohort = ids[:k] + ["IDSENT-UNMATCHED"]
    return write_csv(tmp_path / "AMR.csv", rows, layout=layout), cohort


SAMPLE_SENTINELS = SENTINELS + ["IDSENT", "SENT-FREE", "SENT-MAIL", "SENT-SAMPLE-OUTSIDE", "1985"]


def assert_no_sample_leak(*texts):
    for text in texts:
        for s in SAMPLE_SENTINELS:
            assert s not in text, f"a planted marker leaked: {s}"


@pytest.mark.parametrize("layout", ["old", "2.0.0"])
def test_sample_categories_give_named_categories_and_counts_only(tmp_path, capsys, caplog, layout):
    from src.marisma_schema import SampleSummary, sample_categories
    caplog.set_level(logging.DEBUG)
    path, cohort = sample_fixture(tmp_path, layout)
    summary = sample_categories(path, cohort)
    assert summary == SampleSummary(
        categories={"Urine": 120, "Blood culture": 60, "Rectal swab": 10}, missing=3, conflicting=2,
        suppressed_categories=3, suppressed_isolates=4, matched_isolates=199, unmatched_isolates=1, schema_problem=None)
    out = capsys.readouterr()
    assert_no_sample_leak(repr(summary), str(summary), out.out, out.err, caplog.text)


def test_sample_summary_has_exactly_the_permitted_fields():
    from src.marisma_schema import SampleSummary
    assert set(SampleSummary.__dataclass_fields__) == {
        "categories", "missing", "conflicting", "suppressed_categories", "suppressed_isolates", "matched_isolates",
        "unmatched_isolates", "schema_problem"}


@pytest.mark.parametrize("value", ["Jane Doe 1985-03-02", "a@b.org", "see http://x",
                                   "one two three four five six seven", "x" * 61,"tab\there"])
def test_identifying_or_free_text_values_are_never_named(value):
    from src.marisma_schema import nameable
    assert not nameable(value)


def test_free_text_fields_are_suppressed_entirely(tmp_path):
    from src.marisma_schema import sample_categories
    many = write_csv(tmp_path / "many.csv", [sample_row(f"I{i}", f"Category {chr(65 + i % 26)}{i // 26}")
                                             for i in range(150)])
    summary = sample_categories(many, [f"I{i}" for i in range(150)])
    assert summary.categories == {} and summary.schema_problem and summary.suppressed_isolates == 150
    rare = write_csv(tmp_path / "rare.csv", [sample_row(f"I{i}", "Urine") for i in range(90)]
                     + [sample_row(f"J{i}", f"Rare {chr(65 + i)}") for i in range(6)])
    summary = sample_categories(rare, [f"I{i}" for i in range(90)] + [f"J{i}" for i in range(6)])
    assert summary.categories == {} and "5 %" in summary.schema_problem       # 6 of 96 suppressed: over 5 %


def test_source_classes_return_classes_never_values(tmp_path, capsys):
    from src.marisma_schema import source_classes
    path, cohort = sample_fixture(tmp_path)
    mapping = {"Urine": "clinical", "Blood culture": "clinical", "Rectal swab": "screening"}
    classes = source_classes(path, cohort, mapping)
    assert set(classes.values()) <= {"clinical", "screening", "ambiguous"}
    counts = {c: list(classes.values()).count(c) for c in ("clinical", "screening", "ambiguous")}
    assert counts == {"clinical": 180, "screening": 10, "ambiguous": 9}   # suppressed 4 + missing 3 + conflicting 2
    assert set(classes) <= set(cohort) and "IDSENT-UNMATCHED" not in classes
    assert_no_sample_leak("".join(classes.values()), capsys.readouterr().out)


@pytest.mark.parametrize("mapping", [{"Pus": "clinical"},                 # a suppressed category cannot be mapped
                                     {"Urine": "kept"}])                   # not one of the three classes
def test_source_classes_refuse_an_invalid_mapping_without_content(tmp_path, mapping):
    from src.marisma_schema import source_classes
    path, cohort = sample_fixture(tmp_path)
    with pytest.raises(RestrictedReaderError) as e:
        source_classes(path, cohort, mapping)
    assert_no_sample_leak(str(e.value))
    assert "Pus" not in str(e.value)
