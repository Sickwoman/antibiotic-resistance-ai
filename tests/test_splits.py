"""Tests for the Version 0.2 split infrastructure (synthetic metadata only)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.dataset import row_fingerprint
from src.splits import (
    PARTS,
    DatasetMismatchError,
    LeakageError,
    Split,
    SplitError,
    assert_usable,
    build_splits,
    check_split,
    derive_split,
    external_split,
    load_split,
    make_splits,
    random_split,
    temporal_split,
    within_year_split,
)
from src.utils import load_config


def make_meta(n_a: int = 400, n_b: int = 60, seed: int = 0) -> pd.DataFrame:
    """Site A: ~3 samples per patient group, dates 2016-2018. Site B: one group per sample, no dates."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n_a), unit="D")
    a = pd.DataFrame({
        "site": "DRIAMS-A",
        "year_folder": dates.year.astype(str),
        "code": [f"a{i:04d}" for i in range(n_a)],
        "acquisition_date": dates.strftime("%Y-%m-%d"),
        "label": (rng.random(n_a) < 0.3).astype(int),
        "group_id": rng.integers(0, n_a // 3, n_a),
        "group_source": "patient_no",
    })
    b = pd.DataFrame({
        "site": "DRIAMS-B", "year_folder": "2018", "code": [f"b{i:04d}" for i in range(n_b)],
        "acquisition_date": np.nan,
        "label": (rng.random(n_b) < 0.3).astype(int),
        "group_id": np.arange(n_a // 3, n_a // 3 + n_b),
        "group_source": "sample",
    })
    meta = pd.concat([a, b], ignore_index=True)
    meta.insert(0, "sample_index", np.arange(len(meta)))
    return meta


def test_random_split_is_disjoint_grouped_and_reproducible():
    meta = make_meta()
    s1 = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=42)
    s2 = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=42)
    s3 = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=7)
    for part in PARTS:
        assert np.array_equal(s1.parts()[part], s2.parts()[part])
    assert not np.array_equal(s1.test, s3.test)
    all_rows = np.concatenate([s1.train, s1.validation, s1.test])
    assert np.array_equal(np.sort(all_rows), np.flatnonzero(meta["site"] == "DRIAMS-A"))
    assert 0.12 < s1.test.size / all_rows.size < 0.28
    assert 0.05 < s1.validation.size / all_rows.size < 0.16
    for idx in s1.parts().values():            # both classes present everywhere
        assert set(meta["label"].iloc[idx]) == {0, 1}
    assert s1.dataset_fingerprint == row_fingerprint(meta)
    check_split(meta, s1)


def test_random_split_keeps_large_groups_together():
    meta = make_meta()
    meta.loc[:39, "group_id"] = 999            # one patient with 40 spectra
    split = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=1)
    parts_with_group = [p for p, idx in split.parts().items() if (meta["group_id"].iloc[idx] == 999).any()]
    assert len(parts_with_group) == 1


def test_within_year_split_uses_one_year_folder_only():
    meta = make_meta()
    split = within_year_split(meta, ["DRIAMS-A"], "2017", 0.2, 0.1, seed=42)
    again = within_year_split(meta, ["DRIAMS-A"], 2017, 0.2, 0.1, seed=42)
    expected = np.flatnonzero((meta["site"] == "DRIAMS-A") & (meta["year_folder"] == "2017"))
    assert np.array_equal(np.sort(np.concatenate(list(split.parts().values()))), expected)
    assert all(np.array_equal(split.parts()[p], again.parts()[p]) for p in PARTS)
    assert split.name == "within_year" and "patient groups are complete" in split.notes[0]
    assert 0.12 < split.test.size / expected.size < 0.28
    check_split(meta, split)

    meta.loc[expected[:2], "group_source"] = "sample"
    split = within_year_split(meta, ["DRIAMS-A"], "2017", 0.2, 0.1, seed=42)
    assert any("2 sample(s) have no patient ID" in note for note in split.notes)
    with pytest.raises(SplitError, match="No samples in year folder 2014"):
        within_year_split(meta, ["DRIAMS-A"], "2014", 0.2, 0.1, seed=42)


def test_temporal_split_uses_acquisition_date_not_folder():
    meta = make_meta()
    # a spectrum filed under 2018 but acquired in 2017 must be judged by its date
    meta.loc[0, ["year_folder", "acquisition_date", "group_id"]] = ["2018", "2017-12-30", 5000]
    split = temporal_split(meta, ["DRIAMS-A"], "2017-10-01", "2018-01-01")
    dates = pd.to_datetime(meta["acquisition_date"])
    assert (dates.iloc[split.test] >= "2018-01-01").all()
    assert ((dates.iloc[split.validation] >= "2017-10-01") & (dates.iloc[split.validation] < "2018-01-01")).all()
    assert (dates.iloc[split.train] < "2017-10-01").all()
    assert 0 in split.validation
    assert not set(split.test) & set(np.flatnonzero(meta["site"] == "DRIAMS-B"))
    check_split(meta, split)


def test_temporal_split_removes_groups_that_cross_the_boundary():
    meta = make_meta()
    meta.loc[[0, 1], "group_id"] = 7777
    meta.loc[0, "acquisition_date"] = "2016-05-01"     # train period
    meta.loc[1, "acquisition_date"] = "2018-03-01"     # test period
    split = temporal_split(meta, ["DRIAMS-A"], "2017-10-01", "2018-01-01")
    assert 1 in split.test and 0 not in split.train
    assert any("Removed" in note for note in split.notes)
    check_split(meta, split)


def test_temporal_split_validation():
    meta = make_meta()
    with pytest.raises(SplitError, match="acquisition_date"):
        temporal_split(meta, ["DRIAMS-A"], "2017-10-01", "2018-01-01", date_column="year_folder")
    with pytest.raises(SplitError, match="earlier"):
        temporal_split(meta, ["DRIAMS-A"], "2018-01-01", "2017-10-01")
    with pytest.raises(SplitError, match="No acquisition dates"):
        temporal_split(meta, ["DRIAMS-B"], "2017-10-01", "2018-01-01")
    meta.loc[3, "acquisition_date"] = np.nan
    split = temporal_split(meta, ["DRIAMS-A"], "2017-10-01", "2018-01-01")
    assert 3 not in np.concatenate(list(split.parts().values()))
    assert "1 sample(s) without acquisition_date" in split.notes[0]


def test_external_split():
    meta = make_meta()
    split = external_split(meta, ["DRIAMS-A"], ["DRIAMS-B"], 0.1, seed=3)
    assert set(meta["site"].iloc[split.test]) == {"DRIAMS-B"}
    assert set(meta["site"].iloc[np.concatenate([split.train, split.validation])]) == {"DRIAMS-A"}
    assert split.test.size == 60
    with pytest.raises(SplitError, match="both training and external"):
        external_split(meta, ["DRIAMS-A"], ["DRIAMS-A"], 0.1, seed=3)
    with pytest.raises(SplitError, match="not in this dataset"):
        external_split(meta, ["DRIAMS-A"], ["DRIAMS-C"], 0.1, seed=3)


def test_check_split_detects_leakage():
    meta = make_meta()
    meta.loc[[0, 1], "group_id"] = 123
    with pytest.raises(LeakageError, match="patient group"):
        check_split(meta, Split("bad", np.array([0]), np.array([], dtype=int), np.array([1]), ""))
    with pytest.raises(LeakageError, match="samples shared"):
        check_split(meta, Split("bad", np.array([5, 6]), np.array([], dtype=int), np.array([6]), ""))
    with pytest.raises(LeakageError, match="duplicate"):
        check_split(meta, Split("bad", np.array([5, 5]), np.array([], dtype=int), np.array([9]), ""))
    with pytest.raises(LeakageError, match="outside"):
        check_split(meta, Split("bad", np.array([len(meta)]), np.array([], dtype=int), np.array([9]), ""))


def test_split_save_and_load(tmp_path):
    meta = make_meta()
    split = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=42)
    split.save(tmp_path / "random.json", meta)
    loaded = load_split(tmp_path / "random.json", meta)
    for part in PARTS:
        assert np.array_equal(loaded.parts()[part], split.parts()[part])
    assert loaded.dataset_fingerprint == row_fingerprint(meta)
    saved = (tmp_path / "random.json").read_text(encoding="utf-8")
    assert '"resistant"' in saved and '"dataset_fingerprint"' in saved
    assert "group_id" not in saved and "a0000" not in saved          # no group IDs, no sample codes


def test_saved_split_only_loads_with_its_own_dataset(tmp_path):
    meta = make_meta()
    path = tmp_path / "random.json"
    random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=42).save(path, meta)

    other_label = meta.copy()
    other_label.loc[0, "label"] = 1 - other_label.loc[0, "label"]
    other_group = meta.copy()
    other_group.loc[0, "group_id"] = 99999
    other_order = meta.iloc[::-1].reset_index(drop=True)
    one_row_less = meta.iloc[:-1]           # only a DRIAMS-B row is gone: every index is still in range
    for other in (other_label, other_group, other_order, one_row_less):
        with pytest.raises(DatasetMismatchError, match="another dataset build"):
            load_split(path, other)

    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["dataset_fingerprint"]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetMismatchError, match="no dataset fingerprint"):
        load_split(legacy, meta)


def test_derived_split_keeps_every_sample_in_its_reference_part():
    reference = make_meta()
    ref_split = random_split(reference, ["DRIAMS-A"], 0.2, 0.1, seed=42)
    keep = np.random.default_rng(1).random(len(reference)) > 0.15      # e.g. I results removed
    variant = reference[keep].sample(frac=1, random_state=3)            # another row order on purpose
    variant = pd.concat([variant, variant.iloc[:1].assign(code="new-sample")], ignore_index=True)
    variant["sample_index"] = np.arange(len(variant))

    derived = derive_split(ref_split, reference, variant, "reference_ds")
    ref_part = {code: part for part, idx in ref_split.parts().items() for code in reference["code"].iloc[idx]}
    for part, idx in derived.parts().items():
        assert all(ref_part[code] == part for code in variant["code"].iloc[idx])
    used = set(variant["code"].iloc[np.concatenate(list(derived.parts().values()))])
    assert used == {code for code in variant["code"] if code in ref_part}   # new-sample and site B are left out
    assert any("1 sample(s) are not in reference_ds" in note for note in derived.notes)
    assert any("not used by the reference split" in note for note in derived.notes)
    assert derived.derived_from == {"dataset": "reference_ds", "dataset_fingerprint": ref_split.dataset_fingerprint}
    assert derived.dataset_fingerprint == row_fingerprint(variant)
    check_split(variant, derived)
    with pytest.raises(DatasetMismatchError):
        derive_split(ref_split, variant, variant, "reference_ds")      # reference split paired with wrong data


def test_build_splits_needs_the_primary_dataset_first(tmp_path):
    config = copy.deepcopy(load_config())
    with pytest.raises(SplitError, match="Build it first"):
        build_splits(make_meta(), config, config["dataset"]["name"] + "__intermediate-exclude", tmp_path)


def test_unreachable_fractions_are_rejected_and_real_fractions_reported():
    meta = make_meta()
    with pytest.raises(SplitError, match="cannot be produced"):
        random_split(meta, ["DRIAMS-A"], 0.4, 0.1, seed=1)       # folds would give 50 %, not 40 %
    split = random_split(meta, ["DRIAMS-A"], 0.25, 0.15, seed=1)  # 0.15 / 0.75 = 0.2 is reachable
    assert "requested 25% / 15%" in split.description


def test_make_splits_skips_splits_whose_sites_are_missing():
    config = copy.deepcopy(load_config())
    only_b = make_meta()
    only_b = only_b[only_b["site"] == "DRIAMS-B"].reset_index(drop=True)
    skipped: list[str] = []
    assert make_splits(only_b, config, skipped) == {}
    # one message per configured split: random, within_year, temporal, external, external_ab
    assert len(skipped) == len(config["splits"]) and all("not in this dataset" in s for s in skipped)

    config["splits"]["within_year"]["year_folder"] = "2014"
    skipped = []
    assert "within_year" not in make_splits(make_meta(), config, skipped)
    assert "within_year: year folder 2014 not in this dataset" in skipped
    # This fixture has DRIAMS-A and DRIAMS-B but no DRIAMS-D, so external_ab is skipped for a missing
    # site. Nothing else may be skipped.
    assert [s for s in skipped if "year folder" not in s] == ["external_ab: site(s) ['DRIAMS-D'] not in this dataset"]


def test_make_splits_from_project_config():
    meta = make_meta()
    config = copy.deepcopy(load_config())
    config["splits"]["external"]["test_sites"] = ["DRIAMS-B", "DRIAMS-C"]
    splits = make_splits(meta, config)
    assert set(splits) == {"random", "within_year", "temporal", "external"}
    assert any("DRIAMS-C" in note for note in splits["external"].notes)
    for split in splits.values():
        check_split(meta, split)
    only_a = meta[meta["site"] == "DRIAMS-A"].reset_index(drop=True)
    assert set(make_splits(only_a, config)) == {"random", "within_year", "temporal"}


# --- review fix: a split that cannot carry the metrics must not reach a model -------------------------------------

def test_assert_usable_accepts_a_real_split():
    meta = make_meta()
    assert assert_usable(meta, random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=42)) is None


def test_assert_usable_refuses_empty_and_single_class_parts():
    meta = make_meta()
    resistant = np.flatnonzero(meta["label"].to_numpy() == 1)
    susceptible = np.flatnonzero(meta["label"].to_numpy() == 0)
    both = np.concatenate([resistant[:10], susceptible[:10]])
    empty = Split("bad", both, np.array([], dtype=int), np.concatenate([resistant[10:20], susceptible[10:20]]), "")
    with pytest.raises(SplitError, match="validation part is empty"):
        assert_usable(meta, empty)
    one_class = Split("bad", both, susceptible[20:30], np.concatenate([resistant[10:20], susceptible[30:40]]), "")
    with pytest.raises(SplitError, match="validation part holds 0 resistant"):
        assert_usable(meta, one_class)
    # the same split is fine when only the parts that matter are checked
    assert assert_usable(meta, one_class, parts=("train", "test")) is None


def test_assert_usable_can_require_more_than_one_of_each_class():
    meta = make_meta()
    split = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=42)
    with pytest.raises(SplitError, match="fewer than 10000 of one class"):
        assert_usable(meta, split, min_per_class=10_000)
