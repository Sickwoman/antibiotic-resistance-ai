"""Tests for the Version 0.2 split infrastructure (synthetic metadata only)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.splits import (
    LeakageError,
    Split,
    SplitError,
    check_split,
    external_split,
    load_split,
    make_splits,
    random_split,
    temporal_split,
)
from src.utils import load_config


def make_meta(n_a: int = 400, n_b: int = 60, seed: int = 0) -> pd.DataFrame:
    """Site A: ~3 samples per patient group, dates 2016-2018. Site B: one group per sample, no dates."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n_a), unit="D")
    a = pd.DataFrame({
        "site": "DRIAMS-A",
        "year_folder": dates.year.astype(str),
        "acquisition_date": dates.strftime("%Y-%m-%d"),
        "label": (rng.random(n_a) < 0.3).astype(int),
        "group_id": rng.integers(0, n_a // 3, n_a),
    })
    b = pd.DataFrame({
        "site": "DRIAMS-B", "year_folder": "2018", "acquisition_date": np.nan,
        "label": (rng.random(n_b) < 0.3).astype(int),
        "group_id": np.arange(n_a // 3, n_a // 3 + n_b),
    })
    meta = pd.concat([a, b], ignore_index=True)
    meta.insert(0, "sample_index", np.arange(len(meta)))
    return meta


def test_random_split_is_disjoint_grouped_and_reproducible():
    meta = make_meta()
    s1 = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=42)
    s2 = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=42)
    s3 = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=7)
    for part in ("train", "validation", "test"):
        assert np.array_equal(s1.parts()[part], s2.parts()[part])
    assert not np.array_equal(s1.test, s3.test)
    all_rows = np.concatenate([s1.train, s1.validation, s1.test])
    assert np.array_equal(np.sort(all_rows), np.flatnonzero(meta["site"] == "DRIAMS-A"))
    assert 0.12 < s1.test.size / all_rows.size < 0.28
    assert 0.05 < s1.validation.size / all_rows.size < 0.16
    for idx in s1.parts().values():            # both classes present everywhere
        assert set(meta["label"].iloc[idx]) == {0, 1}
    check_split(meta, s1)


def test_random_split_keeps_large_groups_together():
    meta = make_meta()
    meta.loc[:39, "group_id"] = 999            # one patient with 40 spectra
    split = random_split(meta, ["DRIAMS-A"], 0.2, 0.1, seed=1)
    parts_with_group = [p for p, idx in split.parts().items() if (meta["group_id"].iloc[idx] == 999).any()]
    assert len(parts_with_group) == 1


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
    loaded = load_split(tmp_path / "random.json")
    for part in ("train", "validation", "test"):
        assert np.array_equal(loaded.parts()[part], split.parts()[part])
    saved = (tmp_path / "random.json").read_text(encoding="utf-8")
    assert '"resistant"' in saved and "group_id" not in saved


def test_make_splits_from_project_config():
    meta = make_meta()
    config = copy.deepcopy(load_config())
    config["splits"]["external"]["test_sites"] = ["DRIAMS-B", "DRIAMS-C"]
    splits = make_splits(meta, config)
    assert set(splits) == {"random", "temporal", "external"}
    assert any("DRIAMS-C" in note for note in splits["external"].notes)
    for split in splits.values():
        check_split(meta, split)
    only_a = meta[meta["site"] == "DRIAMS-A"].reset_index(drop=True)
    assert set(make_splits(only_a, config)) == {"random", "temporal"}
