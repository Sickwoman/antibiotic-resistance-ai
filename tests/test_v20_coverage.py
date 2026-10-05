"""The coverage investigation's helpers (scripts/v20_coverage_investigation.py), on synthetic spectra only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing import PreprocessingConfig, bin_intensities, trim

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v20_coverage_investigation.py"


@pytest.fixture(scope="module")
def cov():
    spec = importlib.util.spec_from_file_location("v20_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bin_support_counts_what_the_pipeline_bins(cov):
    cfg = PreprocessingConfig()
    mz = np.sort(np.random.default_rng(1).uniform(1990, 20010, size=30000))
    kept_mz, kept = trim(mz, np.ones_like(mz), cfg.mz_min, cfg.mz_max)
    assert np.array_equal(cov.bin_support(mz, cfg), bin_intensities(kept_mz, kept, cfg.bin_edges).astype(int))


def test_bin_boundaries_are_those_of_the_pipeline(cov):
    cfg = PreprocessingConfig()
    support = cov.bin_support(np.array([2003.0, 19997.0, 20000.0]), cfg)
    assert support[0] == 0 and support[1] == 1           # [2000, 2003) excludes 2003.0
    assert support[5999] == 2                            # the last bin, [19997, 20000], is closed
    assert cov.bin_support(np.array([1999.99, 20000.01]), cfg).sum() == 0     # trimmed away


def test_truncate_keeps_inclusive_bounds(cov):
    mz = np.array([1.0, 2.0, 3.0, 4.0])
    kept, y = cov.truncate(mz, mz * 10, 2.0, 3.0)
    assert np.array_equal(kept, [2.0, 3.0]) and np.array_equal(y, [20.0, 30.0])


def test_edge_extent(cov):
    local = np.zeros(6000)
    local[[0, 3, 5990]] = 0.5
    assert cov.edge_extent(local, 0.01) == (4, 10)
    assert cov.edge_extent(np.zeros(6000), 0.01) == (0, 0)


def test_cutting_only_the_margin_changes_no_interior_feature(cov):
    cfg = PreprocessingConfig()
    spectra = cov.synthetic_spectra(3, seed=7)
    summary, curves = cov.compare_truncations(spectra, [("cut margin", 1999.9, None), ("cut bin 0", 2003.1, None)],
                                              cfg)
    margin = summary["cut margin"]
    assert margin["tic_ratio"]["q100"] < 1                                    # less area, so a different scale
    # but no interior bin changes beyond the float32 rounding of the stored features (about 1e-7 of a bin's value,
    # so a few 1e-6 of the mean bin for a peak bin), far below the 0.1 % and 1 % thresholds the investigation uses
    assert margin["interior_max_abs_local_change"]["q100"] < 1e-4
    # cutting at 2003.1 Da leaves bin 0 with no acquired point, so its feature falls (to zero: the change, relative to
    # the mean bin, is minus that bin's own relative size)
    mz, y = spectra[0]
    assert cov.bin_support(cov.truncate(mz, y, 2003.1, None)[0], cfg)[0] == 0
    assert summary["cut bin 0"]["first_bin_local_change"]["q100"] < 0
    assert curves["cut margin"].shape == (6000,)
