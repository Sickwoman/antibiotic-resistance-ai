"""Tests for src.preprocessing (synthetic spectra; one optional check against real DRIAMS files)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import SpectrumFormatError, read_binned_spectrum
from src.preprocessing import (
    PreprocessingConfig,
    PreprocessingError,
    bin_intensities,
    preprocess_arrays,
    preprocess_file,
    savitzky_golay,
    savitzky_golay_coefficients,
    snip_baseline,
    total_ion_current,
    validate_spectrum,
)
from src.utils import load_config

CFG = PreprocessingConfig()


def synthetic_spectrum(n: int = 4000, lo: float = 1950.0, hi: float = 20150.0, seed: int = 0):
    """TOF-like spacing, decaying baseline, a few Gaussian peaks, Poisson noise (non-negative counts)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(np.sqrt(lo), np.sqrt(hi), n)
    mz = t ** 2
    baseline = 3000 * np.exp(-(mz - lo) / 3000) + 50
    peaks = sum(a * np.exp(-0.5 * ((mz - c) / 8) ** 2)
                for c, a in [(2500, 20000), (4365, 9000), (6255, 12000), (9540, 7000), (15000, 3000)])
    intensity = rng.poisson(baseline + peaks).astype(float)
    return mz, intensity


def write_spectrum(path: Path, mz, intensity, header: bool = True) -> Path:
    lines = ["#  /some/instrument/fid", "#  code"]
    if header:
        lines.append('"mass.myspec..1..." "intensity.myspec..1..."')
    lines += [f"{float(m)!r} {float(i):g}" for m, i in zip(mz, intensity)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- whole pipeline

def test_valid_spectrum_gives_fixed_length_float32_vector():
    mz, inten = synthetic_spectrum()
    features, info = preprocess_arrays(mz, inten, CFG)
    assert features.shape == (6000,) and CFG.n_bins == 6000
    assert features.dtype == np.float32
    assert np.isfinite(features).all() and (features >= 0).all()
    assert info.raw_points == 4000 and 0 < info.nonzero_bins <= 6000
    assert info.points_in_range == int(((mz >= 2000) & (mz <= 20000)).sum())


def test_preprocessing_is_deterministic():
    mz, inten = synthetic_spectrum()
    a, _ = preprocess_arrays(mz, inten, CFG)
    b, _ = preprocess_arrays(mz.copy(), inten.copy(), CFG)
    assert a.tobytes() == b.tobytes()


def test_input_arrays_are_not_modified():
    mz, inten = synthetic_spectrum()
    mz0, inten0 = mz.copy(), inten.copy()
    preprocess_arrays(mz, inten, CFG)
    assert np.array_equal(mz, mz0) and np.array_equal(inten, inten0)


def test_peaks_end_up_in_the_right_bins():
    mz, inten = synthetic_spectrum()
    features, _ = preprocess_arrays(mz, inten, CFG)
    peak_bin = int((6255 - 2000) // 3)
    assert features[peak_bin - 5: peak_bin + 6].max() > 20 * np.median(features[features > 0])


def test_tic_normalisation_scales_by_total_ion_current():
    mz, inten = synthetic_spectrum()
    plain = PreprocessingConfig(normalization="none", dtype="float64")
    normalised = PreprocessingConfig(dtype="float64")
    raw_features, _ = preprocess_arrays(mz, inten, plain)
    tic_features, info = preprocess_arrays(mz, inten, normalised)
    assert np.allclose(tic_features * info.total_ion_current, raw_features, rtol=1e-10)


def test_scaling_the_raw_intensities_does_not_change_tic_normalised_features():
    mz, inten = synthetic_spectrum()
    a, _ = preprocess_arrays(mz, inten, PreprocessingConfig(dtype="float64"))
    b, _ = preprocess_arrays(mz, inten * 4.0, PreprocessingConfig(dtype="float64"))
    assert np.allclose(a, b, rtol=1e-9, atol=1e-15)   # sqrt(4x) = 2 sqrt(x); every later step is linear


def test_float64_option():
    mz, inten = synthetic_spectrum()
    features, _ = preprocess_arrays(mz, inten, PreprocessingConfig(dtype="float64"))
    assert features.dtype == np.float64


# ---------------------------------------------------------------- invalid input

def test_empty_spectrum():
    with pytest.raises(SpectrumFormatError, match="empty"):
        validate_spectrum(np.array([]), np.array([]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_are_rejected(bad):
    mz, inten = synthetic_spectrum()
    inten[100] = bad
    with pytest.raises(SpectrumFormatError, match="NaN or infinite"):
        preprocess_arrays(mz, inten, CFG)
    mz2, inten2 = synthetic_spectrum()
    mz2[5] = bad
    with pytest.raises(SpectrumFormatError):
        preprocess_arrays(mz2, inten2, CFG)


def test_unsorted_mz_is_rejected():
    mz, inten = synthetic_spectrum()
    mz[[10, 11]] = mz[[11, 10]]
    with pytest.raises(SpectrumFormatError, match="not sorted"):
        preprocess_arrays(mz, inten, CFG)


def test_duplicate_mz_is_rejected():
    mz, inten = synthetic_spectrum()
    mz[11] = mz[10]
    with pytest.raises(SpectrumFormatError, match="duplicate m/z"):
        preprocess_arrays(mz, inten, CFG)


def test_negative_intensity_is_rejected():
    mz, inten = synthetic_spectrum()
    inten[3] = -1
    with pytest.raises(SpectrumFormatError, match="negative"):
        preprocess_arrays(mz, inten, CFG)


def test_too_few_points_and_shape_mismatch():
    with pytest.raises(SpectrumFormatError, match="only 50 points"):
        preprocess_arrays(np.linspace(2000, 3000, 50), np.ones(50), CFG)
    with pytest.raises(SpectrumFormatError, match="same length"):
        validate_spectrum(np.arange(1, 200.0), np.ones(198))


def test_spectrum_outside_configured_range():
    mz, inten = synthetic_spectrum(lo=21000, hi=30000)
    with pytest.raises(PreprocessingError, match="No m/z values inside"):
        preprocess_arrays(mz, inten, CFG)


def test_spectrum_partly_outside_range_only_fills_matching_bins():
    mz, inten = synthetic_spectrum(n=2000, lo=1500, hi=2600)
    features, info = preprocess_arrays(mz, inten, CFG)
    assert features.shape == (6000,)
    assert np.count_nonzero(features[200:]) == 0            # nothing above 2600 Da
    assert info.points_in_range == int(((mz >= 2000) & (mz <= 20000)).sum())


def test_all_zero_spectrum_is_unusable():
    mz, _ = synthetic_spectrum()
    with pytest.raises(PreprocessingError, match="no signal"):
        preprocess_arrays(mz, np.zeros_like(mz), CFG)


def test_malformed_file(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("# header\nmass intensity\n2000 10\n2001 not-a-number\n", encoding="utf-8")
    with pytest.raises(SpectrumFormatError):
        preprocess_file(path, CFG)
    with pytest.raises(SpectrumFormatError, match="not found"):
        preprocess_file(tmp_path / "missing.txt", CFG)
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="empty"):
        preprocess_file(empty, CFG)


def test_file_and_array_paths_agree(tmp_path):
    mz, inten = synthetic_spectrum()
    path = write_spectrum(tmp_path / "s.txt", mz, inten)
    from_file, _ = preprocess_file(path, CFG)
    from_arrays, _ = preprocess_arrays(mz, inten, CFG)
    assert np.array_equal(from_file, from_arrays)


# ---------------------------------------------------------------- individual steps

def test_savitzky_golay_coefficients_known_values():
    coef = savitzky_golay_coefficients(2, 3)
    assert np.allclose(coef[2], np.array([-3, 12, 17, 12, -3]) / 35)
    assert np.allclose(coef.sum(axis=1), 1.0)
    assert np.allclose(coef[3:], coef[:2][::-1, ::-1])


def test_savitzky_golay_keeps_cubic_polynomials_including_edges():
    x = np.linspace(-3, 3, 200)
    y = 0.5 * x ** 3 - 2 * x ** 2 + x + 7
    assert np.allclose(savitzky_golay(y, 10, 3), y, atol=1e-8)


def test_savitzky_golay_rejects_too_short_input():
    with pytest.raises(PreprocessingError, match="smoothing window"):
        savitzky_golay(np.ones(15), 10, 3)


def test_snip_baseline():
    x = np.arange(500, dtype=float)
    line = 0.3 * x + 20
    assert np.allclose(snip_baseline(line, 20), line)
    peak = 500 * np.exp(-0.5 * ((x - 250) / 3) ** 2)
    baseline = snip_baseline(line + peak, 20)
    assert (baseline <= line + peak + 1e-12).all()
    assert abs(baseline[250] - line[250]) < 1.0
    assert np.allclose(snip_baseline(np.ones(10), 100), 1.0)   # window larger than data: unchanged


def test_total_ion_current_is_trapezoidal_area():
    mz = np.array([0.0, 1.0, 3.0])
    assert total_ion_current(mz, np.array([2.0, 4.0, 0.0])) == pytest.approx(3.0 + 4.0)


def test_binning_edges():
    edges = PreprocessingConfig().bin_edges
    mz = np.array([2000.0, 2002.999, 2003.0, 19999.0, 20000.0])
    counts = bin_intensities(mz, np.array([1.0, 2.0, 4.0, 8.0, 16.0]), edges)
    assert counts[0] == 3.0 and counts[1] == 4.0 and counts[-1] == 24.0
    assert counts.sum() == 31.0


# ---------------------------------------------------------------- configuration

def test_config_from_project_file():
    cfg = PreprocessingConfig.from_config(load_config())
    assert cfg.n_bins == 6000 and cfg.dtype == "float32"
    assert cfg.baseline_iterations == (20, 100)
    assert cfg.bin_edges[0] == 2000.0 and cfg.bin_edges[-1] == 20000.0
    assert cfg == PreprocessingConfig()


def test_config_validation_and_fingerprint():
    with pytest.raises(ValueError, match="whole number"):
        PreprocessingConfig(bin_width=7.0)
    with pytest.raises(ValueError):
        PreprocessingConfig(normalization="zscore")
    with pytest.raises(ValueError):
        PreprocessingConfig(mz_min=5000, mz_max=4000)
    assert PreprocessingConfig().fingerprint() == PreprocessingConfig().fingerprint()
    assert PreprocessingConfig(bin_width=6.0).fingerprint() != PreprocessingConfig().fingerprint()
    assert PreprocessingConfig(bin_width=6.0).n_bins == 3000


# ---------------------------------------------------------------- real data (skipped when absent, e.g. in CI)

def _real_driams_samples(n: int = 3):
    root = Path(load_config()["paths"]["driams_root"]) / "DRIAMS-B"
    id_file = root / "id" / "2018" / "2018_clean.csv"
    if not id_file.is_file() or not (root / "raw" / "2018").is_dir():
        return []
    ids = pd.read_csv(id_file, dtype=str, usecols=["code", "species"])
    codes = ids.loc[ids["species"] == "Escherichia coli", "code"]
    found = [c for c in codes if (root / "raw" / "2018" / f"{c}.txt").is_file()
             and (root / "binned_6000" / "2018" / f"{c}.txt").is_file()]
    return [(root / "raw" / "2018" / f"{c}.txt", root / "binned_6000" / "2018" / f"{c}.txt") for c in found[:n]]


@pytest.mark.skipif(not _real_driams_samples(), reason="DRIAMS-B raw spectra not available on this machine")
def test_matches_published_driams_binned_spectra():
    cfg = PreprocessingConfig(dtype="float64")
    for raw_path, binned_path in _real_driams_samples():
        ours, _ = preprocess_file(raw_path, cfg)
        reference = read_binned_spectrum(binned_path)
        assert np.abs(ours - reference).max() / reference.max() < 1e-9
