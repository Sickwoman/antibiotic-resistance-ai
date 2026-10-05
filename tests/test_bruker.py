"""The Bruker flex reader (src/bruker.py), on synthetic spectra written by these tests: no research data is read.

The central check is independent of how the conversion is implemented: every converted mass must satisfy the
calibration's inverse, tof = ML2 + ML3*m + sqrt(1e12/ML1)*sqrt(m) (readBrukerFlexData's .tof2mass, solved for tof).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.bruker import BrukerReadError, acqu_values, apply_hpc, hpc_coefficients, read_spectrum

ML1, ML2, ML3, DELAY, DW, TD = 3.19e6, 1000.0, 0.0025, 25000.0, 2.0, 28750   # covers about 1,830-21,140 Da


def write_spectrum(folder, *, td=TD, delay=DELAY, dw=DW, ml=(ML1, ML2, ML3), bytorda=0, intensities=None,
                   acqu_name="acqu", extra=(), omit=()):
    folder.mkdir(parents=True, exist_ok=True)
    fields = {"TD": td, "DELAY": delay, "DW": dw, "ML1": ml[0], "ML2": ml[1], "ML3": ml[2], "BYTORDA": bytorda}
    lines = ["##TITLE= synthetic", "##JCAMPDX= 5.0"]
    lines += [f"##${k}= {v}" for k, v in fields.items() if k not in omit]
    lines += list(extra)
    (folder / acqu_name).write_text("\n".join(lines) + "\n", encoding="latin-1")
    if intensities is None:
        intensities = 100 + (np.arange(td) % 50)
    np.asarray(intensities, dtype=">i4" if bytorda == 1 else "<i4").tofile(folder / "fid")
    return folder


def test_masses_satisfy_the_inverse_calibration(tmp_path):
    s = read_spectrum(write_spectrum(tmp_path / "s"))
    tof = DELAY + np.arange(TD) * DW
    implied = ML2 + ML3 * s.mz + math.sqrt(1e12 / ML1) * np.sqrt(s.mz)
    assert np.allclose(implied, tof, rtol=1e-12, atol=1e-6)
    assert s.calibration == "tof2mass"
    assert s.mz[0] < 2000 < 20000 < s.mz[-1]


def test_the_linear_branch_when_ml3_is_zero(tmp_path):
    s = read_spectrum(write_spectrum(tmp_path / "s", ml=(ML1, ML2, 0.0), td=34000))
    tof = DELAY + np.arange(34000) * DW
    assert np.allclose(ML2 + math.sqrt(1e12 / ML1) * np.sqrt(s.mz), tof, rtol=1e-12)


def test_byte_order_follows_bytorda(tmp_path):
    values = (np.arange(TD) % 997) - 100          # includes negative values
    little = read_spectrum(write_spectrum(tmp_path / "le", bytorda=0, intensities=values))
    big = read_spectrum(write_spectrum(tmp_path / "be", bytorda=1, intensities=values))
    assert np.array_equal(little.intensity, big.intensity)
    assert np.array_equal(little.intensity, np.clip(values, 0, None).astype(float))   # negatives set to 0


def test_acqu_is_preferred_and_acqus_is_the_fallback(tmp_path):
    folder = write_spectrum(tmp_path / "s", acqu_name="acqus")
    only_acqus = read_spectrum(folder)
    write_spectrum(folder, acqu_name="acqu", ml=(ML1 * 1.01, ML2, ML3))   # a different calibration in acqu
    both = read_spectrum(folder)
    assert not np.allclose(only_acqus.mz, both.mz)
    assert np.allclose(ML2 + ML3 * both.mz + math.sqrt(1e12 / (ML1 * 1.01)) * np.sqrt(both.mz),
                       DELAY + np.arange(TD) * DW, rtol=1e-12)


def test_a_short_fid_is_truncated_as_in_the_reference(tmp_path):
    s = read_spectrum(write_spectrum(tmp_path / "s", td=TD + 500, intensities=100 + np.zeros(TD)))
    assert s.mz.size == TD


def test_values_are_parsed_as_the_reference_does():
    lines = ["##$DW= 2,5", "##$NTBCal= <abc def>", "##$PATH= <x=y>"]
    assert acqu_values(lines, "DW") == ["2,5"]
    assert acqu_values(lines, "NTBCal") == ["abc def"]
    assert acqu_values(lines, "PATH") == ["y"]          # greedy up to the last '=', as R's gsub


def test_a_decimal_comma_is_read_as_a_point(tmp_path):
    s = read_spectrum(write_spectrum(tmp_path / "s", extra=("##$DW= 2,0",), omit=("DW",)))
    assert np.allclose(np.diff(DELAY + np.arange(TD) * 2.0), 2.0)
    assert s.mz.size == TD


def hpc_lines(coefficients: str):
    return ("##$HPClUse= yes", "##$HPClBLo= 5000", "##$HPClBHi= 15000", "##$HPClOrd= 2",
            f"##$HPCStr= <V1.0VectorDouble 3 {coefficients} c2 0.0>")


def test_hpc_is_applied_only_inside_its_limits(tmp_path):
    plain = read_spectrum(write_spectrum(tmp_path / "a"))
    hpc = read_spectrum(write_spectrum(tmp_path / "b", extra=hpc_lines("0.05 1e-6 0.0")))   # a sub-Da correction
    inside = (plain.mz >= 5000) & (plain.mz <= 15000)
    assert hpc.calibration == "tof2mass+hpc"
    assert np.allclose(hpc.mz[inside], plain.mz[inside] - (0.05 + 1e-6 * plain.mz[inside]))
    assert np.array_equal(hpc.mz[~inside], plain.mz[~inside])


def test_an_hpc_correction_that_breaks_the_axis_is_refused(tmp_path):
    # The reference subtracts the correction only inside its window, so a correction larger than the point spacing
    # makes the axis step backwards at the window's edge. The reader's check refuses such a spectrum.
    with pytest.raises(BrukerReadError) as e:
        read_spectrum(write_spectrum(tmp_path / "s", extra=hpc_lines("0.5 0.001 0.0")))
    assert e.value.reason == "axis_not_increasing"


def test_hpc_is_ignored_unless_all_its_conditions_hold(tmp_path):
    extra = ("##$HPClUse= no",) + hpc_lines("0.05 1e-6 0.0")[1:]
    assert read_spectrum(write_spectrum(tmp_path / "s", extra=extra)).calibration == "tof2mass"


def test_hpc_helpers():
    assert np.allclose(hpc_coefficients("x V1.0VectorDouble 2 1.5 -2e-3 c2 7"), [1.5, -2e-3])
    with pytest.raises(BrukerReadError) as e:
        hpc_coefficients("no coefficients here")
    assert e.value.reason == "invalid_hpc"
    m = np.array([100.0, 200.0, 300.0])
    assert np.allclose(apply_hpc(m, 150, 250, np.array([1.0, 0.01])), [100.0, 200.0 - 3.0, 300.0])


@pytest.mark.parametrize(("kwargs", "reason"), [
    ({"extra": ("##$NTBCal= <V1.0CTOF2CalibrationConstants 48 0.25 3884 1164156 6.16 -0.06 -34.3 2 "
                "V1.0CTOF2CalibrationConstants>",)}, "unsupported_calibration_ctof2"),
    ({"extra": ("##$SPType= 2",)}, "lift_spectrum"),
    ({"extra": ("##$Lift1= 1.5", "##$Lift2= 2.5")}, "lift_spectrum"),
    ({"extra": ("##$TLift= 3",)}, "lift_spectrum"),
    ({"omit": ("ML1",)}, "no_calibration"),
    ({"ml": (-1.0, ML2, ML3)}, "no_calibration"),
    ({"omit": ("TD",)}, "unreadable"),
    ({"omit": ("BYTORDA",)}, "unreadable"),
    ({"intensities": np.zeros(TD)}, "empty"),
    ({"td": 2000}, "range_not_covered"),
    ({"dw": -2.0, "delay": 82500.0}, "axis_not_increasing"),
])
def test_refusals_carry_a_reason_and_no_data(tmp_path, kwargs, reason):
    with pytest.raises(BrukerReadError) as e:
        read_spectrum(write_spectrum(tmp_path / "s", **kwargs))
    assert e.value.reason == reason
    assert str(e.value) == reason            # the message is the reason code only: no path, no value


def test_a_bruker_spectrum_gives_the_frozen_feature_layout(tmp_path):
    from src.preprocessing import PreprocessingConfig, preprocess_arrays
    cfg = PreprocessingConfig()
    assert cfg.fingerprint() == "347cbd6d5d956ff9"          # the fingerprint both frozen bundles store
    tof = DELAY + np.arange(TD) * DW
    mz = ((-math.sqrt(1e12 / ML1) + np.sqrt(1e12 / ML1 - 4 * ML3 * (ML2 - tof))) / (2 * ML3)) ** 2
    peaks = sum(5000 * np.exp(-0.5 * ((mz - c) / 4.0) ** 2) for c in (2500, 4365, 6255, 9000, 15000))
    intensities = np.round(200 + peaks).astype(int)
    s = read_spectrum(write_spectrum(tmp_path / "s", intensities=intensities))
    features, info = preprocess_arrays(s.mz, s.intensity, cfg)
    assert features.shape == (6000,) and np.isfinite(features).all()
    assert info.nonzero_bins > 0


def test_missing_files_are_refused(tmp_path):
    (tmp_path / "s").mkdir()
    with pytest.raises(BrukerReadError) as e:
        read_spectrum(tmp_path / "s")
    assert e.value.reason == "missing_files"



def test_decode_returns_what_convert_refuses_and_agrees_where_convert_accepts(tmp_path):
    from src.bruker import convert, decode
    folder = write_spectrum(tmp_path / "s")
    lines = (folder / "acqu").read_text(encoding="latin-1").splitlines()
    fid = (folder / "fid").read_bytes()
    accepted, decoded = convert(lines, fid), decode(lines, fid)
    assert np.array_equal(accepted.mz, decoded.mz) and np.array_equal(accepted.intensity, decoded.intensity)
    short = write_spectrum(tmp_path / "short", td=2000)
    lines = (short / "acqu").read_text(encoding="latin-1").splitlines()
    with pytest.raises(BrukerReadError):
        convert(lines, (short / "fid").read_bytes())
    assert decode(lines, (short / "fid").read_bytes()).mz.size == 2000     # decoded, though the check refuses it


# --- amendment D2: the amended coverage rule -------------------------------------------------------------------------

def tof_at(mz: float) -> float:
    """The calibration's inverse for the fixture constants: tof = ML2 + ML3*m + sqrt(1e12/ML1)*sqrt(m)."""
    return ML2 + ML3 * mz + math.sqrt(1e12 / ML1) * math.sqrt(mz)


def window(first: float, last: float, dw: float = 2.0) -> dict:
    delay = tof_at(first)
    return {"delay": delay, "dw": dw, "td": int((tof_at(last) - delay) / dw) + 1}


def amended(folder):
    from src.bruker import convert_amended
    from src.preprocessing import PreprocessingConfig
    lines = (folder / "acqu").read_text(encoding="latin-1").splitlines()
    return convert_amended(lines, (folder / "fid").read_bytes(), PreprocessingConfig().bin_edges)


def test_the_amended_rule_accepts_a_standard_window(tmp_path):
    w = window(1965.0, 20500.0)
    s = amended(write_spectrum(tmp_path / "s", **w, intensities=100 + np.zeros(w["td"])))
    assert s.mz[0] == pytest.approx(1965.0, abs=0.01) and s.mz[-1] >= 20000


@pytest.mark.parametrize(("first", "last", "dw", "reason"), [
    (1965.0, 20500.0, 1.0, "sampling_interval_not_2ns"),
    (1955.0, 20500.0, 2.0, "starts_below_1960_da"),
    (2004.0, 20500.0, 2.0, "feature_bin_without_data"),        # bin 0, 2,000-2,003 Da, gets no point
    (1965.0, 19995.0, 2.0, "feature_bin_without_data"),        # bin 5999, 19,997-20,000 Da, gets no point
    (1955.0, 19995.0, 1.0, "sampling_interval_not_2ns"),       # the order of the checks decides the reason
])
def test_the_amended_rule_refuses_with_its_reason(tmp_path, first, last, dw, reason):
    w = window(first, last, dw)
    with pytest.raises(BrukerReadError) as e:
        amended(write_spectrum(tmp_path / "s", **w, intensities=100 + np.zeros(w["td"])))
    assert e.value.reason == reason


def test_bin_membership_uses_the_pipeline_boundaries():
    from src.bruker import empty_feature_bins
    from src.preprocessing import PreprocessingConfig, bin_intensities, trim
    edges = PreprocessingConfig().bin_edges
    grid = np.arange(2000.0, 20000.0, 2.9)                     # every bin gets a point, and 20,000 itself is absent
    assert empty_feature_bins(grid, edges) == 0
    assert empty_feature_bins(np.r_[2003.0, grid[grid >= 2003.0]], edges) == 1   # 2,003.0 belongs to bin 1
    assert empty_feature_bins(np.r_[grid[grid < 19997.0], 19997.0], edges) == 0  # 19,997.0 is in the last bin
    assert empty_feature_bins(np.r_[grid[grid < 19997.0], 20000.0], edges) == 0  # so is 20,000.0: [a, b] is closed
    assert empty_feature_bins(np.r_[grid[grid < 19997.0], 20000.01], edges) == 1  # beyond 20,000: trimmed away
    mz = np.sort(np.random.default_rng(3).uniform(1990, 20010, size=20000))
    kept, ones = trim(mz, np.ones_like(mz), 2000.0, 20000.0)
    assert empty_feature_bins(mz, edges) == int((bin_intensities(kept, ones, edges) == 0).sum())

