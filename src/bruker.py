"""Read Bruker flex raw spectra (an `acqu`/`acqus` file and a `fid` file) as readBrukerFlexData 1.9.3 does.

DRIAMS converted its spectra in R with MALDIquantForeign, which calls readBrukerFlexData's `readBrukerFlexFile` with
its default arguments (DRIAMS raw text files carry the R column names "mass.myspec..1..."). The frozen models were
trained on features from those conversions. This reader therefore reproduces that function's defaults: useHpc=TRUE,
keepNegativeIntensities=FALSE, filterZeroIntensities=FALSE. Source: CRAN readBrukerFlexData 1.9.3 (GPL >= 3),
R/readBrukerFlexFile-functions.R, R/readAcquFile-functions.R, R/tof2mass-functions.R, R/hpc-functions.R.

- **Metadata** comes from `acqu`, or from `acqus` if there is no `acqu`. Values are taken as the reference does: the
  text after "= ", without angle brackets, with a decimal comma read as a point.
- **Intensities** are 32-bit signed integers, big-endian if ##$BYTORDA is 1, otherwise little-endian. TD of them are
  read, or fewer if the file is shorter, and negative values are set to 0.
- **Time of flight** is DELAY + i * DW, for i = 0 ... n - 1.
- **Mass** uses the quadratic calibration with ML1, ML2 and ML3 (`.tof2mass`; Titulaer et al., BMC Bioinformatics 2006,
  7:403).
- **HPC** is applied when ##$HPClUse is "yes" and HPClBLo, HPClBHi and HPClOrd are all positive. As `.hpc` does, it
  subtracts sum(c_k * m^k) from masses inside [HPClBLo, HPClBHi].

The reader refuses some spectra, each with a reason, rather than converting them by approximation:
- V1.0CTOF2 calibration constants in ##$NTBCal, which the reference itself does not fully support;
- LIFT spectra, which have no mass conversion;
- missing or invalid calibration constants.

After conversion, the checks registered for Version 2.0 apply (docs/v2.0_marisma_plan.md, Cohort and amendment A6):
a spectrum must be non-empty, finite, have a strictly increasing m/z axis, and cover 2,000-20,000 Da. Error messages
carry a reason code only, never a path or a data value.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_RANGE = (2000.0, 20000.0)   # Da; the spectrum must cover it (plan, Cohort: reader failures)
_VALUE = re.compile(r"(^.*= *<?)|(>? *$)")   # readBrukerFlexData's .grepAcquValue, applied with gsub


class BrukerReadError(Exception):
    """A spectrum the reader refuses. `reason` is one of REASONS; the message never contains data or paths."""

    def __init__(self, reason: str):
        if reason not in REASONS:
            raise ValueError(f"unknown reason {reason!r}")
        super().__init__(reason)
        self.reason = reason


REASONS = (
    "missing_files",                   # no fid, or neither acqu nor acqus
    "unreadable",                      # a required field (TD, DELAY, DW, BYTORDA) missing or not a number
    "no_calibration",                  # ML1, ML2 or ML3 missing or invalid (ML1 must be positive)
    "invalid_hpc",                     # HPC requested but its coefficients cannot be read
    "unsupported_calibration_ctof2",   # V1.0CTOF2 constants: not converted by approximation
    "lift_spectrum",                   # LIFT: no mass conversion
    "empty",                           # no points, or every intensity zero
    "non_finite",                      # a mass or intensity is not finite
    "axis_not_increasing",             # m/z is not strictly increasing
    "range_not_covered",               # m/z does not cover 2,000-20,000 Da
)


@dataclass(frozen=True)
class BrukerSpectrum:
    mz: np.ndarray            # float64, Da
    intensity: np.ndarray     # float64, counts, negatives set to 0
    calibration: str          # "tof2mass" or "tof2mass+hpc"
    acquisition_date: str     # ##$AQ_DATE, for batch summaries only ("" if absent)
    instrument: str           # ##$INSTRUM, for batch summaries only ("" if absent)


def acqu_values(lines: list[str], key: str) -> list[str]:
    """All values of `##$<key>=` lines, extracted as readBrukerFlexData's .grepAcquValue does."""
    pattern = f"##${key}="
    return [_VALUE.sub("", line) for line in lines if pattern in line]


def _double(lines: list[str], key: str) -> float | None:
    """Single numeric value of a key, or None if it is absent, repeated or not a number (as .grepAcquDoubleValue)."""
    values = acqu_values(lines, key)
    if len(values) != 1:
        return None
    try:
        return float(values[0].replace(",", "."))
    except ValueError:
        return None


def _text(lines: list[str], key: str) -> str:
    values = acqu_values(lines, key)
    return values[0] if len(values) == 1 else ""


def tof2mass(tof: np.ndarray, c1: float, c2: float, c3: float) -> np.ndarray:
    """readBrukerFlexData's .tof2mass: A = c3, B = sqrt(1e12 / c1), C = c2 - tof."""
    a, b, c = c3, math.sqrt(1e12 / c1), c2 - tof
    if a == 0:
        return (c * c) / (b * b)
    with np.errstate(invalid="ignore"):
        return ((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)) ** 2


def hpc_coefficients(hpc_str: str) -> np.ndarray:
    """readBrukerFlexData's .extractHPCConstants: the numbers between 'V1.0VectorDouble' + 2 and 'c2'."""
    tokens = [t for t in hpc_str.split(" ") if t]
    try:
        start, stop = tokens.index("V1.0VectorDouble") + 2, tokens.index("c2")
        coefficients = np.array([float(t) for t in tokens[start:stop]], dtype=np.float64)
    except (ValueError, IndexError) as exc:
        raise BrukerReadError("invalid_hpc") from exc
    if coefficients.size == 0 or not np.isfinite(coefficients).all():
        raise BrukerReadError("invalid_hpc")
    return coefficients


def apply_hpc(mass: np.ndarray, lo: float, hi: float, coefficients: np.ndarray) -> np.ndarray:
    """readBrukerFlexData's .hpc: inside [lo, hi], mass - sum(c_k * mass^k)."""
    out = mass.copy()
    inside = (mass >= lo) & (mass <= hi)
    m = mass[inside]
    powers = np.vander(m, N=coefficients.size, increasing=True)   # m^0 ... m^l
    out[inside] = m - powers @ coefficients
    return out


def read_spectrum(folder: str | Path) -> BrukerSpectrum:
    """Read the Bruker spectrum in `folder` (which holds `fid` and `acqu`/`acqus`); raise BrukerReadError(reason)."""
    folder = Path(folder)
    fid = folder / "fid"
    acqu = folder / "acqu" if (folder / "acqu").is_file() else folder / "acqus"
    if not fid.is_file() or not acqu.is_file():
        raise BrukerReadError("missing_files")
    try:
        return convert(acqu.read_text(encoding="latin-1").splitlines(), fid.read_bytes())
    except OSError as exc:
        raise BrukerReadError("unreadable") from exc


def decode(lines: list[str], fid_bytes: bytes) -> BrukerSpectrum:
    """Decode one spectrum from its `acqu` lines and `fid` bytes, as readBrukerFlexFile does with its defaults.

    The registered checks on the result (see `convert`) are not applied here, so that the reference check can validate
    the decoding of spectra those checks refuse. The pipeline always uses `convert`.
    """
    td, delay, dw, bytorda = (_double(lines, k) for k in ("TD", "DELAY", "DW", "BYTORDA"))
    if any(v is None or not math.isfinite(v) for v in (td, delay, dw, bytorda)) or td < 1 or td != int(td):
        raise BrukerReadError("unreadable")
    ntbcal = _text(lines, "NTBCal")
    if "V1.0CTOF2CalibrationConstants" in ntbcal:
        raise BrukerReadError("unsupported_calibration_ctof2")
    lift1, lift2, tlift = _double(lines, "Lift1"), _double(lines, "Lift2"), _double(lines, "TLift")
    lift_used = (lift1 is not None and lift2 is not None and lift1 != 0 and lift2 != 0) or \
        (tlift is not None and tlift != 0) or _text(lines, "SPType") == "2"
    if lift_used:
        raise BrukerReadError("lift_spectrum")
    ml1, ml2, ml3 = _double(lines, "ML1"), _double(lines, "ML2"), _double(lines, "ML3")
    if any(v is None or not math.isfinite(v) for v in (ml1, ml2, ml3)) or ml1 <= 0:
        raise BrukerReadError("no_calibration")

    available = len(fid_bytes) // 4
    raw = np.frombuffer(fid_bytes, dtype=">i4" if int(bytorda) == 1 else "<i4", count=min(int(td), available))
    n = raw.size                                      # fewer than TD if the file is shorter, as the reference allows
    if n == 0:
        raise BrukerReadError("empty")
    intensity = raw.astype(np.float64)
    intensity[intensity < 0] = 0.0
    tof = delay + np.arange(n, dtype=np.float64) * dw
    mz = tof2mass(tof, ml1, ml2, ml3)
    calibration = "tof2mass"

    hpc_lo, hpc_hi, hpc_order = _double(lines, "HPClBLo"), _double(lines, "HPClBHi"), _double(lines, "HPClOrd")
    if _text(lines, "HPClUse") == "yes" and all(v is not None and v > 0 for v in (hpc_lo, hpc_hi, hpc_order)):
        mz = apply_hpc(mz, hpc_lo, hpc_hi, hpc_coefficients(_text(lines, "HPCStr")))
        calibration = "tof2mass+hpc"
    return BrukerSpectrum(mz, intensity, calibration, _text(lines, "AQ_DATE"), _text(lines, "INSTRUM"))


def convert(lines: list[str], fid_bytes: bytes) -> BrukerSpectrum:
    """`decode`, then the checks registered for Version 2.0: non-empty, finite, strictly increasing, and covering
    2,000-20,000 Da. A spectrum failing one is refused with its reason."""
    spectrum = decode(lines, fid_bytes)
    mz, intensity = spectrum.mz, spectrum.intensity
    if not intensity.any():
        raise BrukerReadError("empty")
    if not (np.isfinite(mz).all() and np.isfinite(intensity).all()):
        raise BrukerReadError("non_finite")
    if np.any(np.diff(mz) <= 0):
        raise BrukerReadError("axis_not_increasing")
    if mz[0] > REQUIRED_RANGE[0] or mz[-1] < REQUIRED_RANGE[1]:
        raise BrukerReadError("range_not_covered")
    return spectrum
