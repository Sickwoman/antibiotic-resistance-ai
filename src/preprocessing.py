"""Version 0.2 spectrum preprocessing: raw MALDI-TOF spectrum -> fixed-length feature vector.

The steps reproduce the DRIAMS pipeline (Weis et al. 2022; R scripts in BorgwardtLab/maldi_amr,
run with MALDIquant). The algorithms below are ports of MALDIquant 1.22.3:

1. square-root intensity transform (variance stabilisation)
2. Savitzky-Golay smoothing, half window 10, cubic polynomial, MALDIquant edge handling
3. SNIP baseline removal, applied twice (20 iterations, then the MALDIquant default of 100)
4. TIC normalisation: divide by the trapezoidal area of the whole spectrum
5. trim to [mz_min, mz_max] (both ends included)
6. binning: sum of intensities in bins of bin_width Da; the last bin includes mz_max

Negative values produced by steps 1-2 are set to zero, as MALDIquant does.

Every step uses only the spectrum itself (no parameters learned from other samples), so applying
it before a train/test split cannot leak information. Learned transformations (scaling, PCA,
feature selection) are intentionally not part of this module; they belong in model pipelines that
are fitted on training data only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.data_loader import SpectrumFormatError, read_raw_spectrum

TRANSFORMS = ("sqrt", "none")
SMOOTHING = ("savitzky_golay", "none")
BASELINES = ("snip", "none")
NORMALIZATIONS = ("tic", "none")


class PreprocessingError(SpectrumFormatError):
    """A spectrum that was read correctly but cannot be turned into a usable feature vector."""


@dataclass(frozen=True)
class PreprocessingConfig:
    intensity_transform: str = "sqrt"
    smoothing_method: str = "savitzky_golay"
    half_window_size: int = 10
    polynomial_order: int = 3
    baseline_method: str = "snip"
    baseline_iterations: tuple[int, ...] = (20, 100)
    normalization: str = "tic"
    mz_min: float = 2000.0
    mz_max: float = 20000.0
    bin_width: float = 3.0
    dtype: str = "float32"
    min_raw_points: int = 100
    spectrum_folder: str = field(default="raw", compare=False)

    def __post_init__(self) -> None:
        checks = [
            (self.intensity_transform in TRANSFORMS, f"intensity_transform must be one of {TRANSFORMS}"),
            (self.smoothing_method in SMOOTHING, f"smoothing method must be one of {SMOOTHING}"),
            (self.baseline_method in BASELINES, f"baseline method must be one of {BASELINES}"),
            (self.normalization in NORMALIZATIONS, f"normalization must be one of {NORMALIZATIONS}"),
            (self.half_window_size >= 1, "half_window_size must be >= 1"),
            (2 * self.half_window_size + 1 > self.polynomial_order, "window must be larger than the polynomial order"),
            (all(i >= 1 for i in self.baseline_iterations), "baseline iterations must be >= 1"),
            (self.mz_max > self.mz_min > 0, "need 0 < mz_min < mz_max"),
            (self.bin_width > 0, "bin_width must be positive"),
            (self.dtype in ("float32", "float64"), "dtype must be float32 or float64"),
            (self.min_raw_points >= 2, "min_raw_points must be >= 2"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
        n = (self.mz_max - self.mz_min) / self.bin_width
        if abs(n - round(n)) > 1e-9:
            raise ValueError(f"(mz_max - mz_min) / bin_width = {n} is not a whole number of bins")

    @property
    def n_bins(self) -> int:
        return int(round((self.mz_max - self.mz_min) / self.bin_width))

    @property
    def bin_edges(self) -> np.ndarray:
        return np.linspace(self.mz_min, self.mz_max, self.n_bins + 1)

    @property
    def bin_centers(self) -> np.ndarray:
        edges = self.bin_edges
        return (edges[:-1] + edges[1:]) / 2

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PreprocessingConfig:
        p = config["preprocessing"]
        return cls(
            intensity_transform=p["intensity_transform"],
            smoothing_method=p["smoothing"]["method"],
            half_window_size=int(p["smoothing"]["half_window_size"]),
            polynomial_order=int(p["smoothing"]["polynomial_order"]),
            baseline_method=p["baseline"]["method"],
            baseline_iterations=tuple(int(i) for i in p["baseline"]["iterations"]),
            normalization=p["normalization"],
            mz_min=float(p["mz_min"]),
            mz_max=float(p["mz_max"]),
            bin_width=float(p["bin_width"]),
            dtype=p["dtype"],
            min_raw_points=int(p["min_raw_points"]),
            spectrum_folder=p["spectrum_folder"],
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["baseline_iterations"] = list(self.baseline_iterations)
        d["n_bins"] = self.n_bins
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessingConfig:
        """Inverse of to_dict (e.g. the settings stored inside a saved model)."""
        values = {k: v for k, v in d.items() if k != "n_bins"}
        values["baseline_iterations"] = tuple(int(i) for i in values["baseline_iterations"])
        return cls(**values)

    @property
    def matches_driams_binning(self) -> bool:
        """True when the settings are the DRIAMS ones, so results are comparable with binned_6000 files."""
        reference = PreprocessingConfig()
        fields = ("intensity_transform", "smoothing_method", "half_window_size", "polynomial_order",
                  "baseline_method", "baseline_iterations", "normalization", "mz_min", "mz_max", "bin_width")
        return all(getattr(self, f) == getattr(reference, f) for f in fields)

    def fingerprint(self) -> str:
        """Short hash identifying the feature definition (changes whenever a parameter changes)."""
        payload = {k: v for k, v in self.to_dict().items() if k != "spectrum_folder"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------------

def validate_spectrum(mz: np.ndarray, intensity: np.ndarray, min_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Check an in-memory spectrum; return float64 copies. Raises SpectrumFormatError."""
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.ndim != 1 or intensity.ndim != 1 or mz.shape != intensity.shape:
        raise SpectrumFormatError("m/z and intensity must be 1-D arrays of the same length.")
    if mz.size == 0:
        raise SpectrumFormatError("Spectrum is empty.")
    if mz.size < min_points:
        raise SpectrumFormatError(f"Spectrum has only {mz.size} points; at least {min_points} are required.")
    if not (np.isfinite(mz).all() and np.isfinite(intensity).all()):
        raise SpectrumFormatError("Spectrum contains NaN or infinite values.")
    steps = np.diff(mz)
    if np.any(steps == 0):
        raise SpectrumFormatError("Spectrum contains duplicate m/z values.")
    if np.any(steps < 0):
        raise SpectrumFormatError("m/z values are not sorted in increasing order.")
    if mz[0] <= 0:
        raise SpectrumFormatError("m/z values must be positive.")
    if np.any(intensity < 0):
        raise SpectrumFormatError("Raw intensities must not be negative.")
    return mz, intensity


# ------------------------------------------------------------------------------------------------
# MALDIquant-equivalent steps
# ------------------------------------------------------------------------------------------------

@lru_cache(maxsize=16)
def savitzky_golay_coefficients(half_window_size: int, polynomial_order: int) -> np.ndarray:
    """Coefficient matrix as in MALDIquant's .savitzkyGolayCoefficients.

    Row m (the middle) is the usual symmetric filter. Rows 0..m-1 estimate the first m points from
    the first window; the last m rows are the same, mirrored, for the right edge.
    """
    m, k = half_window_size, polynomial_order
    nm = 2 * m + 1
    powers = np.arange(k + 1)
    coef = np.empty((nm, nm))
    for i in range(m + 1):
        positions = (np.arange(nm) - i).astype(np.float64)
        design = positions[:, None] ** powers[None, :]
        coef[i] = np.linalg.solve(design.T @ design, design.T)[0]
    coef[m + 1:] = coef[:m][::-1, ::-1]
    coef.setflags(write=False)
    return coef


def savitzky_golay(y: np.ndarray, half_window_size: int = 10, polynomial_order: int = 3) -> np.ndarray:
    m = half_window_size
    w = 2 * m + 1
    if y.size < w:
        raise PreprocessingError(f"Spectrum has {y.size} points, fewer than the smoothing window ({w}).")
    coef = savitzky_golay_coefficients(m, polynomial_order)
    out = np.empty_like(y, dtype=np.float64)
    out[m:y.size - m] = np.correlate(y, coef[m], mode="valid")
    out[:m] = coef[:m] @ y[:w]
    out[y.size - m:] = coef[m + 1:] @ y[-w:]
    return out


def snip_baseline(y: np.ndarray, iterations: int) -> np.ndarray:
    """SNIP baseline with a decreasing clipping window (MALDIquant C_snip, decreasing=TRUE)."""
    baseline = np.array(y, dtype=np.float64, copy=True)
    n = baseline.size
    for i in range(iterations, 0, -1):
        if 2 * i >= n:
            continue
        mean = 0.5 * (baseline[: n - 2 * i] + baseline[2 * i:])
        np.minimum(baseline[i: n - i], mean, out=baseline[i: n - i])
    return baseline


def total_ion_current(mz: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal area under the spectrum (MALDIquant totalIonCurrent)."""
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(mz)))


def trim(mz: np.ndarray, y: np.ndarray, mz_min: float, mz_max: float) -> tuple[np.ndarray, np.ndarray]:
    keep = (mz >= mz_min) & (mz <= mz_max)
    return mz[keep], y[keep]


def bin_intensities(mz: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Sum of intensities per bin; bins are [a, b) except the last, which is [a, b]."""
    counts, _ = np.histogram(mz, bins=edges, weights=y)
    return counts


@dataclass(frozen=True)
class SpectrumInfo:
    raw_points: int
    points_in_range: int
    nonzero_bins: int
    total_ion_current: float


def preprocess_arrays(mz: np.ndarray, intensity: np.ndarray,
                      cfg: PreprocessingConfig) -> tuple[np.ndarray, SpectrumInfo]:
    """Raw (m/z, intensity) arrays -> feature vector of length cfg.n_bins in cfg.dtype."""
    mz, y = validate_spectrum(mz, intensity, cfg.min_raw_points)
    raw_points = mz.size

    if cfg.intensity_transform == "sqrt":
        y = np.sqrt(y)
    if cfg.smoothing_method == "savitzky_golay":
        y = savitzky_golay(y, cfg.half_window_size, cfg.polynomial_order)
        np.maximum(y, 0.0, out=y)
    if cfg.baseline_method == "snip":
        for iterations in cfg.baseline_iterations:
            y = y - snip_baseline(y, iterations)

    tic = float("nan")
    if cfg.normalization == "tic":
        tic = total_ion_current(mz, y)
        if not np.isfinite(tic) or tic <= 0:
            raise PreprocessingError("Total ion current is zero or invalid; the spectrum has no signal.")
        y = y / tic

    mz, y = trim(mz, y, cfg.mz_min, cfg.mz_max)
    if mz.size == 0:
        raise PreprocessingError(f"No m/z values inside the configured range [{cfg.mz_min}, {cfg.mz_max}].")
    features = bin_intensities(mz, y, cfg.bin_edges).astype(cfg.dtype)
    nonzero = int(np.count_nonzero(features))
    if nonzero == 0:
        raise PreprocessingError("All features are zero after preprocessing.")
    if not np.isfinite(features).all():
        raise PreprocessingError("Preprocessing produced non-finite values.")
    return features, SpectrumInfo(raw_points, int(mz.size), nonzero, tic)


def preprocess_file(path: str | Path, cfg: PreprocessingConfig, *, max_points: int | None = None,
                    max_bytes: int | None = None) -> tuple[np.ndarray, SpectrumInfo]:
    """Read a raw DRIAMS spectrum file and preprocess it.

    `max_points` and `max_bytes` are optional input limits for untrusted input (the Version 0.9 API).
    They are parameters rather than PreprocessingConfig fields on purpose: that dataclass's hash is the
    feature fingerprint every saved bundle is checked against, so a new field there would change
    347cbd6d5d956ff9 and invalidate every saved bundle and every cache. Both default to None, which is
    the behaviour every caller before Version 0.9 relies on.
    """
    values = read_raw_spectrum(path, min_points=cfg.min_raw_points, max_points=max_points,
                               max_bytes=max_bytes)
    return preprocess_arrays(values[:, 0], values[:, 1], cfg)
