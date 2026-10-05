"""The frozen model's distribution package: built from the trusted original, installed from a local file.

**The package** is a ZIP holding exactly the five members of `MEMBERS`. It is stored, not compressed, with fixed
timestamps and attributes, so rebuilding it from the same inputs gives the same bytes on any system.

**Trust does not come from the package.**
- The bundle's SHA-256 is pinned here, in `FROZEN_BUNDLE_SHA256`. The same value has been published in
  `docs/reproduction_guide.md` since 2026-10-01.
- The installer checks the bundle's bytes against that pin before anything is written. The package's own checksum
  file and manifest are only cross-checked: they can refuse a package, never accept one.
- **Nothing here deserialises the bundle.** A joblib file is a pickle, and loading one runs code. Only the project's
  unchanged loader (`src/predict.load_bundle`) ever loads it, after installation.

**The installer never replaces a different model.** An existing different file at the target is left untouched, and
the run is refused.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_BUNDLE_SHA256 = "d59d6d7deafa1af464c33ebefc0f8f641a80eb70f7b51039e6c2ede0fd841c8b"
FROZEN_BUNDLE_SIZE = 636_912
MODEL_VERSION = "v0.4.0-tuned_lightgbm-random-seed42"
PACKAGE_NAME = f"ecoli_ciprofloxacin-{MODEL_VERSION}"
BUNDLE_NAME = "best_random.joblib"
SIDECAR_NAME = BUNDLE_NAME + ".sha256"
TARGET = Path("models") / "v0.4" / "ecoli_ciprofloxacin" / BUNDLE_NAME     # where the unchanged CLI looks
MEMBERS = (BUNDLE_NAME, SIDECAR_NAME, "MANIFEST.json", "MODEL_CARD.md", "NOTICE.md")
ZIP_TIME = (2026, 9, 18, 12, 54, 10)          # the bundle's own creation time, so archives are reproducible

# exit codes of scripts/install_model.py
VERIFICATION_FAILED, REFUSED_OVERWRITE = 2, 3


class PackageError(RuntimeError):
    """The package cannot be used, or installing it would replace something else. Nothing was written."""

    def __init__(self, message: str, code: int = VERIFICATION_FAILED):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Installed:
    target: Path
    changed: bool
    message: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sidecar_text(digest: str | None = None) -> str:
    """The sidecar line, in the format `src/predict.verify_digest` reads (`sha256sum` style); the pin by default."""
    return f"{digest or FROZEN_BUNDLE_SHA256}  {BUNDLE_NAME}\n"


# --- reading and checking a package (never deserialising it) ---------------------------------------------------------

def read_package(path: str | Path) -> dict[str, bytes]:
    """The members of a package, from its ZIP file or an extracted folder. A local path only: nothing is downloaded."""
    text = str(path)
    if "://" in text:
        raise PackageError(f"{text!r} is not a local path. The installer reads a package you already have; it never "
                           "downloads one.")
    path = Path(path)
    if not path.exists():
        raise PackageError(f"No package at {path}.")
    if path.is_dir():
        members = {name: (path / name).read_bytes() for name in MEMBERS if (path / name).is_file()}
        unknown = sorted(p.name for p in path.iterdir() if p.name not in MEMBERS)
    else:
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                unknown = sorted(n for n in names if n not in MEMBERS)
                members = {name: archive.read(name) for name in MEMBERS if name in names}
        except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError) as exc:
            raise PackageError(f"{path.name} is damaged or not a ZIP package ({exc}).") from None
    if unknown:
        raise PackageError(f"{path.name} holds unexpected members {unknown}; a package holds exactly {list(MEMBERS)}.")
    if BUNDLE_NAME not in members:
        raise PackageError(f"{path.name} holds no {BUNDLE_NAME}.")
    return members


def check(members: dict[str, bytes]) -> str:
    """Verify the bundle against the pinned hash; the package's own files may only refuse it. Returns the digest."""
    data = members[BUNDLE_NAME]
    digest = sha256_bytes(data)
    if digest != FROZEN_BUNDLE_SHA256 or len(data) != FROZEN_BUNDLE_SIZE:
        raise PackageError(f"Refused: the bundle's SHA-256 is {digest} ({len(data):,} bytes), not the pinned "
                           f"{FROZEN_BUNDLE_SHA256} ({FROZEN_BUNDLE_SIZE:,} bytes). Nothing was installed.")
    if SIDECAR_NAME in members:
        listed = members[SIDECAR_NAME].decode("utf-8", errors="replace").split()
        if not listed or listed[0].lower() != FROZEN_BUNDLE_SHA256:
            raise PackageError("Refused: the package's own checksum file disagrees with its bundle and the pin; the "
                               "package is inconsistent. Nothing was installed.")
    if "MANIFEST.json" in members:
        try:
            manifest = json.loads(members["MANIFEST.json"])
            listed = manifest["bundle"]["sha256"]
        except (ValueError, KeyError, TypeError):
            raise PackageError("Refused: the package's MANIFEST.json is unreadable. Nothing was installed.") from None
        if listed != FROZEN_BUNDLE_SHA256:
            raise PackageError("Refused: the package's MANIFEST.json names another bundle. Nothing was installed.")
    return digest


# --- installing ----------------------------------------------------------------------------------------------------

def _file_digest(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _sidecar_digest(path: Path) -> str | None:
    words = path.read_text(encoding="utf-8", errors="replace").split()
    return words[0].lower() if words else None


def _place(data: bytes, final: Path) -> None:
    """Write `data` next to `final`, check it on disk, then link it into place, which fails if `final` exists."""
    temporary = final.with_name(f".installing-{os.getpid()}-{final.name}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if _file_digest(temporary) != sha256_bytes(data):
            raise PackageError(f"The copy written to {temporary.parent} does not match what was verified; nothing "
                               "was installed.")
        try:
            os.link(temporary, final)                      # atomic, and never replaces an existing file
        except FileExistsError:
            raise PackageError(f"{final} appeared while installing; it was left untouched.",
                               REFUSED_OVERWRITE) from None
        except OSError:                                    # a file system without hard links
            with final.open("xb") as handle:
                handle.write(data)
    finally:
        temporary.unlink(missing_ok=True)


def install(package: str | Path, *, root: Path = PROJECT_ROOT, target: Path | None = None) -> Installed:
    """Install the frozen bundle from a local package, verified against the pin before anything is written."""
    members = read_package(package)
    check(members)
    final = Path(target) if target is not None else root / TARGET
    sidecar = final.with_name(final.name + ".sha256")
    if sidecar.exists() and _sidecar_digest(sidecar) != FROZEN_BUNDLE_SHA256:
        raise PackageError(f"A different checksum file is already at {sidecar}; it and the model were left "
                           "untouched. Move them aside yourself if you mean to replace them.", REFUSED_OVERWRITE)
    if final.exists():
        if _file_digest(final) != FROZEN_BUNDLE_SHA256:
            raise PackageError(f"A different file is already at {final}; it was left untouched. Move it aside "
                               "yourself if you mean to replace it.", REFUSED_OVERWRITE)
        if not sidecar.exists():
            _place(sidecar_text().encode("utf-8"), sidecar)
            return Installed(final, True, f"The frozen model was already at {final}; its checksum file was added.")
        return Installed(final, False, f"The frozen model is already installed at {final}; nothing changed.")
    final.parent.mkdir(parents=True, exist_ok=True)
    _place(members[BUNDLE_NAME], final)
    if not sidecar.exists():
        _place(sidecar_text().encode("utf-8"), sidecar)
    return Installed(final, True, f"Installed the frozen model at {final} (SHA-256 {FROZEN_BUNDLE_SHA256}, matching "
                                  "the value pinned in this repository).")


# --- building (on the machine that holds the trusted original) -----------------------------------------------------

def zip_members(members: dict[str, bytes]) -> bytes:
    """A reproducible ZIP: stored, in `MEMBERS` order, with fixed timestamps, system and permissions."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in MEMBERS:
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, members[name])
    return buffer.getvalue()


def build(bundle: bytes, manifest: dict, model_card: str, notice: str) -> bytes:
    """The package's bytes. The bundle must be the frozen original, byte for byte."""
    if sha256_bytes(bundle) != FROZEN_BUNDLE_SHA256 or len(bundle) != FROZEN_BUNDLE_SIZE:
        raise PackageError("The source is not the frozen bundle: its SHA-256 or size differs from the pin.")
    if manifest.get("bundle", {}).get("sha256") != FROZEN_BUNDLE_SHA256:
        raise PackageError("The manifest must name the frozen bundle's SHA-256.")
    members = {BUNDLE_NAME: bundle, SIDECAR_NAME: sidecar_text().encode("utf-8"),
               "MANIFEST.json": (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
               "MODEL_CARD.md": model_card.encode("utf-8"), "NOTICE.md": notice.encode("utf-8")}
    return zip_members(members)
