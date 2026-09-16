"""Selectively extract folders from a verified DRIAMS archive in one streaming pass.

The archives are too large to unpack completely (DRIAMS-A is 86 GB compressed and raw spectra
alone are ~0.45 MB each), so only the folders we need are written to disk. At the same time a
manifest of every file in the archive is saved, so later steps can check which spectra exist
without extracting them.

Examples (from the project root, virtual environment active):
    python scripts/extract_driams.py --site B                          # id + binned_6000 (default)
    python scripts/extract_driams.py --site A --folders id
    python scripts/extract_driams.py --site B --folders raw --species "Escherichia coli"   # V0.2

Safety: only regular files whose path starts with the site folder are written; absolute paths,
'..' components, links and device files are refused.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import CODE_COL, SPECIES_COL, DataError, load_site_tables  # noqa: E402
from src.utils import (  # noqa: E402
    ConfigError,
    archive_info,
    archives_dir,
    driams_root,
    free_disk_gb,
    get_logger,
    human_bytes,
    keep_awake,
    load_config,
    manifests_dir,
)

log = get_logger("extract")


class ExtractionError(RuntimeError):
    pass


class CountingReader(io.RawIOBase):
    """Wraps a binary file and counts how many compressed bytes have been read."""

    def __init__(self, fh) -> None:
        self._fh = fh
        self.count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        n = self._fh.readinto(buffer)
        self.count += n or 0
        return n


def safe_parts(name: str, site: str) -> tuple[str, ...]:
    """Validate a tar member name and return its path components."""
    posix = PurePosixPath(name)
    if posix.is_absolute() or name.startswith(("/", "\\")):
        raise ExtractionError(f"Refusing absolute path in archive: {name!r}")
    parts = tuple(p for p in posix.parts if p not in ("", "."))
    if any(p == ".." or "\\" in p or ":" in p for p in parts):
        raise ExtractionError(f"Refusing unsafe path in archive: {name!r}")
    if not parts or parts[0] != site:
        raise ExtractionError(f"Unexpected top-level folder in archive member {name!r} (expected {site}/)")
    return parts


def allowed_codes(config: dict, site: str, species: str) -> set[str]:
    """Spectrum codes of one species, taken from the already-extracted id tables."""
    table = load_site_tables(driams_root(config), site, config["driams"]["id_suffixes"],
                             config["labels"]["missing_values"], config["driams"]["id_folder"])
    if CODE_COL not in table.columns or SPECIES_COL not in table.columns:
        raise ExtractionError(f"id tables have no '{CODE_COL}'/'{SPECIES_COL}' columns; cannot filter by species.")
    codes = set(table.loc[table[SPECIES_COL] == species, CODE_COL].dropna().astype(str))
    if not codes:
        raise ExtractionError(f"No samples of species {species!r} found in {site} id tables.")
    return codes


def extract(config: dict, site_letter: str, folders: list[str], species: str | None, overwrite: bool) -> dict:
    info = archive_info(config, site_letter)
    site = info["site"]
    archive = archives_dir(config) / info["filename"]
    if not archive.is_file():
        raise ExtractionError(f"{archive} not found. Run: python scripts/download_driams.py --site {site_letter}")
    root = driams_root(config)
    spectra_folders = set(config["driams"]["spectra_folders"])
    unknown = set(folders) - spectra_folders - {config["driams"]["id_folder"]}
    if unknown:
        raise ExtractionError(f"Unknown folder(s) {sorted(unknown)}; choose from id and {sorted(spectra_folders)}.")

    codes = None
    if species:
        codes = allowed_codes(config, site, species)
        log.info("[%s] species filter %r: %d spectrum codes", site, species, len(codes))

    min_free = float(config["extraction"]["min_free_disk_gb"])
    if free_disk_gb(root) < min_free:
        raise ExtractionError(f"Less than {min_free} GB free on the drive holding {root}.")

    man_dir = manifests_dir(config)
    man_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = man_dir / f"{site}_manifest.csv"
    manifest_tmp = manifest_path.with_suffix(".csv.tmp")

    total_size = archive.stat().st_size
    stats = {"members_seen": 0, "files_written": 0, "files_skipped_existing": 0,
             "files_filtered_out": 0, "macos_metadata_skipped": [], "bytes_written": 0, "per_folder": {}}
    started = time.monotonic()
    last_log = started

    with archive.open("rb") as raw_fh, manifest_tmp.open("w", newline="", encoding="utf-8") as man_fh:
        counter = CountingReader(raw_fh)
        buffered = io.BufferedReader(counter, buffer_size=4 << 20)
        writer = csv.writer(man_fh)
        writer.writerow(["path", "folder", "year", "filename", "size_bytes"])
        try:
            with tarfile.open(fileobj=buffered, mode="r|gz") as tar:
                for member in tar:
                    stats["members_seen"] += 1
                    if time.monotonic() - last_log > 30:
                        pct = 100.0 * counter.count / total_size
                        log.info("[%s] %.1f%% of archive read, %d members seen, %d files written (%s)",
                                 site, pct, stats["members_seen"], stats["files_written"],
                                 human_bytes(stats["bytes_written"]))
                        last_log = time.monotonic()
                    if member.isdir():
                        continue
                    if not member.isfile():
                        log.warning("[%s] skipping non-regular member %s", site, member.name)
                        continue
                    parts = safe_parts(member.name, site)
                    if parts[-1].startswith("._"):  # macOS AppleDouble metadata, not data (seen in DRIAMS-A id/2016)
                        stats["macos_metadata_skipped"].append("/".join(parts))
                        continue
                    folder = parts[1] if len(parts) > 1 else ""
                    year = parts[2] if len(parts) > 3 else ""
                    writer.writerow(["/".join(parts), folder, year, parts[-1], member.size])
                    folder_stats = stats["per_folder"].setdefault(folder, {"files": 0, "bytes": 0, "written": 0})
                    folder_stats["files"] += 1
                    folder_stats["bytes"] += member.size

                    if folder not in folders:
                        continue
                    if codes is not None and folder in spectra_folders and Path(parts[-1]).stem not in codes:
                        stats["files_filtered_out"] += 1
                        continue
                    dest = root.joinpath(*parts)
                    if dest.exists() and dest.stat().st_size == member.size and not overwrite:
                        stats["files_skipped_existing"] += 1
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src = tar.extractfile(member)
                    if src is None:
                        continue
                    tmp = dest.with_name(dest.name + ".tmp")
                    with tmp.open("wb") as out:
                        while chunk := src.read(1 << 20):
                            out.write(chunk)
                    os.replace(tmp, dest)
                    stats["files_written"] += 1
                    stats["bytes_written"] += member.size
                    folder_stats["written"] += 1
        except (tarfile.TarError, EOFError, OSError) as exc:
            raise ExtractionError(f"Archive {archive.name} could not be read completely: {exc}") from exc

    os.replace(manifest_tmp, manifest_path)
    stats["elapsed_s"] = round(time.monotonic() - started, 1)
    stats.update({"site": site, "archive": str(archive), "folders_extracted": folders,
                  "species_filter": species, "manifest": str(manifest_path)})
    summary_path = man_dir / f"{site}_extraction_{'_'.join(folders)}{'_' + species.replace(' ', '_') if species else ''}.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    log.info("[%s] done in %.0f s: %d files written (%s), %d already present, %d filtered out",
             site, stats["elapsed_s"], stats["files_written"], human_bytes(stats["bytes_written"]),
             stats["files_skipped_existing"], stats["files_filtered_out"])
    for folder, fs in sorted(stats["per_folder"].items()):
        log.info("[%s]   %-14s %8d files  %10s in archive  %8d written",
                 site, folder, fs["files"], human_bytes(fs["bytes"]), fs["written"])
    if stats["macos_metadata_skipped"]:
        log.info("[%s] skipped %d macOS metadata file(s) ('._*'): %s", site,
                 len(stats["macos_metadata_skipped"]), ", ".join(stats["macos_metadata_skipped"][:5]))
    log.info("[%s] manifest: %s", site, manifest_path)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Selectively extract a DRIAMS archive.")
    parser.add_argument("--site", required=True, help="A, B, C or D")
    parser.add_argument("--folders", nargs="+", default=None,
                        help="folders to extract (default from config: id binned_6000)")
    parser.add_argument("--species", default=None,
                        help="only extract spectra of this species (needs the id folder already extracted)")
    parser.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        folders = args.folders or list(config["extraction"]["default_folders"])
        with keep_awake() as awake:
            if awake:
                log.info("Windows sleep is blocked while this command runs (closing the lid still sleeps).")
            extract(config, args.site, folders, args.species, args.overwrite)
        return 0
    except (ExtractionError, ConfigError, DataError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted. Files written so far are complete; re-run to continue (existing files are skipped).")
        return 130


if __name__ == "__main__":
    sys.exit(main())
