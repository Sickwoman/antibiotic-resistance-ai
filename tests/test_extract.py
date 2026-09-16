"""Tests for the selective archive extractor (synthetic tar.gz, no downloads)."""

from __future__ import annotations

import csv
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from extract_driams import ExtractionError, extract, safe_parts  # noqa: E402


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _config(root: Path) -> dict:
    return {
        "paths": {"driams_root": str(root), "archives_subdir": "archives", "manifests_subdir": "manifests"},
        "driams": {
            "archives": {"Z": {"site": "DRIAMS-Z", "filename": "DRIAMS_Z.tar.gz", "url": None,
                               "size_bytes": 0, "checksum_type": "md5", "checksum": ""}},
            "spectra_folders": ["raw", "preprocessed", "binned_6000"],
            "id_folder": "id",
            "id_suffixes": ["strat", "clean"],
        },
        "labels": {"missing_values": ["-", ""]},
        "extraction": {"default_folders": ["id", "binned_6000"], "min_free_disk_gb": 0},
    }


@pytest.fixture()
def archive_root(tmp_path):
    root = tmp_path / "DRIAMS"
    (root / "archives").mkdir(parents=True)
    with tarfile.open(root / "archives" / "DRIAMS_Z.tar.gz", "w:gz") as tar:
        _add(tar, "DRIAMS-Z/raw/2018/aaa.txt", b"2000 1\n")
        _add(tar, "DRIAMS-Z/raw/2018/bbb.txt", b"2000 2\n")
        _add(tar, "DRIAMS-Z/binned_6000/2018/aaa.txt", b"0 0.1\n")
        _add(tar, "DRIAMS-Z/id/2018/2018_clean.csv",
             b"code,species,Ciprofloxacin\naaa,Escherichia coli,R\nbbb,Staphylococcus aureus,S\n")
    return root


@pytest.mark.parametrize("name", ["/etc/passwd", "DRIAMS-Z/../../evil.txt", "C:/evil.txt",
                                  "OTHER/id/x.csv", "DRIAMS-Z/raw/..\\evil.txt"])
def test_safe_parts_rejects_unsafe_names(name):
    with pytest.raises(ExtractionError):
        safe_parts(name, "DRIAMS-Z")


def test_safe_parts_accepts_normal_member():
    assert safe_parts("DRIAMS-Z/raw/2018/x.txt", "DRIAMS-Z") == ("DRIAMS-Z", "raw", "2018", "x.txt")


def test_extract_selected_folders_and_manifest(archive_root):
    stats = extract(_config(archive_root), "Z", ["id", "binned_6000"], None, overwrite=False)
    assert (archive_root / "DRIAMS-Z" / "id" / "2018" / "2018_clean.csv").is_file()
    assert (archive_root / "DRIAMS-Z" / "binned_6000" / "2018" / "aaa.txt").is_file()
    assert not (archive_root / "DRIAMS-Z" / "raw").exists()
    assert stats["files_written"] == 2
    with open(archive_root / "manifests" / "DRIAMS-Z_manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {r["folder"] for r in rows} == {"raw", "binned_6000", "id"}

    again = extract(_config(archive_root), "Z", ["id", "binned_6000"], None, overwrite=False)
    assert again["files_written"] == 0 and again["files_skipped_existing"] == 2


def test_macos_metadata_files_are_skipped(tmp_path):
    root = tmp_path / "DRIAMS"
    (root / "archives").mkdir(parents=True)
    with tarfile.open(root / "archives" / "DRIAMS_Z.tar.gz", "w:gz") as tar:
        _add(tar, "DRIAMS-Z/id/2016/2016_clean.csv", b"code,species\na,Escherichia coli\n")
        _add(tar, "DRIAMS-Z/id/2016/._2016_notes.csv", b"\x00\x05\x16\x07Mac OS X")
    stats = extract(_config(root), "Z", ["id"], None, overwrite=False)
    assert stats["macos_metadata_skipped"] == ["DRIAMS-Z/id/2016/._2016_notes.csv"]
    assert not (root / "DRIAMS-Z" / "id" / "2016" / "._2016_notes.csv").exists()
    with open(root / "manifests" / "DRIAMS-Z_manifest.csv", newline="", encoding="utf-8") as fh:
        assert [r["filename"] for r in csv.DictReader(fh)] == ["2016_clean.csv"]


def test_extract_species_filter(archive_root):
    config = _config(archive_root)
    extract(config, "Z", ["id"], None, overwrite=False)
    stats = extract(config, "Z", ["raw"], "Escherichia coli", overwrite=False)
    assert (archive_root / "DRIAMS-Z" / "raw" / "2018" / "aaa.txt").is_file()
    assert not (archive_root / "DRIAMS-Z" / "raw" / "2018" / "bbb.txt").exists()
    assert stats["files_filtered_out"] == 1


def test_extract_refuses_path_traversal(tmp_path):
    root = tmp_path / "DRIAMS"
    (root / "archives").mkdir(parents=True)
    with tarfile.open(root / "archives" / "DRIAMS_Z.tar.gz", "w:gz") as tar:
        _add(tar, "DRIAMS-Z/id/../../../evil.txt", b"x")
    with pytest.raises(ExtractionError):
        extract(_config(root), "Z", ["id"], None, overwrite=False)
    assert not (tmp_path / "evil.txt").exists()


def test_extract_truncated_archive(tmp_path, archive_root):
    path = archive_root / "archives" / "DRIAMS_Z.tar.gz"
    path.write_bytes(path.read_bytes()[:60])
    with pytest.raises(ExtractionError, match="could not be read"):
        extract(_config(archive_root), "Z", ["id"], None, overwrite=False)


def test_missing_archive_message(tmp_path):
    with pytest.raises(ExtractionError, match="download_driams.py"):
        extract(_config(tmp_path), "Z", ["id"], None, overwrite=False)


def test_unknown_folder(archive_root):
    with pytest.raises(ExtractionError, match="Unknown folder"):
        extract(_config(archive_root), "Z", ["secrets"], None, overwrite=False)
