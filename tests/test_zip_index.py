"""The streaming zip reader (src/zip_index.py) agrees with Python's zipfile, on archives written by these tests."""

from __future__ import annotations

import zipfile

import pytest

from src.zip_index import Member, ZipIndexError, iter_members, read_member


def make_archive(path, *, force_zip64=False):
    contents = {"MARISMa/": b"", "MARISMa/a/fid": bytes(range(256)) * 40, "MARISMa/a/acqu": b"##$TD= 10\n" * 50,
                "MARISMa/b/stored": b"stored bytes", "MARISMa/b/empty": b"", "née/utf8": b"x"}
    with zipfile.ZipFile(path, "w") as z:
        for name, data in contents.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED if "stored" in name or name.endswith("/") else zipfile.ZIP_DEFLATED
            with z.open(info, "w", force_zip64=force_zip64) as fh:
                fh.write(data)
    return contents


@pytest.mark.parametrize("force_zip64", [False, True])
def test_members_and_bytes_agree_with_zipfile(tmp_path, force_zip64):
    path = tmp_path / "a.zip"
    contents = make_archive(path, force_zip64=force_zip64)
    members = dict(iter_members(path))
    with zipfile.ZipFile(path) as z:
        infos = {i.filename: i for i in z.infolist()}
    assert list(members) == list(infos) == list(contents)
    with open(path, "rb") as fh:
        for name, member in members.items():
            info = infos[name]
            assert (member.offset, member.compressed, member.size, member.method, member.crc) == (
                info.header_offset, info.compress_size, info.file_size, info.compress_type, info.CRC)
            assert read_member(fh, member) == contents[name]


def test_a_corrupted_member_is_refused(tmp_path):
    path = tmp_path / "a.zip"
    make_archive(path)
    name, member = next((n, m) for n, m in iter_members(path) if n == "MARISMa/b/stored")
    with open(path, "rb") as fh:
        with pytest.raises(ZipIndexError):
            read_member(fh, Member(member.offset, member.compressed, member.size, member.method, member.crc ^ 1))


def test_a_file_without_end_record_is_refused(tmp_path):
    path = tmp_path / "not.zip"
    path.write_bytes(b"not a zip archive" * 100)
    with pytest.raises(ZipIndexError):
        list(iter_members(path))
