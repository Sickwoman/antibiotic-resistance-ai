"""Read a very large zip archive without holding Python's full index of it in memory.

MARISMa.zip has 3.1 million members. `zipfile.ZipFile` keeps a ZipInfo object for each, about 3 GB on this project's
8 GB laptop, which then swaps. This module streams the central directory once, so the caller keeps only what it needs
(here, every member's name, and the location of the E. coli members), and reads a member from its local header.

It supports what MARISMa.zip uses: Zip64, members stored (method 0) or deflated (method 8), no encryption, one disk.
Anything else raises ZipIndexError. Every member read is checked against the CRC-32 and size in the central directory.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

EOCD = b"PK\x05\x06"
ZIP64_LOCATOR = b"PK\x06\x07"
ZIP64_EOCD = b"PK\x06\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


class ZipIndexError(Exception):
    """The archive is malformed, or uses a feature this reader does not support."""


@dataclass(frozen=True)
class Member:
    offset: int          # of the local file header
    compressed: int
    size: int
    method: int          # 0 stored, 8 deflated
    crc: int


def _directory_location(fh: BinaryIO) -> tuple[int, int, int]:
    """(number of entries, size, offset) of the central directory, from the end records (Zip64 if present)."""
    fh.seek(0, 2)
    end = fh.tell()
    tail_size = min(end, 22 + 65535)
    fh.seek(end - tail_size)
    tail = fh.read(tail_size)
    at = tail.rfind(EOCD)
    if at < 0:
        raise ZipIndexError("no end-of-central-directory record")
    disk, cd_disk, _, entries, cd_size, cd_offset, _ = struct.unpack("<HHHHIIH", tail[at + 4:at + 22])
    if disk != 0 or cd_disk != 0:
        raise ZipIndexError("multi-disk archives are not supported")
    if entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        locator = tail[at - 20:at]
        if len(locator) != 20 or locator[:4] != ZIP64_LOCATOR:
            raise ZipIndexError("Zip64 values without a Zip64 locator")
        (record_offset,) = struct.unpack("<Q", locator[8:16])
        fh.seek(record_offset)
        record = fh.read(56)
        if record[:4] != ZIP64_EOCD:
            raise ZipIndexError("no Zip64 end-of-central-directory record")
        entries, _, cd_size, cd_offset = struct.unpack("<QQQQ", record[24:56])
    return entries, cd_size, cd_offset


def iter_members(path: str | Path) -> Iterator[tuple[str, Member]]:
    """Every member of the archive, in central-directory order, as (name, Member)."""
    fixed = struct.Struct("<4s4xHH4xIIIHHHH6xI")       # the 46-byte fixed part of a central-directory entry
    with open(path, "rb") as fh:
        entries, cd_size, cd_offset = _directory_location(fh)
        fh.seek(cd_offset)
        buffer, pos = b"", 0
        for _ in range(entries):
            if len(buffer) - pos < 46:
                buffer, pos = buffer[pos:] + fh.read(1 << 20), 0
            (signature, flags, method, crc, compressed, size, name_len, extra_len, comment_len, disk,
             offset) = fixed.unpack_from(buffer, pos)
            if signature != CENTRAL:
                raise ZipIndexError("a central-directory entry is malformed")
            total = 46 + name_len + extra_len + comment_len
            while len(buffer) - pos < total:
                more = fh.read(1 << 20)
                if not more:
                    raise ZipIndexError("the central directory is truncated")
                buffer, pos = buffer[pos:] + more, 0
            raw_name = buffer[pos + 46:pos + 46 + name_len]
            extra = buffer[pos + 46 + name_len:pos + 46 + name_len + extra_len]
            pos += total
            if flags & 0x1:
                raise ZipIndexError("encrypted members are not supported")
            if 0xFFFFFFFF in (compressed, size, offset) or disk == 0xFFFF:
                size, compressed, offset = _zip64_values(extra, size, compressed, offset, disk)
            name = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
            yield name, Member(offset, compressed, size, method, crc)


def _zip64_values(extra: bytes, size: int, compressed: int, offset: int, disk: int) -> tuple[int, int, int]:
    """The Zip64 extra field holds, in this order, each value whose 32-bit field is 0xFFFFFFFF."""
    i = 0
    while i + 4 <= len(extra):
        tag, length = struct.unpack("<HH", extra[i:i + 4])
        if tag == 0x0001:
            data, j = extra[i + 4:i + 4 + length], 0
            if size == 0xFFFFFFFF:
                (size,), j = struct.unpack("<Q", data[j:j + 8]), j + 8
            if compressed == 0xFFFFFFFF:
                (compressed,), j = struct.unpack("<Q", data[j:j + 8]), j + 8
            if offset == 0xFFFFFFFF:
                (offset,), j = struct.unpack("<Q", data[j:j + 8]), j + 8
            return size, compressed, offset
        i += 4 + length
    raise ZipIndexError("a Zip64 value is missing its extra field")


def read_member(fh: BinaryIO, member: Member) -> bytes:
    """The member's bytes, decompressed and checked against its CRC-32 and size."""
    fh.seek(member.offset)
    header = fh.read(30)
    if header[:4] != LOCAL:
        raise ZipIndexError("a local file header is malformed")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    fh.seek(member.offset + 30 + name_len + extra_len)
    data = fh.read(member.compressed)
    if len(data) != member.compressed:
        raise ZipIndexError("a member is truncated")
    if member.method == 8:
        data = zlib.decompress(data, -15)
    elif member.method != 0:
        raise ZipIndexError(f"compression method {member.method} is not supported")
    if len(data) != member.size or zlib.crc32(data) != member.crc:
        raise ZipIndexError("a member does not match its size or CRC-32")
    return data
