"""Check MARISMa.zip's integrity: its SHA-256 against provenance.json, and every member's CRC-32 and size.

Version 2.0, step 1 (docs/v2.0_marisma_plan.md). Nothing is extracted to disk and no member's content is shown: each
member is decompressed in memory, checked against the central directory, and discarded. The output holds counts only.

    python scripts/v20_archive_integrity.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import get_logger, keep_awake, load_config  # noqa: E402
from src.zip_index import ZipIndexError, iter_members, read_member  # noqa: E402

log = get_logger("v20_archive_integrity")


def main(argv: list[str] | None = None) -> int:
    data_root = Path(load_config()["paths"]["driams_root"])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=data_root / "MARISMa_v2.0.0" / "MARISMa.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "results/metrics/v2.0/archive_integrity.json")
    args = parser.parse_args(argv)
    provenance = json.loads((args.zip.parent / "provenance.json").read_text(encoding="utf-8"))
    recorded = provenance["files"]["MARISMa.zip"]

    with keep_awake():
        digest = hashlib.sha256()
        with open(args.zip, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                digest.update(block)
        sha256_ok = digest.hexdigest() == recorded["sha256_local"]
        log.info("SHA-256 %s the provenance record", "matches" if sha256_ok else "DOES NOT match")

        outcome: Counter[str] = Counter()
        methods: Counter[int] = Counter()
        uncompressed = members = 0
        with open(args.zip, "rb") as fh:
            for name, member in iter_members(args.zip):     # streamed: no index of all members is kept
                members += 1
                methods[member.method] += 1
                if name.endswith("/"):
                    outcome["directory_entries"] += 1
                    continue
                try:
                    uncompressed += len(read_member(fh, member))
                    outcome["files_crc_and_size_ok"] += 1
                except (ZipIndexError, zlib.error):
                    outcome["files_failed"] += 1
                if members % 500_000 == 0:
                    log.info("checked %d members", members)

    result = {
        "what": "MARISMa.zip integrity: SHA-256 against provenance.json, and every member's CRC-32 and size",
        "checked_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "size_bytes": args.zip.stat().st_size, "size_matches_provenance": args.zip.stat().st_size == recorded[
            "size_bytes"],
        "sha256_matches_provenance": sha256_ok, "md5_published_and_verified": recorded["md5_published_and_verified"],
        "members": members, "outcome": dict(sorted(outcome.items())),
        "compression_methods": {str(k): v for k, v in sorted(methods.items())},
        "uncompressed_bytes": uncompressed,
        "all_ok": sha256_ok and outcome["files_failed"] == 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("written %s", args.out.relative_to(ROOT))
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
