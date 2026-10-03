"""Download the two MARISMa 2.0.0 files that Version 2.0 step 1 authorises, and record their provenance.

docs/v2.0_marisma_plan.md, step 1 (approved 2026-10-03, steps 1-3 only): `MARISMa.zip` and `AMR.csv` from Zenodo record
17201597 (version 2.0.0, doi:10.5281/zenodo.17201597). The record's two statistics files are never downloaded. The files
go outside the repository, under the configured data root. `AMR.csv` goes to its own sealed folder, because only the
restricted schema reader (amendment A2) may open it before scoring.

The download reuses scripts/download_driams.py's parallel, resumable range requests. A file receives its final name
only after its size and the MD5 that Zenodo publishes both match. The script then records local SHA-256 values,
sizes, URLs and UTC timestamps in `provenance.json` beside the archive. It never opens or prints the contents of
either file.

    python scripts/download_marisma.py              # download both files and verify them
    python scripts/download_marisma.py --verify-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_driams import (  # noqa: E402
    DownloadError,
    Progress,
    assemble,
    fetch_segment,
    hash_file,
    probe_server,
    read_meta,
    write_meta,
)
from src.utils import free_disk_gb, get_logger, human_bytes, keep_awake, load_config  # noqa: E402

RECORD_ID = 17201597
VERSION = "2.0.0"
API = f"https://zenodo.org/api/records/{RECORD_ID}"
# What the record published when the plan was approved (read from the Zenodo API on 2026-10-03). Any difference
# means the record changed, and the download stops.
AUTHORISED = {
    "MARISMa.zip": {"size": 16825605739, "md5": "1af8137ec7f890f9b5f6e0aa82a0ace3", "folder": "MARISMa_v2.0.0"},
    "AMR.csv": {"size": 14130346, "md5": "9d8cc36ce213f4cd5254f906234779ca", "folder": "MARISMa_v2.0.0_sealed"},
}
NEVER = {"amr_stats.json", "detailed_dataset_statistics.json"}   # statistics files: not downloaded

log = get_logger("download_marisma")


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_files() -> dict[str, dict]:
    """The live record's files, checked against what was authorised."""
    req = urllib.request.Request(API, headers={"User-Agent": "antibiotic-resistance-ai (academic research)"})
    rec = json.load(urllib.request.urlopen(req, timeout=60))
    if rec["metadata"].get("version") != VERSION:
        raise DownloadError(f"Zenodo record {RECORD_ID} is version {rec['metadata'].get('version')}, not {VERSION}.")
    files = {f["key"]: f for f in rec["files"]}
    for name, want in AUTHORISED.items():
        f = files.get(name)
        if f is None or f["size"] != want["size"] or f["checksum"] != f"md5:{want['md5']}":
            raise DownloadError(f"{name} in record {RECORD_ID} no longer matches the authorised size and MD5.")
    return {name: {**AUTHORISED[name], "url": files[name]["links"]["self"]} for name in AUTHORISED}


def download_one(name: str, info: dict, out_dir: Path, connections: int, segment_mb: int, min_free_gb: float) -> Path:
    """scripts/download_driams.py's download(), for one file described by `info` instead of config.yaml."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / name
    if final_path.exists():
        log.info("[%s] already present (checksum-verified when created)", name)
        return final_path
    size, segment_bytes = info["size"], segment_mb * (1 << 20)
    parts_dir = out_dir / f"{name}.parts"
    parts_dir.mkdir(exist_ok=True)
    meta_path, tmp_path = parts_dir / "meta.json", out_dir / f"{name}.assembling"
    if meta_path.exists():
        meta = read_meta(meta_path)
        if meta.get("url") != info["url"] or meta.get("size_bytes") != size:
            raise DownloadError(f"{parts_dir} belongs to a different download; delete it and retry.")
        segment_bytes = meta["segment_bytes"]
    else:
        meta = {"url": info["url"], "size_bytes": size, "segment_bytes": segment_bytes, "checksum_type": "md5",
                "checksum": info["md5"], "assembled_through": -1}
        write_meta(meta_path, meta)
    segments = [(i, s, min(s + segment_bytes, size) - 1) for i, s in enumerate(range(0, size, segment_bytes))]
    through = meta.get("assembled_through", -1)
    pending, on_disk = [], (segments[through][2] + 1 if through >= 0 else 0)
    for index, start, end in segments[through + 1:]:
        part = parts_dir / f"seg_{index:05d}.part"
        have = part.stat().st_size if part.exists() else 0
        on_disk += min(have, end - start + 1)
        if have != end - start + 1:
            pending.append((index, start, end))
    if free_disk_gb(out_dir) < (size - on_disk) / 1e9 + min_free_gb:
        raise DownloadError(f"Not enough free disk space on {out_dir.drive or out_dir} for {name}.")
    log.info("[%s] %s in %d segments, %d to fetch", name, human_bytes(size), len(segments), len(pending))
    if pending:
        probe_server(info["url"], size)
        progress, stop, active = Progress(size, on_disk), threading.Event(), [0]
        try:
            with ThreadPoolExecutor(max_workers=connections) as pool:
                remaining = {pool.submit(fetch_segment, info["url"], i, s, e, parts_dir / f"seg_{i:05d}.part",
                                         progress, stop, active) for i, s, e in pending}
                last = 0.0
                while remaining:
                    done = {f for f in remaining if f.done()}
                    for fut in done:
                        fut.result()
                    remaining -= done
                    if time.monotonic() - last > 60:
                        log.info(progress.line(name, active[0]))
                        last = time.monotonic()
                    time.sleep(0.5)
        except BaseException:
            stop.set()
            raise
    assemble(name, segments, parts_dir, meta_path, tmp_path, final_path, "md5", info["md5"], size)
    return final_path


def sha256(path: Path) -> str:
    return hash_file(path, "sha256", path.name, hasher=hashlib.sha256()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and verify the authorised MARISMa 2.0.0 files.")
    ap.add_argument("--connections", type=int, default=8)
    ap.add_argument("--segment-mb", type=int, default=128)
    ap.add_argument("--verify-only", action="store_true", help="re-check MD5 and SHA-256 of downloaded files")
    args = ap.parse_args()
    config = load_config()
    root = Path(config["paths"]["driams_root"])
    started = now()
    try:
        files = record_files()
        prov = {"record": RECORD_ID, "version": VERSION, "doi": f"10.5281/zenodo.{RECORD_ID}", "api": API,
                "not_downloaded": sorted(NEVER), "started_utc": started, "files": {}}
        with keep_awake():
            for name, info in files.items():
                out_dir = root / info["folder"]
                path = out_dir / name
                if not args.verify_only:
                    path = download_one(name, info, out_dir, args.connections, args.segment_mb,
                                        float(config["extraction"]["min_free_disk_gb"]))
                if not path.exists():
                    raise DownloadError(f"{path} is missing.")
                md5 = hash_file(path, "md5", name).hexdigest()
                if path.stat().st_size != info["size"] or md5 != info["md5"]:
                    raise DownloadError(f"{name}: size or MD5 differs from the published value.")
                prov["files"][name] = {"url": info["url"], "folder": info["folder"], "size_bytes": info["size"],
                                       "md5_published_and_verified": md5, "sha256_local": sha256(path),
                                       "verified_utc": now()}
                log.info("[%s] size and published MD5 verified; SHA-256 recorded", name)
        prov["finished_utc"] = now()
        out = root / AUTHORISED["MARISMa.zip"]["folder"] / "provenance.json"
        out.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(prov, indent=2))
        return 0
    except DownloadError as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
