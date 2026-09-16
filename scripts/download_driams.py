"""Download DRIAMS archives with parallel, resumable HTTP range requests and verify checksums.

Why parallel? Zenodo limits each connection to roughly 0.3 MB/s (measured 2026-09-16), so the
86 GB DRIAMS-A archive would take about 3 days over a single connection. Splitting the file into
segments and fetching several at once is much faster.

How resuming works: every segment is stored as its own file in `<archive>.parts/`. The size of a
segment file is its progress, so re-running the same command continues where it stopped.
Segments are then joined into the final archive while the checksum is computed. The archive
only receives its final name after the checksum matches the value published by Zenodo/Dryad.

Usage (from the project root, with the virtual environment active):
    python scripts/download_driams.py --list
    python scripts/download_driams.py --site B
    python scripts/download_driams.py --site A --connections 8
    python scripts/download_driams.py --site B --verify-only
    python scripts/download_driams.py --site C --from-file "C:/Users/<you>/Downloads/DRIAMS_C.tar.gz"
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (  # noqa: E402
    ConfigError,
    archive_info,
    archives_dir,
    free_disk_gb,
    get_logger,
    human_bytes,
    keep_awake,
    load_config,
)

USER_AGENT = "antibiotic-resistance-ai/0.1 (academic research; DRIAMS download script)"
READ_CHUNK = 1 << 20  # 1 MiB
MAX_FAILURES_WITHOUT_PROGRESS = 30
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}

log = get_logger("download")
_active_lock = threading.Lock()


class DownloadError(RuntimeError):
    pass


class Progress:
    """Thread-safe byte counter with a rolling speed estimate."""

    def __init__(self, total: int, already: int) -> None:
        self.total = total
        self.done = already
        self._lock = threading.Lock()
        self._history: list[tuple[float, int]] = [(time.monotonic(), already)]

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n

    def line(self, label: str, active: int) -> str:
        now = time.monotonic()
        with self._lock:
            done = self.done
        self._history.append((now, done))
        self._history = [(t, d) for t, d in self._history if now - t <= 120] or [(now, done)]
        t0, d0 = self._history[0]
        speed = (done - d0) / (now - t0) if now > t0 else 0.0
        remaining = self.total - done
        eta = time.strftime("%H:%M:%S", time.gmtime(remaining / speed)) if speed > 0 else "--:--:--"
        pct = 100.0 * done / self.total if self.total else 100.0
        return (f"[{label}] {human_bytes(done)} / {human_bytes(self.total)} ({pct:5.1f}%)  "
                f"{speed / 1e6:5.2f} MB/s  ETA {eta}  active segments={active}")


def _request(url: str, start: int, end: int) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})


def probe_server(url: str, expected_size: int, max_wait_s: float = 1800, sleep=time.sleep) -> None:
    """Check that the server honours range requests and reports the expected total size.

    Temporary failures (e.g. Zenodo answering 504 Gateway Time-out, seen on 2026-09-16) are retried
    with growing pauses for up to `max_wait_s` seconds before giving up.
    """
    deadline = time.monotonic() + max_wait_s
    delay = 10
    while True:
        try:
            with urllib.request.urlopen(_request(url, 0, 0), timeout=60) as resp:
                status = resp.status
                content_range = resp.headers.get("Content-Range", "")
                resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRYABLE_HTTP:
                raise DownloadError(f"Download server refused the request: HTTP {exc.code} {exc.reason}") from exc
            problem = f"HTTP {exc.code} {exc.reason}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
            problem = str(exc)
        if time.monotonic() + delay > deadline:
            raise DownloadError(
                f"Download server still unavailable after {max_wait_s / 60:.0f} min (last error: {problem}). "
                "Nothing was lost; run the same command again later."
            )
        log.warning("Download server not available (%s); retrying in %d s ...", problem, delay)
        sleep(delay)
        delay = min(delay * 2, 120)
    if status != 206:
        raise DownloadError(f"Server did not accept a range request (HTTP {status}); parallel download impossible.")
    total = content_range.rsplit("/", 1)[-1]
    if not total.isdigit() or int(total) != expected_size:
        raise DownloadError(
            f"Server reports size {total!r} but config.yaml expects {expected_size}. "
            "The remote file may have changed; stop and re-check the dataset source."
        )


def fetch_segment(url: str, index: int, start: int, end: int, part: Path,
                  progress: Progress, stop: threading.Event, active: list[int]) -> int:
    """Download bytes [start, end] (inclusive) into `part`, appending to what is already there."""
    expected = end - start + 1
    failures = 0
    with _active_lock:
        active[0] += 1
    try:
        while not stop.is_set():
            have = part.stat().st_size if part.exists() else 0
            if have == expected:
                return index
            if have > expected:  # should never happen; start this segment again
                progress.add(-expected)
                part.unlink()
                continue
            written = 0
            try:
                with urllib.request.urlopen(_request(url, start + have, end), timeout=60) as resp:
                    if resp.status != 206:
                        raise DownloadError(f"Segment {index}: server ignored the range request (HTTP {resp.status}).")
                    content_range = resp.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start + have}-"):
                        raise DownloadError(f"Segment {index}: unexpected Content-Range {content_range!r}.")
                    with part.open("ab") as fh:
                        while not stop.is_set():
                            buf = resp.read(min(READ_CHUNK, expected - have - written))
                            if not buf:
                                break
                            fh.write(buf)
                            written += len(buf)
                            progress.add(len(buf))
                        fh.flush()
                if written:
                    failures = 0
                    continue
                failures += 1
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_HTTP:
                    raise DownloadError(f"Segment {index}: HTTP {exc.code} {exc.reason}") from exc
                failures += 0 if written else 1
                wait = exc.headers.get("Retry-After") if exc.headers else None
                delay = int(wait) if wait and wait.isdigit() else min(60, 2 ** min(failures, 6))
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError):
                failures += 0 if written else 1
                delay = min(60, 2 ** min(failures, 6))
            else:
                delay = min(60, 2 ** min(failures, 6))
            if failures >= MAX_FAILURES_WITHOUT_PROGRESS:
                raise DownloadError(f"Segment {index}: {failures} consecutive failed attempts; check your connection.")
            if failures:
                stop.wait(min(delay, 300))
        return -1
    finally:
        with _active_lock:
            active[0] -= 1


def hash_file(path: Path, algo: str, label: str, hasher=None, limit: int | None = None) -> "hashlib._Hash":
    """Hash `path` (optionally only its first `limit` bytes), printing progress every 15 s."""
    hasher = hasher or hashlib.new(algo)
    total = path.stat().st_size if limit is None else limit
    done = 0
    last = time.monotonic()
    with path.open("rb") as fh:
        while done < total:
            buf = fh.read(min(8 << 20, total - done))
            if not buf:
                break
            hasher.update(buf)
            done += len(buf)
            if time.monotonic() - last > 15:
                log.info("[%s] checksum %s / %s", label, human_bytes(done), human_bytes(total))
                last = time.monotonic()
    return hasher


def read_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def write_meta(meta_path: Path, meta: dict) -> None:
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)


def assemble(label: str, segments: list[tuple[int, int, int]], parts_dir: Path, meta_path: Path,
             tmp_path: Path, final_path: Path, algo: str, expected_checksum: str, size: int) -> None:
    """Join segment files into the archive while hashing; crash-safe and deletes parts as it goes."""
    meta = read_meta(meta_path)
    through = meta.get("assembled_through", -1)
    hasher = hashlib.new(algo)
    if through >= 0 and tmp_path.exists():
        keep = segments[through][2] + 1
        with tmp_path.open("r+b") as fh:
            fh.truncate(keep)
        log.info("[%s] resuming assembly after segment %d; re-hashing %s already joined",
                 label, through, human_bytes(keep))
        hash_file(tmp_path, algo, label, hasher=hasher, limit=keep)
    else:
        through = -1
        tmp_path.unlink(missing_ok=True)

    last = time.monotonic()
    with tmp_path.open("ab") as out:
        for index, start, end in segments[through + 1:]:
            part = parts_dir / f"seg_{index:05d}.part"
            if not part.exists() or part.stat().st_size != end - start + 1:
                raise DownloadError(f"Segment file {part.name} is missing or incomplete; re-run the download.")
            with part.open("rb") as src:
                while True:
                    buf = src.read(8 << 20)
                    if not buf:
                        break
                    hasher.update(buf)
                    out.write(buf)
            out.flush()
            os.fsync(out.fileno())
            meta["assembled_through"] = index
            write_meta(meta_path, meta)
            part.unlink()
            if time.monotonic() - last > 15:
                log.info("[%s] joined %s / %s", label, human_bytes(end + 1), human_bytes(size))
                last = time.monotonic()

    actual_size = tmp_path.stat().st_size
    if actual_size != size:
        raise DownloadError(f"Joined file has {actual_size} bytes, expected {size}.")
    digest = hasher.hexdigest()
    if digest.lower() != expected_checksum.lower():
        bad = tmp_path.with_suffix(".checksum-mismatch")
        os.replace(tmp_path, bad)
        raise DownloadError(
            f"{algo} mismatch: got {digest}, expected {expected_checksum}. The file was kept as {bad} "
            "for inspection. Delete it and the .parts folder, then download again."
        )
    os.replace(tmp_path, final_path)
    shutil.rmtree(parts_dir, ignore_errors=True)
    log.info("[%s] size OK, %s OK -> %s", label, algo, final_path)


def download(config: dict, site: str, connections: int, segment_mb: int, max_wait_min: float = 30) -> Path:
    info = archive_info(config, site)
    label = info["site"]
    if not info.get("url"):
        raise DownloadError(
            f"{label} cannot be downloaded by script. Download {info['filename']} in a browser from "
            f"{info.get('manual_download_page')} and then run:\n"
            f"  python scripts/download_driams.py --site {site} --from-file <path-to-downloaded-file>"
        )
    out_dir = archives_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / info["filename"]
    if final_path.exists():
        log.info("[%s] %s already exists (it was checksum-verified when created). "
                 "Use --verify-only to check it again.", label, final_path)
        return final_path

    size = int(info["size_bytes"])
    segment_bytes = segment_mb * (1 << 20)
    parts_dir = out_dir / f"{info['filename']}.parts"
    parts_dir.mkdir(exist_ok=True)
    meta_path = parts_dir / "meta.json"
    tmp_path = out_dir / f"{info['filename']}.assembling"
    if meta_path.exists():
        meta = read_meta(meta_path)
        if meta.get("url") != info["url"] or meta.get("size_bytes") != size:
            raise DownloadError(f"{parts_dir} belongs to a different download; delete it and retry.")
        segment_bytes = meta["segment_bytes"]  # keep the original layout when resuming
    else:
        meta = {"url": info["url"], "size_bytes": size, "segment_bytes": segment_bytes,
                "checksum_type": info["checksum_type"], "checksum": info["checksum"], "assembled_through": -1}
        write_meta(meta_path, meta)

    segments = [(i, start, min(start + segment_bytes, size) - 1)
                for i, start in enumerate(range(0, size, segment_bytes))]
    assembled_bytes = segments[meta["assembled_through"]][2] + 1 if meta.get("assembled_through", -1) >= 0 else 0
    pending = []
    on_disk = assembled_bytes
    for index, start, end in segments[meta.get("assembled_through", -1) + 1:]:
        part = parts_dir / f"seg_{index:05d}.part"
        have = part.stat().st_size if part.exists() else 0
        on_disk += min(have, end - start + 1)
        if have != end - start + 1:
            pending.append((index, start, end))

    needed_gb = (size - on_disk) / 1e9 + float(config["extraction"]["min_free_disk_gb"])
    if free_disk_gb(out_dir) < needed_gb:
        raise DownloadError(f"Not enough disk space: need about {needed_gb:.1f} GB free on {out_dir.drive or out_dir}.")

    log.info("[%s] %s, %d segments of %s, %d still to fetch, %s already on disk",
             label, human_bytes(size), len(segments), human_bytes(segment_bytes), len(pending), human_bytes(on_disk))

    if pending:
        probe_server(info["url"], size, max_wait_s=max_wait_min * 60)
        progress = Progress(size, on_disk)
        stop = threading.Event()
        active = [0]
        try:
            with ThreadPoolExecutor(max_workers=connections) as pool:
                futures = [pool.submit(fetch_segment, info["url"], i, s, e,
                                       parts_dir / f"seg_{i:05d}.part", progress, stop, active)
                           for i, s, e in pending]
                last_print = 0.0
                remaining = set(futures)
                while remaining:
                    done_now = {f for f in remaining if f.done()}
                    for fut in done_now:
                        fut.result()  # re-raises DownloadError from a worker
                    remaining -= done_now
                    if time.monotonic() - last_print > 30:
                        log.info(progress.line(label, active[0]))
                        last_print = time.monotonic()
                    time.sleep(0.5)
        except KeyboardInterrupt:
            stop.set()
            log.warning("[%s] Interrupted. Progress is saved; run the same command again to resume.", label)
            raise
        except Exception:
            stop.set()
            raise
        log.info(progress.line(label, 0))

    log.info("[%s] all segments downloaded; joining and verifying %s ...", label, info["checksum_type"])
    assemble(label, segments, parts_dir, meta_path, tmp_path, final_path,
             info["checksum_type"], info["checksum"], size)
    return final_path


def verify_existing(config: dict, site: str, path: Path | None = None) -> bool:
    info = archive_info(config, site)
    target = path or archives_dir(config) / info["filename"]
    if not target.is_file():
        log.error("File not found: %s", target)
        return False
    size = target.stat().st_size
    if size != int(info["size_bytes"]):
        log.error("[%s] size mismatch: %d bytes on disk, expected %s", info["site"], size, info["size_bytes"])
        return False
    digest = hash_file(target, info["checksum_type"], info["site"]).hexdigest()
    if digest.lower() != info["checksum"].lower():
        log.error("[%s] %s mismatch: got %s, expected %s", info["site"], info["checksum_type"], digest, info["checksum"])
        return False
    log.info("[%s] size OK, %s OK (%s)", info["site"], info["checksum_type"], target)
    return True


def register_manual_file(config: dict, site: str, source: Path) -> Path:
    """Verify a browser-downloaded archive and move it into the archives folder."""
    info = archive_info(config, site)
    if not verify_existing(config, site, source):
        raise DownloadError(f"{source} does not match the published size/checksum for {info['site']}; not used.")
    dest = archives_dir(config) / info["filename"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.move(str(source), str(dest))
    log.info("[%s] registered at %s", info["site"], dest)
    return dest


def list_status(config: dict) -> None:
    out_dir = archives_dir(config)
    print(f"Archive folder: {out_dir}")
    for letter, info in config["driams"]["archives"].items():
        final = out_dir / info["filename"]
        parts = out_dir / f"{info['filename']}.parts"
        if final.exists():
            status = "complete (verified)"
        elif parts.exists():
            got = sum(p.stat().st_size for p in parts.glob("seg_*.part"))
            status = f"partial: {human_bytes(got)} downloaded (resume with --site {letter})"
        else:
            status = "not downloaded"
        source = "Zenodo (script)" if info.get("url") else "Dryad (browser, then --from-file)"
        print(f"  {letter}  {info['filename']:<16} {human_bytes(int(info['size_bytes'])):>9}  {source:<34} {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify DRIAMS archives.")
    parser.add_argument("--site", help="A, B, C or D")
    parser.add_argument("--connections", type=int, default=8, help="parallel connections (default 8)")
    parser.add_argument("--segment-mb", type=int, default=128, help="segment size in MiB for new downloads")
    parser.add_argument("--verify-only", action="store_true", help="only verify an already downloaded archive")
    parser.add_argument("--from-file", type=Path, help="verify a browser-downloaded archive and move it into place")
    parser.add_argument("--list", action="store_true", help="show the status of all archives")
    parser.add_argument("--max-wait-min", type=float, default=30,
                        help="minutes to keep retrying if the server is temporarily down (default 30)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.list:
            list_status(config)
            return 0
        if not args.site:
            parser.error("--site is required (or use --list)")
        if not 1 <= args.connections <= 16:
            parser.error("--connections must be between 1 and 16")
        with keep_awake() as awake:
            if awake:
                log.info("Windows sleep is blocked while this command runs (closing the lid still sleeps).")
            if args.from_file:
                register_manual_file(config, args.site, args.from_file)
            elif args.verify_only:
                return 0 if verify_existing(config, args.site) else 1
            else:
                download(config, args.site, args.connections, args.segment_mb, args.max_wait_min)
        return 0
    except (DownloadError, ConfigError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
