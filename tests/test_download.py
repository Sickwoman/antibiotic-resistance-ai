"""Tests for the downloader's server check (network calls are simulated)."""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_driams  # noqa: E402
from download_driams import DownloadError, probe_server  # noqa: E402

URL = "https://example.org/archive.tar.gz"


class FakeResponse:
    def __init__(self, status: int, content_range: str) -> None:
        self.status = status
        self.headers = {"Content-Range": content_range}

    def read(self) -> bytes:
        return b"x"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _gateway_timeout():
    return urllib.error.HTTPError(URL, 504, "Gateway Time-out", {}, None)


def test_probe_retries_temporary_server_errors(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _gateway_timeout()
        return FakeResponse(206, "bytes 0-0/100")

    monkeypatch.setattr(download_driams.urllib.request, "urlopen", fake_urlopen)
    probe_server(URL, 100, max_wait_s=600, sleep=sleeps.append)
    assert calls["n"] == 3
    assert sleeps == [10, 20]


def test_probe_gives_up_with_clear_message(monkeypatch):
    def always_down(request, timeout):
        raise _gateway_timeout()

    monkeypatch.setattr(download_driams.urllib.request, "urlopen", always_down)
    with pytest.raises(DownloadError, match="still unavailable"):
        probe_server(URL, 100, max_wait_s=0, sleep=lambda s: None)


def test_probe_does_not_retry_permanent_errors(monkeypatch):
    def not_found(request, timeout):
        raise urllib.error.HTTPError(URL, 404, "Not Found", {}, None)

    monkeypatch.setattr(download_driams.urllib.request, "urlopen", not_found)
    with pytest.raises(DownloadError, match="HTTP 404"):
        probe_server(URL, 100, max_wait_s=600, sleep=lambda s: pytest.fail("should not wait"))


def test_probe_rejects_size_change(monkeypatch):
    monkeypatch.setattr(download_driams.urllib.request, "urlopen",
                        lambda request, timeout: FakeResponse(206, "bytes 0-0/999"))
    with pytest.raises(DownloadError, match="remote file may have changed"):
        probe_server(URL, 100, max_wait_s=0)


def test_probe_rejects_servers_without_range_support(monkeypatch):
    monkeypatch.setattr(download_driams.urllib.request, "urlopen",
                        lambda request, timeout: FakeResponse(200, ""))
    with pytest.raises(DownloadError, match="range request"):
        probe_server(URL, 100, max_wait_s=0)
