"""The project's append-only experiment logs, and a check that the test suite never changes them.

Every test that runs a script must point its production and development logs into pytest's temporary folders.
`tests/conftest.py` hashes the real logs when a test session starts and fails the session if either changed by
the end - a guard over the whole run, so a test that forgets to redirect a log cannot pass unnoticed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GUARDED_LOGS = ("results/experiments/test_evaluations.csv", "results/experiments/development_runs.csv")


def log_digests(paths: list[Path]) -> dict[Path, str | None]:
    """SHA-256 of each file's bytes; None for a file that does not exist."""
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in paths}


def changed_logs(before: dict[Path, str | None]) -> list[str]:
    """The files whose bytes (or existence) differ from `before`."""
    now = log_digests(list(before))
    return sorted(str(p) for p in before if now[p] != before[p])
