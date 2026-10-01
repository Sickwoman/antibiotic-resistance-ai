"""Test automation must surface failures (Version 1.3 closure, docs/v1.3_threshold_plan.md).

A docs commit once landed with two failing tests because a local `pytest ... | tail -1` returned tail's exit
status. These tests pin what keeps a failure visible: CI runs every step with errexit and pipefail, never pipes
the test or lint command, and the whole test session fails if it changed the project's experiment logs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from src.utils import load_config
from tests.log_guard import GUARDED_LOGS, changed_logs, log_digests

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def _job() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["pytest"]


def test_ci_runs_every_step_with_errexit_and_pipefail():
    job = _job()
    assert job["defaults"]["run"]["shell"] == "bash"          # GitHub: bash --noprofile --norc -eo pipefail {0}
    assert all(step.get("shell", "bash") == "bash" for step in job["steps"])
    assert set(job["strategy"]["matrix"]["os"]) == {"ubuntu-latest", "windows-latest"}


def test_ci_never_pipes_the_test_or_lint_command():
    runs = [step["run"] for step in _job()["steps"] if "run" in step]
    checked = [r for r in runs if "pytest" in r or "ruff" in r]
    assert len(checked) == 2 and not any("|" in r for r in checked)


def test_the_log_guard_detects_a_changed_created_or_deleted_log(tmp_path):
    kept, edited, created, deleted = (tmp_path / f"{n}.csv" for n in ("kept", "edited", "created", "deleted"))
    for path in (kept, edited, deleted):
        path.write_text("a,b\n1,2\n", encoding="utf-8")
    before = log_digests([kept, edited, created, deleted])
    assert changed_logs(before) == []
    edited.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    created.write_text("a\n", encoding="utf-8")
    deleted.unlink()
    assert changed_logs(before) == sorted(str(p) for p in (edited, created, deleted))


def test_a_session_that_writes_a_guarded_log_fails_and_one_that_does_not_passes(tmp_path):
    """The real guard, run end to end on a throwaway copy: the project's own logs are never touched."""
    (tmp_path / "tests").mkdir()
    for name in ("conftest.py", "log_guard.py"):
        (tmp_path / "tests" / name).write_text((ROOT / "tests" / name).read_text(encoding="utf-8"), encoding="utf-8")
    for log in GUARDED_LOGS:
        (tmp_path / log).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / log).write_text("header\n", encoding="utf-8")
    body = ("from pathlib import Path\n\n\ndef test_it():\n    log = Path(__file__).resolve().parents[1] / {!r}\n"
            "    {}\n")
    outcomes = {}
    for label, action in (("writes", "log.write_text(log.read_text() + 'row\\n')"), ("reads", "log.read_text()")):
        (tmp_path / "tests" / "test_inner.py").write_text(body.format(GUARDED_LOGS[0], action), encoding="utf-8")
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                             cwd=tmp_path, capture_output=True, text=True, timeout=300)
        outcomes[label] = (run.returncode, run.stdout + run.stderr)
    assert outcomes["writes"][0] == 1 and "modified historical experiment logs" in outcomes["writes"][1]
    assert outcomes["reads"][0] == 0, outcomes["reads"][1]


def test_the_session_guard_watches_the_production_and_development_logs():
    config = load_config()
    assert set(GUARDED_LOGS) == {config["evaluation"]["test_log"], config["v12_development"]["development_log"],
                                 config["v13_threshold"]["development_log"]}
