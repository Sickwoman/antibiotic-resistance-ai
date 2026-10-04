"""The gate in front of Version 2.0's one-time evaluation (docs/v2.0_marisma_plan.md, amendment F5.11).

The evaluation may run only under a scoring authorisation that the owner records in the plan, in a later dated entry
pinned like the others. The entry holds a machine-readable block between the markers below, naming the code commit
the owner approved and the fingerprint of the scoring code at that commit.

`verify` refuses unless all of these hold:
- exactly one authorisation block exists, inside the plan's last pinned segment, and every pin in
  tests/test_preregistration.py still matches;
- the working tree has no change to a tracked file;
- the authorised commit exists, is an ancestor of HEAD, and no scoring-code file changed since it;
- the scoring code's fingerprint now equals the recorded one.

No authorisation block exists when this module is written, so every production path refuses.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/v2.0_marisma_plan.md"
AUTH_BEGIN = "<!-- v2.0 scoring authorisation: begin -->"
AUTH_END = "<!-- v2.0 scoring authorisation: end -->"
CODE_FILES = ("scripts/v20_score_marisma.py", "src/v20_evaluation.py", "src/v20_scoring_guard.py",
              "src/marisma_labels.py", "src/marisma_schema.py", "src/primary_endpoint.py", "src/evaluate.py",
              "src/uncertainty.py", "src/predict.py")


class GuardError(RuntimeError):
    """The evaluation may not run: the reason is in the message."""


@dataclass(frozen=True)
class Authorisation:
    authorised_commit: str
    code_fingerprint: str
    recorded_utc: str


def code_fingerprint(root: Path = ROOT) -> str:
    """SHA-256 over the scoring code's files: each file's path, a NUL byte, and its bytes with LF line endings."""
    digest = hashlib.sha256()
    for name in sorted(CODE_FILES):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def recorded_authorisation(plan_text: str) -> dict | None:
    """The authorisation block's content, or None if the plan holds none. More than one is refused."""
    count = plan_text.count(AUTH_BEGIN)
    if count == 0:
        return None
    if count != 1 or plan_text.count(AUTH_END) != 1:
        raise GuardError("the plan must hold exactly one scoring-authorisation block")
    block = plan_text.split(AUTH_BEGIN, 1)[1].split(AUTH_END, 1)[0].strip()
    if not (block.startswith("```json") and block.endswith("```")):
        raise GuardError("the scoring-authorisation block is not a fenced JSON block")
    record = json.loads(block[len("```json"):-3])
    for key in ("authorised_commit", "code_fingerprint", "recorded_utc"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise GuardError(f"the scoring authorisation lacks {key!r}")
    return record


def _pins(root: Path):
    spec = importlib.util.spec_from_file_location("preregistration_pins", root / "tests/test_preregistration.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=60)


def check_pins(root: Path = ROOT) -> None:
    """Every locked prefix and every pinned amendment of every document still matches."""
    pins = _pins(root)
    for path, (length, digest, _commit) in pins.LOCKED.items():
        data = (root / path).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data[:length]).hexdigest() != digest:
            raise GuardError(f"{path}: its locked text no longer matches its pin")
    for path, segments in pins.LOCKED_AMENDMENTS.items():
        data = (root / path).read_bytes().replace(b"\r\n", b"\n")
        for start, length, digest, _commit in segments:
            if hashlib.sha256(data[start:start + length]).hexdigest() != digest:
                raise GuardError(f"{path}: a pinned amendment no longer matches its pin")


def verify(root: Path = ROOT) -> Authorisation:
    """Refuse unless the owner's scoring authorisation is recorded, pinned, and matches this code (see the module)."""
    plan_bytes = (root / PLAN).read_bytes().replace(b"\r\n", b"\n")
    record = recorded_authorisation(plan_bytes.decode("utf-8"))
    if record is None:
        raise GuardError("no scoring authorisation is recorded in the plan: the evaluation may not run")
    check_pins(root)
    pins = _pins(root)
    start, length, _digest, _commit = pins.LOCKED_AMENDMENTS[PLAN][-1]
    if len(plan_bytes) != start + length or AUTH_BEGIN.encode("utf-8") not in plan_bytes[start:]:
        raise GuardError("the scoring authorisation is not inside the plan's last pinned segment")
    if _git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise GuardError("the working tree has changes to tracked files")
    commit = record["authorised_commit"]
    if _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise GuardError("the authorised commit does not exist in this repository")
    if _git(root, "merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        raise GuardError("the authorised commit is not an ancestor of HEAD")
    if _git(root, "diff", "--quiet", commit, "HEAD", "--", *CODE_FILES).returncode != 0:
        raise GuardError("the scoring code changed after the authorised commit")
    fingerprint = code_fingerprint(root)
    if fingerprint != record["code_fingerprint"]:
        raise GuardError("the scoring code's fingerprint differs from the authorised one")
    return Authorisation(commit, fingerprint, record["recorded_utc"])
