"""Make the project root importable in tests (so `import src...` works without installing), and guard the logs."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.log_guard import GUARDED_LOGS, changed_logs, log_digests  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def historical_logs_are_never_written():
    """Fail the session if any test changed the production or development log (tests write under tmp_path)."""
    before = log_digests([ROOT / p for p in GUARDED_LOGS])
    yield
    changed = changed_logs(before)
    assert not changed, f"The test suite modified historical experiment logs: {changed}"
