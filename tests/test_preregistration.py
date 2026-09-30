"""Pre-registrations may be appended to, never edited where they stand.

Every plan in docs/ was committed before its version's code, and the evaluation protocol says it "changes
only through a dated amendment at the end of this file". That rule used to be a convention, and a sweep on
2026-09-29 found it broken three times — twice in the protocol, and once in the Version 0.9 plan, where a
documentation pass rewrote a row of the locked-items table. Nothing caught it, because nothing checked.

This test checks. Each document's locked text is pinned by its length and SHA-256, and the file on disk must
still begin with exactly those bytes. Anything may be added *after* the locked text — that is what addenda
are — but a change anywhere inside it fails here.

Pinned by digest rather than by `git show <commit>:path` on purpose: CI checks the repository out shallowly,
so the commits these baselines come from are not present there, and a history-based test would fail in CI
for a reason that has nothing to do with the documents.

Line endings are normalised to LF before hashing, because a Windows checkout may present CRLF and the
repository stores LF.
"""

from __future__ import annotations

import hashlib

import pytest

from src.utils import project_path

# path: (length of the locked text in LF-normalised bytes, its SHA-256, the commit it was locked at)
#
# The evaluation protocol was first pinned at e8d13fb, the last version before the 2026-09-29 record: two
# in-place edits made before then (at d3d4d9e and e8d13fb) corrected approved text and are disclosed in that
# record rather than reverted, since reverting would reinstate wording already found to be wrong. Since
# b6016a3 the pin covered the whole file through amendment 8, approved with the Version 1.1 plan, and since
# db48bff through amendment 9, recorded with the Version 1.2 plan; each longer pin begins with the earlier
# one's bytes unchanged, so no earlier guarantee is lost. When a later amendment is recorded, extend the pin
# to it the same way. Since 2242327 the pin runs through amendment 10 (Version 1.3), and since 5600efb through
# amendment 11 (Version 1.4).
LOCKED: dict[str, tuple[int, str, str]] = {
    "docs/v0.4_search_plan.md": (
        7109, "df3544c61d2d0ebf97338bcd19866aa673de7e68116c54d76a91f16e3e67d7a5", "ddeae53"),
    "docs/v0.5_deep_learning_plan.md": (
        5277, "df871c7aefbed8fbcb4390095f73ee266c5894be58971695143068f0c8dbdd57", "ae8ffde"),
    "docs/v0.6_explainability_plan.md": (
        8244, "db0ca8099d4f6d67fcc8af28e4ce8b7dff8a6e97e9307eaf93b89de6c30d0f09", "785e84c"),
    "docs/v0.7_generalisation_plan.md": (
        13673, "4b9987bdae3d6c654f04132131355975a37c4e2932f8ae9b44b57b2a8964507e", "ba7cb3e"),
    "docs/v0.8_adaptive_plan.md": (
        32203, "773b015b0eeb830e549285acb24cad2f7ad103b0a9e8e061fbd2345a53c2ac03", "ba7cb3e"),
    "docs/v0.9_api_plan.md": (
        11246, "f8f08edb5a1ebbd1e09591418a45f83549eb424f1e5f15f501f53e0486404d2b", "57b83fa"),
    "docs/v1.0_plan.md": (
        12030, "4ba8075eb5258c310ae78961d0518f1fd67f5a94aa190ce93efa9f98df22c3a0", "d1718cc"),
    "docs/v1.1_ceftriaxone_plan.md": (
        13983, "c1f8b265ca984178e4b54ea979942ee4acd87333d276a0bd81604b7a0ed7382c", "b6016a3"),
    "docs/v1.2_calibration_plan.md": (
        15384, "f0a7b4e64cb27d39e882fd1cd5d6d206a07bfa875a1c6bbd6770fb9328ee2dc3", "db48bff"),
    "docs/v1.3_threshold_plan.md": (
        16104, "ee8be88e641ee31595c1cf9fad51a1708430372b4e40f7f9bcc854be52639151", "2242327"),
    "docs/v1.4_screening_plan.md": (
        18244, "9c7ab5cfda3ce33cffa2f594388d6792743cddb448a555e9feed59c2670c032b", "5600efb"),
    "docs/evaluation_protocol.md": (
        38155, "093deec943a6dc26a54eb96d72fbf6f2bb283d7d44803970375d87528d751985", "5600efb"),
}


def locked_prefix(path: str, length: int) -> bytes:
    data = project_path(path).read_bytes().replace(b"\r\n", b"\n")
    assert len(data) >= length, f"{path} is shorter than its locked text: something was deleted from it"
    return data[:length]


@pytest.mark.parametrize("path", sorted(LOCKED))
def test_the_locked_text_of_each_plan_is_unchanged(path: str):
    length, digest, commit = LOCKED[path]
    actual = hashlib.sha256(locked_prefix(path, length)).hexdigest()
    assert actual == digest, (
        f"{path} no longer begins with the text locked at {commit}. Pre-registered text may only be followed "
        f"by a dated addendum, never edited in place. If a statement in it was wrong, correct it with an entry "
        f"after the locked text and leave the original standing.")


def test_every_plan_in_docs_is_locked():
    """A new version's plan must be added here when it is committed, or it is not protected at all."""
    plans = {p.relative_to(project_path(".")).as_posix() for p in project_path("docs").glob("v*_plan.md")}
    missing = sorted(plans - set(LOCKED))
    assert not missing, f"these plans have no locked baseline and could be edited silently: {missing}"


def test_the_check_would_notice_an_edit():
    """A guard on the guard: a one-byte change inside the locked text must change the digest."""
    length, digest, _ = LOCKED["docs/v1.0_plan.md"]
    text = bytearray(locked_prefix("docs/v1.0_plan.md", length))
    text[length // 2] ^= 0x01
    assert hashlib.sha256(bytes(text)).hexdigest() != digest
