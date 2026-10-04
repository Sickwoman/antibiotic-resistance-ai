"""Pre-registrations may be appended to, never edited where they stand.

Every plan in docs/ was committed before its version's code, and the evaluation protocol says it "changes
only through a dated amendment at the end of this file". That rule used to be a convention, and a sweep on
2026-09-29 found it broken three times — twice in the protocol, and once in the Version 0.9 plan, where a
documentation pass rewrote a row of the locked-items table. Nothing caught it, because nothing checked.

This test checks. Each document's locked text is pinned by its length and SHA-256, and the file on disk must
still begin with exactly those bytes. Anything may be added *after* the locked text — that is what addenda
are — but a change anywhere inside it fails here. An addendum can in turn be pinned as its own segment
(LOCKED_AMENDMENTS), so recording it never changes the earlier pin.

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
# to it the same way. Since 2242327 the pin runs through amendment 10 (Version 1.3), since 5600efb through
# amendment 11 (Version 1.4), and since 186a9d1 through amendment 12 (Version 2.0).
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
    "docs/v2.0_marisma_plan.md": (
        15422, "d49189da36112e6ed131a8be549dded56c0c80d9f0c7441107bf25ff56d12859", "186a9d1"),
    "docs/evaluation_protocol.md": (
        39768, "94de3fe0f58bb9c356999377c4d359a54a1ab17ea0f3ebc91ed302259b62438d", "186a9d1"),
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


# Dated amendments recorded after a document's locked text are pinned separately, so that the earlier pin stays
# exactly as it was locked: path -> [(start, length, SHA-256, commit), ...] in file order. The first amendment starts
# where the locked text ends, and each later one where the previous amendment ends. Since 873aed5: Version 2.0 plan
# amendment A and protocol amendment 12, note A; since 886076a: plan amendment B and note B (pre-data
# clarifications); since 62deee6: the owner's approval record of steps 1-3 and note C; since 1ee6935: plan amendment C
# and note D (deviation record, sample-source access, coverage investigation); since 6a5ea90: plan amendment D and
# note E (the owner's decisions on the step 3 blockers).
LOCKED_AMENDMENTS: dict[str, list[tuple[int, int, str, str]]] = {
    "docs/v2.0_marisma_plan.md": [
        (15422, 19573, "f8cee0a947f7ba15cec6ebdb1040f8fbc44b52aaf84ebf0e22384d5d3e3589dd", "873aed5"),
        (34995, 5652, "c8204365e7c9b84f23fd07f65132a11cefb20d27e61047d30394c81e8d9c1ca5", "886076a"),
        (40647, 1830, "b46234a911e716c725487bc6d03c3d9662585ed6a444b77e912b9fbe3bafbfa4", "62deee6"),
        (42477, 8246, "145dca7b5a2033b74c5f8d9d01d3b0dc6be5c48eeb29bb378a27700373146e53", "1ee6935"),
        (50723, 7667, "9eda7a39e307d7b6d7857a635f02ad773c171c249e4ee3b850d9147fdfe38b87", "6a5ea90")],
    "docs/evaluation_protocol.md": [
        (39768, 1142, "e7e09fd41721eded12b019e42455358c7593a8a771fdda0728d7aa0fd3c8c197", "873aed5"),
        (40910, 1216, "68e495810e09ae281ecbb15f9c8508f1dc69446a72f7fd9b2ae872ecef871969", "886076a"),
        (42126, 820, "4f46bef1572162c9336cddd088e4aa34392fc468ef8ec3fe1e6cb008dedc10be", "62deee6"),
        (42946, 970, "e801715d98445ecb14d9375c471f3b8e04cddb6cee33726fff4e8b528b5063c8", "1ee6935"),
        (43916, 1551, "d3cbab8fecde7102a548ca3e07cb792d86b6047bfee2f0d33bf8a50559dac972", "6a5ea90")],
}
AMENDMENT_CASES = [(path, i) for path in sorted(LOCKED_AMENDMENTS) for i in range(len(LOCKED_AMENDMENTS[path]))]


def locked_segment(path: str, start: int, length: int) -> bytes:
    data = project_path(path).read_bytes().replace(b"\r\n", b"\n")
    assert len(data) >= start + length, f"{path} is shorter than its pinned amendments: something was deleted from it"
    return data[start:start + length]


@pytest.mark.parametrize(("path", "index"), AMENDMENT_CASES)
def test_each_pinned_amendment_is_unchanged(path: str, index: int):
    start, length, digest, commit = LOCKED_AMENDMENTS[path][index]
    actual = hashlib.sha256(locked_segment(path, start, length)).hexdigest()
    assert actual == digest, (
        f"{path} no longer holds, from byte {start}, the amendment pinned at {commit}. An amendment, like the text "
        f"it amends, may only be followed by a later dated entry, never edited in place.")


def test_pinned_amendments_follow_the_locked_text():
    """Each pinned amendment starts exactly where the locked text, or the amendment before it, ends."""
    for path, segments in LOCKED_AMENDMENTS.items():
        assert path in LOCKED, f"{path} has a pinned amendment but no locked text"
        end = LOCKED[path][0]
        for start, length, _, _ in segments:
            assert start == end, (
                f"{path}: an amendment pinned at byte {start} does not start where the text before it ends")
            end = start + length
