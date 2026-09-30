# Local review checklist: branches, proposed pull requests, and what to check

Prepared 2026-10-01. **Nothing here has been pushed, and no pull request has been opened.** Every branch below exists
only in the local clone. Pushing, opening PRs and merging are the owner's decisions.

## 1. Branch dependencies

A linear stack on top of `main`. Each branch contains the one before it.

```
main (2f665e1, pushed; production log 89 rows)
 └─ v1.1-ceftriaxone  f4581cf  +12 commits  b6016a3..f4581cf   production log 89 → 113 rows
     └─ v1.2-calibration  3fc142b  +6 commits   db48bff..3fc142b   development log created (16 rows)
         └─ v1.3-threshold  c5cfb49  +7 commits   2242327..c5cfb49   development log 16 → 24; CI shell change
             └─ v1.4-screening  0806e74  +8 commits   4ecb130..0806e74   development log 24 → 34
                 └─ v1.4-report  (this review)  documentation only
```

## 2. Proposed pull-request boundaries

| PR | Head → base | Scope | Log changes | Reviewer focus |
|---|---|---|---|---|
| A | `v1.1-ceftriaxone` → `main` | Version 1.1: ceftriaxone dataset option, strict pre-write gate, test parts scored once | +24 production rows (append only) | the gate's dataset-aware key (`83df31c`); the pair-rule shortfall is stated; the Weis 0.74 attribution in `tables.md` is unverified (report 8.1) |
| B | `v1.2-calibration` → `v1.1-ceftriaxone` | Version 1.2 development study and its closure | development log created | pool derivation excludes spent rows *and* their patients; `tables_complete.md` vs `tables.md` |
| C | `v1.3-threshold` → `v1.2-calibration` | Version 1.3 study, the test fix, the closure | +8 development rows | **first CI run under `shell: bash` (`-eo pipefail`) on GitHub**, on both OSes; the session log guard in `tests/conftest.py` |
| D | `v1.4-screening` → `v1.3-threshold` | Discrimination audit, the `--keep-workstation` builder option, Version 1.4 | +10 development rows (1 failed run, 9 results) | screening rows are training-only; A0 = V1.2 arm C reproduced exactly; the count addendum |
| E | `v1.4-report` → `v1.4-screening` | This report, evidence map, reproduction guide, summary, checklist; README corrections | none | every figure against its cited artifact; reference verification notes |

**Squash merges and stacked branches.** The owner squash-merges on GitHub. After A is squash-merged, `main` holds A's
content in one new commit, so B (whose history still contains A's twelve commits) would show them again or conflict.
Rebase each next branch onto the updated `main`, keeping only its own commits:

```bash
git fetch origin
git rebase --onto origin/main f4581cf v1.2-calibration   # after A is squash-merged
git rebase --onto origin/main 3fc142b v1.3-threshold     # after B is squash-merged
git rebase --onto origin/main c5cfb49 v1.4-screening     # after C is squash-merged
git rebase --onto origin/main 0806e74 v1.4-report        # after D is squash-merged
```

The upstream argument is always the previous branch's **original** tip (the ID above), not its branch name, which
moves when that branch is itself rebased. Each merged squash commit has the same tree as the original tip, so each
replay applies cleanly. Rebasing rewrites commit IDs. The provenance records cite
the original IDs (`b6016a3` … `0806e74`), so **tag the originals before any rebase**, e.g.
`git tag archive/v1.1-ceftriaxone f4581cf`, and keep the tags. Alternatively, merge with merge commits (not squash) to
keep the recorded IDs reachable.

**Commits that are not green on their own** (expected, and harmless under squash merge): the four "Record the protocol"
commits (`b6016a3`, `db48bff`, `2242327`, `5600efb`) fail `test_every_plan_in_docs_is_locked` until the next "Pin"
commit; `1a00206` had two failing tests, fixed in `5f833b4`. Every branch *tip* passes.

## 3. Checks before each merge

- [ ] CI green on the PR head (Ruff + pytest, Ubuntu and Windows, Python 3.11 and 3.12).
- [ ] `tests/test_preregistration.py` passes: every plan's locked text and the protocol pin unchanged.
- [ ] `tests/test_privacy.py` passes: no identifier-like hash or UUID in any committed file.
- [ ] The production log's first 89 rows are byte-identical to `main`'s. Only PR A appends, and only 24 rows.
- [ ] The development log only grows. Rows 1–16 (B), 17–24 (C) and 25–34 (D) are unchanged by later PRs.
- [ ] No raw data, `X.npy`, model bundle or identifier is committed (`data/`, `models/` stay git-ignored).
- [ ] The served model is untouched: `models/v0.4/ecoli_ciprofloxacin/best_random.joblib` matches its sidecar
      `d59d6d7d…` (local check; the bundle is not in git).

## 4. Other items for the owner

- **Preserve the Version 0.7/0.8 originals.** `96d0425` (V0.7 pre-registration), `8878bfd` (V0.7 run), `64c6c75`,
  `284c0bd` (V0.8 approval), `4d82a22` (V0.8 run) and `849cad7` are reachable locally only through the reflog, which
  expires. They are held on GitHub by PR #11's ref. To keep them locally:
  `git fetch origin refs/pull/11/head:refs/pull/11` (or tag each commit).
- **Old remote branches** `origin/v0.9.2-followups` and `origin/v1.0-result-page` are merged and can be deleted by
  the owner (branch deletion is not done by the assistant).
- **DRIAMS-C.** No procedure warrants opening its ceftriaxone labels (report, section 12). Any future use needs a
  dated, owner-approved amendment.
- **Model iteration** on the ceftriaxone development pool has stopped by the Version 1.4 plan's rule. A new study needs
  a new pre-registration and new data (report, section 12).
