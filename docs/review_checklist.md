# Review checklist: branches, pull requests, and what to check

Prepared 2026-10-01 and updated the same day after publication. The five branches below are on GitHub as draft pull
requests #26–#30, stacked A → E. **Nothing has been merged.** Merging and deleting branches are the owner's decisions.

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

## 2. Pull-request boundaries

| PR | Head → base | Scope | Log changes | Reviewer focus |
|---|---|---|---|---|
| A (#26) | `v1.1-ceftriaxone` → `main` | Version 1.1: ceftriaxone dataset option, strict pre-write gate, test parts scored once | +24 production rows (append only) | the gate's dataset-aware key (`83df31c`); the pair-rule shortfall is stated; the Weis 0.74 attribution in `tables.md` is unverified (report 8.1) |
| B (#27) | `v1.2-calibration` → `v1.1-ceftriaxone` | Version 1.2 development study and its closure | development log created | pool derivation excludes spent rows *and* their patients; `tables_complete.md` vs `tables.md` |
| C (#28) | `v1.3-threshold` → `v1.2-calibration` | Version 1.3 study, the test fix, the closure | +8 development rows | **first CI run under `shell: bash` (`-eo pipefail`) on GitHub**, on both OSes; the session log guard in `tests/conftest.py` |
| D (#29) | `v1.4-screening` → `v1.3-threshold` | Discrimination audit, the `--keep-workstation` builder option, Version 1.4 | +10 development rows (1 failed run, 9 results) | screening rows are training-only; A0 = V1.2 arm C reproduced exactly; the count addendum |
| E (#30) | `v1.4-report` → `v1.4-screening` | This report, evidence map, reproduction guide, summary, checklist; README corrections | none | every figure against its cited artifact; reference verification notes |

**Merge method: merge commits, in order A → E (none merged yet).** A merge commit keeps the PR's original commits, so
the Version 1.1–1.4 commit IDs that plans, records and logs cite (`b6016a3` … `0806e74`) become part of `main`'s
history. A squash or rebase merge would replace them, and they would then resolve only through the archive
(section 5). Branches are not deleted on merge, so GitHub does not retarget the stacked PRs. After each merge:
1. retarget the next PR to `main` (`gh pr edit <number> --base main`); its diff should then list only its own commits;
2. close and reopen it, because a base change alone does not start the `pull_request` workflow, and wait for CI;
3. complete section 3, mark it ready, and merge it with a merge commit (`gh pr merge <number> --merge`).

**Commits that are not green on their own** (expected; with merge commits they enter `main`'s history, so skip them
when bisecting): the four "Record the protocol" commits (`b6016a3`, `db48bff`, `2242327`, `5600efb`) fail
`test_every_plan_in_docs_is_locked` until the next "Pin" commit; `1a00206` had two failing tests, fixed in `5f833b4`.
Every branch *tip* passes.

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

- **Version 0.7–0.9 originals: preserved (done 2026-10-01; section 5)**, locally by archival refs and on GitHub by
  annotated `research-archive/…` tags. They had been reachable only through the reflog. The off-machine copy of the
  bundle is **pending**.
- **Old remote branches** `origin/v0.9.2-followups` and `origin/v1.0-result-page` are merged and can be deleted by
  the owner (branch deletion is not done by the assistant). Their tips are archived as
  `research-archive/v0.9.2/followups-tip` and `research-archive/v1.0/result-page-tip`, so deleting them loses no
  commit.
- **DRIAMS-C.** No procedure warrants opening its ceftriaxone labels (report, section 12). Any future use needs a
  dated, owner-approved amendment.
- **Model iteration** on the ceftriaxone development pool has stopped by the Version 1.4 plan's rule. A new study needs
  a new pre-registration and new data (report, section 12).

## 5. Archive, and how recorded commits stay resolvable

**Archival refs, published as tags (2026-10-01).** 38 local refs under `refs/archive/` point at every research
milestone that was at risk or held only by a branch pending deletion:
- the V0.7, V0.8 and V0.9 chains, previously reachable only through the reflog;
- the pre-rebase V0.2 history;
- the run and plan commits of V0.2–V0.6;
- the tips of `v0.9.2-followups`, `v1.0-result-page` and V1.1–V1.4;
- the report.

Each was checked against its expected commit subject before it was written, and no branch was moved. On GitHub, each
is an **annotated tag** with the same name under `research-archive/`, pointing at the same commit:
`refs/archive/v0.8/approved-protocol` is published as `research-archive/v0.8/approved-protocol`. A fresh clone from
GitHub was compared with this clone on 2026-10-01: 38 tags, all annotated, none pointing elsewhere. The prefix is not
`archive/`, because a tag `archive/x` would share its short name with the ref `refs/archive/x`, which Git would
resolve ambiguously.

**The refs and tags are fixed snapshots: never move or delete them.** `research-archive/v1.4-report/tip` (like
`refs/archive/v1.4-report/tip`) points at `bf832bd`, this branch's tip when the archive was made. Later commits on
this branch, including this correction, are held by the branch and PR #30, not by that tag.

**Every recorded commit resolves on GitHub through `main` or a tag.** The 84 distinct commits cited by tracked files
of this branch (plans, records, logs and this report) were checked on 2026-10-01 against a fresh clone: each is
reachable from `main` or from a `research-archive/…` tag, so deleting any other branch would orphan none of them.

**Fetch the tags and restore an archived commit.** Each command was tested on 2026-10-01 in a temporary clone. On
Windows, deep folders need `core.longpaths` (reproduction guide, section 5).

```bash
# A fresh clone gets all 38 tags:
git clone https://github.com/Sickwoman/antibiotic-resistance-ai.git
# In an existing clone, a plain `git fetch` brings only the tags whose commits are on a GitHub branch
# (10 of 38 on 2026-10-01). Fetch all of them:
git fetch origin 'refs/tags/research-archive/*:refs/tags/research-archive/*'
git tag -l 'research-archive/*'          # lists 38 tags

# Read the Version 0.8 protocol as approved, and compare it with the merged text (report, section 8.8):
git show research-archive/v0.8/approved-protocol:docs/v0.8_adaptive_plan.md
git diff research-archive/v0.8/approved-protocol main -- docs/v0.8_adaptive_plan.md
# Check out an archived commit in a separate, detached worktree (your own checkout is untouched):
git worktree add --detach ../v0.8-approved research-archive/v0.8/approved-protocol
# Or start a branch from it:
git switch -c restore/v0.7-pre-registration research-archive/v0.7/pre-registration
```

**Bundle.** `archive/research-history-2026-10-01.bundle` (git-ignored; SHA-256 `c913048a…`, recorded in the
`.sha256` file beside it) holds the 18 local branches, 6 remote-tracking refs and 38 archival refs as they stood on
2026-10-01. It predates the tags, so it holds `refs/archive/*`, not `research-archive/…`. It survives reflog expiry
and accidental branch deletion in this clone. It also holds 17 commits that are on no GitHub branch or tag, on nine
local-only branches such as `v0.5-deep-learning`; no tracked file cites them. **The off-machine copy is pending.**
Until the bundle and its checksum file are copied off this machine, those 17 commits exist only here. To verify a
copy and restore from it (tested 2026-10-01):

```bash
sha256sum -c research-history-2026-10-01.bundle.sha256    # in the folder holding both files; prints "OK"
git init verify-tmp                                        # `git bundle verify` needs a repository; any will do
git -C verify-tmp bundle verify /path/to/research-history-2026-10-01.bundle   # "is okay", 63 refs
git clone --mirror /path/to/research-history-2026-10-01.bundle restored.git   # every ref, as archived
git fetch /path/to/research-history-2026-10-01.bundle 'refs/archive/*:refs/archive/*'   # in an existing clone
```

**Why recorded IDs stay valid.**
- Merging A–E with merge commits (section 2) puts their original commits in `main`'s history. A squash or rebase would
  not, but the tags would still resolve them. Cited commits of earlier versions, which reached `main` by squash,
  resolve through the tags either way.
- The protocol pins in `tests/test_preregistration.py` check file **content** (length and SHA-256 of the locked text),
  not commit ancestry, so no merge that leaves a plan's text unchanged can invalidate them. The commit named beside
  each pin is informational and resolves through `main` or the tags.
- The production and development logs record commit IDs as text, so no merge method alters them. Only their
  resolvability depends on the archive.

**Is the five-PR structure coherent? Kept, with one note.** Each PR is one version's pre-registered study, its run,
and its documentation, in dependency order. Every log append happens in exactly one PR. PR C also carries
repository-wide test automation (CI shell, session log guard), because the V1.3 closure commit introduced it.
Splitting it out would rewrite recorded commit IDs, so it stays and is called out in that PR's description.
