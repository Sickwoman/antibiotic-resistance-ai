# Erratum to `tables.md` — dated 2026-09-29

`tables.md` in this directory is **preserved exactly as it was generated** and is not rewritten. This file
records the one error in it. See issue #22.

## What is wrong

The **"Across the fitted seeds"** table reports two rows over six scorings while labelling them as five seeds:

| Experiment | Tested on | As printed in `tables.md` | Correct, over the five refit seeds |
|---|---|---|---|
| `external` | DRIAMS-B | Seeds 5 · 0.816 (0.809–0.824) | Seeds 5 · **0.817 (0.810–0.824)** |
| `external` | DRIAMS-D | Seeds 5 · 0.715 (0.704–0.728) | Seeds 5 · **0.717 (0.712–0.728)** |

The `temporal` and `external_ab` rows are correct as printed.

## Why it happened

`scripts/generalisation_tables.py` grouped the seed-variation rows by split and site only. The saved Version
0.4 model is scored at the same split and site as the refitted seeds, so it joined their group. Its seed is 42,
the same as the first refit seed, so the distinct-seed count still read 5 — while the mean, lowest and highest
were taken over all six rows, the saved model's included. At DRIAMS-D that lowered the minimum from 0.712 to
0.704, the saved model's own score.

The generator now groups by model as well, so the saved model forms a group of one and is left out, and it
refuses any group in which a seed appears twice, so a printed count can no longer disagree with the rows it
summarises. A regression test reproduces the original fault with these exact values and fails against the
old generator.

## What this does not change

**No Version 0.7 conclusion depends on these two cells.** The finding that no generalisation gap was
demonstrated at any site rests on `generalisation_gaps.csv` and `two_sites_versus_one.json`, and the per-site
results for the saved model are in `test_metrics.csv` and the append-only log. None of those is affected.

The README's Version 0.7 section has always carried the correct five-seed figures, and the Version 1.0
consolidated sections deliberately cite neither affected cell.

## Why this is an erratum and not a regenerated table

`tables.md` is a committed output of Version 0.7, and this project corrects its records by adding to them
rather than rewriting them, so that what was published and what was learned afterwards stay distinguishable.
The correct values above were recomputed from `test_metrics.csv` and are reproduced by the fixed generator.
