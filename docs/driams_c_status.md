# DRIAMS-C: what has been accessed, and whether it can still give an independent evaluation

**Audit date: 2026-09-30.** Written from the repository's committed records and the project's own working
history. **No DRIAMS-C file was opened for this audit**, and no DRIAMS-C ceftriaxone label was read.

## Status in one line

| Label | Independent evaluation at DRIAMS-C |
|---|---|
| Ciprofloxacin | **Unavailable.** Its labels fitted models in Version 0.8 and its protected part has been scored once. |
| Ceftriaxone | **Uncertain — conditionally possible, not clean.** No model, threshold or choice has used a single C ceftriaxone label, but their aggregate counts have been seen, the spectra were used for ciprofloxacin, C has no patient identifiers, and the project's own texts close C until the owner decides otherwise. |
| Any other antibiotic | Not assessed; aggregate counts for several pairs have been seen (below). |

## Timeline of access and decisions

| Date | What happened | Record |
|---|---|---|
| 2026-09-16 | DRIAMS-C registered with its Dryad SHA-256; not downloadable by script. | `config.yaml` → `driams.archives.C` |
| 2026-09-17 | Protocol: C to be added "the same way" once downloaded and the selection rule checked on it. | `docs/evaluation_protocol.md` section 8 |
| 2026-09-24 | C reserved for Version 0.8 ("used_in_v07: false"), before it was downloaded. | `config.yaml` → `reservation`; V0.7 addendum |
| 2026-09-24 | Version 0.8's methodology approved and hashed; **then** C downloaded in a browser, verified against its SHA-256, and extracted (`id`, `binned_6000`, `raw`). | `docs/v0.8_adaptive_plan.md` |
| 2026-09-24 | A **ciprofloxacin** cohort built at C: 927 *E. coli* rows, 889 spectra kept, 191 resistant, **none with a patient identifier** (built from `284c0bd`). | `results/metrics/v0.2/ecoli_ciprofloxacin__site-C/dataset_summary.json` |
| 2026-09-24 | That cohort partitioned once, seed 42, each spectrum its own group: adaptation 622 (120 resistant), protected 267 (71 resistant). | `results/metrics/v0.8/…/partition.json`; V0.8 plan 6.5 |
| 2026-09-24 | Version 0.8 run (`4d82a22`, logged 18:21; results committed as `849cad7`): the adaptation part's **ciprofloxacin** labels fitted a recalibration arm (A1), an A + C refit (A2) and A2's refitted zone; the protected part was **scored once** for five arms — prevalence, the saved model, the Version 0.7 external refit, A1 and A2. Recorded by the addendum of 2026-09-27: "It must never be scored again." | V0.8 plan addendum; `test_evaluations.csv` |
| 2026-09-28 | Amendment 8 (Version 1.1): "DRIAMS-C is not scored for any label" in Version 1.1. | `docs/evaluation_protocol.md` |
| **2026-09-30** | **While choosing Version 1.1's pair, the C metadata table was read to count isolates with a result for many species–antibiotic pairs — aggregate counts only, shown in the working session, never committed.** For *E. coli* + ceftriaxone at C: **916 isolates with an R/I/S result, 151 of them R or I (16.5 %)**, before any cohort rule. Counts for other pairs (for example *S. aureus* + oxacillin, 738 / 41) were seen the same way. | this audit |
| 2026-09-30 | Amendment 9 (Version 1.2): C stays closed; opening its ceftriaxone labels needs a separate owner-approved amendment. | `docs/evaluation_protocol.md`; V1.2 plan section 12 |

## By kind of information

- **Ciprofloxacin labels:** used to fit (adaptation part) and scored once (protected part). Spent.
- **Ceftriaxone labels:** never attached to a spectrum in any cohort, never used by a model, a threshold, a
  zone or a choice. **Seen in aggregate once** (916 with a result, 151 resistant, before cohort rules). That is
  the prevalence, not any isolate's label — but a later evaluation cannot claim that nothing about C's
  ceftriaxone labels was known.
- **Spectra:** the 889 *E. coli* spectra of the ciprofloxacin cohort were preprocessed; 622 of them were fitted
  on (the A + C refit) and 267 were scored. A ceftriaxone evaluation at C would largely re-use these spectra,
  as Version 1.1 re-used A, B and D's.
- **Patients:** C has **no patient identifier**. Every spectrum is its own group, repeated isolates from one
  patient cannot be detected, and intervals at C are sample-level and may be too narrow.

## What would have to be true before C could give an independent ceftriaxone evaluation

1. A dated amendment, approved by the owner, that reads the Version 0.8 reservation as covering ciprofloxacin
   only and reopens C for the ceftriaxone label alone.
2. That amendment records the aggregate ceftriaxone counts already seen (above), so the evaluation does not
   present itself as blind to them.
3. One procedure frozen and hashed before any C ceftriaxone label is attached to a spectrum, with its endpoints
   and success rules fixed in the same amendment, and one scoring through the strict gate.
4. The evaluation described as **label-independent but not spectrum-independent**, with sample-level
   intervals.

Until then, DRIAMS-C stays closed, and nothing in Versions 1.1–1.3 may be described as having been evaluated
there.
