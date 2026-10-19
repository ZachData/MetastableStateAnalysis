<!-- trackA_force/design-A5.md -->
# Track A, A5 — generality on the PolyPythias seeds: registration draft (2026-10-18)

**Status: DRAFT. Not frozen, nothing registered, no seed's intermediate step downloaded or read.**
Registering is the user's (`CLAUDE.md` "Literature scans and the registry"); this file proposes
the text and lists the choices in it. ROADMAP §6 A5, §10.7 and §10.9. Scan: `lit-C2.md`.
Numbers: `tools/math_checks/a5_seed_location.py` (3 checks pass), which owns every p and E below.

## What A5 is for

The energy break (E_β stops being monotone across layers) is dated on one run, seed 0, at
128–512 (`p1_mstate_tracking/status-1.md` "The developmental arc"; 0 violations through 128,
19 at 256, 56 at 512, 70 at 143000, `data/analysis/colocation_panel.csv`). A5 asks whether that
step belongs to the run or to the recipe: the 9 PolyPythias 410m seeds (same code, data and
schedule; different init and data order) were never read here between step 0 and 143000, so a
prediction registered before they are downloaded is the project's first held-out test.

**What it cannot show (stated in the entry):** every seed shares Pythia's LR warmup (to about
step 1430), so a high count says the step is not seed 0's; it does not say the step is not the
schedule's (`lit-C2.md` §3).

**Exit condition:** K (below) and its E reported on the 9 seeds, or a recorded reason they
cannot be. **Budget: 3 units + close-out**: (1) the producer and its seed-0 dry run, (2) the
download and the batch (`runner`), (3) the reading.

## Decisions (each yours; recommendation first)

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | what is registered | (a) the break's location, seed by seed, counted (K of 9); (b) within-seed coupling of break and induction onset by `CLAIM-B`'s pairing null over seeds; (c) a conjunction: monotone at step 0 in every seed *and* the break in the window | **(a).** (b) has no power if the seeds agree at grid resolution, and Pythia has no step between 512 and 1000, where induction's previous-token half switches on; keep it descriptive, beside. (c) needs its combination fixed and step 0 is near-certain; report it beside |
| D2 | the energy | (a) β = 1 on the unit sphere, `core.metrics.interaction_energy`, a violation where (E_prev − E_curr)/\|E_prev\| > 1e-3 (`energy_violation_severity`), summed over layers and prompts, as the panel's `p1_energy_nviol_b1` was built; (b) β = 3.5 in the LN1 frame (Blocked 9's β); (c) both, as two entries | **(a)**, the only definition seed 0 has been read with, so the window comes from data. (b) is the project's β but is unread on seed 0: if wanted, run it on seed 0 first (free, seed 0 is the development set) and take its window from that. (c) puts two near-duplicate entries under one claim; only one could enter its product |
| D3 | prompts | (a) the 7 prompts of `build_colocation_panel.META_PROMPTS` (= the v1 scored set; hash pinned at run time); (b) add the 12 held-out 410m prompts; (c) Phase 1's 8 metastability prompts | **(a)**: the set seed 0's numbers come from. (b) spends 410m's confirmation set for no gain here; the seeds are already held out |
| D4 | the break statistic | (a) half-rise: the first grid step with V ≥ V(143000)/2; (b) first step with V > 0; (c) `CLAIM-B`'s change-mass centroid | **(a)**: seed 0 at 512. (b) puts seed 0 at 256 but one stray violation moves it (Phase 2's own count reads 1–2 at steps 0–4). (c) puts seed 0 itself at about 830, outside the window, pulled late by the 16000–32000 rise (the check) |
| D5 | the window | (a) the step is 256 or 512, i.e. the break in (128, 512]; (b) 512 only; (c) 128–1000 | **(a)**, ROADMAP's wording, seed 0 inside by both (a) and (b) of D4. (b) is sharper but seed 0's 512 vs 256 depends on D4; (c) is near-trivial against D6's nulls |
| D6 | the null's p0 (K ~ Bin(9, p0) under H0) | (a) 2/17: the step uniform over the grid's 17 non-zero steps; (b) 1/6: uniform over the 12 non-zero steps up to 4000; (c) 1/3: a "moves early" null | **(b)**: it grants that the project's transitions all sit before 4000 (`CLAIM-B`'s "everything moves early" objection) and asks only whether the step is shared inside that region. E ≥ 20 at K ≥ 7 (E 52; K = 6 gives 15). Under (a) K = 6 suffices (E 39); under (c) only K = 9 (E 70) |
| D7 | which seeds count | (a) all 9; a seed that fails to load or has V(143000) = 0 counts as outside; no exclusions; (b) exclude PolyPythias's named outlier runs, named before download | **(a)**, unless the full text of 2503.09543 names a 410m outlier (`lit-C2.md` Parked 2, before freezing); then decide (b) on the record |
| D8 | the claim, relevance | (a) H-RESIST ("trained weights resist the collapse dynamics"; the break is its founding observation, absent at init); (b) H-EMERGE, beside `CLAIM-B` | **(a)**. H-EMERGE is about co-location with circuit formation, which (a) of D1 does not test. Relevance on the rubric is yours (0.6 is the floor to enter the product) |

**Not a choice, fixed with the entry:** the 18-step grid (`trackA_force/a1a.py` `STEPS`; the
statistic reads steps up to the half-rise and 143000, the rest are descriptive); one-sided; the
p-value exact, P(Bin(9, p0) ≥ K); E by `core.evalues.calibrate` at κ = 0.5; GPU float32 for the
forward passes.

## The draft entry (with the recommendations taken)

| field | text |
|---|---|
| id | `P-A5` |
| claim | H-RESIST |
| statement | In each of the 9 PolyPythias `pythia-410m-seed{1..9}` runs, the energy break (the first of the 18 grid steps at which the layer-monotonicity violations of E_1, summed over the 7 v1 prompts, reach half their step-143000 count) is step 256 or 512. |
| h0 | The break's step is the run's: across seeds it falls anywhere among the 12 non-zero grid steps up to 4000, each with probability 1/12, independently. |
| h1 | The break's step is the recipe's: most seeds' fall in (128, 512], as seed 0's does. |
| falsifier | K ≤ 3 of 9 (E ≤ 1.2): the step is seed-specific, and C2 dates the break for seed 0 only. K = 4–6: insufficient, reported as such. |
| instrument | a new producer `trackA_force/a5_seeds.py`: forward passes on the 18 steps per seed, E_1 per layer, `energy_violation_severity`, the half-rise step; gate `p_value_p_a5` = P(Bin(9, 1/6) ≥ K) |
| null_construction | Exact binomial over seeds, independent by construction (separate runs). Not a test of mechanism: all seeds share the LR schedule (warmup to about step 1430), so a high K cannot separate a training event from the schedule. A seed that fails to load or has no violations at 143000 counts as outside. |
| calibration_record | the seed-0 dry run (below), before registration |
| notes | Descriptive, not scored: per-seed induction onset (7d's score) and its order against the break; violations at step 0 in every seed; the β = 3.5 LN1 reading. Input set at run time: prompts hash, code hash, seed list, revisions. |

## Before it can be registered, in order

1. Your calls on D1–D8 (any you leave take the recommendation, said on the record).
2. **Seed-0 dry run** (free; seed 0 is the development set): the new producer on `pythia-410m`
   reproduces the panel's counts (0 through 128, 19 at 256, 56 at 512, 70 at 143000) and
   half-rise 512; leave-one-prompt-out half-rise on seed 0 (if one prompt moves it out of the
   window, the window or D4 is revisited before registering). This is the known-answer gate.
3. `lit-C2.md` Parked 2 (outlier seeds), for D7.
4. You register (the registry commit is yours); then the seeds are downloaded (about 131 GB for
   16 steps × 9 seeds at 0.91 GB, `HDD_1TB` had 265 GB free on 2026-10-18, ROADMAP §6).
