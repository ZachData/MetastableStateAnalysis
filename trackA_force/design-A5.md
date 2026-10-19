<!-- trackA_force/design-A5.md -->
# Track A, A5 — generality on the PolyPythias seeds: registration text (drafted 2026-10-18, decided 2026-10-19)

**Status: text decided 2026-10-19 (Blocked 68, the user's: D1–D9 all as recommended, listed
below). Not registered; no seed's intermediate step downloaded or read.** Next before registering:
the calibration gate on seed 0 ("Before it can be registered" 2), which is A5's unit 1.
Registering is the user's (`CLAUDE.md` "Literature scans and the registry"); this file proposes
the text and lists the choices in it. ROADMAP §6 A5, §10.7 and §10.9. Scan: `lit-C2.md`.
Numbers: `tools/math_checks/a5_seed_location.py` (3 checks pass), which owns every p and E below.
Nine choices (D1–D9); revised after `/challenge-pr` on #227.

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

Revised 2026-10-18 after `/challenge-pr` on #227 (findings 1–5 taken): D4's reference, D5's
asymmetry and its (c) corrected, D6's label, D7 decided from the known outliers, D9 added, the
calibration gate named.

**Decided 2026-10-19 (Blocked 68, the user's, all as recommended):** D1 (a), D2 (a), D3 (a),
D4 (a), D5 (a), D6 (b), D7 (a), D8 (a), D9 (a), so A5 is labelled a reproducibility check, not a
test of mechanism. The draft entry below is the decided text. It goes back to the user (not edited
here) if any calibration-gate arm fails, including the known answer not reproducing the panel's
counts, or if A3, which runs first, dates the break with another energy or statistic than D2 and D4.

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | what is registered | (a) the break's location, seed by seed, counted (K of 9); (b) within-seed coupling of break and induction onset by `CLAIM-B`'s pairing null over seeds; (c) a conjunction: monotone at step 0 in every seed *and* the break in the window | **(a).** (b) has no power if the seeds agree at grid resolution, and Pythia has no step between 512 and 1000, where induction's previous-token half switches on; keep it descriptive, beside. (c) needs its combination fixed and step 0 is near-certain; report it beside |
| D2 | the energy | (a) β = 1 on the unit sphere, `core.metrics.interaction_energy`, a violation where (E_prev − E_curr)/\|E_prev\| > 1e-3 (`energy_violation_severity`), summed over layers and prompts, as the panel's `p1_energy_nviol_b1` was built; (b) β = 3.5 in the LN1 frame (Blocked 9's β); (c) both, as two entries | **(a)**, the only definition seed 0 has been read with, so the window comes from data. (b) is the project's β but is unread on seed 0: if wanted, run it on seed 0 first (free, seed 0 is the development set) and take its window from that. (c) puts two near-duplicate entries under one claim; only one could enter its product |
| D3 | prompts | (a) the 7 prompts of `build_colocation_panel.META_PROMPTS` (= the v1 scored set; hash pinned at run time); (b) add the 12 held-out 410m prompts; (c) Phase 1's 8 metastability prompts | **(a)**: the set seed 0's numbers come from. (b) spends 410m's confirmation set for no gain here; the seeds are already held out |
| D4 | the break statistic | (a) half-rise: the first grid step with V ≥ half the median of V over the 1000–4000 plateau; (a′) the same against V(143000); (b) first step with V > 0; (c) `CLAIM-B`'s change-mass centroid | **(a)**: seed 0 at 512 (plateau median 57). (a′) also gives 512 on seed 0 (70 at 143000), but it ties an early-location claim to late training: a seed whose count keeps climbing late, or spikes late, reads a later break with the same onset (#227 finding 2). (b) puts seed 0 at 256 but one stray violation moves it (Phase 2's own count reads 1–2 at steps 0–4). (c) puts seed 0 itself at about 830, outside the window, pulled late by the 16000–32000 rise (the check) |
| D5 | the window | (a) the step is 256 or 512, i.e. the break in (128, 512]; (b) 512 only; (c) (128, 1000] | **(a)**, ROADMAP's wording. It is not symmetric about seed 0: interpolated in log step, seed 0 crosses at about 306 (345 against 143000), so a seed may break about 2.4× earlier and still count but only about 1.7× later, and a crossing just past 512 reads 1000 because the grid has nothing between (#227 finding 3). (c) removes that grid edge at a price: p0 = 3/12, and E ≥ 20 needs K ≥ 8 of 9 (not "near-trivial", as the first draft said). (b) is sharper, p0 = 1/12 |
| D6 | the null's p0 (K ~ Bin(9, p0) under H0) | (a) 2/17: the step uniform over the grid's 17 non-zero steps; (b) 1/6: uniform over the 12 non-zero steps up to 4000; (c) 1/3: a "moves early" null | **(b)**: it grants that the project's transitions all sit before 4000 (`CLAIM-B`'s "everything moves early" objection) and asks only whether the step is shared inside that region. E ≥ 20 at K ≥ 7 (E 52; K = 6 gives 15). Under (a) K = 6 suffices (E 39); under (c) only K = 9 (E 70). **Whichever is taken, E is quoted as "against a uniform-location null"**: nobody holds that null, so a large E here is cheap, and the entry leads with its falsifier (#227 finding 4) |
| D7 | which seeds count | (a) all 9, seeds 3 and 4 flagged; a seed that fails to load or has a zero plateau counts as outside; (b) seeds 1, 2, 5–9 scored (7), seeds 3 and 4 read beside | PolyPythias already names **410m seeds 3 and 4 as outliers** (loss spikes, ≥ 2 SD below the mean downstream; `p1d_cluster_ensemble/lit-1d.md` §10 row 1, `design-1d.md`, which kept and flagged them), so this is decided now, not after a read (#227 finding 1). **(a), narrowly.** The claim is that the recipe sets the step, and outlier runs are the recipe too; D4 (a)'s plateau reference keeps a late spike from moving the break. The cost: if both outliers miss, the other 7 must all land (K = 7, E 52). (b) is pre-specified on a criterion independent of the energy and needs K ≥ 6 of 7 (E 44; K = 5 gives 11). Take (b) if you want the test to lose only for an energy reason |
| D9 | what kind of first adjudication this is | (a) a reproducibility check, labelled so: it can say the dated break is not seed 0's, and is cheap to pass against D6's null; (b) make it riskier first (D5 (b), 512 only); (c) let another row carry the first adjudication (`P-ST1`, Blocked 41 (a)) and run A5 as C2's seed-robustness check | **(a)**, because C2 needs to know whether its date is seed 0's, and A5's value is that one fact. But (a) is the project's first adjudication only in name: it tests no mechanism. If "first adjudication" should mean a hypothesis that could lose on its merits, (c) (#227 finding 4) |
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
| statement | In each of the 9 PolyPythias `pythia-410m-seed{1..9}` runs, the energy break (the first of the 18 grid steps at which the layer-monotonicity violations of E_1, summed over the 7 v1 prompts, reach half their median over steps 1000, 2000 and 4000) is step 256 or 512. Seeds 3 and 4, PolyPythias's named outliers, count and are flagged. |
| falsifier (leads) | K ≤ 3 of 9 (E ≤ 1.2): the step is seed-specific, and C2 dates the break for seed 0 only. K = 4–6: insufficient, reported as such. Any E is quoted as "against a uniform-location null". |
| h0 | The break's step is the run's: across seeds it falls anywhere among the 12 non-zero grid steps up to 4000, each with probability 1/12, independently. |
| h1 | The break's step is the recipe's: most seeds' fall in (128, 512], as seed 0's does. |
| instrument | a new producer `trackA_force/a5_seeds.py`: forward passes on the 18 steps per seed, E_1 per layer, `energy_violation_severity`, the half-rise step; gate `p_value_p_a5` = P(Bin(9, 1/6) ≥ K) |
| null_construction | Exact binomial over seeds, independent by construction (separate runs). Not a test of mechanism: all seeds share the LR schedule (warmup to about step 1430), so a high K cannot separate a training event from the schedule. A seed that fails to load or has a zero plateau median counts as outside. |
| calibration_record | the calibration gate's record (below), before registration |
| notes | Descriptive, not scored: per-seed induction onset (7d's score) and its order against the break; violations at step 0 in every seed; the β = 3.5 LN1 reading. Input set at run time: prompts hash, code hash, seed list, revisions. |

## Before it can be registered, in order

1. Your calls on D1–D9: **done 2026-10-19**, all as recommended (Blocked 68).
2. **The calibration gate** (`CLAUDE.md` "Calibration gate"; free, seed 0 is the development
   set), every arm before registering:
   - *known answer:* the new producer on `pythia-410m` reproduces the panel's counts (0 through
     128, 19 at 256, 56 at 512, 57 as the plateau median) and half-rise 512;
   - *planted effect:* seed 0's series moved one grid step later reads 1000, outside;
   - *planted null:* 9 synthetic series with crossings drawn from h0, through the producer's
     statistic, give K's distribution as Bin(9, 1/6) (this checks the code, not the null);
   - *sensitivity:* half-rise on seed 0 at rel_tol 1e-4, 1e-3, 1e-2, and leave-one-prompt-out.
     Phase 2's count sits within half a violation of its own threshold at step 256 (#227
     finding 5), so if either moves seed 0 out of the window, D2's tolerance or D4 is revisited
     before registering.
3. You register (the registry commit is yours); then the seeds are downloaded (about 131 GB for
   16 steps × 9 seeds at 0.91 GB, `HDD_1TB` had 265 GB free on 2026-10-18, ROADMAP §6).
