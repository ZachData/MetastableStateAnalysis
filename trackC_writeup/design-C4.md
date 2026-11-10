<!-- trackC_writeup/design-C4.md -->
# Track C — design: C4, the schedule intervention on `pythia-70m`

Opened 2026-11-09 by STATE Blocked 153 as revised: C7 (i) first, then C4's design. Opened by
Claude as the third of the user's "next 3 items on state, automatically". Row:
`docs/ROADMAP.md` §8 C4. Scan: `trackC_writeup/lit-C4.md`. Unit 0 is writing only; nothing ran,
nothing was downloaded, and nothing is registered. **Proposed; revised after `/challenge-pr` on
#344** (accept with changes, findings 1–6 taken). The reference is A0 on a shared dense grid, not
REF70; the instantaneous hypothesis is dropped on F; A0 trains first and always; the flat maps
have a math check; the warmup-stability scan is added. **Frozen 2026-11-09 on Blocked 155 all (a)**
(taken by Claude on the user's delegation, below); the registration (D4 (a)) is the user's, before
F trains past step 64.

## What C4 is for

C2's two dated events on Pythia, E₁'s break in V (256 or 512) and the token-centred cloud's
date (1000), sit inside the LR warmup (1430 steps). So F1 is open: are the dates fixed in steps,
or do they wait for a summed learning rate? C2 3b could not separate them on Pythia's grid. C6,
on the released `pythia-70m-warmup01` (ten times the ramp), labelled the centred date
"consistent with cumulative", but V's gate refused there. **C4 trains `pythia-70m` itself with
the ramp removed (a flat rate at Pythia's peak) and dates both events.** The three hypotheses
then predict dates an order of magnitude apart. It answers F1 for 70m, and reaches 410m (the
paper's model) only by inference.

## The predictions on the flat arm (fixed now; `tools/math_checks/c4_flat_maps.py`)

Pythia's ramp is η(t) = η̂ · t / W with W = 1430 and η̂ = 1e-3, so its summed rate to t ≤ W is
η̂ t² / 2W. **F runs at η̂ from the first update to step W, then follows Pythia's cosine from W
on**, so F and A0 differ only before W. F's summed rate to t ≤ W is η̂ t.

| hypothesis | maps A0's date t to F's | V (t ≈ 256) | centred (t ≈ 1000) |
|---|---|---|---|
| step | t | ≈ 256 | ≈ 1000 |
| cumulative LR | t² / 2W | ≈ 23 | ≈ 350 |

**The instantaneous-rate hypothesis is not a hypothesis on F** (`/challenge-pr` on #344, finding
3). F holds every rate ≤ η̂ from its first update, so the map is undefined. C2 3b and C6 already
exclude it for both dates.

**The reference is A0, not REF70** (#344, findings 1–2: REF70's ten-seed spread makes the
centred date's step and cumulative bands overlap at 512, and C6's functions run on Pythia's grid
alone). A0 shares F's init, data order and harness, and is read on **F's dense grid**, so its
dates t_A0 fix the predictions. The rule, fixed now:
- each arm's date is `a3.d4a`'s half-rise on its own plateau (1000–4000), read on the shared
  dense grid (below);
- the tolerance is ρ = max(1.25, the grid's largest ratio between neighbours near t_A0, A0's
  rerun ratio if one exists), with 1.25 placed;
- hypothesis h is consistent when F's date lies in [map_h(t_A0/ρ), map_h(t_A0·ρ)];
- the label is "consistent with step", "consistent with cumulative", "neither" or "both".

The two bands are disjoint whenever t_A0 < 2W/ρ³ (≈ 1464 at ρ = 1.25), which holds for both
dates; the math check proves the condition. Beside, not the label: C6's REF70 reading of F, and
seed 0's released dates as the harness check (R1).

## The arms

| arm | schedule | role | horizon |
|---|---|---|---|
| **A0** | Pythia's own: linear ramp to 1e-3 over 1430 steps, cosine to 0.1 × peak at 143 000 | the harness check: its dates must equal the released `pythia-70m`'s (V 256, centred 1000) within REF70's seed spread | 4000 |
| **F** | flat at 1e-3 from the first update, then Pythia's cosine | the intervention | 4000 |
| (W10) | ten times the ramp | **already trained and released** (`pythia-70m-warmup01`, C6); not retrained | n/a |

Common to both: init = the released `pythia-70m` step-0 weights (seed 0's init); data = the first
4000 × 1024 sequences of `EleutherAI/pile-standard-pythia-preshuffled` shard 0, in order; batch
1024 × 2049 by gradient accumulation; Adam (β 0.9, 0.95, ε 1e-8), weight decay **0.1** (decoupled,
not on biases or norms; *corrected in U1 from 0.01: `data/trackC/configs/pythia-70m.yml` says
0.1*), gradient
clip 1.0, Pythia's config; mixed precision on the 3080 (bf16 autocast, fp32 master; Pythia used
fp16, so not bit-equal). **Checkpoints, the same dense grid for both arms:** 0, 1, 2, 4, 8, 12,
16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1000, 1250, 1500,
2000, 3000, 4000. Neighbour ratios are ≤ 1.34 from 48 to 2000 (1.33 at 256/192, 1.30 at
1000/768, 1.33 at 2000/1500) and 1.5 at 3000/2000 (*corrected in U1: the design said ≤ 1.25
from 128 to 1500*; ρ uses the actual ratios, so the bands are as the rule computes them). Pythia's
own grid is a subset, so A0 is compared with the released run on it.

## The harness's known answers (U1, before any arm passes step 64)

| check | against | pass |
|---|---|---|
| H1 init | released step 0 | the loaded state bit-equal |
| H2 data | the first batch's tokens | equal to `batch_viewer.py`'s batch 0 |
| H3 first updates | released steps 2, 4, 8, 16 | the sign agreement of zero-initialised biases' increments ≥ 0.95 at 2 and 4 (C6's method: warmup01 against `pythia-70m` read 1.000 to 4); beside, every parameter's step-2 update sign (#344, finding 5) |
| H6 the counts | the released run's seed-0 records at Pythia's grid to 1000 | A0's V and centred counts per grid step, reported against the released run's before F launches |
| H4 loss | released `pythia-70m` at 1000 and 4000, on 64 held-out sequences | A0's loss within 2 % (placed; fp16 against bf16) |
| H5 the loop | `t5h`'s float64 known answer | exact |

## V's first check (Blocked 153's main risk)

V's gate refused on C6's warmup01 (plant B 0.38 against 0.4), so V must be shown readable on F,
the first non-Pythia arm. *Revised after #344, findings 4–5:* A0 trains first to 1000, the
harness check (H1–H6, cheap and early). Then F trains to 4000, and C6's gate (the same plants
and bar) runs at an off-grid step, 3500. Then A0 continues to 4000. **A0 is trained to 4000
whatever V's gate says**, because it is the reference for both readings. If V refuses on F, C4
reads the centred count alone (C6 U1b's form). F's first 64 steps' loss is looked at before F is
launched past them (Kalra & Barkeshli's catapult; a divergence is a refusal).

## Readings and outcomes (fixed now)

- **R1:** A0's two dates against the released `pythia-70m`'s, on Pythia's grid (harness). If
  A0's date falls outside the released date's grid interval, ±1 grid step, that is a refusal,
  and the flat arm is then not read.
- **R2:** F's centred date, labelled by the rule above against A0's.
- **R3:** F's V date, labelled the same way, if V's gate passed on F.
- **"Not formed by 4000"** (Qu & Liu found a fixed LR forming no induction heads) is a reading,
  not a refusal. It is consistent with none of the three, and says the warmup is needed for the
  event. A diverging loss (spikes over 2× the running median for 50 steps, placed) is a refusal.
- Beside: the loss curves, and the largest per-head induction score at each checkpoint (a
  repeated-random-token prompt, the score defined in U1 on the released `pythia-70m` before any
  arm is read), so "formed" has a second witness.

| outcome on F | what changes |
|---|---|
| "consistent with cumulative" | F1 narrowed for 70m: the dates wait for a summed rate; `paper/` §7 says so, scoped to 70m |
| "consistent with step" | the dates are training's own clock, not the schedule's, for 70m |
| "not formed by 4000" | the event needs the ramp; F1 is not answered, the warmup is causal for it |
| harness refusal | nothing is read; the harness is fixed or C4 closes |

## Beside: the cloud's concentration and effective rank across the ramp (added 2026-11-10, the user's)

Added while A0 trained to 1000, before any arm's count or cloud statistic was read and before F
existed. **Description only:** no label, no rule, not part of R1–R3 or `P-SCH1`, and it moves none
of them. No unit is added to the budget.

**Why.** The middle-state cloud rises, falls and rises again on Pythia-410m (peak at 16, floor at
2000–4000), Dodecahedron (floor at 671–917, inside its 1000-step warmup; second rise from
1253–1465, just after it) and partly on Mistral GPT-2 small (`PROJECT.md` §3.78, §3.80; on
Dodecahedron the warmup link is co-occurrence only). The sphere's effective rank is the same
picture seen from the other side (A3's E5, dated at 8 on 410m), and E₁ moves with the tokens'
mean (C2 5b). F removes the ramp with init, data and harness fixed, so it is the first run in the
project that can show whether the cloud's shape follows the schedule.

**What is read**, on every saved checkpoint of both arms (the dense grid and 3500), on v1's 7
prompts (the same inputs as C4's records and T5's ρ series), float32 eager:
- ρ_ℓ, ρ_mid, ρ_1:5 and ρ_mid⁻⁰ as `trackT_timaeus/t5.py` defines them (`state_stats`, `rho_mid`,
  `rho_15`); on 70m's 6 blocks ρ_mid is the median over states 1–5 and ρ_1:5 their mean;
- m_ℓ, the mean's share of the energy (`state_stats`);
- the sphere's effective rank per state, `core.metrics.effective_rank(h, mode="normed")`, mean
  over prompts, and its median over states 1–5 (er_mid). This is not A3's E5 producer (Phase 1's
  reports), so 410m's E5 values are compared by shape, not level.

Beside A0, the released `pythia-70m` on Pythia's grid to 4000, read the same way.

**What is described, per arm:** the steps of ρ_mid's peak, its half-fall from the peak to its
4000 value, and its minimum on the grid (censored if the minimum is at 4000); the same for er_mid
(its trough, half-rise and maximum). Beside each of F's steps, the step that the step map (t) and
the summed-rate map (t²/2W) give from A0's, as numbers, with no consistency call. Under the
summed-rate map any A0 date ≤ 53 (√(2W)) maps below F's first update, so an early peak would
sit at F's first grid steps.

**When:** in U3 with R1–R3, after the frozen rule has been applied to both arms, so nothing seen
here can shape how the registered reading is applied. The code goes in `c4_read.py` or a module
beside it, never `c4_train.py` (its `code_sha` is the arms'). Cost: one forward pass per checkpoint
and prompt, 29 × 7 per arm plus the released run's, minutes on the 3080.

**Not established by it:** cause; anything past one seed per arm; anything for 410m but by
inference; a reading on the prompt library (v1's 7 are kept for comparison with T5's and C4's own
records, against the library default for new designs).

## Carried in (from the ROADMAP row; not made in U1: `status-C.md` "C4 unit 1", Parked 0)

1. **C2 3c's Parked 0:** `c2_seeds.null_matched_tol` must refuse, not return a tolerance, when
   the null's uncentred rate is 0. A flat arm's early re-inits could hit this.
2. **T5 U3's Parked 3:** `c2_seeds.b_matched_tol` bounds τ at ≤ 1e-2 and reads a miss outside
   the bound as "no τ". Either the bound is widened or the refusal names the bound. Recommended:
   the refusal names it, which changes no earlier reading. A widened bound would be a new
   instrument for every arm read so far.

Not made (U1, 2026-11-09): C4 derives no centred arm (it reads 70m's four frozen τ), and
`c2_seeds.py` is hashed into `c6_warmup`'s and `c2_dose`'s code hashes, which C6 U1b's stored
records carry, so any edit stops them re-reading. Parked to the next unit that derives an arm.

## Cost and data

Data: shard 0 of the standard preshuffled Pile plus `document.idx`, about 30 GB (estimate),
onto `HDD_1TB` (128 GB free on 2026-10-20). Throughput: an estimate of about 50k tokens/s for
70m on the 3080 (half of `t5h`'s 104.8k for 32m), so about 47 h an arm to 4000, about 4 GPU-days
for both. U1 measures it before any arm is launched past step 64.

## Exit condition and budget

**Exit:** R1–R3 read (or a recorded refusal: a harness check missed, V refused, a divergence),
with the label written into `paper/` §7, scoped to 70m. **Budget: 4 units + close-out.**

| unit | what | done when |
|---|---|---|
| 0 | the scan and this design (writing only); the flat maps' math check | Blocked 155 decided |
| 1 | the harness (from `t5h`), the shard downloaded and verified, H1–H5, A0 to 1000 and H6 (about half a day), F's first 64 steps' loss, throughput measured (the two carried-in `c2_seeds` refusals not made, "Carried in"); **the rule's code frozen, and the reading registered by the user if D4 (a), before F trains past 64** | every check, or a refusal |
| 2 | F to 4000 (runner, about 2 days); V's gate at 3500 | F's records, V gated or refused |
| 3 | A0 from 1000 to 4000 (about 1.5 days); R1–R3 read | the labels |
| close-out | one page, `paper/` §7, Parked carried or dropped, the card | the card current |

## Decisions (each yours; recommendation first)

| # | decision | options | recommendation |
|---|---|---|---|
| 155 | C4's design and registration | **D1, the arms:** (a) F (flat at 1e-3) and A0 (Pythia's, the harness check); warmup01 already covers the long ramp; (b) as (a), plus F½, flat at 5e-4 (Qu & Liu's peak-dependence; about 2 GPU-days more); (c) F only, with the harness checked on the first steps alone. **D2, horizon:** (a) 4000, C2's plateau window; (b) 2000, half the cost, without the centred plateau. **D3, order:** (a) A0 to 1000 (the harness, H6), then F to 4000 with V's gate, then A0 to 4000; (b) A0 to 4000 first. **D4, registration:** (a) you register the reading (the A0-referenced rule on F, R2 primary) in `claims/registry.json` after U1 freezes its code and before F trains past step 64; (b) unregistered, tier 1 | **D1 (a), D2 (a), D3 (a), D4 (a).** D1 (a) is the smallest pair that separates step from cumulative with a harness check; (b) only matters if F forms nothing, and can follow then. D3 (a) checks the harness before F, and tests V's risk before A0's second half, at no extra cost. D4 (a): C4 is the one row whose answer could go either way, and it decides a sentence of the paper |

**Decided 2026-11-09: all (a)**, by Claude on the user's "do the next 4 items in state", as
recommended (as Blocked 154 was); yours to overrule. D4 (a) leaves the registration itself to the
user: F does not train past step 64 until it is in `claims/registry.json` (CLAUDE.md "Literature
scans and the registry").

## The rule's details, fixed in U1 before any arm's count was read (`c4_read.py`)

The design left these open; each is fixed in code and tested on planted series (`tests/test_trackC_c4.py`)
before A0's first record:

- **The date** is `a3.d4a`'s half-rise on the dense grid (plateau = median at 1000, 2000, 4000).
  A plateau ≤ 0 is "not formed by 4000"; a step-0 count ≥ half the plateau is unreadable. **92 (i)'s
  early check (any step ≤ 128 at half) is reported beside, not applied**: on F the cumulative
  hypothesis predicts V at ≈ 23, so applying it would refuse the very reading that hypothesis predicts.
- **ρ** = max(1.25, t / prev(t), next(t) / t) on the dense grid at A0's date (no A0 rerun exists):
  1.33 at 256, 1.30 at 1000. The bands stay disjoint at every grid step ≤ 1000 (tested).
- **m_cumulative(t) = t² / 2W**, the continuous map the math check proves; t > W refuses.
- **The centred count** is read at 70m's four frozen τ arms (`c2_dose.TAU_ARMS["70m"]`, as C6 read
  them; none is re-derived, so the `c2_seeds` refusals carried in are not needed here, see
  `status-C.md` "C4 U1"). As in C6, h is consistent when it is at every τ arm gated on F (at 3500)
  and dated on both arms; no such arm is "unreadable"; all gated arms not formed is "not formed by 4000".
- **R1** is per count and per τ arm, and refuses only on counts read (V if gated; gated τ arms).
- **The induction witness** (beside): 8 sequences of 64 random tokens repeated twice (seed
  20261109); a head's score is the mean attention from each second-copy position to the token
  after its first occurrence. On the released run the top head (L3 H6) reads 0.017 at 512 and
  0.93 at 1000.

## The draft entry (D4 (a): the user's to commit, before F trains past step 64)

Drafted by Claude in U1; nothing in `claims/registry.json` was touched. `tools/check_registry.py`
now accepts phase `C` (U1). The registration commit adds this entry and, in the same commit,
the line `r"|P-SCH[0-9]+"` to `ID_PATTERN` there (`P-C1`–`P-C4` are Phase 1d's, so the id is
`P-SCH1`; allowed earlier, the coverage check refuses the id until the entry exists). Validated
against a scratch copy of the registry: 0 errors. **The direction is a choice:**
h1 below follows C6's lean on `pythia-70m-warmup01` (the centred date "consistent with
cumulative"); registering step as h1 instead, or registering the label with no direction, are
the alternatives.

```json
{
  "id": "P-SCH1",
  "claim": "H-RESIST",
  "statement": "On pythia-70m trained from seed 0's init on Pythia's data order with the LR warmup removed (arm F: 1e-3 from the first update to step 1430, then Pythia's cosine), the token-centred count's date (R2) is consistent with the summed learning rate and not with the step count, against arm A0 (Pythia's own schedule, the same init, data and harness) read on the same dense grid.",
  "h0": "The centred date is fixed in steps: F's date lies in [t_A0/rho, t_A0*rho].",
  "h1": "The centred date waits for a summed rate: F's date lies in [(t_A0/rho)^2/2W, (t_A0*rho)^2/2W], W = 1430.",
  "falsifier": "The label is 'consistent with step' or 'neither'. 'Not formed by 4000' and 'unreadable' are reported and score nothing; a harness refusal (R1, H1-H6) or a divergence means nothing is read.",
  "instrument": "trackC_writeup/c4_train.py (the arms) and c4_read.py (records as C2 unit 3b's, float32 eager; the centred count at 70m's four frozen tau arms; the half-rise on the dense grid; rho = max(1.25, the grid's neighbour ratios at t_A0); h consistent iff consistent at every tau arm gated on F at step 3500 by C6's gate and dated on both arms); code hashes recorded at registration",
  "phase": "C",
  "experiment": "C4",
  "gate": "",
  "cost": "F",
  "evaluable": "measurement",
  "null_construction": "None: one seed per arm, a label against bands set by the grid, not by a measured run-to-run spread (no A0 rerun). The bands are disjoint for t_A0 < 2W/rho^3 (tools/math_checks/c4_flat_maps.py). 92 (i)'s early check is reported beside, not applied. V's date (R3) is read beside under the same rule if V's gate passes on F, and does not score.",
  "calibration_record": null,
  "real_run_record": null,
  "relevance": 0.6,
  "source": "trackC_writeup/design-C4.md (frozen 2026-11-09 on Blocked 155 all (a)); status-C.md 'C4 unit 1'",
  "registered_commit": null,
  "registered_date": null,
  "registered_provenance": "gated",
  "notes": "70m only; 410m by inference. Registered after A0's counts to 1000 were seen (H6, the harness check) and F's training loss to step 64 (the divergence check), before any F count. bf16 against Pythia's fp16. Input set at run time: the shard's sha256, the batch-0 hash, the prompts hash, code hashes.",
  "superseded_by": null,
  "status": "active"
}
```

`cost` copies `P-A5`'s code; check it against the registry's convention. `relevance` 0.6 copies
`P-A5`'s, as a row that decides one sentence of the paper.

## Worth challenging (for the PR)

- **A flat rate at Pythia's peak may diverge**, or form nothing (Qu & Liu, a blog on a small
  model). Divergence is a refusal and "not formed" a reading; D1 (b) is the hedge.
- **The harness is not Pythia's** (bf16 against fp16, one GPU against many, a different
  library). H1–H5 and R1 are the guard; R1 compares dates, not weights.
- **One seed per arm, and A0 as the reference** (revised after #344, finding 1). The tolerance
  ρ = 1.25 is the grid's, not a measured run-to-run spread. A second A0 seed (about 2 GPU-days)
  would measure it; without one, a date of F near a band's edge reads "neither". REF70's reading
  is beside.
- **70m only.** 410m's dates are reached by inference, as §7 must say.
- **The cost is an estimate** until U1 measures throughput. *U1 measured it:* 79k tokens/s at
  micro-batch 2 (larger micro-batches run out of the 3080's 10 GB on the 50k-vocabulary logits),
  26.5 s an update, about 29.5 h an arm to 4000, about 2.5 GPU-days for both.
