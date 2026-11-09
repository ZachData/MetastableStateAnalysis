<!-- trackC_writeup/design-C4.md -->
# Track C — design: C4, the schedule intervention on `pythia-70m`

Opened 2026-11-09 by STATE Blocked 153 as revised: C7 (i) first, then C4's design. Opened by
Claude as the third of the user's "next 3 items on state, automatically". Row:
`docs/ROADMAP.md` §8 C4. Scan: `trackC_writeup/lit-C4.md`. Unit 0 is writing only; nothing ran,
nothing was downloaded, and nothing is registered. **Proposed; the decisions and the
registration are Blocked 155, the user's.**

## What C4 is for

C2's two dated events on Pythia, E₁'s break in V (256 or 512) and the token-centred cloud's
date (1000), sit inside the LR warmup (1430 steps). So F1 is open: are the dates fixed in steps,
or do they wait for a summed learning rate? C2 3b could not separate them on Pythia's grid. C6,
on the released `pythia-70m-warmup01` (ten times the ramp), labelled the centred date
"consistent with cumulative", but V's gate refused there. **C4 trains `pythia-70m` itself with
the ramp removed (a flat rate at Pythia's peak) and dates both events.** The three hypotheses
then predict dates an order of magnitude apart. It answers F1 for 70m, and reaches 410m (the
paper's model) only by inference.

## The predictions on the flat arm (fixed now; math check in U1)

Pythia's ramp is η(t) = η̂ · t / W with W = 1430 and η̂ = 1e-3, so its summed rate to t ≤ W is
η̂ t² / 2W. The flat arm runs at η̂ from the first update.

| hypothesis | maps a Pythia date t to the flat arm's | V (t = 256) | centred (t = 1000) |
|---|---|---|---|
| step | t | 256 | 1000 |
| cumulative LR | t² / 2W | ≈ 23 | ≈ 350 |
| instantaneous LR | the first step at which η reaches η̂ t / W | 1 | 1 |

The bands come from C6's frozen reading. `c6_warmup.REF70` holds 70m's dates over its ten seeds,
and each band is mapped by the same functions (`tools/math_checks/c6_bands.py` extended to
W_flat = 0 in U1). The label is C6's: the hypotheses whose mapped band contains the arm's date.
"Consistent with X" is the label when only X's band contains it; "neither" or "two" are labels
too.

## The arms

| arm | schedule | role | horizon |
|---|---|---|---|
| **A0** | Pythia's own: linear ramp to 1e-3 over 1430 steps, cosine to 0.1 × peak at 143 000 | the harness check: its dates must equal the released `pythia-70m`'s (V 256, centred 1000) within REF70's seed spread | 4000 |
| **F** | flat at 1e-3 from the first update, then Pythia's cosine | the intervention | 4000 |
| (W10) | ten times the ramp | **already trained and released** (`pythia-70m-warmup01`, C6); not retrained | n/a |

Common to both: init = the released `pythia-70m` step-0 weights (seed 0's init); data = the first
4000 × 1024 sequences of `EleutherAI/pile-standard-pythia-preshuffled` shard 0, in order; batch
1024 × 2049 by gradient accumulation; Adam (β 0.9, 0.95, ε 1e-8), weight decay 0.01, gradient
clip 1.0, Pythia's config; mixed precision on the 3080 (bf16 autocast, fp32 master; Pythia used
fp16, so not bit-equal). Checkpoints: A0 at Pythia's grid (0, 1, 2, 4, …, 512, 1000, 2000, 3000,
4000); F also at 12, 24, 48, 96, 192, 384, 768, 1250, 1500 (dense where the cumulative
predictions fall).

## The harness's known answers (U1, before any arm passes step 64)

| check | against | pass |
|---|---|---|
| H1 init | released step 0 | the loaded state bit-equal |
| H2 data | the first batch's tokens | equal to `batch_viewer.py`'s batch 0 |
| H3 first updates | released steps 2, 4, 8, 16 | the sign agreement of zero-initialised biases' increments ≥ 0.95 at 2 and 4 (C6's method: warmup01 against `pythia-70m` read 1.000 to 4) |
| H4 loss | released `pythia-70m` at 1000 and 4000, on 64 held-out sequences | A0's loss within 2 % (placed; fp16 against bf16) |
| H5 the loop | `t5h`'s float64 known answer | exact |

## V's first check (Blocked 153's main risk)

V's gate refused on C6's warmup01 (plant B 0.38 against 0.4). So V must be shown readable on F,
the first non-Pythia arm, before A0 trains past step 64. F trains to 4000, and C6's gate (the
same plants and bar) runs at an off-grid step, 3500. If V refuses on F, C4 reads the centred
count alone (C6 U1b's form), and whether to train A0 to 4000 is Blocked, the user's.

## Readings and outcomes (fixed now)

- **R1:** A0's two dates against the released `pythia-70m`'s (harness). A mismatch outside
  REF70's spread is a refusal: the flat arm is then not read.
- **R2:** F's centred date, labelled by C6's rule.
- **R3:** F's V date, labelled, if V's gate passed on F.
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

## Carried in (from the ROADMAP row; done in U1, before any arm's centred arm is derived)

1. **C2 3c's Parked 0:** `c2_seeds.null_matched_tol` must refuse, not return a tolerance, when
   the null's uncentred rate is 0. A flat arm's early re-inits could hit this.
2. **T5 U3's Parked 3:** `c2_seeds.b_matched_tol` bounds τ at ≤ 1e-2 and reads a miss outside
   the bound as "no τ". Either the bound is widened or the refusal names the bound. Recommended:
   the refusal names it, which changes no earlier reading. A widened bound would be a new
   instrument for every arm read so far.

Both come with tests in U1, and C6's and C2's records are re-checked to read unchanged.

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
| 0 | the scan and this design (writing only) | Blocked 155 decided; the reading rule registered by the user if D4 (a) |
| 1 | the harness (from `t5h`), the shard downloaded and verified, H1–H5, A0 to step 64, throughput measured; the flat arm's maps in `tools/math_checks/`; the two carried-in `c2_seeds` refusals with tests | every check, or a refusal |
| 2 | F to 4000 (runner, about 2 days); V's gate at 3500 | F's records, V gated or refused |
| 3 | A0 to 4000 (about 2 days); R1–R3 read | the labels |
| close-out | one page, `paper/` §7, Parked carried or dropped, the card | the card current |

## Decisions (each yours; recommendation first)

| # | decision | options | recommendation |
|---|---|---|---|
| 155 | C4's design and registration | **D1, the arms:** (a) F (flat at 1e-3) and A0 (Pythia's, the harness check); warmup01 already covers the long ramp; (b) as (a), plus F½, flat at 5e-4 (Qu & Liu's peak-dependence; about 2 GPU-days more); (c) F only, with the harness checked on the first steps alone. **D2, horizon:** (a) 4000, C2's plateau window; (b) 2000, half the cost, without the centred plateau. **D3, order:** (a) A0 to 64 (the harness), then F to 4000 with V's gate, then A0 to 4000; (b) A0 to 4000 first. **D4, registration:** (a) you register the reading (C6's rule on F, R2 primary) in `claims/registry.json` before F's first scored checkpoint is read; (b) unregistered, tier 1 | **D1 (a), D2 (a), D3 (a), D4 (a).** (a) is the smallest pair that separates the three hypotheses with a harness check; (b) only matters if F forms nothing, and can follow then. D3 (a) puts V's risk first, at no extra cost. D4 (a): C4 is the one row whose answer could go either way and decides a sentence of the paper |

## Worth challenging (for the PR)

- **A flat rate at Pythia's peak may diverge**, or form nothing (Qu & Liu, a blog on a small
  model). Divergence is a refusal and "not formed" a reading; D1 (b) is the hedge.
- **The harness is not Pythia's** (bf16 against fp16, one GPU against many, a different
  library). H1–H5 and R1 are the guard; R1 compares dates, not weights.
- **One seed per arm.** The reference spread is REF70's ten seeds. A0 against the released run
  is the harness's spread only.
- **70m only.** 410m's dates are reached by inference, as §7 must say.
- **The cost is an estimate** until U1 measures throughput.
