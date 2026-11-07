<!-- trackT_timaeus/design-T5p.md -->
# Track T — design: T5′, what forms with the rise, per U4 arm (proposed 2026-11-07)

Opened 2026-11-07 by `STATE.md` Blocked 137 (a), the user's, as recommended: **Track T is not
closed; it closes with T5′**, after which the user's order of 2026-11-02 puts Track N (N1) next.
Row: `docs/ROADMAP.md` §7b T5′. Scan: `docs/lit-T.md` §5 (2026-11-07). Tier 1, description
throughout, nothing registered. **Proposed, not frozen**: D1–D5 below are the user's (Blocked 138);
no T5′ output exists.

## What T5′ is for

T5 found Dodecahedron's middle-state cloud falls to ρ_mid 0.10–0.11 and rises again, and that its
retrained recipe (A0, three seeds) and the one-change arms (A1 biases, A2 Pythia's init, A3 both)
all rise between about 1253 and 4369 (`status-T.md` "T5 on one page"). It did not read what else
forms then. On `pythia-410m` the paper dates previous-token and induction attention at 1000, the
step where the centred cloud breaks (`trackC_writeup/paper/main.tex` §5, E3, E4). T5′ asks, per arm:

1. **When does induction form, against the arm's rise?** (leads, with, after, or not by 4369)
2. **Does a sink, massive-activation or position-0-norm onset lead the rise?** (T5's Parked item,
   "not determined by the half-rise rule"; description only, see D4)

It is the bridge from C2's 1000 to a model without Pythia's biases or init: if the instrument dates
410m's induction at C2's step (D3), the arms' dates sit beside it.

**What it cannot show.** Co-occurrence, not cause: no arm intervenes on induction. One run per
one-change arm; the three A0 seeds are the only noise floor. The arms stop at 4369.

**The token arithmetic (no lean).** The arms train at 524,288 tokens a step, Pythia at 2,097,152:
Pythia's step 1000 is 2.10B tokens, an arm's step 4000; the arms end at 2.29B tokens. Induction
forms near 2–5B tokens across Pythia sizes (Tigges et al., via Olsson et al.; `lit-T.md` §5), and
Aoyama et al. find its step depends on batch and context size. So "no induction by 4369" is a live
row of the outcome table, not an edge case. The released `-32m` runs to 9536 (D5).

## Inputs (checked on disk 2026-11-07)

| input | where | as |
|---|---|---|
| the six arms, 49 checkpoints each (0, 1–8, 10 … 4369; T2's grid ≤ 4369) | `HDD_1TB/mets_data/trackT/t5u4/{A0,A0p,A0pp,A1,A2,A3}/ckpt/step*.safetensors` | trained at `t5h` `66af413c31fe` (A0), `40289679fc36` (replicates), `81b2e03bdcf7` (arms) |
| their ρ_mid and n0 per step | `…/t5u4/<arm>/read/step*.json` (U4's reads) | seen |
| their S, M, n0 onset series | `…/t5u4/beside/<arm>/onset/` (the beside stage) | seen, post hoc 10 % values in `status-T.md` |
| held-out text | `…/t5u4/data/tail.bin`: the continuation's 1601 × 256 sequences of 2048 tokens, **trained by no arm** (Blocked 134 (a): no continuation ran) | the first 64 sequences |
| released `timaeus/dodecahedron-32m`, 53 checkpoints 1 … 9536 | `HDD_1TB/mets_data/hf` (53 refs) | as U2 |
| `pythia-410m` seed 0 at C2's 18 steps 0 … 143000 | `~/.cache/huggingface` (34 refs) | as C2 |

## The statistics (fixed now)

Random-token sequences: R = 32, each T = 256 token ids drawn uniformly (seed 20261107) from the
model's ordinary vocabulary (special tokens excluded), no BOS (the arms trained on packed sequences
without one). **Repeated:** `[r_1 … r_T, r_1 … r_T]`. **Control:** `[r_1 … r_T, r′_1 … r′_T]`, a fresh
second half. Pythia draws from its own vocabulary under the same seed. Eager attention, float32.

| symbol | definition | role |
|---|---|---|
| PM_h | head h's mean post-softmax attention from query T + j to key j + 1 (the token after the earlier copy), j = 0 … T − 2, over the R sequences (Olsson et al.'s prefix matching) | per head |
| **I** | max over heads of PM_h(repeated) − PM_h(control): the control removes any head that attends at that offset without a repeat | **dated, labelled (D1 (a))** |
| I_top5, argmax head | mean of the top 5 heads' net PM; which head is the max | beside |
| P | max over heads of mean attention from q to q − 1 on the control sequences (E4's analogue) | dated, beside |
| B | mean token loss on the repeated second half (positions T + 1 … 2T − 1) minus the same on the control | dated, beside (behaviour of I) |
| ICL | mean token loss at positions 490–509 minus 40–59 over the 64 held-out sequences (Olsson et al.'s 500 − 50 with a 20-token window); Pythia reads the same text re-encoded in its tokenizer | dated, beside |
| ρ_mid, n0; S, M | as T5 (`design-T5.md`; `t5j`) | read from the records, not recomputed (G1 checks the path) |

Bootstrap SEs over the R sequences (I, P, B) or the 64 (ICL), 1000 resamples, seed fixed.

## The dating rule (D2 (a), fixed now)

For a series x(s) on a run's grid, signed so formation is positive (B and ICL negated):
t = the step of x's minimum before its maximum; peak = max x up to the run's end (4369 for the
arms; 9536 and 4369 both for the released run; 4000 for 410m, as C2's locator). **on_f** = the
first step after t with x(s) − x(t) ≥ f · (peak − x(t)), at **f = 0.1 (the onset) and 0.5 (the
half-rise)**. Refused, "none formed", if peak − x(t) < 4 bootstrap SE (C2's locator's floor).
**Still-rising flag:** the peak is the last step and x rose by ≥ 0.1 of the rise over the last grid
step; the half-rise is then late by construction (`/challenge-pr` on #313, finding 2) and is stated
so. ρ_mid's rise is dated by the same rule from its trough (seen: its half-rise 1713–2340 and post
hoc 10 % 1253–1465 are in `status-T.md`).

## The outcome table (per arm; fixed now)

d = the grid-index difference on_0.1(I) − on_0.1(ρ_mid); the half-rises are read the same way beside.
One grid step near the rise is about +17 % in step (1253 → 1465 → 1713).

| case | label |
|---|---|
| I refused (rise < 4 SE) | **no induction formed by 4369** |
| d ≤ −2 | **induction leads the rise** |
| −1 ≤ d ≤ 1 | **induction with the rise** |
| d ≥ 2 | **induction after the rise** |

**Noise:** if the three A0 seeds' on_0.1(I) span ≥ 2 grid steps, A0's row reads "seed-dependent" and
any arm's departure from the A0 range by less than that span is "not told apart". Across arms:
description only (one run each). S, M, n0 (D4 (a)): dated by the same rule against ρ, no label.

## Calibration gate (U1, before any reading)

| # | check | pass |
|---|---|---|
| G1 | the producer's ρ_mid on A0 at 671 and 4369 and on released `-32m` 4369 equals the read records | ≤ 1e-6 |
| G2 | eager against sdpa (the arms trained under sdpa): hidden states at A0 4369 | max relative ≤ 1e-4 |
| G3 | the scorer on synthetic attention: one planted head at key j + 1 gives PM 1; uniform causal attention gives net I = 0 | exact to 1e-12 |
| G4 | **null at init:** every run's first checkpoint (arms step 0, `-32m` step 1, 410m step 0) | I ≤ 0.05 |
| G5 | **positive control:** 410m at 143000 | I ≥ 0.3; ICL below step 0's by ≥ 4 SE |
| G6 | **the bridge (D3 (a)):** 410m's I half-rise on C2's grid (C2's locator, reference to 4000) | at 512, 1000 or 2000 (1000 ± one grid step) |
| G7 | first output populated: A0 4369's record has I, P, B, ICL, the argmax head, non-null | all |

A failure refuses the reading and goes back to the user.

## Units, exit and budget

**U0** (this design and the scan; docs only). **U1**: producer `trackT_timaeus/t5k.py`, the gate,
the read on six arms × 49 + `-32m` × 53 + 410m × 18 (≈ 420 forward passes, GPU minutes), the
outcome table; with it `t5j.outcome_row`'s refused-run flag (T5's Parked; a refused run named in
the row text; no recorded row moves, none refused). **Close-out**, which also closes Track T
(Blocked 137 (a)). **Exit:** each arm's induction onset and half-rise read against its rise under
the table above, or refused by its gate. **Budget: U0 + U1 + close-out**, as the ROADMAP row.

## Decisions (each yours; recommendation first)

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | the induction statistic | (a) I: the max head's net prefix matching on repeated random tokens, control subtracted; (b) each arm's top 2 heads at 4369, named, traced back (E3's shape); (c) the behavioural ICL score alone | **(a).** It selects no head from the end state, so a date is not pulled toward 4369; Olsson et al.'s and Singh et al.'s measure. (b) matches E3 but names its heads on the step the series ends at. (c) has no head choice but reads packed text with document breaks, so its signal is diluted; it stays beside |
| D2 | the dating rule | (a) from the trough at f 0.1 (onset) and 0.5 (half), refusal at 4 SE, the still-rising flag; (b) C2's locator unchanged (from step 0, half only); (c) a fitted sigmoid's midpoint | **(a).** The question is whether something *leads*, which a half-rise toward a still-climbing end cannot say (#313, finding 2); 0.5 is kept for C2's bridge. (b) dates ρ, which falls first, from the wrong end. (c) adds a model the grid's 49 points may not support |
| D3 | the bridge | (a) a gate: refuse if 410m's I half-rise is not at 512–2000; (b) read 410m beside, no refusal | **(a).** The row's reason is the bridge to C2's 1000 (E3, two named heads, natural text). If the max-head score on random repeats dates 410m elsewhere, the arms' dates measure another event and cannot sit beside C2's |
| D4 | S, M, n0 | (a) dated by D2's rule, described, no label; (b) labelled by the outcome table, as I | **(a).** Their series were read at the beside stage and their post hoc 10 % steps written down (671–1253, 917–1465), so any rule chosen now is chosen with them in view; I, P, B and ICL are the only unseen series |
| D5 | released `-32m` | (a) all 53 checkpoints to 9536, dated to 9536 and to 4369; (b) arms only | **(a).** About 10 GPU-minutes; it reads induction past 4369, where the arms stop, and checks A0's timing against the run A0 reproduces (its fall and late level matched; its rise started earlier, `status-T.md` "T5 U4a") |

## Worth challenging (for the PR)

- Max over heads is noisy at the bottom (the largest of 96 small values on the arms, 16 layers × 6 heads; 384 on 410m) and switches head
  through training; the control subtraction and G4 bound it, I_top5 and the argmax head sit beside.
- f = 0.1 is a judgement; it is also the fraction the beside stage used post hoc on S and n0, so
  choosing it now is not blind for those (D4) though it is for I.
- Random-token repeats are off-distribution; a head can match prefixes there and not on text. B and
  ICL are its behavioural checks, not gates.
- G6 asks a different statistic to land at E3's date; if induction on random repeats forms a grid
  step before E3's named heads on text, G6 still passes (±1 step), which is loose by design.
- Rejected: reading ICL on Dodecahedron's own data (unknown); `-67m` (its trajectory matches `-32m`'s, U2,
  and no arm has its shape); training anything.
