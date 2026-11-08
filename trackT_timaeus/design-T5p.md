<!-- trackT_timaeus/design-T5p.md -->
# Track T — design: T5′, what forms with the rise, per U4 arm (proposed 2026-11-07; revised after `/challenge-pr` on #317, findings 1–5; **frozen 2026-11-07**, Blocked 138: D1–D5 all (a), the user's)

Opened 2026-11-07 by `STATE.md` Blocked 137 (a), the user's, as recommended: **Track T is not
closed; it closes with T5′**, after which the user's order of 2026-11-02 puts Track N (N1) next.
Row: `docs/ROADMAP.md` §7b T5′. Scan: `docs/lit-T.md` §5 (2026-11-07). Tier 1, description
throughout, nothing registered. **Frozen 2026-11-07, before any T5′ output**: D1–D5 below decided
(a), the user's (Blocked 138, "138: all (a)"); every "fixed now" in this file is fixed as written.

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
| held-out text | `…/t5u4/data/tail.bin`: the continuation's 1601 × 256 sequences of 2048 tokens, **trained by no arm** (Blocked 134 (a): no continuation ran) | ~~the first 64 sequences~~ **sequences 64–1087 (1024)**: Blocked 139 (a), the user's, 2026-11-07, post hoc (N set from the noise after G6's ICL leg failed at 64, not from the effect; windows and bars unchanged; rows 0–63 were read by that gate, so not reused) |
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
| ICL | mean token loss at positions 490–509 minus 40–59 over the 1024 held-out sequences (64 until Blocked 139 (a); Olsson et al.'s 500 − 50 with a 20-token window); Pythia reads the same text re-encoded in its tokenizer | dated, beside |
| ρ_mid, n0; S, M | as T5 (`design-T5.md`; `t5j`) | read from the records, not recomputed (G1 checks the path) |
| **I_null** | I's formula with the "repeated" set replaced by a third independent draw `[r_1 … r_T, r″_1 … r″_T]` (no repeat), against the same control: the max over 96 heads of pure noise, at every step of every run | **the planted null** (`/challenge-pr` on #317, finding 1) |

Bootstrap SEs over the R sequences (I, P, B) or the 1024 (ICL), 1000 resamples, seed fixed: beside
each value, not a floor (finding 1: a 4 SE floor on a max over 96 heads, from the trough to the
peak over 49 steps, is cleared by pure noise in 62 % of simulated series).

## The dating rule (D2 (a), revised after `/challenge-pr` on #317, findings 1–3)

For a series x(s) on a run's grid, signed so formation is positive (B and ICL negated):
t = the step of x's minimum before its maximum; peak = max x up to the run's end (4369 for the
arms; 9536 and 4369 both for the released run; 4000 for 410m, as C2's locator). **on_f** = the
first step after t with x(s) − x(t) ≥ f · (peak − x(t)), at **f = 0.1 (the onset) and 0.5 (the
half-rise)**. **Refused, "none formed", if peak − x(t) < 0.1** in I's units (an absolute floor, a
judgement fixed now: uniform causal attention puts about 0.003 on one key at these positions, an
induction head 0.3 or more); G4 checks the floor against I_null on every run, G5 against a planted
rise. For P, B, ICL (beside) the floor is 4 bootstrap SE, as C2's, with finding 1's caveat stated.

**Censored** (finding 2): the peak is the run's last step and x rose by ≥ 0.1 of its rise over the
last grid step. Then the true rise is larger than the measured one, so the true on_f is **at or
after** the measured on_f: a censored date is a lower bound (as C2's paper treats its censored rows).

**ρ_mid is dated in both phases** (finding 3), by the same rule: its **fall** (signed down, from its
early peak, 1.00 at 30–47 on `-32m`, to the trough) and its **rise** (from the trough). Seen: the
rise's half 1713–2340 and post hoc 10 % 1253–1465 are in `status-T.md`; the fall's dates are not
written there, but the series are seen.

## The outcome table (per arm; fixed now; revised after `/challenge-pr` on #317, findings 2–4)

**Phase:** which part of the arm's ρ_mid trajectory on_0.1(I) and on_0.5(I) fall in: the early
rise (before ρ's early peak), **the fall** (the peak to ρ's minimum t), the trough (t to
on_0.1 of the rise), **the rise** (from there). Description: the phase carries no label. Reported for every run, 410m included against its own
ρ_mid (U2's records; its induction at 1000 sits in its fall, 0.81 at 16 → trough 2000–4000).

**Against the rise:** d = on_0.1(I) − on_0.1(ρ_mid's rise) in grid steps (one step near the rise is
about +17 %: 1253 → 1465 → 1713). A label stands only if censoring cannot move it:

| case | label |
|---|---|
| I refused (rise < 0.1) | **no induction formed by 4369** |
| d ≤ −2, I not censored (ρ may be: its censoring only moves its onset later) | **induction leads the rise** (the phase says whether in the fall or the trough) |
| −1 ≤ d ≤ 1, neither censored | **induction with the rise** |
| d ≥ 2, ρ not censored (I may be: its censoring only moves it later) | **induction after the rise** |
| any other case with a censored series | **censored, not ordered** |

**Noise** (finding 4): on the **gap d**, not I's onset alone. If the three A0 seeds' d span ≥ 2 grid
steps, A0's row reads "seed-dependent" and any arm's d inside A0's range widened by that span is
"not told apart". Across arms: description only (one run each). S, M, n0 (D4 (a)): dated by the
same rule against ρ, no label.

## Calibration gate (U1, before any reading)

| # | check | pass |
|---|---|---|
| G1 | the producer's ρ_mid on A0 at 671 and 4369 and on released `-32m` 4369 equals the read records | ≤ 1e-6 |
| G2 | eager against sdpa (the arms trained under sdpa): hidden states at A0 4369 | max relative ≤ 1e-4 |
| G3 | the scorer on synthetic attention: one planted head at key j + 1 gives PM 1; uniform causal attention gives net I = 0 | exact to 1e-12 |
| G4 | **planted null over the real checkpoints** (finding 1; replaces "I ≤ 0.05 at init", which could not fail): I_null on every run (six arms, `-32m`, 410m) | refused ("none formed") on all 8, and max I_null ≤ 0.05 at every step |
| G5 | **planted effect over the real checkpoints:** each run's I_null series plus a logistic rise of height 0.2 (twice the floor) centred at each of 1072, 1465, 2002, 2735, 3738 (410m: 256, 512, 1000, 2000, 4000) | the rule dates on_0.5 within one grid step of the centre in ≥ 38 of 40, and never refuses |
| G6 | **positive control:** 410m at 143000 | I ≥ 0.3; ICL below step 0's by ≥ 4 SE |
| G7 | first output populated: A0 4369's record has I, I_null, P, B, ICL, the argmax head, non-null | all |
| B | **the bridge (D3 (a), revised, finding 5): not a gate.** 410m's I half-rise on C2's grid (C2's locator, reference to 4000) | **at 1000**: the arms' dates sit beside C2's E3; elsewhere the comparison with C2's 1000 is refused and stated, **the arms' own tables still read** |

A G1–G7 failure refuses the reading and goes back to the user.

## Units, exit and budget

**U0** (this design and the scan; docs only). **U1**: producer `trackT_timaeus/t5k.py`, the gate,
the read on six arms × 49 + `-32m` × 53 + 410m × 18 (≈ 420 forward passes, GPU minutes), the
outcome table; with it `t5j.outcome_row`'s refused-run flag (T5's Parked; a refused run named in
the row text; no recorded row moves, none refused). **Close-out**, which also closes Track T
(Blocked 137 (a)). **Exit:** each arm's induction onset and half-rise read against its rise under
the table above, or refused by its gate. **Budget: U0 + U1 + close-out**, as the ROADMAP row.

## Blocked 139 decided, and T5′ paused (2026-11-07, the user's; after G6's ICL leg failed, before any batch)

| change | as | label |
|---|---|---|
| ICL's held-out set | tail rows 64–1087 (1024; rows 0–63 were read by the refused gate), windows and bars unchanged | post hoc (N from the noise, not the effect); committed `938eea3` before any output on it |
| ICL's document breaks, beside | `gate.json` records the share of held-out rows with an `<|endoftext|>` in positions 40–509 (`t5k.break_fraction`; not a gate). On the input text alone: rows 0–63 0.39 (Parked 1's 25 of 64), rows 64–1087 0.45. More rows cut the SE about 4×, not the dilution; ICL's definition is not changed now (`/challenge-pr` on #319, finding 2) | beside |
| **I_text, beside** (the user's: "avoid synthetic data that we choose") | I's formula on repeated real text, **fixed now** (`/challenge-pr` on #319, finding 3): R = 32 tail rows each for x (rows 1088–1119), x′ (1120–1151) and x″ (1152–1183), each the row's first T = 256 tokens; repeated `[x, x]` against control `[x, x′]`; **I_text_null** `[x, x″]` against the same control (no repeat), at every step of every run, max ≤ 0.05 described beside, not a gate. Pythia reads each row decoded and re-encoded, its first 256 tokens (as ICL's text). Dated by D2's rule, no label. The prompt library does not move it | post hoc, description only; **I on random tokens stays the labelled statistic** (D1 (a) not moved after the gate showed A0's I 0.72). **Built and committed 2026-11-08 before its first output** (`8f639ad`; details in `t5k.py`'s docstring); the gate re-run under 139 (a): G1–G3, G6, G7 pass; G4, G5 run in `read`, after the batch (`status-T.md` "T5′ U1 resumed") |
| **T5′ paused** | the user chose to build a larger shared prompt library first (`docs/ROADMAP.md` §5 "Prompt library", §10.14); T5′ resumes on its frozen inputs with the gate under 139 (a). **The library closed 2026-11-07** (`prompt_library/status-L.md` "The library on one page (closed 2026-11-07)"), so T5′ resumes: next, I_text and its null built and committed, then the gate re-run under 139 (a) | nothing dated |

**An exception to the close-out rule, on the record** (`/challenge-pr` on #319, finding 1): `CLAUDE.md`
"Close out before the next opens" and `docs/ROADMAP.md` §3 say a new idea mid-phase goes to the
ROADMAP and the phase finishes first. Here the user chose, on the record (§10.14), to pause T5′ and
build the library first; the exception is the user's, not the rule's. **Open in T5′ on pause:** the
gate re-run under 139 (a); I_text and its null built (code and tests, before output); G4, G5; the
batch (≈ 420 forward passes); `read`; the outcome table; `t5j.outcome_row`'s refused-run flag; the
close-out, which also closes Track T. Parked 1 (document breaks) stands. The library is not a fix
for T5′: T5′ resumes on the inputs fixed in this file, whatever the library holds.

**Budget** (finding 4): unchanged, U0 + U1 + close-out. `938eea3`'s "U1b" is U1 continued, not a new
unit: 139 (a) as offered was "then G4, G5, the batch and the reading as frozen", inside U1; I_text
adds code to U1. If U1 does not finish in one more session it goes back to the user as an extension.

Rejected for I's input: Markov-chain sequences (Edelman et al. 2024, cited from memory, not re-read;
those papers *train* on chain samples, and as a probe the chain's parameters would be ours to
choose); replacing I by I_text (moves D1 after output).

## Decisions (decided 2026-11-07, all (a), the user's: Blocked 138)

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | the induction statistic | (a) I: the max head's net prefix matching on repeated random tokens, control subtracted; (b) each arm's top 2 heads at 4369, named, traced back (E3's shape); (c) the behavioural ICL score alone | **(a).** It selects no head from the end state, so a date is not pulled toward 4369; Olsson et al.'s and Singh et al.'s measure. (b) matches E3 but names its heads on the step the series ends at. (c) has no head choice but reads packed text with document breaks, so its signal is diluted; it stays beside |
| D2 | the dating rule (revised after `/challenge-pr` on #317, findings 1–3) | (a) from the trough at f 0.1 (onset) and 0.5 (half); an absolute floor 0.1 on I, checked by a planted null and a planted rise on every run's real checkpoints (G4, G5); censored dates as lower bounds, a label kept only where censoring cannot move it; ρ dated in its fall and its rise; (b) C2's locator unchanged (from step 0, half only, 4 SE floor); (c) a fitted sigmoid's midpoint | **(a).** The question is whether something *leads*, which a half-rise toward a still-climbing end cannot say (#313, finding 2); 0.5 is kept for C2's bridge. (b) dates ρ, which falls first, from the wrong end, and its 4 SE floor is cleared by noise on a max over heads (finding 1). (c) adds a model the grid's 49 points may not support |
| D3 | the bridge (revised after `/challenge-pr` on #317, finding 5) | (a) not a gate: the comparison with C2's 1000 stands only if 410m's I half-rise is at 1000 exactly, else it is refused and stated, and the arms' own tables still read; (b) a gate refusing the whole reading unless 410m lands at 512–2000 (the first proposal) | **(a).** The arms' tables answer the row's first question on their own; only the sentence that sets them beside C2's E3 needs the bridge. (b) refused too much and passed too easily: 512–2000 holds three of C2's four most common dates |
| D4 | S, M, n0 | (a) dated by D2's rule, described, no label; (b) labelled by the outcome table, as I | **(a).** Their series were read at the beside stage and their post hoc 10 % steps written down (671–1253, 917–1465), so any rule chosen now is chosen with them in view; I, P, B and ICL are the only unseen series |
| D5 | released `-32m` | (a) all 53 checkpoints to 9536, dated to 9536 and to 4369; (b) arms only | **(a).** About 10 GPU-minutes; it reads induction past 4369, where the arms stop, and checks A0's timing against the run A0 reproduces (its fall and late level matched; its rise started earlier, `status-T.md` "T5 U4a") |

## Worth challenging (for the PR)

- Max over heads is noisy at the bottom (the largest of 96 small values on the arms, 16 layers × 6 heads; 384 on 410m) and switches head
  through training; I_null (G4) measures that noise on every run, I_top5 and the argmax head sit beside.
- The floor 0.1 on I and the planted height 0.2 are judgements, fixed before output; G4 and G5 test
  them on the real checkpoints but do not derive them.
- f = 0.1 is a judgement; it is also the fraction the beside stage used post hoc on S and n0, so
  choosing it now is not blind for those (D4) though it is for I.
- Random-token repeats are off-distribution; a head can match prefixes there and not on text. B and
  ICL are its behavioural checks, not gates.
- The bridge (B) asks a different statistic to land at E3's exact step; if random-repeat induction
  forms one grid step before E3's named heads on text, the C2 comparison is refused though the two
  may date one event. Revised from a refusing gate at 512–2000 (`/challenge-pr` on #317, finding 5).
- Censoring keeps a label only where it cannot move (finding 2); with both I and ρ still rising at
  4369, every arm may read "censored, not ordered", and D5's released run to 9536 is then the only
  uncensored read.
- Rejected: reading ICL on Dodecahedron's own data (unknown); `-67m` (its trajectory matches `-32m`'s, U2,
  and no arm has its shape); training anything.
