<!-- trackC_writeup/design-C4p.md -->
# Track C — design: C4′ (a) and (b), no training

**Status: proposed 2026-11-15 (U0, writing only); decisions D1–D5 are STATE Blocked 159.
Revised after `/challenge-pr` on #350 (accept with changes, findings 1–4 taken: (b)'s gate step
counts and what it does not test, the null at fp16, what A1's label covers, (b)'s cost).** Opened
on Blocked 158 (a), the user's (2026-11-15): C4′ (a), then (b), each with its rule fixed before
its output; the replicate pair (C4′ (c)) is decided on what they show. Scan: `lit-C4p.md`.
Nothing below has been computed. Of the inputs, only C6's centred counts on warmup01 (A3's
records) have been seen, as counts; their per-transition split has not been read.

## What it is for

C4 closed with seven leans read after its label (`status-C.md` "C4 on one page", items 6–10).
Two of them can be tested at once, without training:

- **(a) Induction's date follows the summed rate more nearly than the counts do** (item 9: A0
  768 → the summed-rate map 206 on F, F read 320; step map 768). `pythia-70m-warmup01` is
  `pythia-70m`'s init and first batch with a ten times longer ramp at the same peak (`status-C.md`
  "C6 on one page", item 1), a run C4 did not train and whose induction nobody here has computed. Inside both
  warmups the summed-rate map is t′ = √10 · t (`tools/math_checks/c6_bands.py`), so the lean
  says ≈ 2430 from A0's 768, step says 768. Pythia's grid holds 1000, 2000, 4000 between them.
- **(b) Is F A0 on another clock?** (item 10 says no, from the order of seven events.) If F's
  next-token function at each step is A0's at some step, the clock question is well posed and
  the match is the clock; if not, F takes another path and "moves with the schedule" in
  `paper/` §7 means less than it says.

## (a) The third schedule, already released (U1)

**Inputs.** `pythia-70m-warmup01` (the run) on C6's 18-step grid, all cached; `pythia-70m` and
`pythia-70m-seed1..9` (the reference) on the same grid, cached (C6 U0); step 3000 of the run, one
download, beside only (D1). C6 U1b's records on the run and on seed 0 for A3.

**A1, induction's date (the reading).** Each checkpoint's induction score per head is
`c4_read.induction_scores` on C4's sequences (`IND_LEN` 64, `IND_N` 8, `IND_SEED`), the
instrument both C4 arms were read with; the profile is the top head's score by step. It is read
with **C6's frozen rule, unchanged** (`c6_warmup.py`: `ref70`, `bands`, `window`, `date_on`,
`joint`), with the top head's score in place of a count and a single arm in place of the τ arms
(D2):

| item | fixed as |
|---|---|
| the reference (`REF70` for induction) | the half-rise of each of the ten seeds' profiles on 70m's 1000–4000 window, with 92 (i)'s unreadable check, as C6 read the counts |
| schedules | step, cumulative, instantaneous (C6's three, with the run's real schedule, W 14 300) |
| a schedule's band, window, early set | C6's `band`, `window`, early set (tops 128, 512, 2000), unchanged |
| h consistent | the run's profile has a non-zero plateau on h's window, is below half of it on h's early set, and its half-rise is in h's band |
| the label | `joint` over the one profile: "consistent with …", "not told apart", "neither", or "unreadable" |
| **the gate** (before the run's scores are computed) | C6's `calibrate` with `loo=True` on the ten seeds' induction profiles: each seed's profile planted under each schedule (`plant`), read against the other nine seeds' reference. **Passes iff ≥ 8 of 10 cumulative plants read exactly "consistent with cumulative" and ≥ 8 of 10 step plants exactly "consistent with step"**; instantaneous's beside. If it fails, A1 is refused and the run's profile is still computed and reported, as description |
| the lean, stated now | **"consistent with cumulative"** supports item 9's lean on a run C4 did not train; **"consistent with step"** breaks it; anything else leaves it a lean |
| what "consistent with cumulative" covers (`/challenge-pr` on #350, finding 3) | a half-rise in the cumulative band, a set of grid steps (C6's grid has 1000, 2000, 4000 there; the released run's score goes 0.02 at 512 to 0.93 at 1000), **not a point at ≈ 2430**. Inside the warmup every clock that is a function of the summed rate predicts the same band, among them **a fixed training loss** (2408.11029's law with S2 = 0, `lit-C4p.md` 2). So the label supports "the summed rate or a function of it", and cannot tell the summed rate from loss |
| beside, the loss at the date | the held-out loss (`ct.HELD_OUT`'s 64 rows, unseen by either run at or before step 4000) at the run's half-rise and at `pythia-70m`'s; equal losses say a loss clock fits as well as the summed rate, numbers only |

A1 is C4's lean read on a held-out run with its rule fixed before output, so it could be
registered (D3), with the scope in the row above written into its statement.

**A2, the heads (beside, description).** The heads with score ≥ 0.5 (`c4_close.HEAD_BAR`) on the
run, on `pythia-70m` and on each seed at 4000, 8000, 16 000 and 143 000, and the overlap of each
set with `pythia-70m`'s. The seeds' overlaps are the chance scale for another init and order. C4
found F's four heads disjoint from A0's (= the released run's) from the same init and order; the
run shares init and order and differs only in the ramp's length. No prediction is made.

**A3, the per-transition centred set (beside, description).** From C6 U1b's records on the run
and on seed 0, at the three gated arms (9.61e-5, 1e-4, 3e-4), C4's `c4_close.per_transition`,
`majority` and `settled` on C6's grid: the majority set at each step, the step it settles, and
whether there is a **transient**, a transition violated on a majority of prompts at some step
that is not in the final set. C4's leans, stated now: both C4 arms end on {T1, T2}; F has a
transient (T3–T5 at 64–320) and A0 none; on A0 T2 fills before T1. Each is reported for the run
as "holds" or "does not", nothing scored.

**Cost.** 10 models × 18 steps + the run's 21 + step 3000: about 200 forward passes of 8 × 128
tokens, minutes on the 3080; one download (~150 MB).

## (b) Is F A0 on another clock? (U2)

**Inputs.** Both C4 arms' 29 saved checkpoints (`ct.SAVE_STEPS`, `HDD_1TB/mets_data/c4/`),
`pythia-70m` and `pythia-70m-seed1..9` on Pythia's grid to 4000 (cached). The 64 held-out rows
(`ct.HELD_OUT`, iteration 4000's first 64, read by neither arm), all 2048 positions each.

**The statistic.** K(X, Y) = the mean over the 131 072 positions of KL(p_X ‖ p_Y) of the
next-token distributions, in nats, float32 log-softmax. For each F step s ≥ 1: **match(s)** = the
A0 grid step t minimising K(F_s, A0_t), **r(s)** = that minimum. A0's resolution at t:
**δ(t)** = min(K(A0_t, A0_prev(t)), K(A0_t, A0_next(t))) on the dense grid. F is **on A0's path
at s** iff r(s) ≤ δ(match(s)): F's function is as close to an A0 state as A0's neighbouring grid
states are to each other.

| item | fixed as |
|---|---|
| windows | **early**, F's steps 1–256 (C4's transient, before F's induction at 320); **late**, 320–4000 (`lit-C4p.md` 3: per-seed KL reconverges near induction) |
| a window's reading | **"one path"** if F is on A0's path at ≥ ⅔ of the window's steps and match is non-decreasing up to one grid step; **"another path"** if on the path at < ⅓; else **"mixed"** |
| **the gate** (planted null and effect on the real design points, before F is read) | null: `pythia-70m` (same init and order as A0; harness noise) as pseudo-F on its 13 grid steps 1–4000 (early 9: 1–256; late 4: 512–4000), **against A0 rounded to fp16**, since every released weight is fp16-representable (stored as fp32) and early updates below fp16's resolution vanish from it (`/challenge-pr` on #350, finding 2; F is read against A0 at full precision, both fp32 from one harness). Effect: each of `pythia-70m-seed1..9` (another init and order) as pseudo-F on its 12 cached steps 2–4000 (early 8, late 4; no step 1), against A0 at full precision. **A window is readable iff the null reads "one path" with match(s) = s at ≥ ⅔ of its steps and ≥ 8 of 9 seeds read "another path"**; with 4 late steps, ⅔ is 3 of 4 and ⅓ is 2 of 4. An unreadable window reports F's numbers as description and no reading |
| what the gate does not test (finding 1) | a seed differs from A0 in init and order; F differs in neither. The gate shows the instrument sees an init-and-order difference, not a same-init one, so **"one path" is the weaker reading**: a same-init departure smaller than a seed's could read "one path". The seeds reconverge near 2k (2509.26643), so **the late window may well be refused**; then (b)'s reading is the early window's, which holds the lean |
| the clock, beside | in a "one path" window, match(s) against the step map (t = s) and the summed-rate map (`c4_read.m`, inverted); numbers only, no label |
| beside, description | each match's training loss on both arms (the S1 laws predict a loss match, `lit-C4p.md` 2); K in both directions; the seeds' K(released_t, seed_t) at the same step as the "another init" scale; per parameter group, ‖θ_s − θ_0‖ on both arms (same init, so comparable), the A0 step of equal displacement, and cos(θ^F_s − θ_0, θ^A0_match(s) − θ_0) |
| the lean, stated now | item 10 says F is not A0 on another clock: **"another path" early** is the lean; "one path" in both windows breaks it |

**Cost** (revised, `/challenge-pr` on #350, finding 4). About 180 checkpoints with the controls
(29 F + 29 A0 + 13 released + 9 × 12 seeds), each a forward pass of 64 × 2048 tokens. One
model's float32 log-probs on one row are 412 MB, so a row's 29 A0 models do not fit on the GPU
together and a per-pair loop is far over an hour. Instead, per half-row (1024 positions), each
model's float32 log-probs are kept in CPU memory (206 MB each; ≈ 12 GB for F and A0 together),
and at each position K's cross term for every pair is one matrix product
(P_F (n_F × V) · log P_A0ᵀ), the entropy terms per model; the controls likewise. Estimated about an
hour; U2 times one half-row first, refuses to subsample positions, and stops for the user if the
projected total passes 3 h.

## Exit condition and budget

**Exit:** A1 labelled or refused by its gate, A2 and A3 written; (b) read per window or each
window refused by its gate; each lean of C4's items 6–10 that (a) or (b) touches marked held,
broken or untouched, so C4′ (c)'s registration can be decided on them. **Budget: 3 units +
close-out** (U0 this design, U1 (a), U2 (b), close-out). GPU about 1–2 hours in all.

## Carried in

- **The two `c2_seeds` refusals** (C4 U1's Parked 0): not taken. No unit here derives a centred
  arm (A3 reads C6's frozen arms), so they stay with ROADMAP C4′ for the next unit that does.
- **Keep `c4_train.py` untouched** (C4's close-out, U1 (1)): nothing here trains. New code goes in
  `c4p_read.py`, so `c4_read`'s (`220161b9159b`) and `c6_warmup`'s registered hashes stay.

## Decisions (each yours; recommendation first)

| | question | (a), recommended | (b) |
|---|---|---|---|
| D1 | step 3000 on the run | C6's 18-step grid as frozen; step 3000 downloaded and its score printed beside, outside the rule (C6's rule and tests are reused unchanged; the gate shows whether the grid separates) | put 3000 on the run's grid (C6's band and window code re-derived for two grids, its tests re-run) |
| D2 | A1's rule | C6's rule unchanged, on the top head's score (a ten-seed reference, a run-to-run spread C4 lacked) | C4's A0-referenced band (one seed, the grid's ρ; needs a dense grid the run lacks) |
| D3 | register A1 | **as a second `P-SCH` entry, by you, after the gate passes and before the run's scores are computed** (C4's lean on a held-out run, the one confirmatory reading available without training); Claude drafts the entry in U1 | tier 1 only, the rule frozen in code and pushed before output |
| D4 | (b)'s rule | the residual against A0's own resolution, gated on the released run (null) and the nine seeds (effect), per window | description only: the matrix, matches and residuals, no reading |
| D5 | order | U1 (a), then U2 (b), as Blocked 158 (a) | (b) first (its gate needs no download) |

## Worth challenging (for the PR)

1. **A1 uses the top head's score, whichever head it is.** On F the top head changed set; a run
   whose top head switches mid-rise could date the switch, not formation. Rejected: a fixed head
   (none is shared across arms) and the sum of heads ≥ 0.5 (a count with its own threshold).
2. **δ(t) is generous**: a state between two A0 grid states is at most δ from either, so "on the
   path" admits anything as close as one grid step. Rejected: half of δ (KL is not a metric, so
   "half" has no meaning). The gate decides whether the seeds still read "another path" under it.
3. **The null for (b) is the released run, which A0 reproduces.** It tests that the instrument
   finds the identity clock with harness noise, not that it finds a stretched one. No released
   run of 70m is a known stretch; warmup01 is not one (C4's question). A synthetic stretch
   (A0's own checkpoints relabelled) would pass trivially.
4. **A3 reads records whose counts were seen in C6.** The per-transition split was not, and A3
   is description only.
5. **D3 recommends a registration** from a lean read post hoc. It is legitimate only because the
   run's induction has never been computed here; the reference (the seeds) and the gate are
   computed before the entry is drafted.
