<!-- trackC_writeup/design-C5.md -->
# Track C — design: C5 opened on M2, the massive-activation coordinates projected out

Opened 2026-10-23 by Blocked 103 (a), the user's ("C5's M2 first, then C2p"; nothing posted
to arXiv by Claude: posting stays the user's). Row: `docs/ROADMAP.md` §8 C5. Scan: `trackC_writeup/lit-C5.md`
(2026-10-23). Unit 0 (this) is writing only; nothing ran. **Proposed; not frozen.** It freezes
when Blocked 104 is decided; the tolerances and gate are then frozen with a preflight report in
unit 1, before any scored output (as `design-C2.md` "Unit 3s").

## What this opening of C5 is for

One question from the referee's M2 (`referee-C2.md`): C2's unit 5b showed that V at β = 1 nearly
counts the layers at which the tokens' mean norm ‖m‖² falls, and that the mean's term carries
all but 1 of the 1224 drops at 128–1000. **Is that mean the massive-activation coordinates?**
Project them out of every token, recount, and see whether the two dated events (the break at
256 or 512, the centred event at 1000) stay where they are.

Only M2. C5's other items (M1, M5's rows, M6, M7, M8, R's own break) stay in the ROADMAP row,
unscheduled; under 103 (a) the next thread after M2 is C2p, then C4. This opening closes when
M2 is read.

## Definitions

| item | fixed as | why |
|---|---|---|
| the states | the 25 hidden states per (seed, step, v1 prompt), float32, eager, TF32 off, exactly as `c2_seeds.run` (A5's e1 reproduced ≤ 1e-4, the tokens' hash equal, else refuse) | the records C2 dated; the known answer |
| a coordinate set's projection | zero the coordinates in the set in each token's raw state, then L2-normalise (float64); a row shorter than 1e-6 refuses | the same direction as zeroing them in the normalised token and re-normalising, so one definition; it removes the coordinates' weight in each token's norm, which is how they dominate the sphere |
| the energies | E₁ on the projected cloud (uncentred), and on its token-centred, re-normalised cloud (`c2_aniso.centred`) | C2's two counts, after the projection |
| **the set S (per seed)** | read on the seed's own **step 143000** states, the 7 v1 prompts: at each of the 25 states, each coordinate's max \|x\| over all tokens and prompts, divided by the median of those maxima over the 1024 coordinates (S1's `over_median`); **S = the coordinates at ≥ 100 at any state.** Written to disk before any other step of that seed is read | per seed, since the carriers differ across seeds (2610.00423); S1's statistic, so seed 0's set is checkable against `massive.json` (357, 130, 966 reach 100 at states 6–22; 278, 335, 914 never pass 21); 100× not Sun's 1000×, which 410m does not reach (`lit-C5.md` §1). Empty S, or more than 16 coordinates: refuse for that seed, reported |
| the random control | per seed, 20 sets of \|S\| coordinates drawn uniformly outside S (seeded), each projected the same way | 2607.09279's control: removing any \|S\| coordinates of 1024 |
| the dating | `a3.d4a`'s half-rise on each count's own plateau (1000–4000), A5's D2 (a) | C2's rule unchanged; dates only, no size compared across counts (`LESSONS.md` 6, unit 3s) |

## Readings (the scored family: seeds 1–9 × the 18 grid steps × the 7 v1 prompts)

Seed 0 is read first as the development set and the first-output check (its uncentred counts and
‖m‖² profile have been seen), as in unit 3s. Nothing outside the family enters a reading.

| # | reading | fixed as |
|---|---|---|
| R0 | **the share** (description) | per (seed, step, state), q = ‖m_S‖² / ‖m‖², m the token mean of the normalised (unprojected) cloud, m_S its coordinates in S, prompt-averaged; the 20 random sets' median q beside. Reported first, at 128–1000 at the states where ‖m‖² falls |
| R0b | **the coordinates' own date** (description) | per seed, the first grid step at which some coordinate of S passes 100× at some state, with S's largest ratio per step beside |
| P1 | **the break, projected** | per seed, the projected uncentred half-rise at 1e-3, with 1e-4 and 1e-2 (A5's sensitivity tolerances): **kept** if it equals the seed's uncentred break (A5's date) at all three, **moved** if later at all three, else **mixed**; unreadable (plateau 0, or a step 0–128 count ≥ half of it) reads mixed, flagged. **Specific** if at least 18 of the 20 random sets keep the date at 1e-3; else the seed's P1 is reported "not specific" |
| P2 | **the centred event, projected** | per seed, the projected centred half-rise at unit 3s's four arms (`c2_seeds.TAU_ARMS`, re-gated on the projected counts in unit 1's preflight): **kept** if at 1000 at all four arms, **moved** if at another step at all four, else **mixed**; specificity as P1 |
| tally | of 9 | K_kept ≥ 7 "kept"; K_moved (specific) ≥ 7 "the coordinates'"; else "mixed": the count reported and the user decides (as 3s's tally) |

**What each outcome changes (fixed now).**

| outcome | §4.1 and the abstract (in C2p) |
|---|---|
| P1 kept | "the mean whose falls V counts at the break is not carried by the massive-activation coordinates": the projection leaves the date on K of 9 seeds; R0's share and R0b's date beside |
| P1 moved, specific | the break at its step is the coordinates' falls: §4.1, §1, the abstract and the title rewritten; whether the draft keeps its headline goes to the user (a Blocked item) before C2p edits it |
| P2 kept / moved | likewise for the centred event; "moved" with R0b at 1000 would match 2510.06477's "around 1k" |
| mixed, or not specific | the counts reported as such; the user decides what §4.1 says |

**A lean, stated before any output** (`lit-C5.md` §1): 2510.06477 dates 410m's massive
activations, sink and anisotropy p₁ to "around step 1k", so R0b may read 1000 on seed 0,
after the break: P1 would then read kept almost by construction (the coordinates are not yet
large at 512), and P2 is the reading that can move. It is a published curve on one seed, not a
reading.

## Not this opening

A causal statement (projection is a read-out, not an intervention on the forward pass: the
model still computed with the coordinates; 2607.09279's clamping is the intervention, and it
would be its own unit); the attention sink's share per head; M1's other β (R0 does not depend on
β, P1 does); any rung but 410m and the PolyPythias seeds; any separation from the LR warmup (F1,
C4).

## Exit condition and budget

**Exit:** R0, R0b, P1 and P2 read on the 9 seeds with the random control, or a recorded
refusal (an empty or oversized S, a failed gate, a known answer missed). **Budget: 2 units +
close-out**, the low end of 103 (a)'s "2–3 units"; at the budget it closes at what it knows,
unless the user extends it on the record.

| unit | what | done when |
|---|---|---|
| 0 | the scan and this design (writing only) | Blocked 104 decided; the design frozen |
| 1 | `trackC_writeup/c5_massive.py` and pure tests; the preflight (the projected centred counts re-gated per arm on unit 3s's re-inits and plants; plant A placed in S, plant B outside it); the tolerances frozen and pushed before any scored record; the run (10 seeds × 18 steps, cached checkpoints, `HF_HUB_OFFLINE`); R0–P2 read | every reading in the table above, or a refusal |
| close-out | writing only: what M2 established, its Parked items carried (C2p) or dropped; the card current | the card current |

## Decisions (each yours; recommendation first)

| # | decision | options | recommendation |
|---|---|---|---|
| 104 | the coordinate set and the scope | (a) **as above**: per seed, S read at the seed's 143000 by S1's ratio ≥ 100 (seed 0: 357, 130, 966), the 9 seeds scored and seed 0 the development set, 20 random sets per seed; 2 units + close-out. (b) **seed 0 only, S1's six** (357, 130, 966 and the early 278, 335, 914): 1 unit + close-out, no reproduction, three of the six never pass 21×. (c) **a per-step set**: S re-read at each (seed, step) by the same rule; follows the coordinates' birth, but the set changes with the step, so the projected count is a different statistic at each step and its date is not comparable with C2's. (d) **no M2**: 103 (b), C2p now | **(a).** It reads the seeds C2's claims rest on, it is per seed as the life-cycle paper requires, and it fixes one set per seed so the date is one statistic along training; (b) is cheaper by a unit and reproduces nothing; (c)'s question is R0b's, which (a) reports as description |

## Worth challenging (for the PR)

- **The set from the end state.** S is read at 143000, outside the window, so it is fixed before
  any scored step; if the carriers at 512 differ from the end state's (2610.00423's
  "stabilize early" is not dated for Pythia), P1 removes the wrong coordinates. R0 shows it: a
  small q at 512 with the random sets' q equally small says the set is not the mean's carrier
  *at that step*, which is the answer, not a failure. Rejected: (c), for the reason in 104.
- **100×.** A threshold below Sun's, chosen from 410m's end state; seed 0's set is three
  coordinates at 100× and six at 10×. The 10× set is not read here (the three early coordinates
  sit at states 1–4, where ‖m‖² is not where C2's drops are).
- **Read-out, not intervention.** Projecting coordinates out of the states says which
  coordinates the energy reads, not which ones the model needs; clamping inside the forward pass
  is a different unit.
- **The lean.** A published date makes P1 "kept" likely before any output; the design keeps
  P1 because a reader of §4.1 asks it, and puts the weight on P2 and R0.
