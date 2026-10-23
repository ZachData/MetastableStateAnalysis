<!-- trackC_writeup/design-C5.md -->
# Track C — design: C5 opened on M2, the massive-activation coordinates projected out

Opened 2026-10-23 by Blocked 103 (a), the user's ("C5's M2 first, then C2p"; nothing posted
to arXiv by Claude: posting stays the user's). Row: `docs/ROADMAP.md` §8 C5. Scan: `trackC_writeup/lit-C5.md`
(2026-10-23). Unit 0 was writing only; nothing ran. Revised after `/challenge-pr` on #263 (accept with changes: R0′ the
share of the falls, P1's scale-matched arm and magnitude-matched control, each arm against its
own unprojected date, the set as a union over the grid, S1's states). **Frozen 2026-10-23 on
Blocked 104 (a), the user's**, with "Unit 1, fixed before any output" below; the tolerances and
gate are frozen with a preflight report in unit 1, before any scored output (as `design-C2.md`
"Unit 3s").

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
| **the set S (per seed)** | the ratio r: at each of states 0–23 (the block inputs S1 read; the post-final-LN state 24 is not used to choose), each coordinate's max \|x\| over all tokens and prompts, divided by the median of those maxima over the 1024 coordinates (S1's `over_median`); **S = the coordinates at r ≥ 100 at any state 0–23 at any of the seed's 18 grid steps** (the union). Pass 1 reads only r at every step and writes S; pass 2 computes the projected energies; no projected energy exists before S is on disk | per seed, since the carriers differ across seeds (2610.00423); the union keeps one set (one statistic along training) and contains the carriers at 512 if any passes there, which an end-state set need not (`/challenge-pr` on #263); S1's statistic and states, so seed 0's 143000 is checkable against `massive.json` (357, 130 reach 100 at states 6–23, 966 at 6–22; 278, 335, 914 never pass 21); 100× not Sun's 1000×, which 410m does not reach (`lit-C5.md` §1). Empty S, or more than 16 coordinates: refuse for that seed, reported |
| the random control | per seed, 20 sets of \|S\| coordinates drawn uniformly outside S (seeded), each projected the same way | 2607.09279's control: removing any \|S\| coordinates of 1024. A floor only: random coordinates carry almost none of ‖m‖², so they cannot test specificity |
| **the magnitude-matched control** | per seed, the next \|S\| coordinates outside S by their largest r (over steps and states 0–23), projected the same way (seed 0's end state suggests 278, 335, 914, unread on the union) | removes large coordinates that are not the set, so it shrinks E₁'s relative drops as the set does; a date that moves under both is a scale effect, not the set's (`/challenge-pr` on #263, finding 2; `LESSONS.md` 6, unit 3) |
| the dating | `a3.d4a`'s half-rise on each count's own plateau (1000–4000), A5's D2 (a) | C2's rule unchanged; dates only, no size compared across counts (`LESSONS.md` 6, unit 3s) |

## Readings (the scored family: seeds 1–9 × the 18 grid steps × the 7 v1 prompts)

Seed 0 is read first as the development set and the first-output check (its uncentred counts and
‖m‖² profile have been seen), as in unit 3s. Nothing outside the family enters a reading.

| # | reading | fixed as |
|---|---|---|
| R0 | **the level** (description) | per (seed, step, state), q = ‖m_S‖² / ‖m‖², m the token mean of the normalised (unprojected) cloud, m_S its coordinates in S, prompt-averaged; the random and magnitude-matched sets' q beside |
| R0′ | **the share of the falls** (description; the one V counts) | at each transition that violates at 1e-3 (unprojected) at steps 128–1000, the signed share Δ‖m_S‖² / Δ‖m‖² (exact: ‖m‖² = ‖m_S‖² + ‖m_{S^c}‖²); per step the median and the fraction above ½; both controls beside. Reported first. A high level with a low share of the falls, or the reverse, is possible, so R0 alone licenses nothing about the drops (`/challenge-pr` on #263, finding 1) |
| R0b | **the coordinates' own date** (description) | per seed, the first grid step at which some coordinate of S passes 100× at some state 0–23, with S's largest ratio per step beside |
| P1 | **the break, projected** | per seed and arm, the projected uncentred half-rise against **the unprojected count's own date at the same arm** (recomputed from the same pass; A5's date at 1e-3 the known answer). Arms: 1e-4, 1e-3, 1e-2 and the **scale-matched** τ_p = 1e-3 × s_p / s_u (s the median \|relative change\| per (prompt, transition) at the off-grid step 7000, projected and not; frozen in unit 1's preflight before any scored record; s_p / s_u at every grid step beside). **Kept** if equal at all four arms, **moved** if later at all four, else **mixed**; unreadable (plateau 0, or a step 0–128 count ≥ half of it) reads mixed, flagged. **Specific** only if the magnitude-matched control keeps its own dates at all four arms *and* ≥ 18 of 20 random sets keep theirs at 1e-3; else "moved, not specific" |
| P2 | **the centred event, projected** | per seed and arm, the projected centred half-rise at unit 3s's four arms (`c2_seeds.TAU_ARMS`, re-gated on the projected counts in unit 1's preflight) against **the unprojected centred date at the same arm** (unit 3s's records; not a fixed 1000, since seed 6's own dates differ by arm): **kept** / **moved** / **mixed** as P1; specificity as P1 |
| tally | of 9 | K_kept ≥ 7 "kept"; K_moved (specific) ≥ 7 "the coordinates'"; else "mixed": the count reported and the user decides (as 3s's tally) |

**What each outcome changes (fixed now).**

| outcome | §4.1 and the abstract (in C2p) |
|---|---|
| P1 kept | "projecting out the massive-activation coordinates leaves the break's date on K of 9 seeds"; whether they carry the mean's falls is R0′'s, stated beside, with R0b's date |
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

**Exit:** R0, R0′, R0b, P1 and P2 read on the 9 seeds with both controls, or a recorded
refusal (an empty or oversized S, a failed gate, a known answer missed). **Budget: 2 units +
close-out**, the low end of 103 (a)'s "2–3 units"; at the budget it closes at what it knows,
unless the user extends it on the record. Unit 1 carries most of the work (two passes per seed,
two controls); if its run cannot be read inside the unit, the close-out reports what was read and
the rest goes to the user as an extension, not a silent unit 1b.

| unit | what | done when |
|---|---|---|
| 0 | the scan and this design (writing only) | Blocked 104 decided; the design frozen |
| 1 | `trackC_writeup/c5_massive.py` and pure tests; the preflight (the projected centred counts re-gated per arm on unit 3s's re-inits and plants, plant A placed in S, plant B outside it; τ_p and s_p / s_u at step 7000); the tolerances frozen and pushed before any scored record; the run (10 seeds × 18 steps, pass 1 for S, pass 2 for the energies, cached checkpoints, `HF_HUB_OFFLINE`); R0–P2 read | every reading in the table above, or a refusal |
| close-out | writing only: what M2 established, its Parked items carried (C2p) or dropped; the card current | the card current |

## Unit 1, fixed before any output (2026-10-23, with the freeze)

What the table above leaves open, fixed in `trackC_writeup/c5_massive.py` before pass 1 runs.

| item | fixed as |
|---|---|
| r's tokens | every token of every v1 prompt, position 0 included (S1 split position 0 from the rest; the design's "all tokens" is their union). Known answer: at seed 0's 143000, per state 0–23, S1's two top-3 lists (`massive.json`) are recomputed from the same pass, coordinates equal and ratios within 1e-3 relative, else refuse |
| the passes | `pass1` per seed writes r (24 × 1024 per step); `select` writes `set.json` (S, M, the 20 random sets, R0b's inputs); `pass2` refuses unless `set.json` exists, and its first record is opened before the batch goes on |
| the projection | computed from the raw float64 Gram by the rank-\|S\| correction (G − H_S H_Sᵀ, then the normalised and token-centred Grams from it): exact algebra, checked against the direct definition in the pure tests (≤ 1e-10) |
| the known answers in pass 2 | the unprojected e1 against A5's records (≤ 1e-4 relative, as unit 3s) and the unprojected e1c against unit 3s's records (seed 0: unit 3's), ≤ 1e-4 relative, the tokens' hash equal, else refuse |
| the unprojected references | P1: this pass's e1 (A5's float32 path; its V equal to A5's at every step, else refuse). P2: unit 3s's e1c records (seed 0: unit 3's). This pass's float64 unprojected counts beside, as a precision check on the projected path |
| M and the random sets | written to `set.json` from pass 1 alone; the random sets drawn with `numpy.random.default_rng(20261023 + seed)` |
| P1's scale-matched arm | the projected count at τ_p against the **unprojected count at 1e-3**; M at its own τ_M = 1e-3 × s_M / s_u from the same preflight; one τ_p and one τ_M, from seed 0's S on seed 0's step 7000, for every seed (only seed 0's 7000 is cached); every seed's s_p / s_u per grid step beside |
| the random sets' arm | P1: 1e-3 (as above). P2: the B-matched 7.58e-4, the centred arm calibrated to see plant B as the uncentred count sees it at 1e-3 |
| the preflight's plants | unit 3s's plants (`c2_seeds._plants`) on seed 0's S projected out: plant A's common vector placed in S (unit-normalised on S), so the projection removes it and its gate half checks the projection (share 0 up to rounding); plant B's spread zeroed on S; the gate as unit 3s (re-init rate ≤ 0.05, B at 0.25 ≥ 0.4, A at 0.5 ≤ 0.1) per arm; unit 3s's plant A (the full u) beside. A failed arm refuses the run and goes to the user |
| P1's arms on the re-inits | the projected uncentred counts' re-init violation rates at 1e-4, 1e-3, 1e-2, τ_p, beside (A5's D2 arms were not gated either) |
| unreadable | an unprojected centred arm unreadable on unit 3s's rule (seed 6 ties on one arm): that seed's P2 reads mixed, flagged |
| R0′ | at a transition with \|Δ‖m‖²\| < 1e-12 the share is not taken (counted) |
| **frozen by the preflight** (2026-10-23, before any pass-2 record) | seed 0's S = {130, 357, 966} (S1's known answer met, max deviation 0), M = {421, 550, 752}. At step 7000 s_u 0.0141, s_p 0.0161, s_M 0.0144: **τ_p = 1.15e-3, τ_M = 1.02e-3** (projection makes the relative changes larger there, not smaller). The gate passes at all four centred arms (re-init rate ≤ 0.016, B 0.46–0.94, A in S 0: its largest shift 7e-16, the projection's check). `tools/design_preflight` report sha256 `c308803892bf` refuses on its floor, as unit 3s's did: it prices an e-value this design never takes (dates only). Code `c91bfe321bda`; records `data/trackC/c5/2026-10-23-u1/preflight/` (main tree) |

## Decisions (each yours; recommendation first)

| # | decision | options | recommendation |
|---|---|---|---|
| 104 | **decided (a), the user's, 2026-10-23** · the coordinate set and the scope | (a) **as above**: per seed, S the union over the 18 grid steps of S1's ratio ≥ 100 at states 0–23 (seed 0's end state alone: 357, 130, 966), the 9 seeds scored and seed 0 the development set, 20 random sets and one magnitude-matched set per seed; 2 units + close-out. (b) **seed 0 only, S1's six** (357, 130, 966 and the early 278, 335, 914): 1 unit + close-out, no reproduction, three of the six never pass 21×. (c) **a per-step set**: S re-read at each (seed, step) by the same rule; follows the coordinates' birth, but the set changes with the step, so the projected count is a different statistic at each step and its date is not comparable with C2's. (d) **no M2**: 103 (b), C2p now | **(a).** It reads the seeds C2's claims rest on, it is per seed as the life-cycle paper requires, and it fixes one set per seed so the date is one statistic along training; (b) is cheaper by a unit and reproduces nothing; (c)'s question is R0b's, which (a) reports as description |

## Worth challenging (for the PR)

- **The set as a union over the grid** (first proposed from the end state alone; changed after
  `/challenge-pr` on #263). It is fixed before any projected energy, and holds the carriers at
  512 if any coordinate passes 100× there; if none does, R0′ says the falls at the break are not
  in large coordinates at all, which is the answer, not a failure. The cost: the union may hold
  coordinates that are large only late, which removes more than the break's carriers. Rejected:
  (c), for the reason in 104.
- **The scale.** Removing coordinates that carry ‖m‖² shrinks E₁'s relative drops, so the
  date can move later at a fixed tolerance for that reason alone (unit 3's lesson). P1 adds a
  scale-matched arm and a magnitude-matched control and calls "moved" specific only if both
  agree; the uniform random sets are a floor, not the control.
- **100×.** A threshold below Sun's, chosen from 410m's end state; seed 0's set is three
  coordinates at 100× and six at 10×. The 10× set is not read here (the three early coordinates
  sit at states 1–4, where ‖m‖² is not where C2's drops are).
- **Read-out, not intervention.** Projecting coordinates out of the states says which
  coordinates the energy reads, not which ones the model needs; clamping inside the forward pass
  is a different unit.
- **The lean.** A published date makes P1 "kept" likely before any output; the design keeps
  P1 because a reader of §4.1 asks it, and puts the weight on P2 and R0.
