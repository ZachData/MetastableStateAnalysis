<!-- trackB_circuit/status-B.md -->
# Track B — STATUS

<!-- phase-card -->
## Card

- **Question:** Does a paired ± nudge of one token's residual, gated against planted effects and random directions (B0), see a known circuit's particle signature (each token pulled toward its left neighbour inside `L5H2`'s output, ranked against layer 5's other heads, B1) switch on over training, so that "small" has a yardstick before the harness reads unknown mechanisms?
- **Inputs:** `pythia-410m`, the 18 Stage 0 steps, the 7 scored v1 prompts (B0's `prompts_hash`), `resid_pre` at layer 5, 32 seeded positions and K = 32 seeded random directions per prompt, c ∈ {0.01, 0.03, 0.1, 0.3}, GPU float32 — `trackB_circuit/design-B.md` "B1"
- **Results:**
  - Opened 2026-10-13 with a scan, a `sympy` check of the paired-pass algebra and a frozen design (3-unit budget) — `trackB_circuit/design-B.md` "Exit condition and budget"
  - B0's harness passes its calibration gate at every step and arm but one near-tied G4b arm at step 128, waived by the user as that arm only (Blocked 44 (a)) — `trackB_circuit/status-B.md` "B0 over 18 steps"
  - Loss gradients at layer 5 are aligned across prompts at every step, most at 64–512, so a structured direction ranked against random ones is biased upward by the model itself — `trackB_circuit/status-B.md` "Fences (from B0's gate)"
  - B1 passes by its revised pass rule (off at 0, on at 1000–8000): `L5H2`'s signature reads off at steps 0–256 and on at every step from 512; the onset falls in (256, 512], inside the range recorded before the run and before ROADMAP's first window (512, 4000]; read from block 5 alone, on one head's output — `trackB_circuit/status-B.md` "B1 (2026-10-14)"
  - B2a (the 2D spike) opened 2026-10-15 with a scan, a `sympy` check (a head's own QK is exact on a subspace of dimension ≤ 128) and a design not yet frozen; its reduction is 2609.28448's general-value-map section, on V's invariant subspaces — `trackB_circuit/design-B2a.md` "The paper, corrected"
  - B2a's instrument passes its planted gate (circle, rotation, D = 4, positional, null 20 of 20, out of class, arithmetic) at the design's constants; on the weights V's top pair is a D = 2 block (complex or degenerate) in 123 of 252 cells, near-repulsive in 10, and no V block beats the kernel in 31 cells at 1000–32000 — `trackB_circuit/status-B.md` "B2a unit 1 (2026-10-15)"
  - At step 143000 the reduction's premise fails: the real tokens hold about the random share of their norm² in V's invariant subspaces, so no head of 14 has D\* ≥ 1 (no circle); 4 read positional (D\* = 0, `L5H2` post hoc), 8 outside the paper's class (at the placed 0.5), 2 "none"; the isotropic cosine adds at most 0.08 of TV over position — `trackB_circuit/status-B.md` "B2a unit 2 (2026-10-16)"
  - The rule at all 18 steps finds no head of 14 with D\* ≥ 1 at any step: to 256 attention is near uniform and every head reads outside (nothing to fit); positional heads appear from 512 (`L5H2`, `L4H9`), and V's subspaces never carry the content a head reads; B2a is at its exit with its 4 units spent — `trackB_circuit/status-B.md` "B2a unit 3 (2026-10-16)"
  - B2a closed 2026-10-16 at its exit: on 410m no head's attention lives in a small subspace of V, mostly because trained heads are not isotropic content plus position (214 of 252 cells outside); where V's ladder was reached the tokens hold about chance in V's subspaces; V's 2D blocks are at the random-weight rate, training shows in their sign — `trackB_circuit/status-B.md` "B2a on one page (closed 2026-10-16)"
- **Superseded / wrong:** none
- **Registry:** none, because B1 is a positive control on an exploration rung — `trackB_circuit/design-B.md` "Fences"
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - From step 1000 the statistic's size is a float32 detection limit, not a measurement, so only its rank is read — `trackB_circuit/status-B.md` "B1 (2026-10-14)"
  - B1 reads one head's output after block 5 alone; the multi-layer path and the residual and loss readouts B2–B4 use are covered only by B0's gate — `trackB_circuit/status-B.md` "B1 (2026-10-14)"
  - G4b's false-fail rate is not derived, and G2 was checked only at its four design positions — `trackB_circuit/status-B.md` "B0 over 18 steps"
- **After Phase 10:**
  - B2–B4 on this harness, with an on-distribution null for B4 *(forward pass, GPU minutes per step)*
  - The weight-space site (a head's OV moved by ε), gated on its own *(forward pass, GPU minutes per step)*
- **Reviewed:** 2026-10-16 · body `60e612323b`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| 1. design, scan, `sympy` check | **done 2026-10-13**; frozen by the user 2026-10-14 (Blocked 42) | `design-B.md`, `lit-B.md`, `tools/math_checks/b0_antisymmetric_nudge.py` |
| 2. B0, the harness and its gate | **done 2026-10-14**: G0–G4a pass at every step; G4b at 17 of 18, failing at step 128 on one near-tied arm, **waived as that arm only** (Blocked 44 (a), the user's, 2026-10-14; `design-B.md` "If any arm fails") | `core/nudge.py`, `core/position.py`, `trackB_circuit/b0_gate.py`; "B0 over 18 steps" below |
| 3. B1, `L5H2` as positive control | **done 2026-10-14: passes** ("sees") | `trackB_circuit/b1_prev_token.py`; "B1 (2026-10-14)" below |

Budget: 3 units (`design-B.md` "Exit condition and budget"), all spent: Track B's opening is
closed at its exit condition. What follows on these instruments (B2 onward) is `docs/ROADMAP.md`
§7, after Track A (Blocked 40 (d)).

**B2a** (its own budget, `design-B2a.md` "Exit condition and budget"; Blocked 50 sets it):

| unit | state | where |
|---|---|---|
| 0. design, scan, `sympy` check | **done 2026-10-15**; frozen by the user 2026-10-15 (Blocked 50 (a): 4 units, M1 primary) | `design-B2a.md`, `lit-B.md` "B2a", `tools/math_checks/b2a_subspace_ceiling.py` |
| 1. weights only, planted gate | **done 2026-10-15: every arm passes**; weights rows written | `trackB_circuit/b2a.py`; "B2a unit 1 (2026-10-15)" below |
| 2. real tokens at 143000 | **done 2026-10-16: a verdict for all 14 heads; no head has D\* ≥ 1** | "B2a unit 2 (2026-10-16)" below |
| 3. the 18 steps | **done 2026-10-16 (Blocked 53 (a), occupancy first; then the rule at all 18 steps after `/challenge-pr` on #210): no head has D\* ≥ 1 at any step; B2a is at its exit, budget spent** | "B2a unit 3 (2026-10-16)" below |

**B2a closed 2026-10-16** at its exit condition (Blocked 55 (a), the user's): "B2a on one page" below.

## B2a on one page (closed 2026-10-16)

**Asked:** for each of 14 heads (`b2a.heads()`), the smallest subspace on V's invariant ladder whose
reduced dynamics (M1: `β cos(Uᵀu_i, Uᵀu_j) + φ(i − j)`, 2609.28448 §6.1) reproduce the head's real
attention; D = 2 is the paper's circle. **Exit met** (`design-B2a.md` "Exit condition and budget"):
a verdict per head at 143000 (unit 2) and at all 18 steps (unit 3), in the 4 units of Blocked 50.

**Established (tier 1, 410m, v1's 7 prompts `976b62d677bc`, 18 steps, 14 heads):**

1. **No head has D\* ≥ 1 at any step (no circle).** Of the 252 (head, step) cells, 214 read
   outside, 32 positional (D\* = 0), 6 "none"; a cell reaches V's ladder only when M1's full fit is
   inside and content adds, so the Haar nulls ran in 5 (`L9H8` at 8000, `L10H7` at 16000–143000),
   all "none". The largest gain any V rung gives over position alone is 0.092 (`L18H1`, 2000).
   (`/challenge-pr` on #211, findings 1–2: first read "no V subspace below its Haar nulls" and
   "because the tokens do not sit in V's subspaces".) "B2a unit 3".
2. **Why, mostly: the heads are not in M1's class.** To 256 attention is near uniform and nothing
   adds (nothing to fit); after it most heads read outside because even M1's full isotropic cosine
   plus position misses (TV(full) > 0.5). Where M1 fits, it fits by position. So "most trained
   heads on 410m are not isotropic content plus position" is the main reading, and the ladder was
   seldom reached.
3. **Where the ladder was reached, the tokens do not occupy V's subspaces.** Their share of norm²
   there is about chance (U₂ ≤ 0.01 at 143000; over 18 steps only `L5H2` late and `L18H1` stand
   outside step 0's spread), so content that helps (`L7H8`, `L10H7`) is carried by no V subspace.
   Occupancy alone does not decide a head: `L18H1` holds 37× chance in U₂ at 143000 and still
   reads outside (TV(full) 0.717). "B2a unit 2", "unit 3".
4. **Positional heads.** From 512: `L5H2`, `L4H9` (B1's onset for `L5H2` is (256, 512]), `L12H5`
   from 2000; 143000's 4 hold from 16000.
5. **V's 2D blocks are at the random-weight rate; training shows in their sign.** V's top pair is
   a D = 2 block in 123 of 252 cells, but in 7 of 14 heads at step 0 and 68 of 126 cells at
   0–256 against 55 of 126 from 512 (`/challenge-pr` on #211, finding 3). What training adds:
   near-repulsive top pairs in 10 cells, all at 256–32000, none at init; no V block beats the
   kernel in 31 cells, all at 1000–32000 (the OV repulsive phase, `PROJECT.md` §3.12 A). "B2a unit 1".
6. **The instrument sees what it should.** The planted gate passes every arm at the design's
   constants (a faithful rung at TV 0.005, the nearest random subspace ≥ 0.1); G0 at every step.
7. design-A's prior (2D fails for previous-token and induction heads) is not contradicted.

**Not established:**

- Anything off V's ladder: D\* is the smallest *on the ladder*, an upper bound (design "Worth
  challenging"). Whether a head's QK read space is low-rank (M2's ladder, read only beside) is a
  question about QK rank, not the paper's reduction (design "Worth challenging"); that a head reads
  a token only through `W_Q y`, `W_K y` is true by construction (`/challenge-pr` on #211, finding 4).
- Dynamics: each head acts once; whether depth iterates the map, and whether the dynamics
  *selects* where tokens sit, are untested (design "Fences").
- The 4 positional / 8 outside split at 143000 rests on the placed 0.5 (at 0.55 `L6H4`, `L10H14`
  read D\* = 0); no D\* ≥ 1 either way. `L5H2`'s 143000 verdict is post hoc (Blocked 52).
- Heads outside the 14, prompts outside v1, other rungs (410m is exploration; nothing registered).
- A1a's β against M1's: opposite signs in 6 of 14 heads (all M1 −, A1a +), not resolved here.

**Open threads, at close:**

| thread | at close |
|---|---|
| The gate has no position-0 sink arm | **dropped**: the rule excludes position 0 and renormalises, and a softmax row with one key removed and renormalised is exactly the softmax over the rest, so a planted sink changes nothing the rule reads (`/challenge-pr` on #211) |
| The gate has no arm with content only partly V's | **dropped**: it bears only on the 6 cells that reached V's ladder, and there reading 3 (occupancy and M2, independent of the gate) says the content is not in V's subspaces |
| Token-matched null | dropped at unit 3 ("Parked" below) |
| A1a's β against M1's | **carried** to A2 as its first item (Blocked 54 (a)) |
| QK read rank (M2's ladder: how much of the token a head reads; a QK-rank question, not the paper's reduction) | **carried** to B2 (`docs/ROADMAP.md` §7), beside its QK spectra |
| `L10H7` "none" (content helps, V carries none of it), while its V's top pair is near-repulsive at 2000 and 32000 and no V block beats the kernel at 1000–8000 (unit 1's `weights.json`, read at close) | **carried** to B6 (`docs/ROADMAP.md` §7), beside its 410m `L10H7` line |

## B2a unit 3 (2026-10-16): the 18 steps

**Input:** v1's 7 prompts (`976b62d677bc`), Stage 0's runs at each step, the 14 heads; HF forward on
the RTX 3080 float32 (TF32 off), rebuild and fits float64. Two batches (`design-B2a.md` "Unit 3 as
built" and its amendment): the occupancy screen and the rule at its 6 steps, code `37d4f2487738`
(commit `c03076a`), records `/run/media/system/WDS_500/Mets/data/b2a/2026-10-16-occ/`, 42 min; then,
after `/challenge-pr` on #210 found the screen's baseline wrong (the user's pick), the rule at the
other 11 steps, code `939e114f7afb` (commit `e0063b4`; the rule's code unchanged), records
`.../2026-10-16-occ2/` (its `occupancy.json` is the one to read: it carries `chance_ratio`), 61 min;
143000 is unit 2's. Re-run: `python -m trackB_circuit.b2a occupancy --out <dir> --unit2 <unit 2's
real.json>`, `... gate --out <dir>`, `... real --out <dir>/step<S> --gate <dir>/gate.json --step <S>`
per step, `... report --out <dir>`.

**Checks.** Both passes reproduce unit 2's rung occupancies at 143000 exactly (max |Δ| 0). Gate 9 of 9
on each code. G0 passes at all 17 steps (worst scaled 3.80, step 4; bound 10).

**Occupancy** (`b2a report`; the statistic is a median, whose Haar chance is 0.69 × D/d at D = 2 and
0.79 × at D = 3, `chance_ratio`; U₂ moves up to U₃ for `L7H8`, `L9H1`, `L18H1` at 143000, and the set
varies by step). Over chance, U₂ is 0.4–1.8× at step 0 (random weights: the spread of a median over
7 prompts) and up to 2.7–3.4× in some head from 16 on; U₆₄ is at chance to 16, then 1.2–2.2× in most
heads from 512. Only `L5H2` late (7.6× chance at 143000) and `L18H1` (8.2× at 1000, 37× at 143000)
stand well outside step 0's spread. The screen (Blocked 53 (a)'s bar, set against the mean) chose
7 steps; it did not decide anything, since the rule then ran at all 18.

**The rule at every step** (143000 from unit 2):

| step | D\* ≥ 1 | D\* = 0 | outside (TV(full) > 0.5 / nothing adds) | none | largest V-rung gain over D = 0 |
|---|---|---|---|---|---|
| 0, 2, 4, 8, 16 | 0 | 0 | 14 (0 / 14) | 0 | ≤ 0.001 |
| 32, 64, 128, 256 | 0 | 0 | 14 (0 / 14) | 0 | 0.003, 0.010, 0.019, 0.015 |
| 512 | 0 | 2 (`L5H2`, `L4H9`) | 12 (1 / 11) | 0 | 0.017 (`L18H1`) |
| 1000 | 0 | 2 (same) | 12 (3 / 9) | 0 | 0.047 (`L18H1`) |
| 2000 | 0 | 3 (+ `L12H5`) | 11 (10 / 1) | 0 | 0.092 (`L18H1`, U₂ 1.2× chance) |
| 4000 | 0 | 3 (same) | 11 (10 / 1) | 0 | 0.058 (`L7H8`) |
| 8000 | 0 | 5 (+ `L10H7`, `L6H4`) | 8 (7 / 1) | 1 (`L9H8`) | 0.047 (`L9H8`) |
| 16000 | 0 | 5 (`L5H2`, `L12H5`, `L4H9`, `L6H4`, `L9H1`) | 8 (8 / 0) | 1 (`L10H7`) | 0.037 (`L9H8`) |
| 32000, 54000 | 0 | 4 (`L5H2`, `L12H5`, `L4H9`, `L9H1`) | 9 (9 / 0) | 1 (`L10H7`) | 0.038 (`L9H8`) |
| 143000 | 0 | 4 (same) | 8 (8 / 0) | 2 (`L7H8`, `L10H7`) | 0.037 (`L9H8`) |

| reading | what it rests on |
|---|---|
| **No head of the 14 has D\* ≥ 1 at any of the 18 steps (no circle), on v1; the Haar nulls ran in 5 of 252 cells, the rest decided before the ladder** (corrected 2026-10-16, `/challenge-pr` on #211, finding 2: it first read "no V subspace up to D = 64 below its nulls"). This is the rule's verdict at every step, not the occupancy screen's. B2a is at its exit with its 4 units spent | `step<S>/real.json` in both record dirs |
| To 256, attention is near uniform (TV(uniform) 0.06–0.32) and every head reads outside by Blocked 51/52's clause (neither content nor position adds 0.05): "M1 has nothing to fit", not "M1 fits badly". The clause decided no head at 143000 | `real.json` "outside_why", "tv_uniform" |
| Positional heads appear from 512 (`L5H2`, `L4H9`; B1's onset for `L5H2` is (256, 512]), `L12H5` from 2000; `L10H7` (8000), `L6H4` (8000–16000) and `L9H1` (16000 on) come and go; 143000's 4 hold from 16000 | the table; "B1 (2026-10-14)" |
| The largest gain any V rung gives over position alone is 0.092 (`L18H1` at 2000, where its U₂ share is 1.2× chance, inside step 0's spread); where content helps (`L7H8`, `L9H8`, `L10H7`, `L18H1`, up to 0.086 from 2000), no V subspace carries it at any step | "content adds", "rungs" per step |

**Not established:** anything about heads outside the 14 or prompts outside v1; whether a token-matched
null would change a verdict (no V rung beat its Haar nulls at any step, so there is nothing for it to demote).

## B2a unit 2 (2026-10-16): real tokens at 143000

**Input:** code `b2a.code_sha()` = `9fe738d5a008` (commit `2f8c092`), v1's 7 prompts
(`prompts_hash` `976b62d677bc`), Stage 0's stored runs at 143000 (`trackA_force.a1a.run_dir`), HF's
forward on the RTX 3080 float32 (TF32 off), rebuild and fits float64. Records
`/run/media/system/WDS_500/Mets/data/b2a/2026-10-16/` (`gate.json`, `real.json`, `run.log`); 22 min,
16 of them the gate. Rule: design "The rule", with Blocked 51 (a) and 52 (a′) (both the user's, taken
before the batch; 52 after one head, `L5H2`, was read in a smoke test, so `L5H2`'s verdict is post hoc).
Re-run: `python -m trackB_circuit.b2a gate --out <dir>`, then `... real --out <dir> --gate <dir>/gate.json`,
`... report --out <dir>`.

**Checks.** The gate reran on this code and passes all 9 arms: the 7 of unit 1 unchanged, **G4w**
outside at both scales (TV(full) = TV(0) = 0.195, 0.378) and **G2n** D\* = 0 at both (TV(full) 0.163,
0.324 against TV(uniform) 0.449, 0.529). **G0** passes on all 98 (head, prompt) cells: worst
`L13H8`/`camus_letranger`, 2.8e-6 raw, 1.58 of the bound 10.

**The rule's verdicts** (TV is the mean over prompts of the per-prompt mean over rows; "content adds" =
TV(0) − TV(full), "position adds" = TV(uniform) − TV(0)):

| head | set | verdict | TV(full) | content adds | position adds | best V rung's gain over D = 0 (D ≤ 64) | M2 on QK's top 64 | sink |
|---|---|---|---|---|---|---|---|---|
| `L5H2` | previous-token | **D\* = 0** (post hoc) | 0.125 | 0.000 | 0.820 | 0.000 | 0.063 | 0.00 |
| `L7H8` | induction | none | 0.468 | 0.062 | 0.397 | 0.000 | 0.094 | 0.84 |
| `L12H5` | induction | **D\* = 0** | 0.412 | 0.006 | 0.478 | 0.013 | 0.221 | 0.38 |
| `L8H6` | induction | outside | 0.811 | 0.000 | 0.020 | 0.003 | 0.448 | 0.80 |
| `L11H14` | induction | outside | 0.742 | 0.000 | 0.025 | 0.000 | 0.554 | 0.77 |
| `L10H7` | negative interactor | none | 0.469 | 0.072 | 0.088 | 0.024 | 0.309 | 0.60 |
| `L6H4` | reference | outside | 0.522 | −0.010 | 0.262 | 0.000 | 0.335 | 0.65 |
| `L9H8` | reference | outside | 0.542 | 0.079 | 0.143 | 0.037 | 0.311 | 0.58 |
| `L4H9` | reference | **D\* = 0** | 0.261 | 0.000 | 0.646 | 0.000 | 0.135 | 0.00 |
| `L15H7` | reference | outside | 0.648 | −0.029 | 0.235 | 0.003 | 0.562 | 0.57 |
| `L13H8` | reference | outside | 0.598 | −0.003 | 0.099 | 0.001 | 0.306 | 0.43 |
| `L9H1` | reference | **D\* = 0** | 0.460 | 0.006 | 0.238 | 0.009 | 0.222 | 0.51 |
| `L18H1` | reference | outside | 0.717 | −0.009 | 0.020 | 0.000 | 0.624 | 0.75 |
| `L10H14` | reference | outside | 0.524 | −0.004 | 0.290 | 0.004 | 0.230 | 0.75 |

| reading | what it rests on |
|---|---|
| **The reduction's premise fails at 143000: the real tokens do not lie in V's invariant subspaces.** Their share of norm² in U₂ is ≤ 0.01 (below the median's Haar chance, 0.69 × 2/1024, in 8 of the 11 heads whose U₂ is 2-dimensional; above it `L5H2` 7.6×, `L6H4`, `L12H5` near it; corrected 2026-10-16, `/challenge-pr` on #210, finding 1: it first read "below the random 2/1024 in 11 of 12", against the mean), and in U₆₄ 0.6–5.3× the random 64/1024, ≤ 0.13. So no head has D\* ≥ 1 (no circle), and TV on V's ladder is flat in D (largest gain of any rung over D = 0: 0.037, `L9H8`); given the occupancy this is close to expected, not a separate finding (`/challenge-pr` on #209, finding 1) | `real.json` "rungs" "occupancy" |
| M1's isotropic cosine adds at most 0.079 of TV over the offset profile in any head; in 11 of 14 it adds < 0.01 (negative values, down to −0.029: the full model adds one parameter, β, and a β fitted on 6 prompts transfers worse to the 7th than none) | "content adds" |
| 4 heads are positional (D\* = 0): `L5H2`, `L4H9` fitted well (0.125, 0.261), `L12H5`, `L9H1` only as well as M1 fits at all (0.41, 0.46). 8 are outside, **all by TV(full) > 0.5; the clause Blocked 51 and 52 added decided no head**, and the frozen rule gives the same 14 verdicts. Under 51 (a) as first built, all 4 positional heads would have read outside. **The 4 / 8 split rests on the placed 0.5:** `L6H4`, `L10H14`, `L9H8` are outside at 0.522–0.542; at 0.55 the first two would read D\* = 0 and `L9H8` would go to its nulls at D = 64. No D\* ≥ 1 either way | "position adds" ≥ 0.24 for the 4 |
| `L7H8` and `L10H7` are "none": content helps (0.062, 0.072), but no V subspace carries it. `L10H7`'s D = 64 rung meets the TV clause (0.518 ≤ 0.519), and the first 32 Haar subspaces of dimension 64 all tie or beat it (min 0.499; occupancy V 0.07 against the nulls' median 0.06) | `real.json` "nulls" |
| **The content a head reads is in its QK read space, not in V's invariant subspaces** (a consequence of the occupancy row, not independent of it). M2 (the head's own logit with y projected) on V's ladder stays at its bias-only value (0.63–0.94) at every rung; on QK's ladder it falls to 0.06–0.62 at D = 64 and is exact at the full rank 128 (G5 (ii)) | `real.json` "m2" |
| The prior (design-A: 2D fails for previous-token and induction heads) is not contradicted: `L5H2` D\* = 0, the 4 induction members D\* = 0, none, outside, outside | the table |

**Beside.** γ (the head's output over its LN1 row, as γ/(1 − γ)) 0.005–0.146, largest `L5H2`. The
motion cosine at full D is 0.49–0.99 (median over tokens) but mostly V's own direction, since the reduced
and real updates share V; at D = 2 it is −0.10 to 0.38. Occupancy of V's U₂ by the real tokens is ≤ 0.01
except `L18H1` (0.09 at D = 3).

**A1a's β against M1's at full D (the design's "must reproduce it in sign and order"): sign agrees in 8
of 14.** The 6 that disagree (`L6H4`, `L4H9`, `L15H7`, `L13H8`, `L18H1`, `L10H14`) are the heads where
content adds ≤ 0; where it adds ≥ 0.05 (`L7H8`, `L10H7`, `L9H8`) the signs agree and the order is
the same (M1 7.9 > 6.6 > 5.6; A1a 15.5 > 10.3 > 9.8). The two are different estimators: A1a's
`fe_full` is least squares on log A with per-offset effects, every pair weighted equally; M1 is
cross-entropy, weighted by attention mass. **The disagreement has a direction** (`/challenge-pr` on #209,
finding 4): A1a's β is positive in 13 of 14 heads, and all 6 disagreements are M1 negative, A1a
positive, so one estimator is biased where the cosine explains nothing, not both noisy. The check
fails as written; nothing in B2a's rule reads it (A1a's β is beside only, design "Fences"), but A1's
layer β is the median of these per-head βs, so it goes to the user as Blocked 54.

## B2a unit 1 (2026-10-15): the gate and the weights rows

**Input:** code `b2a.code_sha()` = `72882e6f353d` (commit `32ddebd`), records
`/run/media/system/WDS_500/Mets/data/b2a/2026-10-15/` (`gate.json`, `weights.json`, `run.log`).
Gate: synthetic, d = 1024, n = 500, 7 prompts per arm, float64 on the RTX 3080, 9 min. Weights:
the 18 steps, the 14 heads (`b2a.heads()`: the 6 named, then `L6H4`, `L9H8`, `L4H9`, `L15H7`,
`L13H8`, `L9H1`, `L18H1`, `L10H14` drawn at seed 20261015), float64 CPU, 27 s. Constructions
as built, not as the design's table: `design-B2a.md` "Unit 1's gate, as built".
Re-run: `python -m trackB_circuit.b2a gate --out <dir>`, `... weights --out <dir>`, `... report --out <dir>`.

**The gate: every arm passes, with the design's constants (TV 0.5, +0.05) unchanged.**

| arm | verdict | numbers |
|---|---|---|
| G1, circle | D\* = 2, degenerate negative, angle 0 | TV(0) 0.566, TV(1) 0.448, TV(2) 0.0047; 320 nulls at D = 2: min 0.127, median 0.404 |
| G1r, rotation | D\* = 2, rotation | TV(2) 0.0047; nulls min 0.108 |
| G1d, D = 4 | D\* = 4 | TV(2) 0.389, TV(3) 0.260, TV(4) 0.0045; nulls min 0.245 |
| G2, positional | D\* = 0 | TV 2e-9 |
| G3, null | none in 20 of 20 | the best ladder rung's TV 0.099–0.152 against TV(full) ≈ 0, so no rung reached its null comparison |
| G4, out of class | outside | TV(full) 0.815 |
| G5 | (i) 1.6e-15; (ii) 1.2e-11; (iii) 0.81 of the bound (raw 3.8e-4, at `L18H1`'s \|logit\| 8.2e3) | (ii), (iii) in `weights.json` "g5_real" |

The margins are wide: a faithful rung sits at 0.005 and the nearest random subspace at ≥ 0.1. The
gate does not test a head whose content subspace is only partly V's (V's ladder is fixed), nor
attention with a sink (synthetic prompts have no position 0).

**Weights rows (252 cells; V in the LN1 frame at γ = 0.1, placed), descriptive:**

| row | reading |
|---|---|
| \|λ₁\| | 0.013–1.93; ≈ 0.014 for every head at steps 0–256, then growing to a peak at 16000–54000 and falling at 143000 |
| U₂'s type | a complex pair 120 (the paper's rotating circle `W = R_φ`, also a D = 2 case; 8 within 0.05 rad of π, 1 within 0.05 of 0), mixed 129, degenerate negative 2 (`L12H5` at 512, `L15H7` at 4000), degenerate positive 1 (`L10H14` at 16000). **Near-repulsive** (degenerate negative, or a pair within 0.05 rad of π): **10 of 252**; any D = 2 top pair (a complex pair or a degenerate real one): 123. At 143000: a pair for `L5H2`, `L12H5`, `L9H8`, `L4H9`, `L13H8`, `L10H14`; mixed for the rest. (Labelling: a real pair counts as degenerate within 5 %, a complex pair at any angle; `/challenge-pr` on #208, finding 4) |
| kernel beaten | in 221 of 252 cells; the 31 where no block beats `1 − γ` are at 1000–32000 (10 at 1000, 8 at 2000, 7 at 4000), where the top block is a negative real: the OV repulsive phase on these heads (`PROJECT.md` §3.12 A) |
| ρ_⊥ where beaten | 0.949–0.99999: at \|λ\| ≲ 2 and γ = 0.1, \|b\| ≈ 0.9 + 0.1 Re λ, so one application selects nothing; γ is read in unit 2 |
| selection order | the paper's top block (by \|b\|) is V's top by \|λ\| in 64 of 252 cells (by Re λ, γ's small-γ limit: the same 64); V's ladder and the paper's selection order disagree in most cells, as the design expected for negative λ |

## B1 (2026-10-14): `L5H2` as positive control

**Input:** as the card; B0's records `2026-10-14c` (code `1e78e2be9d94`), accepted by
`check_gate` under the waiver. Code `b1_prev_token.code_sha()` in every record. Records:
`/run/media/system/HDD_1TB/mets_data/trackB_b1/2026-10-14b/b1_step*.json`, `b1.json`, `run.log`,
on code `70b28da06186`; the first run (`2026-10-14/`, code `07b452b5a7c2`, a docstring apart)
gives the same `b1.json` steps exactly.
36 s per step on the RTX 3080, 11 min in all.

**Verdict by the frozen rule: PASS: the instrument sees `L5H2`'s signature switch on**, by the
pass rule as revised by `/challenge-pr` on #197, finding 1. The design's exit sentence and
ROADMAP's first window, (512, 4000], are not met as written: the signature is already on at 512. Off at
step 0, on at 1000, 2000, 4000 and 8000. The reading at c = 0.03 (s = log(σ₁ / median σ₂…σ₅),
median over 32 positions; *on* = `L5H2` above ≥ 14 of layer 5's other 15 heads):

| step | 0–8 | 16 | 32 | 64 | 128 | 256 | **512** | 1000 | 2000–54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|
| reading | off | off | off | off | off | off | **on** | on | on | on |
| prompts on (of 7) | 2 | 2 | 2 | 1 | 1 | 3 | 7 | 7 | 7 | 7 |
| `L5H2`'s s, median over prompts | 0.02–0.05 | 0.01 | −0.01 | 0.05 | 0.06 | 0.01 | 2.60 | 9.07 | 8.84–9.23 | 7.97 |
| cos(A, OV-mapped nudge) at k = 1 | 0.98 | 0.99 | 0.98 | 0.97 | 0.97 | 0.97 | 0.85 | 0.99 | ≥ 0.997 | 0.997 |
| ‖A‖ / r_out at k = 1 (the yardstick) | 1.1e-5 | 8.7e-6 | 8.8e-6 | 9.4e-6 | 1.1e-5 | 8.8e-6 | 6.8e-4 | 3.0e-3 | 3.2–3.6e-3 | 2.7e-3 |

(Every step, per prompt and at the other three c: `b1.json`; the reading is the same at all
four c at every step.)

- **Onset in (256, 512].** Recorded before the run: (256, 1000]. Steps 2–512 were to be recorded,
  not judged; 512 reads on at 7 of 7.
- **The yardstick for "small"** (the one place this number lives): `L5H2`'s offset −1 response,
  ‖A‖ / (c · r_out) with r_out block 5's RMS output norm, is **0.10–0.12** from 1000 to 54000
  and 0.09 at 143000, against ~3e-4 before 512: a factor of ~300. It is linear in c up to 0.1
  and 5–7 % lower at c = 0.3 (‖A‖/r_out at 4000: 1.2e-3, 3.6e-3, 1.2e-2, 3.3e-2 at the four c;
  the table's row is c = 0.03), so the ratio, not a single ‖A‖, is the yardstick. The late fade (7d: 15 % of the causal effect left at 143000) is
  ~25 % here (2.7e-3 against the 3.6e-3 peak at 4000–16000), by this measure.
- **The response is the OV-mapped nudge.** cos ≥ 0.997 from 2000 on. At init it is 0.98 too,
  since any head's response at k = 1 is its OV map of the nudge times its attention weight; the
  cosine says the readout is the head's value path, not that the head attends back.
- **`L5H9`** (the second previous-token head, the reason for "14 of 15") reaches s 0.9–2.2 from
  1000 on, against `L5H2`'s 7.3–9.7; `L5H2` beats all 15 at every prompt from 1000.
- **At step 0–256, 1–3 prompts of 7 read on by chance**: `L5H2` is one of 16 heads with s near 0,
  so it tops ≥ 14 of 15 in about 2 of 16 draws.

**The statistic's size from 1000 is a float32 detection limit.** `L5H2` is so sharp that its
attention weight on token j from 3–5 tokens later is ~1e-7–1e-10 (step 1000, `hdbscan_code`:
medians 3.6e-7, 4.8e-10, 3.5e-9; none exactly 0), so the ± change there is under float32
resolution of the head's output and A cancels to exactly 0. 47–73 % of its σ₂…σ₅ are under the rounding floor from 1000 (2 % at 512, 0 before),
and in s they are held at the floor. So s ≈ 9 says "under rounding", not e⁹. Without the
substitution the heads beaten are the same at every step and prompt (checked), so the verdict
does not rest on it. *The substitution was set after the first run (step 1000) refused on an
exact zero, before any verdict was read; `b1_prev_token.py`'s docstring says so.*

**Not established:** what B1 does not cover: it ranks one head's output against its layer's
other heads, from block 5 alone, so the multi-layer path, the ranking against random directions
and the residual and loss readouts B2–B4 read rest on B0's gate only (G2, G3, G4); the
false-on rate of the rule (≈ 2/16 per prompt at chance, read off steps
0–256, not derived); whether s's size tracks `L5H2`'s strength once rounding is not the floor
(a float64 block pass would tell); any head other than `L5H2` as a target.

**Re-run:** `python trackB_circuit/b1_prev_token.py run --out <dir> --gate
/run/media/system/HDD_1TB/mets_data/trackB_b0/2026-10-14c && python
trackB_circuit/b1_prev_token.py report --out <dir> --gate <same>`.

## B0 (2026-10-14): the gate at its first step

**Input:** `pythia-410m` step 143000; the 7 scored v1 prompts (`b0_gate.V1_SCORED`, prompts
hash `976b62d677bc`), ≤ 512 tokens; `resid_pre` at layer 5; B1's 32 positions per prompt
(`core.nudge.draw_positions`); K = 32 directions seeded per (prompt, `resid_pre`, 5). GPU
float32 on the RTX 3080, deterministic, TF32 off (torch 2.10.0+cu130, transformers 4.44.2); G2
on CPU float64. Code `329eba22f8c6`. Records: `/run/media/system/HDD_1TB/mets_data/trackB_b0/
2026-10-14b/{gpu,cpu}_step143000.json`. GPU 660 s for the step, G2 27 s.

**Re-run after `/challenge-pr` on #198** (finding 1: `Stack.block` crashed on transformers 4.57,
and 4.57's eager attention masks only by what it is passed; the block now always passes an
explicit causal mask and 4.57's rotary embeddings; smoke tests pass on 4.44.2 and 4.57.6): the
first record (code `db4804772c65`, `2026-10-14/`) and the re-run agree in all 5,870 GPU and 388
CPU fields but timings. `report` now refuses a record from other code.

| arm | rule (`b0_gate.py` docstring, fixed before output) | at 143000 |
|---|---|---|
| G0 | loop vs HF logits ≤ 1e-5 | **pass**: 0.0 exactly, all 7 prompts |
| G1 | the add, to 8 ulp; 0 off j | **pass**: worst 0.058 of tolerance; off j exactly 0 |
| G2 | A/ε vs JVP, ≤ 5 % at ≥ 90 % of tokens, c = 0.01 | **pass**: 10424/10424; median relative error 1.1e-4, 1.0e-3, 1.1e-2, 9.9e-2 at c = 0.01, 0.03, 0.1, 0.3 (ε² growth; at c = 0.3 only 9.8 % of tokens are within 5 %: the plateau's edge) |
| G3 | planted ∇L: within 10 % and rank 1, ≥ 6/7 | **pass**: 7/7; A/ε over ‖∇L‖ 0.998–1.020 |
| G4a | causality, exact 0 before j | **pass**: 0 exactly at every later layer and the loss |
| G4b | another prompt's ∇L: rank-1 rate ≤ 0.053 (N = 224) | **fail**: 17, 17, 17, 21 of 224 (0.076, 0.076, 0.076, 0.094) at c = 0.01 … 0.3 |
| floor | four same-shape base rows | 0 exactly everywhere: no readout below floor |

**Why G4b fails** (`trackB_circuit/b0_g4b_geometry.py 143000 <out>`, no nudge): ranking
\|∇L_j · g′\| among \|∇L_j · R_k\| from the gradients alone gives **the same 17 rank-1 hits**
(the rank-1 indicator agrees with the nudged c = 0.01 at 224 of 224; the exact rank at 189,
within 1 at 217, at most 3 apart). Loss gradients at layer 5 are aligned
across prompts: mean \|cos\| over cross-prompt pairs 0.031, against 0.025 for random directions.
So the harness reads A = ε∇L·d as G3 says it does, and the excess is the model's gradient
geometry. The arm assumed an on-distribution direction from another prompt ranks like a random
one; at 143000 it does not. The frozen rule still reads **fail**: Blocked 43.

**Not established:** any step but 143000; whether G4b's excess holds at other steps; G2 at
any other step.

**Re-run:** `python trackB_circuit/b0_gate.py run --out <dir> --steps 143000`, then `g2`,
then `report` (it needs all 18 steps for a verdict).

## B0 over 18 steps (2026-10-14, Blocked 43 (a))

**Input:** as "B0 (2026-10-14)" above, all 18 steps; G4b's rule as amended by Blocked 43 (a)
(the user's), committed before the run (`8330f7b`). Code `1e78e2be9d94` in all 36 records.
Records: `/run/media/system/HDD_1TB/mets_data/trackB_b0/2026-10-14c/{gpu,cpu}_step*.json`,
`gate.json`. 207 min (GPU ~11 min per step; G2 on CPU first).

**Verdict by the rule: FAIL**, at one arm. G0, G1, G2 (10424/10424 tokens within 5 %), G3
(7/7) and G4a pass at all 18 steps; G4b passes at 17 of 18.

| step | 0 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rank-1, nudged / linear (of 224) | 32/32 | 32/32 | 32/32 | 36/36 | 34/34 | 49/49 | 86/86 | **114/115** | 119/119 | 82/82 | 36/36 | 20/20 | 15/15 | 7/7 | 14/14 | 9/9 | 12/12 | 17/17 |
| exact rank equal | 223 | 224 | 223 | 223 | 224 | 224 | 224 | 221 | 224 | 224 | 223 | 223 | 222 | 223 | 223 | 220 | 223 | 189 |
| rank-1 rate at c = 0.01 (old bound 0.053, not judged) | 0.143 | 0.143 | 0.143 | 0.161 | 0.152 | 0.219 | 0.384 | 0.509 | 0.531 | 0.366 | 0.161 | 0.089 | 0.067 | 0.031 | 0.062 | 0.040 | 0.054 | 0.076 |

(Every step: within 1 rank at 224 of 224, except 128 (223) and 143000 (217). Rates at the other
c: `gate.json`.)

**The one mismatch is a near-tie.** Step 128, `camus_letranger` j = 82, d from
`hdbscan_code`: linear |∇L·d| = 0.0725662 against the top random direction's 0.0725586 (k = 10),
a relative margin of 1.05e-4; the nudged |A|/ε at c = 0.01 is 0.072468, 9.8e-5 below linear
(1.4e-3 of this projection, 13× the margin), so it ranks 2. In G3's unit (over ‖∇L‖ = 1.165)
the error is 8.4e-5, under the largest G3 saw at step 128 (1.7e-4) and far inside its 10 %. This is the failure the amended
rule's text named (a near-tie inside the linear error), at 1 arm of 4,032. Re-derived by
`python trackB_circuit/b0_g4b_tie.py` (the arm alone on the GPU; ranks as in the record).

**What it shows about the model, beside the gate:** another prompt's loss gradient outranks all
32 random directions in 14 % of arms at init, rising to 53 % at step 256 and falling to
3–8 % from 8000. Loss gradients at layer 5 are aligned across prompts at every step, most
strongly at 64–512. This sharpens the B4 fence below.

**Not established:** the false-fail rate of G4b's amended rule (one near-tie in 4,032 arms is
its only reading); G2 at steps other than J[0:4] and R[0].

**Re-run:** `D=<dir>; python trackB_circuit/b0_gate.py g2 --out $D && python
trackB_circuit/b0_gate.py run --out $D && python trackB_circuit/b0_gate.py report --out $D`.

## Fences (from B0's gate)

- **B4 brings an on-distribution null.** Loss gradients at layer 5 share a direction across
  prompts: mean \|cos\| over cross-prompt pairs 0.031 against 0.025 for random directions; the
  median \|projection\| of a unit gradient on the mean gradient direction is 0.068 (a random unit
  vector: ~0.025). So "a structured direction ranked among random ones" is biased upward by the
  model itself, and B4's eigen-directions against random ones would read that bias as a finding.
  (`/challenge-pr` on #198, finding 3; carried from Parked.)
  Over the 18 steps the rank-1 rate of another prompt's gradient is 3–53 % against 3 % for a
  random direction ("B0 over 18 steps"), so the bias is largest at 64–512.
- **B1's floor.** *(Resolved by B1: the floor is the base row in another batch shape, at
  least one ulp of the readout; `design-B.md` "B1 as built". It does bind from 1000:
  "B1 (2026-10-14)".)* The same-shape base floor is exactly 0 under deterministic algorithms, so
  every later unit fixes a rounding floor of this kind before its run. (`/challenge-pr` on #198, risk 1.)

## Parked

- B2a, a token-matched null (random subspaces drawn to match the real tokens' occupancy): why, V's subspaces share LN1's gain pattern with the tokens and may beat Haar nulls by overlap alone (`/challenge-pr` on #208, finding 2); cost, one more null family per rung; could change a unit 2 D\* from "below the nulls" to "no better than where the tokens are". Unit 2 records each null's occupancy beside its TV first. *(Unit 2, 2026-10-16: no V rung beat its nulls, so nothing for this null to demote at 143000; kept for unit 3.)* **Dropped (unit 3, 2026-10-16):** no V rung beat its nulls at any of the 7 steps read either, so it can change no B2a verdict.
- A1a's per-head `fe_full` β against B2a's M1 β, opposite signs in 6 of 14 heads at 143000, all M1 −, A1a +. **Carried to A2 as its first item** (Blocked 54 (a), the user's, 2026-10-16; `trackA_force/status-A.md` Parked).

## Corrections received

(none)
