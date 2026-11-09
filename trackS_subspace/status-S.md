<!-- trackS_subspace/status-S.md -->
# Track S — STATUS

<!-- phase-card -->
## Card

- **Question:** Do the 384 heads of `pythia-410m` share directions of the residual stream, per subspace family (OV write / read, QK query / key, OV's negative / positive / complex invariant subspaces), beyond a random arrangement of the same subspaces, how does that change over training, and what do the attention layers do inside the shared space (push, rotate, collapse)?
- **Inputs:** `pythia-410m` weights at steps 0, 128, 256, 512, 1000, 2000, 4000, 143000 (and 8000 off the grid for the preflight and the gate); tokens only for unit 3's residual check (v1's 7 scored prompts, `trackB_circuit/b0_gate.py` `V1_SCORED`); code `trackS_subspace/s1.py`; records `data/trackS/s1/` (main tree) — `trackS_subspace/design-S1.md` "Objects, fixed here". S2: the same weights at 0, 256, 512, 1000, 2000, 4000, 143000 (8000 for the preflight) under a per-head OV hook, v1's 7 prompts for V and 7d's 16 repeated-random-token sequences for G; code `trackS_subspace/s2.py`; records `data/trackS/s2/` (main tree) — `trackS_subspace/design-S2.md` "Steps and readouts"
- **Results:**
  - S4 closed 2026-11-09 before its exit, by the frozen rule of Blocked 150 (c), on Blocked 151 (a), the user's (3 units + close-out): on HAPAX's 1B pair the energy counts rise alike with and without the paper's induction transition (one seed a side, description, no noise floor); no released toy carries a gated date; on the one healthy toy (R2) the uncentred count cannot be gated at V's plateau (layer 0 never turnable, layer 1 saturated), and V there is a count of layer 1's drops. The toys go to the ROADMAP's new S4′ row — `trackS_subspace/status-S.md` "S4 on one page (closed 2026-11-09)"
  - S4 unit 2 (Blocked 150 (a), the user's): the toy instrument's calibration fails its opening check, so S4 closes as Blocked 150 (c) under the frozen rule. On 64 library `val` items (116 units a step) plant B is seen by the uncentred count at R2's step 6876 at 0.03 at dose 1 (0.4 needed; 0.05 at dose 2), though at 29101 it passes at dose 1 (0.44) and the null holds. The limit is neither the unit count nor the dose ceiling: plant B at dose ≤ 1 never turns layer 0 (embedding → layer 0 rises steeply from init on), and layer 1 already drops on 56 of 58 units at V's plateau, so the check could not pass at any plateau step (corrected after `/challenge-pr` on #334). Whether S4 closes or is extended with a per-transition count is Blocked 151 — `trackS_subspace/status-S.md` "S4 unit 2, the toy instrument's calibration: the opening check fails (2026-11-09)"
  - S4 unit 1, Blocked 149 applied ((i) (a), (ii) (a), the user's): no released toy carries a gated date. On the late window (18540, 38140, 75000) R2's induction dates 14146, and its gate refuses at 6876 and 29101: plant B is seen by the uncentred count at 0.14–0.17 at dose 0.5 (0.4 needed), as at steps 1–2; doses up to 4 do not pass it either (7–12 units a step). R2's V (description, read after the refusal) rises at ~900 and is gone by 16942, so the late plateau is 0 (946 on U1's window). R3–R6 are description only. What next is Blocked 150 — `trackS_subspace/status-S.md` "S4 unit 1, Blocked 149 applied: R2's gate refuses (2026-11-08)"
  - S4 unit 1 (Blocked 148: D1 (a′), D2–D5 (a), the user's): on HAPAX's pair, removing induction's training pressure removes the paper's induction transition (vanilla's at about 800; Hapax's mean ramps), while both energy counts rise alike: within 5 counts of each other through step 800, half their plateaus at 600–800 on both sides. One seed a side, description; the dated one-step differences are single counts at the bar and are not read. The released toys answer nothing yet: the depth-3 to depth-6 runs get worse on held-out text from about step 2100 (unstable or overfit, not told apart), R1's recipe is unreadable, and on R2 and R3 induction forms at about 9,000–22,000, after the frozen window. No toy's V has been read (Blocked 149) — `trackS_subspace/status-S.md` "S4 unit 1, the released arms (2026-11-08)"
  - S4 opened 2026-11-08 (Blocked 147 (b), the user's): a scan and a proposed design (Blocked 148, D1–D5). The untied toy arm is already released at depths 2–6 (`timaeus/L2`–`L6`: constant LR, no warmup, 451 checkpoints each). So is a Pythia-1b-shaped pair trained with and without the loss on induction-predictable tokens (HAPAX, 2511.05743). U1 reads those before any training — `trackS_subspace/status-S.md` "S4 unit 0, the scan and the design (2026-11-08)"
  - Opened 2026-10-20 with a scan, a `sympy` check and a design frozen as written (3 units + close-out); the rotation null, the norm-matched random draw and step 0 are one law for the mean projector, and the preflight refused per-unit verdicts against it, so S1 was designed to read counts — `trackS_subspace/design-S1.md` "Preflight (`tools/design_preflight.py`, step 8000, off the grid)"
  - Unit 1's gate: the instrument and its null pass (G0, G1, G1iii, G2: the three nulls are one law, step 0 reads no shared direction, the rotation null reads none at any step); the count fails its planted effect on the real pool in all 7 families, because at a trained step the mean projector's spectrum is dense at the null's edge, where each planted direction raises the count only about half the time; N refused by the frozen fallback (Blocked 82); the heads use nearly every dimension but unevenly (participation ratio 344–729 against 880), which alone explains a count near the edge, not the top eigenvalues — `trackS_subspace/status-S.md` "S1 unit 1, the gate (2026-10-20)"
  - Unit 2 (Blocked 82 (b), the user's): R1′ = the participation ratio against its exact Haar value; the null side passes (G0′–G2′); the additive plant passes in the 5 OV families and fails in QK, both by the plant's added dimensions diluting the pool's own concentration (math check fact 6), and with that removed it cannot see the pool, so the effect side is uncalibrated and QK's ρ is Blocked 83. Over training the OV families concentrate from 512 (`ov_write` and `ov_cplx` already from 256: corrected by `/challenge-pr` on #245), most at 2000 (ρ 0.59 `ov_write`), partly relax at 4000 (just after the LR warmup ends), and end at ρ 0.65–0.83; LN1's gain changes nothing; the spectrum is uneven from 512 (the top's null cannot tell a few directions from a graded spectrum) — `trackS_subspace/status-S.md` "S1 unit 2, R1′ and its re-gate (2026-10-21)"
  - Unit 3 (Blocked 83–85 (a), the user's): QK's heads leave the null at 256 with `ov_write` and `ov_cplx`, keep concentrating through 4000, and are the most concentrated of the 7 from 2000 on (`ov_write` is at 512–1000); inside each family's 10 most shared directions the radial sign reads S at no step from 512 (inside random subspaces' range at 512, and at 1000–4000 random subspaces hold no attractive part in 18–20 of 24 layers, so only "less repulsive" can show); the OV families' S is drawn from the heads it compresses, a bias of unknown size, so QK's readings are the clean ones; collapsible negative pairs are rarer than random in QK's S at 1000–4000; no shared space is the residual stream's top variance, and at 143000 the write, negative and complex ones are partly the massive-activation coordinates — `trackS_subspace/status-S.md` "S1 unit 3, QK's ρ, R3 and the residual check (2026-10-21)"
  - S1 closed 2026-10-21 at its revised exit (3 units + close-out): the shared spaces are described against an exact null, never counted or calibrated on the effect side — `trackS_subspace/status-S.md` "S1 on one page (closed 2026-10-21)"
  - S2 closed 2026-10-22 at its exit as extended (4 units + close-out): neither symmetrising every head's OV nor removing its rotation undoes the energy break (takes the violations back toward none), and the one OV-only edit with a known destination does not either, so "not shown" cannot be told from out of reach (that edit, every head's step-128 OV, lowers the violations at 512 and 1000 and raises them at 2000); copying survives the rotation's removal, which is about a tenth of each head's map; S1's shared rotation carries nothing beyond random pairs at a matched selection; structured edits move the stream further than random ones of the same weight norm — `trackS_subspace/status-S.md` "S2 on one page (closed 2026-10-22)"
  - S2 unit 4 (Blocked 90 (b), the user's: one more unit, as description): with each head's 2 pairs most in the shared complex space against 2 of its others removing as much rotation, neither the energy break nor copying moves beyond the random pairs at any step (as its controls in all 9 cells). About half the selected pairs lean into the shared space beyond chance; in 102–163 heads none does, and there the null holds by construction. The selected pairs move the stream further than the random ones at 1000, 4000 and 143000 (about 1.5× at 143000), so the null holds against the larger edit in effect. Against a random plane in each head's own write space, 3.4–5.0 % of pairs lie in the shared space beyond its 99th percentile, where 1 % would by chance — `trackS_subspace/status-S.md` "S2 unit 4"
  - S2 unit 3 (Blocked 89 (a), the user's): removing rotation only in the pairs lying in S1's shared complex space (left out by layer) was meant to be read against removing it in as many random other pairs, but the frozen rule counted 37–69 % of all pairs as "in S" (a random plane in a head's own write space sits as high), so the random side could not match the count and R3 is refused as frozen (unmatched); as description, neither the energy break nor copying moves beyond the fewer random pairs, and with every pair's rotation removed already inert (unit 2) there was little to carry. S1's readings inside the OV families' shared spaces hold on a leave-the-layer-out S from 512 (its early `ov_neg` / `ov_pos` readings were the selection bias) — `trackS_subspace/status-S.md` "S2 unit 3"
  - S2 unit 2 (Blocked 87 (a) and 88 (a), the user's): neither symmetrising every head's OV nor removing its rotation in the Schur planes takes the energy break back toward monotone at any scored step (R1 "not shown" under both; the symmetric part raises the violations beyond random edits of its size at 512 and 143000), and the one OV-only edit tried with a known destination does not either (every head given its step-128 map halves the violations at 512 and 1000 but leaves 29–61); in-context copying survives the rotation's removal at all four steps; the displacement guard was made description for R1 and R2 after it refused both planted effects (R3's guard is Blocked 89) — `trackS_subspace/status-S.md` "S2 unit 2"
  - S2 unit 1 (Blocked 86 (a) and (i), the user's: arm b in Schur planes, 3 units): the hook is the model at its base core and every arm's read-back is exact; at 8000, off the grid, both arms move the residual stream further than all 40 of their size-matched random edits, so the frozen displacement guard would leave their cells unread; arm a's random edits alone break most in-context copying; removing rotation in the Schur planes changes about a tenth of each head's map; the shared write space meets the shared read and key spaces from 512; the freeze is Blocked 87 — `trackS_subspace/status-S.md` "S2 unit 1"
  - S2 opened 2026-10-21 (Blocked 86 since decided (a) and (i)): every head's OV edited at once, (a) to its symmetric part and (b) with its rotation removed (recommended: in an orthonormal basis of each invariant plane, since λ → Re λ in the eigenbasis edits a head by 1.6× its size at 512), each against 40 random edits matched in size and in the edited head's norm, and read only beyond the gain change alone; reading the energy break, in-context copying and S1's shared rotation; no theorem predicts "monotone" for (a) — `trackS_subspace/design-S2.md`
- **Superseded / wrong:** none
- **Registry:** none, because Track S is tier 1 on an exploration rung (`trackS_subspace/design-S1.md` "Fences")
- **Depends on:** none
- **Feeds:** C
- **Open threads:**
  - The toys (a per-transition count gated at its own unsaturated steps; Tu, Tt and Q5; deeper healthy toys), carried to the ROADMAP's S4′ row; H's description carried to C2's rewrite — `trackS_subspace/status-S.md` "S4 on one page (closed 2026-11-09)"
  - Arm a's displacement at 143000 (11.9 against 1.0–1.1) and the selected pairs' at 143000 (about 1.5×), both carried to S3 — `trackS_subspace/status-S.md` "S2 on one page (closed 2026-10-22)"
  - Whether any OV-only edit undoes the break, and the gradient-flow statement's exact hypotheses (`lit-S2.md` Parked 1), both carried to C2's draft — `trackS_subspace/status-S.md` "S2 on one page (closed 2026-10-22)"
  - The massive coordinates in the end-state shared spaces (carried to S3), 2603.13314's full text (carried to C2's draft) — `trackS_subspace/status-S.md` "S1 on one page (closed 2026-10-21)"
- **After Phase 10:**
  - The weighted projectors (singular values as weights), where the rotation null keeps each head's spectrum and differs from Haar *(free, weights only)*
- **Reviewed:** 2026-11-09 · body `331fc20cc5`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| 0. scan, design, `sympy` check, preflight | **done 2026-10-20**; frozen as written (Blocked 81 (a), the user's) | `lit-S1.md`, `design-S1.md`, `tools/math_checks/s1_projector_null.py`, `data/trackS/s1/2026-10-20-preflight/` |
| 1. the instrument and its gate | **done 2026-10-20: G0–G2 pass, G3 fails in all 7 families; N refused** (Blocked 82) | "S1 unit 1" below; `s1.py gate` |
| 2. re-gate + R1′ over the 8 steps (Blocked 82 (b)) | **done 2026-10-21: G0′, G1iii′, G2′ pass; G4 passes in the 5 OV families, fails in QK, and cannot see the pool (`/challenge-pr` on #244); R1′ read as description; QK Blocked 83** | "S1 unit 2" below; `s1.py regate`, `s1.py r1` |
| 3. QK's ρ, R3 on S, the residual check (Blocked 83–85 (a)); R1b, R2 carried to S2 | **done 2026-10-21: QK concentrates most of the 7 (ρ 0.51–0.53 at the end); R3's radial sign at 512–2000 reads the layer, not S; no S is the residual stream's top variance; at 143000 the write, negative and complex S are partly the massive-activation coordinates** | "S1 unit 3" below; `s1.py r1qk`, `s1.py r3`, `s1.py massive` |
| close-out | **done 2026-10-21** (writing only; 85 (a)) | "S1 on one page" below |
| S2 0. scan, design, `sympy` check | **done 2026-10-21; Blocked 86 decided (a) and (i), the user's** | `lit-S2.md`, `design-S2.md`, `tools/math_checks/s2_ov_arms.py` (24 of 24 after `/challenge-pr` on #246) |
| S2 1. code, G0, the preflight at 8000; R1b, R2 beside; then the freeze | **done 2026-10-21: G0 passes (its (b) read-back revised); the preflight's refusals argued; both arms move the stream further than every control at 8000; the freeze is Blocked 87** | "S2 unit 1" below; `s2.py g0`, `preflight`, `beside` |
| S2 2. G1–G3 at the design points; R1, R2 | **done 2026-10-21: Blocked 87 (a) and 88 (a), the user's; the gates pass with the guard as description; R1 "not shown" under both arms; R2 under b "survives" at 4 of 4** | "S2 unit 2" below; `s2.py gates`, `gate-read`, `arms`, `read` |
| S2 3. R3 on S₋ₗ; S1's R3 on S₋ₗ beside | **done 2026-10-22: Blocked 89 (a), the user's; R3 refused as frozen, unmatched (the frozen selection took 37–69 % of pairs; read as description, nothing lower in V); S1's R3 on S₋ₗ as S1's from 512; a matched R3 is Blocked 90** | "S2 unit 3" below; `s2.py r3`, `r3-read`, `s1-r3-loo` |
| S2 4. the matched R3, description (Blocked 90 (b), the user's) | **done 2026-10-22: as its controls in all 9 cells (V 0 of 5, G 0 of 4 read *lower*); R3's G1 0 of 9; 3.4–5.0 % of pairs above their head's write-space q99** | "S2 unit 4" below; `s2.py r3m-select`, `r3m`, `r3m-read` |
| S2 close-out | **done 2026-10-22** (writing only; 90 (b)) | "S2 on one page" below |
| S4 0. scan, design (Blocked 147 (b), the user's) | **done 2026-11-08: frozen on Blocked 148 (D1 (a′), D2–D5 (a), the user's)** | `lit-S4.md`, `design-S4.md`; "S4 unit 0" below |
| S4 1. the released arms | **H read: both energy counts rise alike (within 5 counts through 800) with and without the paper's induction transition (description). Toys: R1 refused (recipe unreadable); R3–R6's held-out loss rises from ~2100; induction forms after the frozen window on every toy; no toy's V read; Blocked 149** | "S4 unit 1" below; `s4.py`, `s4_hapax_fig.py` |
| S4 1b. Blocked 149 applied | **R2's gate refuses at its late steps (plant B unseen, as at steps 1–2); R3–R6 description; no toy dated; Blocked 150** | "S4 unit 1, Blocked 149 applied" below |
| S4 2. the toy instrument's calibration (Blocked 150 (a)) | **opening check fails: plant B unseen at R2's 6876 at doses ≤ 1 on 116 units (0.03); S4 closes as 150 (c); close-out next, U3 not run** | "S4 unit 2" below; `s4.py calib` |
| S4 close-out (Blocked 151 (a), the user's) | **done 2026-11-09** (writing only; closed before its exit as 150 (c)) | "S4 on one page" below |

## Corrections received

- 2026-10-22, C2 unit 1: `lit-S2.md`'s gradient-flow row said the energy "decreases" along the
  flow; the texts say non-decreasing (2312.10794 §3.2, 2605.04279 Theorem 11). S2 counted a drop
  as a violation, the non-decreasing convention, so no S2 reading moves. Row corrected in place;
  `trackC_writeup/lit-C2-unit1.md` §1.
- 2026-10-22, S2 unit 3: S1's "the OV families' R3 readings carry a bias of unknown size" is now
  measured: on a leave-the-layer-out S they read as before from 512, so the bias moves none of
  S1's readings there; before 512 `ov_neg`'s and `ov_pos`'s outside-Haar `share` layers (up to 11)
  were the bias (0–2 on S₋ₗ). "S2 unit 3" below.

## S4 on one page (closed 2026-11-09)

**Asked** (`design-S4.md` "What S4 answers"): on a toy where the induction circuit is known, is
there an energy break to move (Q1); does it move where induction cannot form or is suppressed
(Q2); does it fall at a step without a warmup (Q3); S1's atlas with the circuit named (Q4); the
tied arm (Q5). Arms: the released toys R1–R6, HAPAX's 1B pair H (D1 (a′), energy only), our
trained Tu / Tt. **Closed 2026-11-09 before its exit**, by the frozen rule of Blocked 150 (c)
(U2's opening check failed), on Blocked 151 (a), the user's: 3 units (U0, U1 with U1b, U2) and
this close-out, inside the 4-unit budget (D5 (a)); U3 not run. Writing only here.

**Established (tier 1, description; one seed an arm):**

1. **On HAPAX's pair the energy counts rise alike with and without the paper's induction
   transition** (U1): within 5 counts of each other through step 800, half their plateaus at
   600–800 on both sides; vanilla's induction transition at about 800, Hapax's mean ramps. Q2's
   one reading, and it is H's (1B, Pythia's recipe, Hapax weakens induction rather than removing
   it), with no noise floor, so no label and no gap read.
2. **No released toy carries a gated date** (U1, U1b). R1 refused (recipe unreadable); R3–R6
   description only (held-out loss rising from ~2100; on the late window induction dates R3 at
   16942 and never forms on R4–R6); R2, the one healthy toy, refused by its gate at 6876 and 29101.
3. **The uncentred count does not fit a 2-layer toy at its plateau** (U2, 64 library `val`
   items, 116 units a step). Layer 0 raises E₁ steeply in every population, fresh inits included,
   so plant B at dose ≤ 1 turns 0 of 174 layer-0 units at init and 0 of 58 at 6876; layer 1
   already drops on 56 of 58 units at 6876. The positive control fails at V's plateau by
   construction, whatever the dose ceiling or the unit count; the null holds (re-inits 0).
4. **On R2, V is a count of layer 1's drops**, so R2's transient (V up at ~900 under a constant LR
   with no warmup, gone by 16942 as induction forms, dated 14146) is layer 1's drop rate rising
   and falling. A one-seed lean its own gate cannot test.

**Not established:** a break on any toy (Q1 gated); Q2's toy answer (R1 refused, no suppressed
toy trained); Q3 (R2's rise at ~900 without a warmup is ungated and is layer 1's count, not
Pythia's V); Q4 and Q5 (not run: Tu and Tt untrained, the planted negative sector, B7's read);
that H's 5-count gaps are a difference (no noise floor); when Hapax's 1–2 induction heads form;
anything on 410m or between released checkpoints.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. scan, design | #331 | frozen on Blocked 148 (D1 (a′), D2–D5 (a)) |
| 1. the released arms | #332 | H read (description); no toy dated; Blocked 149 |
| 1b. Blocked 149 applied | #333 | R2's gate refuses at its late steps; R3–R6 description; Blocked 150 |
| 2. the toy instrument's calibration (Blocked 150 (a)) | #334 | opening check fails; cause split by transition; Blocked 151 |
| close-out (Blocked 151 (a)) | this | writing only |

**Parked and open at close:**

| item | fate |
|---|---|
| U0 1. 2506.13688's Pythia result | carried to C2's rewrite (already there: `trackC_writeup/status-C.md` "For the user's rewrite") |
| U0 2. HAPAX's pair as a C2 arm | done: read in U1 under D1 (a′); H's description carried to C2's rewrite as cited context beside 2506.13688 (status-C, same section), with its fences |
| U1 1. H's noise floor (`pythia-1b`'s checkpoints, reserved by `design-8.md`) | stays yours with the reserve (Blocked 7); named in that cited-context line, since without it H carries no label |
| U1 2. HAPAX's two unreleased seeds | carried to ROADMAP §7a S4′ as an input if released (a second seed for H) |
| U1 3. whether 2507.21449's `L4` is these weights | carried to S4′ (it could say whether R3–R6's degeneration is known, which S4′'s deeper-toy option needs) |
| U1b 4. R2's transient | done by U2 (layer 1's drop rate); its dating carried to S4′ with item 6 |
| U1b 5. the digitised Hapax panel's heads | dropped: H carries no label and no gap is read without a noise floor, so a date for Hapax's heads changes no reading; S4′ can take it up if U1 1 opens |
| U2 6. a per-transition instrument | carried to S4′ (its design starts from U2's per-transition table, below) |
| Q4 (the atlas), B7's partner persistence | already carried to ROADMAP §7a (U2's design) |
| Q5 (Tt, the planted negative sector), Tu, Q2–Q3 on trained toys | carried to S4′ (U3 not run) |
| the MLP arm (D4 (a)) | stays in ROADMAP §7a, unscheduled (as at U2) |

## S4 unit 2, the toy instrument's calibration: the opening check fails (2026-11-09)

Blocked 150 decided (a), the user's. U2's design (`design-S4.md` "U2 after Blocked 150 (a)") was
frozen and pushed in `d53bfda` before any U2 output. Code `16bd243e0669` (`s4.py calib`); record
`HDD_1TB/mets_data/trackS/s4/2026-11-09-u2/R2/calib.json` (sha256 `506b507cf4c7`), per-unit drops
in `calib_units.json`, log `calib_R2.log`. Input: the library's `val`, k = 64 (`library_hash`
`2e76b7c7bde5`), retokenized by R2's tokenizer, and the 7 v1 prompts beside. Float32, GPU, 66 s.
**Placed before output** (the first run stopped on it, nothing read): an item longer than the toys'
1024-position context is set beside, like the ¼ bar, and not cut. 6 of the 64 items are beside
(5 over the share bar, 4 over the length, 3 over both), so 58 items give 116 units a population.

**The check fails, so S4 closes as Blocked 150 (c)** (the frozen rule; the close-out is the next
unit, and U3 does not run). Plant B's uncentred share at 1e-3 (the units not already violating that
the plant turns), 58 items:

| population | base rate | 0.25 | 0.5 | **1** | 2 | 4 | ceiling | layer move | rule |
|---|---|---|---|---|---|---|---|---|---|
| re-inits (3, each) | 0 | 0 | 0 | 0.47–0.49 | 0.50 | 0.78–0.83 | 1.0 | 0.94 | null rate 0 ≤ 5 % |
| R2 at 6876 | 0.48 | 0.02 | 0.03 | **0.03** | 0.05 | 0.73 | 1.0 | 0.94 | **fails** |
| R2 at 29101 | 0.05 | 0.03 | 0.17 | **0.44** | 0.71 | 1.0 | 1.0 | 0.83 | passes at 1 |

The 7 v1 prompts (12 units) read alike: 0.14 at 6876 and 0.33 at 29101 at dose 1. U1's 0.14 at 6876
was one unit of 7. More units did not raise it; the share at 6876 is lower on 116 units.

**Why (description, per transition, from `calib_units.json`; corrected after `/challenge-pr` on
#334, finding 1).** The share mixes two transitions that behave differently, and neither can be
turned at V's plateau:

| population | layer 0: median rise, open, turned at dose 1 | layer 1: median rise, open, turned at dose 1 |
|---|---|---|
| re-inits (3) | 0.59–0.61, 58, **0** | 0.41–0.44, 58, 54–57 |
| R2 at 6876 | 0.86, 58, **0** | 0.03, **2**, 2 |
| R2 at 29101 | 0.21, 58, 1 | 0.12, 52, 47 |

"Open" counts the units not already violating; the median rise is theirs. Layer 0 (embedding →
after layer 0) raises E₁ steeply in every population, fresh inits included. So this is the toy's
architecture, not training, and plant B at dose ≤ 1 almost never turns a layer-0 unit (0 of 174 at
init, 0 of 58 at 6876, 1 of 58 at 29101). It does turn the layer-1 units wherever any are open (90–100 %). At 6876, V's plateau,
layer 1 already drops on 56 of 58 units, so 58 of the share's 60 units are layer 0's and the share
is 2/60 = 0.03. **The check could not pass at any step where V is on its plateau, whatever the
dose ceiling or the unit count.** The ceiling (1.0) shows only that a pure-noise state would turn
layer 0 too. Each layer moves the tokens about as far as their own norm (0.83–0.94).

**What U2 established:** on R2 the uncentred count's positive control fails at V's plateau because
layer 0 is never turnable at dose ≤ 1 (from init on) and layer 1 is saturated there. V on this toy
is a count of layer 1's drops, so R2's transient (U1, Parked 4) is layer 1's drop rate going from 0
to ~1 and back. That transient is a one-seed lean this gate cannot test, since no planted
violation can show on a transition that already violates. Blocked 150's recommendation took the
unit count as the limit, and it was not. U1's "1 of 7, flat across doses" already showed the
pattern (`LESSONS.md` 6).

**Parked** (why / cost / which decision it could change):
6. A per-transition instrument: count layer 1's drops alone and gate it at its own unsaturated
   steps (fresh inits: 54–57 of 58 turned at dose 1; 29101: 47 of 52). It would date the rise and
   the fall of layer 1's drop rate, but at the plateau itself the gate has nothing to turn, by
   construction. Deeper healthy toys (R3–R6 lose held-out loss) might leave unsaturated middle
   transitions. Cost: a new design, post hoc on R2. It could change whether a toy instrument
   exists at all. Carried to the close-out (S4′ in ROADMAP), or to an extension: Blocked 151.

## S4 unit 1, Blocked 149 applied: R2's gate refuses (2026-11-08)

Blocked 149 decided (i) (a), (ii) (a), the user's; placed in `993588b` before any toy's V was read
(`design-S4.md` "Decided after U1's released arms"). Code `16463a9740d5`; records as unit 1 below
(`reading.json` sha256 `ff4997ed6c35`; U1's kept as `reading_u1window.json`, each toy's
`induction_reading_u1window.json`). H's entries are identical to U1's.

**R2's gate refuses, so no toy is dated.** On the window 18540, 38140, 75000, induction (random-token
prefix, max over heads) dates **14146**, so the gate's steps are 6876 and 29101. Plant B is not seen
by the uncentred count (`R2/gate_refused.json`, 12 units a step, 6 prompts × 2 transitions):

| dose | ≤ 0.25 | 0.5 | needed |
|---|---|---|---|
| B share, step 6876 (base rate 0.42) | 0 | 0.14 | 0.4 |
| B share, step 29101 (base rate 0) | 0 | 0.17 | 0.4 |

R2 and R3 refused the same way at steps 1–2 in U1, so the refusal does not move with the steps
(LESSONS 6). The positive control was sized on 410m and never on a 2-layer, d-256 model (the
design's "Worth challenging" named it). **Doses above 0.5** (`/challenge-pr` on #333, finding 1;
a diagnostic on a scratch copy, not a record): with 1, 2 and 4 added, the pooled share crosses 0.4
at dose 2. Per step it is 0.14 at 6876 (1 of the 7 units not already violating) and 0.67 at
29101, and the B-matched centred τ is not found in [1e-7, 1e-2]. So the dose alone does not pass,
and 7–12 units a step are too few.

**R2's V, description (gate refused; read after the refusal):** 0 through step 789; 4 of 12 units
from 864–1035; 4–5 through 9012; then 3, 2, 1 at 12926–15481; 0 from 16942 to 75000. So on the
late window V's plateau is 0, and D4 (a) would read "no change" even with a passing gate. On U1's
window (526, 1035, 2100, fixed before any output) V reads 0, 4, 5 and D4 (a) dates **946**
(`/challenge-pr` on #333, finding 2); ungated, so description only. V rises
early and dissolves while induction forms (9,012–18,540). This is a lean, and the instrument that
reads it failed its own positive control. The centred count is not read: it needs the gate's τ.

**R3–R6, description only** (`reading.json`, label "held-out loss rising"): V's maximum is 5–12 (R3
at 1485, R4 9863, R5 5247, R6 789); V on the window: R3 2, 0, 0; R4 1, 1, 0; R5 11, 0, 0; R6 7, 5, 10.
On the late window, induction dates R3 at 16942, and R4–R6 at step 1 (never forms; plateau ≤ 0.006).

**What U1 established:** H's energy counts rise alike with and without the paper's induction
transition (description, below). No released toy carries a gated date: R1 has no readable recipe,
R3–R6 lose held-out loss, and R2's positive control is not seen. What next is Blocked 150.

**Parked:** (4) R2's transient (a one-seed lean; V up at ~900, gone by ~17,000 as induction forms) is the shape
C2 would want, and a gated instrument on a toy could test it. Carried to Blocked 150 (it changes
whether U2's rule must date a rise and a fall, not a plateau). (5) The digitised Hapax panel's own
heads (the paper's second panel; by eye its top head rises from ~1000 to ~2500, later than
vanilla's 800). Cost: one colour-and-alpha pass on the figure. It could date Hapax's 1–2 induction
heads, which the mean cannot.

## S4 unit 1, the released arms (2026-11-08)

Design frozen on Blocked 148 (D1 (a′), D2–D5 (a), the user's) in `7324ac2`, before any arm was
read. Code `trackS_subspace/s4.py` (records made at the code of `meta.json`; reading at
`570bf145a5c0`), `s4_hapax_fig.py`; tests `tests/test_s4.py` (the date rule equals `a3.d4a` on 200
series on Pythia's grid). Records `HDD_1TB/mets_data/trackS/s4/2026-11-08-u1/` (per arm:
`records/`, `induction/`, `meta.json`, `gate.json`, `check.json`; `reading.json` sha256 `f9daea51d762`;
`heldout.json` from `s4.py heldout`; `tl_check.json` from `s4_tl_check.py`, run in a venv with
TransformerLens).
Checkpoints in `HDD_1TB/mets_data/hf` (toys ~31 GB, H ~83 GB). Input: C2's 7 v1 prompts as text
(`texts_hash` in each `meta.json`), retokenized per model; float32, GPU, no TF32.

**Placed in code before output** (no V read on any arm when placed):

| choice | what | why |
|---|---|---|
| toy forward pass | TransformerLens 2.16.1's attention-only Shortformer block, re-implemented (`toy_forward`); no BOS, as C2 | TL is not in the env. **Checked against TL itself** (`tl_check.json`): max \|Δlogit\| ≤ 2.2e-4 on R2–R6 at 75000 and R4 at 1035; each repo's final checkpoint equals its `model.safetensors` (Δ 0.0, R2–R6, `check.json`; `check` now refuses otherwise) |
| toy tokenizer | each repo's `tokenizer.json`, merges rewritten as strings for tokenizers 0.19 (refuses if a merge part holds a space or `ignore_merges` is set) | the env's tokenizers is older than the file's format |
| step check | a checkpoint's optimizer step must equal its file's step | how R1 was caught |
| the toys' window | the log-grid steps nearest 525, 1050, 2100: **526, 1035, 2100** (a tie goes to the later); grid 117 steps (1–26, every second log-grid step, the window, 75000) | design "Dates" |
| induction probe | 16 sequences of 128 tokens uniform on 0–4998 (4999 is `<\|endoftext\|>`), repeated; **beside, the 7 texts' first 128 tokens repeated** | added after R2's last step scored 0.08 on random tokens and 0.48 (L1H6) on repeated text, before any induction date was read |
| gate | plant-B dose and the null- and B-matched τ from the two steps pooled; each step gated on its own; the uncentred count gated on its own (re-init rate ≤ 5 %, plant B seen ≥ 0.4), the centred τ arms as C2 3b; gate steps: the last grid step ≤ ½ induction's date and the first ≥ 2× | the design names two steps but not how they combine |
| H's re-inits | `arch_null.reinit_model` (Pythia's init law, float16-valued) on H's config | H has Pythia's recipe |
| H's induction (D1 (a′)) | D4 (a) on HAPAX's published mean prefix-matching curve, digitised (`s4_hapax_fig.py`, figure sha256 `41afea2f9e95…`, ~64 steps a pixel): vanilla crosses in (721, 785] → **800**; hapax's mean is a ramp with no transition (0.052 at ~980, 0.133 at 2010, 0.197 at 4000), which crosses half its window's median in (1172, 1236], **not a formation date** (`/challenge-pr` on #332, finding 1). Before ~980 hapax's line is hidden under Thresholded-Hapax's, which reads 0.017–0.038 at steps 77–914, so the crossing is the first *observed* one, but no hidden earlier jump fits under that line (CodeRabbit on #332) | no induction score of ours on H. The paper's text gives no step for seed 1234; its Appendix G steps are other seeds'. The panel is a mean over 128 heads, not the toys' max, so it cannot date Hapax's 1–2 induction heads |

**(i) Populated.** Every arm's first records are finite and vary across states. **R1 refuses:** its
file `checkpoint_0049990.pth` stores optimizer step 24,990, so its file numbers are 2× its
optimizer steps (gradient accumulation of 2 would do it). The refusal rests on what is unknown:
its data, batch, tokenizer and position type cannot be read from its files (design "Fences"), so
R1 is not read.

**Discovery: the released depth-3 to depth-6 runs get worse on held-out text from about step
2100.** Mean loss on 16 Pile-CC validation documents (`data/trackA/a3/corpus/val.jsonl.zst`, first
1024 toy tokens; `heldout.json`):

| step | 526 | 2100 | 9012 | 26592 | 75000 |
|---|---|---|---|---|---|
| R2 | 4.64 | 4.25 | 3.87 | 3.55 | 3.56 |
| R3 | 4.69 | 4.72 | 5.01 | 4.88 | 5.21 |
| R4 | 4.85 | 4.82 | 5.94 | 6.69 | 7.00 |
| R5 | 4.87 | 5.52 | 6.95 | 8.88 | 8.97 |
| R6 | 4.72 | 4.74 | 6.13 | 6.33 | 6.67 |

Uniform over the vocabulary is ln 5000 = 8.52. The same holds on C2's 7 texts (R5 ends at 6.6–9.4 a
prompt) and under TransformerLens itself, so it is the released weights. The recipes are identical
(`training.yaml`: AdamW 1e-3 constant, no warmup, clip 1.0, seed 42). **Instability and overfitting
are not told apart** (`/challenge-pr` on #332, finding 3): the training corpus `pile13m` is private
and its size is not stated (`data.yaml` names only the task), 75,000 × 32 × 1024 tokens is 2.5B,
and overconfident memorisation also drives held-out loss past uniform. Either way R2 is the only
toy that keeps learning text it was not trained on. 2507.21449 is said to study `L4` (Track T,
leaning); whether its `L4` is these weights is not checked.

**Induction on the toys** (not V; the max over heads):

| | random-token prefix score rises | repeated-text prefix score rises | previous-token rises | at 75000 (random / text) |
|---|---|---|---|---|
| R2 | 9,012–18,540 | 9,012–18,540 | 1,625–3,054 | 0.08 / 0.46 (L1H6) |
| R3 | 12,926–22,204 | 10,794–18,540 | 6,284–18,540 | 0.21 / 0.36 |
| R4–R6 | never (≤ 0.03) | never (≤ 0.05) | from ~20,000 (to 0.15) | — |

So on every toy **the frozen window (526, 1035, 2100) lies before induction**, and the frozen rule
dates induction at step 1 (a flat window, "no change"). The gate's steps derived from that date (1,
2) test nothing designed. Those gate records are set aside (`gate_setaside_steps1-2.json`; R2 and R3
refused anyway, with plant B unseen by the uncentred count at steps 1–2). **No toy's V has been
read.** The window, and what to do with R3–R6, is Blocked 149.

**(ii)–(iii) H, read in full** (gate at steps 400 and 1600, from vanilla's published 800; 7
prompts × 16 transitions, 336 re-init units):

| | uncentred gate | centred arms passing | V date (plateau) | per-prompt V dates | centred date, gated arms | induction (published) |
|---|---|---|---|---|---|---|
| vanilla | pass (null 0; B 0.88, 0.75) | 3.11e-4, 3e-4, B-matched 3.43e-3 (1e-4 fails) | 600 (32) | 500–900 | 800, 800, 1500 | a transition at ~800 |
| hapax | pass (null 0; B 0.87, 0.69) | 3.11e-4, 3e-4, B-matched 3.69e-3 (1e-4 fails) | 700 (27) | 500–800 | 800, 800, no change | no transition (a ramp) |

The series behind those dates, steps 100–900 (V, then the centred count at 3.11e-4):

| step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 |
|---|---|---|---|---|---|---|---|---|---|
| V vanilla | 0 | 0 | 0 | 0 | 8 | 16 | 18 | 18 | 26 |
| V hapax | 0 | 0 | 2 | 3 | 6 | 11 | 19 | 21 | 21 |
| centred vanilla | 2 | 4 | 5 | 5 | 6 | 12 | 17 | 28 | 34 |
| centred hapax | 0 | 0 | 0 | 1 | 6 | 8 | 16 | 25 | 23 |

The dates turn on single counts at the bar (`/challenge-pr` on #332, finding 2): vanilla's V is
exactly 16 against its bar of 16 at 600 (one count less dates it 700, as hapax), and its centred
count is 17 against 17.5 at 700. The two agreeing centred τ (3.11e-4 and 3e-4) are 4 % apart, so
they are one reading, not two.

The B-matched arms carry almost nothing (plateau 2 on vanilla, whose 1500 rests on it; 0 on
hapax), so the two lower arms are taken as the centred reading. That choice was made after seeing
the dates, and the B-matched arm passed its gate, dating vanilla 1500 and hapax "no change"; so
no centred date (800 / 800 included) is robust across the passing arms (CodeRabbit on #332). V on both sides is not flat at the window's
end (about 50 at 4000, against 26–32 at 2000), and vanilla's dips at 1500–1700 (12–17).

**What H shows (description; one seed a side, no noise floor, no label):** removing induction's
training pressure removes the paper's induction transition (vanilla's mean prefix-matching jumps
at about 750–800; Hapax's ramps with none), while both energy counts rise alike: through step 800
the two sides' V and centred series are within 5 counts of each other (plateaus 27–43), and both
reach half their plateaus at 600–800. The dated differences (V 600 → 700) are one count at the bar
and are not read. So on this pair the energy's rise does not need induction's transition; it says
nothing about Hapax's 1–2 induction heads, which the mean cannot date. Fences: 1B, not 410m; Hapax weakens induction rather than removing it;
H's schedule and data order are unread (the paper says only "Pythia's configuration"); no step 0;
induction's dates come from a figure, at a mean over heads.

**Parked** (why / cost / which decision it could change):
1. H's own noise floor: `pythia-1b`'s checkpoints are the only one, and they stay closed while
   `design-8.md`'s reserve holds. Cost: about 15 GB of reads. It could change whether the 5-count gaps are
   a difference at all. Yours, with the reserve (Blocked 7).
2. HAPAX's two extra seeds (bf16, to 18,000; Appendix G): unreleased as far as the scan found.
   They would give H a second seed.
3. Whether 2507.21449's `L4` is these weights (Track T's lean). Cost: one read of its Appendix E.
   It could change whether R4's degeneration is known.

## S4 unit 0, the scan and the design (2026-11-08)

Writing only; no weights downloaded, no pickle opened. Blocked 147 (b), the user's, opened S4.

| | |
|---|---|
| scan | `trackS_subspace/lit-S4.md`: HF API configs and file lists (confirmed), 2511.05743 §3.1 (read), 2506.13688, 2511.16893, 2601.09588, 2604.11791 (summary) |
| what it changed | the untied arm is released (`timaeus/L2`–`L6`, 451 checkpoints each, constant LR 1e-3, no warmup; `L1` empty, `L1W256-dense` a 1-layer candidate with its recipe unknown); a Pythia-1b-shaped pair with and without the loss on induction-predictable tokens is released (`KeremSahin2210/vanilla_1b`, `hapax_1b`, every 100 steps from 100, no step 0); the L-series' data `timaeus/pile13m` is private |
| design | `trackS_subspace/design-S4.md`, proposed: Q1–Q5; arms R1–R6, H, Tu, Tt; C2's definitions (E₁, V, the centred count, A5's D4 (a) rewritten for any grid; the toys' window at steps 525, 1050, 2100; H's at 1000, 2000, 4000) on C2's prompt text retokenized, fresh inits for step 0, 3b's gate per arm; U1 the released arms (no training), U2 gated on U1, U3 the atlas, circuit, plant and B7; 4 units |
| decisions | Blocked 148: D1 HAPAX's pair in S4 or C5; D2 the "without repeats" arm; D3 the tied arm's depth; D4 the MLP arm; D5 budget. Recommended D1 (a′) (H read for energy only: its recipe is `pythia-1b`'s, reserved against induction measurement by `p8_scale_ladder/design-8.md`), (a) for D2–D5. Revised after `/challenge-pr` on #331 (accept with changes, 1–5 taken: `design-S4.md` "Revised after #331") |
| Parked | 1. 2506.13688's Pythia result (last-layer cosine near 1 early, falling around step 1000), not in `paper/`: **carried to C2's rewrite** (`trackC_writeup/status-C.md` "For the user's rewrite"). 2. HAPAX's pair as a C2 arm: in U1 under D1 (a); **carried to ROADMAP C5** under D1 (b) |

## S2 on one page (closed 2026-10-22)

**Asked** (`design-S2.md`): with every head's OV edited at once on 410m, is the energy break (V,
A5's violation count) OV's non-symmetric part (Q1, arm a: each OV made symmetric) or its rotation
(arm b: rotation removed in each head's Schur planes); does in-context copying survive (Q2, G on
7d's probe); and is either carried by the pairs in S1's shared complex space (Q3, R3)? Each arm
against 40 random edits matched in size and in the head's norm, read only beyond the gain change
alone. **Exit met as extended** (Blocked 90 (b): 4 units + this close-out, #246–#250): R1's 10
cells, R2's 8 and R3's 9 labelled with D and the gain arm beside (R3 refused as frozen, unmatched;
its matched form read as description); the carried items run. Writing only here.

**Established (tier 1, one rung; v1's 7 prompts and 7d's 16 probe sequences):**

1. **The harness is the model.** One hook with a per-head core reproduces 410m to 3.4e-6; every
   arm reads back exactly (arm a at 143000 on all 384 heads to 4.3e-8); base V equals A5's panel
   at all 7 steps. G2 (the 10 induction heads' OV zeroed) loses copying at 4 of 4 steps (G 1.1–2.6
   against 7.7–12.5); G3 (the whole model moved to step 128) reaches V 0 at every scored step,
   below its controls at 512–4000.
2. **R1, "not shown" under both arms** (0 of 5 each). The symmetric part never lowers V beyond its
   controls at a scored step and raises it at 512 and 143000; rotation's removal is as its controls
   everywhere. G3′, every head given its own step-128 OV, never reaches V ≤ 7 (29–61): the one
   OV-only edit with a known destination does not undo the break, so "not shown" cannot be told
   from "out of an OV edit's reach". Its readings change sign along training (`gate_read.json`,
   description): *lower* at 512 and 1000 (V 29, 31 against controls from 50 and 38, about half
   their medians), ***higher* at 2000** (56 against 33–51), as its controls at 4000 and 143000.
   Step 128's OV maps lower the violations in the break's first steps and raise them at the start
   of the OV repulsive phase (1000–2000), every cell moved further.
3. **R2 under b: copying survives the rotation's removal at 4 of 4 steps** (G 0.02–0.03 below base
   at 1000–4000, 0.21 above at 143000). Under a (description) a random edit of a's size alone
   breaks most copying (G −0.1 to 4.6 against base's 8.2–12.4), and the symmetric part takes it
   below every one at 1000–4000. (b-eig) beside breaks it (G ≤ 2.4).
4. **Rotation inside a head's write space is a small part of its map.** (b-schur) removes about
   0.1 ‖K‖ and keeps the gain; at 8000, 0.88 of ‖K‖² reads from outside the head's write space and
   the skew part (b) removes is about 1 %. "Survives" and "as its controls" are read at that size.
5. **R3: S1's shared rotation carries nothing that shows.** The frozen selection did not select (a
   random plane in a head's own write space sits as high in S, so 37–69 % of pairs were "in S"),
   and R3 was refused, unmatched. Matched (each head's 2 pairs most in S₋ₗ against 2 of its others
   removing as much rotation, ratio 0.98–1.00): as its controls in all 9 cells. Against the right
   null 3.4–5.0 % of pairs lean into S (1 % by chance) and bS holds 84–94 % of them, but about half
   of bS is filler and 102–163 heads have no such pair.
6. **Structured edits move the stream further than random edits of the same weight norm** at most
   cells (the reason Blocked 88 made the guard description): arm a at 143000 by 11.9 against
   1.0–1.1; bS against bR by about 1.5× at 143000. Every "moved further" label is description.
7. **Beside, weights only** (S1's carried items): the shared write space meets the shared read and
   key spaces from 512 (about 20× Haar); sharing within layers beyond the global sharing rises
   256–4000 and falls by 143000; on a leave-the-layer-out S, S1's OV readings stand from 512 and
   its early `ov_neg` / `ov_pos` ones were the selection bias (Corrections received, above).

**Not established:** that the break is not OV's (G3′ is one edit and does not reach V ≤ 7); that
no OV-only edit can undo it; that copying needs no rotation in general (b edits about 1 % of
‖K‖²); anything under arm a beyond description (its controls are damaged models); any label at
matched displacement; a selection by the write-space test itself, or m ≠ 2; a pair's plane apart
from the Schur form's block order; why the selected pairs move the stream further (coherence
across heads, or the massive coordinates at 143000); which heads carry anything (Track B);
anything between grid steps or apart from the LR warmup; any other rung.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. scan, design, `sympy` check | #246 | arm b in Schur planes (Blocked 86 (a), (i)) |
| 1. code, G0, the preflight at 8000; R1b, R2 beside | #247 | G0 passes; refusals argued; frozen with the one-sided guard (Blocked 87 (a)) |
| 2. G1–G3; R1, R2 | #248 | the guard as description (88 (a)); R1 "not shown" under both; R2 under b "survives" 4 of 4 |
| 3. R3 on S₋ₗ; S1's R3 on S₋ₗ | #249 | R3 refused as frozen, unmatched (89 (a)); S1's OV readings stand from 512 |
| 4. the matched R3 (Blocked 90 (b)) | #250 | as its controls in all 9 cells (description) |
| close-out | this | writing only |

**Parked and open at close:**

| item | fate |
|---|---|
| `lit-S2.md` Parked 1 (the gradient-flow statement's exact hypotheses) | carried to C2's draft (ROADMAP §8): C2 states the energy against the theory, and "no theorem predicts monotone" is worded from summaries |
| arm a's D at 143000 (unit 2) | carried to S3 (already there), with the massive coordinates |
| bS's extra displacement (unit 4) | at 143000 carried to S3 with arm a's D (one projection-out pass reads both); at 1000 and 4000 dropped: the null is stronger for it and no reading uses it |
| whether any OV-only edit undoes the break (G3′'s reach) | carried to C2's draft as a sentence C2 may not go past (the break is neither shown OV's nor shown not OV's), with G3′'s readings as description (lower at 512 and 1000, higher at 2000); S4's toy can ask it with the circuit known |
| the matched-distance fallback (design "Pre-authorised fallbacks") | dropped: R1's positive answer needs V ≤ 7, and no arm cell came within 22 of it (lowest 29), so re-drawn controls could move a label but not R1's answer; R2's "survives" never rested on the guard |
| a selection by the write-space test, m ≠ 2, the Schur order's hold | dropped: arm b on every pair is as its controls (unit 2), so a subset has no effect to carry; m = 1 fails the match (43–63 heads unmatched). The same reason held before units 3 and 4: Q3 asked whether S's pairs carry what R1 and R2 read, and after unit 2 they read nothing under b. Unit 3 ran by the frozen scope; unit 4 ran by Blocked 90 (b), the user's, against the recommended (a), which gave this reason. It read as predicted (`LESSONS.md` 6, the S2 unit 3 instance) |
| S1's R1b, R2, R3 on S₋ₗ (carried in) | done, units 1 and 3 |
| the LR-warmup confound | carried to C2 (as A5's and S1's) |
| Phase 2b's `W_OV := S` (its open thread and After Phase 10 item) | done as arm a; a result line on 2b's card, its open thread narrowed to what was not run (an answer to an open thread, not a correction of a 2b result) |
| weighted projectors | stays on the card, "After Phase 10" |

## S1 on one page (closed 2026-10-21)

**Asked** (`design-S1.md`): at 8 steps of 410m, do the 384 heads share directions of the residual
stream, per subspace family, beyond a random arrangement; how does that change over training; and
what do the layers do inside the shared space? **Exit met as revised**: R1′'s 56 cells as
description, R3's table, the residual check (85 (a): the budget kept, 3 units + this close-out,
#243–#245). Writing only here.

**Established (tier 1, one rung, weights only; the residual check on v1's 7 prompts):**

1. **One null for the mean projector.** The rotation null, a norm-matched Gaussian draw and step 0
   are one law (math check fact 4), and the instrument passes it at every step (unit 1's G0–G2,
   unit 2's G0′–G2′). PR has an exact null value (fact 5).
2. **A count of shared directions cannot be calibrated on a trained pool,** and neither can ρ's
   effect side: the spectrum is dense at the null's edge (unit 1's G3), and a plant that adds
   dimensions only measures its own dilution (unit 2's G4, `/challenge-pr` on #244). So S1 reads ρ
   as an exact description and no count.
3. **Concentration over training** (ρ, 1 at the null): every family reads the null through 128;
   `qk_query`, `qk_key`, `ov_write` and `ov_cplx` leave it at 256, the rest at 512. `ov_write` is
   the most concentrated at 512–1000; its write, negative and complex families are most
   concentrated at 2000 and partly relax by 4000; read and positive concentrate steadily; QK falls
   through 4000 and is the most concentrated of the 7 from 2000 on (0.51–0.53 at the end against
   OV's 0.65–0.83). LN1's gain changes none of it. The spectrum is uneven; whether that is a few
   directions is not readable.
4. **Inside the 10 most shared directions** the radial sign reads S at no step from 512: at 512
   S sits inside random subspaces' range, and at 1000–4000 random subspaces hold no attractive
   part in 18–20 of 24 layers, so only "less repulsive than random" can show (in QK's S it does,
   often by a share under 0.05). The OV families' S is built from the heads it compresses, a bias
   of unknown size (their 256 "turn repulsive first" may be built in); QK's S is not, and there a
   collapsible negative pair is rarer than in a random subspace at 1000–4000.
5. **Not the residual stream's top variance** at any step (S holds 9–17× a random subspace's share
   at 512, never half the top-10's); **at 143000 the write, negative and complex shared spaces are
   partly 410m's massive-activation coordinates** (357, 130, 966; 278, 335, 914 early), which they
   avoid through 4000.

**Not established:** that a shared direction is used or causes anything (S2, S5); a calibrated
effect size for ρ in any family; that the concentration is a few directions rather than a graded
spectrum; anything between grid steps, or apart from the LR warmup (to ≈ 1430), where every first
change sits; the 2000 dip as more than one checkpoint; the massive-coordinate share between 4000
and 143000; R1b, R2; weighted projectors; R3 with attention weights; any other rung.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. scan, design, `sympy` check, preflight; 1. the gate | #243 | frozen as written (Blocked 81 (a)); G0–G2 pass, G3 fails in all 7, N refused (Blocked 82) |
| 2. re-gate + R1′ | #244 | the null side passes; G4 cannot see the pool; R1′ as description; QK Blocked 83 |
| 3. QK's ρ, R3, the residual check; close-out | this | as above (Blocked 83–85 (a)) |

**Parked and open at close:**

| item | fate |
|---|---|
| `lit-S1.md` Parked 1 (2603.13314's full text) | carried to C2's draft (ROADMAP §8): how S1's key and value rows are stated against OLMo-2 |
| `lit-S1.md` Parked 2 (410m's massive dims) | done, unit 3 |
| R1b (overlap between families' S), R2 (within-layer sharing, description) | carried to S2 (ROADMAP §7a row) |
| R3 on a leave-the-layer-out S for the OV families (the selection bias, `/challenge-pr` on #245, finding 3) | carried to S2 (ROADMAP §7a row): S2 intervenes on OV, so it needs OV's readings free of it |
| the massive coordinates in the end-state S | carried to S3 (ROADMAP §7a row): S3 projects tokens into S, so it states them or projects them out |
| a dims-kept gate for ρ (Blocked 83 (c), not taken) | dropped: no planted effect is known that S2 or S5 would need sized in ρ; S2's ablations read their own controls |
| the LR-warmup confound | carried to C2 (as A5's) |
| weighted projectors | stays on the card, "After Phase 10" |

## S1 unit 1, the gate (2026-10-20)

**Run.** `python -m trackS_subspace.s1 gate --out data/trackS/s1/2026-10-20-u1-gate` (main
tree; `HF_HOME` the main tree's `data/hf`), RTX 3080 for the Haar draws and rotations (float64),
CPU float64 for the bases. 2000 Haar draws per cell; per-cell thresholds for the invariant
families at the 8 scored steps and 8000 in `nulls.json` (kept for unit 2). The first run crashed
in G3 at `ov_pos` (fewer than k heads with ≥ 4 positive-real dimensions); fixed to record "not
plantable" and resumed with `--arms G3 G3l` on the saved thresholds (`log_resume.txt`); G3's
first five families had already printed k* None. G0's first, standalone run read 5e-8 against
1e-8: its metric √(1 − σ²) floors at √eps; replaced by ‖(I − UUᵀ)W‖₂ before the full gate
(`LESSONS.md` 6, 2026-10-20).

**Code per record** (`s1.py`'s sha256, first 12; the arm files of this run carry none, which
`_save` now adds, `/challenge-pr` on #243 finding 4): the exploration
`data/trackS/s1/2026-10-20-explore/explore.json` (file sha `124ed4de88cb`) `b1ec7b5ddcb9`; the
preflight `8a5b46bfc7b7`; the gate's nulls, G0, G1, G1iii, G2 `c2591b5e01de` (`nulls.json`);
G3 and G3l (resumed) `dfed5443a46c`, reconstructed: the committed file minus one later comment
label (`placed (design G0)`), so its hash is derived, not recorded. The committed `s1.py` at
#243's first commit is `e9606a4dc844`; none of the edits after the runs changes a computed value.

| arm | result | pass |
|---|---|---|
| G0 wiring | low-rank bases against dense SVD / eig, 20 random + 4 real heads: ≤ 6.5e-13 (singular), ≤ 4.7e-14 (invariant); Gram against dense P̄ 1.1e-15; GPU against numpy Haar λ₁ law KS p 0.65 (layer), 0.38 (all) | yes |
| G1 one null law | Gaussian-weight heads (norm-matched to 8000's: ‖W_Q‖ 6.39, ‖W_K‖ 6.09, ‖W_V‖ 5.43, ‖W_O‖ 4.42) and 8000's heads rotated, against Haar, both pool shapes, all 7 families: every KS p > 0.001 (`G1.json`) | yes |
| G1iii step 0 | N = 0 in all 7 families (pool `all`) | yes |
| G2 planted null | the rotation null at all 8 scored steps × 7 families: N = 0 in 56 of 56 | yes |
| G3 planted effect | 4 directions each in k of 384 heads; on a Haar carrier ΔN ≥ 4 in 100 % of 50 draws at every k (98 % once, `ov_cplx` at 32); on step 8000's real pool **0 % at every k in every family** (table below) | **no** |
| G3l (recorded) | one direction in k of a layer's 16 on the shuffle carrier: seen at 100 % from k = 4 (OV families) and k = 8 (`qk_key`); `qk_query` at k = 8 0.5 % (its shuffle maximum 0.550 > 8/16), 100 % from 12; `ov_pos` at 12 not plantable | — |

G3 on the real pool (median ΔN, range over 50 draws; the planted 4 directions are seen, at
λ ≈ k/384, in every draw):

| family | N at 8000 | T | k = 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| `ov_write` | 88 | 0.0312 | +1 [0, 2] | 0 [0, 2] | 0 [−1, 1] | −1 [−2, −1] | −3 [−4, −3] | −4 [−5, −4] |
| `ov_read` | 80 | 0.0312 | +2 [0, 3] | +1 [0, 3] | +1 [0, 1] | 0 [−2, 0] | −3 [−4, −1] | −5 [−6, −4] |
| `qk_query` | 87 | 0.0312 | +2 [1, 2] | +2 [0, 2] | +1 [0, 2] | +1 [0, 1] | 0 [−1, 0] | −2 [−3, −1] |
| `qk_key` | 79 | 0.0312 | +1 [0, 2] | +1 [−1, 2] | 0 [−1, 1] | −1 [−2, 0] | −2 [−3, −1] | −4 [−5, −3] |
| `ov_neg` | 23 | 0.0101 | +1 [0, 2] | 0 [−1, 1] | −1 [−2, −1] | −4 [−5, −2] | −7 [−9, −7] | −11 [−11, −10] |
| `ov_pos` | 15 | 0.0065 | −1 [−2, 0] | −2 [−3, −1] | −5 [−6, −3] | −8 [−8, −7] | not plantable (33 heads eligible) | — |
| `ov_cplx` | 67 | 0.0271 | +1 [0, 3] | +1 [0, 1] | 0 [−1, 1] | −1 [−3, −1] | −4 [−5, −3] | −6 [−8, −6] |

**Why it fails (diagnosed on 8000 after the arm, off the grid; mechanism corrected by
`/challenge-pr` on #243, finding 1).** At a trained step P̄'s eigenvalues are dense at the
threshold (`ov_write`: 0.0317, 0.0316, … 0.0312 | 0.0311, 0.0310, …; spacing ≈ 1e-4). A
strong planted direction takes one slot at λ ≈ k/n, and the rest of the spectrum is then, to
first order, P̄ compressed to u's complement, which interlaces P̄'s: it keeps N or N − 1
eigenvalues above T. In a dense edge which one is close to a coin flip, so each plant raises N by
1 only about half the time, **whatever the dose**: G3's larger doses could not have rescued it,
and the dense edge was visible before the freeze ("Seen before": 79–89 eigenvalues above T at
8000; `LESSONS.md` 6, 2026-10-20). A plant that only adds a dimension (nothing falls) gives ΔN
2–4 per 4 plants, as interlacing predicts; the frozen plant also removes each head's 16th
direction, so it loses more (and its planted mass is partly cancelled: m rose by +0.010 to
+0.013 against 0.021 planted, 10 draws, k = 32). Above 1.5 × T the spectrum thins (`ov_write`
26, `ov_read` 15, `qk_query` 43, `qk_key` 33 eigenvalues; above 3 × T: 0, 0, 13, 10).

**How much of the space the heads use (for Blocked 82; `/challenge-pr` on #243, finding 2 and
question 1).** No family at 8000 has a null space (≤ 1 eigenvalue below 1e-3; 99 % of the
mass in 978–989 of 1024 dimensions), but the use is uneven: the participation ratio
(Σλ)² / Σλ², the inverse of the mean pairwise kernel (math check fact 2), is 661, 729, 344, 412
(`ov_write`, `ov_read`, `qk_query`, `qk_key`) against 879–880 at step 0. Haar subspaces drawn
inside P̄'s top-PR eigenspace (20 draws) put **more** eigenvalues above T than the real pool
(145, 104, 321, 301 against 88, 80, 87, 79), so a count or excess near T is explained by uneven
dimension use alone; their λ₁ (0.042, 0.039, 0.070, 0.061) is 2–5× below the real one (0.084,
0.083, 0.347, 0.294), so the top of the spectrum is not. Inside the 99 %-mass span the same draws
give N = 2–3.

**What unit 1 establishes.** The instrument is exact and its null is right: Haar, the rotation
null, a norm-matched Gaussian draw and Pythia's step 0 are one law (G1, G1iii, G2). **A count of
eigenvalues above the null's edge is unstable at a trained step, because the spectrum is dense
there**; it does not show that no separated groups exist above the edge. N is refused in all 7
families by the frozen fallback.

**Not established.** Anything at the scored trained steps (no trained scored cell was read);
whether any statistic other than N is calibratable; whether the top eigenvalues form separated
groups; R3.

**Blocked 82 (yours; revised after `/challenge-pr` on #243).** (a) Go on under the frozen
fallback: unit 2 draws each cell's spectrum against the Haar band and step 0's, no statistic.
(b) Revise R1 and re-gate in unit 2: **R1 = the participation ratio** (threshold-free, the mean
pairwise kernel's inverse; null Haar 880, step 0 beside), per family per step, with the
invariant families' read per head dimension; **beside, the top eigenvalues against Haar drawn
inside the top-PR eigenspace** (does the top exceed what uneven dimension use gives), described,
not counted; the gate re-planted with the additive plant (nothing displaced), its recovery bar
fixed before the re-gate. (c) Close S1 at the gate (writing only); S2 takes each family's top
directions descriptively. **Recommended: (b)**, within the 3-unit budget (unit 2 = re-gate +
R1): it asks "how concentrated is the heads' use of the space, and when", which the data can
answer, and separates the top of the spectrum from dimension use; the excess over T (this
unit's first proposal) is dropped, since near T it measures dimension use.

## S1 unit 2, R1′ and its re-gate (2026-10-21)

Blocked 82 decided (b), the user's ("the planted test removed weight as it added it; the next
gate uses a plant that only adds"). Bars fixed in `design-S1.md` "Blocked 82 decided (b)" and
committed (`c9c8ecd`) before the code was written.

**Run.** `python -m trackS_subspace.s1 regate --out data/trackS/s1/2026-10-20-u2-regate`
(main tree's `data/`, `HF_HOME` its `data/hf`; run from `../Mets-work`; the directory is named
for the first launch, 2026-10-20, and its files were written 2026-10-21), code `e0b82943e83e`
(commit `9de8523`), 7,441 s under `systemd-inhibit`; a first launch was killed after an 8 h suspend
(`log_killed_after_suspend.txt`; `LESSONS.md` 11, 2026-10-21). Then `s1.py r1 --gate <that dir>`
→ `data/trackS/s1/2026-10-21-u2-r1/r1.json` (sha256 `c54df0db84f8`), code `55c4859e42e9` (commit
`bd85227`: the per-step write and resume only), 2,849 s. After the runs, `pr_of` takes d from
the basis instead of 1024 (found by fact 6's test on d = 96); every run had d = 1024, so no
computed value changes. RTX 3080 float64 for the Haar bands and
the top's null; CPU float64 for bases and Grams. Re-gate files (sha256, first 12): `bands.json`
`c947528c5218`, `G0p` `87ea22c44c1a`, `G1iii_p` `44e5d797d509`, `G2p` `8d952843df8a`, `G4`
`33669c326203`, `G4t` `461a841eb7ca`.

| arm | result | pass |
|---|---|---|
| G0′ | PR by Frobenius against eigenvalues ≤ 4.4e-16 relative; the 28 distinct bands' mean tr P̄² against the closed form: max \|z\| 2.96 | yes |
| G1iii′ | step 0's PR inside its band in all 7 (`ov_write` 879.5 in [878.9, 880.6]; `ov_neg` 204.9 in [204.0, 206.0]) | yes |
| G2′ | the rotation null inside the band in 56 of 56 | yes |
| G4 | additive plant: detected on Haar in 100 % at every dose, every family; on the real pool the size bar holds at k = 32 in the 5 OV families and **fails in `qk_query` and `qk_key` at every dose to 64**; both by dilution, not by the pool (below) | **no** (QK), and uninformative |
| G4r | the displacing plant (recorded): on Haar the same Δχ as the additive one (±5 % in the singular families); on the real pool (dimensions kept, so no dilution) 0.66 (`qk_query`) to 0.99 (`ov_pos`) of its Haar response at k = 32 | — |
| G4t | the top (recorded): an added direction is recovered above the m-dim null in every draw from k = 32 (`ov_write`, `ov_read`), 24 (`ov_cplx`), 12 (`ov_neg`, `ov_pos`); QK at most 0.16 (`qk_query`) and 0.66 (`qk_key`, at 48) to k = 64, the planted direction mixing with the pool's own top (median \|cos\| 0.71–0.92 from k = 32) | — |

G4 on the real pool, the median Δχ as a share of the Haar carrier's (bar: ≥ 0.75 in 80 % of draws):

| family | χ₀ at 8000 | k = 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| `ov_write` | 3.8e-4 | 0.56 | 0.67 | 0.79 | 0.84 | — | — |
| `ov_read` | 2.3e-4 | 0.73 | 0.80 | 0.87 | 0.90 | — | — |
| `qk_query` | 1.8e-3 | −1.08 | −0.53 | 0.00 | 0.26 | 0.51 | 0.63 |
| `qk_key` | 1.3e-3 | −0.51 | −0.12 | 0.27 | 0.45 | 0.64 | 0.73 |
| `ov_neg`, `ov_pos`, `ov_cplx` | 3.5e-4, 7.1e-4, 2.2e-4 | 0.79–0.96 | | | 0.92–0.98 | — | — |

**Why QK fails (diagnosed after the arm).** The additive plant adds dimensions, raising tr P̄ by
a share ε = 4k/6144, and χ = (tr P̄² − E₀)/(tr P̄)² falls by 2εχ₀ on that alone (math check
fact 6). It predicts the shares above from χ₀: `qk_query` −1.08 at k = 12 (observed −1.08) and
0.24 at 32 (0.26); `ov_write` 0.56 at 12 (0.56). So the deficit is the pool's own concentration
diluted by the plant's added dimensions, not a plant absorbed. **The design's "χ is additive
under a plant" was wrong on a concentrated pool**, and the dilution was derivable before the run
from unit 1's PR (344, 412). The singular families keep k_h = 16 at every step, so in the data
tr P̄ never moves and PR and tr P̄² are one-to-one; the dilution exists only in the plant.

**The additive plant does not see the pool (`/challenge-pr` on #244, finding 1; reproduced).**
Re-scored from the saved draws on κ = tr P̄² − E₀ tr P̄² (the dilution removed), the real pool
responds at 0.97–1.00 of the Haar carrier's at every dose in `ov_write`, `ov_read`, `qk_query`
and `qk_key` (every draw past 0.75 from k = 12): a random direction added to k heads overlaps any
pool about equally, so dilution was the only pool-dependent part of G4. **G4's pass and fail
therefore say nothing about the trained pool, and k* is not a resolution** (it is where the
plant outruns the dilution at 8000's concentration). The only plant that reads the pool is the
dims-kept one (G4r): on the real pool `qk_query` 0.03, 0.31, 0.56, 0.66, 0.78, 0.84 of its Haar
response at k = 12–64, `qk_key` 0.32–0.88, `ov_write` 0.68–0.89, `ov_read` 0.80–0.93 (k ≤ 32).
So what unit 2 calibrates is the null side (G0′, G1iii′, G2′); ρ is an exact description of the
weights against an exact null, and its effect side is uncalibrated.

**R1′ (pool `all`, r = 16, raw frame). ρ = PR / PR₀** (1 at the null; QK spectrum only):

| family | 0 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 143000 |
|---|---|---|---|---|---|---|---|---|
| `ov_write` | 1.000 | 0.999 | 0.994 | 0.889 | 0.662 | **0.586** | 0.735 | 0.653 |
| `ov_read` | 1.000 | 1.000 | 1.000 | 0.994 | 0.982 | 0.951 | 0.896 | 0.814 |
| `ov_neg` | 0.999 | 1.000 | 1.000 | 0.925 | 0.779 | **0.686** | 0.845 | 0.825 |
| `ov_pos` | 1.001 | 0.999 | 1.000 | 0.985 | 0.967 | 0.910 | 0.865 | 0.810 |
| `ov_cplx` | 1.000 | 1.000 | 0.997 | 0.935 | 0.761 | **0.675** | 0.822 | 0.793 |

(`r1.json` carries a `k_star` per cell from G4; by the paragraph above it is not a resolution and
is not read.) ρ is a ratio, so equal moves in the excess kernel shrink as ρ falls; "most at 2000"
is a statement in ρ's units. The invariant families' pools are small at some steps: `ov_pos`
holds 26 and 23.5 effective dimensions (PR₀) at 512 and 1000 across all 384 heads, so its ρ
there compares a pool of ~25 dimensions with one of ~380 at 143000.

PR itself (`ov_write` 879.5 → 515.5 at 2000 → 574.6) and the invariant families' PR₀ (which
moves with the heads' type mix: `ov_pos` 209.5 → 23.5 at 1000 → 383.6) are in `r1.json`. The
band's half-width in ρ is ≤ 0.0011 (singular, `ov_cplx`) and ≤ 0.0063 (`ov_neg`, `ov_pos`), so every move from 512 on
is far outside the null; ρ is exact on the weights (no sampling), so a step-to-step difference
has no noise to clear, and no calibrated size says how large a difference matters.

**Beside (ρ).** r = 8 and 32: the same shape, r = 8 more concentrated (`ov_write` 0.534 at 2000),
r = 32 less (0.669). LN1's gain folded in: equal to raw within 0.003 everywhere except at 143000 in `ov_neg`
(0.797 against 0.825) and `ov_pos` (0.801 against 0.810), so the gain does not create the sharing.

**The top, described (λᵢ over the largest λ₁ of Haar in R^m, m = round(PR); no count).** Below
1 at steps 0–128 in every family (the m-dim null is conservative: it packs the same dimensions
into fewer). λ₁ crosses it from 512 in `ov_write` (2.14; 3.06 at 2000, 3.26 at 143000), `ov_cplx`
(1.61; 2.45, 2.01), `ov_read` (1.13 → 2.05 at 143000) and `ov_neg` (1.04; 1.59, 1.33); `ov_pos`
only at 143000 (1.16). λ₁₀ is above it in `ov_write` and `ov_cplx` from 512 (1.16–1.54) and in `ov_read`
from 2000 (1.05–1.34). QK's λ₁ (spectrum only): `qk_query` 0.031 (Haar) → 0.347 at 4000 → 0.187 at 143000;
`qk_key` → 0.259 → 0.230. **What this null can say (`/challenge-pr` on #244, finding 2):** for
any spectrum λ₁ ≥ tr P̄ / PR, with equality only when it is flat, and the m-dim null is that flat
case plus its edge; a smoothly graded spectrum with no special directions also exceeds it. So
"above the m-dim null" means the spectrum is uneven, not that a few directions stand apart.

**What unit 2 establishes (tier 1, one rung, weights only).** In the five OV families the heads'
subspaces read as an independent random arrangement through step 256, as at init (*corrected
2026-10-21, `/challenge-pr` on #245: `ov_write` and `ov_cplx` are already below their bands at 256,
ρ 0.9935 and 0.9975; the other three through 256*); from 512 they
concentrate on fewer directions of the residual stream (ρ falls), most at step 2000 in the write,
negative and complex families in ρ's units, partly relaxing by 4000; the read and positive
families concentrate steadily to 143000. The null side is calibrated (G0′–G2′); the effect side
is not (G4 cannot see the pool), so these are descriptions against an exact null.

**Not established.** QK's ρ (Blocked 83; its λ₁ is descriptive); that the concentration is a few
directions rather than a graded spectrum (the top's null cannot tell them apart); that the 2000
dip is more than one checkpoint's reading (no step between 1000 and 4000 but 2000), **and the
dip and partial relaxation sit just after Pythia's learning-rate warmup ends (1 % of 143000 ≈
step 1430)**, which this design cannot separate from anything the heads do (cf. A5's warmup
confound); how large a change in ρ matters (no calibrated effect size); which directions are
shared or whether they are the residual stream's top variance (unit 3's beside check); any other rung.

**Blocked 83 (decided (a), 2026-10-21, the user's; "S1 unit 3"): QK's ρ, now that G4 is known not to see the pool** (rewritten after
`/challenge-pr` on #244; the first version's (b), a κ-scored re-gate, cannot fail: the saved
draws already pass it). (a) Read ρ in all 7 families as a description against the exact null,
QK included, with the effect side stated as uncalibrated and G4r's dims-kept response beside;
QK's 16 cells need only tr P̄² (CPU, minutes). (b) Keep QK at spectrum only, the frozen
fallback's letter, though the failure that triggered it is the plant's, not the pool's.
(c) A dims-kept gate (G4r's plant) with a bar fixed before it runs, for every family. **Recommended:
(a)**: G4 gives no evidence for or against any family, so passing OV and refusing QK on it is
arbitrary; against it, the fallback was frozen, and (a) sets it aside after seeing the failure.

**Blocked 84 (decided (a), 2026-10-21, the user's): S for R1b and R3** (N defined it; refused). (a) S = each
family's top 10 eigenvectors at each step, R3's Haar subspace of the same dimension as its
comparison, as designed. (b) S = the eigenvectors above the m-dim null's largest λ₁ (a count;
and that null only says "uneven", finding 2 above). (c) Drop R1b; R3 reads only the top 1.
**Recommended: (a)**: fixed width, no count, and R3 keeps its designed null.

**Blocked 85 (decided (a), 2026-10-21, the user's): the budget** (`/challenge-pr` on #244, question 4). Unit 3 would hold R1b,
R2, R3, the residual check, QK's ρ (83 (a)) and the close-out. (a) Keep 3 units: unit 3 = QK's
ρ, R3 on S, the residual check and the close-out; R1b and R2 carried to S2 (R2 is description
only since the preflight). (b) Extend to 4 on the record: unit 3 = QK, R1b, R2; unit 4 = R3,
the residual check, the close-out. **Recommended: (a)**: the budget rule, and S2 reads the
shared directions next anyway.

## S1 unit 3, QK's ρ, R3 and the residual check (2026-10-21)

Blocked 83–85 decided (a), (a), (a), the user's ("your letters"). Readings fixed in `design-S1.md`
"Blocked 83–85 decided" and committed (`1b11fd3`) before any run of the code.

**Run.** From `../Mets-work`, `HF_HOME` the main tree's `data/hf`, under `systemd-inhibit`, side by
side; code `1ab2936d1e67` (`1b11fd3`). `s1.py r1qk --gate data/trackS/s1/2026-10-20-u2-regate
--out data/trackS/s1/2026-10-21-u3-r1qk`: 4,918 s, `r1_qk.json` sha256 `f6985b2d6a26`. `s1.py r3
--out data/trackS/s1/2026-10-21-u3-r3`: 5,254 s, `r3.json` `292f76fe08a7`. Then `s1.py massive
--out data/trackS/s1/2026-10-21-u3-massive` (`lit-S1.md` Parked 2), code `bfc14bb35d60`
(`3620806`), `massive.json` `798552804cb3`. RTX 3080: the forward passes (float32) and the top's
null; CPU float64 for the rest. First outputs opened before the rest ran: step 0 reads the null in
both (QK ρ 1.0004, 1.0001; S's variance share 0.039 against a Haar S's 0.040).

**QK's ρ (83 (a)): description against the exact null; the effect side is uncalibrated, as in
every family.**

| family | 0 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 143000 |
|---|---|---|---|---|---|---|---|---|
| `qk_query` | 1.000 | 1.000 | 0.996 | 0.938 | 0.722 | 0.498 | **0.403** | 0.525 |
| `qk_key` | 1.000 | 1.000 | 0.997 | 0.973 | 0.858 | 0.660 | 0.521 | **0.513** |

QK leaves the band at 256 (PR 875.8 and 877.0 against [878.9, 880.6]), with `ov_write` (0.9935)
and `ov_cplx` (0.9975), and falls through 4000 with no dip at 2000. `ov_write` is the more
concentrated at 512 and 1000 (0.889, 0.662); `qk_query` from 2000 (0.498 against 0.586), and at
143000 QK is the most concentrated of the 7 (OV 0.65–0.83, unit 2). *(First written "a step before
the OV families", from unit 2's "OV reads the null through 256"; both corrected by `/challenge-pr`
on #245, finding 1.)* Beside: r = 8 more concentrated (`qk_query` 0.249 at 4000), r = 32 less
(0.613); LN1's gain folded in equals raw within 0.004. The top: λ₁ above the m-dim null from 256
(1.39, 1.23), λ₁₀ from 1000 (1.41, 1.35); as in unit 2, that says "uneven", not "a few".
G4r's dims-kept plant (unit 2, step 8000) was hardest to see in QK's pool (`qk_query` 0.03–0.84,
`qk_key` 0.32–0.88 of its Haar response at k = 12–64), so a planted change may move QK's ρ less
than OV's; not calibrated.

**R3 on S** (84 (a): S = P̄'s top 10 eigenvectors). Layer median over the 24 layers, then (layers
above / below Haar's [min, max] over 100 draws). The attractive share of C's symmetric part:

| family | 0 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 143000 |
|---|---|---|---|---|---|---|---|---|
| `ov_write` | 0.50 (0/1) | 0.45 (1/0) | 0.02 (0/19) | 0.00 (0/1) | 0.00 (3/0) | 0.00 (3/0) | 0.05 (13/1) | 0.52 (10/7) |
| `ov_read` | 0.52 (0/0) | 0.48 (0/1) | 0.24 (0/3) | 0.00 (0/3) | 0.00 (1/0) | 0.00 (0/0) | 0.00 (7/2) | 0.68 (7/5) |
| `qk_query` | 0.52 (0/0) | 0.50 (0/0) | 0.15 (3/10) | 0.00 (0/0) | 0.00 (10/0) | 0.03 (22/0) | 0.08 (19/2) | 0.40 (11/6) |
| `qk_key` | 0.52 (0/0) | 0.50 (0/0) | 0.22 (0/2) | 0.00 (0/0) | 0.00 (7/0) | 0.04 (17/0) | 0.13 (19/2) | 0.54 (10/7) |
| `ov_neg` | 0.38 (0/5) | 0.34 (0/7) | 0.15 (0/11) | 0.00 (0/3) | 0.00 (1/0) | 0.00 (0/0) | 0.01 (9/2) | 0.43 (9/7) |
| `ov_pos` | 0.65 (3/0) | 0.59 (4/0) | 0.43 (3/0) | 0.02 (4/0) | 0.00 (4/0) | 0.00 (8/0) | 0.12 (12/0) | 0.79 (12/7) |
| `ov_cplx` | 0.48 (0/1) | 0.45 (0/1) | 0.04 (0/20) | 0.00 (0/3) | 0.00 (3/0) | 0.00 (0/0) | 0.00 (9/2) | 0.54 (9/7) |

Mean |arg λ| of C's complex eigenvalues, same form:

| family | 0 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 143000 |
|---|---|---|---|---|---|---|---|---|
| `ov_write` | 1.60 (0/0) | 1.66 (0/0) | 2.62 (14/0) | 3.03 (18/0) | 3.03 (4/2) | 2.91 (0/8) | 2.68 (1/10) | 1.53 (5/9) |
| `ov_read` | 1.46 (0/1) | 1.57 (1/0) | 2.07 (1/0) | 2.96 (15/0) | 3.03 (5/1) | 2.97 (1/2) | 2.89 (0/5) | 1.50 (4/6) |
| `qk_query` | 1.55 (0/0) | 1.58 (0/1) | 2.04 (1/0) | 2.77 (1/0) | 2.93 (1/8) | 2.86 (1/12) | 2.47 (2/13) | 1.55 (8/8) |
| `qk_key` | 1.54 (0/0) | 1.63 (0/0) | 1.86 (0/1) | 2.85 (9/0) | 2.97 (1/4) | 2.86 (1/16) | 2.42 (2/16) | 1.30 (4/10) |
| `ov_neg` | 1.82 (3/0) | 1.66 (0/0) | 2.25 (5/0) | 3.00 (19/0) | 3.03 (3/1) | 2.98 (1/4) | 2.89 (2/7) | 1.51 (4/7) |
| `ov_pos` | 1.45 (0/0) | 1.41 (0/0) | 1.73 (0/0) | 2.74 (2/3) | 2.99 (0/3) | 2.97 (3/4) | 2.67 (0/13) | 1.19 (3/9) |
| `ov_cplx` | 1.54 (0/0) | 1.50 (3/0) | 2.58 (12/0) | 3.02 (17/0) | 3.06 (9/2) | 2.95 (0/6) | 2.78 (2/12) | 1.39 (7/9) |

Negative-pair flag, layers flagged (Haar's mean count): `ov_write` 3 (1.5), 6 (2.1), 6 (7.5),
12 (16.2), 15 (21.7), 15 (21.7), 3 (19.8), 6 (7.5); the other families alike, within S fewer
than Haar's mean at 512–4000 in all 7 (`r3.json`). The complex share sits at 0.4–0.8 with 0–6
layers outside Haar in every cell.

- **The selection bias.** In the OV families S is built from the same heads whose summed OV it
  compresses, so it is not a random subspace for V_l even at init: steps 0–128 read 0–7 layers
  outside on `share` (`ov_neg` 5–7, `ov_pos` 3–4), 0–3 on |arg|; Haar's [min, max] alone puts
  about 0.5 of 24 outside. That floor does not bound it at trained steps, where V_l's structure
  grows (`/challenge-pr` on #245, finding 3), so **the OV families' R3 readings carry a bias of
  unknown size, and QK's (S from W_Q, W_K) are the clean ones.** A leave-the-layer-out S, which
  removes it, is carried to S2.
- **The radial sign cannot read S from 512.** At 512 S sits inside random subspaces' range (0–4
  layers outside). At 1000–4000 random subspaces hold no attractive part at all in 18–20 of 24
  layers (Haar's range [0, 0]: A3's E2 seen in compressions), and the share cannot go below 0,
  so there only "less repulsive than random" can show, and any attractive part counts. *(First
  written "at 512–2000", and the QK reading below as "less repulsive than random" without this;
  corrected by `/challenge-pr` on #245, finding 2.)*
- **What it shows.** At 256 the write, negative and complex S are more repulsive than random
  (below Haar in 11–20 layers; OV, so possibly built in). From 1000 the QK S hold a small
  attractive part in 10–22 layers where random subspaces hold none, often a share under 0.05
  (`qk_query` at 2000: 22 layers, 13 of them below 0.05). At 256–512 the write, read, negative and
  complex S hold eigenvalues nearer π than random (12–19 layers); from 2000 the QK S hold smaller
  angles (12–16 layers below).
- **Collapsibility.** Inside S the two most negative directions are a near-equal pair in fewer
  layers than in a random subspace at 512–4000 in every family; in QK's S (no selection),
  `qk_query` 10, 9, 6 against Haar's 21.9, 21.8, 19.9 at 1000, 2000, 4000.
- **Step 0's comparison** (the trained S on step 0's V_l): `share` medians 0.35–0.61 at every
  step and family, so the trained readings' fall to 0 is V_l's training, not S's choice. **Its
  "outside" counts are not read:** `summary_step0` scores step 0's compressions against the trained
  step's Haar draws, the wrong null (found reading the output; `LESSONS.md` 2, 2026-10-21).
  `r3.json` (code `1ab2936d1e67`) still holds them; since `/challenge-pr` on #245 (finding 4)
  `r3_summary` writes no "outside" count without a Haar set, `summary_step0` is medians only, and
  a test pins it (no value read here changes).

**The residual check.** No family × step has a layer-median q ≥ 0.5, so **no S is reported as
the residual stream's top variance.** q median (max over layers); a Haar S's q beside:

| family | 0 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 143000 |
|---|---|---|---|---|---|---|---|---|
| `ov_write` | 0.04 (0.04) | 0.04 (0.10) | 0.14 (0.19) | **0.33** (0.37) | 0.24 (0.33) | 0.12 (0.15) | 0.15 (0.18) | 0.26 (0.52) |
| `ov_read` | 0.04 (0.04) | 0.02 (0.04) | 0.04 (0.04) | 0.18 (0.21) | 0.21 (0.24) | 0.16 (0.25) | 0.15 (0.26) | 0.02 (0.13) |
| `qk_query` | 0.04 (0.04) | 0.02 (0.04) | 0.16 (0.18) | 0.27 (0.29) | 0.24 (0.30) | 0.22 (0.27) | 0.17 (0.20) | 0.17 (0.19) |
| `qk_key` | 0.04 (0.04) | 0.02 (0.04) | 0.11 (0.14) | 0.27 (0.29) | 0.23 (0.35) | 0.22 (0.33) | 0.18 (0.29) | 0.07 (0.10) |
| `ov_neg` | 0.04 (0.04) | 0.02 (0.04) | 0.03 (0.04) | 0.17 (0.19) | 0.18 (0.28) | 0.07 (0.08) | 0.09 (0.10) | 0.15 (0.28) |
| `ov_pos` | 0.04 (0.05) | 0.02 (0.04) | 0.02 (0.04) | 0.04 (0.07) | 0.07 (0.10) | 0.12 (0.14) | 0.10 (0.12) | 0.05 (0.10) |
| `ov_cplx` | 0.04 (0.05) | 0.03 (0.04) | 0.09 (0.13) | 0.29 (0.34) | 0.20 (0.28) | 0.08 (0.09) | 0.13 (0.16) | 0.17 (0.45) |
| Haar S | 0.040 | 0.017 | 0.016 | 0.019 | 0.034 | 0.046 | 0.046 | 0.015 |

S holds 9–17× a random subspace's share of the top-10 variance at 512 (all but `ov_pos`), less
later. Position 0 kept changes q by ≤ 0.07. The uncentred second moment raises the end in the
write, negative and complex families (median 0.39, 0.26, 0.32; max 0.66, 0.50, 0.60): what those
S hold at 143000 is partly the stream's mean, where the massive activations sit.

**Massive activations (`lit-S1.md` Parked 2).** 410m's are coordinates 357, 130 and 966 from block
8's input on (max |x| 500–650 at layers 8–20, 140–350× the median coordinate's; position 0 and the
rest alike), with 278, 335 and 914 at layers 1–4 (|x| 10–35). **At 143000 the write, complex and
negative S are partly these coordinates:** their S-weight Σᵢ ‖U_Sᵀ eᵢ‖² over the six is 1.54, 1.10
and 0.88 of S's 10 dimensions (a Haar S: 0.06; 278 alone 0.65 in `ov_write`); `qk_query` 0.26,
`ov_pos` 0.16, `qk_key` 0.14, `ov_read` 0.02. By S's top-3 coordinates (with 125, a seventh), they
fill 22 of `ov_write`'s 30 slots at 143000 and 0–2 at every step through 4000 (`r3.json`
`top_coords`). So the concentration of 512–4000 is not these coordinates; the end state partly
is. The preflight's "≤ 1.8 %" (`lit-S1.md` §4) was step 8000's top direction only.

**What unit 3 establishes (tier 1, one rung, weights; the residual check on v1's 7 prompts).**
QK's heads leave the null at 256 with `ov_write` and `ov_cplx`, keep concentrating through 4000,
and are the most concentrated family from 2000 on. Inside the 10 most shared directions the
radial sign reads S at no step from 512 (random subspaces are as repulsive or more); in QK's S,
which carries no selection bias, a collapsible negative pair is rarer than in a random subspace
at 1000–4000. No shared space is the residual stream's top variance; at 143000 the write,
negative and complex ones are partly the massive-activation coordinates.

**Not established.** That any of this is used (S2, S5); QK's ρ against a calibrated effect; R3 on
attention-weighted OVs (fenced); anything between grid steps, or separating the 256–4000 changes
from the LR warmup (to ≈ 1430); whether the end state's massive-coordinate share holds at
8000–54000 (only 143000 and the grid read); S at another width than 10; R1b and R2 (carried to S2);
the OV families' R3 readings free of selection (a leave-the-layer-out S, carried to S2); the
residual check for the read-side spaces in the frame the heads read (LN1's centring and gain;
raw against raw here, as the design's frame; `/challenge-pr` on #245, finding 5).

## S2 unit 1, the code, G0, the preflight at 8000, R1b and R2 (2026-10-21)

Blocked 86 decided (a) and (i), the user's: arm (b) is (b-schur), (b-eig) beside with no label;
3 units + close-out. Code `trackS_subspace/s2.py` (choices fixed in code: `design-S2.md` "Unit 1:
fixed in code"); tests `tests/test_trackS_s2.py` (11, pure), `tests/test_trackS_s2_smoke.py` (4,
tiny GPT-NeoX in float64). No scored step's arm, control, V or G was computed.

**Run.** From `../Mets-work`, `HF_HOME` the main tree's `data/hf`, under `systemd-inhibit`; RTX 3080,
float32, TF32 off for the passes; frames and arms float64 on the CPU. Records in the main tree's
`data/trackS/s2/`:

| job | code (file sha256) | record | time |
|---|---|---|---|
| `s2.py g0` | `16b25495163b` (`077b31c`) | `2026-10-21-u1-g0b/g0.json` `95522987245d` | 87 s |
| `s2.py preflight` (the GPU part) | `98bffaa38aca` (`dfb9f70`'s file without the `preflight-tool` split, which moves no number) | `2026-10-21-u1-preflight/preflight_runs.json` `5ed46da5f616` | 129 s |
| `s2.py preflight-tool` (the reports) | `2fac8f6c646a` (`dfb9f70`) | `preflight_report_a.json` sha256 `2156db3ca9dc0474…`, `_b.json` `769d0f66f778e3f5…` | — |
| `s2.py beside` | `b672917e92f9` (between `955763b` and `077b31c`; `beside` is the same in both) | `2026-10-21-u1-beside/beside.json` `f9cb4925e444` | 1,124 s |

All ran under the `mets` env's transformers 4.44.2. After them, `77ef95a` takes the attention
weights by shape (4.57 returns them in another place); under 4.44 it selects the same tensor, so no
record moves; the smoke tests pass under both (`LESSONS.md` 2).

Set aside, not read: two OOM runs of `g0` (8 feeds' logits held on the GPU; the hook kept the
previous step's model alive), one of the preflight (G3's two weight copies on the GPU), the first
preflight-tool input (per-prompt counts against a summed null) — `data/trackS/s2/oom-attempts/`,
`2026-10-21-u1-preflight/first-tool-input/`. The first G0 (`077b31c`'s parent) failed its (b)
read-back; it is the reason for the revision below.

**G0 passes.**

| check | worst | bar |
|---|---|---|
| hook at the base core against the model, logits and raw hidden states (7 v1 prompts, the probe) | 2.0e-6 at 0, 3.4e-6 at 8000 | 1e-4 |
| (b-eig) by the hook against (b-eig) written in W_V (logits, centred hidden states) | 2.3e-6, 6.8e-6 | 1e-4 |
| read-back, 20 heads at 8000: (a) against sym K; (b), (b-eig) maps against the exact K′ | 3.6e-8; 4.6e-8, 1.2e-7 of ‖K‖ | 1e-6 |
| the exact core's eigenvalues against the arm's: (b) (and their imaginary parts), (b-eig) | 1.9e-10 (0.0), 1.0e-15 | 1e-6 |
| a doubly matched control's two norms, every head, both arms | 3.7e-16 | 1e-6 |
| base V against the panel at 0, 256, 512, 1000, 2000, 4000, 143000 | 0, 19, 56, 56, 61, 57, 70: equal | equal |

**G0's (b) read-back, revised before the freeze (Blocked 87).** As written it gated the
eigenvalues of the float32 core the hook holds. For (b-schur) they missed by 3.1e-4 ‖K‖ (imaginary
parts 2.4e-4) while the float32 map equals the exact K′ to 2.9e-8 and the exact core has the arm's
eigenvalues to 1.9e-10: (b-schur) leaves near-repeated eigenvalues (gaps 7.5e-7–2.4e-5 ‖K‖ on the
20 heads), which float32 rounding moves by about √eps. The gate now checks the map and the exact
core; the float32 core's eigenvalues are recorded beside.

**Sizes (every arm, every head; median [min, max] of ‖K′ − K‖ / ‖K‖; ‖K′‖ / ‖K‖ median).**

| step | a | b (b-schur) | b-eig (beside) | g_a | g_b |
|---|---|---|---|---|---|
| 0 | 0.71 [0.71, 0.71]; 0.71 | 0.12 [0.11, 0.13]; 0.99 | 1.74 [1.31, 7.31]; 1.75 | 0.29 | 0.01 |
| 256 | 0.71; 0.71 | 0.12 [0.11, 0.13]; 0.99 | 1.72; 1.74 | 0.29 | 0.01 |
| 512 | 0.70; 0.71 | 0.12 [0.10, 0.13]; 0.99 | 1.60 [1.16, 5.47]; 1.65 | 0.29 | 0.01 |
| 1000 | 0.67; 0.74 | 0.09 [0.08, 0.13]; 1.00 | 0.66; 1.09 | 0.26 | 0.00 |
| 2000 | 0.66; 0.75 | 0.08 [0.05, 0.12]; 1.00 | 0.48; 1.03 | 0.25 | 0.00 |
| 4000 | 0.68; 0.74 | 0.09 [0.04, 0.16]; 1.00 | 0.78; 1.11 | 0.26 | 0.00 |
| 8000 | 0.68 [0.38, 0.71]; 0.73 | 0.09 [0.02, 0.18]; 1.00 | 0.98; 1.21 | 0.27 | 0.00 |
| 143000 | 0.69 [0.37, 0.71]; 0.72 | 0.10 [0.02, 0.23]; 0.99 | 1.23 [0.05, 4.48]; 1.38 | 0.28 | 0.01 |

(b-schur) removes about 10 % of a head's norm and keeps its gain (g_b ≤ 0.03): 10,368–10,968 of the
12,288 eigenvalue pairs are complex at every step, but at 8000 (all 384 heads, medians of ‖K‖²'s
shares) 0.88 of a head's map reads from outside its write space (N, which (b-schur) keeps), 0.04 is
the restriction's coupling between planes and 0.08 its diagonal blocks, of which the skew part
(what (b-schur) removes) is 0.008. Rotation inside the write space is a small part of these maps.
(b-eig) edits by 0.48–1.74 ‖K‖ (no head refused for its condition number).

**The preflight at 8000 (off the grid; not a reading).** V per configuration (sum over the 7
prompts, at most 168), G and D:

| | V | G | D |
|---|---|---|---|
| base | 59 | 12.32 | — |
| arm a / gain g_a / G1's 41st draw | 21 / 50 / 37 | 1.36 / 10.50 / 2.61 | 0.949 / 0.306 / 0.709 |
| a's 40 controls (min, median, max) | 30, 36, 39 | 2.09, 2.75, 3.22 | 0.693, 0.713, 0.735 |
| arm b / gain g_b / G1's draw | 57 / 59 / 60 | 12.30 / 12.31 / 12.29 | 0.110 / 0.005 / 0.102 |
| b's 40 controls | 56, 59, 61 | 12.21, 12.26, 12.33 | 0.100, 0.102, 0.106 |
| b-eig (beside) | 42 | 1.59 | 0.829 |
| G3, t = 0.25 / 0.5 / 1 (toward step 128) | 34 / 18 / 0 | 3.83 / 0.12 / 0.00 | 0.553 / 0.840 / 0.981 |

The tool (`tools/design_preflight.py`; V, "less", null the arm's 40 controls, dose 0 the gain arm
and G1's draw, doses G3's t, `decision_p` 1/41, bounds [0, 168]): no size, headroom, dose-side or
power refusal for either arm (G3 at t = 1 and 0.5 below every control of both; 0.25 below b's, not
a's). It refuses, for both arms, `trained` (one value: V is one number per model, as the design
foresaw) and `floor` (1/41 calibrates to E 3.2), and for arm a `band:controls` (base's 59 lies
outside a's controls' [31, 39]). Argued in `design-S2.md` "The preflight's refusals".

**What the preflight shows for the freeze.**
- **Both arms move the stream further than every control** (a: D 0.949 against 0.693–0.735; b: 0.110
  against 0.100–0.106). Under the frozen displacement guard such a cell is "moved further" and not
  read; if the scored steps look like 8000, R1 and R2 read nothing. Blocked 87 (after
  `/challenge-pr` on #247) makes the guard one-sided: it withholds only the labels displacement
  can produce (R1 *lower*, R2 *lost* / *weakened*).
- Arm a's controls alone move V from 59 to 30–39 and G from 12.3 to 2.1–3.2: a random edit of arm
  a's size in the head's span already breaks most in-context copying, so R2 under a compares
  damaged models (87: R2 under a is description).
- Arm b at its size moves nothing measurable beyond its controls (V 57 in 56–61, G inside).

The records above carry the code sha but not the transformers version or the checkpoints' commits
(`/challenge-pr` on #247, finding 6): `meta_backfill.json` in each record directory holds both,
taken after the run from the same HF cache (4.44.2; step 8000 at `8e89b393da98…`); new runs write
them (`s2.run_meta`).

**R1b and R2 (beside, weights only, description; S1's raw frame, r 16, S = P̄'s top 10).**

| step | write–read | write–key | neg–pos | R2 layers above the shuffle max: write, read, query, key, neg, pos, cplx |
|---|---|---|---|---|
| 0 | 0.009 | 0.008 | 0.012 | 0, 0, 0, 0, 0, 0, 0 |
| 128 | 0.009 | 0.008 | 0.010 | 1, 0, 0, 0, 0, 0, 0 |
| 256 | 0.017 | 0.032 | 0.012 | 10, 0, 8, 6, 0, 0, 3 |
| 512 | 0.213 | 0.238 | 0.024 | 5, 5, 11, 11, 0, 1, 10 |
| 1000 | 0.114 | 0.203 | 0.030 | 15, 5, 19, 22, 3, 1, 15 |
| 2000 | 0.179 | 0.098 | 0.037 | 21, 8, 22, 17, 6, 0, 21 |
| 4000 | 0.100 | 0.154 | 0.073 | 14, 4, 19, 18, 1, 0, 15 |
| 143000 | 0.208 | 0.090 | 0.288 | 7, 8, 5, 5, 0, 0, 6 |

R1b is tr(P_S P_S′) / 10; two Haar 10-dim spaces give 0.0098 (2000 draws; 99.9th percentile 0.015),
which steps 0 and 128 match. The shared write space meets the shared read and key spaces from 512
(about 20 times Haar at 512), and at 143000 the negative and positive spaces overlap (0.29). R2
counts, per layer pool of 16, P̄'s eigenvalues above the largest λ₁ of 2000 random 16-head sets of
the step's 384: step 0 reads none, sharing within layers beyond the step's global sharing rises
from 256 to 2000–4000 (up to 22 of 24 layers in QK and `ov_cplx`) and falls back by 143000 (5–8).
Not established: anything these overlaps do (S2's arms, S5); R2 is description only (S1's preflight
refused it as a verdict for QK).

## S2 unit 2, the gates, R1 and R2 (2026-10-21)

Blocked 87 decided (a), the user's: the design frozen as revised (the one-sided guard; R2 under arm
a description). Two changes before any arm was run, both the user's: **G2's null** (the frozen
doubly matched one was infeasible for a zeroing: every control equalled the arm) became zeroing 10
non-members, each from its member's layer; and **Blocked 88 (a)**, after the gates: the
displacement guard as description (`design-S2.md` "Unit 2: fixed in code"). Tier 1, one rung.

**Run.** From `../Mets-work`, `HF_HOME` the main tree's `data/hf`, under `systemd-inhibit`; RTX 3080,
float32, TF32 off; transformers 4.44.2. Records `data/trackS/s2/2026-10-21-u2/` (main tree):

| job | code (file sha) | records (sha256, 12) | time |
|---|---|---|---|
| `s2.py gates` (the `runner` agent) | `13e7454e0d1a` (`e5a5b98`) | `gates_{0,256,512,1000,2000,4000,143000}.json` `a33b70ee784d`, `c731487dba9f`, `2eaabe00e9a8`, `5d075fe8fab7`, `fb767b05a802`, `355ad8011730`, `1842ddd4bc62`: 85, 85, then 292 configurations each | 1,752 s |
| `s2.py gate-read` | `5ffab7b163ab` (`455050c`; the readers only differ from `e5a5b98`) | `gate_read.json` `4e03ca7c7a92` | — |
| `s2.py arms` | `5ffab7b163ab` | `arms_{step}.json` (base re-run equal to the gate pass's at every step) | 63 s |
| `s2.py read` | `5ffab7b163ab`; re-run at `37a2d6a9046d` after `/challenge-pr` on #248 (the gates' refusals enforced; every label the same) | `readings.json` `da8567dcac9c`, then `c5ecf6863def` | — |
| read-back at 143000 (`/challenge-pr` on #248, question 3; a scratch script calling `s2.readbacks`) | `b87b9405b7f2` | `readback_143000.json`: G0's read-backs on 20 heads pass (worst 4.7e-8 of ‖K‖); arm a's read-back equals sym K on all 384 heads to 4.3e-8 | — |

After the runs, one comment in `s2.py` (a threshold's provenance label) moved its sha to
`b87b9405b7f2`; no line that computes changed. After CodeRabbit on #248, G3′'s base check
(`check_g3p_base`, D ≤ `G3P_BASE_TOL`) runs on the stored record, so a resumed gates pass and
`gate-read` refuse too (before, a resume skipped it); `abf4db781a29`. The five recorded `base_D`
are 4.8e-7 to 1.6e-6; `gate-read` re-run at `abf4db781a29` on a scratch copy of the gate records
equals `gate_read.json` but for its code field. Base V at the 7 steps: 0, 19, 56, 56, 61, 57, 70 (the panel's). Base G: ≈ 0 at 0–512, 8.21 at
1000, 11.89, 12.44, 12.13 after.

**The gates (`gate_read.json`; under the frozen guard beside).**

| gate | reading | frozen guard | revised (88 (a)) |
|---|---|---|---|
| G1, a 41st control read as the arm | 0 of 10 R1 cells, 0 of 8 R2 cells labelled | pass | pass |
| G2, the 10 members' OV zeroed, G (controls: 10 other heads of the same layers) | 1000: 1.06 against [7.68, 8.30]; 2000: 1.62 against [11.19, 12.02]; 4000: 1.90 against [11.74, 12.53]; 143000: 2.57 against [11.08, 12.55]. D further than every control at 1000–4000 | "lost" 1 of 4: **fails** | 4 of 4: pass |
| G3, the model toward step 128, V at t = 1 (controls: Gaussian, same per-tensor norm) | 0 at every step, against [53, 59], [47, 60], [38, 61], [19, 40], [0, 11]; D further at 512–2000, less at 4000 and 143000 | 1 of 5 (4000): **fails** | 4 of 5 (512–4000; 143000 as its controls, median 0): pass |
| G3′, every head's K set to its step-128 K, V | 29, 31, 56, 44, 61 at 512 … 143000 (controls [50, 66], [38, 58], [33, 51], [36, 52], [54, 66]; gain arm 54, 46, 33, 32, 68) | never ≤ 7 | never ≤ 7: **the one OV-only edit with a known destination does not reach R1's positive answer**; it halves V at 512 and 1000, beyond its controls and gain arm |

G3 at t = 0.25: *lower* at 512–4000 (V 36–45); at 0.5: *lower* at 512 and 1000 (V 36, 26), as its
controls at 2000 and 4000 (V 57); both *higher* at 143000 (V 36–45 against Gaussian controls at 2–31: random edits of that size already
make E₁ nearly monotone at the end). G3′ at 512 and 1000 is *lower* beyond its gain arm (V 29, 31),
moved further: step 128's OV maps alone halve the violations there, without reaching 7.

**G1′, the gain arms against the controls (description).** g_b is as its controls everywhere. g_a
(each head scaled to ‖sym K‖, ≈ 0.71) is *higher* in V at 2000, 4000, 143000 (V 50, 46, 63 against
medians 35, 33, 37) and far *higher* in G from 1000 (7.5–11.3 against 0.4–4.5): arm a's random
edits break copying, the gain change alone barely does.

**R1, V (no cell counts).**

| step | arm a: V; controls [min, median, max]; gain; D (controls) → label | arm b: same |
|---|---|---|
| 0 (beside) | 11; [0, 6, 17]; 9; 0.480 (0.469–0.503) → as its controls | 0; [0, 0, 2]; 0; 0.095 → as its controls |
| 256 (beside) | 30; [9, 11, 17]; 15; 0.343 (0.306–0.321) → higher, moved further | 18; [17, 19, 23]; 18; 0.050 → as its controls |
| 512 | 60; [45, 49, 55]; 50; 0.406 (0.332–0.348) → **higher**, moved further | 54; [54, 55, 58]; 54; 0.050 (0.045–0.049) → as its controls, moved further |
| 1000 | 41; [40, 46, 54]; 49; 0.519 → as its controls, moved further | 56; [54, 57, 60]; 57 → as its controls, moved further |
| 2000 | 29; [29, 35, 41]; 50; 0.628 → as its controls (a tie at the minimum), moved further | 61; [57, 60, 62]; 60 → as its controls, moved further |
| 4000 | 36; [27, 33, 38]; 46; 0.697 → as its controls, moved further | 54; [52, 56, 60]; 57 → as its controls, moved further |
| 143000 | 65; [23, 37, 51]; 63; **11.93** (1.03–1.14) → **higher**, moved further | 61; [56, 65, 70]; 70; 0.316 → as its controls |

**R1: "not shown" under both arms** (0 of 5 each). Under arm a the symmetric part never lowers V
beyond its controls at a scored step and raises it at 512 and 143000 (both moved further); at 8000
(the preflight) it was below them (21 against 30–39), so the sign varies along training. Under arm
b removing rotation in the Schur planes (≈ 0.1 ‖K‖) moves V no more than random edits of its size.
With G3′ never at V ≤ 7, "not shown" cannot be told from "out of an OV edit's reach" (G3′ is one
edit, so that reach is not established either).

**R2, G (read under arm b only).**

| step | arm b: G (base); controls; gain → label | arm a (description): G; controls; gain |
|---|---|---|
| 1000 | 8.19 (8.21); [8.16, 8.22]; 8.22 → **survives**, moved further | 4.01; [4.38, 4.62]; 7.47 (below every control, beyond its gain) |
| 2000 | 11.87 (11.89); [11.81, 11.91]; 11.89 → **survives**, moved further | 3.19; [3.68, 4.20]; 9.50 (likewise) |
| 4000 | 12.41 (12.44); [12.36, 12.45]; 12.43 → **survives**, moved further | 2.33; [2.84, 3.40]; 10.35 (likewise) |
| 143000 | 12.34 (12.13); [11.58, 12.52]; 12.12 → **survives** | 0.05; [−0.11, 0.74]; 11.33 (as its controls) |

**R2 under b: copying survives the rotation's removal at 4 of 4 steps** (also under the frozen
guard: "survives" was never withheld). At 0–512 G ≈ 0 (no copying yet), so the half-of-base rule
is degenerate there (beside). Under a (description), the symmetric part takes copying below every
random edit of its size at 1000–4000.

**(b-eig), beside, no label:** V 0, 17, 59, 54, 49, 43, 61; G 0.0, −0.07, −0.12, 2.41, 1.95, 0.98,
0.89; D 0.63–1.53 (it edits each head by 0.48–1.74 ‖K‖): λ → Re λ in the eigenbasis breaks most copying.

**Not established.** That the break is not OV's antisymmetric part or rotation: the one OV edit
tried with a known destination (G3′) does not reach the monotone regime, so R1 was unlikely to
answer yes. That no OV-only edit can. That copying needs
no rotation in general: (b-schur) edits only the diagonal blocks' skew parts, about 1 % of ‖K‖²
(unit 1), and 88 % of a head's map reads outside its write space and is kept. Any reading under arm
a beyond description: its controls are damaged models (G 0.4–4.6). The "moved further" cells
(most of them) cannot be told from moving further (OV1's hazard, kept as description by 88 (a)).

**Parked.**
- Arm a's D at 143000 is 11.9 against its controls' 1.0–1.1 (at 4000, 0.70 against 0.66): symmetrising
  the end-state heads moves the stream an order of magnitude further than random edits of the same
  size. Not the hook: arm a's read-back at 143000 equals sym K on every head (4.3e-8). Why: likely
  the massive-activation coordinates S1 found at 143000. Cost: one pass of D per
  coordinate on the records' configuration (minutes). Could change: whether R1's 143000 *higher*
  under a is about energy or about those coordinates. *Carried to S3* (with the massive
  coordinates, S1's Parked item).

**After `/challenge-pr` on #248** (accept with changes): (1) Blocked 88 changed no unit-2 label (no
R1 cell is raw *lower*; every scored R2 cell under b survives); only the gates' pass flags moved,
so under the frozen rules R1 and R2 would have been refused on the same numbers. Its effect
lands on R3, where both sides are structured edits: R3's guard is Blocked 89, before unit 3. (2)
`read` now enforces G1 and G3, not only G2 (`s2.gate_refusals`, tested), re-run with the same
labels. (3) The design's guard text marked superseded where 88 replaced it. (4) "Out of reach of an
OV-only edit" narrowed to the one edit tried, its halving at 512 and 1000 stated.

**Re-run.** `s2.py gates --out <dir>` (≈ 30 min), `gate-read`, `arms` (refuses without
`gate_read.json`), `read`, each `--out` the same directory.

## S2 unit 3, R3 on S₋ₗ and S1's R3 beside (2026-10-22)

Blocked 89 decided (a), the user's: R3 keeps the one-sided guard (`design-S2.md` "Unit 3: fixed in
code", written before any R3 output). Tier 1, one rung.

**Run.** From `../Mets-work` (the `runner` agent), env as unit 2; code `e62e8518cd16` (`45da0c5`).
Records `data/trackS/s2/2026-10-22-u3/` (main tree): `plan.json` `b5122ae16a40`, `r3_{512,1000,2000,4000,143000}.json`
`c9f2fd0bc252`, `9a350416d8ff`, `207796542514`, `c4138c274501`, `613d6b8c07ff` (44 configurations
each: base, bS, g_bS, bR 0–40), `r3_readings.json` `f438fc7c6fb2`, `s1_r3_loo.json` `56a091701b94`.
`r3` 249 s, `s1-r3-loo` 210 s. Base V and G equal unit 2's at every step.

**The selection did not select, and its null was the wrong one.** The Haar threshold (a random
plane's 99th-percentile share in a 10-dim space of R¹⁰²⁴) is 0.0183 (mean 0.0098). The median share
of *every* pair is 0.016, 0.020, 0.025, 0.020, 0.020 at 512 … 143000. But a pair's plane lies in
its head's 64-dim write space, which S1 found overlaps the shared space: a random plane *inside
each head's write space* has median expected share tr(P_S P_B) / 64 of 0.018, 0.031, 0.024 at 512,
2000, 143000, at or above the pairs' own (scratch check after `/challenge-pr` on #249, finding 1).
So the over-selection is the write space's overlap, not pairs leaning into S. "In S" took most of
the model's rotation:

| step | pairs in S / all | heads with any | bR shortfall (pairs, the same in every draw) | edit, median over heads (‖K‖): bS; bR (median draw) | D: bS; bR [min, max] |
|---|---|---|---|---|---|
| 512 | 3,966 / 10,819 (37 %) | 367 | 577 | 0.075; 0.063 | 0.025; [0.019, 0.021] |
| 1000 | 6,237 / 10,645 (59 %) | 377 | 3,697 | 0.075; 0.039 | 0.043; [0.024, 0.025] |
| 2000 | 7,225 / 10,450 (69 %) | 372 | 5,609 | 0.070; 0.022 | 0.043; [0.025, 0.028] |
| 4000 | 5,871 / 10,485 (56 %) | 382 | 3,180 | 0.068; 0.039 | 0.063; [0.030, 0.032] |
| 143000 | 5,978 / 10,390 (58 %) | 380 | 3,013 | 0.077; 0.045 | 0.267; [0.119, 0.125] |

A head with more pairs in S than out cannot give bR its count, so bR edits fewer pairs, by less,
and bS moves the stream further than every bR at every step. **The comparison is unmatched**: it
reads "S's pairs" against fewer, smaller edits, not against as many random ones. On a short head
every draw takes all its other pairs, so the 40 draws are partly one edit: they give 2, 3, 4, 3, 7
distinct V values (35 of 41 at V 61 at 2000). And with 37–69 % of pairs in it, bS is nearly unit
2's arm b on every pair (V 54 against 54 at 512, 60 against 61 at 2000): R3 mostly re-ran that edit.

**R3 (`r3_readings.json`): refused as frozen (unmatched) in V and G.** bR is short at every step,
and `r3_cells` now refuses such a step (after `/challenge-pr` on #249, finding 2: refuse rather
than degrade); the labels below are description. R3's G1 (a 41st bR read as bS): 1 of 9 labelled
(G at 2000). The gain arm g_bS is itself *higher* in V at 512, 4000 and 143000 (it stays at base
while every bR lowers V slightly), *as its controls* elsewhere and in every G cell.

| cell | bS (base); bR [min, median, max]; gain → label |
|---|---|
| V 512 | 54 (56); [54, 55, 55]; 56 → as its controls, moved further |
| V 1000 | 57 (56); [56, 56, 58]; 56 → as its controls, moved further |
| V 2000 | 60 (61); [60, 61, 63]; 60 → as its controls, moved further |
| V 4000 | 53 (57); [53, 54, 55]; 57 → as its controls (a tie at the minimum), moved further |
| V 143000 | 67 (70); [61, 66, 69]; 70 → as its controls, moved further |
| G 1000 | 8.199 (8.214); [8.203, 8.211, 8.217]; 8.211 → **lower, moved further: not read** (89 (a)) |
| G 2000 | 11.891 (11.894); [11.882, 11.887, 11.897]; 11.894 → as its controls, moved further |
| G 4000 | 12.416 (12.437); [12.432, 12.441, 12.452]; 12.433 → **lower, moved further: not read** (89 (a)) |
| G 143000 | 12.311 (12.126); [12.060, 12.135, 12.187]; 12.119 → higher, moved further |
| G 512 (beside) | −0.003 (−0.003) → as its controls, moved further |

**As description: no cell reads *lower* in V; in G two read *lower, moved further*.** Under the
frozen rule as first run (before the refusal) the answer was "not shown" (0 of 5, 0 of 4); under
88's rule (option (b)) the two G cells would read, 2 of 4, still short of 3: Blocked 89's choice
changed no answer. The G
differences from base are 0.003–0.19 on 8–12. In V, editing 1.2–3.2 times the random pairs' size inside S moves
the violations no more than the random pairs do. With R1 under arm b (every pair) as its controls
at every step and copying surviving it (unit 2), there was nothing in V or G for S's pairs to
carry: Q3 was conditional on an effect Q1 and Q2 did not show.

**S1's R3 on S₋ₗ (beside, description; `s1_r3_loo.json`, S1's own Haar draws).** From 512 the
leave-the-layer-out S reads as S1's S: `share` 0 in every OV family at 512–2000, layers outside
Haar within 0–2 of S1's count at 512–2000 and 143000 (`arg` within 0–3), the end state's shares
as before (0.52–0.74). At 4000 `ov_pos`'s share median falls (0.118 → 0.041; outside 12 → 11).
Before 512 the bias was real: `ov_neg`'s layers outside on `share` fall from 5, 7, 11 to 2, 1, 0
at 0, 128, 256, `ov_pos`'s from 3, 4, 3 to 0, 0, 2. So **the OV families' R3 readings from 512 on
carry no selection bias that moves them**; S1's "reads nothing of S's own from 512" stands for OV
as for QK; its early `ov_neg` / `ov_pos` readings were the bias (Corrections received, above).

**Not established.** Whether the most shared pairs carry anything at a matched selection (each
head's top pairs by share against as many of its others, matched in size): the frozen rule could
not ask it (unit 4 asked it: as its controls). That the heads' planes lean into S: against a random
plane in each head's own write space the median pair does not (above); in the tail 3.4–5.0 % do
("S2 unit 4"). Which pairs are "in S" also depends on the Schur form's block order:
only the first block's plane is invariant, so a pair's plane, and its share, is the order's
(finding 5; it would bind a top-m selection too).

**Parked.**
- A matched R3 (fixed top-m pairs per head by share against a random plane in the head's write
  space; bR from the same head's rest, matched in edit size). Why: the frozen selection took
  37–69 % of pairs. Cost: one unit (the pass is 4 min). Could change: R3 from refused to read.
  Against it: arm b on every pair moves nothing in V and leaves copying (unit 2), so a subset has
  little to carry; the Schur order binds its selection. *Blocked 90, decided (b), the user's:
  run as unit 4 ("S2 unit 4").*

**After `/challenge-pr` on #249** (accept with changes; findings 1–5 taken): (1) the threshold's
null was the wrong one, not only unchecked (the write-space check above); (2) R3 refused as frozen
(unmatched), code `63e6a57839e5`, `r3-read` re-run (`r3_readings.json` `bcc3baf3ebaa`; every label
the same, now description); (3) the shortfall is the same in every draw and the draws are partly
one edit; (4) the gain arm's own label stated; (5) the Schur order's hold on the selection.

**Re-run.** `s2.py r3 --out <dir>` (≈ 4 min), `r3-read`, `s1-r3-loo` (≈ 4 min, weights only).


## S2 unit 4, the matched R3 (2026-10-22)

Blocked 90 decided (b), the user's: one more unit, a matched R3 **as description**, then the
close-out (`design-S2.md` "Unit 4", written with the selection's numbers before any V or G output,
`e02cb61`). Tier 1, one rung.

**Run.** From `../Mets-work`, env as unit 2; code `1345c1e2b1d3` (`e02cb61`). Records
`data/trackS/s2/2026-10-22-u4/` (main tree): `select_{step}.json` (weights only, CPU; shas in the
design), `r3m_{512,1000,2000,4000,143000}.json` `abea6302be26`, `b8f84463c505`, `919c08b03dee`,
`36c0db327523`, `7c81d9d9a626` (44 configurations each: base, bS, g_bS, bR 0–40),
`r3m_readings.json` `e18cb36d47a6`. `r3m` 309 s (the `runner` agent). Base V and G equal unit 2's
at every step.

**The match held.** bS takes each head's 2 pairs with the largest share in S₋ₗ (768 pairs, 7.1–7.4 %
of all); each bR takes 2 of the head's others within 10 % of bS's edit. Per draw, bR's edit over
bS's has a median of 0.98–1.00 at every step. At most 15 of 384 heads are unmatched (nearest
subset, the same in every draw). The edit sizes (median over heads, ‖K‖) are bS 0.024–0.029 and bR
0.024–0.028, about a third of unit 3's bS. R3's G1 (bR's 41st draw against draws 0–39): 0 of 9
labelled. The gain arm g_bS is *as its controls* in every cell.

| cell | bS (base); bR [min, median, max] → label (description) |
|---|---|
| V 512 | 55 (56); [54, 55, 56] → as its controls |
| V 1000 | 57 (56); [56, 56.5, 58] → as its controls, moved further |
| V 2000 | 62 (61); [60, 60, 62] → as its controls |
| V 4000 | 55 (57); [55, 57, 57] → as its controls (a tie at the minimum), moved further |
| V 143000 | 73 (70); [68, 70, 73] → as its controls (a tie at the maximum), moved further |
| G 1000 | 8.207 (8.214); [8.206, 8.214, 8.219] → as its controls, moved further |
| G 2000 | 11.892 (11.894); [11.887, 11.894, 11.903] → as its controls |
| G 4000 | 12.445 (12.437); [12.425, 12.441, 12.448] → as its controls, moved further |
| G 143000 | 12.171 (12.126); [12.094, 12.159, 12.227] → as its controls, moved further |
| G 512 (beside) | −0.0030 (−0.0032) → as its controls |

**As description: neither V nor G moves beyond the random pairs at any step** (0 of 5 read *lower*
in V, 0 of 4 in G; the frozen rule's answer would be "not shown" in both). With the selection and
the edit size matched, the pairs that lie most in S1's shared complex space carry no more of the
energy break or of copying than as much rotation removed elsewhere in the same head. Given unit 2
(arm b on every pair as its controls in V, copying intact), this is the expected null.

**Beside, the displacement.** At 1000, 4000 and 143000 bS moves the stream further than every bR
at the same edit size (D 0.019 against 0.016–0.017; 0.027 against 0.023–0.024; 0.142 against
0.085–0.095, about 1.5×). At 512 and 2000 it moves as far as the bR draws do. bR is matched on the
weight edit, but V and G read the stream. Two explanations, neither tested: the edits in S add
coherently across heads (S is shared), or, at 143000, S partly overlaps the massive-activation
coordinates (S1, "S1 on one page"), so an edit of the same size acts on more input. Either way bS
was the larger edit *in effect* and still read as its controls, which makes the null stronger
there, not weaker (`/challenge-pr` on #250, finding 2).

**The selection, against the right null** (weights only; `design-S2.md` "Unit 4"). Against each
head's own write space (the null unit 3's review named), 3.4–5.0 % of all pairs lie above their
head's 99th percentile of a random plane's share, where 1 % would if the planes were random (they
are Schur planes, so this is description). 41–58 % of bS's pairs pass. So a minority of planes
lean into S, and bS holds 84–94 % of all the pairs that pass. But only about half of bS's pairs
pass, and 102–163 of 384 heads have no passing pair at all. In those heads bS against bR compares
one chance-level pair with another, so the null there holds by construction. **The null reads on a
selection that is about half filler** (`/challenge-pr` on #250, finding 1). bR's pool keeps only
23–83 passing pairs over the whole model.

**Not established.** Whether a selection by the write-space test itself (a variable count per
head) would read otherwise. Whether m other than 2 would; m = 2 was fixed before output, m = 1 is
purer (58–74 % pass) with a smaller edit. Whether the extra displacement is coherence across heads.
Every limit of unit 3 on the Schur order holds here.

**Parked.** None new. The extra displacement goes to the close-out as a description line. Why:
it is the one thing in which bS differs from bR. Cost: none. It could change nothing in R3.

**After `/challenge-pr` on #250** (accept with changes; findings 1–3 taken): (1) the selection is
about half filler, and the null holds by construction in the 102–163 heads with no passing pair
(above); (2) the extra displacement is about 1.5× at 143000, with the massive coordinates named as
the other explanation; (3) m's choice has a tracked producer, `r3m-msweep` (`msweep.json`
`07f4d745e930`), which also shows m = 1 would be refused (43–63 heads unmatched).
`R3M_MAX_UNMATCHED` is labelled placed. `r3m` now refuses a selection whose content sha is not the
design's (`R3M_SEL_SHA`). Code after the run: `08a7d72d9d64`. No line that computes `r3m`'s or
`r3m-read`'s numbers changed.

**Re-run.** `s2.py r3m-select --out <dir>` (weights only, ≈ 3 min CPU), `r3m` (≈ 5 min GPU),
`r3m-read`; `r3m-msweep` (≈ 8 min CPU) for m's choice.
