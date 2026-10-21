<!-- trackS_subspace/status-S.md -->
# Track S — STATUS

<!-- phase-card -->
## Card

- **Question:** Do the 384 heads of `pythia-410m` share directions of the residual stream, per subspace family (OV write / read, QK query / key, OV's negative / positive / complex invariant subspaces), beyond a random arrangement of the same subspaces, how does that change over training, and what do the attention layers do inside the shared space (push, rotate, collapse)?
- **Inputs:** `pythia-410m` weights at steps 0, 128, 256, 512, 1000, 2000, 4000, 143000 (and 8000 off the grid for the preflight and the gate); tokens only for unit 3's residual check (v1's 7 scored prompts, `trackB_circuit/b0_gate.py` `V1_SCORED`); code `trackS_subspace/s1.py`; records `data/trackS/s1/` (main tree) — `trackS_subspace/design-S1.md` "Objects, fixed here"
- **Results:**
  - Opened 2026-10-20 with a scan, a `sympy` check and a design frozen as written (3 units + close-out); the rotation null, the norm-matched random draw and step 0 are one law for the mean projector, and the preflight refused per-unit verdicts against it, so S1 was designed to read counts — `trackS_subspace/design-S1.md` "Preflight (`tools/design_preflight.py`, step 8000, off the grid)"
  - Unit 1's gate: the instrument and its null pass (G0, G1, G1iii, G2: the three nulls are one law, step 0 reads no shared direction, the rotation null reads none at any step); the count fails its planted effect on the real pool in all 7 families, because at a trained step the mean projector's spectrum is dense at the null's edge, where each planted direction raises the count only about half the time; N refused by the frozen fallback (Blocked 82); the heads use nearly every dimension but unevenly (participation ratio 344–729 against 880), which alone explains a count near the edge, not the top eigenvalues — `trackS_subspace/status-S.md` "S1 unit 1, the gate (2026-10-20)"
  - Unit 2 (Blocked 82 (b), the user's): R1′ = the participation ratio against its exact Haar value; the null side passes (G0′–G2′); the additive plant passes in the 5 OV families and fails in QK, both by the plant's added dimensions diluting the pool's own concentration (math check fact 6), and with that removed it cannot see the pool, so the effect side is uncalibrated and QK's ρ is Blocked 83. Over training the OV families concentrate from 512 (`ov_write` and `ov_cplx` already from 256: corrected by `/challenge-pr` on #245), most at 2000 (ρ 0.59 `ov_write`), partly relax at 4000 (just after the LR warmup ends), and end at ρ 0.65–0.83; LN1's gain changes nothing; the spectrum is uneven from 512 (the top's null cannot tell a few directions from a graded spectrum) — `trackS_subspace/status-S.md` "S1 unit 2, R1′ and its re-gate (2026-10-21)"
  - Unit 3 (Blocked 83–85 (a), the user's): QK's heads leave the null at 256 with `ov_write` and `ov_cplx`, keep concentrating through 4000, and are the most concentrated of the 7 from 2000 on (`ov_write` is at 512–1000); inside each family's 10 most shared directions the radial sign reads S at no step from 512 (inside random subspaces' range at 512, and at 1000–4000 random subspaces hold no attractive part in 18–20 of 24 layers, so only "less repulsive" can show); the OV families' S is drawn from the heads it compresses, a bias of unknown size, so QK's readings are the clean ones; collapsible negative pairs are rarer than random in QK's S at 1000–4000; no shared space is the residual stream's top variance, and at 143000 the write, negative and complex ones are partly the massive-activation coordinates — `trackS_subspace/status-S.md` "S1 unit 3, QK's ρ, R3 and the residual check (2026-10-21)"
  - S1 closed 2026-10-21 at its revised exit (3 units + close-out): the shared spaces are described against an exact null, never counted or calibrated on the effect side — `trackS_subspace/status-S.md` "S1 on one page (closed 2026-10-21)"
  - S2 unit 1 (Blocked 86 (a) and (i), the user's: arm b in Schur planes, 3 units): the hook is the model at its base core and every arm's read-back is exact; at 8000, off the grid, both arms move the residual stream further than all 40 of their size-matched random edits, so the frozen displacement guard would leave their cells unread; arm a's random edits alone break most in-context copying; removing rotation in the Schur planes changes about a tenth of each head's map; the shared write space meets the shared read and key spaces from 512; the freeze is Blocked 87 — `trackS_subspace/status-S.md` "S2 unit 1"
  - S2 opened 2026-10-21 (Blocked 86 since decided (a) and (i)): every head's OV edited at once, (a) to its symmetric part and (b) with its rotation removed (recommended: in an orthonormal basis of each invariant plane, since λ → Re λ in the eigenbasis edits a head by 1.6× its size at 512), each against 40 random edits matched in size and in the edited head's norm, and read only beyond the gain change alone; reading the energy break, in-context copying and S1's shared rotation; no theorem predicts "monotone" for (a) — `trackS_subspace/design-S2.md`
- **Superseded / wrong:** none
- **Registry:** none, because Track S is tier 1 on an exploration rung (`trackS_subspace/design-S1.md` "Fences")
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - A leave-the-layer-out S for OV's R3 (S2 unit 3; R1b and R2 done in S2 unit 1, description; `design-S2.md` "Carried in from S1"); S2's freeze (Blocked 87); the massive coordinates in the end-state shared spaces (carried to S3), 2603.13314's full text (carried to C2's draft) — `trackS_subspace/status-S.md` "S1 on one page (closed 2026-10-21)"
- **After Phase 10:**
  - The weighted projectors (singular values as weights), where the rotation null keeps each head's spectrum and differs from Haar *(free, weights only)*
- **Reviewed:** 2026-10-21 · body `e2cc49f7a3`
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
| S2 2. G1–G3 at the design points; R1, R2 | not started | `design-S2.md` "The gate", "Readings" |
| S2 3. R3 on S₋ₗ; S1's R3 on S₋ₗ beside | not started | `design-S2.md` "The arms" |
| S2 close-out | not started | — |

## Corrections received

(none)

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
