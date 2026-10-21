<!-- trackS_subspace/status-S.md -->
# Track S — STATUS

<!-- phase-card -->
## Card

- **Question:** Do the 384 heads of `pythia-410m` share directions of the residual stream, per subspace family (OV write / read, QK query / key, OV's negative / positive / complex invariant subspaces), beyond a random arrangement of the same subspaces, how does that change over training, and what do the attention layers do inside the shared space (push, rotate, collapse)?
- **Inputs:** `pythia-410m` weights at steps 0, 128, 256, 512, 1000, 2000, 4000, 143000 (and 8000 off the grid for the preflight and the gate), no tokens; code `trackS_subspace/s1.py`; records `data/trackS/s1/` (main tree) — `trackS_subspace/design-S1.md` "Objects, fixed here"
- **Results:**
  - Opened 2026-10-20 with a scan, a `sympy` check and a design frozen as written (3 units + close-out); the rotation null, the norm-matched random draw and step 0 are one law for the mean projector, and the preflight refused per-unit verdicts against it, so S1 was designed to read counts — `trackS_subspace/design-S1.md` "Preflight (`tools/design_preflight.py`, step 8000, off the grid)"
  - Unit 1's gate: the instrument and its null pass (G0, G1, G1iii, G2: the three nulls are one law, step 0 reads no shared direction, the rotation null reads none at any step); the count fails its planted effect on the real pool in all 7 families, because at a trained step the mean projector's spectrum is dense at the null's edge, where each planted direction raises the count only about half the time; N refused by the frozen fallback (Blocked 82); the heads use nearly every dimension but unevenly (participation ratio 344–729 against 880), which alone explains a count near the edge, not the top eigenvalues — `trackS_subspace/status-S.md` "S1 unit 1, the gate (2026-10-20)"
- **Superseded / wrong:** none
- **Registry:** none, because Track S is tier 1 on an exploration rung (`trackS_subspace/design-S1.md` "Fences")
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - Which statistic says "how much is shared" on a trained pool, now that the count is refused: the spectrum only, the participation ratio with the top eigenvalues against a dimension-matched null, or S1 closed at the gate (Blocked 82) — `trackS_subspace/status-S.md` "S1 unit 1, the gate (2026-10-20)"
- **After Phase 10:**
  - The weighted projectors (singular values as weights), where the rotation null keeps each head's spectrum and differs from Haar *(free, weights only)*
- **Reviewed:** 2026-10-20 · body `97f7fe7f12`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| 0. scan, design, `sympy` check, preflight | **done 2026-10-20**; frozen as written (Blocked 81 (a), the user's) | `lit-S1.md`, `design-S1.md`, `tools/math_checks/s1_projector_null.py`, `data/trackS/s1/2026-10-20-preflight/` |
| 1. the instrument and its gate | **done 2026-10-20: G0–G2 pass, G3 fails in all 7 families; N refused** (Blocked 82) | "S1 unit 1" below; `s1.py gate` |
| 2. R1, R1b, R2 over the 8 steps | waits on Blocked 82 | — |
| 3. R3 and the residual check; close-out | waits on 2 | — |

## Corrections received

(none)

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
