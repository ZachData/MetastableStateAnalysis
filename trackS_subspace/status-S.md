<!-- trackS_subspace/status-S.md -->
# Track S — STATUS

<!-- phase-card -->
## Card

- **Question:** Do the 384 heads of `pythia-410m` share directions of the residual stream, per subspace family (OV write / read, QK query / key, OV's negative / positive / complex invariant subspaces), beyond a random arrangement of the same subspaces, how does that change over training, and what do the attention layers do inside the shared space (push, rotate, collapse)?
- **Inputs:** `pythia-410m` weights at steps 0, 128, 256, 512, 1000, 2000, 4000, 143000 (and 8000 off the grid for the preflight and the gate), no tokens; code `trackS_subspace/s1.py`; records `data/trackS/s1/` (main tree) — `trackS_subspace/design-S1.md` "Objects, fixed here"
- **Results:**
  - Opened 2026-10-20 with a scan, a `sympy` check and a design frozen as written (3 units + close-out); the rotation null, the norm-matched random draw and step 0 are one law for the mean projector, and the preflight refused per-unit verdicts against it, so S1 was designed to read counts — `trackS_subspace/design-S1.md` "Preflight (`tools/design_preflight.py`, step 8000, off the grid)"
  - Unit 1's gate: the instrument and its null pass (G0, G1, G1iii, G2: the three nulls are one law, step 0 reads no shared direction, the rotation null reads none at any step); the count fails its planted effect on the real pool in all 7 families, because at a trained step the mean projector's spectrum is a continuum through the null's edge, so a count above the edge is not a number of shared directions; N refused by the frozen fallback (Blocked 82) — `trackS_subspace/status-S.md` "S1 unit 1, the gate (2026-10-20)"
- **Superseded / wrong:** none
- **Registry:** none, because Track S is tier 1 on an exploration rung (`trackS_subspace/design-S1.md` "Fences")
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - Which statistic says "how much is shared" on a trained pool, now that the count is refused: the spectrum only, a continuous excess over the null's edge, or S1 closed at the gate (Blocked 82) — `trackS_subspace/status-S.md` "S1 unit 1, the gate (2026-10-20)"
- **After Phase 10:**
  - The weighted projectors (singular values as weights), where the rotation null keeps each head's spectrum and differs from Haar *(free, weights only)*
- **Reviewed:** 2026-10-20 · body `caba301948`
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

**Why it fails (diagnosed on 8000 after the arm, off the grid).** At a trained step P̄'s
eigenvalues run continuously through the threshold (`ov_write`: 0.0317, 0.0316, … 0.0312 | 0.0311,
0.0310, …; spacing ≈ 1e-4). The plant replaces each chosen head's 16th direction, which moves
eigenvalues near the edge by more than their spacing, so several fall below T while the 4 planted
rise above it. A plant that only adds a dimension (Loewner-monotone, so nothing can fall) still
raises N by 2–4, not 4: a random direction partly overlaps the existing shared space and is
absorbed. Above 1.5 × T the spectrum thins (`ov_write` 26, `ov_read` 15, `qk_query` 43, `qk_key`
33 eigenvalues; above 3 × T: 0, 0, 13, 10), but stays a sequence, not separated groups. The mass
above T (m) moves the right way under the frozen plant, by about half the planted mass (+0.010
to +0.013 against 0.021; 10 draws, k = 32, singular families).

**What unit 1 establishes.** The instrument is exact and its null is right: Haar, the rotation
null, a norm-matched Gaussian draw and Pythia's step 0 are one law (G1, G1iii, G2). **At a
trained step the heads' shared space is a continuum through the null's edge, not a countable
set of subspaces,** so "how many shared subspaces" has no answer by this count. The frozen
fallback refuses N in all 7 families and leaves R1 as spectra only.

**Not established.** Anything at the scored trained steps (no trained scored cell was read);
whether a continuous excess is calibratable (only m's 10-draw diagnostic); R3.

**Blocked 82 (yours).** (a) Go on under the frozen fallback: unit 2 draws each cell's spectrum
against the Haar band and step 0's, with m beside, no count; R3's S is the eigenvectors above T,
labelled a threshold crossing. (b) Revise R1 to a continuous statistic and re-gate in unit 2: the
excess over the null's edge, X = Σ_i (λ_i − T)₊, its planted recovery (ΔX against the planted
excess Σ(k/n − T)) gated on both carriers at a recovery fraction fixed before the re-gate, the
spectrum drawn, the count above 2 × T beside as description. (c) Close S1 at the gate (writing
only): the continuum is the finding, and S2 takes each family's top directions descriptively.
**Recommended: (b)**, within the 3-unit budget (unit 2 = re-gate + R1): it keeps S1's question
(how much is shared, and when) on a scale the gate can calibrate, and (a) has no calibrated
statistic at all.
