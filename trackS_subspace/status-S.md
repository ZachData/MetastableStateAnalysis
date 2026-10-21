<!-- trackS_subspace/status-S.md -->
# Track S — STATUS

<!-- phase-card -->
## Card

- **Question:** Do the 384 heads of `pythia-410m` share directions of the residual stream, per subspace family (OV write / read, QK query / key, OV's negative / positive / complex invariant subspaces), beyond a random arrangement of the same subspaces, how does that change over training, and what do the attention layers do inside the shared space (push, rotate, collapse)?
- **Inputs:** `pythia-410m` weights at steps 0, 128, 256, 512, 1000, 2000, 4000, 143000 (and 8000 off the grid for the preflight and the gate), no tokens; code `trackS_subspace/s1.py`; records `data/trackS/s1/` (main tree) — `trackS_subspace/design-S1.md` "Objects, fixed here"
- **Results:**
  - Opened 2026-10-20 with a scan, a `sympy` check and a design frozen as written (3 units + close-out); the rotation null, the norm-matched random draw and step 0 are one law for the mean projector, and the preflight refused per-unit verdicts against it, so S1 was designed to read counts — `trackS_subspace/design-S1.md` "Preflight (`tools/design_preflight.py`, step 8000, off the grid)"
  - Unit 1's gate: the instrument and its null pass (G0, G1, G1iii, G2: the three nulls are one law, step 0 reads no shared direction, the rotation null reads none at any step); the count fails its planted effect on the real pool in all 7 families, because at a trained step the mean projector's spectrum is dense at the null's edge, where each planted direction raises the count only about half the time; N refused by the frozen fallback (Blocked 82); the heads use nearly every dimension but unevenly (participation ratio 344–729 against 880), which alone explains a count near the edge, not the top eigenvalues — `trackS_subspace/status-S.md` "S1 unit 1, the gate (2026-10-20)"
  - Unit 2 (Blocked 82 (b), the user's): R1′ = the participation ratio against its exact Haar value; re-gated with an additive plant, it passes in the 5 OV families (resolution 12–24 of 384 heads) and fails in both QK families, where the added dimensions dilute the pool's own concentration (math check fact 6), so QK is spectrum only by the frozen fallback (Blocked 83). Over training the OV families read the null's value through step 256, concentrate from 512, most at 2000 (ρ 0.59 `ov_write`), partly relax at 4000, and end at ρ 0.65–0.83 at 143000; LN1's gain changes nothing; the top eigenvalue rises above a dimension-matched null from 512 (`ov_write` 3.3× at 143000) — `trackS_subspace/status-S.md` "S1 unit 2, R1′ and its re-gate (2026-10-21)"
- **Superseded / wrong:** none
- **Registry:** none, because Track S is tier 1 on an exploration rung (`trackS_subspace/design-S1.md` "Fences")
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - Whether the QK families get R1′: spectrum only as the fallback has it, a QK re-gate scored on the excess kernel at fixed dimensions, or QK left to S2 (Blocked 83) — `trackS_subspace/status-S.md` "S1 unit 2, R1′ and its re-gate (2026-10-21)"
  - What a family's shared space S is for R1b and R3, now that N is refused (Blocked 84) — `trackS_subspace/design-S1.md` "Blocked 82 decided (b)"
- **After Phase 10:**
  - The weighted projectors (singular values as weights), where the rotation null keeps each head's spectrum and differs from Haar *(free, weights only)*
- **Reviewed:** 2026-10-21 · body `4d5a4d3ef8`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| 0. scan, design, `sympy` check, preflight | **done 2026-10-20**; frozen as written (Blocked 81 (a), the user's) | `lit-S1.md`, `design-S1.md`, `tools/math_checks/s1_projector_null.py`, `data/trackS/s1/2026-10-20-preflight/` |
| 1. the instrument and its gate | **done 2026-10-20: G0–G2 pass, G3 fails in all 7 families; N refused** (Blocked 82) | "S1 unit 1" below; `s1.py gate` |
| 2. re-gate + R1′ over the 8 steps (Blocked 82 (b)) | **done 2026-10-21: G0′, G1iii′, G2′ pass; G4 passes in the 5 OV families, fails in QK (spectrum only, Blocked 83); R1′ read** | "S1 unit 2" below; `s1.py regate`, `s1.py r1` |
| 3. R1b, R2, R3 and the residual check; close-out | waits on Blocked 83, 84 | — |

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

## S1 unit 2, R1′ and its re-gate (2026-10-21)

Blocked 82 decided (b), the user's ("the planted test removed weight as it added it; the next
gate uses a plant that only adds"). Bars fixed in `design-S1.md` "Blocked 82 decided (b)" and
committed (`c9c8ecd`) before the code was written.

**Run.** `python -m trackS_subspace.s1 regate --out data/trackS/s1/2026-10-20-u2-regate`
(main tree's `data/`, `HF_HOME` its `data/hf`; run from `../Mets-work`), code `e0b82943e83e`
(commit `9de8523`), 7,441 s under `systemd-inhibit`; a first launch was killed after an 8 h suspend
(`log_killed_after_suspend.txt`; `LESSONS.md` 11, 2026-10-21). Then `s1.py r1 --gate <that dir>`
→ `data/trackS/s1/2026-10-21-u2-r1/r1.json` (sha256 `c54df0db84f8`), code `55c4859e42e9` (commit
`bd85227`: the per-step write and resume only), 2,849 s. RTX 3080 float64 for the Haar bands and
the top's null; CPU float64 for bases and Grams. Re-gate files (sha256, first 12): `bands.json`
`c947528c5218`, `G0p` `87ea22c44c1a`, `G1iii_p` `44e5d797d509`, `G2p` `8d952843df8a`, `G4`
`33669c326203`, `G4t` `461a841eb7ca`.

| arm | result | pass |
|---|---|---|
| G0′ | PR by Frobenius against eigenvalues ≤ 4.4e-16 relative; the 28 distinct bands' mean tr P̄² against the closed form: max \|z\| 2.96 | yes |
| G1iii′ | step 0's PR inside its band in all 7 (`ov_write` 879.5 in [878.9, 880.6]; `ov_neg` 204.9 in [204.0, 206.0]) | yes |
| G2′ | the rotation null inside the band in 56 of 56 | yes |
| G4 | additive plant: detected on Haar in 100 % at every dose, every family; on the real pool the size bar holds at k = 32 in the 5 OV families (k* `ov_write` 24, `ov_read` 16, `ov_neg`, `ov_pos`, `ov_cplx` 12) and **fails in `qk_query` and `qk_key` at every dose to 64** | **no** (QK) |
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

**R1′ (pool `all`, r = 16, raw frame). ρ = PR / PR₀** (1 at the null; QK spectrum only):

| family | 0 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 143000 | k* |
|---|---|---|---|---|---|---|---|---|---|
| `ov_write` | 1.000 | 0.999 | 0.994 | 0.889 | 0.662 | **0.586** | 0.735 | 0.653 | 24 |
| `ov_read` | 1.000 | 1.000 | 1.000 | 0.994 | 0.982 | 0.951 | 0.896 | 0.814 | 16 |
| `ov_neg` | 0.999 | 1.000 | 1.000 | 0.925 | 0.779 | **0.686** | 0.845 | 0.825 | 12 |
| `ov_pos` | 1.001 | 0.999 | 1.000 | 0.985 | 0.967 | 0.910 | 0.865 | 0.810 | 12 |
| `ov_cplx` | 1.000 | 1.000 | 0.997 | 0.935 | 0.761 | **0.675** | 0.822 | 0.793 | 12 |

PR itself (`ov_write` 879.5 → 515.5 at 2000 → 574.6) and the invariant families' PR₀ (which
moves with the heads' type mix: `ov_pos` 209.5 → 23.5 at 1000 → 383.6) are in `r1.json`. The
band's half-width in ρ is ≤ 0.0011 (singular, `ov_cplx`) and ≤ 0.0063 (`ov_neg`, `ov_pos`), so every move from 512 on
is far outside the null; ρ is exact on the weights (no sampling), so a step-to-step difference
has no noise to clear, only k*'s resolution as the size a plant must have to move it.

**Beside (ρ).** r = 8 and 32: the same shape, r = 8 more concentrated (`ov_write` 0.534 at 2000),
r = 32 less (0.669). LN1's gain folded in: equal to raw within 0.003 everywhere except at 143000 in `ov_neg`
(0.797 against 0.825) and `ov_pos` (0.801 against 0.810), so the gain does not create the sharing.

**The top, described (λᵢ over the largest λ₁ of Haar in R^m, m = round(PR); no count).** Below
1 at steps 0–128 in every family (the m-dim null is conservative: it packs the same dimensions
into fewer). λ₁ crosses it from 512 in `ov_write` (2.14; 3.06 at 2000, 3.26 at 143000), `ov_cplx`
(1.61; 2.45, 2.01), `ov_read` (1.13 → 2.05 at 143000) and `ov_neg` (1.04; 1.59, 1.33); `ov_pos`
only at 143000 (1.16). λ₁₀ is above it in `ov_write` and `ov_cplx` from 512 (1.16–1.54) and in `ov_read`
from 2000 (1.05–1.34). QK's λ₁ (spectrum only): `qk_query` 0.031 (Haar) → 0.347 at 4000 → 0.187 at 143000;
`qk_key` → 0.259 → 0.230.

**What unit 2 establishes (tier 1, one rung, weights only).** In the five OV families the heads'
subspaces are an independent random arrangement through step 256, as at init; from 512 they
concentrate on fewer directions of the residual stream, most at step 2000 in the write,
negative and complex families, partly relaxing by 4000; the read and positive families
concentrate steadily to 143000. The concentration is not only uneven dimension use: the top
eigenvalues exceed what Haar in the pool's effective dimension gives, from 512 on.

**Not established.** QK's ρ (refused by G4; its λ₁ is descriptive); that the 2000 dip is more than
one checkpoint's reading (no step between 1000 and 4000 but 2000); which directions are shared or
whether they are the residual stream's top variance (unit 3's beside check); any other rung.

**Blocked 83 (yours): QK's R1′.** (a) Keep QK at spectrum only, as the frozen fallback says.
(b) A QK re-gate scored on the excess kernel κ = tr P̄² − E₀ tr P̄², whose response to an added
direction has no dilution term (fact 6's −2εχ₀ comes from dividing by (tr P̄)²), same plant,
same 0.75 bar, fixed before it runs; if it passes, read QK's ρ (equivalent to κ at k_h = 16).
~20 min of GPU plus QK's 16 R1′ cells; from unit 3's budget. (c) Close QK in S1 at its spectrum;
S2 takes its top directions. **Recommended: (b)**: QK is the most concentrated family (PR 344,
412 at 8000), the failure is the plant's added dimensions, which the data never have, and the
score is fixed before the run. Against it: it changes the scored quantity after a failure,
which is what the gate exists to stop; the rule's text and the failing numbers stay on record.

**Blocked 84 (yours, before unit 3): S for R1b and R3** (N defined it; refused). (a) S = each
family's top 10 eigenvectors at each step (the described top's width), R3's Haar subspace of the
same dimension as its comparison, as designed. (b) S = the eigenvectors above the m-dim null's
largest λ₁ (a count, which (b) of Blocked 82 kept out). (c) Drop R1b; R3 reads only the top 1.
**Recommended: (a)**: fixed width, no count, and R3 keeps its designed null.
