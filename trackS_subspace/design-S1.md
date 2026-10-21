<!-- trackS_subspace/design-S1.md -->
# Track S — design: S1, the shared-subspace atlas

Opened 2026-10-20 by `docs/ROADMAP.md` §9 row 6 (A3 → A5 → the preflight tool → **S1** → S2 →
C2's draft). Row: `docs/ROADMAP.md` §7a S1. Scan: `trackS_subspace/lit-S1.md`. Math:
`tools/math_checks/s1_projector_null.py`. Code: `trackS_subspace/s1.py`. Preflight:
`data/trackS/s1/2026-10-20-preflight/` (main tree), `summary.json` sha256 `932fb7c6b6a5`, eight
reports cited by sha below. Tier 1 throughout: exploratory, unregistered; 410m is an exploration
rung, so nothing here scores a registered row. **Frozen 2026-10-20 as written (Blocked 81 (a), the
user's): 3 units + close-out, raw frame, r = 16, counts not verdicts; unit 1's record is
`status-S.md` "S1 unit 1".** **Unit 1's gate: G0–G2 pass; G3 fails on step 8000's real pool in all 7
families (the trained spectrum is dense at the null's edge), so N is refused by the
fallback below; R1's statistic is Blocked 82, the user's.**

## What S1 asks

At 8 steps of 410m (0, 128, 256, 512, 1000, 2000, 4000, 143000), weights only:

1. **How many directions do the heads share**, per subspace family, across all 384 heads (heads
   and layers), against a random arrangement of the same subspaces; and how much of a head's
   subspace lies in them.
2. **Beside:** how much of that is sharing *within* a layer beyond the step's own global sharing.
3. **What the attention layers do inside each family's shared space**: push in or out (radial
   sign), rotate (complex pairs and their angle), and the negative-pair gap (B2's flag, the d = 2
   reduction's "collapsibility").

Not asked: whether a shared direction is used (S2, S5), clusters inside it (S3), causes (S4).

## Exit condition and budget

**Exit:** R1's 56 counts (7 families × 8 steps), each with its mass and the gate's resolution;
R3's table at the 8 steps; the beside columns. Or a recorded reason the gate cannot pass after
its fallbacks. **Budget: 3 units + a writing-only close-out** (unit 1 the gate; unit 2 R1, R1b,
R2; unit 3 R3 and the residual check). At the budget S1 closes at what it knows.

## The math (`tools/math_checks/s1_projector_null.py`)

For a pool of n heads with orthonormal bases U_h (d × k_h), P̄ = (1/n) Σ U_h U_hᵀ. Its
eigenvalues lie in [0, 1]; λ along u is the pool's mean of uᵀ P_h u, the share of heads holding u
(fact 1). Σ λ² is the pool's mean pairwise projection kernel, 2601.10266's measure (fact 2). A
direction held fully by k of n heads forces λ₁ ≥ k/n (fact 3), which sets the doses. **P̄ sees a
head only through its subspace, so the three nulls of ROADMAP §7a are one law** (fact 4):
rotating each trained head by its own Haar map gives independent Haar subspaces of the same
dimensions; a norm-matched Gaussian draw gives the same; Pythia's i.i.d. normal init is one draw
of it. The ROADMAP's reason for the rotation null (keep each head's spectrum) does not apply to
an unweighted projector. Weighted projectors, where it would, are not read (fenced).

## Objects, fixed here

| object | fixed as | beside |
|---|---|---|
| frame | raw residual (weights as stored); column convention, V = W_O W_V, M_0 = W_Qᵀ W_K / √64 (B2) | LN1 gain folded (`diag(g)` on the read side): unit 2, R1 only |
| families (7) | `ov_write`, `ov_read` (V's top-r left / right singular vectors); `qk_query`, `qk_key` (M_0's, offset 0); `ov_neg`, `ov_pos`, `ov_cplx` (V's invariant subspaces among its top-r eigenvalues by modulus, a complex pair counting 2, by type) | — |
| r | **16** (placed: B2's C6) | 8, 32 (unit 2, R1 only) |
| pools | `all` (384 heads); `L0`–`L23` (16 each) | — |
| steps | 0, 128, 256, 512, 1000, 2000, 4000, 143000 (ROADMAP) | — |
| null draws | 2000 per pool shape (singular families: one shape; invariant families: per cell, each head's dimension matched), GPU float64 | — |

## R: the readings (fixed before any scored-step output)

**R1 (the family: 7 families × 8 steps = 56 cells, pool `all`).** N = the number of eigenvalues
of P̄ above the cell's null threshold, the **largest λ₁ over 2000 Haar draws**. Its false count is
the threshold's for every rank at once: P(N ≥ 1 | null) ≤ 1/2001 per cell, ≤ 2.8 % over the 56.
With N: the mass m = (Σ of the counted λ) / (mean k_h), the share of an average head's subspace
inside the shared space; and the gate's resolution k* (G3), quoted beside every N as "directions
held by fewer than k* of 384 heads are not counted". N is a count, not a verdict (see
"Preflight"). Drawn: P̄'s top 200 eigenvalues per cell against the null's band.

**R1b (beside):** the overlap between families' shared spaces at each step,
tr(P_S P_S′) / min(N, N′) (S the span of a family's counted eigenvectors): write against read and
against key (B2's C6 at the population level), negative against positive. Each counted
direction's three largest coordinates (the massive-activation dims, `lit-S1.md` §4).

**R2 (beside, no reading):** each layer pool's count above its **layer-shuffle null** (2000
random 16-head sets of the same step's 384, same family): sharing within a layer beyond the step's
global sharing. Description only, because the preflight refuses it as a verdict for QK (below).

**R3 (unit 3; per family's shared space S with N ≥ 2, per layer l, per step).** C = U_Sᵀ V_l U_S
with V_l = Σ_h V_h (the layer's heads' OV summed: every head attending one token, not
attention-weighted). Read: its symmetric part's attractive share (B2's `share`: radial sign); the
complex share of C's eigenvalues (in pairs) and their mean |arg λ| (rotation); B2's `top_weight`
and negative-pair flag (|λ₂|/|λ₁| ≥ 0.8, both negative; the gap). Description across the 8 steps,
each against the same compression at step 0 (S fixed from the trained step, applied to step 0's
V_l) and against S replaced by a Haar subspace of the same dimension at that step (100 draws).

**Beside R3 (unit 3):** the residual stream's top-variance directions at each layer's input
(v1 battery prompts only, `core/holdout.py`; one forward pass per step, GPU): the share of their
variance inside each S. A shared space that is the residual stream's top variance is reported as
that, not as sharing.

## Unit 1's gate (before any scored-step count; CLAUDE.md "Calibration gate")

| arm | what | pass |
|---|---|---|
| G0 | wiring: the low-rank singular and invariant bases equal dense SVD / eig on 20 random heads and on 4 heads at step 8000 (subspace distance < 1e-8); P̄'s eigenvalues by Gram equal dense; GPU Haar draws match the numpy draws' λ₁ law (KS, 500 each) | all |
| G1 | one null law: λ₁ under (i) Gaussian-weight heads run through `head_bases` (norm-matched to 8000's), (ii) 8000's heads rotated, against the Haar draws (KS, 500 draws, both pool shapes, every family; invariant families with dims matched) | KS p > 0.001 in every cell |
| G2 | planted null at the design points: the rotation null at every scored step (each head's subspace turned, its dimension kept), every family, pool `all` | N = 0 in ≥ 55 of 56 cells |
| G3 | planted effect: 4 random directions each put into k random heads, on (a) a Haar pool and (b) **step 8000's real pool** (the count is read on top of existing sharing), k ∈ {12, 16, 24, 32} of 384, 50 draws, every family; read ΔN = N(planted) − N(unplanted) | ΔN ≥ 4 in ≥ 80 % of draws at k = 32 on both carriers, every family; k* = the smallest k passing that on (b) |
| G3l | R2's planted effect (beside, for R2's description): one direction in k of a layer's 16 heads on the shuffle carrier, OV k ∈ {4, 8, 12}, QK k ∈ {8, 12, 16} (the preflight's dose-4 refusal) | recorded, no pass |

Step 0's 7 cells (pool `all`, one per family) are G1's (iii) beside: each reads N = 0 if Pythia's init is the Haar law (it is
read as a check of the null, not as a reading).

**Pre-authorised fallbacks** (ROADMAP §5; scope stays the user's):
- G1 fails for a family (the law differs, not the code): that family's null becomes its rotation
  null drawn per cell (2000 draws), threshold the larger of the two maxima; re-gate in the unit.
- G2 reads N > 0 in more than 1 cell: thresholds at the maximum of 4000 draws; re-gate once.
- G3 fails at k = 32 in a family: add k = 48, 64; a family failing at 64 (17 % of heads) has N
  refused and reports its spectrum only.
- Step 0 reads N > 0 in a family: that family's threshold is the larger of Haar's and step 0's
  λ₁, stated beside; re-gate.

## Preflight (`tools/design_preflight.py`, step 8000, off the grid)

Statistic: λ₁ of a layer pool (24 layers as the units, the form the tool takes); planted: a
random direction in k of the pool's 16 heads, on the null's own carrier; decision p 6.25e-4 (the
E bar); 2000 null draws (floor 5.0e-4, E up to 22.4).

| family | Haar null | report | layer-shuffle null | report |
|---|---|---|---|---|
| `ov_write` | refused: `band` (24 of 24 layers outside [0.1317, 0.1375]); doses 3, 4, 6 seen 1.00 | `51f6ca8633ff` | pass (doses 4, 8, 12 seen 1.00) | `72f656bbb056` |
| `ov_read` | refused: `band` (24 of 24) | `d1d86b95a59a` | pass | `99aaffb4fb51` |
| `qk_query` | refused: `band` (24 of 24) | `eb263a2c27e0` | refused: `dose:4` (mean 0.384, below the shuffled pools' own) | `6549affa7c10` |
| `qk_key` | refused: `band` (24 of 24) | `ab7ab5a1ddb3` | refused: `band` (24 of 24 outside [0.268, 0.419]) and `dose:4` | `48b880f605cc` |

**What the refusals change.** (1) Against Haar, "does this unit share?" is answered yes for every
layer at a trained step, so S1 reads no per-unit verdict: R1 is a count with its resolution, and
Haar plays the role of a moved mark at trained steps (the tool's term) and of the null at step 0.
The count's calibration is G3 on step 8000's real pool, which the band check does not test.
(2) R2 is demoted to description, and its QK doses start at 8 (G3l). (3) Key subspaces share
within every layer beyond the step's global sharing (cf. 2603.13314 on OLMo-2); stated, not led
with. The tool's report is cited, not re-run, for R1's own statistic (a single pool per step has
no unit population to give it).

## Five ways to answer "how many shared directions, per step"

Written in this session, not by a fresh-context pass (ROADMAP §5's row); `/challenge-pr` is the
fresh pass. Simplest first:

1. The mean pairwise projection kernel per step (2601.10266): one scalar, no count. It is R1's
   Σ λ² (fact 2), drawn beside.
2. **The mean projector's spectrum against a simulated Haar maximum (taken):** the count and the
   directions come from one eigendecomposition with an exact null; the directions feed R3.
3. Clustering heads by kernel (spectral clustering) into groups that share: names the heads,
   but the group count needs its own null (1d's whole problem).
4. Activation-based sharing (2603.13314's linear predictability, CCA of heads' outputs): needs
   tokens; S1 is weights only, and S3 reads tokens.
5. A tensor decomposition of the stacked OVs (CP / Tucker): shared components with per-head
   weights, but non-convex, rank chosen by hand, no null.

## Seen before the readings were fixed

`s1.py explore` (code as of this design) on steps **0** and **8000**: per family, P̄'s top 32
eigenvalues for every pool, and a sequential per-rank count at the 95th percentile.
- Step 0 (a scored step): every family's layer λ₁ inside Haar's range (0.131–0.138); pool `all`
  top 5 at 0.030–0.031; sequential counts 0 in pool `all`, and 1–4 in 4 of 96 layer cells (the
  per-rank 5 % rule's false rate; R1 uses the maximum instead).
- Step 8000 (off the grid): pool `all` holds 79–89 eigenvalues above Haar's maximum (0.0311) per
  singular family, mass 3.3–5.7 of 16; top λ₁ 0.084 (`ov_write`), 0.083 (`ov_read`), 0.347
  (`qk_query`), 0.294 (`qk_key`); `ov_neg` λ₁ 0.017 and `ov_cplx` 0.054 (no matched null drawn).
  The top direction's largest coordinate holds ≤ 1.8 % of it. Layer-shuffle counts as in the
  preflight. No other step was loaded.

## Fences

- Weighted projectors, the folded frame beyond R1, r beyond 8 / 32: not read.
- R3 sums a layer's OVs without attention weights; it is the "every head attends the same token"
  reading, not the dynamics (B2a: the reduced dynamics reproduce no head).
- No claim that a shared direction is used, or that sharing causes anything (S2, S5).
- Any other rung, seed, or the 12 held-out v2 prompts.

## Worth challenging (for the freeze)

- **Raw frame primary.** The head reads LN1(x), so its effective read space includes the gain;
  folding the gain makes a layer's heads share its large-gain coordinates trivially. Rejected:
  folded primary. Raw keeps the weights' own geometry and matches B2's C6.
- **r = 16, unweighted.** A head's top-16 of 64 directions, each counted equally; a head with a
  flat spectrum contributes noise directions. Rejected: weighted projectors (need a per-head
  spectrum null and lose fact 4's single law).
- **Counts, not verdicts,** after the preflight's band refusal. The alternative was to look for a
  null that trained layers do not all leave (the layer shuffle); it fails for keys too.
- **The max-based threshold** undercounts a bulk shifted below the null's top edge. Rejected:
  per-rank quantiles (Horn's parallel analysis), whose false count grows with the number of ranks.
- **The invariant families' "top-r by modulus"** mixes types inside one budget, so a head with
  many complex pairs holds few real directions; a type's count partly measures that mix.
- **Planting by exact inclusion** (cos 1) only; a direction held partially by many heads is
  calibrated by fact 3's bound, not by G3.
