<!-- trackS_subspace/design-S2.md -->
# Track S — design: S2, what rotation does

Opened 2026-10-21 by `docs/ROADMAP.md` §9 row 6 (… → S1 → **S2** → C2's draft). Row:
`docs/ROADMAP.md` §7a S2. Scan: `trackS_subspace/lit-S2.md`. Math:
`tools/math_checks/s2_ov_arms.py` (19 of 19 pass). Harness reused: Phase 10's OV1 writer
(`tools/run/p10_ov1_cut.py`: LN1 folded, centred frame; the head's constant moved to the dense
bias; read-back) and A5's pass (`trackA_force/a5_seeds.py` `one_pass`: E₁, V). Tier 1 throughout:
exploratory, unregistered; 410m is an exploration rung, so nothing here scores a registered row.
**Proposed, not frozen: the shape is Blocked 86, the user's.** No code is written and nothing has
run under S2's name.

## What S2 asks

With every attention head's OV edited at once, on 410m at the steps around and after the energy
break (512):

1. **Q1, the energy.** Is the break the OV's non-symmetric part, or its rotation? Does V (A5's
   violation count, 0 through 128, 19 at 256, 56 at 512, 56–76 after) fall back toward 0 under
   arm (a) or (b), more than under a random edit of the same size in the same spaces?
2. **Q2, induction.** Does in-context copying survive each arm (7d's repeated-random-token probe)?
3. **Q3, S1's shared rotation.** Is what Q1–Q2 read carried by the complex pairs lying in S1's
   shared `ov_cplx` space, against the same number of the heads' other pairs?

Not asked: which heads matter (Track B), whether a shared direction is used outside rotation (S5),
clusters (S3), causes across training (S4). No theorem predicts "monotone" for either arm: the
gradient-flow results tie V to QᵀK, and a symmetric V with untied QK breaks E₁'s monotonicity in
53 of 200 toy flows (math check C6; `lit-S2.md` §2). Q1 is empirical.

## Exit condition and budget

**Exit:** R1's 10 cells, R2's 8 and R3's 9, each labelled (or refused by the gate) with its
displacement beside; the carried items run or homed. Or a recorded reason the gate cannot pass
after its fallbacks. **Budget: 3 units + a writing-only close-out** (unit 1 code, gate, preflight
and the carried weights-only items; unit 2 R1 and R2; unit 3 R3, its leave-the-layer-out S and
any matched-distance fallback). The ROADMAP estimated 2; the third is S1's carried items and Q3.
At the budget S2 closes at what it knows.

## The arms (`tools/math_checks/s2_ov_arms.py`)

Per head, in OV1's frame: B = Π W_O, C = W_V diag(γ) Π (LN1's gain γ folded in, Π the centring),
K = B C (the map a head applies to the attention-weighted x̂; its constant c = W_O(W_V β + b_V)
moves to the dense bias, OV1's C1), A = C B (k × k, K's nonzero eigenvalues), Z = [B, Cᵀ]
(d × 2k). **Every arm is K' = Z M Zᵀ with a 2k × 2k core M** (C1), so one hook runs them all and
the base model is the core E = [[0, I], [0, 0]]:

| arm | core M | what it is | size per head |
|---|---|---|---|
| base | E | the model (G0 checks the hook reproduces it) | 0 |
| **a**, symmetric | J = (E + Eᵀ)/2 | K := sym(K): rank up to 2k = 128, writes into the read space too (so not rotation alone; the ROADMAP's caveat) | ‖skew K‖ (C4; OV1 measured ‖K − S‖ at 0.66–0.71 ‖K‖) |
| a_ctl × 40 | E + R, R Gaussian, scaled | K plus a random edit in the same 2k-dim span | = arm a's, per head (C3: exact through the Gram ZᵀZ) |
| **b**, rotation removed | [[0, g(A)], [0, 0]], g(A) = Q diag(Re λ / λ) Q⁻¹ | each complex pair λ → Re λ on its own invariant plane (the rotation generator removed, the radial part kept); real eigenvalues, eigenvectors and write space kept; rank drops only where Re λ = 0 (C2) | ‖B (g(A) − I) C‖ |
| b_ctl × 40 | [[0, I + R′], [0, 0]], R′ Gaussian k × k, scaled | a random edit with b's read and write spaces | = arm b's, per head |
| **bS** (unit 3) | as b, g on the selected pairs only (C5) | pairs whose plane lies in S1's shared `ov_cplx` space (below) | its own |
| bR × 40 (unit 3) | as bS on random pairs | per head, the same number of its other complex pairs, drawn at random | recorded beside |

**The ROADMAP row says "angle set to 0 (λ → Re λ)"**; these differ (angle 0 keeps |λ|, Re λ
shrinks it by |cos θ|). S2 takes **λ → Re λ**: in each invariant plane the block is a·I + b·J′, and
the arm removes b·J′ and nothing else. The modulus-kept version is not run (Blocked 86 (c)).

Arm b is also writable in the weights (W_V := g(A) W_V, b_V adjusted so c is unchanged), which G0
uses as a second implementation. Arm a cannot be (rank 128 in a 64-wide head), so it runs through
the hook: per head, Y = X̂ Z (n × 2k), then P Y, then (P Y) M Zᵀ, added to the layer's output with
the head's V path zeroed.

**Q3's S and selection.** S = P̄'s top 10 eigenvectors of the `ov_cplx` family at the step
(S1's definition), **left out by layer**: for layer l's heads, S₋ₗ is built from the other 23
layers' 368 heads (S1's carried selection-bias fix, `status-S.md` "S1 on one page"). A head's
complex pair is "in S" when its plane's share in S₋ₗ (‖P_S U‖²_F / 2, U the plane's orthonormal
basis) is at or above the 99th percentile of that share for a Haar plane against a 10-dim space
in 1024 (drawn, 10⁵ draws). Fewer than 100 such pairs over the model at a step: R3 refused there.

## Steps and readouts

| item | fixed as |
|---|---|
| scored steps | 512, 1000, 2000, 4000, 143000 (the break and after; induction forms in (512, 4000]) |
| beside | 0 (wiring; V = 0 there), 256 (pre-break, V = 19) |
| preflight | 8000 (not scored) |
| V | A5's `one_pass` E₁ on the 25 hidden states of v1's 7 prompts, violations at rel_tol 1e-3, summed (at most 168); known answer: the panel's `p1_energy_nviol_b1` |
| G, the repeat gain | 7d's probe (`tools/run/induction_rank_sweep.py`: 96 random tokens in [1000, 40000), repeated; `EVAL_SEED`), 16 sequences: mean NLL on the first half's positions minus on the second's (7d read the second only; the first half separates general damage from lost copying) |
| D, displacement | per layer, mean over tokens and prompts of the centred ‖h_arm − h_base‖ / ‖h_base‖, median over layers 1–24 (OV1d's "centred distance") |
| beside | v1 NLL; the four 7d members' +1 induction scores (A5's `ind`) |

Each record carries the code sha, the step's revision, the arm, the draw and the per-head sizes.

## Readings (fixed before any scored-step output)

A cell's **label** against its 40 controls: **lower** (below every control), **higher** (above
every control), else **as its controls**; ties go to "as its controls". Per cell the false-label
rate is 2/41 ≈ 0.05 under exchangeability.

**The displacement guard (every cell).** If the arm's D is above every control's D, the label is
written "… , moved further" and **not read** (OV1's lesson, `lit-S2.md` §1). If it is below every
control's, the label is written "… , moved less" and read with that stated.

| reading | cells | statistic | answer, fixed now |
|---|---|---|---|
| **R1** (Q1) | arms a, b × 5 scored steps = 10 | V | per arm, **"the break goes with it"** if the cell is *lower* with V ≤ 7 (at most one violation per prompt, placed) at ≥ 4 of 5 steps; **"not shown"** otherwise |
| **R2** (Q2) | arms a, b × 1000, 2000, 4000, 143000 = 8 | G | per cell, **"survives"** if G_arm ≥ 0.5 G_base and not *lower*; **"lost"** if *lower* and G_arm < 0.5 G_base; else **"weakened"**. 512 beside |
| **R3** (Q3, unit 3) | bS against bR × 5 steps on V, and × 1000–143000 on G = 9 | V, G | per cell, the label against bR; "S1's shared rotation carries it" if bS is *lower* than bR in V (or G) at ≥ 4 of 5 (3 of 4) |

Not read: any label marked "moved further"; a lean (a cell inside its controls but at an extreme
rank); differences between arms a and b (their edits differ in size and span).

## What a reading can mean

- R1 "the break goes with it" under a: removing OV's antisymmetric part, at that size in those
  spaces, takes V back toward 0 beyond a random edit there. It does not say Pythia follows the
  theory's gradient flow (C6), or which heads carry it.
- Under b: the same for OV's rotation alone (the radial part kept).
- "As its controls" everywhere: an edit of that size there moves V as much, so S2 cannot separate
  the rotation from damage; quoted with the controls' spread.
- R2 "lost" under b and not under b_ctl: in-context copying needs OV's rotation.

## The calibration gate (unit 1, before any scored-step output; CLAUDE.md "Calibration gate")

| arm | what | pass |
|---|---|---|
| G0 | wiring: the hook at core E reproduces the model's logits and hidden states (relative 1e-4, float32) at steps 0 and 8000; arm b by hook equals arm b written in W_V (1e-4); on 20 heads at 8000, the read-back K′ has eigenvalues Re λ (b) and equals sym K (a) (1e-6 of ‖K‖); base V reproduces the panel at all 7 steps (0, 19, 56, 56, 61, 57, 70) | all |
| G1 | planted null at the design points: a 41st control draw read as the arm, every scored step, both arms | labelled in ≤ 2 of 10 R1 cells and ≤ 2 of 8 R2 cells |
| G2 | planted effect for R2 (a positive control): the four 7d members' OV zeroed (`L5H2`, `L7H8`, `L8H6`, `L12H5`, `p7_motifs/status-7.md`), against 40 random edits of the same per-head size on the same heads | "lost" at ≥ 3 of 4 R2 steps |
| G3 | planted effect for R1 (a known answer): the whole model interpolated toward step 128 (V = 0 there), W(t) = W_s + t (W₁₂₈ − W_s), t ∈ {0.25, 0.5, 1}, against 40 Gaussian edits of the same per-tensor norm | at t = 1: *lower* with V ≤ 7 at ≥ 4 of 5 scored steps; the curve at 0.25, 0.5 recorded |

**Preflight** (`tools/design_preflight.py`, ROADMAP §5): at step 8000, the controls as the null,
G1's draw at dose 0 and G3's t as doses, `decision_p` = 1/41; the report's sha256 is cited here
before any scored step runs. If the tool cannot take a rank-against-controls statistic (it wants
≥ 20 per-unit trained values), unit 1 records why and G1–G3 stand in for it, as in S1.

**Pre-authorised fallbacks** (scope stays the user's):

- G0 fails: a code defect; fix and re-run (no rule changes).
- G1 labels more than 2 cells: 100 controls per arm (false-label rate 2/101), re-gate once.
- G2 fails: R2 is refused and G reported as description.
- G3 fails because the controls are monotone too (control median V ≤ 7 at t = 1): a random edit
  of that size already makes E₁ monotone, so R1 is refused at any cell whose controls' median
  V ≤ 7, and that fact is the reading there.
- A head whose eigenvector matrix has condition number above 1e8 (C2 does not cover a defective
  A): it keeps its K in arm b, counted; more than 5 % of heads at a step: arm b refused there.
- Cells marked "moved further" (unit 3, if the budget allows): their controls re-drawn at 1.5, 2
  and 3 times the size and the label read at matched D by linear interpolation (OV1d's method).

## Carried in from S1 (`status-S.md` "S1 on one page")

| item | where in S2 |
|---|---|
| R3 on a leave-the-layer-out S for the OV families (the selection bias) | unit 3: S₋ₗ is Q3's S, and S1's R3 table is re-read on it beside (weights only) |
| R1b, overlap between families' shared spaces (write against key) | unit 1, beside, weights only, description (`s1.shared_space` at the 8 S1 steps) |
| R2, within-layer sharing above the layer-shuffle null | unit 1, beside, weights only, description |

## Compute

Per step: base, 2 arms, 80 controls, 1 G1 draw, each on v1's 7 prompts and 16 probe sequences:
about 2,000 forward passes; 7 steps, about 14,000 (unit 2), plus the gate's at 8000 and the
scored steps (unit 1). RTX 3080, float32, eager, TF32 off (A5's settings); the hook adds
16 × (n·d·2k) per layer. Estimated under an hour per unit on the GPU; the `runner` agent watches.
