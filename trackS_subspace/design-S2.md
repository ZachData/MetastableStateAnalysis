<!-- trackS_subspace/design-S2.md -->
# Track S — design: S2, what rotation does

Opened 2026-10-21 by `docs/ROADMAP.md` §9 row 6 (… → S1 → **S2** → C2's draft). Row:
`docs/ROADMAP.md` §7a S2. Scan: `trackS_subspace/lit-S2.md`. Math:
`tools/math_checks/s2_ov_arms.py` (24 of 24 pass). Harness reused: Phase 10's OV1 writer
(`tools/run/p10_ov1_cut.py`: LN1 folded, centred frame; the head's constant moved to the dense
bias; read-back) and A5's pass (`trackA_force/a5_seeds.py` `one_pass`: E₁, V). Tier 1 throughout:
exploratory, unregistered; 410m is an exploration rung, so nothing here scores a registered row.
**Blocked 86 decided (a) and (i), the user's, 2026-10-21: arm (b) is (b-schur), (b-eig) beside
with no label; scope 3 units + close-out, with Q3.** Not frozen: the freeze follows unit 1's
preflight and is the user's (Blocked 87). Code `trackS_subspace/s2.py`; unit 1's choices made in
code, before any scored-step output, are in "Unit 1: fixed in code" below. Revised after
`/challenge-pr` on #246 (accept with changes, findings 1–7 taken; "Revised after review" at the end).

## What S2 asks

With every attention head's OV edited at once, on 410m at the steps around and after the energy
break (512):

1. **Q1, the energy.** Is the break the OV's non-symmetric part, or its rotation? Does V (A5's
   violation count, 0 through 128, 19 at 256, 56 at 512, 56–76 after) fall back toward 0 under
   arm (a) or (b), more than under a random edit of the same size and the same output gain in the
   same spaces, and more than under the same gain change alone?
2. **Q2, induction.** Does in-context copying survive each arm (7d's repeated-random-token probe)?
3. **Q3, S1's shared rotation.** Is what Q1–Q2 read carried by the complex pairs lying in S1's
   shared `ov_cplx` space, against the same number of the heads' other pairs?

Not asked: which heads matter (Track B), whether a shared direction is used outside rotation (S5),
clusters (S3), causes across training (S4). No theorem predicts "monotone" for either arm: the
gradient-flow results tie V to QᵀK, and with untied QK a toy attention-only flow breaks E₁'s
monotonicity in 53 of 200 draws with V symmetrised, against 59 of 200 with the same V unsymmetrised
(math check C6; `lit-S2.md` §2). In the toy, symmetrising V barely changes monotonicity: a weak
prior against R1's positive answer. Q1 is empirical.

## Exit condition and budget

**Exit:** R1's 10 cells, R2's 8 and R3's 9, each labelled (or refused by the gate) with its
displacement and its gain arm beside; the carried items run or homed. Or a recorded reason the
gate cannot pass after its fallbacks. **Budget: 3 units + a writing-only close-out.** Unit 1:
code, G0, the preflight at 8000 and the carried weights-only items, then the freeze (the user's).
Unit 2: G1–G3 at the design points, then R1 and R2. Unit 3: R3, its leave-the-layer-out S and
any matched-distance fallback. The ROADMAP estimated 2; the third is S1's carried items and Q3.
At the budget S2 closes at what it knows.

## The arms (`tools/math_checks/s2_ov_arms.py`)

Per head, in OV1's frame: B = Π W_O, C = W_V diag(γ) Π (LN1's gain γ folded in, Π the centring),
K = B C (the map a head applies to the attention-weighted x̂, column convention; its constant
c = W_O(W_V β + b_V) moves to the dense bias, OV1's C1), A = C B (k × k, K's nonzero
eigenvalues), Z = [B, Cᵀ] (d × 2k). **Every arm is K′ = Z M Zᵀ with a 2k × 2k core M** (C1; C7
for the Schur-plane arm), so one hook runs them all and the base model is the core
E = [[0, I], [0, 0]]. Row-wise the hook adds, per head, X̂ ↦ P (X̂ Z) Mᵀ Zᵀ (n × d; X̂ the layer's
normalised inputs, P the head's attention), with the head's V path zeroed. Arm (a) cannot be
written into a 64-wide head (rank 128), which is why the hook carries every arm.

| arm | what | edit size per head (‖K′ − K‖) |
|---|---|---|
| base | core E: the model (G0 checks the hook reproduces it) | 0 |
| **a**, symmetric | core J = (E + Eᵀ)/2: K := sym K. Rank up to 128, and it writes into the read space too, so it is not rotation alone (the ROADMAP's caveat) | ‖skew K‖ (C4), 0.66–0.71 ‖K‖ (OV1; #246's review measured ‖sym K‖ / ‖K‖ at 0.71–0.75) |
| **b**, rotation removed | **two definitions, Blocked 86** (next section) | see there |
| g_a, g_b, the gain arms | K := c K with c = ‖K′‖ / ‖K‖ of arm a or b: the arm's change in output gain and nothing else | \|1 − c\| ‖K‖ |
| controls × 40 per arm | K + D with D doubly matched (C8): D's component along K equals the arm's, and the rest is a random direction in the arm's own span, orthogonal to K, of the arm's remaining size. So ‖D‖ = ‖K′ − K‖ **and** ‖K + D‖ = ‖K′‖ exactly | = the arm's |
| **bS** (unit 3) | arm b on the selected pairs only (C5 for the eigenbasis form) | its own |
| bR × 40 (unit 3) | arm b on the same number of each head's other complex pairs, drawn at random | recorded beside |

**Arm (b): two definitions (Blocked 86).** Both remove every complex pair's rotation, and both
keep the write space (C2, C7) and every real eigenvalue ((b-eig) also its eigenvector, C2;
(b-schur) by the quasi-triangular form, not checked numerically). They differ in
the metric the rotation is removed in:

| | (b-eig), λ → Re λ in A's eigenbasis | (b-schur), symmetric blocks in an orthonormal basis |
|---|---|---|
| definition | g(A) = Q diag(Re λ / λ) Q⁻¹, K′ = B g(A) C | U from B = U R; T = R C U, K's restriction to col(B); real Schur T = V S Vᵀ; each 2 × 2 block of S replaced by its symmetric part; K′ = U V S′ Vᵀ Uᵀ + U N, N = R C (I − U Uᵀ) |
| eigenvalues | Re λ, twice per pair, on K's own eigenvectors | a ± (b + c)/2 per block [[a, b], [c, a]]: real, with the pair's trace kept |
| "radial part kept" | **in A's eigenbasis only**, which is far from orthogonal on 410m (#246's review, 15 heads: median log₁₀ cond(Q) ≈ 1.8) | in the residual's metric, plane by plane: (b-schur) is arm (a) restricted to each invariant plane |
| edit size | **large**: on 410m (15 heads, review) median ‖K′ − K‖ / ‖K‖ 1.62 at 512, 0.66 at 1000, 0.80 at 143000, and ‖K′‖ / ‖K‖ 1.69 at 512; on Gaussian heads 0.95 (C7) | ‖S′ − S‖, the blocks' skew parts only; on Gaussian heads 0.19 (C7); on 410m not yet measured (G0) |
| the ROADMAP row's words | "λ → Re λ" | "angle set to 0" in an orthonormal plane basis |

**Recommended: (b-schur).** (b-eig) at 512 edits each head by more than its own size and makes
its controls a broken model. (b-schur) removes rotation in the metric E₁ is measured in, and its
edit is bounded by the blocks' skew parts. (b-eig) stays beside, with its sizes recorded and
no label. The modulus-kept variant (λ → sign(Re λ)·|λ|) is not run.

**Q3's S and selection.** S = P̄'s top 10 eigenvectors of the `ov_cplx` family at the step
(S1's definition), **left out by layer**: for layer l's heads, S₋ₗ is built from the other 23
layers' 368 heads (S1's carried selection-bias fix, `status-S.md` "S1 on one page"). A head's
complex pair (its plane: the Schur block's two columns of U V, or for (b-eig) the orthonormalised
eigenplane) is "in S" when its share in S₋ₗ (‖P_S U_pair‖²_F / 2) is at or above the 99th
percentile of that share for a Haar plane against a 10-dim space in 1024 (10⁵ draws). Fewer than
100 such pairs over the model at a step: R3 refused there.

## Steps and readouts

| item | fixed as |
|---|---|
| scored steps | 512, 1000, 2000, 4000, 143000 (the break and after; induction forms in (512, 4000]) |
| beside | 0 (wiring; V = 0 there), 256 (pre-break, V = 19) |
| preflight | 8000 (not scored) |
| V | A5's `one_pass` E₁ on the 25 hidden states of v1's 7 prompts, violations at rel_tol 1e-3, summed (at most 168); known answer: the panel's `p1_energy_nviol_b1` |
| G, the repeat gain | 7d's probe (`tools/run/induction_rank_sweep.py`: 96 random tokens in [1000, 40000), repeated; `EVAL_SEED`), 16 sequences: mean NLL on the first half's positions minus that on the second half's. 7d read the second half only; the first half separates general damage from lost copying, though on random tokens it sits near uniform, so G may mostly track the second half (`/challenge-pr` on #246, the author's doubt 6) |
| D, displacement | per layer, mean over tokens and prompts of the centred ‖h_arm − h_base‖ / ‖h_base‖, median over layers 1–24 (OV1d's "centred distance") |
| sizes | per head and step, for every arm: ‖K′ − K‖ / ‖K‖ and ‖K′‖ / ‖K‖ (median, range) |
| beside | v1 NLL; the four 7d members' +1 induction scores (A5's `ind`) |

Each record carries the code sha, the step's revision, the arm, the draw and the per-head sizes.

## Readings (fixed before any scored-step output)

A cell's **label** against its arm's 40 doubly matched controls: **lower** (below every control),
**higher** (above every control), else **as its controls**; ties go to "as its controls". The
false-label rate is 2/41 ≈ 0.05 per cell under exchangeability.

**The gain arm (every cell).** The arm's gain arm (g_a or g_b) is labelled against the same
controls. A label is **read** for the arm only if the arm is also beyond its gain arm in the same
direction (V_arm < V_gain for *lower*; G likewise). Otherwise it is written "… , as its gain" and
not read as being about skew or rotation. A gain arm that is itself *lower* or *higher* is stated
beside: random structure at that size moves V by itself.

**The displacement guard (every cell).** If the arm's D is above every control's D, the label is
written "… , moved further" and **not read** (OV1's lesson, `lit-S2.md` §1). If it is below every
control's, it is written "… , moved less" and read only through the gain arm above, whose D is
recorded beside.

| reading | cells | statistic | answer, fixed now |
|---|---|---|---|
| **R1** (Q1) | arms a, b × 5 scored steps = 10 | V | per arm, **"the break goes with it"** if the cell is read *lower* with V ≤ 7 (at most one violation per prompt on average, placed) at ≥ 4 of 5 steps; **"not shown"** otherwise |
| **R2** (Q2) | arms a, b × 1000, 2000, 4000, 143000 = 8 | G | per cell, **"survives"** if G_arm ≥ 0.5 G_base and not read *lower*; **"lost"** if read *lower* and G_arm < 0.5 G_base; else **"weakened"**. 512 beside |
| **R3** (Q3, unit 3) | bS against bR × 5 steps on V, and × 1000–143000 on G = 9 | V, G | per cell, the label against bR; "S1's shared rotation carries it" if bS is *lower* than bR in V (or G) at ≥ 4 of 5 (3 of 4) |

Not read: any label written "moved further" or "as its gain"; a lean (a cell inside its controls
at an extreme rank); differences between arms a and b (their edits differ in size and span).

## What a reading can mean

- R1 "the break goes with it" under a: removing OV's antisymmetric part takes V back toward 0
  beyond a random edit of that size and gain in those spaces, and beyond the gain change alone.
  It does not say Pythia follows the theory's gradient flow (C6), or which heads carry it.
- Under b: the same for OV's rotation alone, in the metric arm (b) is defined in (Blocked 86).
- "As its controls" or "as its gain" everywhere: S2 cannot separate the edit's structure from its
  size and gain; quoted with the controls' spread.
- R2 "lost" under b: in-context copying needs OV's rotation, beyond the gain and size of removing it.

## The gate

**Unit 1, before the freeze (no scored-step output).**

| arm | what | pass |
|---|---|---|
| G0 | wiring: the hook at core E reproduces the model's logits and hidden states (relative 1e-4, float32) at steps 0 and 8000; arm b by hook equals arm b written in W_V (1e-4, (b-eig) only: (b-schur) changes the read map and is checked by read-back alone); on 20 heads at 8000 the read-back K′ has the arm's eigenvalues (b) and equals sym K (a) (1e-6 of ‖K‖); a doubly matched control's two norms equal the arm's (1e-6); base V reproduces the panel at all 7 steps (0, 19, 56, 56, 61, 57, 70). **Recorded:** every arm's per-head sizes at all 7 steps and 8000, before any scored V or G is computed | all |
| preflight | `tools/design_preflight.py` at 8000: the controls as the null, the gain arm and G1's draw at dose 0, G3's t as doses, `decision_p` = 1/41. Its report's sha256 is cited here before the freeze (Blocked 80 (a), recommended). If the tool cannot take a rank-against-controls statistic (it wants ≥ 20 per-unit trained values), unit 1 records why and G1–G3 stand in for it, as in S1 | no refusal, or the refusal argued here |

**Unit 2, after the freeze, at the design points (CLAUDE.md "Calibration gate").**

| arm | what | pass |
|---|---|---|
| G1 | planted null, code: a 41st control draw read as the arm, every scored step, both arms. It is exchangeable with the 40 by construction, so this checks the labeller only | labelled in ≤ 2 of 10 R1 cells and ≤ 2 of 8 R2 cells |
| G1′ | planted null of the arm's form: the gain arms g_a, g_b against the controls at every scored step. Not a pass / fail: where a gain arm is labelled, the comparison is biased by gain, and the gain-arm rule above already makes such cells unreadable for the arm | recorded per cell |
| G2 | planted effect for R2 (positive control): the 10 heads with the largest single-ablation ΔNLL in 7d's catalogue (`data/analysis/redundancy_catalog.json`, step 16000, 8 sequences: `L5H2`, `L7H8`, `L12H5`, `L8H6`, `L11H14`, `L8H9`, `L15H14`, `L7H1`, `L9H13`, `L10H9`) with their OV zeroed, against 40 doubly matched random edits on the same heads. B3 found three members carry about 23 % of a related response (`trackB_circuit/status-B.md`), so four heads may not halve G; ten are expected to, not known to | "lost" at ≥ 3 of 4 R2 steps |
| G3 | planted effect for R1 (known answer): the whole model interpolated toward step 128 (V = 0 there), W(t) = W_s + t (W₁₂₈ − W_s), t ∈ {0.25, 0.5, 1}, against 40 Gaussian edits of the same per-tensor norm | at t = 1: *lower* with V ≤ 7 at ≥ 4 of 5 scored steps; 0.25, 0.5 recorded |
| G3′ | reach of an OV-only edit: every head's K replaced by its own step-128 K, everything else at step s, against 40 doubly matched controls | recorded. If it is not *lower* with V ≤ 7 at ≥ 4 of 5 steps, R1's positive answer is out of reach of any OV-only edit, and that is stated before R1 is read |

**Pre-authorised fallbacks** (scope stays the user's):

- G0 fails: a code defect; fix and re-run (no rule changes).
- G1 labels more than 2 cells: 100 controls per arm (false-label rate 2/101), re-gate once.
- G2 fails: R2 is refused and G is reported as description.
- G3 fails because the controls are monotone too (control median V ≤ 7 at t = 1): a random edit
  of that size already makes E₁ monotone, so R1 is refused at any cell whose controls' median
  V ≤ 7, and that fact is the reading there.
- (b-eig) only: a head whose eigenvector matrix has condition number above 1e8 (C2 does not
  cover a defective A) keeps its K, and is counted; more than 5 % of heads at a step: refused
  there.
- Cells written "moved further" (unit 3, if the budget allows): their controls re-drawn at 1.5, 2
  and 3 times the size, and the label read at matched D by linear interpolation (OV1d's method).

## Carried in from S1 (`status-S.md` "S1 on one page")

| item | where in S2 |
|---|---|
| R3 on a leave-the-layer-out S for the OV families (the selection bias) | unit 3: S₋ₗ is Q3's S, and S1's R3 table is re-read on it beside (weights only) |
| R1b, overlap between families' shared spaces (write against key) | unit 1, beside, weights only, description (`s1.shared_space` at the 8 S1 steps) |
| R2, within-layer sharing above the layer-shuffle null | unit 1, beside, weights only, description |

## Compute

Per step: base, 2 arms, 2 gain arms, 80 controls, 1 G1 draw, each on v1's 7 prompts and 16 probe
sequences: about 2,000 forward passes; 7 steps, about 14,000 (unit 2), plus G2, G3, G3′ (about
3 × 41 configurations per scored step) and the preflight at 8000 (unit 1). RTX 3080, float32,
eager, TF32 off (A5's settings); the hook adds 16 × (n·d·2k) per layer. Estimated under an hour
per unit on the GPU; the `runner` agent watches.

## Unit 1: fixed in code (2026-10-21, before any scored-step output)

| item | fixed as | why |
|---|---|---|
| coordinates | Z = [B, Cᵀ] = Q R, Q orthonormal (d × 2k); every arm is a 2k × 2k core in Q's basis (`s2.head_frame`) | norms, symmetric parts and eigenvalues are the core's; no ill-conditioned Z in the hook |
| the write along 1 | each head's (1/d) 1 1ᵀ W_O W_V diag(γ) P x̂ is kept as the model's own in every arm | the centred frame leaves it out; no LayerNorm reads it, but V does (A5's E₁ normalises the raw state, `core.models.layernorm_to_sphere` is plain L2), so an arm that dropped it would move V by a term no arm defines. It is the same in every arm and control |
| G0's comparisons | base hook against the model: logits and raw hidden states; (b-eig) by hook against written: logits and *centred* hidden states | writing g(A) W_V changes the write along 1 too, which the hook keeps |
| controls' span | (a): all of Z; (b-schur): col(B) both sides (its edit is U V (S′ − S) Vᵀ Uᵀ); (b-eig): col(B), row(C). The random part is made orthogonal to K's projection on the span, so it stays inside and ⟂ K | `lit-S2.md`: "the edit's own read and write spaces"; C8's double match holds exactly (`tests/test_trackS_s2.py`) |
| draws | per head, `seed_for("s2-ctl", step, arm, draw, layer, head)`; draw 40 is G1's 41st | — |
| the preflight's input | V per arm at 8000: null the arm's 40 controls; dose 0 its gain arm and G1's draw; doses G3's t (one model each, no G3 controls in unit 1); `trained` the base's 7 per-prompt counts (V is one number per model: no per-unit population of 20 exists, so the tool refuses `trained`, as foreseen); `decision_p` 1/41 | "The gate" |
| R1b | S1's raw frame, r 16, S = P̄'s top 10 (84 (a)); tr(P_S P_S′) / 10 for write–read, write–key, negative–positive; Haar reference 2000 draws (mean 10/1024) | `design-S1.md` R1b |
| R2 | per layer pool: λ₁, the count of P̄'s eigenvalues above the step's layer-shuffle maximum λ₁ (2000 random 16-head sets of the 384, same family), λ₁'s rank p | `design-S1.md` R2; description only |

## The preflight's refusals (unit 1, 2026-10-21; numbers `status-S.md` "S2 unit 1")

Reports: `data/trackS/s2/2026-10-21-u1-preflight/preflight_report_a.json` sha256
`2156db3ca9dc0474fddbc90d981a983d70f5479cf569c2757164ce2f35c38dbd`, `preflight_report_b.json`
`769d0f66f778e3f5c807256e08d5adc9c689c147c699c4018cfd91e0149a58c6`. No size, headroom, dose-side or
power refusal for either arm.

| refusal | arm | argued |
|---|---|---|
| `trained` (1 value < 20) | both | V is one number per model; no per-unit population exists. Foreseen ("The gate"); G1–G3 stand in |
| `floor` (1/41 → E 3.2 < 20) | both | S2 computes no E: tier 1, unregistered, a cell's label has false rate 2/41 by exchangeability and R1 is a count of cells. Nothing here is scored as a factor |
| `band:controls` (base 59 outside [31, 39]) | a | the controls are the null for "an edit of this size here", not for "no edit": a random edit of arm a's size moves V by about 24 by itself. That is what R1 reads against, and the gain arm states it beside. It is a moved mark by construction, declared here |

**G0's (b) read-back, revised (proposed in the freeze, Blocked 87).** "The read-back K′ has the
arm's eigenvalues (b)" becomes: the float32 read-back map equals the exact K′ (1e-6 of ‖K‖) and the
exact core has the arm's eigenvalues, real (1e-6); the float32 core's eigenvalues are recorded
beside. Why: (b-schur) leaves near-repeated eigenvalues, which float32 rounding moves by about √eps
(3.1e-4 ‖K‖ at 8000) while the map is exact to 3e-8. The revised check passes.

**Blocked 87, the freeze (the user's).** At 8000 both arms move the stream further than all 40
controls (a: D 0.949 against 0.693–0.735; b: 0.110 against 0.100–0.106), so the displacement guard
as frozen would write every such cell "moved further" and read nothing.
(a) Freeze as revised, with the matched-distance fallback moved from unit 3 ("if the budget
allows") into unit 2 for both arms at every scored step: each arm's controls also drawn at 1.5, 2
and 3 times its size (40 each), and a cell the arm moves further than its controls is labelled at
matched D by linear interpolation of the controls' V (or G) against D (OV1d's method), a cell the
arm moves less than its 1× controls read as now. About 240 more configurations per step (≈ 5 min
per step on the GPU at unit 1's 1.3 s per configuration, ≈ 25 min over the 5); units unchanged.
(b) Freeze as written (with the G0 revision): "moved further" cells are not read; the fallback
stays in unit 3 if the budget allows.
(c) Hold the freeze and change the controls instead (e.g. match D rather than norm at 1×).
**Recommended: (a)**: the guard exists so that "moved more" is not read as "this structure"
(OV1's lesson), and matched D keeps that while giving the reading a chance; (c) reopens the
controls' definition after seeing 8000.

## Revised after review (`/challenge-pr` on #246, 2026-10-21, at `719eb50`)

| finding | change |
|---|---|
| 1. arms shrink the head's output while their controls inflate it (‖sym K‖ ≈ 0.71 ‖K‖ against ‖K + N‖ ≈ 1.2 ‖K‖) | controls doubly matched (C8: same size, same edited-head norm); gain arms g_a, g_b, and a label is read only beyond its gain arm |
| 2. (b-eig) is not "radial part kept" in the residual's metric; ‖K′ − K‖ ≈ 1.6 ‖K‖ at 512 | the prose says where it holds; (b-schur) added (C7) and recommended; every arm's sizes recorded in G0 before any reading; the choice is Blocked 86 |
| 3. G1 cannot fail for the reason that matters | G1 kept as the labeller's check; G1′ (the gain arms) added as the null of the arm's form |
| 4. G3 calibrates the whole model, not an OV-only edit | G3′ added |
| 5. the hook formula applied Kᵀ | row-wise X̂ ↦ P (X̂ Z) Mᵀ Zᵀ |
| 6. G2 may fail by design at four heads | ten heads from 7d's catalogue, named here before any run |
| 7. C6 had no baseline | added: 59 of 200 unsymmetrised against 53 symmetrised |
| the author's question 3: the preflight after the freeze | moved before the freeze (unit 1); the freeze follows its report |
