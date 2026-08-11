# Update Plan — Revision 2

Supersedes the original. Cost tags unchanged: **[R]** report-only, data already on disk ·
**[W]** needs weights, no forward pass · **[F]** needs forward passes · **[D]** doc-only.

Status tags: **DONE** · **BLOCKED** (waiting on artifacts or a decision) · **OPEN**.

---

## 0. Where things stand

Sections 0–4 of the original plan are complete. Phases 1c and 2d are implemented, validated on
synthetic data and on configurations with known exact answers, and **not yet run against Pythia
artifacts** — by design, since the predictions were pre-registered and the artifacts are being
regenerated in this same cycle.

Nothing below has been adjudicated. No result in this project has changed. What has changed is
that six predictions are registered, the citations are correct, and four measurements that were
described as "we have never done this" now exist as code.

**Seven things found during implementation contradict the original plan.** They are collected in
§8 rather than scattered, because several change what should be run and in what order.

---

## 1. Completed

### §0 — citation corrections · **DONE** · [D]

Wider than the original table. The same errors appeared in four files the plan did not list,
including **three figure suptitles** (`pair_comparisons.py`, `checkpoint_sweep.py`,
`checkpoint_heatmaps.py`) — so every energy-trajectory PNG in the project has the wrong
citation baked into the image, and §0 is not purely doc-only in effect. Those figures need
regenerating; no data changes.

`core/config.py`, `core/models.py` and `plots.py` attached the fp32 precision guard to
"status-1's Thm 6.1 falsification". Doubly wrong: `eig_frac_pos_real` and friends say nothing
about dimension. Retargeted to what they actually decide — the attractive/repulsive call
(§3.2, §9.1) and which row of Table 1 (§9.2) a head falls under. The guard stands; it guards a
different claim.

**Left alone, out of phase scope, listed for a decision:** `status-2.md:74,231`; and a separate
error in Phase 5 — `v_alignment.py:185-196`, `run_5.py:535-542`, `design-5.md:57`,
`status-5.md:14` cite **Thm 6.3** for an attractive-subspace-dominance prediction about
intra-cluster mass, which Thm 6.3 (the $d\ge n$ exponential-rate result) does not make.
`run_1b.py:846` and `design-1b.md:14` call cone collapse "Theorem 6.3" when it is Lemma 6.4 —
that one is Phase 1b and cheap to fold in.

### §1 `status-1.md` · **DONE** · [D]
### §2 `design-1.md` · **DONE** · [D]
### §3 `PREDICTIONS.md` · **DONE** · [D]

P-γ1, P-γ2, P-H1, P-S1, P-T1, P-M1 registered with falsifier, instrument and cost.
**One amendment outstanding — see §8.1.**

### §4 — metric and reporting fixes · **DONE except two** · [R]

All implemented. Two carry-overs:

| Item | Why not done | Status |
|---|---|---|
| Re-derive `DEGENERATE_RANK_THRESHOLD` and `FIEDLER_ACTIVE_RANK_THRESHOLD` on the normed scale | Needs the actual normed-rank distribution across the sweep | **BLOCKED** on artifacts |
| Persist per-head Fiedler | Fix is in `p1_io._save_sinkhorn`, but existing artifacts do not contain the values | **BLOCKED** — D2 costs exactly one rerun |

The duplicate `ENERGY_VIOLATION_REL_TOL` in `checkpoint_scalars.py` was a hand-synced literal
with a comment asking editors to remember. It now parses the constant out of `core/metrics.py`
with `ast` — no torch import, and it raises at import if the constant is renamed rather than
falling back to a stale value.

### §5 — Phase 1c · **DONE, all six sub-experiments** · plus driver and IO

`p1c_frames/`: `gamma_ode.py`, `integration_time.py` (A), `gamma_null.py` (B), `moments.py` (C),
`frame_table.py` (D), `hemisphere_feasibility.py` (E), `design_test.py` (F), `run_1c.py`,
`p1c_io.py`, `design-1c.md`, `status-1c.md`.

Validation that actually constrains: the ODE reproduces all 28 entries of `MATH.md` §3.2's
collapse-time table to 0.005; the step estimator recovers an injected Euler step exactly; the
Gegenbauer code recovers $t=3$ for the octahedron and $t=5$ for the icosahedron; Wendel
reproduces the textbook $n{=}3,d{=}2\to0.75$; the cone margin gives $\cos30°$ on a $30°$ cone.

### §6 — Phase 2d · **DONE, all four sub-experiments** · plus driver and IO

`p2d_operator_activation/`: `gradient_flow_condition.py` (D1), `operator_pairing.py` (D2, D4),
`table1_predictions.py` (D3), `p2d_io.py`, `run_2d.py`, `design-2d.md`, `status-2d.md`.

The join is isolated in `p2d_io.py` because it is the only genuinely dangerous step in the
phase: three guards (revision, width, missing $W_Q/W_K$), all refusing rather than degrading.

---

## 2. Outstanding work, in order

### A. Amend P-T1 · **DONE** · [D]

Dated addendum in `PREDICTIONS.md`, not an edit — it was pre-registered. Adds the second row-2
condition, equal spacing, a required control arm, and bandwidth-stability adjudication.

### B. Wire `run_2d.py`'s LN frame · **DONE** · [W]

`p2d_io.resolve_ln_params` loads the model *at the operators' revision* and resolves the frame
through `core.ln_frame.frame_for_hidden_state`. The extraction convention is passed, not
guessed (`--keep-embedding`, `--last-is-post-final-ln`), and the driver prints which state
indices resolved to the identity frame.

### C. Feed P-M1 its violation counts · **DONE** · [R]

`run_2d.violation_counts` derives them from `energies.json` with the shared relative rule. One
correction in the doing: a violation is an event between two adjacent layers, so the series is
a per-boundary **indicator**, not a per-layer count. See §5.9.

### D. The $\beta_{\rm eff}$ head-to-layer reduction · **DE-BLOCKED** · [R]

**The choice still matters, unlike the clusterer question — but it no longer has to be made
first.** Measured spread in $\gamma_\beta(T_{\rm eff})$ across $\beta\in[0.5,5]$ at
$T_{\rm eff}=3$: 0.89 at $n{=}20$, 0.62 at $n{=}128$, 0.26 at $n{=}467$ — larger than any
residual we could hope to measure.

But $\gamma_\beta(t)$ is **monotone in $\beta$**, verified over 984,246 grid points per model
($n\in\{5,20,64,128,467,512\}$, $t\in[0,8]$, $\beta\in[0.01,10]$): (SA) decreasing with **zero**
violations, (USA) *increasing*. So the per-head $\beta$ range brackets the null without any
reduction being chosen, and `p1c_frames/beta_reduction.py` reports a residual **bracket** rather
than a point estimate. Where the observed curve falls outside the envelope, the conclusion holds
for every reduction and the decision is moot; where it falls inside, the decision matters and
the band is the uncertainty. Wired into `run_1c`'s sub-experiment B.

Remaining dependency: `geometry.json` must carry `beta_eff_per_head`. If it carries only a
scalar `beta_eff`, the residual is a point estimate with an unreported error bar, and `run_1c`
says so in the artifact.

### E. Check artifact readiness across the sweep · **TOOL BUILT** · [R] · run it

`tools/preflight_1c.py` answers this and four adjacent questions in one pass, reading only file
headers and JSON keys — no activations loaded. Reports capability coverage, runnable count per
sub-experiment with the blocking reason, and the consequence of each gap. Exits non-zero when
`norms` or `beta_eff` is incomplete, so it can gate a scheduling script rather than only inform
a reader.

It surfaced a **third blocker not previously on this list**: Phase 2d's join needs the revision,
and D2/D3/D4 additionally need the extraction convention. Both are already in
`p1_io._PROVENANCE_FIELDS` — see §5.10.

### F. Fix one clusterer for sub-experiment F · **DONE** · [D + R]

Resolved differently than expected. The premise — that $m$ moves the reference — was measured
and does not hold: the matched-$(m,d)$ baseline keeps $Q_k/Q_k^{\rm random}$ flat at 1 across a
32× range in $m$. F is now wired into `run_1c` with `--f-method` (default kmeans, the only arm
whose centroids Phase 1 persists). `p1c_frames/centroids.py` adds the effect-size floor P-S1 was
missing. See §5.8.

### G. Run 1c-A and 1c-B · [R] · **then re-read §1's verdict table**

### H. Run 1c-C, E · [R] · then 1c-D · [W] · then 1c-F once F above is decided

### I. Run Phase 2d · after 1c-B, per the original sequencing

### J. Regenerate the energy-trajectory figures · [R]

Three suptitles carried the wrong citation into every PNG.

### K. Fold the [F] item into the next forward-pass run

Per-head Fiedler persistence. Do not schedule a rerun for it alone.

---

## 3. Revised sequencing

1. **Run `tools/preflight_1c.py` first.** It is seconds of work and it determines which
   sub-experiments are runnable on which part of the sweep. Everything below is conditional on
   its output.
2. Then 1c-A and 1c-B, reading the residual **bracket** rather than a point estimate.
3. Then the rest of 1c; then 2d, which additionally needs the revision and the extraction
   convention in `geometry.json`.

The $\beta_{\rm eff}$ reduction no longer gates anything — the envelope replaces the decision
wherever the observed curve falls outside it.

---

## 4. Explicitly not doing yet

Unchanged from the original: items 9–11 (the 1.4B sweep, still gated behind claim (c)); anything
needing new checkpoints; BBGKY (§8), diffusive regularization (Problem 7), the singular
$\beta\to\infty$ limit (§9.3).

Added: the Phase 5 Thm 6.3 mis-citations (§1 above) — real but out of the phase at hand.

---

## 5. Findings that changed the plan

These are not results about Pythia. They are things the plan asserted or assumed that turned out
to be wrong, found by implementing and testing.

### 5.1 P-T1 as registered omits half of Table 1's row-2 hypothesis

Row 2 requires $\langle Q\varphi_1, K\varphi_1\rangle > 0$ — i.e. $\varphi_1^\top M\varphi_1 > 0$
— **in addition to** $\lambda_1(V) > 0$ simple. Testing without it would falsify a claim the
paper does not make: structurally the same error as the "Thm 6.1 unsupported" verdict row, made
again in the same document that retracted it. The code checks both and labels
`row2_eigen_only_qk_fails` distinctly. Because P-T1 was pre-registered, this needs a **dated
addendum**, not a silent correction.

### 5.2 `MATH.md` §8's step-size definition understates $T_{\rm eff}$ by ~5.7×

§8 writes $h_\ell = \|P^\perp(\Delta x_\ell)\|/\|x_\ell\|$, which is the sphere *displacement* —
the numerator of the Euler step. It equals the step size only if $\|\mathcal{X}\| = 1$, and the
paper's bound is $\le 1$ with equality only at full collapse. Measured against an injected
$h = 0.0200$: `h_calibrated` recovers 0.0200, §8's form reads 0.0035, the field runs at 0.176.

**P-γ2 predicts $T_{\rm eff} \ll t^\ast$, which the §8 definition makes nearly true by
construction**, in the direction that would make Blog 1's headline an artifact of depth. Three
definitions are computed; `verdict()` refuses a call when they straddle $t^\ast$.

### 5.3 The calibrated step makes the residual rate-invariant — a better result than planned

Damping the field 0.3× gives residual $-0.0009$: correctly, because damping is slower
integration, not resistance, and $T_{\rm eff}$ absorbs it. So the residual measures whether the
network moves in a **different direction** from the identity-weight field, not how much of it it
applies. A network that merely attenuates attention reads zero and should. **P-γ1's reading
should be restated in these terms.**

### 5.4 The $\beta = 5$ energy column cannot be dropped

§4 said to replace four $E_\beta$ columns with the cumulant ladder. Reconstruction error is
0.00% / 0.07% / 0.80% at $\beta = 0.1/1/2$ and **26.6% at $\beta = 5$**; twelve moments are
needed there, not three. Three of four columns are redundant; the fourth is not.

Relatedly: computing the ladder from `ip_histogram` needs the off-diagonal→full conversion.
At $n = 20$ the naive off-diagonal $\kappa_1$ reads $+0.0030$ against a true $+0.0523$ — an
order of magnitude, and precisely on the prompt where status-1's only $\beta$ gradient lives.

### 5.5 Two peak-detection bugs, opposite in direction

`inner_product_modes` scored the **octahedron** — two distinct inner products, the sharpest
configuration in $\mathbb{R}^3$ — as *unimodal*, because an interior-only local-max scan drops
the $-1$ peak sitting in bin 0, and sharp configurations put their mass at the boundaries. The
same scan found five modes in 200 i.i.d. points.

Separately, `projection_modality` scored a plain Gaussian at **nine** modes and a genuinely
trimodal cloud at four. Replaced with a KDE plus a bandwidth-stability scan: any distribution
can be made unimodal by over-smoothing and multimodal by under-smoothing, so a modality claim at
a single unstated bandwidth is not a measurement. **P-T1 should be adjudicated on
`stable_n_modes` only.**

### 5.6 A wrong trace contraction that the $M = I$ anchor could not catch

D2's $\mathrm{tr}(M^\top CMC)$ was implemented as `sum((C@M)*(C@M.T))`, which contracts to
$\mathrm{tr}(CMMC)$ — a different quantity that **coincides at $M = I$ and at any symmetric
$M$**. The sanity anchor passed while the value was wrong for every real head. Caught only
because a derived quantity came out negative, which is impossible:
$\mathrm{tr}(M^\top CMC) = \|C^{1/2}MC^{1/2}\|_F^2 \ge 0$. On a generic $M$: $-72.08$ against a
true $+167.00$. Wrong in three places.

**General lesson: an anchor that only tests the identity case tests almost nothing about a
bilinear form.** Every anchor in this project should have a non-symmetric arm.

### 5.10 (SA) and (USA) respond to $\beta$ in opposite directions

$\gamma_\beta(t)$ is monotone decreasing in $\beta$ under (SA) and monotone **increasing** under
(USA) — zero violations vs ~35% of grid points, over 984k points each. The partition function is
what reverses the sign. So using the surrogate as a stand-in for the normalized dynamics gets
the *direction* of the $\beta$-dependence backwards, not merely the magnitude, and the envelope
endpoints swap between models. This is a sharper version of `MATH.md` §3.2's observation that
the two separate at large $\beta$ and small $n$.

### 5.11 The extraction convention is recorded and was being asked for by flag

`p1_io._PROVENANCE_FIELDS` already writes `revision`, `checkpoint_step`,
`hidden_state_0_is_embedding` and `final_hidden_state_is_post_ln` at `geometry.json`'s top
level — precisely so downstream code need not be told. `run_2d` was taking the convention as
command-line flags anyway, which is the error class those fields exist to prevent: get either
wrong and $M_h$ is applied in the wrong frame, silently. It now reads them via
`p2d_io.extraction_convention`, ignores the flags when the artifact records them (saying so),
and refuses without `--assert-convention` when it does not.

### 5.8 The clusterer premise for sub-experiment F was wrong, and P-S1 had no effect-size floor

F was left unwired because the centroid count $m$ was assumed to move the random baseline.
Measured: $Q_k/Q_k^{\rm random}$ is flat at 1 for $m$ from 4 to 128, and a regular simplex stays
at 0.000 at every $m$. The clusterer must be fixed for a clean comparison but not by matching
$m$.

The same measurement supplied what P-S1 was missing — a threshold. The random ratio's 2σ band is
0.16–0.19 at $k=1$, 0.013–0.015 at $k=2$, 0.002 at $k=3$. Without it, random-vs-random returns a
PARTIAL verdict on nothing.

**And it corrected a claim I had written into the code.** I asserted that discriminating power
sits at low $k$, because the simplex's $Q_2$ ratio is $\approx 0.98$ and looks like no signal.
Wrong: the $k=2$ band is 0.014, so 0.023 is outside it. Higher degrees are *more* sensitive in
relative terms — both deviation and noise shrink with $k$, at different rates, which is why a
fixed absolute tolerance is wrong in a different direction at every degree.

### 5.9 A violation "count per layer" is a category error

A violation is an event between two adjacent layers, so there is exactly one per boundary and
the series is an indicator, not a count. Correlating a per-layer regime score against it is
correlating against a boolean, and P-M1's input now returns as one. Layer 0 is zero by
construction; reported rather than dropped, since dropping it would misalign the regime series.

### 5.7 The cone condition is nearly vacuous as a boolean

Wendel gives probability 1 for $d > n$, which every prompt satisfies, so P-H1 is close to
guaranteed as stated. But the margin *shrinks* as $n \to d$: on i.i.d. clouds at $d = 1024$ it
is 0.221 at $n{=}20$ and 0.030 at $n{=}512$. The reportable object is the margin and the layer
at which it first crosses zero — not the boolean. Similarly, $Q_k$ cannot be compared against a
fixed tolerance: $E[Q_k] = 1/n$ exactly for i.i.d. points, so every large-$n$ configuration
reads as a spherical design under an absolute threshold. P-S1 is adjudicated on the ratio to a
matched random baseline.

---

## 6. Standing rules this cycle established

Written down because each came from a defect that cost real work.

1. **If a quantity appears in a report, it is persisted.** D2's per-head Fiedler existed only in
   the session that produced it.
2. **Every data-dependent fallback records the branch it took.** On a model where no eigengap
   ever exists, the fallback *is* the metric.
3. **Every gate records which quantity it read and whether it passed**, per layer. A gate
   reading a constant that may since have changed cannot be reconstructed from the artifact.
4. **Refuse rather than degrade.** No unit-norm substitute for missing norms, no inferred
   revision, no invented $\beta$, no silent raw-frame fallback. A number from mismatched inputs
   is worse than no number: it is unfalsifiable from the output alone.
5. **Anchors need a non-symmetric arm.** See §5.6.
6. **A threshold that has not been derived from a distribution is labelled as placed, not
   calibrated** — in the code, next to the value.
