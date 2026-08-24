# Phase 2d — MATH (study notes)

## 0. What this document is

Companion to `math-2.md` and `math-2b.md`. **Read `math-1.md` §1A.6 first** — the reweighted-metric
gradient-flow condition derived there is the subject of this phase's first sub-experiment.

Phase 2d is where the paper's *hypotheses* finally get checked. Phase 1 reports energy-monotonicity
violations rising from 3 at step 0 to 64 by step 512 and reads that as the theorem failing. **But
the theorem has hypotheses, and this project had never checked whether the model satisfies them.**
Phase 2d checks them, and converts three more of the paper's results into falsifiable claims about
quantities already on disk.

It needs $M_h = W_Q^{(h)\top}W_K^{(h)}/\sqrt{d_{\text{head}}}$ and $W_{OV}^{(h)}$ — Phase 2's
output — which is why it is a Phase 2 extension rather than part of Phase 1c (deliberately
Phase-2-independent so it can run now).

**Sequencing: 2d runs after 1c-B lands.** If $T_{\rm eff}\ll t^\ast$, the network never integrates
far enough for the asymptotic energy argument to bind, and attributing a monotonicity break becomes
attributing something that was not going to happen anyway. The $T_{\rm eff}$ result determines
whether the break is even the right thing to explain.

Everything here is validated on synthetic data and on configurations with known exact answers.
**None of it has been run against Pythia artifacts.**

---

## 1. D1 — the gradient-flow condition (P-M1)

### 1.1 The condition, and why it reframes the question

From `math-1.md` §1A.6: (SA) is *not* a Wasserstein gradient flow in the standard metric — its
field is a logarithmic derivative (eq. 3.7). It **is** a gradient flow in the reweighted metric

$$
\langle a, b\rangle_X = \sum_i Z_{\beta,i}(X)\,\langle a_i, b_i\rangle
$$

under a standing condition the paper states once and never revisits:

$$
\boxed{\ Q^\top K \ \text{symmetric}\quad\textbf{and}\quad V = Q^\top K\ }
$$

A head satisfying both is inside the gradient-flow regime and **must** show monotone $E_\beta$. A
head far outside carries **no monotonicity guarantee at all** — in either direction.

So the sharp question is not *"is the theorem violated"* but *"which heads are outside its
hypotheses, and do the violations localize there?"* **That converts a falsification into an
attribution**, which is a strictly more informative experiment. It is prediction **P-M1**.

### 1.2 What is measured, per head

$$
\texttt{asymmetry} = \frac{\lVert \mathrm{Skew}(M_h)\rVert_F}{\lVert M_h\rVert_F},
\qquad \mathrm{Skew}(A) = \tfrac12(A - A^\top)
$$

$$
\texttt{ov\_qk\_align} = \frac{\langle W_{OV},\ M_h\rangle_F}{\lVert W_{OV}\rVert_F\,\lVert M_h\rVert_F}
$$

**Calibrating `asymmetry` is worth doing by hand**, because the interesting reading is relative to
chance rather than to zero. Since $\lVert M\rVert_F^2 = \lVert\mathrm{Sym}(M)\rVert_F^2 +
\lVert\mathrm{Skew}(M)\rVert_F^2$ (the two parts are orthogonal in the Frobenius inner product),
and for a matrix with i.i.d. entries the symmetric and antisymmetric parts carry equal expected
energy:

| value | meaning |
|---|---|
| $0$ | perfectly symmetric — **condition 1 satisfied** |
| $\approx 1/\sqrt2 \approx 0.707$ | **a generic random matrix** |
| $1$ | perfectly antisymmetric — pure rotation |

So "is this head symmetric?" is really "is `asymmetry` materially below $0.707$?", and a head at
0.7 is not *somewhat* symmetric — it is indistinguishable from random. This built-in null is a
nice contrast with the phases that lacked one (`math-1b.md` §6, `math-2.md` §10.3).

### 1.3 Three design points that are not obvious

**The cosine is signed, deliberately.** It is scale-invariant, so it tests $V \propto Q^\top K$
rather than equality — the right relaxation, because *a positive rescaling of $V$ rescales time in
the ODE* without changing the gradient-flow structure or the sign of $dE/dt$. A **negative**
constant does change it: that is exactly the $V = -I_d$ repulsive case (`math-1.md` §1.4). So
`ov_qk_align` near $-1$ is recorded as its own regime, `repulsive_aligned`, **where a violation is
the *predicted* behaviour rather than an anomaly** — not lumped in with "aligned", and not lumped
in with "fails the condition" either. Taking $|\cos|$ would have destroyed exactly the distinction
the phase exists to draw.

**Alignment is reported against $M$ and against $\mathrm{Sym}(M)$ separately.** If $M$ is far from
symmetric then *no* symmetric $V$ can match it, so a head can fail condition 2 either because $V$
is wrong or because $M$ was never symmetric in the first place. Those are different failures with
different implications, and the plain cosine conflates them.

**A continuous `regime_distance` alongside the four-way label.** P-M1 is a *correlation* claim, and
a two-bin split discards most of the statistical power. This is the same lesson as
`math-1.md` §15.6 (the plateau vote's discarded gradation) and `math-1b.md` §7.1
(`normalized_margin` vs the regime label): **thresholding early throws away the signal the test
needs.**

### 1.4 The limitation, stated rather than resolved

D1's correlation is between a **per-layer** violation count and an aggregate of **per-head** regime
scores — because there is no per-head energy. **The aggregate is a choice, and the answer depends
on it.** Mean, min, and max are all reported, and

> if they disagree in sign then P-M1 is not adjudicable from per-layer energies and needs head
> ablation.

That is a real result about the experiment's resolution, and recording it as one — rather than
resolving it by picking whichever aggregate confirms — is the right call. Note also
(`math-1.md` §5.4) that a violation is an event *between adjacent layers*, so the target series is a
per-boundary **indicator**, not a count: D1 is correlating a continuous per-head aggregate against a
boolean.

---

## 2. D2 — operator-conditioned rank

### 2.1 The derivation

The paper's $E_\beta$ assumes $Q^\top K = I$. The model's own coupling is $M_h$, acting on the LN'd
states $y$ that attention actually reads (`math-1.md` §3.3):

$$
E_\beta^{(h)} = \frac{1}{2\beta}\Big\langle \exp\big(\beta\, y_i^\top M_h\, y_j\big)\Big\rangle
$$

Expanding the exponential as in `math-1.md` §5.2, but now with the operator inside:

$$
\textbf{first order:}\quad \big\langle y^\top M y'\big\rangle = \bar y^\top M \bar y
$$

a quadratic form of the QK operator **at the centroid**. Negative when the centroid overlaps
$\mathrm{Sym}(M)$'s negative eigenspace. *This is `V_repulsive_local` as a scalar on the same axis as
$E_\beta$* — which is what makes the Phase 2 verdict and the energy comparable at all, rather than
two parallel stories.

$$
\textbf{second order:}\quad \big\langle (y^\top M y')^2\big\rangle = \mathrm{tr}\big(M^\top C M C\big)
= \sum_{a,b}\lambda_a\lambda_b\,\big|\langle u_a,\ M u_b\rangle\big|^2
$$

with $C$ the token covariance and $(\lambda_a, u_a)$ its eigenpairs.

**That second-order term is the bilinear pairing of the operator spectrum against the activation
covariance spectrum. Phase 2 has the left factor. Phase 1 has the right factor. Nobody had computed
the product.** That sentence is the phase's reason to exist.

It defines an **operator-conditioned rank**:

$$
\boxed{\ \mathrm{PR}_M = \frac{(\mathrm{tr}\,MC)^2}{\mathrm{tr}(M^\top C M C)}\ }
$$

— how many of the cloud's directions this head actually couples.

### 2.2 The anchor, and the companion

**Sanity anchor:** at $M = I$, $\mathrm{PR}_M$ reduces to $(\mathrm{tr}\,C)^2/\mathrm{tr}(C^2)$ —
the ordinary participation-ratio rank of the activation cloud, i.e. exactly `math-1.md` §5.2's
$\mathrm{PR} = 1/m_2$. Asserted in tests rather than assumed; agrees to $10^{-8}$. §2.4 explains why
that anchor was not enough.

**Companion:** $\texttt{coupling\_efficiency} = \mathrm{PR}_M/\mathrm{PR}_C$, because $\mathrm{PR}_M$
alone conflates *"the head is selective"* with *"the cloud is low-rank"* — and separating those is
the entire point of the pairing.

### 2.3 The hypothesis it targets

**Heads with large $\lVert M\rVert$ and small $\mathrm{PR}_M$ are strong operators pointed where the
tokens are not.** That is a candidate explanation for a specific unexplained Phase 1 observation:
**the $\beta$-independence of violations after step 512.**

The mechanism: if $M$ concentrates on few directions, the higher moments of $y^\top My'$ collapse and
only $\langle G\rangle$ survives — and by the MGF expansion (`math-1.md` §5.2), an energy dominated
by its first moment is insensitive to $\beta$, since $\beta$ enters only through the higher terms.
**Exactly as observed.** This is a genuine, mechanistic, testable explanation for a Phase 1 anomaly,
derived rather than guessed.

### 2.4 The trace contraction bug — the most valuable lesson in the phase

$\mathrm{tr}(M^\top C M C)$ was implemented as `sum((C@M)*(C@M.T))`, which contracts to
$\mathrm{tr}(CMMC)$ — **a different quantity that coincides with the intended one at $M = I$.**

So **the sanity anchor of §2.2 passed while the value was wrong for every real head.** On a generic
$M$ the buggy form reads $-72.08$ against a true $+167.00$. It was wrong in three places.

It was caught only because a derived quantity came out **negative**, which is impossible:

$$
\mathrm{tr}(M^\top C M C) = \mathrm{tr}\big((C^{1/2}M^\top C^{1/2})(C^{1/2}MC^{1/2})\big)
= \big\lVert C^{1/2} M C^{1/2}\big\rVert_F^2 \;\ge\; 0
$$

using $C = C^{1/2}C^{1/2}$ for the symmetric PSD covariance. The correct contraction is
`((C@M) * (M@C)).sum()`, and the non-negativity is now a runtime guard rather than a hoped-for
property.

**The general lesson, stated in `UPDATE_PLAN.md` §5.6 and worth carrying everywhere:**

> **An anchor that only tests the identity case tests almost nothing about a bilinear form.
> Every anchor in this project should have a non-symmetric arm.**

$M = I$ is symmetric, commutes with everything, and is invariant under transpose — so it cannot
distinguish $M^\top$ from $M$, nor $MC$ from $CM$. A single non-symmetric, non-commuting test matrix
would have caught this immediately. This generalizes past this project: **identity anchors are
nearly free of diagnostic power for anything bilinear**, which is most of numerical linear algebra.

---

## 3. D3 — Table 1 as a geometric prediction (P-T1)

### 3.1 What Table 1 says, and why row 2 is the sharpest prediction available

Table 1 (§9.2, from [GLPR24]) maps a classification this project **already has** — the sign and
multiplicity of $\lambda_1(V)$ per head, from Phase 2 — onto a statement about activations
(`math-2.md` §1.3). Row 2:

$$
\lambda_1(V) > 0 \text{ real, simple}\quad\textbf{and}\quad \langle Q\varphi_1, K\varphi_1\rangle > 0
\;\Longrightarrow\; \text{three parallel hyperplanes normal to } \varphi_1
$$

i.e. the scalar projection $\langle\varphi_1, x_i\rangle$ should be **trimodal**. It is the
sharpest falsifiable prediction anywhere in the paper, and **it costs a projection and a
histogram.**

### 3.2 The amendment: P-T1 as registered omits half the hypothesis

The registered wording carried only the condition on $V$. Table 1's row 2 requires a second
condition, on $QK$:

| | condition | in the registered wording? |
|---|---|---|
| on $V$ | $\lambda_1(V) > 0$, simple | yes |
| on $QK$ | $\varphi_1^\top M_h\varphi_1 > 0$ | **no** |

A head with a positive simple top eigenvalue but a negative QK form **is not in row 2 at all**, and
testing it against row 2's conclusion would falsify a prediction the paper does not make — *the same
structural error as the retracted "Theorem 6.1 unsupported" verdict row*, made a second time in the
same document that retracted the first. That it was recorded rather than quietly fixed is the right
call.

Because P-T1 was pre-registered, this required a **dated addendum**, not an edit. The amended
version is strictly *harder* to satisfy: two operator conditions instead of one, plus approximately
equal mode spacing (the prediction is three *parallel* hyperplanes, so spacing regularity is part of
it and the original wording omitted that too), plus a control arm, plus bandwidth stability.
`row2_eigen_only_qk_fails` labels heads that would have counted under the original wording, **so the
size of the error is recoverable from the output.**

### 3.3 The rescaling caveat is also the falsifier

Table 1 describes the limit geometry of $z_i = e^{-tV}x_i$, **not** of $x_i$ — and $t$ is not
observable from a fixed-depth network. So a direct test on raw activations tests a related but
weaker claim.

The honest procedure, and what the code does: test the raw projection; report unimodality as
evidence about the **transfer** rather than about the theorem; and separately report
`rescaled_modality` at several candidate $t$. The reading:

> If trimodality appears at some $t > 0$ and not at $t = 0$, **the structure is real and the
> rescaling is what hides it** — a different conclusion from "Table 1 does not transfer," and one
> the raw test cannot reach.

**The eigenvector condition number is reported per $t$**, because the amplification grows with $t$
and a rescaling through an ill-conditioned eigenbasis produces noise that looks like structure. (This
is the same non-normality hazard as `math-2.md` §3.1 — $V$'s eigenvectors are not orthogonal, so
$e^{-tV}$ can be badly conditioned.) Note also `math-2.md` §10.2: the *same* $t$-scale question
applies here, and Phase 1c's $T_{\rm eff}$ is the principled candidate set.

### 3.4 Two peak-detection bugs, opposite in direction

Both are worth knowing because they are archetypal.

**`inner_product_modes` scored the octahedron as *unimodal*.** The octahedron has exactly two
distinct pairwise inner products — it is the *sharpest* configuration in $\mathbb R^3$ (a sharp
configuration in the sense of `math-1c.md` §8). The scan was an interior-only local-maximum search,
which **drops the $-1$ peak sitting in bin 0**: sharp configurations put their mass at the
*boundaries* of the inner-product range, which is precisely where an interior-only scan cannot look.
The same scan found **five modes in 200 i.i.d. points.**

**`projection_modality` scored a plain Gaussian at *nine* modes and a genuinely trimodal cloud at
four.** Replaced with a KDE plus a **bandwidth-stability scan**, on the principle:

> Any distribution can be made unimodal by over-smoothing and multimodal by under-smoothing, so **a
> modality claim at a single unstated bandwidth is not a measurement.**

Hence: **P-T1 is adjudicated on `stable_n_modes` only** — the mode count that survives the bandwidth
scan — and **`None` (no stable count) is a legitimate outcome that must be reported as such rather
than resolved.** This is the same discipline as refusing a verdict when the three step-size
definitions straddle $t^\ast$ (`math-1c.md` §1.2).

### 3.5 The control arm

Trimodality rate is reported among row-2 candidates **and among non-candidates**. If non-candidates
are trimodal at the same rate, trimodality is a property of the activations rather than of the
classification, and **a candidates-only number would read as confirmation.** This is the cheapest
possible guard against the failure mode `math-2b.md` §3.2 catalogues, and it costs one extra
histogram.

---

## 4. D4 — the model's own energy

### 4.1 Replacing the proxy

$$
E_\beta^{(h)} = \frac{1}{2\beta}\Big\langle\exp\big(\beta\,y^\top M_h\,y'\big)\Big\rangle
$$

on LN'd states, with the first-order term $\bar y^\top M_h\bar y$ reported alongside the
identity-weight $E_\beta$. **The first-order sign *is* the attractive/repulsive call for that head**,
computed from the model's own operator rather than from the $Q^\top K = I$ proxy.

`monotonicity_compare` then does the thing that had never been done: count violations under **both**
energies and report the ones that exist under the proxy and **disappear** under the head's own
energy.

$$
\textbf{Each of those is a Phase 1 violation that was an artifact of the substitution } Q^\top K \to I.
$$

This is the most direct possible audit of Phase 1's headline, and it is [R+W] cost — weights plus
saved activations, no forward pass.

### 4.2 Two numerical points that carry meaning

**The generalized energy overflows without a shift.** At $\beta = 5$, $d = 1024$, with an untamed
$M$, $\exp(\beta\,y^\top My')$ exceeds float64 — **and it surfaces as `inf` in one head of one layer
rather than as an error**, which is how a NaN reaches an aggregate. The exponent is shifted by its
maximum and carried analytically (the same log-sum-exp discipline as `gamma_ode`'s overflow-safe
factoring, `math-1c.md` §2.2), with `overflow_guarded` flagged per $\beta$.

**Rows are normalized before the energy is computed.** The paper's $E_\beta$ is defined for unit-norm
particles; leaving the norms in makes the exponent scale with $\lVert y\rVert^2$, which for a
transformer's growing residual stream **would make the energy a norm measurement wearing a geometry
costume.** `norm_cv` records the size of what was removed — so the confound is quantified rather than
merely avoided. (Same concern, same resolution, as raw-vs-normed effective rank in
`math-1.md` §6.2.)

---

## 5. The join, and why it is the dangerous step

Combining Phase 2 operators with Phase 1 activations requires that they came from the *same model at
the same revision under the same extraction convention*. `p2d_io.py` isolates the join precisely
because it is "the only genuinely dangerous step in the phase," with **three guards, all refusing
rather than degrading**: revision mismatch, width mismatch, and missing $W_Q/W_K$.

**The extraction convention was being asked for by flag, which is the error class those fields exist
to prevent.** `p1_io._PROVENANCE_FIELDS` already writes `revision`, `checkpoint_step`,
`hidden_state_0_is_embedding` and `final_hidden_state_is_post_ln` at `geometry.json`'s top level —
*precisely so downstream code need not be told.* `run_2d` was taking the convention as command-line
flags anyway. Get either wrong and $M_h$ is applied in the wrong frame, **silently**
(`math-1.md` §2.2 for why the off-by-one is real). It now reads them via
`p2d_io.extraction_convention`, ignores the flags when the artifact records them (saying so), and
**refuses without `--assert-convention` when it does not.**

`run_2d.py` also resolves the LN frame properly: `p2d_io.resolve_ln_params` loads the model *at the
operators' revision* and resolves the frame through `core.ln_frame.frame_for_hidden_state`, printing
which state indices resolved to the identity frame. And `run_2d.violation_counts` derives P-M1's
target series from `energies.json` under the shared relative rule — with the correction that a
violation is a per-boundary indicator (§1.4).

---

## 6. Code map

| File | Sub-exp | Role |
|---|---|---|
| `gradient_flow_condition.py` | D1 | Per-head asymmetry, signed OV/QK cosine (vs $M$ and vs $\mathrm{Sym}(M)$), four-way regime label plus continuous `regime_distance` |
| `operator_pairing.py` | D2, D4 | Token covariance, $\mathrm{PR}_M$ with the non-negativity guard, `coupling_efficiency`, the model's own $E_\beta^{(h)}$ with overflow shift, `monotonicity_compare` |
| `table1_predictions.py` | D3 | `classify_ov_row` (both row-2 conditions), `row2_eigen_only_qk_fails`, projection histograms, KDE modality with bandwidth-stability scan, `rescaled_modality` with per-$t$ condition numbers, control arm |
| `p2d_io.py` | — | The join: three refusal guards, `extraction_convention`, `resolve_ln_params` |
| `run_2d.py` | — | Driver; `violation_counts` from `energies.json` under the shared rule |

---

## 7. Open questions

Tracked: nothing has been run against Pythia artifacts; 2d is sequenced after 1c-B; and D1's
aggregate-choice limitation (§1.4) may prove it un-adjudicable at per-layer resolution.

Surfaced by writing this document:

1. **D2's $\beta$-independence hypothesis is directly testable *within* D2, and the test is not
   currently specified.** §2.3 derives the mechanism — small $\mathrm{PR}_M$ ⟹ higher moments
   collapse ⟹ only $\langle G\rangle$ survives ⟹ $\beta$-insensitivity. That predicts a specific
   **correlation across layers**: layers whose heads have low $\mathrm{PR}_M$ should be the layers
   whose violation counts are flattest in $\beta$. Phase 1 already stores per-$\beta$ violation
   counts per layer, and the $\beta$-gradient it found (43/33/22/6 at steps 128–256, vanishing after
   512) is exactly the series to regress against. This turns a plausible story into a measurement
   with no new computation.

2. **D4's `monotonicity_compare` should also report violations that *appear* only under the head's
   own energy.** The design specifies counting violations that exist under the proxy and disappear
   under $E^{(h)}_\beta$ — artifacts of the substitution. The reverse set is equally informative and
   is not mentioned: **violations the proxy *misses*.** Those would be real monotonicity failures of
   the model's own dynamics that Phase 1's instrument cannot see, which is a stronger result than the
   artifact count (it would mean Phase 1 *understates* the phenomenon). Both directions come from the
   same comparison at no extra cost, and reporting only one is a directional choice that should be
   deliberate.

3. **The four-cell cross-tabulation from `math-2b.md` §8.4 is most naturally computed here.** D1
   partitions heads on symmetry of $M$; Phase 2 partitions eigenvalues on $\mathrm{sign}(\mathrm{Re}\,
   \lambda)$ of $W_{OV}$; Phase 2b partitions them on real-vs-complex. D3 additionally needs
   $\lambda_1(V)$'s sign and multiplicity. **All four classifications are computed in this phase or
   its inputs, and none of the pairwise cross-tabulations exist.** The specific one worth having:
   among row-2 candidates (D3), what is the `asymmetry` distribution (D1)? Row 2 requires
   $\varphi_1^\top M\varphi_1 > 0$, which is a statement about $\mathrm{Sym}(M)$ only — so a row-2
   candidate can be arbitrarily asymmetric and therefore arbitrarily far outside D1's gradient-flow
   regime. **Whether P-T1's candidates and P-M1's in-regime heads overlap at all is unknown, and the
   two predictions may be about disjoint populations.**

4. **`asymmetry`'s $1/\sqrt2$ null is a *pointwise* baseline, not a distributional one.** §1.2 gives
   the expected value for a random matrix, but not its spread — and at $d_{\text{head}} = 64$ or 128
   the concentration is tight but finite. A head at 0.68 is "below random" only relative to a
   standard deviation nobody has computed. It is $O(1/d_{\text{head}})$ by standard concentration and
   could be stated in closed form or sampled in one line; without it, `asymmetry` inherits the same
   "threshold not derived from a null" problem this project keeps rediscovering
   (`math-1.md` §8.3, `math-1b.md` §7.3, `math-1c.md` §8.6, `math-2.md` §10.3).

5. **D3 tests trimodality of $\langle\varphi_1, x_i\rangle$, but row 2 predicts *equally spaced*
   parallel hyperplanes, and spacing is a separate, sharper test.** The amendment records that
   spacing regularity is part of the prediction. Three modes at arbitrary positions is much weaker
   evidence than three modes at $\{-a, 0, +a\}$ — and the spacing test has a *much* better null,
   since random trimodality will not be equally spaced. Given how hard modality is to measure
   robustly (§3.4), **the spacing statistic may be the more reliable instrument of the two**, and it
   is currently the secondary one.

6. **Nothing checks the condition $Z_{\beta,i}$ actually weights.** §1.1's metric is
   $\sum_i Z_{\beta,i}\langle a_i,b_i\rangle$, and D1 tests the two conditions on $Q^\top K$ and $V$
   — but never looks at $Z_{\beta,i}$ itself, which is what makes the metric non-standard in the
   first place. This is the same gap flagged in `math-1.md` §15.12: the per-token partition function
   is a metric weight, it is one line from the Gram matrix, and no phase computes it. **D1 is the
   natural home** — a head can satisfy both operator conditions and still sit in a badly distorted
   metric if $Z_{\beta,i}$ has a wide spread across tokens, and that spread is exactly what an
   attention sink produces.
