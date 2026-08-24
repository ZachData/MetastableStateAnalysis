# Phase 2b — MATH (study notes)

## 0. What this document is

Companion to `math-2.md`, which ends at the observation that OV's Schur and symmetric
decompositions disagree — and that **when they disagree, rotation matters.** Phase 2b takes the
rotational component as its object.

It is also the most instructive phase in the project to read carefully, for a reason that has
nothing to do with transformers:

> **The phase's headline finding was withdrawn — not falsified by new data, but identified as an
> algebraic identity that was never falsifiable.** `elim_rotation = 0.0` in 35/35 runs, across
> every model, every $\beta$, encoder and decoder alike, was forced by construction before any
> data was read.

That is a better epistemic outcome than a wrong number, and the way it was caught (someone
noticed the operator was orthogonal) is worth internalizing. §3 does the algebra in full.

**Naming.** Directory `p2b_imaginary/`, documentation name "Phase 2b", legacy artifacts
`phase2i_*`. New output is `phase2b_results.json`; `p2b_io.refuse_legacy_run_dir` **raises** on a
directory containing the old names, because the violation-counting rule changed underneath them
(§6) — a nice example of refusing rather than silently reading incomparable data.

---

## 1. The question

Phase 2 found OV matrices are **~98% complex** — the overwhelming majority of eigenvalue
dimensions come in complex-conjugate pairs — across every model tested. That is a direct
challenge to Phase 2's own conclusions:

*If the dominant spectral feature of OV is rotational rather than signed, does rotation
contribute to energy violations? Or could the imaginary structure be confounding the
attractive/repulsive attribution entirely?*

Phase 2's whole mechanism (`math-2.md` §1.2) is a **sign** story: attractive subspace,
repulsive subspace, violations from the latter. A complex eigenvalue $\lambda = \rho
e^{\pm i\theta}$ has a real part with a sign, but its dominant action is *rotation* — and
rotation neither attracts nor repels. If 98% of the spectrum is doing something the
attractive/repulsive dichotomy has no vocabulary for, the dichotomy might be describing a 2%
residue while the real dynamics happen elsewhere.

---

## 2. The S/A decomposition

### 2.1 The split

Every real square matrix decomposes uniquely into symmetric and antisymmetric parts:

$$
V = S + A,\qquad S = \tfrac12(V + V^\top),\qquad A = \tfrac12(V - V^\top)
$$

- $S$ is symmetric: **real eigenvalues, orthogonal eigenvectors.** This is the component whose
  sign structure Phase 2 attributes violations to.
- $A$ is antisymmetric: **purely imaginary eigenvalues, generates rotations.** For any
  antisymmetric $A$, $e^{A}$ is **orthogonal** — the fact §3 turns on.

This is the natural decomposition here because it isolates exactly the two possibilities: either
the signed residue does all the causal work, or the rotational majority contributes too.

### 2.2 Real Schur blocks: what a rotation plane is

The real Schur form $V = ZTZ^\top$ (`math-2.md` §3.2) has $T$ upper *quasi*-triangular — meaning
the diagonal carries $1\times1$ blocks (real eigenvalues) and $2\times2$ blocks (complex
conjugate pairs). A $2\times2$ block

$$
\begin{pmatrix} a & b \\ c & a\end{pmatrix},\qquad bc < 0
$$

has eigenvalues $a \pm i\sqrt{-bc}$, i.e. modulus and angle

$$
\rho = \sqrt{a^2 - bc},\qquad \theta = \mathrm{atan2}\big(\sqrt{-bc},\ a\big)
$$

and the two corresponding Schur vectors span a **rotation plane**: a 2-dimensional invariant
subspace on which $V$ acts as a scaling by $\rho$ composed with a rotation by $\theta$.

Two implementation notes that carry real content:

- **$\theta$ must be on $[0,\pi]$, not folded to $[0,\pi/2]$.** The previous version used
  $\mathrm{atan2}(\sqrt{-bc}, |a|)$, so a *repulsive* rotation ($\mathrm{Re}\,\lambda < 0$,
  $\theta$ near $\pi$) was reported as its reflection. The sign survived in a separate field so
  nothing was lost — but `theta_mean` was **not the mean rotation angle**, which matters the
  moment $\theta$ is regressed against depth or training step. Fixed to
  $\mathrm{atan2}(\sqrt{-bc}, a)$.
- **The $2\times2$ detection threshold must be relative.** The test was
  $|T_{i+1,i}| > 10^{-10}$ absolute, on a matrix whose scale varies by orders of magnitude across
  layers and checkpoints. Now relative to $\lVert T\rVert_F$. Same defect class as every other
  un-rederived threshold in this project.
- **Planes are stored as $(d,2)$ orthonormal bases, never as $(d,d)$ projectors.** The previous
  version materialized 32 dense projectors plus two combined ones *per layer* and retained
  $T, Z$ on top: **~7 GB at $d{=}1024\times24$ layers, ~27 GB at $d{=}2048$.** Contracting
  through the basis is also $O(ndk)$ rather than $O(nd^2k)$. A memory bug, but one that made the
  phase unrunnable at the scale it was designed for.

### 2.3 Three different definitions of "rotational fraction"

The phase has reported all three as "how rotational $V$ is". They are not the same number.

| definition | formula | question it answers |
|---|---|---|
| **per-eigenvalue energy** (Henrici convention) | $\dfrac{2\sum_{2\times2}\rho^2}{\sum_{1\times1}\lambda^2 + 2\sum_{2\times2}\rho^2}$ | how much eigenvalue *energy* is rotational |
| **per-block** (the historical 84–97.5% figure) | as above but counting $\rho^2$ **once** per block | — (understates by ~2× relative to the first) |
| **relative eigenvalue criterion** | $\tfrac1d\#\{|\mathrm{Im}\,\lambda| > \texttt{tol}\,(|\mathrm{Re}\,\lambda| + \epsilon)\}$, $\texttt{tol} = 0.01$ | how many *dimensions* are complex |

The first two lived **in the same file** with the discrepancy unflagged: a $2\times2$ block holds
*two* eigenvalues and occupies *two* dimensions, so counting $\rho^2$ once understates rotational
energy by about a factor of two relative to the Henrici convention used elsewhere in the same
module. The rewrite makes per-eigenvalue the standard — it is the convention under which
$\texttt{complex\_energy} + \texttt{real\_energy} = \texttt{eigenvalue\_energy}$ is an **identity
rather than an approximation** — and keeps the old number under a name that says which
convention it is, so the historical figure stays checkable rather than silently restated.

The third is separately suspect: `core/precision_policy.py` flags (item P2) that a *relative*
imaginary-part criterion may be an **fp16-storage artifact** — an exactly-real eigenvalue pair
perturbed at fp16 epsilon splits into a complex pair with a tiny imaginary part, which a relative
criterion counts as rotation. At $\texttt{tol} = 0.01$ that is a low bar.

`dim_complex_fraction` ($2n_{\text{complex}}/d$) and `complex_energy_fraction` are now reported
side by side with distinct names, because *"how many dimensions rotate"* and *"how much energy is
in rotation"* are different questions that were previously printed adjacent without one.

### 2.4a The headline statistic may be about an object the model never forms

`head_circuits.py` makes an argument that undercuts §2.3's headline more sharply than any of the
convention problems: the summed operator $\mathrm{ov\_total} = \sum_h W_{OV}^{h}$ **is not a
thing the model ever forms.** Attention applies each head's $W_{OV}^h$ to that head's own
attention-weighted value sum; the heads' outputs are added *after* the operator acts, not before.
So the spectrum of the sum is not the spectrum of anything the forward pass computes, and
**"84–97.5% of OV's spectral energy is rotational" is a statistic of a fiction.**

The per-head numbers are the real ones, and they are very different: the same head measures 5.5%
of dimensions rotating in full space against 87.5% within its own core subspace. Any downstream
use of the aggregate figure — `math-5.md` §3.1's non-orthogonality argument, `math-6.md` §7.2's
dimension estimate — should be restated per head.

### 2.4 Henrici non-normality — and what it actually predicts

$$
\mathrm{henrici}(V) = \lVert T\rVert_F^2 - \sum_i |\lambda_i|^2
$$

Zero iff $V$ is normal, and otherwise a measure of how much the Schur blocks interact — the
scalar that says whether the S/A split is informative or decorative.

*(A common gloss says Henrici "equals the squared Frobenius norm of $T$'s strict upper
triangle." That holds for the **complex** Schur form, or a wholly real spectrum. It is **false
for the real Schur form this phase uses**: a standardized $2\times2$ block
$\left[\begin{smallmatrix}a&b\\c&a\end{smallmatrix}\right]$ contributes $(|b|-|c|)^2$ to
Henrici while $\mathrm{triu}(T,1)$ contributes $b^2$. Checked numerically: $d=6$ gives 12.26 vs
19.08, $d=10$ gives 58.28 vs 73.75. The same gloss appears in `rotational_schur.py`'s docstring
and should be corrected there too.)*

**Here is the connection the docs do not make, and it is the useful one.** Compute the
commutator of $V$ with its transpose:

$$
[V, V^\top] = [S+A,\ S-A] = -[S,A] + [A,S] = -2\,[S,A]
$$

So

$$
\boxed{\ V \text{ normal} \iff [V,V^\top] = 0 \iff [S,A] = 0 \;\Longrightarrow\; e^{-(S+A)} = e^{-S}e^{-A}\ }
$$

*(The last arrow is one-way as stated. $e^Xe^Y = e^{X+Y}$ implies $XY = YX$ only under spectral
side conditions — Wermuth's theorem, requiring eigenvalue differences to avoid $2\pi i\mathbb Z$ —
so "the exponentials agree" does not by itself force commutation. Everything below needs only the
forward direction.)*

**So Henrici and the full-vs-signed gap vanish together**: by Baker–Campbell–Hausdorff,
$e^{-S}e^{-A} = e^{-(S+A) + \frac12[S,A] + \cdots}$, so the leading discrepancy is
$\tfrac12[S,A]$, and $[S,A] = 0$ is exactly the normality condition Henrici measures.

**They are not proportional, though**, and it would be wrong to read Henrici as a calibrated
predictor of the gap's size. Measured on random matrices, $\mathrm{henrici}/\lVert[S,A]\rVert_F$
runs $\approx1.0$–$1.4$ at $d=4$, $1.5$–$2.0$ at $d=8$, $2.4$–$3.3$ at $d=16$ — it scales with
$d$ and with $\lVert V\rVert$. What survives is a **monotone conjecture** (larger Henrici,
larger gap), which is still worth testing (§8.1) but is a hypothesis rather than an identity.

This is a cheap, falsifiable, cross-block prediction that nobody has tested: **layers with near-
zero Henrici should show $\texttt{elim\_full} \approx \texttt{elim\_signed}$, and the gap between
the two frames should grow with Henrici.** Both quantities are already computed. Henrici is
weights-only, per-layer, and per-checkpoint free. See §8.

---

## 3. The withdrawal: why `elim_rotation = 0` was an identity

### 3.1 The construction, and the proof

Block 1b built three rescaled frames from $V = S+A$ and compared violation counts:

$$
z_{\text{full}} = x\,(e^{-V})^\top,\qquad
z_{\text{signed}} = x\,(e^{-S})^\top,\qquad
z_{\text{rotation}} = x\,(e^{-A})^\top
$$

$\texttt{elim\_rotation} = 0.0$ in 35/35 runs was reported as the headline: *rotation is
dynamically neutral.*

**It is not a finding.** $A$ is real antisymmetric, so $R = e^{-A}$ satisfies

$$
R^\top R = e^{-A^\top}e^{-A} = e^{A}e^{-A} = I
$$

$R$ is **orthogonal**, and so is any cumulative product of such matrices. Now: **every quantity
Block 1b measures is a function of the Gram matrix**, and

$$
(XR^\top)(XR^\top)^\top = X R^\top R X^\top = XX^\top \qquad\text{exactly}
$$

So the Gram matrix is *unchanged*. Energies, effective rank, `ip_mean`, `ip_mass_near_1` —
identical to the unrescaled trajectory, bit for bit up to floating point. The subsequent
re-projection to the unit sphere changes nothing either, since an orthogonal map already
preserves norms.

Measured residual over 24 accumulated layers at $d = 1024$: **$\sim10^{-15}$**, against a
violation threshold of $10^{-3}$ relative. So $n_{\text{rotation}} = n_{\text{original}}$ and
$\texttt{elim\_rotation} = 0$ were forced **in every run, on every model, at every $\beta$,
before any data was read.**

### 3.2 What was done about it, and why that response is right

The row is now **pinned as an identity** by
`tests/test_phase2b_rescaled.py::TestRotationFrameIsAnIdentity`. The frame is retained but
**demoted to an invariance control**: returned with `is_invariance_control=True` and a measured
residual, and **refused as an input to `interpret_comparison`**. Its job is now to fail loudly if
the orthogonality it depends on ever stops holding numerically — not to answer a question.

That is the correct disposition. The alternative (delete it) would lose a genuinely useful
numerical self-check; the alternative (leave it) would keep manufacturing a result.

**The general lesson, which recurs three times in this project.** `math-1b.md` §4.4 found
`strong_bipartition` near-unreachable given cone-collapse; `math-2.md` §10.4 argues
`V_repulsive_via_attn` is unreachable given branch precedence; and here a measurement was
*exactly* constant by construction. In all three cases a null result was reported as evidence
when the test could not have come out otherwise.

**The check that would have caught all three, cheaply: before running a test, ask what its
output would be on data where the hypothesis is maximally true, and on data where it is
maximally false. If those two answers are the same, the test is not a test.** Every one of these
would have failed that check in under a minute.

### 3.3 What survives, and why

$e^{-(S+A)} \ne e^{-S}e^{-A}$ unless $S$ and $A$ commute (§2.4), so **`elim_full` vs
`elim_signed` genuinely differ**, and their contrast is exactly what `status-2.md`'s
"next experiments" item 2 asks for. Concretely:

- If **signed-only rescaling recovers $\approx1.0$ while full-$V$ rescaling gives the 2.1%
  Study B measured**, the failure is *rotational interference in the matrix exponential* — and
  $V$ is still causal, we were just computing $e^{-V}$ in a way the rotational part corrupts.
- If **signed-only also fails**, the mechanism does not transfer to Pythia.

That is a real discriminating test, and it is the phase's remaining live question.

Note it also matters for `math-2.md` §10.2's finding: the rescaled frame is applied at the wrong
time scale, i.e. $e^{-\ell\,\mathrm{OV}}$ instead of $e^{-T_{\rm eff}\mathrm{OV}}$. Over-rescaling
amplifies the non-commutativity too, since the BCH correction grows with the exponent's
magnitude. **The two defects compound**, and both must be fixed before `elim_full` means
anything.

### 3.4 What a real rotation test would be

Something **not invariant under a global orthogonal map of the residual stream.** Two are
reachable from code that already exists:

1. **Weight-space ablation** (`core/intervention.py`): set $W_{OV} := S$ per layer, re-run the
   forward pass, recount violations. The *composition* with attention and the FFN is not
   orthogonally invariant even though the Gram-based metric is — because attention's logits are
   computed from $W_Q, W_K$ in a *fixed* basis, so rotating the residual stream between layers
   does change the routing.
2. **Readout-space measurement** (`core/functional_distance.py`): the decoded next-token
   distribution depends on `embed_out`, which is **fixed**, so rotating the residual stream *does*
   change it. This is the clean discriminator between *"rotation is inert"* and *"rotation
   happens to be orthogonal to the metric we chose."*

The second is the sharper one, and it is worth stating why: **Phase 2b's entire measurement
apparatus is rotation-invariant by construction**, because everything is built on the Gram
matrix (`math-1.md` §4). Asking a Gram-based instrument about rotation is asking a question in a
language that cannot express it. The functional/KL frame is the one place in the project where
the basis is pinned by something outside the residual stream.

---

## 4. The other three blocks, and why each is degenerate

All three were built, none has produced a number, and each fails for its own mathematical
reason. Reading them together is a short course in how a metric can be vacuous.

### 4.1 Block 2 — hemispheric tracking: gated on a constant

Fiedler tracking and rotation–hemisphere alignment were **conditional on Block 1b returning
`rotation_contributes` for some model**. The design doc presents this as good discipline:
expensive analysis runs only if there is a positive signal to explain, and "the uniform
`rotation_neutral` result means it was correctly never triggered."

But §3.1 shows that gate was **a constant**. It could never open. "Correctly never triggered" no
longer holds — the block was gated on an identity.

### 4.2 Block 3 — imaginary ablation: the projector is the identity

`build_imaginary_projector` projects onto $\mathrm{col}(A)$, intending "the rotational subspace."
But **a real antisymmetric matrix in even dimension is generically full rank.** Antisymmetric
matrices have eigenvalues in pairs $\pm i\mu$; a zero eigenvalue is forced only in *odd*
dimension. At $d = 1024$: measured rank **1024**, $\lVert\Pi - I\rVert = 1.6\times10^{-15}$.

So the projector is the identity, and the "ablation" **zeroes every activation at every depth
threshold**. It is not a weak intervention; it is a total one, and it would have destroyed the
signal it was meant to isolate.

The conceptual error is worth naming: **the rotational subspace is not $\mathrm{col}(A)$.** $A$
acts on essentially all of $\mathbb R^d$; what is "rotational" is not a *subspace* but the
*character of the action* — the $2\times2$ plane structure of §2.2. An ablation should remove the
$A$ *component of the operator* ($W_{OV} := S$, §3.4), not project activations onto a subspace.

### 4.3 Block 4 — the LayerNorm Jacobian: three degeneracies and a `NameError`

It raises `NameError` on every prompt (`analyze_layernorm_jacobian` and
`layernorm_jacobian_to_json` used but never imported), caught by the per-prompt handler — so it
has silently never run. Underneath that, the math is degenerate on three counts.

**Deriving the Jacobian properly** (worth doing, because it connects LayerNorm to `math-1.md`
§1.2's tangent projection). With $\mu = \frac1d\mathbf 1^\top x$, $\sigma^2 = \frac1d\lVert
x - \mu\mathbf 1\rVert^2$, and $y = (x - \mu\mathbf 1)/\sigma$:

$$
\frac{\partial y_i}{\partial x_j} = \frac{1}{\sigma}\left(\delta_{ij} - \frac1d - \frac{y_iy_j}{d}\right)
\qquad\Longrightarrow\qquad
J = \frac{1}{\sigma}\left(P_{\mathbf 1} - \frac{yy^\top}{d}\right)
$$

Since $\lVert y\rVert^2 = d$ exactly, $yy^\top/d$ **is** the orthogonal projector onto
$\mathrm{span}(y)$. So

$$
\boxed{\ J = \frac{1}{\sigma}\,\Pi_{\{\mathbf 1,\,y\}^\perp}\ }
$$

**the LayerNorm Jacobian is $1/\sigma$ times the orthogonal projector onto the complement of
$\mathrm{span}\{\mathbf 1, y\}$.** It annihilates the all-ones direction (mean removal) and the
current state direction (radial motion) and acts as a pure scaling on everything else. That is
*exactly* the tangent projection $P^\perp_x$ of the particle model, plus the mean-zero
constraint, plus a $1/\sigma$ gain — which is a clean derivation of why the sphere idealization
is the right one and where it acquires a scale factor.

Now the degeneracies:

1. **`ln_curvature` is identically 1 by algebra.** It computes $\kappa = \lVert x-\mu\rVert^2/(d\sigma^2)$ —
   but $\sigma^2 \equiv \lVert x-\mu\rVert^2/d$ *by definition*, so $\kappa \equiv 1$. Measured:
   0.9999995. **The regressor has zero variance, so the Pearson $r$ against it is always NaN.**
2. **`inflation` cannot exceed the classification threshold.** It is bounded by $\approx1.02$
   because the base fraction is $\approx0.98$, while `_classify` tests $> 1.5$. So the classifier
   returns `H2_UNSUPPORTED` **unconditionally** — the same unreachable-branch pattern as §3.2,
   for the third time in this phase alone.
3. **The Jacobian omits Pythia's learned $\mathrm{diag}(\gamma)$**, so it is the plain-LN
   Jacobian, not the one the model actually applies (`math-1.md` §1.1).

---

## 5. Counting-rule divergence: four ways to manufacture an elimination rate

Independent of the identity, the phase's numbers were not comparable to anything else in the
project, and the fixes are worth listing because each is a distinct failure mode.

**The counting rule diverged.** Phase 2b scored violations with an absolute $-10^{-6}$ threshold
and an `eff_rank >= 3.0` gate, **in three separate hardcoded copies**. The project's rule is
relative $10^{-3}$ (`ENERGY_VIOLATION_REL_TOL`) with `DEGENERATE_RANK_THRESHOLD = 2`. So **no
elimination rate this phase produced was comparable to any Phase 1 or Phase 2 number.** This is
status-1's D7 and D8 landing inside Phase 2b. Now centralized in `p2b_energy.py`.

**Effective rank was a different statistic under the same name.** The local implementation
normalized *unsquared* singular values; `core.metrics.effective_rank` squares them
(`math-1.md` §6.1). Same name, different quantity — **used as a gate.**

**Truncation was computed and discarded.** `max_valid_layer` was dropped by the serializer. This
is Phase 2's verification item V1 landing in the phase where it does the most damage: $e^{-A}$ is
orthogonal and **cannot** overflow, while $e^{-S}$ **can** — so `elim_signed = 1.0` is *precisely
the value an early-truncating signed frame produces for free.* The headline for the surviving
comparison was the value its own numerical failure mode manufactures.

**Three mechanisms manufacture an elimination rate**, all now refused rather than reported:
overflow ($e^{-S}$ diverging); **underflow** ($e^{-S}$ contracting until rows fall below
`l2_normalize`'s $10^{-12}$ floor, after which every energy is the constant $1/(2\beta)$ and the
frame reports zero violations — see `math-1.md` §1.3 for why $1/(2\beta)$ is the no-structure
value); and rank-gate divergence between frames. The third scales with $\lVert V\rVert$, which is
**Phase 2's OV spectral-norm confound** (partial $\rho$ to $-0.71$) — the regime the models are
already known to be in.

**`elim = 0.0` on a clean run.** `_elim_rate` returned float `0.0` when $n_{\text{original}} = 0$,
indistinguishable from "rescaling did nothing" — and that value then entered a $\beta$ majority
vote. **90 of 243 Pythia runs are `no_violations`, and steps 8–64 are clean on all 9 prompts**, so
the phase would have returned a verdict *by vacuity* at exactly the checkpoints where the theorem
holds. Now `None` with an explicit status.

**Substring model matching.** `find_phase2_runs` matched `model_stem in d.name`. On the Pythia
sweep `pythia-410m-step1` matches `step16`, `step128`, `step1000`, `step128000` — **eight of
twenty-seven stems collide.**

---

## 6. Code map

| File | Block | Role |
|---|---|---|
| `rotational_schur.py` | 1a | Schur block parse, $\rho/\theta$ per plane, three named complex-fraction conventions, Henrici, top rotation planes as $(d,2)$ bases, null comparison |
| `rotational_rescaled.py` | 1b | The three frames; the rotation frame demoted to an invariance control with measured residual |
| `p2b_energy.py` | — | **The only place the phase counts a violation.** Relative tolerance, project rank gate, unclipped elimination rate (so negatives — overcorrection — survive) |
| `p2b_io.py` | — | Artifact contract, checkpoint axis, manifest, frame ledger; `refuse_legacy_run_dir` |
| `imaginary_ablation.py` | 3 | Degenerate as written (§4.2) |
| `layernorm_jacobian.py` | 4 | Degenerate as written (§4.3); also home to the *third* complex-fraction definition |
| `fiedler_tracking.py`, `rotation_hemisphere.py` | 2 | Never run; gated on a constant (§4.1) |
| `head_circuits.py` | 1a | Per-head OV circuits — and the argument that $\mathrm{ov\_total} = \sum_h W_{OV}^h$ **is not a thing the model ever forms**, so the 84–97.5% figure is a statistic of a fiction (§2.5) |
| `ffn_rotation.py` | — | Conditional; tests whether the FFN re-introduces rotational displacement at violation layers |
| `p2b_report.py` | — | Report assembly |
| `run_2b.py` / `run_2i.py` | — | Orchestration |

---

## 7. What Phase 2b closes for Phase 2

- **V2 — unclipped rescaled violation counts.** `analysis_p2.py:153` applies
  $\max(0, n_{\text{phase1}} - n_{\text{rescaled}})$, which **destroys the sign** distinguishing
  "rescaling has no effect" from "rescaling makes it worse." `p2b_energy.elimination_rate` does
  not clip — so ALBERT's full-rescaling *overcorrection*, previously a prose caveat, becomes a
  measurable negative number.
- **V1 — `n_valid_layers` per run**, recorded per frame and serialized.
- **Next-experiment 2 — signed-only rescaling on Pythia**, the discriminating test for Study B's
  inert rescaled frame.

---

## 8. Open questions

Tracked in `status-2b.md`: the developmental trajectory of the complex fraction (weights-only,
the cheapest item in the phase, 27 checkpoints); whether Henrici tracks Phase 2's `frac_repulsive`
decay; whether the step 8→16 collapse is rotational; a null (a norm-matched Gaussian is ~100%
complex, so without one Block 1a's headline may be a fact about square matrices); the real
rotation test (§3.4); and which frame to run in (Block 1b runs in `l2_sphere`, but the claim is
about the operator *attention* applies, so the LN frame is arguably correct and is currently
unreachable from this phase).

Surfaced by writing this document:

1. **Henrici and the full-vs-signed gap vanish together, which is a testable cross-block
   prediction nobody has stated.** §2.4: $V$ normal $\iff [S,A] = 0 \Rightarrow e^{-(S+A)} =
   e^{-S}e^{-A}$, and Henrici vanishes exactly when $[S,A]$ does (though it is *not* proportional
   to $\lVert[S,A]\rVert$ — the ratio scales with $d$). So the phase's *one surviving question*
   (§3.3) has its answer's magnitude already predicted by its *other* block's already-computed
   scalar. Two concrete checks, both free: (i) per layer, does $|\texttt{elim\_full} -
   \texttt{elim\_signed}|$ correlate with `henrici_relative`? (ii) are there layers with
   near-zero Henrici, where the two frames must agree, giving a built-in positive control?
   Currently Block 1a and Block 1b are reported as separate findings with no quantitative link.

2. **The "98% complex" headline may be a fact about real square matrices, and the null is one
   line.** A real matrix with i.i.d. Gaussian entries has essentially all eigenvalues in complex
   conjugate pairs — this is standard (the real Ginibre ensemble has $O(\sqrt d)$ real
   eigenvalues out of $d$, so the complex fraction $\to 1$). **So the *expected* complex fraction
   for a random $d\times d$ matrix is $1 - O(d^{-1/2})$, i.e. $\approx0.97$ at $d = 1024$** —
   which is inside the reported 84–97.5% range. The headline may be reporting the Ginibre
   baseline. `core/nulls.py` and `sigma_from_null` exist; the comparison is one function call.
   **Until it is run, Block 1a's descriptive claim should not be cited as a fact about trained
   models.**

3. **The phase's instrument cannot see its own object.** Everything is computed from the Gram
   matrix, which is orthogonally invariant, while rotation is *precisely* an orthogonal action
   (§3.4). This is not just why the headline was an identity — **it is a standing constraint on
   the phase.** Any future Gram-based rotation metric will be vacuous for the same reason. The
   phase needs to move to a basis-pinned observable (readout/KL, or weight-space ablation
   composed through attention) or accept that it can only characterize, never test.

4. **Does the *sign* of $\mathrm{Re}\,\lambda$ within rotation planes matter, and is anyone
   tracking it?** §2.2 fixes $\theta$ to $[0,\pi]$ specifically so that repulsive rotations
   ($\theta$ near $\pi$) are distinguishable from attractive ones. But `math-2.md` §1.2's
   attractive/repulsive split is computed on $\mathrm{Re}\,\lambda$ over the *whole* spectrum,
   complex pairs included — so a complex pair with $\mathrm{Re}\,\lambda<0$ already counts toward
   `frac_repulsive`. **The two phases are partitioning the same eigenvalues on different
   criteria and neither reports the cross-tabulation.** The four-cell table (real/complex ×
   attractive/repulsive, energy-weighted) is one pass over `eig_real`/`eig_imag`, already
   persisted in every `ov_decomp_*.npz`, and it would say directly whether the repulsive
   *energy* Phase 2 attributes violations to lives in rotation planes or in the real spectrum.

5. **The LayerNorm Jacobian result of §4.3 is worth keeping even though the block is dead.**
   $J = \sigma^{-1}\Pi_{\{\mathbf 1,y\}^\perp}$ is a clean statement, and it has a use the block
   never got to: **$1/\sigma$ is a per-token, per-layer gain on the update**, so LayerNorm
   amplifies displacements for low-variance tokens and damps them for high-variance ones. That is
   a candidate mechanism for the attention-sink behaviour that keeps appearing
   (`math-1.md` §2.5, §6.2), it is computable from activations already on disk, and it connects
   to Phase 1c's step-size calibration — $h_\ell$ and $\sigma$ are measuring adjacent things.
