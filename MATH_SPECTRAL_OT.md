<!-- MATH_SPECTRAL_OT.md -->
# Spectral analysis, optimal transport, and the second spectrum

**Status:** reference document, written 2026-08-22. No results in it. It exists because three
methodological gaps were identified across `p2_eigenspectra/`, `p2b_imaginary/` and
`p2d_operator_activation/` that each cost the project evidence, and each has a standard fix that
the code is already most of the way to.

**This is not the `MATH.md` that `PREDICTIONS.md` cites.** That one holds the Phase 1c
derivations against Geshkovski et al. (arXiv:2312.10794v5) and does not exist in this repo. This
document is about the *tools*, not the paper's results; where it needs a paper result it cites the
section.

**Companion module:** `core/dissipation.py` implements §5 and §6. Everything in §3 and §4 is
proposed, not built.

---

## 1. Four things physics means by "eigenvalue analysis"

The phrase covers four settings with different rules. Which one you are in determines whether an
eigenvalue means anything.

| Setting | Operator | Eigenvalues mean | Eigenvectors |
|---|---|---|---|
| Normal modes, small oscillations | real symmetric | squared frequencies; a negative one is an unstable direction | orthogonal; modes evolve independently |
| Quantum mechanics | Hermitian `H` | possible measurement outcomes; `e^{-iHt}` is unitary | orthogonal; degeneracy corresponds to a symmetry |
| Open / non-Hermitian systems | `H = H_0 - i*Gamma` | Re = oscillation frequency, Im = decay rate | **not orthogonal**; they coalesce at exceptional points |
| Statistical mechanics, stochastic dynamics | Markov generator, transfer operator | spectral gap = relaxation rate; eigenvalues near zero = **metastable states** | the sign structure of the leading ones *is* the partition into metastable sets |

Row 4 is where the term "metastable state" in this project's name comes from. Rows 1 and 2 are
where the project's methods come from. Row 3 is where the OV circuit actually lives.

The rules that make rows 1 and 2 comfortable — real eigenvalues, orthogonal eigenvectors,
independent modes, eigenvalues that control the dynamics — are consequences of symmetry. Drop
symmetry and all four fail at once. Row 3 is what "drop symmetry" looks like.

---

## 2. Where OV sits, and what the S/A split actually is

`p2_eigenspectra/weights.py::eigendecompose` computes, for the composed OV circuit:

- full complex eigenvalues, and the fractions with `Re > 0` / `Re < 0`;
- an ordered real Schur form, giving orthonormal bases for the attractive and repulsive
  *invariant subspaces* (correct, and the right choice for a non-normal matrix — Schur vectors are
  orthonormal where eigenvectors are not);
- separately, the eigendecomposition of the symmetric part `S = (OV + OV^T)/2`;
- an `agree` flag: do the two methods give the same sign split within 10%?

`p2b_imaginary/` then splits `V = S + A` with `A = (V - V^T)/2`.

That split is not bookkeeping. It is exactly the Hermitian / anti-Hermitian decomposition of an
open-system Hamiltonian: `A` generates orthogonal motion (norm-preserving — the classical analogue
of unitary evolution), `S` generates gain and loss. Two consequences follow that the phase docs
currently state as empirical findings, and which are actually theorems.

### 2.1 At short times, the symmetric part is the whole story

For the linear flow `x' = -Vx`,

    d/dt ||e^{-tV} x||^2  at t=0   =  -2 * x^T S x

so `A` contributes **exactly zero to first order**, and the largest possible initial growth rate is

    omega(V) = lambda_max(S)          the *numerical abscissa*

`eigendecompose` already computes this — it is `sym_eigenvalues.max()` — and never names it.

This matters for reading Phase 2b. Its surviving claim is that the signed residue `S` carries the
causal weight even though `A` dominates structurally. At `T_eff << 1` that is not a surprising
empirical result; it is forced. Saying so *strengthens* the claim (it now has a derivation) and
explains why the withdrawn half was never falsifiable.

### 2.2 `Re lambda` is the wrong sign to read at depth 24

Three different "abscissas" govern three different time regimes of `||e^{-tV}||`:

| Quantity | Formula | Governs |
|---|---|---|
| numerical abscissa | `lambda_max(S)` | `t -> 0`, the initial slope |
| Kreiss constant / `sup_t ||e^{-tV}||` | — | the intermediate transient |
| spectral abscissa | `max Re lambda(V)` | `t -> infinity`, the asymptotic rate |

For a **normal** matrix all three coincide and the eigenvalues tell you everything. For a
non-normal one they can disagree by orders of magnitude: an operator with every eigenvalue in the
stable half-plane can still amplify enormously before it decays. This is the standard
non-normal-transient-growth result from hydrodynamic stability (Trefethen & Embree, *Spectra and
Pseudospectra*); it is the reason linearly "stable" shear flows go turbulent.

The project's own registered prediction P-gamma2 says the network integrates to
`T_eff << t* ~ 4.2`. So `frac_attractive` / `frac_repulsive` / the Schur attractive-repulsive
projectors — all of which describe the `t -> infinity` regime — are being read in a regime where
they do not govern. The decision-relevant object is the transient curve `sup_t ||e^{-tV}||` and the
`t` at which it peaks, read against `T_eff`.

The project already measures non-normality and already knows it is large.
`p2_eigenspectra/visualization/spectra.py` plots `||OV||_2 / rho(OV)` against a null line of 1.0,
and `eigendecompose`'s `agree` flag is described in that file's own docstring as marking
"where Phase 2's projectors are method-dependent and the verdict should be read with care". The
measurement exists. What is missing is using it dynamically rather than as a caveat.

### 2.3 Exceptional points

Where two eigenvectors coalesce (not just two eigenvalues — a *defective* matrix), the operator has
an exceptional point. Near one, eigenvalues are extraordinarily sensitive to perturbation, and the
transient contains a `t * e^{lambda t}` term that no eigenvalue predicts.

The right per-eigenvalue diagnostic is the **eigenvalue condition number**

    kappa_i = 1 / |<w_i, v_i>|          left and right eigenvectors, both normalized

`kappa_i` large means that eigenvalue is numerically fictitious. Phase 2d's open item 3 says
"`simple_tol` and `align_tol` are placed, not derived" and that eigenvalues of a d=1024 non-normal
OV circuit "come in near-degenerate clusters". `kappa_i` is the quantity that *derives* the
tolerance instead of placing it, and near-degenerate clustering in a non-normal matrix is precisely
what proximity to an exceptional point looks like.

---

## 3. Nulls: the counting statistics sit on top of chance

Every eigenvalue-counting statistic in Phase 2 and Phase 2b is reported without a null, and the
null values are close to the reported numbers.

**Real eigenvalues.** A real Ginibre matrix (iid Gaussian entries, no symmetry) has

    E[# real eigenvalues] = sqrt(2d/pi) * (1 + o(1))         Edelman-Kostlan-Shub

which at d=1024 is about 25 out of 1024. So `frac_complex ~ 0.975` **at chance**, and it can
essentially only fall. A *decrease* is the trained signal — eigenvalues condensing onto the real
axis means the map is becoming more symmetric.

Phase 2b's headline is "OV is 84-97.5% rotational across 7 architectures". The top of that band is
the chance value. `status-2b.md` already flags this as caveat (i) — "it is not known to distinguish
trained from random" — and proposes running a norm-matched Gaussian null. The point of this section
is that the null has a closed form and does not need to be simulated to be quoted (though
simulating it is still the right check, because composed OV is a *sum over heads of products* of
Gaussian blocks, not a single Ginibre matrix; the `sqrt(d)` scaling is generic for real asymmetric
ensembles but the constant is not guaranteed).

**Sign split.** At initialization the composed OV spectrum is symmetric about the imaginary axis,
so `frac_repulsive = 0.5` at chance, flat across depth, with `O(d^{-1/2})` scatter. Phase 2's
unexplained `frac_repulsive` decay — 1.00 -> 0.50 -> 0.80 across ~90k steps with violation count
flat (`status-2b.md` new-question 2) — passes exactly through the chance value at its midpoint.
That is at minimum a coincidence worth ruling out before a mechanism is proposed for it.

`visualization/spectra.py` documents **both** of these null lines in its docstring and draws
`frac_repulsive` on a diverging colormap centered at 0.5 for exactly this reason. The analysis
modules that produce the verdicts never use them.

**One number that predicts both.** The elliptic law's reciprocity parameter

    tau = sum_ij V_ij V_ji / sum_ij V_ij^2        in [-1, 1]

measures how symmetric `V` is. The bulk spectrum fills an ellipse with semi-axes `(1+tau, 1-tau)`:
`tau = 1` is symmetric (real spectrum, an interval), `tau = 0` is Ginibre (a disk), `tau = -1` is
antisymmetric (imaginary spectrum). It predicts the complex fraction and the cloud shape together,
it is one line of numpy, and it is the natural scalar to plot against training step.

**Bulk versus outliers.** An unweighted count over all `d` eigenvalues gives every direction an
equal vote regardless of magnitude, so it is dominated by the bulk — which is noise. The trained
structure lives in the outliers past the bulk edge (the BBP-transition picture). Two fixes, both
cheap: report `|lambda|`-weighted fractions alongside the counts, and report an explicit outlier
count and their positions.

---

## 4. Eigenvectors: the half of the analysis that is missing

The project computes eigen*values* and invariant sub*spaces*. Individual eigenvectors appear in
exactly one place — `p2_eigenspectra/vocab_projection.py`, which projects symmetric-part
eigenvectors through the unembedding to get token readouts, and which is a good idea that stops
after one use. Four standard, weights-only additions:

1. **Eigenvalue condition numbers** `kappa_i = 1/|<w_i, v_i>|` — §2.3. Says which eigenvalues are
   real and which are numerical fiction, and how close the operator is to an exceptional point.

2. **Localization (inverse participation ratio)**

       IPR(v) = sum_k v_k^4 / (sum_k v_k^2)^2

   The Anderson-localization diagnostic. `IPR ~ 1/d` means the direction is spread over the whole
   residual stream; `IPR ~ 1` means it lives on a handful of dimensions. Asked of the repulsive
   directions, this is the difference between "the repulsive mechanism is a few identifiable
   features" and "it is a diffuse property of the whole space" — and Phase 2b already observes that
   ALBERT-xlarge's violations are sustained by a ~2.5% signed residue, i.e. *something* is highly
   concentrated. IPR measures that directly instead of inferring it.

3. **Level-spacing statistics.** The distribution of gaps between adjacent eigenvalues:
   Poisson (uncorrelated, "integrable") versus Wigner-Dyson (level repulsion, "chaotic"). A
   standard structural diagnostic, one histogram per checkpoint, and it distinguishes a spectrum
   with genuine structure from one that is a random cloud in a way that a mean or a fraction
   cannot.

4. **Cross-checkpoint tracking by overlap, not by sorted order.** Any plot of eigenvalues against
   training step currently risks spurious jumps: sorting by magnitude re-labels modes whenever two
   cross. Matching by `|<v_i^(t), v_j^(t+1)>|` (a linear assignment on the overlap matrix) turns
   those artifacts into visible avoided crossings, which are themselves interpretable —
   an avoided crossing is where two modes exchange character, i.e. a candidate mechanism for a
   dated transition.

---

## 5. Optimal transport: the frame that makes the rest one subject

### 5.1 Why it is native here, not imported

The paper's dynamics are a mean-field interacting particle system. Its state is not a point in
`R^d`; it is a **measure** `mu_l` on the sphere. The natural geometry of the space of measures is
the Wasserstein metric, and under the paper's gradient-flow hypotheses (§3.4, which is exactly what
Phase 2d's D1 tests) the layer dynamics *are* the Wasserstein gradient flow of `E_beta`:

    v = -grad (dE_beta / dmu)

    dE_beta/dt = integral < grad(dE_beta/dmu), v > dmu           (the dissipation identity)

Everything the project measures — `ip_mean`, effective rank, energy violations, Fiedler value —
is a scalar shadow of that identity.

### 5.2 The gradient has a closed form and nothing computes it

`core/metrics.py::interaction_energy` defines

    E_beta = (1 / (2*beta*n^2)) * sum_ij exp(beta * <x_i, x_j>)

Differentiating — the factor 2 from the symmetric double sum and the `beta` from the chain rule
both cancel:

    dE_beta/dx_i = (1/n^2) * sum_j exp(beta * <x_i, x_j>) * x_j

Two lines. It is worth staring at: **that is an unnormalized attention-weighted average of the
other tokens** — literally what an attention head computes when `Q^T K = I`. The paper's
gradient-flow condition stops being a hypothesis you test with a proxy and becomes an identity you
can see. It also means the comparison "is this head doing gradient descent on `E_beta`?" can be
made *directly*, per head, per layer, by comparing the head's actual output against this vector,
rather than inferred from the weights-only algebraic condition in D1.

Nothing in the repository computes it. `E_beta` is the project's central object, every "violation"
is a sign of `Delta E_beta`, and the gradient that produces that sign has never been evaluated.

### 5.3 What the identity buys

On the sphere, with `P^perp_u = I - u u^T` the tangential projector:

    G_i = P^perp_{x_i} (dE_beta/dx_i)                the tangential energy gradient
    v_i = (1/||x_i||) P^perp_{x_i_hat} (Delta x_i)   the tangential velocity

and `||v_i||` is **exactly** Phase 1c's existing step size `h_l = ||P^perp(Delta x)|| / ||x||`. So
this is the rigorous form of a heuristic already in use, not a competing one.

**(a) Every energy violation becomes an exact accounting.** A violation is
`sum_i <G_i, v_i> > 0` — the network pushing uphill on `E_beta`. Because the sum is over particles,
each violation is attributable to specific tokens rather than being a layer-level event with no
internal structure. That is the missing link between Phase 5c's "particles first" framing and
Phase 1's energy series.

**(b) The attention/FFN split becomes exact.** On Pythia's parallel residual
(`core/sublayer_streams.py`, `use_parallel_residual=True`)
`Delta x_i = Delta x_i^attn + Delta x_i^ffn` holds exactly, and `P^perp` is linear, so

    sum_i <G_i, v_i>  =  sum_i <G_i, v_i^attn>  +  sum_i <G_i, v_i^ffn>

exactly. This is what `status-2.md` asks for ("re-enables the attn-vs-FFN energy panels") and it is
strictly stronger than GPT-2's `decompose.py`, which carried a sequential-ordering confound that
`design-2.md` documents at length.

**(c) The spectral subspaces join the same identity.** Projecting `v_i` through Phase 2's existing
Schur projectors (`weights.build_subspace_projectors`) factors the dissipation a second way,
giving a **channel x spectral-subspace x particle** decomposition of every violation. Phase 2's
displacement test and Phase 1's energy series stop being two correlated measurements and become two
marginals of one exact quantity.

**(d) The linearization residual is itself a result.** `Delta E_beta - sum_i <G_i, v_i>` is the
second-order term. It measures whether the continuum limit the entire project assumes — treating a
residual block as a forward-Euler step of an ODE — is actually valid at that layer. If it is large,
that is a finding about the project's framing, not an error term.

### 5.4 Transport proper

**A coordinate-free `T_eff`.** Phase 1c defines `T_eff = sum_l h_l` with `h_l` the hand-rolled step
size above. The principled version is the **Wasserstein arc length** `sum_l W_2(mu_l, mu_{l+1})`.
Both are computable on the same artifacts; the comparison puts P-gamma2 on a defensible footing
instead of a convention.

**A new observable, free.** For two `n`-point clouds with equal weights:

- `W_2` under the *identity coupling* (token i to token i) is an upper bound, and equals the
  current convention.
- True `W_2` uses the *optimal coupling*, which at `n ~ 500` is an exact linear assignment
  (`scipy.optimize.linear_sum_assignment`, already available — scipy is a dependency).

The **gap between them** measures how much of a layer's displacement is tokens *swapping places* —
motion that leaves the distribution completely unchanged — versus genuine motion of the measure.
No existing metric in the project separates these, and they mean very different things for a claim
about clustering.

**Straightness.** Arc length versus endpoint distance `W_2(mu_0, mu_L)`. A long path with a short
net displacement is the signature of *dwelling*, which is what metastability is — measured on the
measure, rather than inferred from whether HDBSCAN found clusters.

**Gromov-Wasserstein, for cross-architecture comparison.** `PREDICTIONS.md` claim (c) asks whether
the phenomenology transfers from gpt2-large to pythia-1.4b. Those live in spaces of different
dimension with no shared basis. GW compares metric-measure spaces — it needs only the internal
pairwise-distance structure of each cloud, which is exactly the Gram matrix the project already
computes everywhere. It is the correct instrument for that claim; `linear_cka` (in `core/metrics.py`)
is the current best available and requires paired samples in comparable spaces.

**Unbalanced OT, for merge and split events.** Clusters gaining and losing members across layers is
mass creation and destruction, which balanced OT cannot represent. Unbalanced OT can, and gives a
soft particle-to-cluster correspondence in place of nearest-neighbour matching.

**Entropic OT / Schrodinger bridges.** The entropically-regularized problem is Schrodinger's 1931
question about the most likely evolution of a cloud of particles between two observed
distributions. It is the natural noisy-dynamics extension, it is cheap (Sinkhorn iterations), and
it is the historical bridge to the quantum side of this document: the Schrodinger bridge problem is
formally the imaginary-time counterpart of the Schrodinger equation, which is where Nelson's
stochastic mechanics comes from.

---

## 6. The deepest gap: there are two spectra, and we compute one

This is the summary point, and it is why §2-§4 and §5 are one subject rather than two.

**Eigenvalues of `V` linearize the motion of a particle in `R^d`.** That is what
`p2_eigenspectra` computes.

**Metastability is a property of the measure.** It is governed by the spectrum of the Wasserstein
Hessian of `E_beta` at a configuration: near-zero eigenvalues are slow directions, i.e. long-lived
states, and the escape rate over a barrier is set by the single negative eigenvalue at the saddle
(Eyring-Kramers; the same structure as Kramers' escape-rate formula and as transition-state theory
in chemistry). The number of small eigenvalues counts the metastable states; the sign structure of
the corresponding eigenvectors *is* the partition into them.

The project has only ever computed the first spectrum and inferred the second. Every claim of the
form "these are metastable states" currently rests on a clustering algorithm plus a scalar (the
Fiedler value) rather than on the operator whose spectrum defines the term.

Concretely, three things this predicts are worth building, in this order:

- **`core/dissipation.py`** (built; §5) — the first-order piece of the identity. Exact, runs on
  artifacts already on disk.
- **A non-normal-dynamics module** (weights-only: numerical abscissa, `sup_t ||e^{-tV}||`, Kreiss
  constant, epsilon-pseudospectra, eigenvalue condition numbers, the nulls of §3). Weights-only is
  the key property — it runs on all 27 Pythia checkpoints immediately, which is the same argument
  `p2b_imaginary/rotational_schur.py` already makes for Block 1a.
- **A metastability module**: a transfer-operator / Markov-state model over clusters, implied
  timescales `t_i = -1/log|lambda_i|`, and PCCA+ — the standard method from molecular dynamics for
  extracting metastable sets from a transition operator's leading eigenvectors. The project's
  Fiedler vector is the `k=2` special case of PCCA+. Then the Hessian of `E_beta` and its Morse
  index, which is the definition the word "metastable" is standing in for.

---

## 7. Two stale claims found while writing this

Reported, not fixed — both need an owner's decision, and both are the kind of correction
`status-2b.md` insists be made explicitly rather than silently.

1. **`p6_subspace/design-6.md` rests on the withdrawn Phase 2b result.** Its first paragraph reads:
   "Phase 2b established that the antisymmetric/imaginary component A is dynamically neutral —
   removing it from OV leaves energy violations unchanged; the symmetric/real component S carries
   100% of violation causality." `status-2b.md` withdrew that as an orthogonal-invariance identity
   that was never falsifiable. Phase 6's entire four-part design is built on it as a premise.

   The phase is recoverable rather than dead: §2.1 above supplies a *correct* short-time version of
   the same claim (`A` contributes zero to the initial energy change by algebra, so at `T_eff << 1`
   the causal asymmetry between `S` and `A` is real, just for a different reason). Re-grounding
   Phase 6 on that would preserve most of its design.

2. **`design-2b.md`'s "Interpretation of the result" section still states the withdrawn
   conclusion.** `status-2b.md` explicitly lists this as needing correction ("the 'Interpretation
   of the result' section and the 'why Phase 2c is separate' argument both rest on the withdrawn
   finding"). It has not been done, so a reader arriving at `design-2b.md` first gets the retracted
   version with no marker.

---

## 8. References

- Geshkovski, Letrouit, Polyanskiy, Rigollet, *A mathematical perspective on Transformers*,
  arXiv:2312.10794v5 — the paper this project tests. §3.4 (gradient-flow condition), §6
  (metastability, Wendel's theorem), §9.1-9.2 (sharp configurations, Table 1).
- Trefethen & Embree, *Spectra and Pseudospectra* (2005) — non-normality, transient growth, the
  Kreiss constant, why eigenvalues mislead.
- Edelman, Kostlan & Shub, "How many eigenvalues of a random matrix are real?" (1994) — the
  `sqrt(2d/pi)` null in §3.
- Sommers, Crisanti, Sompolinsky & Stein (1988) — the elliptic law and the reciprocity parameter.
- Villani, *Optimal Transport: Old and New*; Ambrosio, Gigli & Savare, *Gradient Flows in Metric
  Spaces and in the Space of Probability Measures* — Wasserstein gradient flows, the dissipation
  identity, displacement convexity.
- Peyre & Cuturi, *Computational Optimal Transport* (2019) — Sinkhorn, sliced Wasserstein,
  Gromov-Wasserstein, unbalanced OT.
- Deuflhard & Weber, "Robust Perron cluster analysis in conformation dynamics" (2005) — PCCA+.
- Bovier, Eckhoff, Gayrard & Klein (2004) — Eyring-Kramers formulas, the potential-theoretic
  approach to metastability.
