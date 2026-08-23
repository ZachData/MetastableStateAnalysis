# Phase 5b — MATH (study notes)

## 0. What this document is

Companion to `math-5.md`. **Phase 5b is fully specified and has never been run** — code, tests,
eight pre-registered falsification predictions, and per-sub-experiment thresholds all exist; no
result file does. That makes this the cleanest phase in the project to study, because nothing
here has been contaminated by post-hoc reasoning yet, and it is the best place to look for
problems *before* they become findings.

It is also the phase that reaches furthest outside the project: it asks whether this project's
unsupervised cluster centroids are **the same geometric objects** as the concept centroids in
Wurgaft et al. (2026), *Manifold Steering Reveals the Shared Geometry of Neural Network
Representation and Behavior*.

---

## 1. The hypothesis

### 1.1 What Wurgaft et al. establish

Given **concept-labeled** data, they:

1. fit an **activation manifold** $M_h$ — concept centroids in activation space, threaded with a
   spline;
2. fit a **behavior manifold** $M_y$ — the corresponding output probability distributions;
3. show $M_h$ and $M_y$ are approximately **isometric** (pairwise geodesic distances correlate);
4. show that steering *along* $M_h$ produces natural behavioral trajectories, while **linear**
   steering does not — it produces "teleportation," non-adjacent jumps in output distribution.

### 1.2 What Phase 5b asks

> **Are our metastable-state cluster centroids the same objects as Wurgaft's concept centroids?**

The substitution — Phase 1's unsupervised HDBSCAN centroid trajectories in place of labeled
concept data — **is the entire test.** If it holds, this project's cluster-tracking apparatus
becomes unsupervised evidence for a claim Wurgaft made with supervision, and more importantly it
supplies **a causal upstream his framework does not have**:

$$
\underbrace{V \text{ eigenstructure}}_{\text{Phases 2, 2b, 6}}
\;\longrightarrow\; \underbrace{\text{metastable attractor landscape}}_{\text{Phase 1}}
\;\longrightarrow\; M_h \;\underset{\text{isometry}}{\longleftrightarrow}\; M_y
$$

i.e. *$M_h$ has the geometry it has **because** V's eigenstructure determines which states are
metastable.* That is a genuinely ambitious claim and the phase is structured so each link is a
separate sub-experiment.

**One caution about the chain's first link.** It is exactly Phase 2's claim, and Phase 2b has
since withdrawn its supporting "signed component carries 100% of causal weight" result as an
algebraic identity (`math-2b.md` §3.1), while Phase 2's own rescaled-frame evidence on Pythia is
inert for reasons still being diagnosed (`math-2.md` §10). **The chain currently rests on a link
whose evidence changed after this phase was designed.** That does not invalidate B, C, or D —
they test the downstream links independently — but the causal reading of a positive result should
be stated more cautiously than `design-5b.md` currently does.

---

## 2. The mathematics

### 2.1 Manifold fitting (Sub-exp A)

Reduce centroids by PCA, parameterize the centroid path across layers by **arc length**, and fit
a cubic spline. Arc-length parameterization is the right choice because it makes the intrinsic
coordinate $u$ metrically meaningful — equal steps in $u$ are equal distances along the manifold —
which is what makes the geodesic distances of §2.3 well defined.

Two implementation points that carry content:

**`_complete_basis` pads with honest zeros.** When more components are requested than the data has
rank for ($k > \min(n,d)$), the basis is extended with arbitrary orthonormal directions from the
orthogonal complement. Those columns **carry zero variance and are not principal components in any
meaningful sense** — they exist so the returned basis satisfies its declared $(d,k)$ shape
contract. `pca_reduce` reports their explained-variance ratio as **exactly 0.0**, so a caller
reading `evr` can always tell which columns are real. This is the project's "refuse rather than
degrade" discipline in its softest form: degrade, but label the degradation.

**Bounded memory on purpose.** The obvious fix for the old truncation bug was
`np.linalg.svd(..., full_matrices=True)` — which materializes a $(d,d)$ matrix, **134 MB at
$d = 4096$**, to use $k \ll d$ columns of it. The chosen route never allocates more than $(d,k)$.
The seed is fixed rather than drawn from global state, so two runs on the same data return the
same basis.

### 2.2 The Hellinger geometry, and why it is the right choice

$$
d_H(p,q) = \frac{1}{\sqrt2}\big\lVert \sqrt p - \sqrt q\big\rVert_2 \;\in\; [0,\ 1]
$$

This is not an arbitrary divergence. **The map $p \mapsto \sqrt p$ sends the probability simplex to
the positive orthant of the unit sphere in $L^2$**, since $\lVert\sqrt p\rVert^2 = \sum_v p_v = 1$.
Three consequences, all load-bearing:

1. **The behavior side becomes a sphere geometry**, structurally identical to the activation side
   after L2 normalization (`math-1.md` §1.1). The two manifolds being compared live in the same
   *kind* of space, which is what makes an isometry claim meaningful rather than a comparison of
   incommensurables.
2. **It is the chordal distance of the Fisher–Rao metric.** The Fisher information metric on the
   simplex has geodesic distance $2\arccos\big(\sum_v\sqrt{p_vq_v}\big)$ — the great-circle
   distance on the radius-2 sphere reached by $z = 2\sqrt p$, on which the Fisher–Rao metric is
   exactly the round metric; on the *unit* sphere where $\sqrt p$ lives the geodesic is
   $\arccos\big(\sum_v\sqrt{p_vq_v}\big)$, and $d_H$ is that chord divided by $\sqrt2$. So
   Hellinger is not merely convenient — it is the Euclidean shadow of the canonical
   information-geometric metric on distributions, chord to its arc, agreeing with it up to a
   fixed constant factor in the small-separation limit.
3. **It fixes the aggregation convention upstream.** This is exactly why `math-1.md` §9.3's
   `compute_behavior_trajectories` defaults to averaging in the Hellinger embedding:
   mean-then-renormalize in $\sqrt p$ space **is** the spherical mean, matching the activation
   side's mean-then-renormalize. Averaging in probability space instead is entropy-increasing and
   blurs peaked distributions toward uniform, compressing precisely the distances this phase
   exists to resolve.

### 2.3 Geodesic distances

$$
d_{M}(u_i, u_j) = \int_{u_i}^{u_j}\left\lVert \frac{d\gamma}{du}\right\rVert du
\;\approx\; \sum_{k=1}^{n_{\rm pts}-1}\big\lVert \gamma(u_{k+1}) - \gamma(u_k)\big\rVert
$$

Discretized into $n_{\rm pts} = 150$ waypoints, evaluated on the spline, accumulated in PCA space.
For a periodic manifold the wrap case takes **the shorter of the two arcs**, computed both ways
rather than assumed.

### 2.4 The isometry test (Sub-exp B)

Pearson correlation between the two pairwise-distance vectors:

$$
r_{\rm manifold} = \mathrm{corr}\Big(\big\{d_{M_h}(i,j)\big\}_{i<j},\ \big\{d_{M_y}(i,j)\big\}_{i<j}\Big)
$$

compared against $r_{\rm linear}$, the same correlation using straight-line rather than
along-manifold distance on the activation side. **This is the load-bearing test**: if
$r_{\rm manifold}$ is not clearly greater than $r_{\rm linear}$, the rest of the causal chain has
nothing to attach to.

---

## 3. The metric layer: free parameters that were silently fixed

`p5b_distances.py` exists because Sub-exp B is *a correlation between two distance vectors*, and
**which distance is used on each side is a free parameter** that had been fixed without being
recorded: activation side to L2-sphere chordal, behavior side to arc length along a fitted
Hellinger spline. The module makes both explicit, named, and swappable, **with no verdict logic** —
it computes distance matrices and nothing else, keeping readings separate from verdicts.

$$
\texttt{ACTIVATION\_FRAMES} = (\texttt{sphere},\ \texttt{ln},\ \texttt{raw}),
\qquad
\texttt{BEHAVIOR\_METRICS} = (\texttt{hellinger},\ \texttt{sym\_kl})
$$

**Cross-architecture comparability requires the LN frame.** The model never reads the raw residual
stream; GPT-2 is sequential, Pythia is parallel-residual with two LayerNorms per block of which
`input_layernorm` — not `post_attention_layernorm`, despite the name — is what attention reads
(`math-1.md` §2.2). Without routing through `core/ln_frame.py`, *"distance between cluster
centroids" denotes a different operation per architecture*, and any cross-model comparison is
confounded by extraction convention.

**`sym_kl` as an arbiter.** Symmetrized KL is the divergence the readout actually implies, computed
straight from cached probabilities with no spline and no sphere embedding in the way. Hellinger
stays primary (it is what Wurgaft used, and the 0.7 threshold was calibrated against his numbers),
but **if $r_{\rm manifold}$ is strong under one and collapses under the other, that is information
about the fit rather than about the model.** Carried limitation: sym-KL is **not a metric** — the
triangle inequality fails — which is fine for correlating distance vectors and **must not** be
reused in Sub-exp D's subspace scoring, in any MDS embedding meant to be trusted metrically, or in
anything calling `scipy.spatial`.

### 3.1 The subtlest point in the phase: LayerNorm is not linear

$$
\mathrm{LN}\Big(\tfrac1{|C|}\textstyle\sum_{i\in C} x_i\Big) \;\ne\; \tfrac1{|C|}\textstyle\sum_{i\in C}\mathrm{LN}(x_i)
$$

**So frames must be applied to token activations *before* masking and averaging, never post-hoc to
an already-built centroid.** That is why `frame_centroids` takes raw activations rather than the
output of `load_plateau_centroids`.

This is easy to get wrong and impossible to detect downstream — both orderings produce a
plausible $(m, d)$ centroid array. It generalizes: **any nonlinear frame must be applied before
any aggregation**, and this project has at least three nonlinear frames (LN, L2 normalization, and
the $\sqrt p$ Hellinger map). The Hellinger case is handled correctly for the same reason in
`math-1.md` §9.3.

---

## 4. Sub-experiments C and D

### 4.1 C — merge-event teleportation (a prediction Wurgaft's framework alone does not make)

Wurgaft finds that **linear** steering produces "teleportation" — non-adjacent jumps in the output
distribution. Phase 5b's novel prediction: **if merge events are the model's own transitions
between metastable states, they should show the same teleportation signature.**

This is where the project's dynamical-systems framework earns its keep — merge events come from
Phase 1's Jaccard/Hungarian tracking (`math-1.md` §9.2), and nothing in Wurgaft's setup would
suggest looking at them. It is also a genuine prediction rather than a re-description: a merge
could perfectly well be a smooth along-manifold transit, in which case C fails informatively.

### 4.2 D — S-subspace isometry

Does restricting $M_h$ to the **S** (real/symmetric) subspace improve isometry with $M_y$,
relative to full or **A**-restricted $M_h$? This directly cross-validates Phase 6's S/A
division-of-labour hypothesis using the isometry framework, and **it is the test that would
confirm V's eigenstructure is *the* coordinate system $M_h$ lives in, not merely correlated with
it.** A partial analog of Wurgaft's pullback procedure, not the full thing.

---

## 5. Pre-registration, and why it matters more here

Every sub-experiment has explicit pass/fail thresholds set in advance — e.g. **P5b-B2:
$r_{\rm manifold} > 0.7$**, calibrated against Wurgaft's reported 0.89–0.999 for concept-labeled
tasks. `design-5b.md` states the reason plainly:

> the entire premise (cluster centroids = concept centroids) is an **identity claim** that could
> otherwise be argued into a positive result from ambiguous correlations.

And the null is made informative in advance: **"B fails" would mean cluster structure is
dynamically meaningful (established by Phases 1–4) but not *semantically* structured in Wurgaft's
specific sense** — a real result, not an absence of one.

**Scope boundary, stated up front:** this phase does *not* implement the steering intervention
(replacing activations with spline-interpolated targets). The goal is identity verification — *are
these the same geometric objects?* — not steering. Wurgaft's pullback procedure is also not
attempted; D is a partial analog.

---

## 6. Alignment with Phase 1, enforced rather than assumed

The behaviour side consumes `math-1.md` §9.3's machinery, and the alignment discipline there is
what makes Phase 5b well posed:

- `compute_behavior_trajectories` masks **the same tokens the centroid used**, at each chain step,
  so activation-side and behaviour-side descriptions are guaranteed to be about the same
  population.
- `stack_behavior_by_traj_ids` **iterates `traj_ids`** — the identity list for the whole phase —
  rather than the behaviour dict's own keys, which is what guarantees row $i$ of the distribution
  stack and row $i$ of the centroid array describe the same cluster.
- **Coverage is returned, not hidden.** A trajectory covered at 1 of 5 chain layers and one covered
  at 5 of 5 are not equally good measurements; the caller decides whether to drop low-coverage
  trajectories, and the function does not decide for them.

The design note in `cluster_tracking.py` records why the behaviour half lives in Phase 1 rather
than here: it is the same tracking operation applied to a different per-token quantity, and
**Phase 5b previously compensated for its absence by taking a global mean over *all* tokens at
each plateau layer** — which decoupled $M_h$'s population from $M_y$'s and silently disabled
Sub-exp B.

---

## 7. Code map

| File | Sub-exp | Role |
|---|---|---|
| `manifold_fit.py` | A | PCA reduction with honest zero-variance padding; arc-length parameterization; cubic/periodic spline fits for $M_h$ and $M_y$ |
| `p5b_distances.py` | A/B | The metric layer: three activation frames × two behaviour metrics, no verdict logic; `frame_centroids` (§3.1) |
| `isometry_test.py` | B | Hellinger distance, spline arc length with periodic wrap, $r_{\rm manifold}$ vs $r_{\rm linear}$ |
| `merge_teleportation_subspace.py` | C | Teleportation signature at Phase 1 merge events |
| `subspace_isometry.py` | D | S- vs A- vs full-$M_h$ isometry, consuming Phase 2/6 projectors |
| `logit_cache.py` | — | The new infrastructure: intermediate-layer output distributions via one re-forward pass |
| `p5b_io.py`, `p5b_report.py`, `run_5b.py` | — | IO, reporting, CLI |

Outputs per model/timestamp: `run_config.json`, `fit_summary.json`, `mh_params.npz`,
`my_params.npz`, `isometry.json`, `isometry_mds.npz`, `merge_teleportation.json`,
`teleportation_summary.json`, `subspace_isometry.json`, `p5b_report.txt`.

---

## 8. Open questions

Tracked: nothing has been run; `logit_cache.py` must be verified working before Sub-exp A; and
A must run first since B, C, D all consume its output.

Surfaced by writing this document:

1. **Sub-exp D has no dimension-matched control, and without one its result is unfalsifiable in
   the same way `math-1b.md` §6's cone verdict was.** Restricting $M_h$ to the S subspace *reduces
   dimension*, and lower-dimensional embeddings mechanically change distance correlations — often
   improving them, since projection removes variance that is noise with respect to the behaviour
   side. So "S-restriction improves isometry" is not distinguishable from "**any** restriction to
   $\dim = k$ improves isometry" without a **random subspace of matched dimension** as a third arm.
   Given that S and A generically have different dimensions, D as specified compares three
   conditions that differ in dimension *and* in content simultaneously. This is the cheapest fix
   in the phase and it should land before D runs, not after.

2. **Pearson $r$ on pairwise distances is a weak notion of isometry, and the threshold's transfer
   is unexamined.** $r$ is invariant to affine rescaling, so it tests *rank-like* agreement rather
   than isometry proper; two manifolds related by a large uniform stretch would score $r = 1$.
   Stronger, and nearly free given the code already emits `isometry_mds.npz`: **Procrustes
   disparity or MDS stress** between the two distance matrices, which does test metric agreement.
   Separately, $r$ has a sample size of $m(m-1)/2$ pairs, so its sampling distribution depends
   strongly on the number of centroids $m$ — **Wurgaft's 0.89–0.999 was obtained at his $m$, not
   ours**, and a fixed 0.7 threshold does not obviously transfer across $m$. Report a confidence
   interval, or calibrate the threshold against a label-permutation null at our own $m$
   (`core/nulls.py` already provides the machinery).

3. **`n_pts = 150` is an unvalidated numerical parameter on a quantity the whole phase rests on.**
   Arc length by polyline summation converges as $O(n_{\rm pts}^{-2})$ for a smooth spline, so 150
   is probably ample — but "probably ample" is exactly the standard `math-1c.md` §2.2 refuses for
   the ODE, where the step is halved until the answer stops moving and non-convergence is reported
   rather than hidden. The same three-line check applies here, and the phase's own falsification
   thresholds are stated to three digits.

4. **Coverage filtering is an unregistered researcher degree of freedom.** §6 correctly returns
   coverage rather than deciding for the caller — but that means *the caller decides*, after seeing
   the data, which trajectories enter the correlation. With $m$ small and $r$ thresholded at 0.7, a
   post-hoc coverage cutoff is a powerful lever. **The coverage threshold should be registered in
   advance alongside P5b-B2**, and the correlation reported at two or three fixed cutoffs so
   sensitivity is visible.

5. **Sub-exp C's teleportation signature has no stated statistic.** "Non-adjacent probability
   jumps" is a qualitative description; making it a falsifiable prediction needs a definition —
   e.g. the ratio of the behaviour-side step $d_{M_y}$ across a merge boundary to the median
   step at non-merge boundaries, with the null from `label_permutation` over which boundaries are
   labelled merges. P5b-C1/C2/C3 exist as prediction IDs; the operational definition behind them is
   not in the design doc, and C is the phase's only genuinely novel prediction — it deserves the
   sharpest formulation, not the vaguest.

6. **The behaviour manifold inherits Group E's lens problem if it is ever built from anything but
   the final head.** `math-5.md` §6.2: a tuned lens trained to match the final layer decodes the
   *eventual output* at every depth. If $M_y$ were fit to tuned-lens distributions rather than the
   real head's, **the behaviour manifold would be nearly degenerate by construction** — every layer
   reporting the same output-like distribution — which would depress $M_y$'s distance spread and
   could drive $r_{\rm manifold}$ either way depending on how the spline absorbs it. `logit_cache`
   should be pinned to the true unembedding, and the artifact should record which readout produced
   the distributions.
