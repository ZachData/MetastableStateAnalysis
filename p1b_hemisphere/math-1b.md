# Phase 1b — MATH (study notes)

## 0. What this document is

Companion to `math-1.md`. Same purpose: derive the mathematics, state what the code literally
computes, and record what the two together do and don't license. `design-1b.md` gives the
rationale and `status-1b.md` the verdicts; this fills in the derivations neither has room for.

**Read `math-1.md` §1 first.** Everything here assumes the idealized particle model, the
sphere/LN frame distinction, and the Fiedler machinery of §7 there.

Phase 1b is unusual in this project in one respect worth flagging up front: **its own status
document retracts two of its five verdict rows and reinterprets two more.** The code has moved
ahead of the results, nothing has been rerun, and the reason the retractions happened is
mathematically interesting rather than merely procedural — in two separate places a test was
constructed whose positive outcome was close to unreachable given what another test had already
established. §4.4 and §7.3 make those unreachability arguments exact, which neither existing
document does; the sharpened versions turn out to be *inequalities with explicit thresholds*
rather than the informal "essentially cannot" the current docs state, and the thresholds are
measurable.

---

## 1. The question, and why it is a question

### 1.1 Two incompatible geometries

Phase 1 observed that the spectral eigengap heuristic on the token Gram's normalized Laplacian
returns $k = 2$ on long prompts — an apparently dominant bipartition — while HDBSCAN
simultaneously finds 30–60 local clusters at the same layer. The Fiedler vector's sign pattern
defines that bipartition. Phase 1b asks what it is.

The reason this matters beyond bookkeeping is that the paper's convergence theorems have a
**hemisphere hypothesis**, and the naive reading of "$k=2$ bipartition" is its negation:

- **Cone-collapse (the paper's hypothesis).** There exists $w$ with $\langle w, x_i\rangle > 0$
  for *every* token $i$ — all tokens sit inside one open half-space. Under this condition
  (Lemma 6.4, feeding Theorem 6.3), the dynamics converge exponentially to a single point.
- **Split (the naive reading of $k = 2$).** Two populated, antipodally-separated half-spaces,
  each internally compact.

These are **mutually exclusive**, so testing both as an exhaustive pair is what makes the result
a clean falsification rather than an ambiguous partial finding. That was the design intent. §4.4
shows the two tests as originally implemented were much closer to one test than the design
believed, which is the phase's central methodological lesson.

### 1.2 Why the cone condition is a condition on the *initial data*

Worth making explicit, because it is the cleanest structural fact in this corner of the theory.
Lemma 6.4's proof uses **only the positivity of the attention weights**. From `math-1.md` §1.4:
softmax output is strictly positive, unconditionally, for any finite logits — no assumption
about $Q$, $K$, or the content of the prompt. So if all tokens start in an open half-space
$\{x : \langle w, x\rangle > 0\}$, then for every $i$,

$$
\frac{d}{dt}\langle w, x_i\rangle \;=\; \Big\langle w,\, P^\perp_{x_i}\Big(\sum_j a_{ij}x_j\Big)\Big\rangle
$$

and because each $x_j$ has $\langle w, x_j\rangle > 0$ and each $a_{ij} > 0$, the interaction
term pulls every particle *further* into the half-space; the cone is forward-invariant and
contracts. **The hemisphere hypothesis is therefore entirely a condition on the embedding
layer's output**, not on the learned attention weights at all. That is what makes it cheap to
test and what makes a *failure* interesting: infeasibility would mean the embedding layer is
doing something specific to escape a regime that otherwise forces exponential collapse.

### 1.3 Wendel's theorem: the boolean is nearly free

The reason the boolean version of this test is close to vacuous is a classical result (the
paper's Theorem 6.7). For $n$ points drawn i.i.d. from any distribution symmetric about the
origin in $\mathbb R^d$, the probability that they all lie in *some* half-space through the
origin is

$$
P_{n,d} \;=\; 2^{-(n-1)}\sum_{k=0}^{d-1}\binom{n-1}{k}
$$

For $n \le d$ the sum is the entire binomial expansion, so $P_{n,d} = 1$ **exactly**. The
textbook sanity check the codebase uses is $n = 3$, $d = 2 \Rightarrow 0.75$.

Pythia runs at $d = 1024$ with $n \le 512$: every prompt satisfies $n \le d$, so a separating
half-space exists with probability 1 *for random points*, before any transformer is involved.
Hence `PREDICTIONS.md`'s framing of **P-H1**: the prediction is stated as near-certain *on
purpose*, and

> The interesting outcome is therefore the failure or the near-failure: infeasibility, or
> feasibility with a margin $\max_w \min_i \langle x_i, w\rangle$ near zero. **Report the
> margin, not just the boolean.**

Measured on i.i.d. clouds at $d = 1024$: margin $0.221$ at $n = 20$, $0.030$ at $n = 512$. The
margin *shrinks* as $n \to d$, which is the informative direction — so the reportable object is
the margin and the layer at which it first crosses zero.

---

## 2. The Fiedler axis: what the object actually is

### 2.1 Token space vs activation space

The Fiedler vector $f \in \mathbb R^n$ lives in **token space** — one coefficient per token. It
is not directly comparable across anything: two prompts have different $n$; two checkpoints of
the same prompt have Fiedler vectors whose coordinates coincide only by accident of
tokenization. What *is* comparable is its image in activation space,

$$
a \;=\; \frac{X^\top f}{\lVert X^\top f\rVert}, \qquad X \in \mathbb R^{n\times d}
$$

the coefficient-weighted combination of token directions — the direction along which the sign
pattern is realized geometrically. This is the object commensurable with a principal component,
an OV eigenvector, or the same axis at a different checkpoint (`axis_identity.py`).

### 2.2 The axis is mean-orthogonal by construction

With $L = I - D^{-1/2}AD^{-1/2}$ and $D = \mathrm{diag}(A\mathbf 1)$, the trivial eigenvector is
$D^{1/2}\mathbf 1$ with eigenvalue 0:

$$
L\,D^{1/2}\mathbf 1 = D^{1/2}\mathbf 1 - D^{-1/2}A\mathbf 1 = D^{1/2}\mathbf 1 - D^{-1/2}D\mathbf 1 = 0
$$

The Fiedler vector is the *second* eigenvector, hence $f \perp D^{1/2}\mathbf 1$. When degrees
are roughly uniform this is approximately $f \perp \mathbf 1$, i.e. $\sum_i f_i \approx 0$ — so
in $X^\top f = \sum_i f_i x_i$, whatever component **every** token shares cancels.

**This is why the first version of `axis_identity.py` was structurally unable to find anything.**
It asked whether the axis is the *mean token direction*; measured $|\cos(a, \bar x)|$ came out
between 0.000 and 0.085 across every fixture — not an empirical finding but a restatement of the
orthogonality above. The module now retains $\cos(a,\bar x)$ **as a degeneracy diagnostic
rather than a verdict**: it *should* be ≈ 0, and a materially non-zero value (tolerance 0.5)
signals a disconnected graph, a degenerate eigenspace, or poor convergence, meaning the axis at
that layer should not be trusted at all.

The docstring is explicit that shipping the original branch "would have repeated exactly the
defect this revision flags in `bipartition_detect.classify_regime`" — the same error caught
twice in one phase, which is why §4.4 below is worth doing properly.

### 2.3 The real redundancy question, and its baseline

Since the axis is mean-orthogonal, the commensurable comparison is against **centered** PC1 and
the centered top-$k$ PC subspace. The question is sharp and cheap: *transformer residual streams
are strongly anisotropic; if the Fiedler axis is (up to sign) PC1, then "leading variance
direction" is not an interpretation of the result — it is the whole result*, and every
downstream use of the axis as a probe feature (Phase 5's hemisphere centroids, Phase 6's
Fiedler-difference vector) is rediscovering PC1 under a more expensive name.

The baseline matters as much as the number. For a uniformly random unit $u$ and any fixed unit
$v$ in $\mathbb R^d$, $\langle u,v\rangle$ has mean 0 and variance $1/d$, so

$$
\mathbb E\big|\langle u,v\rangle\big| = \sqrt{\tfrac{2}{\pi d}} \approx \frac{0.798}{\sqrt d},
\qquad \mathrm{sd} = \frac{1}{\sqrt d}
$$

The module reports $1/\sqrt d$ alongside every cosine, for the obvious reason: at $d = 1024$,
$1/\sqrt d \approx 0.031$, so a raw $|\cos| = 0.3$ is ~10σ from chance — but without the
baseline printed, "0.3" reads as "not much alignment." On synthetic fixtures the axis is
frequently PC1 to within $|\cos| \ge 0.9$; if that reproduces on real runs, the handoff note to
Phase 5 is that it "is using PC1 under a more expensive name and should say so."

### 2.4 Two Laplacians, one premise

`extract_bipartition_spectrum` originally built its own normalized Laplacian with a hardcoded
$10^{-4}/n$ connectivity floor, while `core.metrics.fiedler_and_eigengap` — the function Phase 1
actually ran — builds one **without**. Phase 1b's entire premise is explaining Phase 1's $k{=}2$
result, and it was doing so **on a different graph**.

The floor is defensible in itself (clipping negative Gram entries can disconnect an antipodal
graph entirely, leaving $\lambda_2 = 0$ and a degenerate eigenspace), which is precisely why it
now has to be a *recorded parameter of a single shared implementation* rather than a silent
difference between two. `CONNECTIVITY_FLOOR = 1e-4` preserves 1b's prior numerics; passing 0.0
reproduces Phase 1's graph exactly. **Any comparison between a Phase 1 `spectral.json` and a
Phase 1b run must state which floor each used.**

---

## 3. The cone-collapse test as a linear program

### 3.1 Formulation

The containment question — *does there exist $w$ with $\langle x_i, w\rangle > 0$ for all $i$?*
— is exactly a linear feasibility question, which is why the code solves an LP rather than a
heuristic. To get a *quantity* rather than a boolean, maximize the worst-case margin:

$$
\max_{w,\;\gamma}\ \gamma \quad\text{s.t.}\quad \langle x_i, w\rangle \ge \gamma\ \ \forall i,
\qquad w \in [-1,1]^d
$$

In `cone_margin_lp` this is assembled with decision vector $(w,\gamma) \in \mathbb R^{d+1}$,
$c = (0,\dots,0,-1)$ (minimize $-\gamma$), and $A_{ub}(w,\gamma) \le 0$ with
$A_{ub} = [-X \mid \mathbf 1]$, i.e. $-Xw + \gamma \le 0$. Solved with HiGHS.

$$
\gamma^\ast > +\texttt{tol} \Rightarrow \textbf{cone\_collapse}, \qquad
|\gamma^\ast| \le \texttt{tol} \Rightarrow \textbf{borderline}, \qquad
\gamma^\ast < -\texttt{tol} \Rightarrow \textbf{split}
$$

with $\texttt{tol} = 10^{-4}$.

**Why the box constraint is needed at all.** Without a bound on $w$, the problem is unbounded
whenever it is feasible: if $(w,\gamma)$ is feasible then so is $(\lambda w, \lambda\gamma)$ for
any $\lambda>0$. Some normalization must be imposed, and it must be one that keeps the problem
*linear*. The natural choice $\lVert w\rVert_2 \le 1$ is a second-order cone constraint, not
linear; $\lVert w\rVert_\infty \le 1$ is a box, which keeps the whole thing an LP. This is a
sound engineering choice and it has a consequence the code does not currently account for — §7.1.

**The degenerate-$w$ case.** The LP can achieve $\gamma = 0$ with $w = 0$, which satisfies every
constraint while supplying no separating direction. That happens exactly when the token set
spans both hemispheres — i.e. it is the *split* case wearing a borderline mask. The code detects
$\lVert w\rVert < 10^{-4}$ together with $\gamma < \texttt{tol}$ and pushes $\gamma$ below
$-\texttt{tol}$ so the classifier says `split` rather than `borderline`. Worth understanding as
a genuine feature of the geometry rather than a numerical patch: $w = 0$ *is* the LP's honest
answer that no direction works.

### 3.2 Binding constraints, and the sink question

$\gamma^\ast$ is determined by the tokens whose constraint is **tight** at the optimum:
$\langle x_i, w^\ast\rangle = \gamma^\ast$ to within `BINDING_REL_TOL`. `binding_tokens` records
their original indices (correctly mapped back through any `drop_indices` filtering), and
`drop_indices` lets a caller re-solve without them.

This matters specifically because of `math-1.md` §2.5: **on GPT-NeoX, position 0 is an attention
sink and a high-norm outlier**, and is therefore a plausible sole determinant of the enclosing
half-space. Running with and without it is what turns "a cone exists" into "a cone exists, and
it is / is not held up by one token" — an entirely different claim about the model. Nothing in
the pre-revision run recorded this.

### 3.3 PCA is sound in exactly one direction

The original docstring claimed the cone question is "invariant under orthogonal projections."
It is not, and the asymmetry is the useful part.

**Reduced-space feasibility lifts.** Let $X_r = XV_k^\top$ with $V_k$ the top-$k$ right singular
vectors (rows orthonormal). Given $w_r$ feasible in the reduced problem, set $w = V_k^\top w_r$.
Then

$$
Xw = XV_k^\top w_r = X_r w_r
$$

**exactly** — every inner product is preserved, so if all reduced inner products are positive,
so are all full-space ones. A `cone_collapse` verdict found under PCA is therefore sound as it
stands.

**Full-space feasibility need not survive projection.** A full-space witness's component
orthogonal to the retained subspace is discarded, so a `split` verdict under PCA may be a
projection artifact. `escalate_on_split=True` re-solves at full $d$ whenever the reduced problem
returns anything other than `cone_collapse`, and flags `escalated=True`. **Only the direction
that can lie gets escalated** — which is the right asymmetry and is cheap because the
interesting case is rare.

**A refinement the docstring gets slightly wrong, and it matters for §7.1.** The *verdict* lifts,
but the *margin value* does not. If $w_r \in [-1,1]^k$ then the lifted $w = V_k^\top w_r$ has

$$
|w_j| = \Big|\sum_t V_k[t,j]\,w_r[t]\Big| \le \sqrt k\,\lVert V_k[:,j]\rVert_2 \le \sqrt k
$$

so $w$ may sit outside the unit box and must be rescaled by up to $1/\sqrt k$ to be feasible,
scaling $\gamma$ by the same factor. Positivity is scale-invariant, so the *sign* — hence the
verdict — is preserved; the numerical margin is not. Any statement of the form "compare
`normalized_margin` across layers/models/checkpoints" inherits this.

---

## 4. Regime classification: the two vocabularies

### 4.1 What Block 0 measures per layer

Given the sign partition $a_i = \mathbb 1[f_i \ge 0]$ of the Fiedler vector:

| Quantity | Definition | Reading |
|---|---|---|
| `bipartition_eigengap` | $(\lambda_3-\lambda_2)/\lambda_3$ | high ⟹ the $k{=}2$ partition dominates the spectrum |
| `centroid_angle` | $\arccos\langle \hat c_A, \hat c_B\rangle$ | absolute angular separation of the two halves |
| `within_half_ip` | mean cosine over the strict upper triangle of each half's self-Gram | internal compactness, per half |
| `between_half_ip` | mean of $X_A X_B^\top$ (all cross pairs) | $<0$ antipodal, $\approx 0$ orthogonal, $>0$ same-leaning |
| `separation_ratio` | between / mean(within) | $<1$ real contrast; $\approx 1$ none; $<0$ antipodal |
| `fiedler_boundary_frac` | fraction with $|f_i| < 0.30\,\mathrm{sd}(f)$ | $\approx 0$ bimodal; $\approx 1$ everyone hugs the boundary |
| `minority_fraction` | $\min(|A|,|B|)/n$ | is the second half populated at all |
| `clip_fraction` | fraction of off-diagonal Gram entries negative *before* clipping | how much geometry the clipping threw away |

`clip_fraction` is computed **unconditionally** now — it was previously filled only when
`clip_negative=True`, so the no-clipping case reported `nan` and any assertion about it passed
vacuously. The fraction of negative off-diagonal entries is a property of the geometry, not of
what was subsequently done to it.

### 4.2 The antipodal classifier (legacy vocabulary)

```
collapsed          minority < 0.05, or any input nan
weak_bipartition   minority ∈ [0.05, 0.10)  OR  centroid_angle < π/2
strong_bipartition minority ≥ 0.10, centroid_angle ≥ π/2, within_half_ip ≥ 0.30 both halves
diffuse            minority ≥ 0.10, centroid_angle ≥ π/2, but one half has within_half_ip < 0.30
```

### 4.3 The relative classifier (cone-compatible)

```
collapsed  minority < 0.05, or any input nan
separated  both halves populated AND separation_ratio ≤ 0.90
graded     separation_ratio ∈ (0.90, 0.98)
uniform    separation_ratio ≥ 0.98
```

This asks **nothing about absolute angle**, so it stays informative inside a single open
hemisphere. The distinction it can draw that the antipodal one cannot is exactly the geometry
Phase 1b found and had no label for: *"separated" and "not antipodal" simultaneously*.
Calibration: two clusters 60° apart with separation ratio 0.45 read as `weak_bipartition`
antipodally and `separated` relatively.

### 4.4 Why "0% strong bipartition" was nearly uninformative — made exact

Both `design-1b.md` and `status-1b.md` state that under cone-collapse "two centroids inside a
single open half-space **essentially cannot** be $\pi/2$ apart." That is the correct intuition
and it is not a theorem as stated — two vectors can both have positive inner product with $w$
and still be nearly antipodal (take $w = e_1$, $c_A \propto (\varepsilon, 1, 0)$,
$c_B \propto (\varepsilon,-1,0)$). What *is* true is a quantitative bound, and since the phase
now reports a margin, the bound is checkable.

**Claim.** Let $\hat w$ be a *unit* witness with $\langle \hat w, x_i\rangle \ge m \ge 0$ for every
unit-norm token, and let $\hat c_A, \hat c_B$ be the normalized half-centroids. Then

$$
\boxed{\ \cos\angle(\hat c_A, \hat c_B)\ \ge\ 2m^2 - 1\ }
\qquad\text{hence}\qquad
\angle(\hat c_A,\hat c_B) \le \arccos(2m^2-1)
$$

*Proof.* Each centroid $c = \frac{1}{|A|}\sum_{i\in A} x_i$ satisfies $\langle \hat w, c\rangle =
\mathrm{mean}_i\langle \hat w, x_i\rangle \ge m$, and $\lVert c\rVert \le 1$ by the triangle
inequality on unit vectors, so $\langle \hat w, \hat c\rangle \ge m$. Write
$\hat c_A = m_A\hat w + \sqrt{1-m_A^2}\,u_A$ with $u_A \perp \hat w$ unit, similarly for $B$.
Then

$$
\langle \hat c_A, \hat c_B\rangle = m_Am_B + \sqrt{1-m_A^2}\sqrt{1-m_B^2}\,\langle u_A,u_B\rangle
\;\ge\; m_Am_B - \sqrt{1-m_A^2}\sqrt{1-m_B^2}
$$

The right side is increasing in each of $m_A, m_B$ (differentiate: $\partial_{m_A} = m_B +
m_A\sqrt{1-m_B^2}/\sqrt{1-m_A^2} > 0$), so it is minimized at $m_A = m_B = m$, giving
$m^2 - (1-m^2) = 2m^2-1$. $\square$

**Consequence.** `strong_bipartition` requires $\angle \ge \pi/2$, i.e. $\cos \le 0$, which by
the bound requires

$$
2m^2 - 1 \le 0 \iff m \le \tfrac{1}{\sqrt 2} \approx 0.707
$$

So the antipodal classifier is **provably unreachable whenever the L2-normalized cone margin
exceeds $1/\sqrt 2$**, and increasingly constrained below that. This is a strictly better
statement than "essentially cannot," for three reasons:

1. It is a *falsifiable* structural claim rather than an intuition.
2. It makes the two tests' dependence **measurable** — report $m$ per layer and the reader can
   see immediately whether Block 0's null carried any information at that layer.
3. It says the two tests are *not* literally the same test. Note the bound only ever **forbids**
   the classifier (when $m > 1/\sqrt2$); below that threshold the obstruction simply lifts, which
   is not the same as establishing reachability. On i.i.d.
   clouds at $d{=}1024$ the measured margins are $0.221$ ($n{=}20$) and $0.030$ ($n{=}512$) —
   **well below** the threshold. So at face value the antipodal classifier should have been
   reachable, and "0% strong bipartition" may be carrying more information than the retraction
   credits it with. That tension is not resolved anywhere and it turns on which margin is being
   compared — see §7.1, which is the reason it cannot currently be settled from the artifacts.

### 4.5 The knock-on: Block 1's zero events were partly foreclosed

Block 1's persistence and birth/collapse/swap detection were hardcoded to the
`strong_bipartition` label. If that label is unreachable, **every persistence length was 0 by
construction while appearing measured** — the same defect class, one module downstream.
`regime_key="regime_relative"` runs the identical machinery on the reachable vocabulary. The
identity-persistence verdict itself (mean match overlap > 0.5) does not depend on the regime
label and stands.

---

## 5. Tracking the axis across layers

### 5.1 Sign alignment is a $k=2$ assignment problem

The Fiedler vector's overall sign is arbitrary at every layer (if $f$ is an eigenvector so is
$-f$), so labels must be aligned before anything can be tracked. At $k=2$ the two possible
alignments are exactly *identity* and *flip*, so the Hungarian assignment over labels
$\{0,1\}$ **is** the global-sign-flip decision — there is no hemisphere-specific matching logic
to own, and Block 1 delegates to `cluster_tracking.match_layer_pair` (`math-1.md` §9.2). The
reported *score* stays local: the mean of the two halves' Jaccards, which is what
`IDENTITY_THRESHOLD` and every existing result are stated against. A `matcher="local"` path
keeps the previous in-module comparison so the delegation stays checkable against it.

**The hazard wiring this up surfaced is worth generalizing.** Exact Jaccard ties let the
assignment solver return *either* pairing — observed on 4 of 500 random label pairs — and
because trajectories are built by anchor chaining, a single arbitrary flip propagates through
the remainder of the run. Not known to have fired in the recorded run; the tie-break is now
pinned. The general form: **any downstream chaining of a solver's output inherits that solver's
tie-breaking as a correctness assumption**, and ties are not rare in integer-valued overlap
measures.

### 5.2 Axis rotation

$$
\theta_L = \arccos\big|\langle v_L,\, v_{L+1}\rangle\big|
$$

The absolute value is doing the same sign-invariance work as the alignment above. Per-transition
$\theta_L$, its nan-safe cumulative sum, per-token `crossing_count`, and a sustained-rotation
"drift" detector make up Block 1's event vocabulary (`birth`, `collapse`, `swap`, `shear`,
`drift`), each cross-referenced against Phase 1's merge events and energy-violation layers.

**The checkpoint version is the one that matters and did not exist.** `compute_axis_rotation`
measures rotation between *adjacent layers within one model*. The identical statistic across
*checkpoints at a fixed layer* is what `PREDICTIONS.md` claim (b) needs — and
`axis_settling_step` (when the Fiedler axis reaches its trained direction) is **the only
quantity anywhere in this project that tracks the axis's *direction* over training, as opposed
to $\lambda_2$'s magnitude.** The design had no training-step axis at all, so a Pythia pilot
would render as $N$ unrelated models; `aggregate_by_checkpoint` groups families and reports
against $\log_{10}(\text{step}+1)$.

### 5.3 Per-token membership

Block 2 produces, per token: `hemisphere_trajectory` (aligned label per layer),
`stability_score` (fraction of valid transitions with no switch), `border_index` (mean of
$|f_L(i)|/\mathrm{mean}_L|f_L|$ — large means deep inside one side), `first_stable_layer`, and
`dominant_hemisphere`.

The HDBSCAN **nesting test** asks whether local density clusters respect the global bipartition:
with $r_c$ the fraction of cluster $c$'s tokens in hemisphere 0,

$$
r_c < \texttt{tol} \Rightarrow \texttt{nested\_B},\quad
r_c > 1-\texttt{tol} \Rightarrow \texttt{nested\_A},\quad
|r_c - 0.5| < \texttt{half\_width} \Rightarrow \texttt{mixed},\quad
\text{else } \texttt{partial}
$$

"Near chance" means clusters straddle the bipartition as often as a random assignment of their
members would — i.e. the two scales of structure are independent.

### 5.4 `border_vs_noise`: two existing quantities, never crossed

The per-token distance from the Fiedler boundary and HDBSCAN's noise labels both already
existed; nothing had ever crossed them. `border_vs_noise` computes a **rank AUC** of
`border_index` against the noise indicator — i.e. *is the unclustered population the boundary
population?*

This is Phase 5c's object of study answered with quantities already on disk, and it is a good
example of a cheap result hiding in the gap between two modules. It also gives "unclustered" a
candidate *geometric* definition rather than leaving it as a clusterer-defined primitive — worth
checking before any later phase builds on it.

---

## 6. Null models: the part the original run skipped

`status-1b.md` R3 is blunt: the cone-collapse verdict was "a binary regime label" and
"$n$ points in $d$ dimensions admit a separating witness for free unless they positively span."
The run used `pca_n_components=64` on prompts of 100–200 tokens, and could not distinguish
transformer geometry from dimension counting. Two matched nulls now exist.

**Shuffled-dimension null** (`core/nulls.py`). Each feature dimension is independently permuted
across tokens, then rows are re-normalized onto the sphere. This preserves every dimension's own
marginal distribution while destroying all cross-token joint geometry. It answers: *is the
containment more than the marginals?*

**Uniform-sphere null.** $n$ points drawn uniformly on $\mathbb S^{d-1}$ at matched $(n,d)$. The
pure dimension-counting control. It deliberately carries *none* of the observed cloud's
structure — not even its marginals — which is why it lives in `cone_collapse.py` rather than
`core/nulls.py`, whose constructions are all shuffles of observed data.

Both nulls are evaluated **through the identical LP path, PCA setting included**
(`normalized_margin_of` is the `metric_fn` handed to the null machinery). Like-for-like is the
whole point: a null computed at a different $d_{\text{eff}}$ answers a different question.

**A degeneracy the code handles well and that generalizes.** A uniform null at matched $(n,d)$
is frequently *degenerate* — every draw lands on the same sentinel margin because every draw
positively spans the space — so $\mathrm{sd} \approx 0$ and the $z$-score is correctly `nan`.
That is the control's answer, not its failure, but a `nan` is useless in a table. So the
**fraction of null draws that are themselves cone-collapsed** is reported alongside: *"observed
collapses, 0/N matched draws collapse"* is the readable form of a `nan` $z$. On synthetic data
the two nulls discriminate cleanly (100% of shuffled-dimension draws cone-collapse, 0% of
uniform-sphere draws do), so the test has power — it simply was never run.

**The reportable quantity is `normalized_margin` and its $z$/percentile against the nulls, not
the regime label.** A 100% cone-collapse fraction sitting at the uniform null's median is a
statement about dimension, not about transformers.

---

## 7. Open questions

### 7.1 `normalized_margin` is not scale-free in the way it is documented to be — likely a live defect

`cone_margin_lp` documents `normalized_margin` = $\gamma^\ast/\max_i\lVert x_i\rVert$ as "the
scale-free quantity; compare this across layers, models and checkpoints, never `cone_margin`."
I don't think that survives inspection, for two compounding reasons.

**(a) The witness norm is not divided out, and the box's L2 radius grows as $\sqrt{d_{\text{eff}}}$.**
The LP constrains $w \in [-1,1]^{d}$, and its optimum generally sits near a vertex, so
$\lVert w^\ast\rVert_2$ can be as large as $\sqrt{d_{\text{eff}}}$. The geometrically meaningful
margin — the $m$ of §4.4, and the quantity Wendel-style reasoning is about — is

$$
m = \frac{\gamma^\ast}{\lVert w^\ast\rVert_2 \cdot \max_i\lVert x_i\rVert}
$$

Dividing by the row norm alone leaves a factor that scales roughly like $\sqrt{d_{\text{eff}}}$.

**(b) $d_{\text{eff}}$ varies across prompts, by a lot.** The reduction is
`k = min(pca_n_components, max(n-1, 1), d)`. With `pca_n_components = 64`:

| prompt length | $d_{\text{eff}}$ | $\sqrt{d_{\text{eff}}}$ |
|---|---|---|
| $n = 20$ | **19** (the $n-1$ term binds) | 4.36 |
| $n = 242$ | 64 | 8.00 |
| $n = 512$ | 64 | 8.00 |

So the short prompt is normalized differently from the long ones by a factor near 2, before any
geometry is involved. **This is structurally the same defect as Phase 1's Fiedler length
confound** (`math-1.md` §8.3), where an $n$-dependent baseline was averaged across prompts of
different lengths — and it is on the exact quantity `status-1b.md` R3 nominates as the one a
falsification table should adjudicate.

Three things follow, and none is currently done: report $\lVert w^\ast\rVert_2$ and
$d_{\text{eff}}$ next to every margin; either divide the witness norm out or state the margin
per-$d_{\text{eff}}$; and re-check §4.4's $1/\sqrt2$ threshold against the corrected $m$, since
the face-value comparison in §4.4 used the *published* margins and may flip sign under the
correction. The $z$-scores against the nulls are **unaffected** — nulls are computed at matched
$(n,d)$ through the same path — which is a good argument for treating the null-referenced
numbers as the primary output and the raw margin as diagnostic.

### 7.2 Does the axis attenuate in the LN frame? (blocked, and it is the sharpest test available)

Phase 1b's conclusion is that the $k{=}2$ eigengap marks an **anisotropy axis**, not a
separator. That conclusion makes a direct prediction: the L2-sphere frame keeps the cloud's mean
offset, while LN centers it (`math-1.md` §3.3), so **if the axis is anisotropy, it should
attenuate in the frame attention actually reads.**

`bipartition_detect` already threads a `FrameSpec` through `frame_gram`, so the machinery
exists. It is blocked only because `run_1b`'s entry point doesn't supply per-model LN parameters
to `apply_frame` (`status-1b.md` open blocker 4). This is the phase's own conclusion testable
with code that already exists, and it is one plumbing change away.

### 7.3 Is the relative classifier's threshold reachable, and what is *its* structural ceiling?

§4.4 derived the antipodal classifier's ceiling exactly. The obvious follow-up — which nobody
has asked — is whether `classify_regime_relative`'s `separation_ratio ≤ 0.90` has an analogous
structural bound under cone-collapse. It should: with all tokens in a cone of margin $m$, every
pairwise cosine is bounded below (roughly $\ge 2m^2-1$ by the same argument applied to
individual tokens rather than centroids), so both within- and between-half means are pushed
toward 1 and their *ratio* is compressed toward 1 as $m$ grows. If that compression is strong at
the observed margins, the relative classifier inherits a weaker version of the same defect it
was introduced to fix, and its 0.90/0.98 thresholds need deriving from $m$ rather than being
placed. **The thresholds are currently labelled "a reporting convention" in the code**, which is
honest but leaves the question open. This is a one-page derivation that would tell you whether
the replacement test is actually reachable in the regime it will be run in.

### 7.4 Layer 0 is a different object and is being averaged in

`status-1b.md` blocker 5: layer 0 is the embedding output, pre-any-LN, and is still averaged into
per-model means. Given §1.2 — the cone condition is *entirely* a condition on the embedding
layer's output — layer 0 is not a nuisance row to be averaged over, it is **the single most
theoretically load-bearing layer in the phase**. It should be reported separately by default,
and P-H1's "the layer at which the margin first crosses zero" is measured from it.

### 7.5 Does the binding-token set identify the sink, and does dropping it change the verdict?

§3.2's machinery exists but the crossing has not been run. Two specific questions: is position 0
in `binding_tokens` at most layers on Pythia, and does `drop_indices=[0]` move
`normalized_margin` materially? A "yes/yes" would mean the universal cone-collapse result is
substantially a statement about one high-norm outlier — which is a *different and more
interesting* finding than the one currently recorded, and directly connects Phase 1b to
`math-1.md` §6.2's sink argument and to the `pos0_policy` ledger field.

### 7.6 Nothing has been rerun

Every result in `status-1b.md` predates the revision. Two rows are retracted, two reinterpreted,
and the ALBERT row is **empty rather than inconclusive** — a path-construction mismatch
(`{model}_{prompt}_d{depth}` vs Phase 1's `{model}_{depth}iter_{prompt}`) meant no ALBERT
extended run ever resolved, so `hdbscan_labels` never loaded and the nesting test had nothing to
test. Fixed via `p1_io.find_phase1_run_dir`, unrerun. The pure-numpy blocks are covered by
tests; the model-touching paths (`--fast`, `--from-phase1`, `write_manifest`, ALBERT extraction)
are unverified.

---

## 8. Code map

| File | Block | What it computes |
|---|---|---|
| `bipartition_detect.py` | 0 | Laplacian spectrum + Fiedler vector via the shared `core.metrics` implementation; within/between-half inner products; separation ratio; boundary fraction; centroid angle; both regime classifiers; frame-aware throughout |
| `hemisphere_tracking.py` | 1 | Sign alignment (delegated to the Hungarian matcher), axis rotation, crossings, persistence, event detection, Phase 1 cross-referencing |
| `hemisphere_membership.py` | 2 | Per-token trajectories, stability, border index, HDBSCAN nesting classes, `border_vs_noise` AUC |
| `cone_collapse.py` | 3 | The margin LP, regime classification, binding tokens, PCA escalation, both matched nulls |
| `axis_identity.py` | A | Token-space → activation-space axis map; redundancy against centered PC1 and top-$k$ subspace; $1/\sqrt d$ baseline; mean-orthogonality diagnostic; cross-checkpoint axis rotation and settling step |
| `hemisphere_mechanism.py` | 5 | Axis alignment vs OV/PCA/embedding/heads — needs Phase 2 artifacts, runs silently if absent |
| `hemisphere_semantics.py` | 6 | Token-attribute contingency and mutual information — same Phase 2 dependency |
| `core/nulls.py` | — | Shuffled-dimension and label-permutation nulls, `sigma_from_null` |

Reused rather than duplicated: `fiedler_tracking.py`, `rotation_hemisphere.py`, and
`core.metrics.fiedler_and_eigengap`.

---

## 9. The standing result, stated carefully

**As recorded (pre-revision):** cone-collapse at 100% of layers in every model; strong
bipartition at 0%; the Fiedler axis stable and identity-preserving across layers. Paper
alignment = `cone_collapse`. The Phase 1 $k{=}2$ eigengap is a real, stable **anisotropy axis**,
not an antipodal bipartition — all tokens remain in one open hemisphere throughout.

**What survives review:** the *direction* of that conclusion, and the identity-persistence
verdict. The reconciliation it offers is genuine and is the phase's contribution — Phase 1's
empirical $k{=}2$ and the paper's hemisphere hypothesis are not in conflict, because a dominant
Fiedler axis inside a cone is not a separator.

**What does not survive:** the magnitude. The cone-collapse fraction is unquantified against any
null (§6), the margin it is stated in is arguably not comparable across the prompts it was
averaged over (§7.1), the "0% strong bipartition" null is structurally constrained by the
cone-collapse result in a way that is now exactly characterized but not yet measured (§4.4), the
ALBERT row is empty rather than inconclusive, and Block 1's zero-event count was partly
foreclosed by construction (§4.5).

The forward-looking corollary is unchanged and is what later phases actually consume: **do not
treat the bipartition sign as a two-class label.** The axis, as a continuous projection, remains
a legitimate candidate feature — subject to §2.3's caveat that it may simply be PC1.
