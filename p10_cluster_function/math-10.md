<!-- p10_cluster_function/math-10.md -->
# Phase 10 — MATH (derivations, checked)

**Five results, four of them symbolically checked, three of them corrections to
things this repository currently asserts.** Written 2026-09-20 alongside
`lit-10.md`'s second scan, under `CLAUDE.md`'s rule that a closed-form
derivation can be checked mechanically and should be.

Checks, all passing, each stating on its face what it does *not* prove:

| file | checks | subject |
|---|---|---|
| `tools/math_checks/causal_mask_attention_baseline.py` | 9/9 | §1, §2 |
| `tools/math_checks/cone_margin_gamma_gradient.py` | 5/5 | §3 |
| `tools/math_checks/ari_size_profile_null.py` | 6/6 | §4 |
| `tools/math_checks/parking_scaling_slope.py` | 8/8 | §5 |

---

## 1. The causal mask puts a three-orders-of-magnitude tilt under the attention flip

`attention-10.md` §2.2 argued that received attention has a built-in advantage
for early tokens and that nothing divides it out. Here is the size of it.

Positions `0 .. n−1`; query `i` attends over keys `0 .. i`. **Content-free**
means uniform within the row, `a[i,j] = 1/(i+1)` for `j ≤ i`. Then

```
    received(j) = Σ_{i=j}^{n−1} 1/(i+1) = H_n − H_j
```

and `Σ_j received(j) = n` exactly, so **the layer mean is exactly 1 and
`received(j)` *is* the "× layer average" quantity the flip reports.** The
baseline and the measurement are directly comparable with no rescaling.

At the battery's `n = 264`:

| position | baseline, × layer average |
|---|---|
| 0 (the sink) | **6.155** |
| median (132) | 0.691 |
| last (263) | **0.0038** |

**A span of about 1 600× from first to last, before any content enters.** The
observed contrast is 1.6× against 0.5× — a factor of 3.2, sitting inside a
structural gradient three orders of magnitude wide.

Two consequences, both checked:

- **The observed numbers are reproducible with zero content.** The content-free
  baseline already equals 1.6× at position ≈ **53** and 0.5× at position ≈
  **160**. A partition whose unclustered members average early and whose
  clustered members average late reproduces the flip exactly, with no learned
  behaviour anywhere.
- **The two reported numbers pin the population split.** Since the populations
  partition the same tokens and the mean is 1, `f·1.6 + (1−f)·0.5 = 1` gives
  `f = 5/11 ≈ 0.4545`. `status-5c.md` independently reports 40–50 % unclustered.
  That is an internal-consistency check, **not** evidence of content — it is
  what any partition into a 1.6× and a 0.5× population must satisfy.

**So `attention-10.md`'s row A0 is not a tidying step. Until the mask baseline
is divided out, the flip is not known to be a measurement of anything.**

The same argument gives the row-side baseline for free: content-free, row `i`
has entropy exactly `log(i+1)` and the layer mean is `log(n!)/n` — **4.590 nats
at `n = 264`.** `attention_entropy_per_head` is stored for every layer of the
sweep, and its raw value is mostly position. **The deviation from `log(i+1)` is
the content**, and nothing reads it that way.

---

## 2. Under a causal mask, "high `Z` = sink" reverses — and Pythia is masked

`math-1.md` §1A.6 makes an identification the project has never measured:

> the partition function is not noise to be normalized away — it is a metric …
> **a high-`Z` token (a sink) is one the metric makes expensive to move.**

Take `Z_{β,i} = Σ_j exp(β⟨x_i, x_j⟩)` in the concentration regime, where every
pairwise inner product is a common `γ`:

- **Unmasked** (the model §1A.6 is written in): `Z_i = n·e^{βγ}` — *position
  independent*. Any spread in `Z` is therefore content, and a high-norm token
  whose inner products with everything are large does have a large `Z`. **The
  identification is reasonable here.**
- **Masked** (Pythia): row `i` sums over `j ≤ i` only, so `Z_i = (i+1)·e^{βγ}`
  — *linear in position*, and **position 0 is the minimum**, `Z_0 = e^{βγ}`.

Meanwhile §1's `received(j) = H_n − H_j` **decreases** in position, by exactly
`1/(j+1)` per step.

> **The two structural baselines are anti-aligned.** Under a causal mask the
> sink at position 0 is simultaneously the **largest** received attention and
> the **smallest** `Z`. On the metric reading that makes it the **cheapest**
> token to move, not the most expensive.
>
> **§1A.6's identification is an unmasked-model statement.** It should be
> labelled as one.

This is not a defect — it is a sharper instrument. It says the sink occupies a
*specific corner* of `attention-10.md` §4.3's paid/received square (low `Z`,
high received), distinct from a parked particle (low both) and from a carrier
(high both), and that `Z` is what separates *pinned* from *parked* only once the
structural `(i+1)` is divided out. **Measure `Z_i/(i+1)`, not `Z_i`.**

---

## 3. The cone margin's response to a γ patch, in closed form

`plan-9.md` §4.4 claims the effect of a candidate metric patch on the cone
margin is free to compute. Here is the derivative.

Points `p_i = Γ x̂_i + b`; margin `m² = min_{λ∈Δ} ‖c(λ)‖²`, `c(λ) = Σ λ_i p_i`
(`math-1c.md` §7.3). The Phase 9 patch `Γ → Γ + ε u uᵀ` moves each point by
`ε (u·x̂_i) u`, so `c(λ,ε) = c₀(λ) + ε s(λ) u` with `s(λ) = uᵀ X̂ᵀ λ`. By
Danskin's envelope theorem at a unique minimiser `λ*`:

```
    d(m²)/dε |_{ε=0}  =  2 · (uᵀ X̂ᵀ λ*) · (uᵀ c(λ*))
```

Both factors come from stored activations plus the optimal simplex weight the
existing QP already returns. Checked against a full re-solve by central
differences at `(n,d) = (6,4)` and `(9,5)`, relative error `< 1e-9`.

Three readings:

- **It reads the binding set and nothing else.** `λ*` is supported on the
  binding tokens, so `s(λ*)` depends only on them. **A patch aimed at a subspace
  no binding token occupies has zero first-order effect on the margin, however
  large its `D`.** That is the closed-form target `plan-9.md` §4.4 claimed and
  did not have.
- **Two ways to get zero**, and they are different: `u ⊥ X̂ᵀλ*` (the raw
  weighted centroid) or `u ⊥ c(λ*)` (the frame-space one). They coincide only
  when `Γ = I, b = 0`.
- **The second-order term is `s(λ*)²‖u‖² ≥ 0`** at fixed `λ*`, so along a
  first-order-null direction the margin cannot *decrease* to second order
  without `λ*` itself moving.

Limits stated in the check: Danskin needs a **unique** minimiser, and the
degenerate case is precisely the near-zero-margin configuration `math-1c.md`
§7.2 calls the informative one — **a runner must report whether `λ*` was
unique.** It is first order at `ε = 0`, and it is the effect **at the patched
layer**; everything downstream still needs a forward pass.

---

## 4. Correction: the ARI is already centred, so the null is for variance

`plan-9.md` §2.3 and `notes-10.md` §4.4 both say a size-profile-preserving null
is needed *before any ARI is quoted*, because two labellings dominated by one
giant cluster "agree at high ARI for reasons that have nothing to do with
content."

**That is right about the Rand index and wrong about the adjusted one.** The
adjustment subtracts exactly the expectation under the permutation model with
both size profiles held fixed, so `E[ARI] = 0` under that null **by
construction**, whatever the profiles. Confirmed by Monte Carlo at three
regimes, including a 200 + 8×8 giant-cluster profile: means within 4 standard
errors of zero in all three.

**What changes is the reason, and the instruction survives it.** The null is
needed for the **variance**, not the centering — and the variance is where the
real problem was all along:

| size profile (N = 260) | null's 95th percentile |
|---|---|
| balanced, 10 × 26 | **+0.009** |
| one giant cluster, 200 + 8×8 | **+0.092** |
| giant vs balanced | **+0.002** |

**A 57× range.** An ARI of 0.05 is unremarkable under one profile and a strong
signal under another. So:

> **A fixed ARI threshold is not comparable across layers, checkpoints or
> models.** This is `compare_rungs.py`'s no-absolute-threshold rule one level up
> — and HDBSCAN's cluster-size profile changes across exactly those axes.

A separate hazard no adjustment touches, named in the check: **HDBSCAN noise is
not a cluster.** Treating `−1` as one cluster versus dropping those points gives
different ARIs on the same data, `adjusted_rand_index(ignore_noise=...)` exposes
the choice, and ~40–50 % of tokens are noise. **The design fixes it before
looking.**

---

## 5. Correction: the parking law is in β and d, not in n — and that makes it a better test

> **UPGRADED TO `[R]` 2026-09-20.** The paper has since been read in full —
> `docs/readings/2411.04990.md`. Everything in this section that was inferred
> from search summaries is confirmed, and **the paper states the `d_eff`
> rescue in §5.2 below as its own open conjecture.** Three additions are marked
> inline; a fourth result, Lemma C.1, is §5.4.

### 5.1 What the field actually says

`lit-1.md` §4 describes the Rényi-parking prediction as *"a density constant
(the Rényi constant ≈ 0.7476) and hence an expected number of occupied cells as
a function of `n`"*, and rates checking it the project's best cheap experiment.
`lit-10.md` §2's second pass finds that reading is wrong on both halves. Two
independent search summaries **[S]** give:

- **early tokens act as nuclei.** In causal models the early tokens *"act as
  'nuclei' that serve as centers for cluster formation … where particles fill up
  space and prevent others from collapsing into them."*
- **the frequency of Rényi and strong-Rényi centers is `Θ(β^((d−1)/2))`**,
  confirmed at `β^(1/2)` for `d = 2`.

So the law is a scaling in **β and dimension**, not in `n`; and there is no
0.7476 in it. Taken literally at `d = 1024` the exponent is **511.5**, and with
a measured β near 0.5 the prediction is indistinguishable from zero. **That is
the `d ≫ 1` problem `design-1.md` already records for the source paper's
Figure 3, inherited.**

**This would have wrecked an F0 registration**, and it is exactly what
`CLAUDE.md` trigger 2 exists to catch — a literature fact changing the
statistic before the wording freezes.

### 5.2 The manipulation that rescues it, and it is better than the original

Write the law with an unknown effective dimension and constant:

```
    count(n, β) = C · n^b · β^a ,        a = (d_eff − 1)/2
```

`PROJECT.md` §3.40 records that **β's unit convention is undecided and worth a
factor of 8** (the model's own `1/√head_size` logit scale, applied or not). A
factor is a constant, and **in log space a constant moves the intercept, not the
slope**:

```
    log count = log C + b·log n + a·(log β_raw − log k)
```

> **`a` is invariant under β → β/k. The convention decision gates the intercept
> and does not gate the slope.** Checked symbolically and by recovery on
> synthetic data: fitting on `β` and on `8β` returns slopes identical to `1e-9`,
> with the intercept shifting by exactly `−a log 8`.

So **a test that looked blocked on an undecided convention is not.** And what it
returns is better than a yes/no:

```
    d_eff = 2a + 1
```

the **effective dimension of the geometry the clustering behaves as**, directly
comparable to numbers already on disk:

| comparator | `d` | predicted slope `a` |
|---|---|---|
| ambient (pythia-410m) | 1024 | 511.5 |
| effective-rank plateau | ~225 | 112.0 |
| ambient participation ratio (`status-7e.md`) | 22 | 10.5 |
| the published check | 2 | 0.5 |

**Five orders of magnitude apart, so the regression measures rather than
confirms**, and any of those landing is informative. A bad log-log fit is the
other informative outcome and the design must report the fit, not only the
slope.

> **`[R]`: this is the paper's own conjecture, and proving it is open.** §5, p. 6,
> verbatim: *"For general matrices `V`, our empirical observations suggest that
> particles rapidly converge to a lower-dimensional subspace spanned by `d₁ ≪ d`
> principal eigenvectors. Consequently, **we conjecture that the number of
> meta-stable clusters should rather be `β^((d₁−1)/2)`, where the ambient
> dimension `d` is replaced by the effective dimension `d₁`.** While a rigorous
> proof of this dimension-reduction remains an open problem for future
> investigation..."*
>
> So the regression is not a workaround — **it is a direct empirical attack on a
> named open problem.** And `d₁` is not free: the paper identifies it as the
> dimension of **`L`, the top eigenspace of `V`**, which Phase 2's `sym_*` /
> `schur_*` projectors already compute for all 19 checkpoints. **Predict `d₁`
> from the OV spectrum, then test the slope against it** — that makes it
> differential rather than exploratory.

**One failure mode, checked:** `k` must be the same for every row in one
regression, and it is not — `head_size` is 64 on gpt2-large and 128 on
pythia-1.4b. A cross-model regression on raw β mixes two conventions; in the
synthetic test that pulls a true slope of 3.25 down to 1.24. **Per-model
regressions, or fix the convention first.**

### 5.4 Lemma C.1: an exact count that does not grow with `n` — and it is the carrying-capacity finding

`[R]`, App. C.4, checked in `tools/math_checks/parking_center_count.py` (8/8):

```
    average number of strong Rényi centres  =  1 / σ_{d−1}(B_δ)  ~  1/δ^{d−1}
```

proved for **any spherically symmetric measure in any dimension** (ordinary
Rényi centres are much harder above `d = 2`, where the classical `c·2π/δ` with
`c ≈ 0.75` applies). With `δ = cβ^{-1/2}` this is exactly the `Θ(β^((d−1)/2))`
frequency, so the paper's two statements are one.

**The property that matters here: it is a limit over an infinite sequence, so
the count SATURATES in `n`.** More tokens do not buy more centres.

> **Phase 1's unexplained finding is that shape.** Max simultaneously-alive
> clusters **invariant at 50–55 across all 27 checkpoints** while mean lifespan
> falls 7.0 → 4.5 and births rise 113 → 164 — a fixed capacity with rising
> turnover. `lit-1.md` grades it *"Looks new"*; **it is a saturating parking
> count**, and the 0.7476 constant lives in the `d = 2` *ordinary*-centre
> formula, not in anything this project measures.

**And inverting it constrains β's undecided convention.** Solving
`1/σ_{d_eff−1}(B_δ) = 52.5` for `δ`, then `c = δ√β`, at the measured median
β = 0.50 (scaled) and 4.0 (unscaled, ×8):

| `d_eff` | `c` at β = 0.50 | `c` at β = 4.0 |
|---|---|---|
| 2 | 0.042 | 0.120 |
| 5 | 0.411 | **1.161** |
| 8 | 0.568 | **1.608** |
| 22 | 0.793 | **2.242** |

Lemma 5.1 needs **`c > 1`**. **Under the scaled convention no `d_eff` up to 22
reaches it; under the unscaled convention `d_eff ≥ 5` does.** That is the first
evidence in this project bearing on the factor-of-8 decision §3.40 flagged as
undecided.

**Held loosely, and the check says so on its face:** this is a consistency
calculation under the paper's i.i.d. spherically-symmetric hypothesis, which
token embeddings do not satisfy; HDBSCAN clusters are neither strong nor
ordinary Rényi centres by definition; and the count is asymptotic in sequence
length while `n ∈ [20, 512]`. **It says which `(convention, d_eff)` pairs could
produce the observed count, not which one does.**

*(One discrepancy recorded, not resolved: App. C.4 prints the `d = 3` count as
`(3 sin²(δ/2))⁻¹`; the cap area gives `1/sin²(δ/2)`. The factor is constant in
`δ`, so it moves an intercept and not an exponent — but do not quote an absolute
`d = 3` count from either form.)*

### 5.3 The collision with §1, which is the interesting part

The parking reading says **early tokens are the cluster nuclei.** §1 says the
causal mask makes **early tokens structurally attention-rich.** These are two
consequences of the same mask, and Phase 10 currently treats one of them as a
finding and the other as a confound.

> **The position axis is not a nuisance variable in this phase. It is the
> mechanism the theory names.**

That reframes three things at once:

1. **F0 becomes an anchor test, not a count test.** "Cluster anchors are early
   tokens" is a per-token, position-indexed prediction, checkable for free
   against `hdbscan_labels.json` plus positions **today**, with no β, no
   convention decision and no reading of the paper. It is the cheapest
   theory-vs-measurement comparison available and it survives §5.1's correction
   intact.
2. **The attention flip gains a rival explanation that is not a confound.** If
   nuclei are early and early tokens are mask-favoured, then "attention goes to
   unclustered tokens" and "clusters nucleate on early tokens" may both be
   position, and **the flip's sign could invert depending on whether the nuclei
   themselves end up labelled clustered or unclustered.** Which they are is
   measurable and nobody has looked.
3. **It gives `H-PARK` a sharper form.** Parking says a particle is captured by
   whichever nucleus reaches it first and then stops. That is *exactly* the
   parked reading, with a mechanism and an arrival order — and it predicts
   cluster membership should correlate with **position distance to the nearest
   early nucleus**, which is again free to check.
4. **`[R]`: Theorem 4.1 makes position 0 a theorem, not a confound.** With
   `V = Id` and **arbitrary** `Q, K`, for almost every initial configuration
   `lim_{t→∞} x_k(t) = x₁(0)` for every `k` — all tokens converge on **the first
   token's initial position**, which never moves because token 1 is autonomous
   under the mask. §1's structural tilt and §5's nuclei are two faces of that.
5. **`[R]`: Lemma 5.1's stationarity time carries the token index.** The
   sufficient condition is `T_j · s_j < e^{c²/2 − c⁴/(24β)}·ε`, so a centre's
   quasi-stationary lifetime falls **inversely with its own token index**. The
   repo measures cluster lifespan (mean 7.0 → 4.5). **Whether lifespan falls with
   the anchor's position is a free, direct test of the lemma** and nobody has
   asked it.

---

## 6. Threads not pulled

Ranked by what a derivation would change. None is attempted here.

1. **Does the `Θ(β^((d−1)/2))` law have an `n` dependence at all?** §5's
   regression carries `b` as a free coefficient because "frequency" is `[S]` and
   might be per-token or a count. `b = 1` versus `b = 0` is a real distinction
   and one reading of the paper settles it.
2. **The cone margin under the *anisotropic* patch, not the rank-1 one.**
   §3 does `Γ + ε u uᵀ`. `plan-9.md` §6 wants `Γ + U D Uᵀ`. The derivative
   generalises immediately (sum over the columns of `U`), but the *optimal* `D`
   subject to a norm budget is a small SDP and has a closed form worth having.
3. **Whether the hemisphere lemma has a subspace form**, `plan-9.md` §4.5's open
   question. Still open, still the one where a plausible-sounding generalisation
   would be easy to believe and wrong.
4. **The implied-timescale readout under a non-gradient-flow dynamics.**
   `plan-9.md` §5.1a records that the masked system is not a mean-field gradient
   flow. A transfer operator needs only a transition structure — but *reversibility*
   is what standard MSM spectral theory assumes, and a causal, non-reversible
   transition matrix has complex eigenvalues. **`t_i = −1/log|λ_i|` still reads,
   but PCCA+'s sign-structure argument may not.** Worth deriving before building.
5. **A closed form for the paid/received square under the mask.** §1 and §2 give
   the two marginals; the joint would say what the content-free square looks
   like, and therefore what an enrichment in each corner means.
