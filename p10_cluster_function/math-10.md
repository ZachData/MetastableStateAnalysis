<!-- p10_cluster_function/math-10.md -->
# Phase 10 — MATH (derivations, checked)

> **UPDATE 2026-09-20 — §1 and §2 are now CONFIRMED on real checkpoints**, and
> §4's instruction is superseded by a measured hazard.
>
> **§1.** Divide the derived baseline out of the attention flip and 94 % of the
> gap disappears; at initialisation all of it does (`status-10.md` §1.1).
> **§2.** Over 11 400 units, raw `log Z` is **99.5 % position** at β = 1, the
> sink sits at percentile **0.0014** of raw `Z` and **0.9986** of corrected `Z`
> — identical to four decimals across β ∈ {1, 2, 4}, so the result does NOT
> depend on the unit convention `docs/AXES.md` §4 flags (`status-10.md` §1.4).
> `math-1.md` §1A.6 is right about the corrected quantity and inverted about
> the raw one, as derived.
> **§4.** The ARI variance argument stands, and a **fourth** hazard was measured
> that no size-profile null addresses: the partition is not reproducible run to
> run — 16.7 % of layer label-vectors differ between two sweeps agreeing to
> 7.9e-05 (`status-10.md` §3).
>
> One caveat §2 did not state: sphere-projected activations have a **unit
> diagonal**, so what they realise is `Z_i = e^β + i e^{βγ}` — affine in `i`
> rather than proportional to `(i+1)` — and the `(i+1)` correction therefore
> leaves a known residual (measured `position_r2_corrected` ≈ 0.32). Both forms
> are test fixtures in `tests/test_p10_partition_function.py`.
>
> **UPDATE 2026-09-20 (later the same day) — `2411.04990` has now been READ,
> and §7 is what it says.** Three things above change:
> **§5.1's "0.7476 does not appear in it" is wrong** (it does, for `d = 2`
> ordinary Rényi centres, as `δ → 0`); **§5.1's `d ≫ 1` verdict applies only to
> the power-law asymptotic**, not to the exact count law, which is
> distribution-free and dimension-free in the form this project can evaluate;
> and **§6 threads 1 and 4 are both answered**. §5.2's `d_eff` manipulation
> turns out to be **the paper's own conjecture**, which licenses it and removes
> its novelty. §7 also corrects `notes-10.md` §10.1's gradient-flow hazard in
> the project's favour. **Read §7 before §5.**

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

**§7 (2026-09-20) has no check file yet** and says so on its face (§7.6). It is
the first section here written from a paper read as primary text rather than
from a search summary, and two of its closed forms — the finite-`n` saturation
identity and the `δ → c²/β` inversion — are exactly the kind `CLAUDE.md` says
should be checked mechanically.

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

> **PARTLY SUPERSEDED 2026-09-20 by §7, which reads the paper instead of a
> summary.** The headline — a law in β and dimension, not in `n` — is
> **confirmed**. Two of the supporting statements are not: the Rényi constant
> *does* appear in the paper (§7.2(b)), and the `d ≫ 1` objection applies to the
> asymptotic power law but **not** to the exact count law (§7.2(c)). §5.2's
> slope test survives as the asymptotic form of §7.2's exact one, and its
> `d_eff` move is the paper's own conjecture (§7.3). Kept as written because the
> reasoning that produced F0 is what §7 then had to correct.

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

**One failure mode, checked:** `k` must be the same for every row in one
regression, and it is not — `head_size` is 64 on gpt2-large and 128 on
pythia-1.4b. A cross-model regression on raw β mixes two conventions; in the
synthetic test that pulls a true slope of 3.25 down to 1.24. **Per-model
regressions, or fix the convention first.**

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

---

## 6. Threads not pulled

Ranked by what a derivation would change. None is attempted here.

1. **ANSWERED 2026-09-20, see §7.2(a): no — `b = 0`, with a computable
   finite-`n` saturation correction. Does the `Θ(β^((d−1)/2))` law have an `n` dependence at all?** §5's
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
4. **ANSWERED 2026-09-20, see §7.4. The implied-timescale readout under a non-gradient-flow dynamics.**
   `plan-9.md` §5.1a records that the masked system is not a mean-field gradient
   flow. A transfer operator needs only a transition structure — but *reversibility*
   is what standard MSM spectral theory assumes, and a causal, non-reversible
   transition matrix has complex eigenvalues. **`t_i = −1/log|λ_i|` still reads,
   but PCCA+'s sign-structure argument may not.** Worth deriving before building.
5. **A closed form for the paid/received square under the mask.** §1 and §2 give
   the two marginals; the joint would say what the content-free square looks
   like, and therefore what an enrichment in each corner means.

---

## 7. The parking law, from the paper rather than from a summary (2026-09-20)

`2411.04990` has been **read as primary text** (`lit-10.md` §11). This section
replaces §5.1's `[S]`-grade description with the paper's own statements, and
**§5.1's correction turns out to have over-shot in one place and under-sold the
result in another.**

### 7.1 The two acceptance rules, verbatim

For a token sequence `(x_j)_{j≥1}` on `S^{d−1}` with geodesic distance `dist`
and a separation parameter `δ > 0`:

```
  Rényi centres           (x_{s_j}) :  dist(x_{s_j}, x_{s_i}) > δ   for all i < j
  strong Rényi centres    (x_{s_j}) :  dist(x_{s_j}, x_i)     > δ   for all i < s_j
```

The first excludes against previously **accepted centres**; the second excludes
against **all previous tokens**. Strong ⊂ Rényi. Both are greedy, sequential,
and defined on **positions and distances alone** — no partition, no clustering
algorithm, no `β` except through `δ`.

The scale is set by the interaction range: attraction is maximal at distances of
order `β^{−1/2}` and decays rapidly beyond, so the paper takes

```
    δ = c · β^{−1/2},     c sufficiently large
```

**Position is arrival order, and that is the entire reason the correspondence
exists only under a causal mask.** Token `k` interacts with `j ≤ k`; under full
attention there is no order and no parking problem. §5.3's claim that *"the
position axis is not a nuisance variable in this phase, it is the mechanism the
theory names"* is now confirmed from the source.

### 7.2 The count law is Lemma C.1, and it is exact, distribution-free and dimension-free

For an infinitely long i.i.d. sequence with law `μ` on `S^{d−1}`:

```
    E[ #strong Rényi centres ]  =  ∫_{S^{d−1}} 1/μ(B_δ(x)) dμ(x)        (C.1)
                                =  1/σ^{d−1}(B_δ)      [μ spherically harmonic]
```

with the closed forms `π δ^{−1}` at `d = 2` and `(3 sin²(δ/2))^{−1}` at `d = 3`,
and growth `δ^{−(d−1)}`. Substituting `δ = cβ^{−1/2}` gives
`c^{−(d−1)} β^{(d−1)/2}` — **so the `Θ(β^{(d−1)/2})` "frequency" §5.1 quotes is
not an independent result; it is (C.1)'s small-`δ` asymptotic.**

Three consequences, and each corrects or completes something above.

**(a) `b = 0`. §6 thread 1 is answered.** §5.2's regression carries a free
coefficient `b` on `log n` because "frequency" was `[S]` and might have been a
count or a rate. (C.1) is the expected count in an **infinitely long** sequence:
there is no `n` in the law. The finite-`n` form is the partial sum

```
    E_n = ∫ Σ_{k=1}^{n} ( 1 − μ(B_δ(x)) )^{k−1} dμ(x)
```

which increases in `n` and **saturates** at (C.1). So `b = 0` asymptotically,
with a saturation correction that is itself computable and is largest exactly
where `δ` is large relative to the cloud — which is the regime a short prompt is
in. **A fitted `b` materially above 0 is therefore evidence of unsaturation or
of non-exchangeability, not of a different law**, and that is a better thing for
`b` to mean than "unknown".

**(b) CORRECTION to §5.1 — the Rényi constant does appear.** §5.1 says *"there
is no 0.7476 in it"*, on two search summaries. Appendix C.4 says the opposite,
for the case the summaries were paraphrasing: at `d = 2`, as `δ → 0`, the
average number of **ordinary** Rényi centres approaches `c·2π/δ` with
`c ≈ 0.75` the Rényi constant (Dvoretzky–Robbins 1964). `lit-1.md` §4's original
description was **half right** — the constant is real and it does give an
expected count — and wrong only in making the count a function of `n` rather
than of `δ`. The correction to `lit-1.md` should say that, not §5.1's
over-correction.

**(c) The `d ≫ 1` objection dissolves — for strong centres.** §5.1 rejects the
prediction as unusable at `d = 1024` because the exponent is 511.5. That is a
statement about the **asymptotic power law** and it is true. It is false of
**(C.1) itself**, whose middle expression

```
    E[ #strong centres ]  =  E_{x∼μ}[ 1 / μ(B_δ(x)) ]
```

is a **reciprocal local-density average**: estimate `μ(B_δ(x))` as the fraction
of tokens within geodesic distance `δ` of `x`, average its reciprocal. No
exponent, no ambient dimension, no `β`, no unit convention — **only `δ`, which
is a distance and can be swept.** The paper is explicit that this is the
asymmetry between the two rules: for strong centres the computation *"works for
any distribution regardless of the dimension"*, while for ordinary centres the
extension to general distributions *"remain[s] open … particularly in higher
dimensions (d > 2)"*.

> **So the phase's quantitative test is not the log-log slope. It is
> observed-vs-(C.1), swept in `δ`, on strong centres.** The slope test (§5.2)
> survives as the *asymptotic* version of the same statement and keeps its own
> virtue — it **measures** `d_eff` — but it is no longer the only available form
> and it is the weaker one, because it needs a `β` and (C.1) does not.

**A reading that falls out for free.** Since `δ = cβ^{−1/2}`, the `δ` at which
the observed strong-centre count matches (C.1) **reads back `c²/β`**.
`PROJECT.md` §3.40's undecided factor-of-8 in β's unit convention becomes a
quantity to measure rather than a decision to take — with the caveat that `c` is
itself only bounded below (Eq. 2 of the paper: `c > β^{1/2} arccos((−1+√(4β²+1))/(2β))`,
and `c > 1` suffices for `β > 1`), so what is read back is `c²/β`, not `β`.

### 7.3 `d_eff`: the conjecture is the paper's, and the candidate list was missing one

§5.2 introduces `d_eff` as *"the manipulation that rescues"* the law. **It is
the paper's own conjecture, not this project's manipulation** (§5, verbatim):
for general `V` the particles *"rapidly converge to a lower-dimensional subspace
spanned by `d_1 ≪ d` principal eigenvectors … we conjecture that the number of
meta-stable clusters should scale as `β^{(d_1−1)/2}`, where the ambient
dimension `d` is replaced by the effective dimension `d_1`. While a rigorous
proof of this dimension-reduction remains an open question…"*

Two things follow. **The construction is licensed** — it is not an unsupported
stretch of the published result. **And it is not novel as a construction**; what
is unclaimed is the empirical check, which nobody appears to have run.

`2501.10573` supplies the measurement (`lit-10.md` §12.3), and it changes the
table in §5.2:

| candidate | kind | `d_eff` | predicted slope `a = (d_eff−1)/2` |
|---|---|---|---|
| ambient (pythia-410m) | — | 1024 | 511.5 |
| effective-rank plateau | **linear** (covariance spectrum) | ~225 | 112.0 |
| ambient participation ratio | **linear** (covariance spectrum) | 22 | 10.5 |
| **kNN intrinsic dimension** (GRIDE / TLE / ESS) | **manifold** | **≈ 7–15** | **≈ 3–7** |
| the published check | — | 2 | 0.5 |

> **`d_1` is the dimension of the set the particles lie on, which is a manifold
> dimension. Effective rank and participation ratio are functionals of a
> covariance spectrum and measure a different thing.** The theoretically-correct
> candidate was missing from §5.2's list, it is ~30× smaller than the nearest
> rival, and it is the only one in a range where the predicted slope is
> measurable at all.

This gives the regression an **independent comparator**: fit `a` from counts,
estimate ID directly from the same `activations.npz`, and compare. Agreement
supports the `d_1` conjecture the paper leaves open; disagreement localises
which half fails. Neither side needs a forward pass.

### 7.4 CORRECTION to §6 thread 4 — the non-reversible readout has a named tool

§6 item 4 asks whether PCCA+'s sign-structure argument survives a causal,
non-reversible transition matrix, and answers *"worth deriving before
building"*. `2601.02932` §5.1.1 answers it directly (`lit-10.md` §13.3):
for a reversible chain the leading eigenpairs are real and PCCA+ applies; **for
a non-reversible one, use the singular values and singular vectors, or the
leading complex eigenvalues and the elements of the real Schur decomposition.**

That paper enforces reversibility with a constrained MLE because its particle
system *is* a reversible gradient diffusion. **A transformer's depth dynamics is
not** — depth is one-way, §7.5 shows the masked flow is only sequentially
gradient, and a split is not the time-reverse of a merge. So:

- `t_i = −τ/log|μ_i|` still reads;
- the reversibility-constrained estimator must **not** be transported;
- the correct spectral tool is the **real Schur decomposition**, which this
  repository already has and which `notes-10.md` §4.2 was looking for an excuse
  to point at a function-defined operator.

Thread 4 is therefore closed as a derivation question and open as an
implementation one.

### 7.5 The masked system is a **sequential** gradient flow — a correction in the project's favour

`notes-10.md` §10.1 and `plan-9.md` §5.1a both record that the masked system
*"cannot be interpreted as a mean-field gradient flow"* and place the Hessian
framing at risk. The paper says that (§4) **and then says what replaces it**
(§5.2, Lemma 5.3): the causal dynamics *"is, in fact, a sequential gradient
flow, where each particle minimizes a slightly different energy"* —

```
    φ̇_k = − ( 1 / Z_k(φ_1,…,φ_k) ) · ∂E_k(φ_1,…,φ_k)/∂φ_k ,     0 < c < Z_k < C
```

with, for the frozen-token case (App. C.3),

```
    E_k(φ_1,…,φ_k) = − ( Σ_{j<k} e^{β(cos(φ_k−φ_j)−1)} + Σ_j a_j e^{β(cos(φ_k−θ_j)−1)} )
```

**What is absent is a single global potential for the ensemble** — so
Łojasiewicz does not apply, and neither does any argument that needs one `E_β`
whose Wasserstein Hessian is the object. **What is present is a per-particle
energy and a genuine gradient flow in it.** The curvature statement Phase 9
wants therefore has to be made about `∂²E_k/∂φ_k²`, per token and causally
ordered, not about an ensemble Hessian. Narrower, specific, and not void.

**And §2's `Z` is this equation's prefactor.** `1/Z_k` multiplies particle `k`'s
own gradient: large `Z_k` ⇒ slow, small `Z_k` ⇒ cheap to move. `math-1.md`
§1A.6's metric reading of `Z` is thereby **an equation rather than an
interpretation**, and `status-10.md` §1.5's parked-vs-pinned argument inherits
that upgrade. §2's conclusion is unchanged and is the operative one: `Z_k` is
the **row** normaliser, the mask makes it `(i+1)`-tilted, and **`Z_i/(i+1)` is
the quantity to read.**

Caveats, stated because the transfer is not free: Lemma 5.3 is proved on `S¹`
with `Q = K = V = I`, the paper ties weights across layers, and it has no MLP.

### 7.6 What is NOT proved here

- Nothing in §7 is checked symbolically yet. §7.2(a)'s saturation identity and
  §7.2's `δ → β` inversion are both closed-form and **should** get a
  `tools/math_checks/` file under `CLAUDE.md`'s rule; §7.3's table is
  arithmetic on published numbers and does not need one.
- **None of this is a theorem about Pythia.** Tied weights, no MLP, `V = I`,
  `d = 2`, `Q = K = I` — §5's results carry all five. The correspondence is a
  hypothesis to test on a real model, which is the point, not a result to
  inherit.
- (C.1) assumes the `X_i` are **i.i.d.** A token sequence is not. **That is not
  a defect; it is the discriminant** — the gap between the observed strong-centre
  count and (C.1)'s i.i.d. prediction is precisely "how far this token cloud
  departs from an exchangeable one", which is `notes-10.md` §3.2's
  packing-versus-content question with an exact null attached.
