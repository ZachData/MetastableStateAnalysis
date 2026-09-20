<!-- p9_metric_intervention/plan-9.md -->
# Phase 9 — PLAN (pre-design, nothing frozen)

**Read `notes-9.md` first.** This file is its continuation, not its
replacement. `notes-9.md` established *what the lever is* (LayerNorm's learned
diagonal, the metric attention reads in) and *what already stands in the way*
(self-repair, identification, the two collapses). This file asks the question
that sits underneath the one the notes answered:

> We want to use clusters — form them, dissolve them, move them — as a tool.
> **Nothing in this repository establishes what a cluster does.** Until that is
> settled, every intervention below produces a picture and no claim.

**Superseded in one place, 2026-09-20: tiers 0 and 1 are now Phase 10.**
§2 and §3 below opened a question large enough to carry its own literature and
its own registrations, and it has been given its own directory,
`p10_cluster_function/` (`notes-10.md`, `lit-10.md`; `PROJECT.md` §3.48). Read
those for the current state of *what a cluster is and does* — §2 and §3 here are
kept as the argument that produced the phase, not as its live state. Two things
found there change this document and are marked inline: **§5.1's Wasserstein-Hessian
framing is at risk under the causal-mask theory** (§5.1a), and **the functional
labelling §2.1 wanted is measurable per layer after all** (§2.1a). Everything from
§4 onward — the congruence algebra, the insertion points, the sign prediction — is
unaffected and stays Phase 9's.

**Phase 9 is parked, not closed.** `notes-10.md` §9 states what each phase owes
the other.

**This is still not `design-9.md`.** No construction is frozen, no `P-*` id
names anything here, `claims/registry.json` is untouched, and `CLAUDE.md`
trigger 1's literature scan remains undischarged — §11 of `notes-9.md` names
the searches, and §12 below adds four more that this document's arguments
depend on. A plan that names an experiment is not a design that specifies one;
the difference is deliberate and is what keeps the scan able to do its work.

---

## 0. Two things called gamma, and they are not the same thing

This bites immediately and it has already bitten once in conversation.

| symbol | what it is | where |
|---|---|---|
| **`gamma`** (`Γ = diag(γ)`) | LayerNorm's **learned per-channel diagonal**. It is the *metric* — the thing that turns the sphere `ln_plain` projects onto into an axis-aligned ellipsoid. This is the lever. | `core/ln_frame.py`, `p1c_frames/frame_table.py`, `math-1c.md` §6.2 |
| **`gamma_beta(t)`** | the paper's **concentration scalar** — the common value all pairwise inner products are predicted to approach under Theorem 6.9, integrated from an ODE. It is a *prediction about the state*, not a parameter of the model. | `p1c_frames/gamma_ode.py`, `beta_reduction.py`, predictions `P-gamma1`/`P-gamma2` |

"Use the gamma metric to cause or take away clusters" means the **first**. The
second is one of the readouts that would say whether it worked. Every sentence
below that says *the metric* means `Γ`; every sentence that says *the envelope*
means `gamma_beta`.

---

## 1. The shape of the programme, in one page

Four tiers. Each is a prerequisite for the one after it, and the project has
historically skipped from tier 0 straight to tier 3 and then had to withdraw
things.

**Tier 0 — What is a cluster?** Three independent labellings of the same tokens
exist as code and have never been run against each other on real data. Until
their agreement is a number, "cluster" names an algorithm's output, not an
object. (§2)

**Tier 1 — What does cluster membership do?** Given a partition, does membership
carry function? Four incompatible accounts are live, all consistent with what is
on disk, and they are separable with instruments that already exist. (§3)

**Tier 2 — What can the metric actually move?** `Γ` is a specific and
surprisingly constrained operation on the dynamics. Most of what it can and
cannot do follows in closed form and can be written down — and checked — before
a single forward pass. (§4–§5)

**Tier 3 — What is it for?** Forgetting, localisation, capacity, plasticity.
Three of the four have an honest answer already, and it is mostly *no*; the
fourth is the one worth building. (§6–§7)

The temptation is to start at tier 3 because that is where the applications are.
The reason not to is §2.5.6: **an intervention validated on the spectrum can be
functionally wrong.** `t = 0` and `t = 1` on the isometric path have identical
singular values by construction and opposite functional outcomes. A cluster
intervention validated on `cluster_count` is exposed to exactly the same error,
and tier 0 is what closes it.

---

## 2. Tier 0 — nobody here knows what a cluster is, and three instruments are sitting unused

### 2.1 What "cluster" currently means in this repository

HDBSCAN, on L2-normed per-layer activations, with a count from an agglomerative
threshold sweep beside it (`p1_mstate_tracking/clustering.py`). Two of
`CLAIM-C`'s six registered metrics — `cluster_count` and `cluster_membership` —
are read from that block and from nothing else. Everything else the project
calls clustering evidence is a scalar summary of the Gram matrix: `fiedler_mean`,
`mass_near_1`, `effective_rank`, `ip_mean`.

That is a *geometric* labelling. It is one of three the tree can produce:

| labelling | derived from | code | run on real data? |
|---|---|---|---|
| **geometric** | distances between activations (sphere frame, or LN frame) | `p1_mstate_tracking/clustering.py`, `core/ln_frame.py` | yes, everywhere |
| **functional** | pairwise KL between the *decoded next-token distributions* of each token | `core/functional_distance.py` (`kl_matrix`, `functional_clusters`) | **never** |
| **mechanistic** | degeneracy in the dominant eigenspace of the operator that generates the motion (`S`-projections) | `p6_subspace/` + Phase 2's `sym_*` projectors | **never** |

`core/functional_distance.py::frame_agreement` computes pairwise Adjusted Rand
Index across a set of labellings. It is built, tested, and **has never been
called on a real run.** `p6_subspace/math-6.md` §4 (item 4) proposes the third
labelling explicitly and notes that it is *"the only one of the three derived
from the operator that generates the dynamics"* and *"the closest thing to a
principled cluster definition anywhere in the project."*

**The single most informative unrun number in this repository is the ARI between
the geometric and the functional partition.** It costs one forward pass per
prompt with an LM head and a matmul.

### 2.1a Amendment (2026-09-20): the functional labelling is measurable per layer

Written above as though the only producer were the final-layer LM head, which
would have given one functional partition per prompt rather than one per layer.
That is no longer the constraint. **The Jacobian lens** of Gurnee et al. 2026 —
the paper `p2_eigenspectra/lens_band.py` already cites, with published Apache-2.0
code — gives `lens_l(h) = unembed(J_l @ h)` with `J_l = E[∂h_final/∂h_l]`, so
every layer has a decoded distribution per token. `core/functional_distance.py`'s
docstring already names "per-layer decoded distributions" as its input, so a
J-lens readout drops into that slot with nothing to change.

`p10_cluster_function/lit-10.md` §1 is the scan, including the four caveats that
decide how it may be used (`pythia-70m-deduped` is not `pythia-70m`; the
published lens carries no checkpoint axis; it is an average of a linearization;
a lens is not ground truth).

### 2.2 The LM-head blocker is stale, and this matters

`archive/p5c_unclustered/status-5c.md` blocker 3 says the functional readout is
unusable because *"no model in the registry has an LM head."* **That is no
longer true.** `core/lm_loading.py` exists: same registry keys, same repo id and
pinned revision as the bare load, so the LM-head model at a checkpoint is
provably the same checkpoint the extraction pipeline analysed. It refuses masked
LMs by design, and it refuses `random_init` entries — with
`load_causal_lm_from_state_dict` as the named path for the random twin, so the
trained-vs-random contrast the whole unclustered-population finding rests on is
reachable rather than blocked.

Two callers already use it (`p2_eigenspectra/vocab_projection.py`,
`p7d_redundancy/backup_sweep_full.py`), so the path is exercised. **Phase 5c's
Group D — "force-collapse / force-disperse with matched controls, read out as
next-token cross-entropy delta and KL" — is the experiment this phase wants, it
was fully designed in 2026, and the blocker that stopped it has been gone for
some time without anyone noticing.** That is the cheapest large win on the
table and §8's ladder puts it near the front.

### 2.3 Chance, and the trap in reading an ARI

**Amended 2026-09-20 — the instruction below stands, its reason was wrong.**
`math-10.md` §4 shows the **adjusted** Rand index already subtracts the
expectation under the size-profile-preserving null, so `E[ARI] = 0` by
construction. **The null is needed for the variance, not the centering** — and
the variance is the real problem: measured 95th percentiles under the null range
**+0.002 to +0.092** across plausible size profiles, a **57× spread**, so a fixed
ARI threshold is not comparable across layers, checkpoints or models.

Two labellings both dominated by a single giant cluster will agree at high ARI
for reasons that have nothing to do with content. Before any ARI is quoted, the
same three comparisons must be run against label permutations *within the
observed cluster-size profile* — the size distribution held fixed, membership
shuffled. `core/nulls.py` is where that belongs. Without it this is the
`math-6.md` §7.2 error again: a number that measures dimension (here, size
profile) rather than content. **Third instance in this project of a
dimension/scale confound eating a headline; the pattern is worth naming in the
design.**

---

## 3. Tier 1 — four accounts of what a cluster does, and what separates them

All four are consistent with everything currently on disk. This is the honest
state and it is why the phase cannot start at the intervention.

### H1 — A cluster is discarded individuation

Membership means the network has decided this token's *identity* no longer
matters downstream; what is read is the cluster's attractor, not the token.

**Evidence for.** The strongest trained-vs-random contrast in the project:
trained GPT-2-large routes ~**1.6×** layer-average attention toward tokens
HDBSCAN never clusters and ~**0.5×** toward clustered ones; trained ALBERT-base
is more extreme (>2× / ~0.5×); the sign is **flipped** under random weights in
every model examined (`archive/p5c_unclustered/status-5c.md`). If clustering
were where the information was, attention would go there.

**Discriminating prediction.** Replace each clustered token's activation with its
cluster centroid. Next-token KL should be small, and small *relative to a
matched-magnitude random-displacement control*. Do the same to unclustered
tokens (substituting the nearest centroid) and the KL should be large.

### H2 — A cluster is a computed category, and the label is used

Membership is itself information — the downstream computation reads "which group"
rather than "which token."

**Evidence against, and why it is not decisive.** Phase 6's first run: LDA
alignment with the real repulsive subspace **0.067** against **0.887** for the
imaginary one, **0/49** layers in the predicted direction, real-only probe at
**0.152** (chance) against imaginary-only **0.564** and full-activation
**0.590**. Read naively: cluster identity is not linearly encoded where the
theory says it should be. But `math-6.md` §7.2's explanation (c) is unresolved —
alignment with a subspace scales as `dim U / d`, and a dimension ratio near **13**
between `U_A` and `U_neg` would reproduce the observed ratio with no content
explanation at all. And `0/49` is one weight-tied ALBERT decomposition observed
49 times, not 49 measurements.

**And explanation (c) has since been settled — against the raw reading.**
`p6_subspace/subspace_geometry.py` is the rebuild (not a copy — `archive/
README.md` rule 2), it exposes `normalized_alignment` rather than a raw one, and
its header records the measurement: at albert-xlarge-v2's shape the audit
(`claims/audits/p6_projector_labels.json`) gives **dim(U_A)/dim(U_neg) = 24.9**
against an observed alignment ratio of **13.2**. *The dimension correction is
nearly twice the effect it would explain.* The live null
(`p6_subspace/r2_r4_null.py`) goes further and holds dimension fixed by
construction — a random subspace of matched dimension — rather than trusting the
`k/d` identity.

So the inversion is **not** evidence that cluster identity is absent from the
repulsive subspace; it is a dimension artifact with the sign of the correction
pointing the other way. **Phase 6's headline negative on H2 should not be quoted
as one**, and any Phase 9 argument that leans on "clusters aren't linearly
encoded" is leaning on a withdrawn reading.

**`P6-R4` is the registered form of this question** ("S-only projection
preserves cluster membership"), it is the one row in Phase 6 whose inputs exist
on disk today — it projects onto `u_s`, which Phase 2's `sym_*` artifacts supply
— and it is blocked on one thing only: the registry names no exchangeable unit
and the gate refuses to pick one rather than silently defaulting. **That is a
decision, not compute** — the same class of blocker the whole e-value audit found
everywhere (`PROJECT.md` §3.45). Its sibling `P6-R2` is blocked differently and
more deeply: its second arm is the **antisymmetric/imaginary** subspace, and no
such projector exists in any of the 19 `p2_eigenspectra_*` directories — the
artifacts carry `{Schur, symmetric} × {attracting, repelling}` and nothing else,
because the rotational channel was *deliberately* not measured (`status-6.md`).
**Phase 9 must not assume an imaginary-channel projector exists.**

### H3 — A cluster is an epiphenomenon of concentration

Theorem 6.9 says pairwise inner products concentrate; HDBSCAN run on a
concentrating cloud will find groups whether or not they mean anything.

**Discriminating prediction.** Functional ARI at chance under §2.3's size-profile
null; centroid substitution costs the same as matched random displacement.

**This is the outcome the phase must be able to survive.** If H3 holds, "cause or
remove a cluster" is a statement about a visualisation, and the phase's object
becomes the *measure's* slow directions (§5.5) rather than the partition. That
is still a phase — it is the one `MATH_SPECTRAL_OT.md` §6 actually asks for —
but it is a different one, and it should be pre-registered as a legible outcome
rather than discovered.

### H4 — A cluster is a capacity ledger entry

Clustering frees dimensions. Effective rank plateaus near **200–250** across
models whose `d_model` spans **768–1600**; if the network simply used what
training gave it, rank would scale with width and it does not. Read that way,
~40–50% of tokens staying unclustered is a **bounded dimensionality budget spent
on particles that must stay individuated** (`PUBLICATION_IDEAS.md` idea 3).

**Caveat carried, not dropped.** That budget number is not publishable as it
stands — the plateau has to be re-established on *normed* rank and ~200
reconciled against Blog 1's ~250 (open-question register item 3). Any Phase 9
argument that leans on the budget inherits that debt.

**Discriminating prediction.** Force additional tokens into a cluster and measure
whether effective rank elsewhere *rises* (freed capacity taken up) or stays flat
(the budget is not fungible within a forward pass). In a frozen model the honest
prior is flat — nothing retrains — which makes this the one account that
genuinely needs the fork-retrain protocol to test, and therefore the one to
defer.

### Why enumerating four is the design, not indecision

`archive/p5c_unclustered/design-5c.md` §"Why Group C tests three specific
outcomes" and `math-5c.md` §4's three-outcome rank test are the same discipline:
register the outcome that would embarrass the phase alongside the favoured one.
Four accounts, four legible outcomes, one battery.

---

## 4. Tier 2 — what the metric lever actually is, in closed form

Everything in this section is derivable without a forward pass and most of it
should go to `tools/math_checks/` before it goes into a design.

### 4.1 A gamma-patch is a congruence on the QK bilinear form, shared by every head in the block

Attention reads `LN(x) = Γ x̂ + b`, with `x̂ = (x − μ)/σ`. For head `h` the
pre-scale logit is

```
    logit_ij = (Γ x̂_i + b)ᵀ Wq ᵀ Wk (Γ x̂_j + b)
             = x̂_iᵀ (Γ W_QK Γ) x̂_j          <- the pairwise term
             + bᵀ W_QK Γ x̂_j                 <- constant along i
             + x̂_iᵀ Γ W_QK b                 <- constant along j
             + bᵀ W_QK b                      <- a scalar
```

So `Γ → Γ' = Γ + U D Uᵀ` sends `W_QK → Γ' W_QK Γ'`: a **two-sided congruence**,
not a one-sided reweighting. Three consequences, each load-bearing:

1. **It is shared.** Every head reading that LayerNorm sees the same `Γ'`. The
   metric is not a per-head lever, and a design that wants per-head control has
   to say so and pay for it differently.
2. **It is symmetric in query and key.** A congruence cannot make *i* attend to
   *j* without making *j* attend to *i* through the same form. Asymmetric
   routing changes are outside what `Γ` can express. **§2.5.6 is exactly the
   warning here:** the transpose preserves the spectrum and copying stays broken
   (`1.040218472480774` both ways), which says read/write alignment — the
   asymmetric part — carries causal weight the symmetric part does not. A
   gamma-only phase is, by construction, blind on the axis where this project's
   one designed intervention found its surprise. **That is an argument for a
   write-side arm, not against the phase.**
3. **The bias terms are the sink channel.** `bᵀ W_QK Γ x̂_j` is constant in `i` —
   a term that shifts one token's logit for *every* query at once, which is the
   functional shape of an attention sink. The LN bias also puts a floor of
   `||β_LN||²` under `<G>` with nothing to do with the tokens
   (`design-1.md`). A gamma-patch moves this term too, and a design that does
   not separate the pairwise from the bias contribution will attribute a sink
   effect to a metric effect.

> **CORRECTION 2026-09-20, from reading `2411.04990` in full
> (`docs/readings/2411.04990.md` §1).** The paper's §2 remarks that a trainable
> RMSNorm diagonal *"can be equivalently achieved by **multiplying `K, Q, V`
> matrices by `D`**"*. In Pythia the attention block's `input_layernorm` feeds
> `W_Q`, `W_K` **and** `W_V` alike, so **a γ-patch there is not a read-side
> lever — it moves the attention pattern and the displacement together.** Two
> consequences:
>
> - **`notes-9.md` §7's three insertion points are not separable by γ.**
>   Isolating the write side needs `W_V`'s input scaled directly, which the
>   LayerNorm does not permit.
> - **§4.7's sign-differential test cannot be run with a γ patch alone**, since
>   it turns on separating "who attends to whom" from "where the content pushes
>   you". It needs a `W_V`-side arm, and the design must add one.
>
> This narrows the lever and sharpens it: a γ-patch is a **simultaneous**
> QK-congruence and OV-rescale, which is a specific, nameable object rather than
> a vague "metric change".

### 4.2 On Pythia the MLP can be excluded exactly, and this answers the "MLP or attention" question precisely

Pythia's parallel residual gives **two LayerNorms per block reading the same
input**: `input_layernorm` (what attention reads) and `post_attention_layernorm`
(what the MLP reads, despite the name) — `core/ln_frame.py` encodes this.

So the user's "augment the space either in the MLP or in the self-attention
mechanism" is not a rough choice here: **patching `input_layernorm.weight`
changes the metric for attention only; patching `post_attention_layernorm.weight`
changes it for the MLP only; patching both changes it for the block.** The split
is exact on this architecture, is free, and gives a three-arm factorial where
most architectures would give one blurred knob. `dissipation_by_channel` already
computes the matching attn/FFN split of the dissipation on the same residual, so
the readout is aligned with the intervention by construction.

This also sharpens §3.29's framing of MLP 6: *"an external field, not an
interaction — it cannot change the pairwise coupling; what it changes is the
cost geometry the coupling is computed in."* A metric patch on
`post_attention_layernorm` is an intervention in **the same channel the MLP's
repair operates in**. See §6.3 — that is the phase's best single question.

### 4.3 What the metric cannot do: Lemma 6.4 survives every gamma

Lemma 6.4: all `x_i` in an open hemisphere ⟹ exponential collapse to a point,
and **only positivity of the attention weights is used** in the proof
(`math-1c.md` §7.1). Softmax weights are positive for any `Γ`, any `Q`, any `K`.

> **No metric deformation can prevent collapse-from-a-hemisphere.** What resists
> must come from `V` or from outside the paper's model (the MLP, causal masking,
> RoPE).

**Under causal masking the ceiling is stronger, and needs no hypothesis at all.**
`2411.04990` Theorem 4.1 (**[R]**): with `V = Id` and **arbitrary** `Q, K`, for
almost every initial configuration the masked dynamics converge to a single
cluster, and the limit is **`x₁(0)`** — the first token's initial position, which
never moves because token 1 is autonomous under the mask. **No hemisphere
condition is required, and no QK-side intervention, γ included, can prevent it.**
The paper notes this is strictly weaker in hypotheses than the unmasked results,
which need `QᵀK = V` or `QᵀK = Id`. It also means the causal mask does **not**
supply the resistance the sentence above hopes for.

This is the hard ceiling on "use the metric to stop clustering," and it should be
stated in `design-9.md`'s first paragraph so no result is read as beating it.

### 4.4 What the metric *can* do to the cone margin — and the computation is free

The margin is exact and needs only the Gram matrix (`math-1c.md` §7.3):

```
    m = dist(0, conv{x_i}) = sqrt( min_{λ ∈ Δ} λᵀ G λ )
```

`n`-dimensional regardless of width, a convex QP with an exact optimum, and the
optimal `λ`'s support **names the binding tokens**. Under a gamma-patch the
points attention reads become `Γ' x̂ + b`, so `G` — and therefore `m` and its
binding set — is recomputable from activations already on disk with **no forward
pass**. The first-order effect of a candidate patch on the configuration is
free.

Two honest limits:

- Wendel gives `m > 0` almost surely whenever `d ≥ n`, and `d > n` holds for
  *every* prompt in the current grid. So the cone condition generically holds
  and the interesting reading is the **margin's size, never the boolean**, which
  is already this project's registered position (`P-H1` is a measurement with no
  valid null, by Wendel).
- The lemma's rate bound is `α' ≥ (1 − α)/(2 n e^{2β})` — it depends on `n` and
  `β`, **not on `m`**. So do not claim the margin tunes the collapse rate. The
  margin governs whether the guarantee applies; the rate in this bound does not
  see it. A relation between margin and observed rate would be an empirical
  finding, and a genuinely interesting one.

### 4.5 A derivation worth checking symbolically before it is believed

The cone condition is stated in the full space. Its **subspace-relative** version
— does `0` lie in the convex hull of the *projections* `P_U x_i` — is not the
lemma and is not implied by it. But it is the natural object for an intervention
that targets a `k`-dimensional subspace, and for `k < n − 1` it can fail where
the global condition holds.

Two things must be established before any design leans on it, and both are
`tools/math_checks/` work in the sense `CLAUDE.md` means:

1. Whether the hemisphere argument has a subspace form at all under `V ≠ I` — the
   proof's forward-invariance step is a statement about the full dynamics and
   projecting it is not free.
2. If it does not, say so and drop it. **A plausible-sounding subspace
   generalisation of a real theorem is exactly the kind of thing this project has
   caught itself on before** (`elim_rotation = 0.0`, withdrawn as an identity;
   `math-6.md` §0).

Instantiating at `n = 4` is evidence, not a general-`n` proof, and the check must
say so on its face.

### 4.6 The write-side lever is a subspace-restricted clock

`math-1c.md` §1.4: damping the field by 0.3× leaves a residual of **−0.0009** —
essentially zero — because `h_calibrated` absorbs it into a shorter `T_eff`.
Damping is not resistance, it is slower integration. Run backwards:

> **Scaling an attention branch's write by `c` runs that layer's dynamics `c`
> times longer.**

Restrict that scaling to a subspace and you have a **subspace-restricted
integration time** — the cleanest available meaning for the user's "accelerate
or decelerate a subspace." Pythia's parallel residual makes the attn/FFN split
exact, so the clock can be set for attention alone.

The prediction is quantitative and must be **written down before the run**:
integrate `gamma_beta` to `c · T_eff` and compare. Two conditions on reading it:

- Read against the **envelope**, never a point estimate. The spread of
  `gamma_beta` across `β ∈ [0.5, 5]` is **0.89 at n = 20**, larger than any
  residual worth reporting.
- **`β`'s unit convention is undecided and it is worth a factor of 8**
  (`status-1c.md` audit, finding 2). With the model's own `1/sqrt(head_size)`
  scale applied, measured `β` on pythia-410m step143000 is median **0.50**, IQR
  **[0.26, 0.75]**, range **[−0.84, 2.19]**, with **0% above 5**. Without it
  every number is 8× larger, and `head_size` is 64 on gpt2-large against 128 on
  pythia-1.4b, so an unscaled `β` is not even comparable across `CLAIM-C`'s own
  arms. **This decision gates every `gamma_beta` prediction Phase 9 would make**,
  and it is a decision nobody has taken, not a computation nobody has run.

### 4.7 Sign: the same intervention has opposite geometric outcomes, and Phase 2 predicts which

Read-side `Γ` changes *who attends to whom*. It does not change *where the
attended-to content pushes you* — that is `V`. `V = −I_d` flips the sign of the
whole Lyapunov identity exactly (`math-1.md` §1A.2), and Phase 2 supplies
per-layer attracting/repelling projectors.

> **Raising mutual attention within a token set produces convergence if the set's
> displacement lies in the attracting subspace and divergence if it lies in the
> repelling one. Same intervention, opposite outcome, predicted in advance from
> the operator's spectrum.**

This is the best *differential* prediction in the plan and it is the shape
§3.29's tier-2 discipline demands: a rival account (attention-pattern
interpretability, where more attention means more influence, full stop) predicts
one sign in both conditions. It also carries the project's own warning against
itself — **§2.4.6 established that no weights-only spectral quantity computed
here identifies the copier**, `L7H8` and `L2H10` agreeing to three digits on
every eigenvalue-derived field with opposite-signed causal effects. So the
sign-split test is simultaneously a test of the prediction *and* of whether the
projectors carry any causal information at all. Both outcomes are informative and
both should be registered.

---

## 5. Readouts — and why cluster_count is the wrong one

### 5.1 The partition is an algorithm's opinion; the timescale is a measurement

`MATH_SPECTRAL_OT.md` §6, restated: every claim of the form "these are metastable
states" currently rests on a clustering algorithm plus a scalar. The operator
that defines the word is the **Wasserstein Hessian of `E_beta`**; near-zero
eigenvalues are slow directions, the number of small eigenvalues counts the
metastable states, and the sign structure of the eigenvectors *is* the partition.

The affordable standing-in for it is the molecular-dynamics construction the same
section names: a **transfer operator over clusters**, implied timescales
`t_i = −1/log|λ_i|`, and **PCCA+**, of which the project's Fiedler value is the
`k = 2` special case. `p1_mstate_tracking/cluster_tracking.py` already matches
clusters across adjacent layers by Jaccard overlap and records births, deaths and
merges — **that is a transition structure, one step from a transfer operator**,
and the step has not been taken.

> Stated in that language the phase's proposal stops being a knob: **"stretch a
> subspace" becomes "change the curvature of a near-zero eigendirection," and the
> predicted effect is a computable shift in an implied timescale** rather than a
> different integer from HDBSCAN.

### 5.1a Hazard (2026-09-20): the Hessian framing presupposes a gradient flow, and a decoder may not have one

`2411.04990` (*Clustering in Causal Attention Masking*, Karagodin–Polyanskiy–
Rigollet) states that **the causally-masked system cannot be interpreted as a
mean-field gradient flow.** Pythia is causal. `docs/LITERATURE.md` row 6 already
asks whether that voids Phase 2d's gradient-flow framing; **it bears on §5.1 the
same way and nobody had said so.**

- **At risk:** the Wasserstein-Hessian-of-`E_beta` framing, and with it
  `notes-9.md` §8's "stretch a subspace = change the curvature of a near-zero
  eigendirection". That restatement presupposes the structure the masked theory
  says is absent.
- **Survives:** the transfer-operator / implied-timescale / PCCA+ readout below.
  It needs a *transition* structure, not a gradient-flow structure, and
  `cluster_tracking.py` already computes transitions. **This is now a reason to
  prefer the timescale readout on its own merits, not only on cost.**
- **Also survives:** everything in §4. Lemma 6.4 is proved from positivity of
  `a_ij` alone, so it is mask-agnostic; so are the congruence and the cone margin.

**`[S]`-grade — confirm by reading `2411.04990` before retracting anything.**
It is the top item in `notes-10.md` §12's queue for exactly this reason: two
phases depend on that one paper and neither has read it.

### 5.2 Transport separates spreading from shuffling, and nothing else here does

`core/dissipation.py`'s transport half — `w2_identity`, `w2_optimal`,
`sliced_w2`, `wasserstein_arc_length`, `straightness` — is implemented, tested,
and **called by no runner** (`notes-9.md` §9, verified by grep). The gap between
`w2_identity` (token `i` to token `i`, the current convention, an upper bound)
and `w2_optimal` (the true coupling) measures **how much of a layer's
displacement is tokens swapping places — motion that leaves the distribution
unchanged — versus genuine motion of the measure.**

For an intervention whose entire purpose is to spread a region apart or pull it
together, that is *the* discriminator, and `straightness` (endpoint `W_2` over
arc length) gives dwelling measured **on the measure** rather than inferred from
whether HDBSCAN found something. No `W_2`, arc length or straightness appears
anywhere in `PROJECT.md` or `docs/`.

### 5.3 The readout panel a design should commit to

| readout | what it answers | cost | exists? |
|---|---|---|---|
| next-token KL / CE delta | did the output change | forward pass | `core/intervention.py`, `core/lm_loading.py` |
| functional ARI vs geometric | is the partition the functional partition | forward pass + matmul | `core/functional_distance.py`, unrun |
| `w2_optimal` vs `w2_identity` | spreading, or shuffling | free | built, unrun |
| `straightness` | dwelling, on the measure | free | built, unrun |
| implied timescales `t_i` | is it metastable, in the word's own sense | cheap | **not built** |
| cone margin `m` + binding set | where the configuration sits, and which tokens hold it | free | `hemisphere_feasibility.py` |
| `E_beta`, `<G_i, v_i>` | energy and per-particle gradient alignment | free | `core/dissipation.py`, run |
| `gamma_beta` envelope residual | does the theory's clock predict it | free | `gamma_ode.py`; **gated on §4.6's β decision** |
| effective rank (normed), PR | capacity | free | `core/metrics.py`; budget claim carries §3-H4's debt |
| `cluster_count`, `cluster_membership` | what HDBSCAN thinks | cheap | the *weakest* row here; never the primary |

---

## 6. Tier 3 — what it is for, with the honest prior on each

### 6.1 Forgetting: the naive version is already refuted here, and that is useful

"Find where the fact lives, collapse it, the fact is gone" fails against two
measurements this repository already owns:

- **Self-repair.** 44/45 pairwise co-ablation cells positive at step 16000, no
  block structure (`PROJECT.md` §3.12-V), which the Hydra Effect predicts and the
  2026-09-10 scan classified as **not new**. Collapse one site and the set
  re-forms the function.
- **The set is not a small subspace.** `L11H14` sits at mean cosine **−0.033** to
  the rest of its set and members write into private low-variance bandwidth: 90%
  of joint effect needs **355 ambient directions** against an ambient
  participation ratio of **22** (`status-7e.md`). There is no compact thing to
  collapse.

And `notes-9.md` §2's rule stands: **collapse is a release operation, not a
deletion one.** On the original distribution it is plasticity, not removal; what
refills freed capacity is decided by the post-collapse gradient.

**So the phase should not promise unlearning.** What it can honestly ask is
narrower and better:

> A metric patch is **read-side, inference-time, weight-preserving, and exactly
> invertible** — a rank-`k`-plus-diagonal object at one LayerNorm. If a
> capability can be suppressed that way, "unlearning" becomes a *filter you can
> switch off*, which is a different product from weight editing and has a
> different threat model.

**Amended 2026-09-20 (`lit-10.md` §6): the genus is occupied and the differentia
is narrower than this.** `2605.12765` (GUARD-IT) is training-free, gradient-free,
*"entirely in activation space"*, and executes unlearning as *"a controlled
geometric transformation"* — specifically **pure rotations in the residual
stream, preserving the activation norm**. So inference-time geometric unlearning
exists. **What survives is exact and worth stating precisely: a rotation is an
isometry; a γ-patch is a congruence** (§4.1's `W_QK → Γ'W_QK Γ'`), which is the
one thing an isometry is not. The claim to rewrite is *"we change the metric,
not the coordinates"*, not *"nobody does inference-time editing"*.

**And `2505.16831` (*Unlearning Isn't Deletion*, ICML 2026) is support, not a
threat** — models *"appear to forget while their original behavior is easily
restored"*, information *"merely suppressed rather than genuinely erased"*. That
is `notes-9.md` §2's release-not-deletion rule, reached independently, with an
evaluation framework attached. **Any Phase 9 unlearning result must be tested for
reversibility** or it measures the thing that paper says is routinely
mismeasured. All `[S]`.

The falsifier is clean: a metric patch that suppresses a behaviour but that a
matched-magnitude random patch suppresses equally has shown nothing.

### 6.2 The one question I would build the phase around

Self-repair has only ever been measured against interventions that **remove a
component** (zero- and mean-ablation). MLP 6's repair, per §3.29, is *"an
external field... it cannot change the pairwise coupling; what it changes is the
cost geometry the coupling is computed in."*

A metric patch **is a change of cost geometry.** Intervention and repair are then
in the *same channel* — which ablation could never test, because ablation is in
the other one.

> **Does self-repair engage against a metric deformation the way it engages
> against an ablation?**
>
> - If **yes**, repair is indifferent to channel and is better described as a
>   downstream error-correction property than as a geometric one. That narrows
>   the Hydra-Effect story in a way the literature has not.
> - If **no**, there is a class of intervention the network does not compensate
>   for, and every capability-removal result measured against ablation has been
>   measuring the wrong intervention.
>
> Both answers are publishable, they are differential, and the matched control is
> obvious (equal-magnitude patch, random `U`).

This is the only item in this document I would rank as tier-2 registerable on
its own merits, and it does not depend on tier 0 resolving in any particular
direction.

### 6.3 Forming a cluster as a *probe*, not an edit

Pick a target set `T` — a fact's token positions, an induction pair's `(j−1, j)`,
an SAE feature's active positions. Raise mutual attention within `T` via a
congruence on `U = span(T's LN-frame directions)`. Both outcomes say something:

- `T` converges, task survives ⟹ the network had already discarded `T`'s
  individuation. Consistent with H1; the capability does not need these tokens
  apart.
- `T` converges, task breaks ⟹ **the capability requires these tokens to stay
  individuated.** That is a localisation claim of a different type from anything
  in mechinterp's usual vocabulary: not *"this capability lives in this
  direction"* but *"this capability requires this separation."*

The second is why forming a cluster is worth doing even if you never want to ship
a cluster-forming edit.

### 6.4 Spreading, and the honest version of "more independent"

"Distribute it so the space expands and everything is more independent" has a
precise form: raise effective rank / participation ratio **restricted to `T`**,
and verify with `w2_optimal` vs `w2_identity` that the mass actually moved rather
than permuted.

But in a frozen model, freed capacity is not taken up by anything — nothing
retrains. So the spreading arm is only interesting in one of two ways:

- **Oversmoothing repair (cheap, sharp).** Deep layers collapse. Stretch at layer
  `L` and ask whether *late-layer functional distinguishability* recovers —
  measured with `functional_distance`, not with a cluster count. If it recovers,
  the collapse was costly and the network was not choosing it; if it does not,
  collapse is chosen. Either way it lands on Blog 1's headline about learned
  resistance, and it connects directly to the rank-collapse literature
  (Dong–Cordonnier–Loukas) and to attention sinks.
- **Fork-and-retrain (expensive).** The only way to test H4's capacity ledger.
  `status-8.md`'s protocol, and the existing 70m retrain is a **fork**, not
  pythia-70m — the two must stay labelled apart.

### 6.5 Localisation: three ways to get `U`, and the rule for SAEs

`notes-9.md` §3(c) sets the prior and it should be a design rule, not a caveat:
**an SAE-derived subspace is a candidate generator, never an identifier.**
Structural proxies have failed against causal ground truth twice here — §2.4.6's
spectral non-identification, and `status-7e.md`'s `useful_rank --bottom`
inversion where keeping `L11H14`'s top singular direction is *worse than deleting
the head* (`−0.096` at `r = 1`), so Eckart–Young's provably optimal approximation
of the **operator** is the worst of three at preserving the **function**.

That converts into a three-arm comparison worth running for its own sake:

| source of `U` | derived from | prior |
|---|---|---|
| SAE features | sparse reconstruction of activations | weak — the objective is not aligned with slow directions of a dynamical operator |
| `S`-subspace projectors | the operator that generates the motion | the only operator-derived option (`math-6.md` §4) |
| causal search | ablation / patching ground truth | strongest, most expensive |

Same intervention, same readout, three localisers. **If the SAE-derived `U`
underperforms the operator-derived one, that is a clean negative result on SAEs
as dynamical localisers** — consistent with two prior failures here and with
`OVERVIEW.md`'s record that sparse dictionary decompositions in this project
"came back null or were shelved."

---

## 7. The questions worth asking, ranked by what the answer would change

1. **Is the geometric partition the functional partition?** (ARI, §2.1.) If no,
   most of this project's cluster vocabulary needs restating, and Phase 9's
   object moves from the partition to the measure's slow directions.
2. **Does self-repair engage against a change of cost geometry?** (§6.2.) The
   one item here that is registerable on its own.
3. **Does raising mutual attention converge or diverge a set, and does Phase 2's
   sign predict which?** (§4.7.) Differential against attention-pattern
   interpretability, and a simultaneous test of whether the projectors carry
   causal information.
4. **What is Pythia's `γ` dynamic range, per layer, across training?** The
   paper's own licensing check (ALBERT: mean **0.44**, sd **0.008**), written as
   `frame_table.py` sub-experiment D and **never run**. It bounds what counts as
   an in-distribution patch, and if the range is wide, *every sphere-frame metric
   in Phase 1 inherits a distortion*.
5. **Is a capability's dependence on a token set a dependence on *separation*
   rather than on a direction?** (§6.3.)
6. **Does late-layer distinguishability recover under a mid-layer stretch?**
   (§6.4.) Bears on whether collapse is chosen or suffered.
7. **How much of each layer's displacement is genuine motion of the measure
   versus tokens swapping places?** (§5.2.) Free, unrun, and it re-reads every
   displacement number already on disk.
8. **Do implied timescales move as predicted under a metric patch?** (§5.1.) The
   difference between a knob and a phase.
9. **Does the attention-flip sign reversal survive on Pythia and across
   checkpoints?** The project's strongest trained-vs-random contrast was measured
   on GPT-2 and ALBERT and never carried to the checkpoint axis.
10. **Which localiser finds a `U` that works?** (§6.5.)
11. **What is the margin at `n > d`, and which tokens bind it?** Unreachable at
    `d = 1024` with the current grid; reachable on pythia-70m (`d = 512`, 2048
    window). Report the margin and the binding set, never the boolean.
12. **Does a Sinkhorn-normalised head behave as the theory says?** Remark 3.5:
    doubly-stochastic attention *is* a Wasserstein gradient flow and clustering
    there is **open in the paper**. `p1_mstate_tracking/sinkhorn.py` already
    implements the normalisation with the causal-mask confound fixed. This is
    theorem-adjacent rather than heuristic and is different in kind from
    everything above it.

---

## 8. The ladder — cost, dependency, and what each would falsify

Ordering is a proposal. Nothing here is registered and nothing here is frozen.

| # | experiment | forward pass? | depends on | falsifies / decides |
|---|---|---|---|---|
| **E0** | **Transport observables on artifacts already on disk.** `tools/run/transport.py`, on the pattern of `tools/run/dissipation.py`. Identity-vs-optimal gap, arc length, straightness, per layer per checkpoint. | no | nothing | gives every existing displacement number a spread/shuffle decomposition |
| **E1** | **`γ` calibration.** Run `frame_table.py` (sub-exp D) on Pythia: dynamic range per layer, `sphere_license`, `bias_energy_floor`. | no | nothing | bounds "in-distribution patch"; a wide range means Phase 1's sphere metrics carry a distortion |
| **E2** | **Three-frame agreement.** geometric / functional / `S`-derived labellings + ARI, against a size-profile-preserving null. | yes (LM head) | `core/lm_loading.py` (unblocked, §2.2) | H3. **The tier-0 gate.** |
| **E3** | **Cluster-as-function battery.** Phase 5c Group D, revived: centroid substitution, force-collapse, force-disperse, each with its matched control, on trained model and random twin. Readout: next-token KL / CE delta. | yes | E2 | H1 vs H2 vs H3 |
| **E4** | **Carry Phase 6's chance-relative correction into the Phase 9 record.** The instrument is rebuilt (`subspace_geometry.py`, `normalized_alignment`) and the ratio is measured (24.9 vs 13.2); what is missing is that no document outside that module's header says the inversion has been explained. A note, not a run. | no | nothing | stops H2 being dismissed on a withdrawn reading |
| **E5** | **`P6-R4`'s exchangeable unit.** A registry decision, not a run. | no | a human call | the one Phase 6 row whose inputs exist today |
| **E6** | **The metric patch itself.** `Γ → Γ + U D Uᵀ`, one LayerNorm, sweep `D`; three arms — attention-LN only, MLP-LN only, both (§4.2). Full readout panel (§5.3), `gamma_beta` prediction written down first. | yes | E1, E2, and §4.6's β decision | whether the lever moves anything beyond a matched random patch |
| **E7** | **The sign-differential test.** Same patch on sets whose displacement lies in the attracting vs the repelling subspace. | yes | E6 | §4.7; and whether Phase 2's projectors carry causal information at all |
| **E8** | **Repair-channel test.** Metric patch vs matched-magnitude ablation, same target, measured for self-repair. | yes | E6 | §6.2 — the phase's best question |
| **E9** | **Implied timescales.** Transfer operator over clusters from `cluster_tracking.py`'s transitions, `t_i = −1/log\|λ_i\|`, PCCA+. | no | E0, E2 | turns the readout from an integer into a timescale |
| **E10** | **Cone margin at `n > d`** on pythia-70m. Margin and binding set, not the boolean. | yes (long context) | nothing | whether the hemisphere lever is reachable at all |
| **E11** | **Localiser bake-off.** SAE / `S`-subspace / causal `U`, same patch, same readout. | yes | E6 | §6.5 |
| **E12** | **Sinkhorn head.** | yes | E0 | Remark 3.5, theorem-adjacent |
| **E13** | **Collapse-and-regrow.** Fork-retrain; does the structure re-form, where, how fast. | training | everything | H4; §2.4.6's *"the developmental question is where the structure is"* |

**E0, E1 and E4 are free and unblocked today** — E4 is a paragraph. E5 is a
decision a human takes. E2 is one forward pass away from the most informative
unrun number in the tree.

**Standing constraint, inherited** (`MATH_SPECTRAL_OT.md` §6.1): this is a new
subphase directory importing existing outputs **read-only**. Nothing here is
added as a stage inside `p2_eigenspectra/run_2.py`.

---

## 9. Registration discipline for this phase, decided before any result exists

`PROJECT.md` §3.29's three tiers, applied:

- **E0, E1, E2, E4, E9, E10 are tier 1** — exploratory, labelled, no p-value.
  Most of the phase lives here and should say so.
- **E7 and E8 are the only tier-2 candidates.** Both are differential: a rival
  account predicts a different answer. Neither is an instance-level claim of the
  kind §3.27/§3.28 nearly registered by mistake.
- **Nothing here spends a reserved rung.** `pythia-1b` and `pythia-1.4b` stay
  reserved; explore on 70m and 410m. And if a Phase 9 prediction ever does name
  1b, it must name **mean-ablation** — 1b is 8 heads/layer, where zero-ablation's
  off-distribution bias (`~1/n_heads`) distorts (`status-8.md`).
- **39 registrations, 0 adjudications.** The marginal value of a 40th is low.
  This phase should aim to *adjudicate* something, not to register more.

---

## 10. How this fails, stated in advance

1. **Tier 0 returns H3** and the partition turns out to be an artifact of
   concentration. Mitigation: it is a legible pre-registered outcome and the
   phase becomes the measure-level one `MATH_SPECTRAL_OT.md` §6 asks for.
2. **The metric patch does nothing a matched random patch does not.** Mitigation:
   the matched control is in the design from the start, not added after.
3. **The patch does something large and uninterpretable** — off-distribution
   activations, the model degenerating rather than the cluster moving.
   Mitigation: E1 bounds the in-distribution range *before* E6 chooses `D`;
   report where each swept `D` sits relative to it.
4. **`β`'s convention is never decided** and every `gamma_beta` comparison stays
   unquotable. Mitigation: it is a named blocker (§4.6) with a factor of 8 on it;
   raise it as a decision, do not route around it.
5. **The sink channel is mistaken for the metric channel** (§4.1, consequence 3).
   Mitigation: separate the pairwise term from the bias terms in the reported
   decomposition.
6. **The phase drifts into a knob zoo** — many `D`s, many layers, no prediction.
   Mitigation: §3.29's standing hazard is *depth in one place against sparseness
   everywhere else*; the `gamma_beta` prediction and the timescale readout are
   what make each run a test rather than a picture.

---

## 11. Stale statements found while writing this

Recorded rather than silently dropped, per `notes-9.md` §9's precedent.

1. **`archive/p5c_unclustered/status-5c.md` blocker 3 is stale.**
   "No model in the registry has an LM head" — `core/lm_loading.py` exists, is
   registry-consistent and revision-pinned, has two live callers, and supplies
   `load_causal_lm_from_state_dict` for the random twin. **Phase 5c's Group D is
   unblocked on this axis and nobody noticed.** (Its blockers 1, 2 and 4 —
   `noise_tracking.py`, `causal_tests.py`'s rewiring, and the missing package —
   are not checked here and should be re-audited before E3 is designed.)
2. **`p6_subspace/math-6.md` §7.2's "until one of these is done, the inversion
   is not evidence" has been discharged, and only the module header says so.**
   `subspace_geometry.py` reports alignment relative to chance and the audit
   measured the ratio at 24.9 against an alignment ratio of 13.2. `math-6.md`
   §7.1's table and `status-6.md`'s summary still read as though the inversion
   were live evidence; `archive/README.md` rule 3's bullet on Phase 6 does too
   ("still carrying two live explanations, neither ruled out"). Three documents
   are behind one module.
3. **`p6_subspace/math-6.md` §4 item 4's proposal is unbuilt, not merely
   unprioritised.** The operator-derived cluster labelling is the third frame
   `frame_agreement` was written for, and no code produces it.

---

## 12. Before any of this becomes `design-9.md`

`CLAUDE.md` trigger 1 is still undischarged. `notes-9.md` §11 lists five
searches. This document's arguments add four more, and each of them could change
a construction rather than a citation:

- **Machine unlearning by inference-time representation editing** — as against
  weight editing and fine-tuning. §6.1's entire claim to novelty is that a metric
  patch is reversible and weight-preserving, and that claim is unverified.
- **Self-repair / Hydra Effect against non-ablation interventions.** §6.2 is the
  phase's best question *only if* nobody has already measured repair against a
  geometry-changing intervention. **Scanned 2026-09-20 (`lit-10.md` §7): every
  route the field names is an ablation route**, and the landscape summarises as
  self-repair *"affects the interpretation of every ablation experiment"*, with
  the automation methods working by joint ablation or by de-biasing a node's own
  score. **The question looks open** — but on `[S]` evidence only, which is too
  weak to register against, so it stays queued.
- **Oversmoothing and rank-collapse mitigation at inference time** — **scanned
  2026-09-20 (`lit-10.md` §8) and it is populated.** `2303.06562` **ContraNorm**
  is *a normalisation-layer modification that spreads representations apart* —
  the closest published object to a Phase 9 metric intervention, and it must be
  read before `design-9.md` freezes anything. Also `2410.07799` (*Mind the Gap*,
  spectral analysis of rank collapse) and `2602.09297` (*Laplacian Heads*, the
  deliberate-smoothing direction — i.e. the cluster-*forming* arm from the other
  side). **§6.4 is a comparison, not a discovery**; its value is predicting the
  effect from `gamma_beta` and reading it on the measure, which those do not.
- **Transfer-operator / PCCA+ / implied-timescale methods applied to transformer
  representations.** §5.1 is load-bearing and is imported wholesale from
  molecular dynamics; whether it has been tried here matters for positioning.

`arxiv.org` and other scholarly hosts are blocked by the session egress proxy
(`docs/LITERATURE.md`), so a scan run from a session like this one yields leads
marked `[S]`/`[N]`, not readings. That constraint should be recorded on the scan,
not worked around.

---

## 13. What this plan is not

- Not a design. No construction is specified to the level `design-9.md` requires.
- Not a registration. No `P-*` id names anything here, and
  `claims/registry.json` is untouched.
- Not a claim that clusters have a function. §3 enumerates four accounts
  precisely because the question is open, and H3 — that they do not — is a
  legible outcome, not a failure mode.
- Not a claim that metric intervention removes a capability. §6.1 says the
  default expectation is the opposite, and says why.
- Not a licence to read `cluster_count` as the phase's headline. It is the
  weakest row in §5.3's panel and should never be the primary readout.
