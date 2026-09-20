<!-- p10_cluster_function/notes-10.md -->
# Phase 10 — NOTES (pre-design, nothing frozen)

**What clusters are, and what they do.**

Opened 2026-09-20 out of `p9_metric_intervention/plan-9.md`, whose finding was
that Phase 9 cannot start at the intervention: you cannot use clusters as a tool
before you know what a cluster is. That prerequisite is a different question with
different instruments, it is large enough to carry its own literature and its own
registrations, and `design-5c.md` already recorded the reason not to bundle two
questions into one phase. **So it is its own phase, and Phase 9 parks on its
notes and plan until this one has an answer.**

> **ENTRY POINT, 2026-09-20: read `status-10.md` first.** Four rows of §8's
> ladder have run on real checkpoints. This file is kept as written — a
> pre-design workshop record — and results are NOT folded back into it, so
> anything here about what "is unrun" or "would be measured" is the state
> before 2026-09-20. `claims/registry.json` is still untouched and nothing that
> ran is registered.

**This is a workshop record, not a design.** No construction is frozen, no
instrument is specified to the level `design-10.md` requires, no `P-*` id names
anything here, and `claims/registry.json` is untouched. `CLAUDE.md` trigger 1 is
**partially** discharged: `lit-10.md` answers one question properly (the J-lens
artifact) and lists what it did not scan. §12 names the rest.

---

## 0. The claim this phase is built to test

In the user's framing, and it is the sharpest statement of the question anyone
has made in this project:

> A cluster might be **trash collection** — a compressed cleanup. One
> representation standing for one thing, with a lot of particles put into it in
> order to keep them stationary, keep them stagnant.

That is a real hypothesis with real consequences, it is not what "metastable
state" has meant in this project so far, and **three independent results already
on disk point at it.** §2 lays out the ledger; §3 turns the hypothesis into
something that can fail.

---

## 1. Why this is Phase 10 and not Phase 9a

Recorded because the alternative was live and the reasons are not obvious.

1. **Different instruments.** Phase 9's are interventional (a metric patch, a
   matched control, a KL readout). Phase 10's are a lens, a partition, a
   packing prediction and a particle table. Almost no overlap.
2. **Different literature.** Phase 9 sits against activation steering, concept
   erasure and unlearning. Phase 10 sits against the global-workspace/lens line,
   the causal-mask theory and the compression-valley line. **One `lit-N.md`
   cannot serve both**, and `CLAUDE.md` trigger 1 wants the scan that precedes
   the constructions it will change.
3. **The precedent is explicit.** `archive/p5c_unclustered/design-5c.md`,
   §"Why Group B is descoped rather than folded in": bundling a second research
   thread into a phase already scoped around one question *"would force Phase 5c
   to carry two separate questions to publication."* Same shape, same call.
4. **Phase 10 can succeed while Phase 9 is still parked, and is worth doing on
   its own.** If clusters turn out to be parking spaces, that is a result with
   no intervention attached.

**Directory: `p10_cluster_function/`.** Named for the question, on the pattern of
`p9_metric_intervention/`, not for the method.

**What Phase 10 does not take from Phase 9.** `plan-9.md`'s §2 and §3 (tier 0 and
tier 1) move here wholesale and are not duplicated there; `plan-9.md` will be
amended to point at this directory rather than restating it. Everything in
`plan-9.md` §4 onward — the congruence algebra, the insertion points, the sign
prediction — stays in Phase 9.

---

## 2. The evidence ledger: what is already measured about clusters

Nothing here is new work. It is scattered across five documents and has never
been read as one argument, which is most of why the hypothesis in §0 has not been
stated before.

| # | finding | where | bears on |
|---|---|---|---|
| 1 | **~50 % of tokens are HDBSCAN-clustered at any layer**, consistently across models and prompts | `math-1.md` §13.1 | the population split is stable, not incidental |
| 2 | **Unclustered tokens absorb ≈90 % of attention mass** by late layers, on ≈50 % of tokens — absent in the random case | `math-1.md` §13.1 | clustered tokens are where attention is *not* |
| 3 | Trained GPT-2-large routes **~1.6×** layer-average attention to unclustered tokens and **~0.5×** to clustered; ALBERT-base >2× / ~0.5×; **the sign is flipped under random weights in every model examined** | `archive/p5c_unclustered/status-5c.md`, cited in `PREDICTIONS.md` claim (a) | the split is *learned*, not architectural |
| 4 | **Cluster carrying capacity is invariant** — max simultaneously-alive clusters holds at **50–55 across all 27 checkpoints** — while **turnover is not**: mean lifespan 7.0 → 4.5, births 113 → 164 | `math-1.md` §13.2, `status-1.md` | a fixed number of slots with rising exchange |
| 5 | **Effective rank plateaus near 200–250** across models whose `d_model` spans **768–1600** | `PUBLICATION_IDEAS.md` idea 3 | a budget that does not scale with width |
| 6 | Trained models have **more, smaller, better-separated** clusters; random models have **fewer, larger, tighter** ones | `math-1.md` §13.1 | training *makes* this structure |
| 7 | Within-cluster cohesion stays **high and flat** across depth; between/noise inner products decline mid-model then rise near the known merge event. **The energy plateau is carried entirely by within-cluster pairs** | `archive/p5c_unclustered/status-5c.md` | once in, a particle stops moving relative to its neighbours |

**Read together, 1–7 describe a fixed-capacity store of low-attention,
high-cohesion, stationary particles, refilled faster as training proceeds.**
That is the user's hypothesis, and it is already most of the way to being
measured. What is missing is the one thing that would make it a claim about
*function* rather than about geometry: **nobody has asked what a clustered
particle is doing for the output.** §4 is the instrument that can.

**Debt carried, not hidden.** Finding 5's budget number is not publishable as it
stands — the plateau must be re-established on **normed** rank and ~200
reconciled against Blog 1's ~250 (open-question register item 3). Any Phase 10
argument leaning on it inherits that.

---

## 3. "Trash collection", made falsifiable

### 3.1 Four signatures of a parked particle, and the hypothesis is that they coincide

| signature | statement | instrument | built? |
|---|---|---|---|
| **kinematic** | its displacement per layer is small relative to the layer's | `core/dissipation.py` `tangential_velocity`, `w2_*` | yes |
| **attentional** | it receives below-average attention | `p1_mstate_tracking/visualization/noise_importance_proxy.py` (finding 3) | yes, **live** — and **unaudited**, see `attention-10.md` |
| **functional** | its readout is *not disposed to make the model say anything* — low-norm, high-entropy, unchanging across layers | **J-lens** (§4) | **the new one** |
| **causal** | resampling or ablating it costs little next-token KL | `core/intervention.py` + `core/lm_loading.py` | yes |

> **H-PARK: the four coincide.** A cluster is a parking space; membership means a
> particle has been set down, and being set down means kinematically still,
> attentionally ignored, functionally silent and causally cheap.
>
> **H-CAT (the rival): they dissociate.** Clustered particles are *quiet but
> load-bearing* — low attention and low motion, yet functionally and causally
> significant, because the cluster is a computed category whose identity is read
> downstream.

These are separable on the four-way concordance and nothing weaker separates
them. Note that findings 1–7 establish only the first two signatures; **the
entire discrimination lives in the two columns nobody has measured.**

**Amendment 2026-09-20 — the attentional column is not as solid as this table
implies.** `attention-10.md` audits it and finds four unchecked confounds, two
of them structural: position 0 is the sink and is unclustered by construction,
and the causal mask gives early tokens a mechanical advantage that the statistic
does not divide out. The measure is also a mean summed over all heads, and it
has never run on Pythia or on a checkpoint axis — though `attentions.npz` sits
in **152/152** directories of the 410m sweep, so the developmental version is
free. **Read `attention-10.md` before quoting the flip**, and treat the
attentional signature as one column to be re-established rather than as one
already in hand.

### 3.2 The quantitative form: Rényi parking

`2411.04990` (*Clustering in Causal Attention Masking*, Karagodin–Polyanskiy–
Rigollet) connects metastable states under **causal** masking — which is what
Pythia is — to the **Rényi parking problem**, and parking gives a density
constant (≈ 0.7476) and hence an expected number of occupied cells **as a
function of `n`**. `lit-1.md` §4 and `docs/LITERATURE.md` §6 both rate checking
it the project's best cheap experiment and say *"do this one first"*; it has not
been done.

It is the right experiment for this phase for a reason beyond cheapness:

> **If a cluster is a parking space, its count is set by packing — by `n` and by
> the geometry — and is largely insensitive to content. If a cluster is a
> computed category, its count tracks what the prompt is about and departs from
> the packing law.**

Phase 1 holds cluster counts per layer, per prompt length, at 27 checkpoints,
already on disk. It costs no forward pass. And `claims/adjudications/` holds
**zero entries against thirty-nine registrations** — a published quantitative
prediction tested against measurement is exactly the missing artifact.

**Corrected 2026-09-20, and the correction improves the test.** `lit-10.md` §5's
second scan finds the description above — inherited from `lit-1.md` §4 — is wrong
on both halves. The published scaling is **`Θ(β^((d−1)/2))`**, a law in **β and
dimension, not in `n`**, and the Rényi constant 0.7476 does not appear in it. At
`d = 1024` the exponent is 511.5 and the prediction is unusable — the `d ≫ 1`
problem `design-1.md` already records for Figure 3. **Registering F0 on the old
reading would have frozen the wrong statistic**, which is what `CLAUDE.md`
trigger 2 exists to prevent.

What replaces it, from `math-10.md` §5, is better:

- **The slope test.** Fit `log count ~ a·log β + b·log n`. The coefficient
  `a = (d_eff − 1)/2` is **invariant to β's undecided unit convention** — a
  constant factor moves the intercept, not the slope — so a test that looked
  blocked on that decision is not. And it **measures** the effective dimension
  the clustering behaves as rather than confirming a number: the candidates
  already on disk (ambient 1024, effective-rank plateau ~225, participation
  ratio 22) predict slopes five orders of magnitude apart.
- **The anchor test, and it is free today.** The mechanism the paper supplies is
  that **early tokens act as nuclei** for cluster formation. That is a
  per-token, position-indexed prediction checkable against `hdbscan_labels.json`
  plus positions with no β, no convention decision and no reading of the paper.
  **This is now F0.**

The exact correspondence remains `[S]` and is the top item in `lit-10.md` §10.

### 3.3 The turnover question is the phase's other half, and the instrument exists

Finding 4 has two readings that cluster-level statistics **cannot** separate,
because both produce identical births/deaths/lifespan curves:

- **(a)** the *same* particles cycle through clusters faster;
- **(b)** *different* particles cluster at late checkpoints.

`turnover_decomposition` is the test, it is **a groupby, not a new experiment**
(`math-5.md` §8.1), it was built and validated on two synthetic sweeps with
identical falling mean lifespan — same-particles gives J = 1.000 at every
threshold and rank correlation 1.000; different-particles gives J = 0.000
first-vs-last and rank correlation −0.667 — and it is **awaiting the real
sweep**. It lives in `archive/p5_single_mstate_analysis/particle_join.py`, so
under `archive/README.md` rule 2 it is **rebuilt against `core/particles.py`, not
lifted** — and `core/particles.py` is already the schema it would be rebuilt
onto, which is the cheapest possible version of that rule.

Its design carries a discipline worth keeping: **D-20 — `turnover_decomposition`
returns no verdict**, deliberately, with a test asserting no `verdict` key
exists, because the two readings are not exhaustive and real numbers can land
between them.

Why it matters for H-PARK: parking with a fixed capacity and rising births is
**reading (a) plus eviction** — the same particles cycling. Reading (b) says the
*population being parked* changes over training, which is a claim about what
training learns to discard, and a much more interesting one.

### 3.4 What would falsify H-PARK

Stated now so it cannot be chosen later.

1. The four signatures **dissociate** — in particular, clustered particles that
   are causally expensive to resample.
2. Cluster count **tracks content** and departs from the packing law in a way
   prompt length does not explain.
3. Functional ARI between the geometric partition and a J-lens-derived one is at
   **chance** under §4.4's null — which would say the geometric partition is not
   a partition of function at all, and would push the phase toward `plan-9.md`
   §3's H3 (an epiphenomenon of concentration).

**Outcome 3 is a legible pre-registered outcome, not a failure.** Same discipline
as `math-5c.md` §4's three-outcome rank test and `math-6.md`'s three-reading
table: register the outcome that embarrasses the phase.

---

## 4. The instrument that changes what is possible: the Jacobian lens

`lit-10.md` §1 is the scan. The short version, and then what it unlocks. **`math-10.md` is this phase's derivations**, four symbolic-check files, three of them corrections to statements this repository currently makes.

### 4.1 What it is

`J_l = E[∂h_final/∂h_l]`, the **averaged input–output Jacobian** from layer `l` to
the final layer, expectation over prompts, source positions and all target
positions; readout `lens_l(h) = unembed(J_l @ h)`. Reference implementation
`github.com/anthropics/jacobian-lens`, Apache 2.0, read directly. Pre-fitted
lenses for 38 open models at `neuronpedia/jacobian-lens` **[S]**, reported to
include `pythia-70m-deduped`, one `[d_model, d_model]` fp16 matrix per layer.

**This project has been citing the paper for two months** — `lens_band.py` and
`archive/p5_single_mstate_analysis/status-5.md` both — while recording that
training the lens was *"deliberately out of scope"*. It is now a licensed,
~100-prompt procedure with published code.

### 4.2 `J_l` is a `d × d` operator, and this project is built for those

The point that makes this more than a nicer logit lens.

Every operator instrument in this repository takes a `d × d` (or head-shaped)
matrix and decomposes it: the real Schur decomposition, the S/A split, the
attracting/repelling projectors (`p6_subspace/subspace_geometry.py`),
`φ(M) = ‖Λ‖²/‖M‖²`, `core/dual_reading.py`'s fields. All of them have only ever
been pointed at **weights** — `M_OV = W_Oᵀ W_Vᵀ`, `M_QK = W_Q W_Kᵀ`.

`MATH_SPECTRAL_OT.md` §2.4.6 is the standing negative result: **no weights-only
spectral quantity computed by this project identifies the copier** — `L7H8` and
`L2H10` agree to three digits on every eigenvalue-derived field and their causal
effects have opposite sign.

> **`J_l` is not weights-only.** It is an expectation over real activations of the
> forward influence of layer `l` on the output. It is a `d × d` operator
> *defined by function*. Running this project's existing operator decompositions
> on `J_l` is a one-line reuse against exactly the kind of object §2.4.6 said the
> weights do not supply.

That is a genuinely new thing to do with the lens and it is native here in a way
it would not be anywhere else. It is speculative — nobody has shown `J_l`'s
spectrum means anything — and it should be labelled exploratory. But the cost is
an import.

### 4.3 The functional partition becomes measurable per layer, which it was not

`plan-9.md` §2.1's tier-0 gate wanted a **functional** cluster labelling to set
against the geometric one. `core/functional_distance.py` builds it: pairwise KL
over decoded next-token distributions in a single matmul, HDBSCAN on the
divergence matrix, and `frame_agreement` for ARI across labellings. It is built,
tested, and **has never been called on real data.**

Its docstring says it consumes *"the per-layer decoded distributions
(`p5b_manifold/logit_cache.py`'s arrays, or `tuned_lens_cluster.py`'s
frozen-head/tuned-lens outputs)"*. The available producers were the logit lens —
noisy exactly in the early layers where the cluster story begins — and a tuned
lens with a documented skip-to-output pathology.

> **A J-lens readout drops into that slot with nothing to change.** `unembed(J_l @ h_i)`
> per token is a per-layer distribution, and the functional partition becomes
> measurable at **every** layer rather than only where the logit lens is
> trustworthy.

So the single most informative unrun number in the tree — the geometric/functional
ARI — is not only reachable, it is reachable *per layer*, which is the axis the
whole phase lives on.

### 4.4 Chance, and the null this needs before any ARI is quoted

**Corrected 2026-09-20 — the instruction stands, the reason was wrong.**
This section previously said a size-profile-preserving null was needed to remove
a bias. `math-10.md` §4 shows the **adjusted** Rand index already subtracts
exactly that expectation, so `E[ARI] = 0` under that null by construction,
whatever the size profiles — confirmed by Monte Carlo at three regimes including
a 200 + 8×8 giant-cluster profile.

**The null is needed for the variance, not the centering — and the variance is
where the real problem was.** Measured 95th percentiles under the null: **+0.009**
(balanced), **+0.092** (one giant cluster), **+0.002** (giant vs balanced). A
**57× range**. An ARI of 0.05 is unremarkable under one profile and a strong
signal under another, so **a fixed ARI threshold is not comparable across
layers, checkpoints or models** — `compare_rungs.py`'s no-absolute-threshold rule
one level up, on an axis where HDBSCAN's size profile is exactly what moves.
Build the null in `core/nulls.py` for the p-value.

**And a hazard no adjustment touches: HDBSCAN noise is not a cluster.** Treating
`−1` as one cluster versus dropping those points gives different ARIs on the same
data, `adjusted_rand_index(ignore_noise=...)` exposes the choice, and 40–50 % of
tokens are noise. **Fix it before looking.**

This project has been eaten by a scale/dimension confound three times
(`math-2b.md` §2.3's energy-vs-dimension, `math-6.md` §7.2's alignment-vs-dimension,
`math-1b.md` §6's un-nulled cone verdict). **Naming it as a recurring class in
`design-10.md` is cheaper than catching it a fourth time.**

### 4.5 Four caveats, each of which changes what may be claimed

1. **`pythia-70m-deduped` is not `pythia-70m`.** Different training run, different
   corpus. Every registered 70m decision in this project names `pythia-70m`. The
   published lens is not licensed on the ladder's model without a check.
2. **The published lens has no checkpoint axis.** It is one fit, presumably at a
   final revision. **The developmental question — which is this project's whole
   object — needs our own fits.** At `d = 512` and 6 layers that is ~3 MB and
   ~100 prompts per checkpoint; `jlens.merge` parallelises it.
3. **It is an average, and an average of a linearization.** `MATH_SPECTRAL_OT.md`
   §2.1/§2.2 already state the project's discipline about reading linearizations
   at depth, and it applies here. `J_l` tells you the *mean* forward influence,
   not this token's.
4. **A lens is a readout, not a ground truth.** The project's own record is that
   structural proxies have failed against causal ground truth twice
   (§2.4.6; `status-7e.md`'s `useful_rank --bottom` inversion). **The J-lens is a
   candidate generator that happens to be defined by function — better prior than
   an SAE, still not an identifier.** Every functional claim wants the causal
   column of §3.1 beside it.

---

## 5. Two different things are being called a cluster

This distinction is not currently drawn anywhere in the project and it is
underneath the MLP question in §7.

- **A token cluster** is a set of **particles** — points in the residual stream,
  indexed by position. This is what HDBSCAN finds, what Phase 1 tracks, what
  `core/particles.py` is keyed on, and what the theory is about.
- **A direction cluster** is a set of **features** — an SAE dictionary element, a
  neuron, a singular direction. These are not particles. They do not move, they
  are not indexed by position, and no theorem in this project's literature is
  about them.

"Form a cluster around an SAE feature" mixes the two: the feature is a direction,
the thing you would cluster is the set of *tokens on which it is active*. That
is a coherent operation, but it should be written as **"cluster the particles
selected by a direction"**, because the alternative reading — cluster the
directions — is a different experiment with different mathematics.

**The J-lens is the bridge**, and it is the reason this is worth writing down
rather than just avoiding: `J_l` maps *any* residual direction into vocabulary
space. A token's state and a feature's direction can be read in the same units.
That makes "what is this cluster about?" and "what is this feature about?" the
same measurement for the first time in this project.

---

## 6. The intervention taxonomy, and the cell that is empty

The user's passive/ablative versus active/forming distinction, crossed with the
object acted on. Every intervention in this project so far:

| | **subtractive / passive** — "was it needed?" | **additive / active** — "what if I make it happen?" |
|---|---|---|
| **weights** | zero- and mean-ablation (7d/7e), rank truncation (`useful_rank`), projecting out a subspace | the `2²` sign factorial (§2.4.2), the isometric path (§2.5), writing `mu_cond` into MLP 6's slot (§3.23) |
| **metric / geometry** | — | **empty. This is Phase 9.** |

Two things fall out.

1. **The empty cell is not the only gap** — there is no *subtractive* metric
   intervention either, and it is the cheaper of the two to interpret. "Remove
   the metric's anisotropy on this subspace" (set `Γ` to isotropic on `U`) is
   the natural passive control for every active `Γ` patch, and it is the
   matched-control shape the design wants anyway.
2. **Writing `mu_cond` into MLP 6's slot is already an additive weights
   intervention**, and it worked (§3.23). It is the closest precedent in the
   project to what Phase 9 proposes, and it should be read as one.

---

## 7. The MLP: the object is a direction, and what "collapse the space" can mean there

Asked directly: *can we find a specific object in the MLP and see what happens
when we collapse the space around it?*

**The object has already been found, and it is not a neuron.**
`PROJECT.md` §3.23 / `p7d_redundancy/mlp6_decode_direction.py`: with `L5H2`
ablated, `L7H8`'s matching depends on a specific direction in MLP 6's output.
Writing MLP 6's post-ablation mean `mu_cond` into its slot **restores the
matcher** (attention 0.643 / 0.567 at steps 16000 / 143000), while its clean-state
mean `mu_clean` restores **nothing** (0.038 / 0.016 — on a par with zero and with
a norm-matched random constant). The rotation between them is only
`cos = 0.837`, so **the entire effect lives in the component of `mu_cond`
orthogonal to what MLP 6 was already writing.**

And `mlp6_content_vs_scale.py` establishes the character: the load-bearing
contribution is carried by the **constant** component, not by what MLP 6 computes
per token — or else `zero` is off-distribution in the way §3.15 warns about and
`mean` is the honest number. The runner separates those.

### 7.1 What follows, and it constrains the experiment

§3.29's reading: MLP 6's repair is **an external field, not an interaction.** It
is position-independent, so **it cannot change the pairwise coupling**; what it
changes is the **cost geometry** the coupling is computed in.

> **You cannot form a token cluster inside an MLP by pulling particles together.**
> A token-wise map moves every particle by (approximately) the same vector plus a
> token-wise nonlinearity; there is no particle-particle term for it to act on.
> What an MLP-side augmentation does is **move the whole cloud and reshape the
> geometry the next attention layer's coupling is computed in.**

So the attention-side and MLP-side experiments are **different operations, not
the same operation at two sites**:

- **Attention-side** (`input_layernorm`): change *who couples to whom*. This is
  where a cluster can be formed or dissolved.
- **MLP-side** (`post_attention_layernorm`): change *the geometry the coupling is
  computed in*. This is where a cluster can be made **easier or harder to form**
  at the next layer.

`plan-9.md` §4.2's three-arm factorial is exactly the instrument that separates
them, and Pythia's parallel residual makes the split exact. **The MLP arm's
prediction is second-order and delayed by a layer** — that is a design constraint
worth stating, because an MLP-side patch measured at its own layer will look like
it did nothing.

### 7.2 The honest MLP "collapse" experiments, in order

1. **Collapse toward the known direction.** Project MLP 6's output onto
   `span(mu_cond)` — keep only the direction that is known to carry the function
   — and measure. The complement arm (project it *out*) is the passive control.
   Both are cheap, both use a runner that already exists, and the answer is a
   fact about whether "the object" is the whole object.
2. **Neuron-basis collapse.** Tie a group of MLP neurons to a shared write
   direction (a *direction* cluster, §5) and ask what happens to the *token*
   cluster structure downstream. This is the one experiment that genuinely tests
   whether the two senses of "cluster" are related.
3. **The metric arm**, which is Phase 9's and belongs there: patch
   `post_attention_layernorm`'s `Γ` and read the effect at layer `l+1`'s
   attention, not at layer `l`.

**Carry `status-7e.md`'s hold into all three:** any SVD-ordered rank truncation
misleads on `L11H14`-like objects, where keeping the top singular direction is
*worse than deleting the head* (−0.096 at `r = 1`). `--bottom` is checked first,
every time.

---

## 8. The ladder

Costs and dependencies. Ordering is a proposal; nothing is registered.

| # | experiment | forward pass? | depends on | decides |
|---|---|---|---|---|
| **F0** | **The anchor test**: are cluster nuclei early tokens, as the parking reading says? Per-token, position-indexed, against `hdbscan_labels.json` + positions. | no | **nothing** | §3.2 as corrected. Needs no β, no convention decision and no reading of the paper |
| **F0b** | **The slope test**: fit `log count ~ a·log β + b·log n`; `d_eff = 2a + 1`. | no | the β producer (`docs/AXES.md` §4) | §3.2. Convention-free in the slope; **measures** `d_eff` rather than confirming a constant |
| **F1** | **Transport observables** (`w2_identity` vs `w2_optimal`, arc length, `straightness`) on artifacts on disk — still the cheapest open action in the tree, carried over from `notes-9.md` §9. | no | nothing | the kinematic signature of §3.1, and it re-reads every displacement number already recorded |
| **F2** | **Verify the J-lens artifacts** on the research machine: does `neuronpedia/jacobian-lens` carry `pythia-70m-deduped`, what shape, what revision. | no | HF access (blocked from a cloud session) | whether §4 borrows or fits |
| **F3** | **Fit a J-lens on `pythia-70m`**, one checkpoint first, then the axis. ~100 prompts, ~3 MB per checkpoint. | yes (backward) | F2 | **the developmental J-space nobody appears to have** |
| **F4** | **Per-layer functional partition + three-frame ARI**, against §4.4's size-profile null. | yes | F3 | §3.4 outcome 3. The tier-0 gate `plan-9.md` was written around |
| **F5** | **Four-signature concordance** per particle: kinematic, attentional, functional, causal. | yes | F1, F4 | **H-PARK vs H-CAT.** The phase's central test |
| **F6** | **`turnover_decomposition`, rebuilt** against `core/particles.py`, run on the real sweep. | no | the particle table populated | §3.3's (a) vs (b). Instrument validated in 2026, awaiting data ever since |
| **F7** | **Centroid substitution** with matched random-displacement control, on trained model and random twin — Phase 5c's Group D, revived. | yes | F4 | the causal column, and `plan-9.md` §3's H1/H2/H3 |
| **F8** | **Operator decomposition of `J_l`** — Schur, S/A split, `φ`, the attracting/repelling projectors — against the same decompositions of `M_OV`. | no | F3 | §4.2. Exploratory, labelled, and cheap |
| **F9** | **MLP object collapse**: project MLP 6's output onto / out of `span(mu_cond)`. | yes | nothing | §7.2 item 1 |
| **F10** | **Neuron-basis collapse** and its effect on token clusters. | yes | F4 | §7.2 item 2 — whether the two senses of "cluster" are related |
| **F11** | **The attention audit**, `attention-10.md` §6 rows A0–A8. A0 (sink and causal-mask baselines) gates the rest; A2 (the checkpoint axis) and A4 (the population×population mass matrix) carry the most information per unit of work. | no | nothing | the attentional column of §3.1, and **A4 is a direct `H-PARK` vs `H-CAT` test** |
| **F12** | **`Z_beta,i` per token** — the trained per-token metric, never examined. | no | nothing | parked vs **pinned**: two kinds of stationary that displacement alone cannot separate (`attention-10.md` §5) |

> **AMENDMENT 2026-09-20 — all four of those rows have RUN. See
> `status-10.md`, which is now the phase's entry point; this file stays a
> pre-design workshop record and is NOT updated with results.** In brief: A0
> found the attention flip is ~94 % causal mask and entirely mask at
> initialisation; **F0 came back against its own prediction** (nuclei late, not
> early) under both the ordinary null and a restricted one that had to be added
> because the ordinary one could not tell nucleation from the clustered/noise
> position split; F1 found the identity coupling exactly optimal in 99.5 % of
> boundaries and the kinematic signature to be a **window** at steps 32–512;
> F12 confirmed §2's masked-`Z` derivation β-independently and, read against a
> step-0 baseline, joins F1 in saying **parked rather than pinned** — in that
> same window, not in the trained model. **Two blockers had to be cleared
> first**: `hdbscan_labels.json` was empty in 152/152 directories, and the
> partition is not reproducible run to run.
>
> **§3.2's F0 as written is not what ran.** The row needed a second null, and
> §4.4's hazard list needed a fourth entry — a **measurement-reproducibility
> floor** that no size-profile null accounts for. Both are in `status-10.md`.

**F0, F1, F11 and F12 are free and unblocked today** — and F0 is now the cheapest of them. F2 costs a directory
listing. F6 is a rebuild of a validated instrument, not a new one.

**`docs/AXES.md` is the grid this ladder draws on** — which axes exist, which
cells are populated, which producers do not exist, and the ten rules that
constrain combining them.

**Standing constraint, inherited** (`MATH_SPECTRAL_OT.md` §6.1): a new subphase
directory importing existing outputs read-only. Nothing is added as a stage
inside `p2_eigenspectra/run_2.py` or `p1_mstate_tracking/run_1.py`.

---

## 9. What each phase owes the other

**Phase 10 → Phase 9.** The tier-0 answer. If H-PARK holds, Phase 9's "form a
cluster" becomes *"park these particles"* and its forgetting application has a
mechanism rather than a hope. If H-CAT holds, forming a cluster destroys
information and the intervention is an edit rather than a release. If §3.4
outcome 3 holds, Phase 9's object moves off the partition entirely.

**Phase 9 → Phase 10.** The causal test of the packing account, and it is a
quantitative one: **if cluster count is set by packing, a metric patch that
changes the effective volume available should move the count by the amount the
packing law predicts, and a content intervention should not.** That is a
differential prediction which needs both phases and neither can make alone.

---

## 10. Hazards

### 10.1 A correction to Phase 9's own framing, found here

`2411.04990`'s first claim: **the causally-masked system cannot be interpreted as
a mean-field gradient flow.** Pythia is causal. `docs/LITERATURE.md` row 6
already asks whether this voids Phase 2d's gradient-flow framing.

> **It bears on Phase 9 the same way, and nobody has said so.** `notes-9.md` §8
> and `plan-9.md` §5.1 are built on the **Wasserstein Hessian of `E_beta`** —
> "stretch a subspace" restated as "change the curvature of a near-zero
> eigendirection". That restatement presupposes the gradient-flow structure the
> masked theory says is absent.

What survives and what does not, stated carefully because the distinction is the
whole value of noticing it:

- **At risk:** the Hessian-of-`E_beta` framing, and any claim that a metric patch
  moves a curvature. Also `p2d_operator_activation/gradient_flow_condition.py`'s
  premise is narrower than it looks — it tests the paper's condition
  (`QᵀK` symmetric **and** `V = QᵀK`) for the **unmasked** reweighted-metric
  result, and a decoder may be outside the framing before that condition is asked.
- **Survives:** the **transfer-operator / implied-timescale / PCCA+** readout
  (`plan-9.md` §5.1's second half). A transfer operator needs a transition
  structure, not a gradient-flow structure, and `cluster_tracking.py` already
  computes transitions. **This is a reason to prefer the timescale readout over
  the Hessian one, independent of cost.**
- **Also survives:** everything algebraic in `plan-9.md` §4 — the congruence, the
  Lemma 6.4 ceiling (which is proved from positivity of `a_ij` alone and is
  therefore mask-agnostic), the cone margin, the sign prediction.

**This is `[S]`-grade and must be confirmed by reading `2411.04990`** before
anything is retracted. It is recorded now because it is the kind of thing that
gets discovered after a design freezes.

### 10.2 The rest

1. **The parking correspondence turns out to be qualitative.** If `2411.04990`'s
   link is a structural analogy rather than a formula in `n`, F0 is not an
   adjudication and should be relabelled exploratory rather than stretched.
2. **The lens becomes the ground truth.** §4.5 caveat 4. The causal column exists
   to stop this.
3. **Four signatures, four thresholds, one story.** Concordance across four
   binarised signatures invites threshold-shopping. The design should fix them
   before looking, or report the concordance as a function of threshold and never
   as a single number — `status-7e.md`'s `--bottom` sweep is the pattern.
4. **`pythia-70m-deduped` quietly becomes "70m".** §4.5 caveat 1. A labelling
   discipline, and the project has the machinery (`core/naming.py`,
   `core/model_selection.py`) to enforce it.
5. **The phase becomes a lens demo.** The J-lens is a means. The question is what
   clusters do.

---

## 11. Registration discipline

`PROJECT.md` §3.29's three tiers, applied before any result exists.

- **F1, F2, F3, F4, F6, F8 are tier 1** — exploratory, labelled, no p-value.
- **F0 is the phase's tier-2/3 candidate and the best one in the project.** A
  published quantitative prediction, a measurement already on disk, and a rival
  account (content-driven clustering) that predicts a different answer. It is
  differential in the sense §3.29 demands, and it is not an instance-level claim.
- **F5 is tier-2 if the four signatures are fixed in advance and tier-1 if they
  are not.** That is a design decision, and it should be taken in `design-10.md`
  rather than after the concordance is seen.
- **No reserved rung is spent.** 70m and 410m only; `pythia-1b` and `pythia-1.4b`
  stay reserved.
- **Thirty-nine registrations, zero adjudications.** This phase should aim to
  produce the first, not the fortieth registration.

---

## 12. Before any of this becomes `design-10.md`

`lit-10.md` discharged one question. The searches that would change a
construction rather than a citation:

- **`2411.04990` read as full text** — the exact parking correspondence (§3.2),
  and the gradient-flow claim (§10.1). **Two phases depend on this one paper and
  neither has read it.** Highest priority in the project's whole verification
  queue.
- **Has anyone clustered tokens in a lens basis** rather than in the residual
  basis? §4.3 is the phase's central instrument and its novelty is unchecked.
- **Has the parking prediction been checked empirically** by anyone, on any
  model? F0's whole value is that it has not.
- **Spectral analysis of averaged Jacobians / transport operators** in
  interpretability. §4.2 is speculative and would be much less so if it has been
  tried.
- **"What do clusters in the residual stream mean"** as a literature in its own
  right — the compression-valley line (`2510.06477`), the developmental-geometry
  line (`2509.23024`), and whatever sits between them. Phase 10's novelty must be
  located against these, and `lit-1.md` already grades both **NOT NEW** as
  phenomena.

`arxiv.org`, `transformer-circuits.pub`, `huggingface.co` and `neuronpedia.org`
are blocked by the session egress proxy; **`github.com` is not** (`lit-10.md`
opens with the table). Companion-code repositories are a primary source from
here.

---

## 13. What this phase is not

> **AMENDMENT 2026-09-20.** Everything in this section still holds — but four
> rows have now RUN, and **`status-10.md` is the phase's entry point.** This
> file remains what it says it is: a pre-design workshop record, kept as
> written so the reasoning that produced the ladder stays legible. Results are
> not folded back into it.

- Not a design. Nothing is specified to the level `design-10.md` requires.
- Not a registration. No `P-*` id, and `claims/registry.json` is untouched.
- Not a claim that clusters are trash collection. §3 makes it falsifiable; §3.4
  lists what would falsify it, including the outcome that would dissolve the
  phase's object.
- Not a replacement for Phase 9. Phase 9 is parked on `notes-9.md` and
  `plan-9.md`, not closed, and §9 states what each owes the other.
- Not a licence to treat the J-lens as ground truth (§4.5 caveat 4) or to quote
  `pythia-70m-deduped` results as `pythia-70m` (§4.5 caveat 1).
