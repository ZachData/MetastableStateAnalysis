# PARTICLE_ONTOLOGY.md — interpretability constructs, defined in the particle paradigm

POPPER_PLAN.md item C1. Written 2026-08-23, **before any of the code it
describes**, per the norm `core/DESIGN_dual_reading.md` already follows: a
schema written after its implementation documents what happened, not what was
decided.

## The constraint this document is written under

This project studies tokens as **particles on $S^{d-1}$ evolving under an
interacting particle system** — the Geshkovski et al. dynamics, with each
residual block read as a forward-Euler step of

$$\dot{x}_i = \mathbb{P}^\perp_{x_i}\Big(\sum_j a_{ij}\, V x_j\Big),\qquad
a_{ij} = \frac{e^{\beta\langle Qx_i,\,Kx_j\rangle}}{\sum_k e^{\beta\langle Qx_i,\,Kx_k\rangle}}.$$

Mainstream interpretability studies the same forward pass with a different
vocabulary: induction heads, steering vectors, SAE features, lenses, patching,
probes. Both are descriptions of the same computation.

**A translation table between the two would be worthless.** Re-describing an
induction head as "a coupling term" predicts nothing, cannot be wrong, and buys
only the appearance of unification. Popper's criterion applies to the bridge
itself, not only to the experiments underneath it.

So the rule for this document, applied without exception:

> Every entry must state a **differential prediction** — something the particle
> account says that the standard account does not, at a point where the two
> disagree observably. An entry with no differential prediction is a glossary
> line and does not belong here.

Where a construct genuinely admits no differential prediction yet, that is
recorded as such rather than papered over with a plausible-sounding one. Two of
the six below are in that state.

## Why `core/dual_reading.py` is the bridge primitive, not a new module

The bridge does not need new machinery. `core/dual_reading.py` was designed to
return, for one point of interest, a **paired** reading: a *geometric* half
(V-attractive/repulsive fractions, real/imaginary split, effective-rank
contribution) and a *semantic* half (frozen-head decode top-k and entropy,
LDA projection, probe membership). That pairing is exactly the bridge —
the same particle, read in both vocabularies, in one call.

Every entry below is therefore written against that existing schema
(`geometric.v_attractive_frac`, `geometric.imag_frac`, `semantic.decode_top1_token`,
…) rather than proposing a parallel one. Where an entry needs a quantity
`dual_reading` does not return, that is called out as a required extension.

## The unit of analysis

`core/particles.py` already fixes it: a **particle record** keyed by
`(model, checkpoint_step, prompt_key, layer, token_position)`, with cluster
label and population as *annotations* rather than as the unit. Plan v2 promoted
Phase 5c's framing — "every particle and how it evolves; clustering is one
annotation on a particle" — to be the project's. Every definition below is a
statement about particles or about the field that moves them, and reduces to
operations on that table.

---

## 1. Induction head

**Standard definition.** A head that, having seen `[A][B] … [A]`, attends from
the second `[A]` to the token after the first one and increases the logit of
`[B]`. Identified by an induction score: mean post-softmax attention on
prefix-matching pairs.

**Particle-paradigm definition.** An **inter-particle coupling with a matching
kernel**: a term in the velocity field whose weight $a_{ij}$ is determined by a
*token-identity match between the predecessors of $i$ and $j$* rather than by
the current positions $x_i, x_j$. This is the specific thing that makes it
strange in the paradigm — the Geshkovski dynamics are position-coupled by
construction ($a_{ij}$ depends on $\langle Qx_i, Kx_j\rangle$), so a coupling
keyed on a match at *other positions* is a non-local transport term the base
model does not contain.

**Already in this repo.** `p6_subspace/induction_ov.py` scores heads by mean
post-softmax attention on canonical induction pairs and asks whether their OV
write directions land in the imaginary/antisymmetric subspace $U_A$ (P6-I1);
`p6_subspace/qk_decompose.py` measures QK antisymmetry for induction versus
same-content pairs (P6-I2). `p6_subspace/head_classify.py` supplies the
semantic-head contrast arm.

**Differential prediction (candidate `PB-IND1`).** If induction is a
matching-kernel *transport* term rather than a feature-copying circuit, then
ablating it must change **inter-particle geometry at the matched positions** —
the pairwise-distance distribution among the particles the head couples — and
not only the logit at the copied token. The standard account predicts the logit
effect and is silent about the geometry.

*Falsifier, stated in the direction that hurts:* a large logit effect with a
pairwise-distance change indistinguishable from the matched control means the
head moves the readout without moving the particles, and the transport reading
is wrong. Adjudicated on the joint outcome, not on the geometric effect alone —
a geometric effect with no logit effect would falsify it equally, in the other
direction.

*Instrument:* `p6_subspace/induction_ov.py` for the head set,
`core/intervention.py` for the ablation, `core/functional_distance.py` for the
readout. Needs `dual_reading` to expose a **pairwise** quantity; today every
`geometric` field is per-point. That extension is the chunk's first task.

---

## 2. Activation steering

**Standard definition.** Add a vector $s$ to the residual stream at some layer
and observe a directed behavioural change; $s$ is read as a "feature direction",
and effect size is expected to scale with $\|s\|$.

**Particle-paradigm definition.** An **exogenous impulse added to a particle's
velocity**. In a system whose intrinsic field has an attractive and a repulsive
part — the $V$-eigenbasis split this project already builds
(`p6_subspace/subspace_build.py`'s `U_pos`/`U_neg`) — an impulse is not
characterized by its magnitude but by its **decomposition in that basis**. A
component along $U_\text{pos}$ adds to the collapsing field; a component along
$U_\text{neg}$ opposes it.

**Already in this repo.** `p5b_manifold_steering/`, `core/intervention.py`,
`p3_crosscoder/steering.py` (frozen). `dual_reading`'s
`geometric.v_attractive_frac` / `v_repulsive_frac` *are* the decomposition this
definition needs, already specified.

**Differential prediction (candidate `PB-STEER1`).** Two steering vectors of
**equal norm** whose $V$-eigenbasis decompositions have opposite sign —
predominantly attractive versus predominantly repulsive — produce
**opposite-signed changes in effective rank** of the token population at the
injection layer. The standard account predicts effect size scales with $\|s\|$
along a feature direction and makes no sign prediction at all.

*Falsifier:* effect on effective rank tracks $\|s\|$ and is insensitive to the
decomposition; or both signs move effective rank the same way. Either outcome
says the impulse reading adds nothing over "a vector was added".

*Why this one is worth running first:* it is cheap, it needs no new
instrument, and it is the entry where the two accounts make *incompatible*
rather than merely different predictions. Norm-matching is the whole design —
without it the comparison is confounded by exactly the quantity the standard
account cares about.

---

## 3. SAE / crosscoder feature

**Standard definition.** A dictionary element of a sparse autoencoder trained on
activations; read as a "feature the model uses", with sparsity as evidence of
having found the model's own basis.

**Particle-paradigm definition.** A claimed **coordinate of the particle
configuration** — a direction along which some subpopulation of particles is
distributed with structure. Under this reading the dictionary is a claim about
*where the population actually has extent*, which is a measurable property of
the configuration and not of the autoencoder.

**Differential prediction (candidate `PB-SAE1`).** Dictionary elements should
concentrate in the **repulsive** subspace $U_\text{neg}$ — the directions along
which the dynamics permit particles to stay individuated — rather than
distributing isotropically across $U_\text{pos}$ and $U_\text{neg}$. Along
attractive directions the field is actively collapsing the population, so there
is progressively less extent for a dictionary element to encode.

*Falsifier:* dictionary elements distribute isotropically across the two
subspaces, or concentrate in $U_\text{pos}$.

**Status: registered and deliberately unrun.** Phase 3 and Phase 4's
`low_rank_ae.py` are **frozen-for-deletion** (`INDEX.md`; reintroduction
requires activation caches at ≥4 checkpoints *and* a specific particle-dynamics
question needing a dictionary). This entry is the second condition arriving
first. It does not unfreeze anything: a registered prediction whose instrument
is frozen is in a correct state, and registering it now is what makes the
freeze reversible on evidence rather than on enthusiasm.

*Caveat that must travel with this prediction.* P6-R2 and P6-R4 already came
back **inverted** — LDA alignment 0.887 with the *imaginary* subspace $U_A$
against 0.067 with the real repulsive $U_\text{neg}$, 0 of 49 layers in the
predicted direction (`p6_subspace/status-6.md`). That is the closest existing
evidence to this prediction's premise and it points the other way. `PB-SAE1` is
registered in the direction the theory implies, not in the direction the
existing measurement suggests, precisely so that the theory can lose.

---

## 4. Logit lens / tuned lens

**Standard definition.** Project an intermediate residual-stream vector through
the unembedding to read "what the model is thinking" at that layer; the tuned
variant fits an affine correction per layer.

**Particle-paradigm definition.** The **readout map** from particle position to
the vocabulary simplex, $x \mapsto \mathrm{softmax}(W_U x)$. Under this reading
"the lens works" is a claim that the map is *approximately isometric on the
directions the population occupies* — that particle-space distances correspond
to readout-distribution distances. The tuned lens's affine correction is then
not a fix for a broken probe but a statement that the readout frame **rotates
with depth**.

**Already in this repo.** `p2_eigenspectra/lens_band.py`,
`p2_eigenspectra/vocab_projection.py`,
`p5_single_mstate_analysis/tuned_lens_cluster.py`,
`p5b_manifold_steering/logit_cache.py`. `dual_reading`'s entire `semantic` half
*is* a lens reading.

**Differential prediction: none yet.** The particle reading and the standard
reading agree on every currently measurable consequence I can state. `P5b-B1`
(the frame the model reads in predicts behaviour better than a Euclidean
baseline) is adjacent but is a claim about the *manifold*, not about the lens.

Recording this as an open gap rather than inventing a prediction is the point of
the rule at the top of this document. The obvious candidate — that tuned-lens
correction magnitude per layer should track the rotational/imaginary content
measured by `p2b_imaginary/rotational_schur.py` — is *plausible* and *not yet
sharp enough to falsify*: it needs a stated functional form before it is a
prediction rather than a hope. That sharpening is a chunk; it is not done here.

---

## 5. Ablation / activation patching

**Standard definition.** Zero, mean-ablate, or substitute a component's output
and measure the change in a behavioural metric; attribute the difference to the
component.

**Particle-paradigm definition.** **Deleting or substituting one term in the
velocity field**, then measuring the divergence of the resulting trajectory from
the unablated one. The reframe is not cosmetic: in a dynamical system, the
effect of removing a term is a property of the *trajectory*, not of the
endpoint, and it compounds with depth in a way a single-layer readout difference
does not capture.

**Already in this repo.** `p2_eigenspectra/head_ablation.py`,
`p2b_imaginary/imaginary_ablation.py`, `core/intervention.py`,
`p6_subspace/dissociation.py` (P6-DD1/DD2 — zero the imaginary channel and ask
whether induction drops while clusters survive, and the converse).

**Differential prediction (candidate `PB-ABL1`).** Because ablation removes a
term from a *field* rather than a value from a sum, the trajectory divergence
should grow **superlinearly with the number of layers after the ablation
point**, whereas an additive-contribution account predicts a divergence that is
constant in the remaining depth once the ablated contribution is removed.

*Falsifier:* divergence flat in remaining depth, or growing linearly, across
ablation points.

*Confound that must be controlled, or the result is meaningless:* later layers
have more opportunity to diverge for reasons unrelated to the field structure.
The control arm is a **matched random-direction ablation of equal magnitude at
the same layer** — the same design `design-5c.md` already requires for its
force-collapse and force-disperse arms.

---

## 6. Probe / LDA direction

**Standard definition.** Train a linear classifier on activations to predict
some property; accuracy is read as evidence the property is "linearly
represented".

**Particle-paradigm definition.** A **hyperplane in configuration space**. A
probe's accuracy is a statement about *particle separation* along a direction —
about the configuration — and not about the existence of a "feature". The
distinction has teeth: the same probe accuracy is produced by a population that
genuinely separates and by one whose separation is inherited from the readout
frame, and the particle reading says which by asking whether the direction lies
in the subspace the dynamics permit extent along.

**Already in this repo.** `p6_subspace/probe_subspace.py` (P6-R4),
`p6_subspace/eigenspace_degeneracy.py` (P6-R2, the LDA alignment measurement),
`dual_reading`'s `semantic.lda_projection` and `semantic.probe_predicted_label`.

**Differential prediction: superseded by an existing measurement, and it
failed.** The natural prediction here is P6-R2 — that LDA directions align with
the real repulsive subspace $U_\text{neg}$, the directions along which the
population is permitted extent. It has been run. The alignment is 0.887 with the
*imaginary* subspace $U_A$ and 0.067 with $U_\text{neg}$; **0 of 49 layers** show
the predicted direction, and P6-R4's probe accuracies invert the same way
(real-only 0.152, at chance; imaginary-only 0.564 against 0.590 for the full
activation).

Registering a fresh differential prediction for this construct before that
inversion is understood would be fitting the theory to a result already in hand
— the specific thing the pre-registration gate exists to prevent. **The correct
next step is the null construction for P6-R2**, not a new prediction: `0/49`
becomes a p-value only once the independent unit is decided, and
`status-6.md`'s own caveat is that 49 ALBERT layers are not 49 independent
observations.

---

## Summary: what this bridge is worth so far

| construct | differential prediction | state |
|---|---|---|
| Induction head | `PB-IND1` — geometric effect at matched positions, not just logit | draftable; needs a pairwise `dual_reading` field |
| Activation steering | `PB-STEER1` — sign of effective-rank change tracks the $V$-decomposition at matched norm | **run this first**: cheap, no new instrument, incompatible predictions |
| SAE / crosscoder | `PB-SAE1` — dictionary elements concentrate in $U_\text{neg}$ | registered, instrument frozen, existing evidence points the other way |
| Logit / tuned lens | none yet | open gap, recorded rather than filled |
| Ablation / patching | `PB-ABL1` — divergence superlinear in remaining depth | draftable; needs the matched-random control arm |
| Probe / LDA | superseded by P6-R2 | **already falsified**; needs a null construction, not a new prediction |

Four of six yield a differential prediction. One is an open gap. One is
already falsified in the direction that disfavours the particle account — and
that is the entry worth taking most seriously, because a bridge whose only
adjudicated prediction failed is not yet a bridge.

## What happens next with this document

Item C2 turns `PB-IND1`, `PB-STEER1`, `PB-SAE1` and `PB-ABL1` into registry
entries under claim **H-BRIDGE**, with falsifier, instrument, null construction
and relevance — at which point the pre-registration gate applies to them and
their timestamps are *gated* rather than backfilled. That will be the first set
of predictions in this project registered prospectively under the machinery
rather than reconstructed into it.

Nothing here is registered yet. The predictions above are drafts in a design
document; a draft that has not passed `tools/check_registry.py` cannot be
adjudicated, which is the intended state until C2 runs.
