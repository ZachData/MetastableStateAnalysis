# Phase 5c — MATH (study notes)

## 0. What this document is

Companion to `math-5.md` and `math-5b.md`. Phase 5c is the only phase in this project with **no
code directory** — its preliminary findings came from standalone visualization scripts run against
Phase 1 output — and it is simultaneously **the phase whose framing was promoted to organize the
entire transition project.** Both facts are worth holding at once.

Its object is the population every other phase treated as residue:

> The ~40–50% of tokens HDBSCAN never assigns to a cluster.

Phase 5 reconstructed the mechanism behind one cluster end to end; the unclustered tokens were
implicitly "everything else." Phase 5c asks whether that is the wrong frame entirely.

There is little novel *mathematics* here — the phase's contribution is a **reframing plus a
resource-allocation hypothesis**, and most of its instruments are existing quantities recomputed
over a different population mask. So this document is shorter than its siblings, and weighted
toward the argument structure and the measurement design rather than derivations.

---

## 1. The question, stated as two competing stories

The Geshkovski dynamics drive **every** token toward a single collapsed point. GPT-2-large does
not reach total collapse. So what are the tokens that never joined a cluster?

**(a) Overflow.** Tokens the architecture fails to collapse — computationally inert relative to
the clustered population. Under this story, clustering *is* the computation and the remainder is
slack.

**(b) Deliberate individuation.** The place where the network keeps representations distinct
*because collapsing them would cost something on the actual task* — induction, n-gram completion,
position tracking: anything needing **a specific token**, not its cluster's attractor.

The working hypothesis is that **both are true of different tokens**, and that the right
resource-allocation unit is neither "in cluster" nor "out of cluster" but something like a **fixed
dimensionality budget** the network spends on whatever still needs distinguishing.

### 1.1 The motivating anomaly: effective rank does not scale with $d_{\rm model}$

Effective rank plateaus near **~200** across models whose $d_{\rm model}$ spans roughly
**768–1600** (`design-5c.md`'s figures). Blog 1's Fig. 10 states the observation slightly
differently — *"the models like to use ~250 dimensions ... despite the models having various
activation space dimensionalities from 700-1600"* — so the two sources disagree on the plateau
value by ~25%, which is worth reconciling given that the exact number is the anchor for a
quantitative budget hypothesis.

$$
\text{If the network simply used whatever space training gave it, } \mathrm{erank} \propto d_{\rm model}.
\qquad \textbf{It doesn't.}
$$

That is the observation Phase 5c tries to *explain* rather than record as a curiosity, and the
budget hypothesis is the explanation on offer: a roughly model-independent number of directions
are kept individuated, and the rest is spent on collapse.

**A caveat this phase should carry and does not.** `math-1.md` §6.2 shows raw-mode effective rank
degenerates in the near-orthogonal limit to the participation ratio of the *norm distribution
alone* — a sink count with zero directional content. **If the ~200 plateau was measured in raw
mode, the budget hypothesis may be a statement about outlier-norm token counts rather than about
dimensionality.** `norm_participation_ratio` and `effective_rank(mode="normed")` both exist and
`math-1c.md` §5's `rank_panel` computes all four surrogates on the same axes; the plateau should
be re-established on the normed quantity before it carries a hypothesis. This is the single
highest-value check for this phase and it is [R]-cost.

---

## 2. The primary evidence: the attention flip

| condition | unclustered tokens receive | clustered tokens receive |
|---|---|---|
| Random GPT-2 / ALBERT | near parity, or clustered-favoured | — |
| **Trained GPT-2-large** | **~1.6×** layer-average attention | ~0.5× |
| **Trained ALBERT-base** | **>2×** | ~0.5× |

**The sign flip is trained-specific in every model examined.** This is the phase's strongest
result, and it is the primary trained-vs-random evidence rather than the energy-plateau asymmetry
(which is corroborating context).

### 2.1 Why this reshapes Group D's design before Group D is written

`design-5c.md` makes an inference worth internalizing as methodology:

> This changes the null hypothesis Group D should be designed against: **not** "unclustered tokens
> are inert overflow" (a null predicting no attention preference either way) **but a
> moderate-to-large effect.** Group D should be powered accordingly rather than sized for a null
> result.

That is a correct and unusually disciplined move — using a correlational result to set the
*power* of a later causal experiment, rather than to pre-empt its conclusion. Compare
`math-1c.md` §8.6, where a measured null band set P-S1's effect-size floor.

Note the connection to Blog 1's own reading: unclustered tokens absorb ~90% of attention mass on
~50% of tokens by late layers. If clustering discards what is safe to discard, then **attention
mass is being spent precisely on what was *not* discarded**, which is what story (b) predicts and
story (a) does not.

### 2.2 The corrected motivating evidence, kept visible

An earlier draft cited ALBERT-base's random-vs-trained two-timescale ratio (13.44 vs ≤1.25) as
evidence that random weights resist collapse *more* than trained ones. **That is a misreading.**
The ratio is *mean plateau width / collapse onset layer* — a relative measure of plateau
*duration*, not of whether or how fast total collapse is reached. Random-weight ALBERT-base
collapses fully and reliably (MaxMass = 1.0000 every seed); the ratio is ALBERT-specific and says
nothing about GPT-2.

The surviving comparison: **GPT-2-large's energy stays close to monotonic under random weights but
is flat-to-slightly-decreasing across layers 5–30 under trained weights** — training actively
pushing against the architecture's default, not a property already present pre-training.

The correction is **preserved in the document rather than removed**, because the reasoning behind
it constrains how Groups C and D should be read. That is the right disposition, and the same one
`math-2b.md` §3 takes with the withdrawn identity.

---

## 3. Why the unit of selection has to be new

Phase 5's `select_cluster.py` scores **cluster trajectories**. Phase 5c's unit is a **token**, and
persistence means *"stayed unclustered,"* not *"stayed in the same cluster."*

`cluster_tracking.py` cannot supply this: it matches and chains labels $\ne -1$ across layers
(`math-1.md` §9.2) — the noise label is **excluded from the Jaccard matching by construction**, so
"this token has been noise for $N$ consecutive layers" is not expressible in it. Hence
`noise_tracking.py` as new primitive work rather than a wrapper.

### 3.1 Five criteria, mirroring Phase 5's structure with one addition

Persistence, attention-received rank, content-word status, matched-control availability, and —
new — **cross-prompt positional consistency.** The last exists to distinguish:

- a **structurally fixed** unclustered position (an *architectural role* — e.g. position 0, the
  attention sink of `math-1.md` §2.5), from
- a **content-flipping** one (*tracks specific information*).

**This distinction changes how every downstream result should be read**, so it is built into
selection rather than added as post-hoc interpretation. It is also a direct hedge against the
failure mode `math-1.md` §6.2 describes: if the selected "unclustered token" is always position 0,
Phase 5c is studying the sink, not the population.

---

## 4. Group C: the rank-budget test, pre-registered to fail cleanly

`rank_decomposition.py` computes effective rank **separately over the clustered and unclustered
populations** across models of differing $d_{\rm model}$. Three outcomes are registered in advance
as legible, not just the favoured one:

| outcome | reading |
|---|---|
| unclustered rank scales with $d_{\rm model}$, clustered does not | **population-mixture story** — the plateau is an average over two differently-behaved populations |
| **neither** scales | **a genuinely fixed, model-independent budget** — the strongest form of the hypothesis |
| both scale proportionally | **the plateau is a superposition artifact** and the motivating observation does not survive decomposition |

Registering all three is what makes this a test rather than a demonstration — and the third
outcome would retract the phase's own motivating anomaly, which is exactly the discipline
`PREDICTIONS.md` asks of every phase.

**A measurement caution.** Effective rank is not additive over a partition: for populations
$P_1, P_2$ with combined cloud $P$, $\mathrm{erank}(P) \ne \mathrm{erank}(P_1) +
\mathrm{erank}(P_2)$ in general, and it is not even monotone in an obvious way, since the spectral
entropy of a union depends on how the two subspaces overlap. So "does the *budget* split between
populations" needs a stated decomposition, not just two numbers — the natural one being to compare
$\mathrm{erank}(P_1)$ and $\mathrm{erank}(P_2)$ against $\mathrm{erank}(P)$ *and* against the
principal angles between the two populations' leading subspaces. Two subpopulations occupying the
**same** 200 directions and two occupying **disjoint** 100-dimensional subspaces both produce
$\mathrm{erank}(P) \approx 200$ and would be indistinguishable from the marginal ranks alone.

---

## 5. Group D: force-collapse and force-disperse as symmetric questions

The design runs **both directions**, with matched controls for each:

| arm | question | control |
|---|---|---|
| **force-collapse** | does the network *need* this token individuated? | collapse-control, random-displacement-control |
| **force-disperse** | does the network *rely on* this token being collapsed? | disperse-control |

> Collapsing only unclustered tokens and calling it a day would leave the "clustering discards what
> is safe to discard" half of the hypothesis untested.

Readout is **next-token cross-entropy delta and KL divergence**, run on **both the trained model
and its random-weight twin per arm** — so every causal result carries the same trained-vs-random
contrast that motivated the phase.

This is the best-designed causal experiment specified anywhere in the project: symmetric arms,
matched controls per arm, and the motivating contrast preserved inside the intervention rather than
assumed from outside it. It has not been run.

### 5.1 The dependency chain, and where it actually breaks

All three blockers have been *partially* resolved by `core/intervention.py`, and the residue is
instructive:

1. **`causal_tests.py` has not been rewired.** `run_model_with_hook` is exactly the model-agnostic
   replacement needed (the old GPT-2 branch of `_run_albert_with_hook` **never called `hook_fn`**),
   and Phase 5's copy *has* been migrated with per-architecture dispatch (`math-5.md` §7.1) — but
   Phase 5c's has not.
2. **No model in the registry has an LM head.** `MODEL_CONFIGS` / `pythia_registry.py` load bare
   model classes with no `.logits` output, **so even a correctly-wired call returns
   `logits = None`.** Since Group D's entire readout is cross-entropy and KL over next-token
   distributions, this is the hard blocker, not the hook wiring. `core/lm_loading.load_causal_lm`
   now closes it.
3. **The Group D module itself does not exist.** Every primitive is built — runner, KL/loss,
   population selector, dual-reading — and *assembling them into the experiment is separate,
   phase-level work.* Recording that explicitly, rather than treating "the primitives exist" as
   "the phase is nearly done," is a distinction most project plans blur.

---

## 6. Two things deliberately not in this phase

**Group B (routing/flow) is descoped, not dropped.** GPT-2's attention heads being universally
content-independent — fixed at training, identical across inputs — is flagged as *possibly the
project's largest single finding*, and asking what that fixed routing computes (induction, n-gram
completion, skip-trigrams) is a research thread on its own. Bundling it here would force one phase
to carry two questions to publication. It is kept as a candidate "Phase 7" with a concrete starting
point (Phase 1's per-head Fiedler tables, Phase 6's `induction_ov.py` / `head_classify.py`).

> Worth flagging: that finding rests on Phase 1's per-head Fiedler analysis, which `math-1.md` §14
> shows was **mislabeled, vacuously thresholded, and never persisted** (defect D2 — every head
> classified identically by construction). **"Attention heads are universally content-independent"
> may be the same artifact wearing a different name**, and it should be re-established on the
> corrected quantity before a phase is built on it.

**Constraints carried in, not re-derived:** cone-collapse is universal (Phase 1b), so nothing here
should be read as testing antipodal separation — as recorded, it never happens, at any layer, in any
model tested. **Read that constraint as directional, not quantitative:** `math-1b.md` §9 shows the
cone-collapse *magnitude* does not survive review — unquantified against any null, stated in a
margin that is arguably not comparable across the prompts it was averaged over, with the ALBERT row
empty rather than inconclusive. The direction (no antipodal separation) is safe to carry; "100%,
every layer" is not yet a calibrated number. And final-layer LM-head contamination in gpt2-small/medium is excluded from any selection
or readout treating late-layer geometry as dynamics.

---

## 7. The reframe: this phase's frame becomes the project's

The transition plan's opening section promotes Phase 5c's central object to the organizing unit for
the whole project:

> **The object of study going forward is every particle and how it evolves. Clustering is one
> annotation on a particle, not the unit of analysis.**

Two concrete consequences:

1. **The per-particle-record schema generalizes `noise_tracking.py`.** This phase needed a
   primitive `cluster_tracking.py` lacks — *"this specific token has been noise for $N$ consecutive
   layers."* The canonical artifact shape — a long table keyed by
   `(prompt, token position, layer, step)` with columns for cluster label, population tag,
   V-projection, and dual-reading output — is that same idea generalized. Once it exists, **Group
   A's persistence tracking and the population selector are both just filters on the same table**
   (`math-5.md` §8), and this phase needs no bespoke tracking module at all. Blocker 1 is therefore
   *relocated rather than closed*: nothing populates that table with real data yet.
2. **The two-baseline policy is a direct extension of this phase's methodology.** The attention-flip
   sign reversal — present in trained models, absent in random twins — is exactly the contrast the
   norm-matched `pythia-1.4b-random` construction is designed to preserve on the new architecture,
   and it is why that baseline must be norm-matched rather than fresh-init (`math-1.md` §13.2).
   This phase's evidence is now part of the project-level falsification record, cited under
   `PREDICTIONS.md` claim (a), *"collapse-resistance is learned, not initial."*

---

## 8. Status, honestly

| Group | State |
|---|---|
| A (persistence structure) | Not started; gated on the particle table |
| B (routing/flow) | **Descoped** to a candidate Phase 7 |
| C (rank budget) | Not started; **gated on nothing new** — reuses existing effective-rank code across populations |
| D (causal) | Not started; module unwritten, primitives now available |

Preliminary correlational findings, all from visualization scripts against Phase 1 output:
the **attention flip** (strongest); **punctuation fraction** is *not* trained-specific (same ratio
under random weights — reflects embedding-space geometry, not learned behaviour); a **trained-only
negative-IP mode** at density $\sim10^{-4}$ (rare but real token pairs — consistent with
cone-collapse universality rather than contradicting it, since a cone can have a thin antipodal
tail without the *whole* cloud leaving the half-space); the **within/between/noise IP
decomposition** showing the energy plateau is carried entirely by within-cluster pairs; and
**cluster cohesion direction** as a coin flip across depth. `attractor_alignment.py` has been run
on synthetic data only.

---

## 9. Open questions

Tracked: `noise_tracking.py` / the particle table; the Group D rewiring and module; no code
directory.

Surfaced by writing this document:

1. **Re-establish the rank plateau on `effective_rank(mode="normed")`, and reconcile ~200 against
   Blog 1's ~250, before building on either** (§1.1). The entire budget hypothesis rests on a number that, if measured in raw mode, may
   be a sink count. `rank_panel` computes every surrogate at once and this is report-only cost.
   **This is the cheapest way to either strengthen or dissolve the phase's motivating anomaly.**

2. **Group C needs a subspace-overlap statistic, not just two marginal ranks** (§4). Effective rank
   is not additive over a partition, and two populations occupying the *same* directions versus
   *disjoint* ones are indistinguishable from marginals alone. Principal angles between the
   clustered and unclustered leading subspaces would settle it, cost one SVD pair per layer, and
   directly test the budget framing: **a shared budget predicts high overlap; independent
   allocation predicts low.** As specified, Group C cannot distinguish its own outcomes 1 and 2
   from a case where both populations simply share one 200-dimensional subspace.

3. **`border_vs_noise` already gives the unclustered population a geometric definition, and this
   phase does not use it.** `math-1b.md` §5.4 crosses per-token distance from the Fiedler boundary
   against HDBSCAN's noise labels as a rank AUC — *is the unclustered population the boundary
   population?* That is a candidate **definition** of "unclustered" that does not depend on
   HDBSCAN's hyperparameters at all, which matters because every result here is currently
   conditional on `min_cluster_size = 2` and a cosine metric. **If the AUC is high, Phase 5c gains a
   clusterer-independent population definition; if low, the two notions of "outsider" are different
   objects and that is itself a finding.** Cross-referenced from Phase 1b's handoff notes; not
   picked up here.

4. **The attention-flip result should be checked against the sink.** Position 0 on NeoX carries a
   norm one to two orders above every other token and is a plausible permanent member of the
   unclustered set. If unclustered tokens receive 1.6–2× layer-average attention, **the obvious
   confound is that a small number of sink tokens dominate that average.** The fix is the same
   `pos0_policy` ledger field the frame machinery already provides: report the flip with position 0
   included and excluded. Given that the flip is the phase's headline and the motivating evidence
   for a project-wide reframe, it deserves the one-line control.

5. **"Content-independent attention heads" may be defect D2 in disguise** (§6). Before a Phase 7 is
   scoped around it, re-establish it on raw $\lambda_2$ with re-derived, baseline-scaled thresholds
   and per-length reporting.

6. **The two stories in §1 make a directly testable differential prediction that is not registered.**
   Story (a) — inert overflow — predicts that force-collapsing unclustered tokens costs **little**
   next-token CE. Story (b) — deliberate individuation — predicts it costs **a lot**. That is
   precisely Group D's force-collapse arm, but the *quantitative* form is unregistered: how much CE
   delta counts as "a lot"? The natural calibration already exists in the design — **the
   random-weight twin arm** — so the registerable prediction is the *ratio* of trained to random CE
   delta, with the same sign-flip logic the attention result uses. Registering it as a ratio makes
   it robust to scale differences between models, exactly as `math-1c.md` §8.5's $Q_k$ ratio is.
