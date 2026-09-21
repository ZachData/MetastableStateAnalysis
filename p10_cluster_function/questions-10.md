<!-- p10_cluster_function/questions-10.md -->
# Phase 10 — QUESTIONS (hypotheses, not a design; nothing frozen, nothing registered)

**Written 2026-09-20, immediately after five papers were read as primary text**
(`lit-10.md` §11–§15, `math-10.md` §7). This file is the *"what are we not
asking?"* pass that read licensed. **`handoff-10.md` is the ordered plan;
this file is the reasoning behind it.**

**Status of everything here: conjecture.** No `P-*` id, no entry in
`claims/registry.json`, nothing quotable as a result. Where a statement is
`[R]` from a paper it is marked; where it is this project's own measurement it
is cited; **everything else is a hypothesis someone thought of on 2026-09-20
and it should be read with that discount.**

---

## 0. The premise that motivated this, and why it has to be restated

The question that opened Phase 10 (`notes-10.md` §0) is the user's:

> A cluster might be **trash collection** — a compressed cleanup. One
> representation standing for one thing, with a lot of particles put into it in
> order to keep them stationary.

The strongest evidence for it was Blog 1 / Phase 5c's **attention flip**:
unclustered tokens receive ~1.6× layer-average attention, clustered ~0.5×,
with the sign flipped under random weights.

> **Row A0 removed ~94 % of that effect** (`status-10.md` §1.1). Corrected for
> the causal mask, the 0.592 gap is **0.036**; at initialisation it is **0.004**
> against a raw 0.26 — *entirely* mask. A learned residual appears at step
> ~2000–4000 and grows to 0.172 by step143000.

**This does not refute trash collection. It changes what has to be explained.**
The object is no longer a large static routing asymmetry; it is a **small
learned residual with an onset**, and every downstream experiment should be
aimed at the onset rather than at the sweep mean.

### 0.1 And the theory predicts the opposite sign, which is the useful part

`2411.04990` §5.1 **[R]**: strong Rényi centres *"serve as primary attractors
for subsequent tokens."* Attractors **receive** attention. Trash collection says
clustered tokens **do not**. Same population, opposite predictions.

The reconciliation is available and has never been stated here:

> **H-ANCHOR/BALLAST.** A cluster is **one attended anchor plus many ignored
> members**. The mean over the cluster is low — which is what `noise_importance_proxy`
> measures, and which looks like trash collection — while the anchor is a sink,
> which is what the theory predicts.

**HDBSCAN gives membership and never centrality**, so this distinction has been
structurally invisible to every measurement in this project. `notes-10.md` §8's
F13 supplies centrality for free, and the cross with received attention is the
real form of the trash-collection question. **This is the single highest-value
free experiment identified in this pass** and `handoff-10.md` Stage 2 is it.

---

## 1. The axis the project owns and has never read

`2411.04990` §A.1 **[R]**: *"the time parameter in our dynamics corresponds to
network depth."*

So there are two clocks. **The algorithmic drive runs in DEPTH. SGD runs in
TRAINING STEPS.** The sweep is a 19 × 24 grid of the two, and **every result so
far collapses one axis or the other** — A0 reports a sweep mean then splits by
checkpoint; F1 and F12 report per-checkpoint curves. Nothing reads the joint
object, and the user's question is about the joint object.

> **H-WARP** — training acts on the clustering dynamics as a **monotone
> reparametrisation of the depth clock**. It changes how fast collapse runs, not
> what collapses with what.
>
> **H-STRUCT** — training changes which particles park where, beyond any
> rescaling of the clock.

**These separate by curve collapse.** Fit one monotone warp per checkpoint and
ask whether all 18 distinct depth-profiles fall onto a single master curve.

- If they collapse: *training does not teach the model what to cluster; it sets
  the clock.* That is a large, clean, and falsifiable headline.
- If they do not: **the residual after the best warp is the learned structure,
  isolated from the architectural drive** — which is the object the whole phase
  is trying to name.

**The instrument exists and was built to throw this away.** `T_eff` is Phase
1c's Distance #3, titled *"the depth confound"*; `β_eff` is Distance #4.
`math-1.md` §15 item 3 calls `T_eff` *"the highest-value unrun quantity, at
report-only cost — every input is already on disk."* **The confound is the
answer.** `2411.04990` §5 supplies the mechanism that would make H-WARP true:
metastable configurations hold until `t = exp(Ω(√β))`, so a checkpoint's
position on the collapse trajectory is set by `β_eff` and depth jointly.

**Why this project and not someone else's.** The theory has no training axis.
`2501.10573` has no training axis. `2601.02932` has no training axis. A
developmental read of a depth dynamics is the thing this repository uniquely
holds.

---

## 2. H-THERMOSTAT: a mechanism for Blog 1's headline

Blog 1's result — *trained transformers resist the collapse their architecture
predicts* — is phenomenological. `PUBLICATION_IDEAS.md` says what it needs is a
mechanism.

In `ẋ_k = (1/Z_k) Σ_{j≤k} e^{β⟨Qx_k,Kx_j⟩} V x_j`, a token that receives
enormous attention from everyone contributes a large term to **every other
token's `Z_k`**, dividing down every *other* pairwise interaction.

> **H-THERMOSTAT.** The attention sink is how a trained transformer resists
> collapse: it absorbs mass into the normaliser, lowering the effective pairwise
> coupling among the remaining tokens, and slowing the clustering dynamics. An
> attention sink is a **denominator attack on coupling strength.**

Three reasons it is worth the time:

1. **It is free today.** `attentions.npz` is in all 152 directories. Take
   `a_{i,0}`, renormalise each row over `j ≠ sink`, measure what is left. If the
   sink's mass share predicts the energy-monotonicity violation rate per layer
   per checkpoint, that is the mechanism.
2. **It unifies three threads.** Phase 9 wants to intervene on the metric;
   `math-1.md` §1A.6 says `Z` **is** the metric; `math-10.md` §7.5 now shows
   `1/Z_k` is literally the prefactor on particle `k`'s gradient **[R]**. So
   **the model has already learned a metric intervention, and the sink is it.**
   Phase 9 stops being "let us try a metric patch" and becomes "let us do
   deliberately what training discovered."
3. **It is distinct from the sink literature.** `2510.06477` proves sinks ⇒
   compression. This claims sinks ⇒ *resisted* compression among the non-sink
   population. Different claim, and testable against the same artifacts.

**The index trap, already solved and easy to re-fall into.** `math-10.md` §2
established the sink is the **minimum** of its *own* row sum `Z_0`. The
thermostat is a **column** effect — the sink's contribution to *everyone else's*
row sum. Same quantity, opposite index. The decomposition is
`Z_k = z_k^{sink} + z_k^{rest}`, and it is one line from the Gram.

**What would falsify it:** sink mass share uncorrelated with violation rate
across the 18 checkpoints; or the correlation present at every checkpoint
including step 0, which would make it architectural rather than learned.

---

## 3. The weight-space → activation-space bridge, which is the user's question made quantitative

`2411.04990` §5 **[R]** conjectures the count law runs in `d₁`, the dimension of
`L` — the top eigenspace of `V`. **`L` is a weights-only object**, and Phase 2's
`sym_*` / `schur_*` projectors already identify it, on disk for all 19
checkpoints.

> **Predict the activation-space cluster count from the weight-space OV
> spectrum, per checkpoint**, via Lemma C.1 with `d₁ = dim L`.

`MATH_SPECTRAL_OT.md` §2.4.6 is the standing negative result that **no
weights-only spectral quantity computed by this project identifies the copier**.
This would be the first weights → activation spectral prediction that works —
and if it fails, the failure localises where SGD does something the architecture
does not. **Either outcome answers "how does the algorithmic drive relate to the
structure SGD imposes on weight space".**

---

## 4. The loss, which this project has never touched

`2501.10573` **[R]** finds `ρ(log ID, surprisal) ≈ 0.6–0.8` across four models,
three estimators, `p < 0.01`. **This repository has never connected geometry to
loss at all**, and the user's question is about minimising loss.

> **Is a clustered token a low-surprisal token?** Trash collection predicts yes
> — you park what you have already resolved.

- **Cost:** one forward pass per prompt per checkpoint, no backward. **No logits
  are on disk** (checked: `activations.npz`, `attentions.npz`, `tokens.txt`,
  `clusters.npz`, `hdbscan_labels.json`, and the per-layer JSONs — no logits).
- **The developmental version is the prize:** does the clustered/unclustered
  surprisal gap open at the **same step** as A0's learned residual (~2000–4000)?
  Two independent onsets co-locating would be strong.
- **Mandatory control:** position. Clustered tokens sit later
  (`clustered_position_bias` +0.051) and later tokens are easier. Without a
  position covariate this measures the mask again.

---

## 5. Three methodological upgrades the papers hand us

### 5.1 Replace ARI with a variance decomposition

§3.51.4's reproducibility floor bites hardest on **partition-comparison**
statistics — which is exactly what F4 and F5 are built on. But the trash
hypothesis has a form that needs no second partition:

> **Within-cluster functional variance should be near zero for trash and high
> for a computed category.**

A variance decomposition is far more robust to label noise than an ARI, and it
is available on the phase's central test at no cost. **This is the single
cheapest rigour improvement identified.**

### 5.2 Cluster in the metric the model reads, not only the one HDBSCAN defaults to

`2411.04990` §5.1 **[R]**: the separation *"extends naturally to distances
induced by `⟨Qx, Ky⟩`"*. So there are three metrics on the token cloud —
**geodesic** (what HDBSCAN uses), **attention** (what the model reads,
`core/ln_frame.py`'s Gram), **functional** (what the output cares about, the
J-lens). Clustering only in the first is a choice nobody here has defended.

### 5.3 Add the input-side control the project does not have

Trained-vs-random changes the **model**. `2501.10573` §4's graded block shuffle
(`b_S = N/4^S`, six levels, unigram-preserving, BLEU/BERTScore-calibrated)
changes the **input**. Blog 1's resistance headline has never been tested
against *"does the input have to be language?"* — and the `repeated_tokens`
degenerate control is the extreme version of exactly this, which `math-1.md`
§13.2 calls *"the cleanest single result in the sweep."* The graded version is
the obvious extension.

---

## 6. Forgetting: what the read actually licenses

### 6.1 Two things the unlearning literature does not have

**A capacity bound.** Lemma C.1 **[R]** gives the number of available parking
spaces as `1/σ^{d−1}(B_δ)` with `δ = cβ^{−1/2}`. A metric patch with budget `ε`
therefore changes the number of basins by a **computable** amount. *"How much
can you forget with budget ε"* acquires a closed-form upper bound. **No one in
unlearning has a capacity theorem**, and this is the differential prediction
`notes-10.md` §9 says needs both phases and neither can make alone.

**A mechanism for the field's most robust finding.** `2505.16831` **[R]**:
unlearning is reversible because information is *suppressed, not erased*. **The
parking account predicts exactly that and says why** — parking is a release
operation (`notes-9.md` §2), the particle is still present, it is merely
stationary. Having a mechanism for someone else's empirical result is a stronger
position than proposing another intervention.

### 6.2 The version that would be irreversible

`2505.16831` Table 1: *irreversible, non-catastrophic* forgetting is the ideal
and they observe it essentially never.

> **Evict and fill.** Park the target, then **occupy** the space so it cannot
> come back. `2411.04990` Theorem 5.2 — freeze the centres and everything
> converges to them — is the occupation operator, and it is the only experiment
> in this phase whose predicted outcome is a theorem.

### 6.3 A better lever than `γ`, and Phase 9 should probably switch

`plan-9.md`'s lever is LayerNorm's `γ`. But `2411.04990` §2 **[R]** states the
RMSNorm diagonal is absorbable into `K`, `Q` **and `V`** — and Pythia's QKV is
fused off one LayerNorm, so **a γ-patch is not a temperature knob; it moves the
value path too.**

> **The clean lever is a patch on the attention logits themselves.** That is
> `β` exactly — the theory's own parameter — weight-preserving, exactly
> invertible, and it leaves `V` alone.

And it has a **capacity reading**: `δ = cβ^{−1/2}`, so raising `β` shrinks `δ`
and by Lemma C.1 *increases* the number of parking spaces; lowering it merges
everything. **Temperature is a capacity knob with a formula.** I would make the
logit-temperature patch Phase 9's primary instrument and `γ` the secondary.

---

## 7. What would make all of this more rigorous, ranked

1. **Register F14 before looking.** F0 was the phase's best tier-2/3 candidate
   and running it exploratory capped it at tier 1 (`status-10.md`, decision
   2026-09-20). With **39 registrations and zero adjudications**, F14 (observed
   count vs Lemma C.1 — a published prediction, an exact null, one free
   parameter) is the second chance. Do not burn it the same way.
2. **The power problem is a prompt-count problem, and it is now the FIRST
   ACTION** (`handoff-10.md` **Stage 0**). **Eight prompts.** `2501.10573` used
   2 244. Layer-units inside one forward pass are not independent samples of
   anything; the `average` e-merger is valid under arbitrary dependence and
   correspondingly low-powered.

   **The work is already licensed.** `core/prompts.py` carries battery **v2, 21
   prompts, hash `06790b90dcfe`**, extended under a rule committed ahead of the
   text (`PROJECT.md` §3.42). Phase 1's sweep used 8; **12 have never been
   through Phase 1**, and because they were chosen blind under a written rule,
   **running them is not a new selection decision.** 8 → 20 usable prompts,
   2.5× the exchangeable unit, and the enlarged battery **can carry a registered
   prediction** where the current one cannot.

   **Budget, measured:** 12 × 19 = 228 directories at 325 MB = **74 GB** against
   164 GB free — or **~40 GB** once the verified `plateau_attentions.npz` /
   `attentions.npz` duplication (148 MB per directory, ≈22 GB across the
   existing sweep) stops being written twice. **Attention scales n², so more
   short prompts beat fewer long ones** for this project's purpose, which is
   independent units rather than per-prompt intrinsic dimension.
3. **Make the reproducibility floor a standard error, not a caveat.** Re-cluster
   `R` times per directory and report every partition-derived number with a
   re-measurement error bar. `backfill_hdbscan.py` does 152 directories in 69 s.
4. **Own position as a covariate once, in `core/`.** A0 found 94 % mask, F12
   found 99.5 % position, F0 needed a bespoke restricted null. Three rows have
   invented three position corrections; it is a shared abstraction.
5. **Never quote a sweep mean again without its checkpoint split.** A0's mean
   hid the finding and `status-10.md` §1.1 says so. This is now a pattern, not
   an incident.

---

## 8. What this file does not claim

- **Nothing here is registered, adjudicated, or quotable.** Tier 1 at best,
  conjecture at worst.
- **None of the papers' results is a theorem about Pythia.** `2411.04990` ties
  weights across layers, omits the MLP entirely (*"a significant open
  challenge"*), and proves its meta-stability results at `V = I`, `Q = K = I`,
  `d = 2`.
- **H-THERMOSTAT, H-WARP and H-ANCHOR/BALLAST were invented on 2026-09-20** and
  have no literature check behind them. `CLAUDE.md` trigger 2 applies before any
  of them reaches `claims/registry.json`.
- **This file was written by one session with a partial read of a 7 000-line
  `PROJECT.md`.** Some fraction of it may already be recorded somewhere not
  reached. Anything that turns out to be a duplicate should be deleted here
  rather than kept as a second account — see §3.52.5 for why.
