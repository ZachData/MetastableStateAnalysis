<!-- p10_cluster_function/lit-10.md -->
# Phase 10 — LITERATURE (partial scan, 2026-09-20)

**Status: THREE passes, and trigger 1 is now discharged for the five papers
that mattered.** The first pass (2026-09-20, §1) settled one factual question
that arrived with the phase — *does a fitted Jacobian lens exist for a Pythia
model, or must one be trained?* The second pass (§5–§9, same day) worked the
topic list in `notes-10.md` §12 and `plan-9.md` §12 under a blocked egress, so
everything in it is `[S]`. **The third pass (§11–§15, same day) read five papers
as primary text** — the user supplied the PDFs — and it **corrects the second
pass in three places**. §15 is the remaining queue.

**Read §11 before anything else here.** It is the paper two phases were built
on and neither had read, and it changes F0, the count law, and the
gradient-flow hazard.

**It has already paid for itself twice.** §5 finds the Rényi-parking prediction
is not what `lit-1.md` says it is, which would have wrecked an F0 registration;
§6 finds Phase 9's unlearning novelty claim is narrower than `notes-9.md` §8
states. Both are `CLAUDE.md` trigger-2 catches arriving at trigger-1 time.

## The egress result, which is itself worth recording

Ranked by what it unblocks, because `docs/LITERATURE.md` records the opposite
constraint and it has shaped three scans.

| host | reachable from a cloud session? |
|---|---|
| `github.com`, `raw.githubusercontent.com` | **YES** — full README and source text |
| `arxiv.org` | no — `EGRESS_BLOCKED` |
| `transformer-circuits.pub` | no — `EGRESS_BLOCKED` |
| `huggingface.co` | no — `EGRESS_BLOCKED` |
| `neuronpedia.org` | no — `EGRESS_BLOCKED` |
| web search (titles, snippets, summaries) | yes |

`docs/LITERATURE.md` item 18 already guessed this — *"GitHub may be reachable
where arXiv is not — this is the fastest route"* — and it is correct. **A
companion-code repository is a readable primary source from here even when the
paper is not.** That is a route every future scan should try before settling for
`[S]` marks: the README of `anthropics/jacobian-lens` gave the lens's defining
equation, its fitting corpus size and its API, none of which a search summary
carried.

Marks below follow the project convention: **[R]** read as primary text,
**[S]** search-engine summary only, **[N]** title/id only.

---

## 1. The Jacobian lens — **[R]** on the code, **[S]** on the paper

**Paper.** *Verbalizable Representations Form a Global Workspace in Language
Models*, Gurnee et al., Transformer Circuits, published **2026-07-06**;
`transformer-circuits.pub/2026/workspace/index.html`, arXiv **2607.15495**
**[S]**. This is **the same paper `p2_eigenspectra/lens_band.py` already cites**
(as "Gurnee et al. 2026, §4.1 Fig. 28") and the same one
`archive/p5_single_mstate_analysis/status-5.md`'s 2026-07-19 note is built on.
The project has been citing it for two months without an arXiv id; **the id
should be added to both files.**

**Code.** `github.com/anthropics/jacobian-lens` **[R]** — the `jlens` package,
**Apache 2.0**, described as a reference implementation and not actively
maintained. Read directly:

```
    lens_l(h) = unembed( J_l @ h ),    J_l = E[ ∂h_final / ∂h_l ]
```

The expectation is over **prompts, source positions, and all target positions**;
the implementation sums cotangents over target positions and then averages over
source positions. API: `jlens.from_hf`, `jlens.fit`, `JacobianLens.from_pretrained`,
`.apply`, `.save`, `.merge` (for combining lenses fitted on disjoint slices).

**Cost, from the README [R]:** the paper's lenses use **1000 sequences of 128
tokens** from a pretraining-like corpus, quality **saturates quickly (§9.3) and
~100 prompts is usable**, and fitting time is *"dominated by the model's own
backward pass"* — not optimised, but parallelisable by slice-and-merge.

**Artifacts [S], not verified directly — `huggingface.co` is blocked.** Search
summaries report pre-fitted lenses for **38 open models** at
`neuronpedia/jacobian-lens`, **including `pythia-70m-deduped`**, with an
interactive explorer at `neuronpedia.org/jlens`, lenses stored as `.pt`, one
`[d_model, d_model]` matrix per layer, fp16. **Every one of those statements is
`[S]` and must be verified on the research machine before anything is built on
it.** The check is one `huggingface_hub` listing and costs nothing.

### 1.1 What this does to two of this project's recorded decisions

1. **`p2_eigenspectra/lens_band.py`'s stated deviation is now a choice rather
   than a constraint.** Its header says it uses the logit lens *"because no
   averaged Jacobian has been trained for these checkpoints and training one is
   deliberately out of scope (see `CHANGES_jlens_adjacent.md`)"*. Fitting one is
   now a documented, licensed, ~100-prompt procedure. The header's consequence
   note — that the detected band **onset is an upper bound** because the logit
   lens is noisier early — is exactly the error a fitted lens removes, and early
   layers are where Phase 1's cluster story starts.
2. **`archive/p5_single_mstate_analysis/status-5.md`'s blocker 4 has a third
   option that is now the cheapest.** That note lays out three routes for Group
   E (decoding what a mid-layer cluster centroid represents): stay with the
   frozen head, state the caveat, or train the affine lens and validate it
   against the skip-to-output pathology. **The J-lens is the route the note's own
   source recommends and the one it could not cost.** The pathology it warns
   about is specific to correlationally-trained affine translators; the averaged
   Jacobian is not one.

### 1.2 The caveat that decides how it can be used

**`pythia-70m-deduped` is not `pythia-70m`.** They are different training runs on
different corpora. This repository's ladder holds `pythia-70m` (19 revisions,
step0 → step143000, `PROJECT.md` §1) and every registered 70m decision names
that model. A lens fitted to deduped activations is **not** licensed on
`pythia-70m` without a check, and the published lens presumably corresponds to a
single final revision, so **it carries no checkpoint axis at all** — which is the
axis this project exists to study. Two honest routes, and they are different
phases of work:

- **Borrow**: add `pythia-70m-deduped` as a separate, labelled model and use the
  published lens on it. Cheap, immediate, and a different model from the ladder.
- **Fit**: run `jlens.fit` on `pythia-70m` per checkpoint. 6 layers, `d = 512`, so
  each `J_l` is 512×512 — about 0.5 MB fp16 per layer, ~3 MB per checkpoint.
  ~100 prompts per fit. **This is the option that gives a developmental J-space**,
  and nobody appears to have one.

## 2. The causal-mask theory — **[S]**, and it bears on Phase 9

*Clustering in Causal Attention Masking*, arXiv **2411.04990** (Karagodin,
Polyanskiy, Rigollet, NeurIPS 2024) **[S]**. Already in `lit-1.md` §1.3 and
`docs/LITERATURE.md` row 6; re-surfaced here because two of its three claims
are Phase 10 and Phase 9 material rather than Phase 1 material.

- **Claim 1: the masked system cannot be interpreted as a mean-field gradient
  flow.** `docs/LITERATURE.md` already asks whether this voids Phase 2d's
  framing. **It bears on Phase 9 the same way and nobody has said so** — see
  `notes-10.md` §10.1.
- **Claim 3: metastable states connect to the Rényi parking problem**, which
  predicts a **number of clusters as a function of `n`** (parking density
  constant ≈ 0.7476). This is the quantitative form of Phase 10's central
  hypothesis and it is the project's best-rated cheap experiment
  (`docs/LITERATURE.md` §6 item 1, `lit-1.md` §4 item 1: *"Do this one first."*).

**Still `[S]`.** The exact form of the correspondence — what the "cells" are,
what plays the role of car length, whether the count is per layer or asymptotic —
is precisely what a Phase 10 prediction would need and precisely what a search
summary does not give. `lit-1.md` §5 already has it in the verification queue.

## 3. Leads, unread

- **`2609.01924`** *Looped Transformers under the Jacobian Lens: Does the Global
  Workspace Survive Recurrence?* **[N]** — a J-lens follow-up using a
  virtual-unrolling adapter. Relevant only if Phase 10 ever wants a depth-recurrent
  comparison; recorded so it is not rediscovered.
- **`2510.06477`** *Attention Sinks and Compression Valleys are Two Sides of the
  Same Coin* **[S]**, already in `lit-1.md` — the nearest prior art to the
  "compressed cleanup" reading, and it **proves** massive activations necessarily
  produce representational compression. Phase 10's novelty has to be located
  against it, not beside it.
- **`2509.23024`** *Tracing the Representation Geometry…* **[S]**, already in
  `lit-1.md` — its "compression-seeking consolidation" phase is the developmental
  version of the same idea, on Pythia.

## 4. What is NOT scanned, and must be before `design-10.md`

`notes-10.md` §12 carries the list. The two that would change constructions
rather than citations: whether anyone has clustered tokens **in a lens basis**
rather than in the residual basis, and whether the parking correspondence has
already been checked empirically by anyone.

---

**Sources for §1, recorded because they are the primary text this scan actually
read:** `github.com/anthropics/jacobian-lens` (README, `[R]`);
`raw.githubusercontent.com/anthropics/jacobian-lens/main/README.md` (`[R]`).
Everything attributed to `neuronpedia/jacobian-lens`, to the paper, or to
`arxiv.org/abs/2607.15495` is `[S]` — the hosts are blocked from here.

---

# Second pass (2026-09-20) — the topic list worked

Same egress constraints as §0. Everything below is **[S]** unless marked.

## 5. The Rényi-parking prediction is not a law in `n`, and `lit-1.md` has it wrong

> **SUPERSEDED 2026-09-20 by a full reading: `docs/readings/2411.04990.md`.**
> Every `[S]` inference below is confirmed, and the paper adds four things the
> summaries did not carry: Theorem 4.1's limit is **`x₁(0)`**, the first token's
> initial position, for arbitrary `Q, K`; the `d_eff` rescue is **the paper's own
> open conjecture** with `d₁ = dim L`; **Lemma C.1** gives an exact
> distribution-free count that **saturates in `n`**; and §2 remarks that a
> trainable RMSNorm diagonal is **absorbable into `K, Q, V`**, which corrects
> `plan-9.md` §7. Read that file, not this section.

Two independent search summaries agree, and they correct the project's standing
description of this result.

- **The scaling.** *"The frequency of both Rényi and strong Rényi centers is
  predicted to be `Θ(β^((d−1)/2))`, confirming a `β^(1/2)` scaling for `d = 2`."*
- **The mechanism.** *"In causal models, **early tokens in a sequence act as
  'nuclei'** that serve as centers for cluster formation, a process compared to
  the Rényi parking problem, where particles fill up space and prevent others
  from collapsing into them."*
- **The gradient-flow claim, in the paper's own framing.** *"This modification
  translates into an interacting particle system that **cannot be interpreted as
  a mean-field gradient flow**."* Second independent confirmation of
  `plan-9.md` §5.1a's hazard.

`lit-1.md` §4 item 1 says the prediction gives *"a density constant (the Rényi
constant ≈ 0.7476) and hence an expected number of occupied cells as a function
of `n`"`. **Both halves are wrong**: the law is in β and dimension, and 0.7476
does not appear in it. At `d = 1024` the exponent is 511.5 and the prediction is
unusable — the `d ≫ 1` problem `design-1.md` already records for Figure 3.

**`math-10.md` §5 shows the test survives in better form**: fit
`log count ~ a·log β + b·log n`, where `a = (d_eff − 1)/2` is **invariant to β's
undecided unit convention** and measures the effective dimension the clustering
behaves as. And the nuclei reading gives a free, position-indexed anchor test
that needs no β at all.

**`lit-1.md` §4 item 1 and `docs/LITERATURE.md` §6 item 1 should be corrected**,
and the exact statement remains the top item in the verification queue.

## 6. Unlearning by inference-time representation editing is populated

`notes-9.md` §8 and `plan-9.md` §6.1 rest a novelty claim on a metric patch
being *"read-side, inference-time, weight-preserving, and exactly invertible"*.
The nearest prior work is closer than either document assumes.

- **`2605.12765`, *Inference-Time Machine Unlearning via Gated Activation
  Redirection* (GUARD-IT)** — *"training- and gradient-free … operates entirely
  in activation space, executing unlearning as a controlled **geometric
  transformation**"*, applying interventions as *"pure **rotations** in the
  residual stream, preserving the original activation norm."*
- **`2505.16831`, *Unlearning Isn't Deletion*** (ICML 2026 poster) — models
  *"appear to forget while their original behavior is easily restored through
  minimal fine-tuning … information is merely **suppressed** rather than
  genuinely erased"*, with a representation-level evaluation framework.
- **`2605.24614`** *Measuring the Depth of LLM Unlearning via Activation
  Patching*; **`2605.31293`** *Divergence Decoding*; **`2602.02139`** *EvoMU*.
  A survey repo exists: `github.com/chrisliu298/awesome-llm-unlearning` — and
  GitHub is reachable (§0.1), so it is the fastest route to this whole area.

**What this does to Phase 9.** The genus — inference-time, weight-preserving,
geometric unlearning — is occupied. **The differentia survives, narrowly and
specifically: GUARD-IT applies a *rotation*, which is an isometry; a γ-patch is
a *congruence*, which changes the metric.** `plan-9.md` §4.1 already derives
that a γ-patch is `W_QK → Γ'W_QK Γ'`, and an isometry is exactly what it is
*not*. That is a real distinction and `notes-9.md` §8's claim should be rewritten
to make it rather than to claim the category.

**And `2505.16831` is direct support for `notes-9.md` §2**, which says on its own
reasoning that *"collapse is a release operation, not a deletion one."* The field
now says suppression-not-erasure with an evaluation framework attached. **That is
a citation Phase 9 should take rather than a threat**, and it means any Phase 9
unlearning result must be evaluated for reversibility or it is measuring the
thing that paper says is routinely mismeasured.

## 7. Self-repair, and the question §6.2 asks appears genuinely open

`notes-10.md` §6.2 and `plan-9.md` §6.2 propose: *does self-repair engage against
a change of cost geometry the way it engages against an ablation?*

The search returns the expected landscape — `2307.15771` (Hydra), `2607.01940`
(CoAx, already in `docs/LITERATURE.md` row 1), `2310.04625` (Copy Suppression),
backup-head work — and summarises the field as: *"self-repair appears to be a
general property … and it **affects the interpretation of every ablation
experiment**"*, with methods that *"automate around self-repair"* doing so by
joint-ablation analysis or by de-biasing a node's own score.

**Every route named is an ablation route.** Nothing surfaced measures repair
against a non-removing intervention. **The question looks open**, which is the
answer it needed before being worth building — and `[S]` is a weak basis for a
novelty claim, so it stays in the queue rather than in a registration.

## 8. Oversmoothing has an inference-side literature, and one paper is the spreading arm

`plan-9.md` §6.4's spreading arm — stretch a subspace and ask whether late-layer
distinguishability recovers — is adjacent to a populated line.

- **`2303.06562`, *ContraNorm: A Contrastive Learning Perspective on
  Oversmoothing and Beyond*** — **a normalisation-layer modification that spreads
  representations apart.** This is the closest published object to a Phase 9
  metric intervention and it should be read before `design-9.md` freezes
  anything.
- **`2410.07799`** *Mind the Gap: a Spectral Analysis of Rank Collapse*;
  **`2602.09297`** *Laplacian Heads Improve Transformers by Smoothing Token
  Representations* — the deliberate-smoothing direction, i.e. the cluster-forming
  arm, from the other side;
  **`2312.04234`** *Graph Convolutions Enrich the Self-Attention in Transformers*.
- Framing already familiar here: *"self-attention acts as a low-pass filter"*,
  and residual connections plus LayerNorm *"slow down the collapse rate"* —
  which is `notes-9.md` §4's resistance question in the oversmoothing dialect.

**Consequence: the spreading arm is a comparison, not a discovery.** Its value is
that this project can predict the effect from `gamma_beta` and read it on the
measure, which ContraNorm does not — but the phase must say so.

## 9. Two more, one adjacent and one a direct neighbour

- **`2601.02932`, *Data-driven Reduction of Transfer Operators for Particle
  Clustering Dynamics*** **[N]** — the title is `plan-9.md` §5.1's construction.
  Searches for PCCA+ / implied timescales applied to **transformer
  representations** returned molecular dynamics, behaviour and climate and
  **nothing on transformers**, so §5.1's move looks open; this paper is the
  nearest thing and must be read.
- **`2501.10573`, *The Geometry of Tokens in Internal Representations of Large
  Language Models*** (Viswanathan, Gardinazzi, Panerai, Cazzaniga, Biagetti,
  Jan 2025) — **the closest neighbour Phase 10 has.** Uses *"the notion of
  empirical measure, which encodes the distribution of token point clouds across
  transformer layers and drives the evolution of token representations in the
  **mean-field interacting picture**"*; metrics are intrinsic dimension,
  neighbourhood overlap and cosine similarity per layer; finds a **correlation
  between token geometry and next-token cross-entropy**, with *"prompts with
  higher loss … represented in higher-dimensional spaces"*; validated against a
  **shuffled-token control**.

  Three things it does to this phase: it is the same object in the same
  language; its loss correlation is the **functional column** of `notes-10.md`
  §3.1 done a different way; and its shuffled-token control is a null this
  project should consider adopting. **Read before `design-10.md`.**

## 10. Queue, for a machine with arXiv access

> **SUPERSEDED 2026-09-20 by §15.** Items 1, 2, 3 and 5 below were read as
> primary text in the third pass and are `[R]`; the list is kept as written
> because the *reasons* it gives for the priority order are what the third pass
> then confirmed or refuted.

In priority order. The first two change constructions.

1. ~~**`2411.04990`**~~ — **DONE 2026-09-20**, read in full from the PDF:
   `docs/readings/2411.04990.md`. It opens five new pointers, of which
   **Castin, Ablin & Peyré 2024** is the one that could move a decision: a
   reparametrisation that *"allows them to recast causal attention as mean-field
   dynamics"*. **If it restores mean-field structure it may restore the
   gradient-flow framing `plan-9.md` §5.1a wrote off.** Also `2410.23228`
   (Bruno et al.) and Cowsik et al. 2024, which includes MLP layers — the gap
   §6 of the paper names and Phase 10 §7's question.
2. **`2501.10573`** — the geometry/loss correlation and the shuffled-token null.
3. **`2605.12765`** and **`2505.16831`** — whether §6's rotation-vs-congruence
   distinction is as clean as it looks.
4. **`2303.06562`** — ContraNorm, before any spreading arm is designed.
5. **`2601.02932`** — the transfer-operator reduction.
6. **`2607.15495`** — the J-lens paper itself; only its companion code has been
   read.

---

# Third pass (2026-09-20) — **five papers READ as primary text**

**Trigger 1 is now discharged for the five that mattered.** The user supplied
PDFs for queue items 1, 2, 3 (both) and 5 of §10 — `2411.04990`, `2501.10573`,
`2605.12765`, `2505.16831`, `2601.02932` — so everything in this section is
**[R]**, read from the paper, not from a search summary. Items 4
(`2303.06562`, ContraNorm) and 6 (`2607.15495`, the J-lens paper)
remain **[S]** and stay in the queue.

**It paid for itself four times.**

1. §11 gives the parking correspondence exactly, and it **dissolves the `d ≫ 1`
   problem** §5 thought was fatal — Lemma C.1 is distribution-free and
   dimension-free in the form the project can actually evaluate.
2. §11.4 finds that **F0 as run does not test the paper's claim.** It tests a
   proxy. The paper's nucleus is a *geometric* object with a definition, not
   "the earliest member of a density cluster".
3. §11.5: **the masked system IS a gradient flow — a sequential one**, with a
   per-token energy and a `1/Z_k` prefactor. `notes-10.md` §10.1 and
   `plan-9.md` §5.1a are both too pessimistic, and the prefactor is *literally*
   the quantity F12 measured.
4. §12.3: `2501.10573` measures, per layer and per prompt, **exactly the
   effective dimension `d_1` that `2411.04990` conjectures the count law runs
   in** — and it is ≈ 10, not ≈ 225. The project's candidate list for
   `math-10.md` §5.2's `d_eff` was missing the theoretically-correct one.

Corrections this pass makes to earlier passes are marked **CORRECTION** and are
not quietly folded in.

---

## 11. `2411.04990` — *Clustering in Causal Attention Masking* **[R]**

Karagodin, Polyanskiy, Rigollet. NeurIPS 2024; v2, 10 Nov 2024, 22 pp.
**The paper two phases depended on and neither had read.**

### 11.1 The model, stated so the transfer is auditable

Tokens are particles `x_k(t) ∈ S^{d−1}`; `t` is **depth**, treated continuously.
Full attention (SA) has token `k` summing over all `j ∈ [n]`; the paper's object
is **causal** attention (CSA), summing over `j ≤ k`:

```
    ẋ_k = P_{x_k} ( (1/Z_k) Σ_{j≤k} e^{β⟨Q x_k, K x_j⟩} V x_j ),
    Z_k  = Σ_{j≤k} e^{β⟨Q x_k, K x_j⟩}
```

`P_x y = y − ⟨x,y⟩x/|x|²` is the tangent projection; RMSNorm is what puts the
particles on the sphere, and its trainable diagonal `D` is **absorbed into
`K, Q, V`** (§2 — worth knowing, because it is the same `Γ` Phase 9 wants to
patch, and this framing says a `Γ`-patch is a `Q,K,V` congruence by
construction, which is what `plan-9.md` §4.1 derives independently).

**Four simplifications bound every transfer** (§6, verbatim): **tied weights
across layers**, **no MLP** (*"incorporating the MLP dynamics into our
theoretical framework remains a significant open challenge"*), no residual
term in (SA) itself, and §5 additionally fixes `V = I_d`, `Q = K = I_2`,
`d = 2`. Pythia has none of those properties. **Nothing below is a theorem
about Pythia**, and the phase must say so wherever it leans on this.

### 11.2 The parking correspondence, exactly — what the cars and the street are

§5.1, and this is the answer §10 item 1 asked for.

- **The street** is the sphere `S^{d−1}` with **geodesic distance**.
- **Arrival order** is **token position**. This is the whole reason the parking
  analogy exists only under a causal mask: token `k` sees only `j ≤ k`, so the
  sequence indexes a *sequential* acceptance process. Under full attention there
  is no arrival order and no parking problem.
- **Car length** is the separation parameter `δ > 0`, and §5.1 fixes its scale:
  particles *"exert maximal attractive force at distances of order at most
  `β^{−1/2}`, with rapid decay beyond this scale"*, so the paper takes
  **`δ = c β^{−1/2}`** with `c` sufficiently large.
- **Rényi centre**: the subsequence `(x_{s_j})` with
  `dist(x_{s_j}, x_{s_i}) > δ` **for all previous centres** `i < j`.
- **Strong Rényi centre**: `dist(x_{s_j}, x_i) > δ` **for all previous tokens**
  `i < s_j`. Strong centres are a **subset** of Rényi centres.

Both are **greedy sequential acceptance rules over the token sequence**, needing
nothing but positions and a distance. **Neither mentions clusters.** The claim
is that these accepted tokens *become* the nuclei, not that they are identified
by any clustering algorithm.

**What each kind does, and they differ** (§5.1, Fig. 2–3):

| | strong Rényi centres | Rényi centres |
|---|---|---|
| separation from | **all** previous tokens | previous **centres** only |
| count | fewer | more |
| stationarity | **near-stationary for a time exponential in `c²`** (Lemma 5.1) | *"lack positional stability"*; move, merge, disappear |
| coverage | does not explain all clusters | captures more meta-stable clusters |

> **This is a two-sided, graded prediction, and F0's single scalar cannot
> express it.** The informative statement is the *pair*: strong centres are the
> stationary anchors and undercount the clusters; ordinary centres cover more
> and drift. A test that collapses both into one number cannot come back
> "right".

### 11.3 The count law, and it is better than §5 believed — **CORRECTION**

§5 above, from two search summaries, concluded: *"the law is in β and dimension,
and 0.7476 does not appear in it."* **The first half is right. The second half
is wrong, and the correction matters.** Appendix C.4, verbatim:

> *"In the case `d = 2`, the result from Dvoretzky and Robbins (1964) implies
> that as `δ → 0`, the average number of elements in the sequence approaches
> `c·2π/δ` superexponentially fast, where `c ≈ 0.75` is the Rényi constant."*

So the Rényi constant **is** in the paper — as the `d = 2`, `δ → 0` asymptotic
for **ordinary** (not strong) Rényi centres. `lit-1.md` §4's original
description was *half* right, not wholly wrong, and §5's correction over-shot.
Both should be fixed to the statement above.

**And the load-bearing result is Lemma C.1, which neither pass had.** For an
infinitely long i.i.d. sequence with law `μ` on `S^{d−1}`:

```
    E[ #strong Rényi centres ] = ∫_{S^{d−1}} 1/μ(B_δ(x)) dμ(x)  =  1/σ^{d−1}(B_δ)
```

the **reciprocal spherical-cap measure**, `π δ^{−1}` at `d = 2`,
`(3 sin²(δ/2))^{−1}` at `d = 3`, growing as `δ^{−(d−1)}`. With `δ = cβ^{−1/2}`
that is exactly the `Θ(β^{(d−1)/2})` frequency §5 quotes — **the frequency law
IS Lemma C.1**, not an independent result.

> **Three things follow, and the third is the one that changes the phase.**
>
> 1. **There is no `n` in it.** `math-10.md` §6 item 1 asks whether the law
>    carries an `n` dependence; the answer is **no in the limit** — it is the
>    expected count in an *infinitely long* sequence, so `b = 0` asymptotically,
>    with the finite-`n` form `∫ Σ_{k≤n} P(dist(x,X_1) > δ)^{k−1} dμ` saturating
>    from below. **Thread 1 is answered.**
> 2. **The last equality needs uniformity; the middle expression does not.** The
>    paper says so explicitly: for strong centres the average-count computation
>    *"works for any distribution regardless of the dimension"*, in contrast to
>    ordinary Rényi centres, whose extension *"remain[s] open … particularly in
>    higher dimensions (d > 2)"*.
> 3. **Therefore the `d ≫ 1` objection dissolves for strong centres.** §5 above
>    and `math-10.md` §5.1 both reject the prediction as *"unusable at `d =
>    1024`, exponent 511.5"*. That is true of the **power-law asymptotic** and
>    false of the **exact form**: `E_{x∼μ}[ 1/μ(B_δ(x)) ]` is a reciprocal
>    kNN-density average, estimable directly from a layer's token cloud, with no
>    exponent, no ambient dimension and no `β` — only `δ`, which is a distance
>    and can simply be **swept**.

This is the single most useful thing in the paper for this phase: a **parameter-
free, distribution-free null for the number of nuclei**, computable on artifacts
already on disk.

### 11.4 What this does to F0 — **the row does not test the paper's claim**

`tools/run/p10_anchor.py` computes `mean_nucleus_position`: the mean over
HDBSCAN clusters of the normalised position of each cluster's **earliest
member**, with the direction fixed to *"less"*. Its docstring calls this *"the
only reading of 'nucleus' a causal mask permits"*.

**It is a defensible proxy and it is not the paper's object.** The paper's
nucleus is defined by a **geometric exclusion condition** — separated by more
than `δ` from every preceding token — evaluated on positions and distances
alone, with no partition involved. The two can disagree completely: a density
cluster's earliest member need not be `δ`-separated from anything, and a strong
Rényi centre need not be in any HDBSCAN cluster at all.

What Lemma 5.1's remark actually claims is narrower and sharper:

> *"it is easy to prove that indexes `s_j` are mostly small. Thus … early strong
> Rényi centres are almost stationary for a time that is exponential with the
> square of separation magnitude."*

i.e. **the accepted set is index-biased early** — which is a near-tautology of
sequential exclusion, since a later token has more predecessors to avoid — and
**the early ones among them are the stationary ones**. Neither statement is
about where a density cluster's first member sits.

> **Consequence, and it is a relief rather than a retraction.** F0's result
> (`status-10.md` §1.2 — nuclei at 0.316 against a null 0.208, failing in the
> predicted direction under both nulls) stands as a fact about *this statistic*
> and is **not** evidence against the parking account, because the statistic is
> not the account's. `status-10.md` §6's line *"F0 is not an adjudication of the
> parking account"* was written for a different reason and turns out to be true
> for a stronger one.
>
> **And the real test is free.** It needs positions, a distance, and a `δ`
> sweep. Nothing else. See §11.7.

### 11.5 The gradient-flow claim, verbatim — and it is **not** what two phases recorded

`notes-10.md` §10.1 and `plan-9.md` §5.1a both record, `[S]`-grade, that *"the
causally-masked system cannot be interpreted as a mean-field gradient flow"*,
and put the Wasserstein-Hessian-of-`E_β` framing **at risk** on that basis.

The paper says both halves, and the second one was missing. §4:

> *"our (CSA) does not have a gradient-flow structure and thus techniques of
> Łojasiewicz are not applicable."*

and then, §5.2 and **Lemma 5.3**:

> *"Since our dynamical system is not a gradient flow, the classical Łojasiewicz
> convergence theorem does not apply. Instead, we establish convergence by
> observing that the causal dynamics (both with and without frozen tokens) is,
> in fact, a **sequential gradient flow, where each particle minimizes a
> slightly different energy**."*

Lemma 5.3 gives it explicitly, for `n` particles on `S¹`:

```
    φ̇_k = − (1 / Z_k(φ_1,…,φ_k)) · ∂E_k(φ_1,…,φ_k)/∂φ_k
```

with `E_1,…,E_n` a family of `C¹` energies and `0 < c < Z_k(φ) < C`. Theorem
5.2's proof (App. C.3) writes the `E_k` down for the frozen-token case:
`E_k = −( Σ_{j<k} e^{β(cos(φ_k−φ_j)−1)} + Σ_j a_j e^{β(cos(φ_k−θ_j)−1)} )`.

> **CORRECTION to `notes-10.md` §10.1 and `plan-9.md` §5.1a.** What is absent is
> a *single global* potential whose gradient flow the whole system is — so
> Łojasiewicz, and any argument that needs one `E_β` for the ensemble, does not
> apply. What is **present** is a per-particle energy `E_k` and a gradient flow
> in it, causally ordered. The Hessian framing is therefore **not dead; it is
> per-token and sequential**, and a curvature statement has to be made about
> `∂²E_k/∂φ_k²` rather than about a Wasserstein Hessian of one ensemble
> functional. That is a weaker and more specific claim than Phase 9 was making
> and a much stronger position than "at risk".

**And the prefactor is F12's measurement.** `1/Z_k` multiplies particle `k`'s
own gradient. Large `Z_k` ⇒ small velocity per unit energy gradient; small
`Z_k` ⇒ cheap to move. That is precisely `math-1.md` §1A.6's metric reading of
`Z`, **now an equation rather than an interpretation** — and `status-10.md`
§1.5's parked-not-pinned reading rests on it. Two caveats, both real: Lemma 5.3
is stated on `S¹` with `Q = K = V = I`, and `Z_k` here is the **row**
normaliser, which `math-10.md` §2 already shows is `(i+1)`-tilted under the
mask. **`Z_i/(i+1)` remains the quantity to read.**

### 11.6 Three more things the paper has that this project can use

1. **Theorem 4.1 — the asymptotic configuration is token 1's initial position.**
   For `V = I_d` and arbitrary `Q, K`, from almost any start, `x_k(t) → x_1(0)`
   for every `k`. Table 1's atlas then maps `(sign λ_max(V), mult, real/complex)`
   to five final configurations: one point at `x_1(0)`; one point in `L`; two
   points `±ξ`; a cloud around `L`; two clouds around `±ξ`. **Position 0 is not
   a nuisance in this theory — it is the attractor**, which is a third reading
   of the sink to set beside `attention-10.md` §3's three.
2. **§B.3 measures the `V`-spectra of `albert-xlarge-v2`** — the exact model of
   Phases 1–6 — and reports: most heads have **real** `λ_max` (which bears on
   Phase 2b's imaginary line); **heads 1 and 6 have negative `λ_max`**; and
   Gaussian initialisation puts the spectrum *"far from the left half-plane"*,
   so a negative `λ_max` is **learned**. They read it as functional. **This is
   direct prior art for the project's V-attractive / V-repulsive split** and
   should be cited wherever that split is claimed as this project's own.
3. **Theorem 5.2 — freezing the centres is a designed intervention.** If the
   quasi-stationary tokens are artificially frozen (*"analogous to
   cross-attention in encoder–decoder architectures"*), all other tokens
   converge to them. That is an *additive/active* intervention on the
   **particle**, and `notes-10.md` §6's taxonomy has no row for it.

**The paper's own open problem, §6, is measurable here.** *"A complete
meta-stability theory would require demonstrating that each Rényi center
captures `Ω(n)` particles in `O(1)` time … even the weaker claim of capturing
`ω(1)` particles remains unproven, presenting a crucial direction for future
research."* Their Figure 3 is the empirical version: *percentage of particles
consumed by Rényi and strong Rényi centres over time*, averaged over 5 000
simulations at `n = 200, d = 2, β = 64`. **The real-model analogue — coverage as
a function of depth, on 152 directories — is free, and it is a figure nobody
has.**

### 11.7 What this licenses building (no code written here; rows proposed in `notes-10.md` §8)

- **The centre scan.** Greedy sequential acceptance over token positions at each
  layer, both rules, **swept in `δ`**. Returns: the accepted index sets, their
  index distribution, the count against Lemma C.1's `E[1/μ(B_δ)]` estimate, and
  the ARI between "is a centre" and "is a cluster core".
- **The coverage curve.** Fraction of tokens within `δ` of some accepted centre,
  as a function of depth — Figure 3 for a real model, and the paper's §6 open
  problem.
- **δ as a measurement rather than a convention.** `δ = cβ^{−1/2}` means the
  `δ` at which the observed centre count matches Lemma C.1's prediction
  **reads back `c²/β`**. `PROJECT.md` §3.40's undecided factor-of-8 becomes
  something to measure instead of something to decide.

---

## 12. `2501.10573` — *The Intrinsic Dimension of Prompts…* **[R]**

Viswanathan, Gardinazzi, Panerai, Cazzaniga, Biagetti. **v2, 21 Aug 2026.**

**CORRECTION to §9.** The title has changed between versions: §9 records *"The
Geometry of Tokens in Internal Representations of Large Language Models"*, which
was v1. v2 is *"The Intrinsic Dimension of Prompts in Internal Representations
of Large Language Models"*, and the paper is narrower and sharper than §9's
description — it is an **intrinsic-dimension** paper with a safety case study,
not a general token-geometry paper. Code:
`github.com/RitAreaSciencePark/token_geometry` (GitHub is reachable, §0.1).

### 12.1 What it does

Prompts as token clouds; the **empirical measure** at each layer probed by
**intrinsic dimension** with three kNN estimators — **GRIDE** (likelihood on
kNN distance ratios, range scaling 4), **TLE**, **ESS** (`k = 10, 20`). Models:
Llama-3 8B, Mistral 7B, **Pythia 6.9B-deduped**, OPT 6.7B, all 32 layers,
`d = 4096`; 2 244 Pile-10K prompts truncated to `N = 1024` tokens. Local
homogeneity checked with PAk: neighbourhoods are approximately constant-density
**up to `k* ≈ 20`**, which is the bound on every kNN estimate here.

Results: **ID peaks in early-to-middle layers**; **ID rises under shuffling**;
`ρ(log ID, surprisal) ≈ 0.6–0.8` across layers, all four models, all three
estimators, `p < 0.01`. Mechanism, in three steps: last-layer ID ↔ logit ID
(`ρ = 0.98`), logit ID ↔ contextual entropy (`ρ = 0.60`), contextual entropy ≈
average surprisal (empirically `x = y`). Toy result: for a Dirichlet on a
`D`-simplex, `⟨S⟩ = ψ(D+1) − ψ(2) → log D − 0.42`.

### 12.2 The shuffling null — **adopt this, it is better than what the phase has**

Not a plain token shuffle. The prompt is split into `nBlocks` blocks of size
`b_S = 1024/4^S` and the blocks are permuted, `S ∈ {0,…,5}` — a **six-level
graded** disruption from intact to fully shuffled, with the unigram distribution
preserved at every level, **calibrated** by BLEU (1.00 → 0.03) and BERTScore F1
(1.00 → 0.76).

> A graded, calibrated null gives a **dose–response curve** where a binary
> control gives a yes/no. `notes-10.md` §4.4 wants a null for the ARI's variance
> and `status-10.md` §3 wants something to read the reproducibility floor
> against; this is the shape both need, and the block construction is free to
> implement on the battery prompts.

One caveat the paper carries and the phase would inherit: ID estimates need
`N ≳ 500` tokens, and degrade below that (Table 3). The project's battery
prompts are much shorter than 1 024.

### 12.3 The number `math-10.md` §5.2 is missing — **and it changes the answer**

`2411.04990` §5 conjectures that for general `V` the ambient `d` in the count
law is replaced by an **effective dimension `d_1`**, the dimension of the
subspace spanned by the few principal eigenvectors the particles collapse onto
(§11.3). `math-10.md` §5.2 offers three candidates for that `d_eff`:

| candidate | value | predicted slope `a = (d_eff−1)/2` |
|---|---|---|
| ambient (pythia-410m) | 1024 | 511.5 |
| effective-rank plateau | ~225 | 112.0 |
| ambient participation ratio | 22 | 10.5 |
| **kNN intrinsic dimension (this paper)** | **≈ 7–15** | **≈ 3–7** |

> **The first three are all *linear* dimensions and the theory wants a manifold
> dimension.** Effective rank and participation ratio are functionals of a
> covariance spectrum; `d_1` is the dimension of the set the particles lie on.
> A kNN ID estimator measures the second and the others do not, so **GRIDE/TLE/
> ESS is the theoretically-correct candidate and it was not on the list.** It is
> also ~30× smaller than the nearest rival, which is the difference between a
> prediction that is measurable and one that is not.

That gives `math-10.md` §5.2's regression an **independent comparator**: fit `a`
from cluster counts, estimate ID directly, and ask whether they agree. Agreement
would be a non-trivial confirmation of the `d_1` conjecture; disagreement
localises which half is wrong. Neither needs a forward pass — both read
`activations.npz`.

### 12.4 Three more joins, and one tension

- **The loss correlation is `notes-10.md` §3.1's functional column done another
  way**, at the *prompt* level rather than the particle level, and it is already
  positive. Any Phase 10 functional claim has to say how it differs from "higher
  ID ⇒ higher surprisal".
- **Pythia 6.9B-deduped is in the model set** and the tuned-lens appendix covers
  Pythia 160M/410M/2.8B/6.9B — so the ID profile has been measured on the
  project's model family, though not on its checkpoints. **The developmental
  axis is untouched here too.**
- **Their §6 probe is Phase 9's readout panel in disguise**: a linear classifier
  on the per-layer ID profile, 90–95 %, beating Llama Guard and Shield Gemma
  (60–70 %), and matching a TunedLens-entropy probe. Geometry-as-readout is a
  validated instrument, not a hope.
- **The tension, and it is worth stating.** Footnote 1: *"The dynamics of a token
  `i` depends on the position of all the tokens `x_j(ℓ)` but not on their
  labels, which is an assumption in the mean-field interacting particle
  framework."* Exchangeability is exactly what a causal mask destroys, and
  `2411.04990` is built on that destruction. **The two papers this phase leans
  on hardest disagree about whether position is a label.** Phase 10 sits on the
  masked side (`math-10.md` §5.3: *"the position axis is not a nuisance variable
  in this phase"*), and should say which framework each borrowed instrument
  comes from.

---

## 13. `2601.02932` — *Data-driven Reduction of Transfer Operators for Particle Clustering Dynamics* **[R]**

Wehlitz, Pavliotis, Schütte, Winkelmann (ZIB / FU Berlin / Imperial). v2, 9 Apr
2026. **`plan-9.md` §5.1's construction, built.**

### 13.1 The architecture, which is the transferable part

A three-stage reduction (their Fig. 1), each stage analytically defined *before*
any data-driven step:

```
 P^τ_N  (Perron–Frobenius on particle configs, T^N)
   → Galerkin onto discretised CONCENTRATIONS      → P^τ_N  (finite)
   → Galerkin onto a COARSE PARTITION of those     → P^τ    (the reduced operator)
   → estimated from data: Diffusion Maps + partition + transition counting
```

> **The state is the empirical measure, not the particle.** That is the move.
> `cluster_tracking.py` builds transitions between *particle→cluster
> assignments*; this builds transitions between *whole configurations*. They are
> different operators and `plan-9.md` §5.1 does not currently distinguish them.

Data-driven half: **Diffusion Maps** with the anisotropic normalisation
(`α = 1`, which *"removes the influence of the empirical data density so that
the resulting diffusion process reflects the intrinsic geometry rather than
artifacts of uneven sampling"*) → embedding coordinates `ξ_1,…,ξ_d` → partition
by **uniform grid or K-means Voronoi** → **Ulam's method** transition counts at
lag `τ` → transition matrix. Then: **implied timescales `T_i = −τ/log μ_i`**,
**PCCA+** for metastable macrostates (MSMTools), **MFPT**, **transition-path
theory** committors.

### 13.2 The metric result, which this project should read as validation

They state outright that the choice of distance between two configurations
decides what the embedding sees, and that a pointwise metric fails once cluster
centres move:

- **translation-invariant `L²`** for the multichromatic potential, where cluster
  positions are fixed — `O(K log K)` via FFT cross-correlation;
- **translation-invariant Wasserstein-1** for the Morse potential, because
  *"the `L²`-metric is not suitable, since cluster centers may drift and merge,
  and simple point-wise comparison does not adequately capture their relative
  positions."*

> **That is `PROJECT.md` §3.29's programme — optimal transport as the right
> geometry on measures — arrived at independently, for a measured reason, in a
> neighbouring field.** And this project already computes `w2_identity`,
> `w2_optimal`, `sliced_w2`, `wasserstein_arc_length` and `straightness`
> (`core/dissipation.py`), which `plan-9.md` §5.3 lists as *built, unrun*. The
> distance the reduction needs is the one already on the shelf.

### 13.3 The reversibility hazard — **`math-10.md` §6 thread 4 is answered**

`math-10.md` §6 item 4 asks whether PCCA+'s argument survives a non-reversible
transition matrix. The paper answers it, §5.1.1:

> *"If the process is reversible, the leading eigenvalues and eigenvectors are
> real-valued … For non-reversible processes, one has to analyze its **singular
> values** and the related singular vectors, or the leading complex-valued
> eigenvalues and respective elements of the **Schur decomposition**."*

They then **enforce** reversibility with a constrained MLE (their Eqs. 21–25),
and they are entitled to: their particle system is a gradient diffusion,
reversible w.r.t. a Gibbs measure.

> **A transformer's depth dynamics is not.** Depth is a one-way index, the
> masked system is only a *sequential* gradient flow (§11.5), and cluster
> splitting is the reverse of a merge. **So the reversibility-constrained
> estimator must NOT be transported**, `t_i = −τ/log|μ_i|` still reads, and the
> correct spectral tool for the non-reversible case is the **real Schur
> decomposition** — which is `core/`'s existing instrument, pointed at a
> function-defined operator exactly as `notes-10.md` §4.2 wants.
>
> This is the cleanest available answer to §4.2's *"nobody has shown `J_l`'s
> spectrum means anything"*: for a **transfer** operator, the literature says
> what the spectrum means and which decomposition to use when detailed balance
> fails.

### 13.4 Two empirical statements to test against Phase 1's turnover data

Free, and neither is currently checked:

1. *"The characteristic times between successive cluster merges increase roughly
   exponentially as the system evolves, reflecting the progressive slowdown of
   the dynamics as the number of clusters decreases."* Phase 1's finding 4 has
   mean lifespan **falling** 7.0 → 4.5 across *training*; this is a claim about
   *depth*. The two axes must not be conflated, and the depth version has never
   been plotted.
2. *"reverse events of cluster splitting are highly unlikely and have never been
   observed throughout the simulation time."* `cluster_tracking.py` records
   births; whether any of them are splits is a groupby.

---

## 14. `2605.12765` (GUARD-IT) and `2505.16831` (Unlearning Isn't Deletion) **[R]**

Read together, because §6 above put a Phase 9 novelty claim on the distinction
between them and one half of that distinction does not survive.

### 14.1 GUARD-IT, precisely — and §6's "rotation" is looser than §6 assumed

Turani et al., v3, 14 Jul 2026. Offline: embed the *forget* corpus with a
sentence-transformer, L2-normalise, **k-means with `k` chosen by mean
silhouette**, one **Prototype Steering Vector** per cluster (mean-pooled
residual at a target layer). Online: a **similarity gate** `K(x) = {j :
sim(c_j, φ(x)) ≥ T}`; if empty, **no intervention at all**; else average the
active PSVs, project **perpendicular to the retain direction** (Eq. 6), rescale
to the mean activation norm (Eq. 7), and apply

```
    h′ = (h − α v̂(x)) · ‖h‖ / ‖h − α v̂(x)‖          (Eq. 8)
```

Defaults: first-quartile layer, `T = 0.55`, `α = 0.2`. TOFU/MUSE, 12 baselines,
Llama-3.2 1B/3B and 3.1 8B; the only method that simultaneously preserves
utility, suppresses memorisation and avoids collapse across all settings; stable
under 4/8-bit quantisation because no weights move.

**Three corrections to §6, all in the same direction — the differentia is
narrower than §6 claimed but it is still there.**

1. **Eq. 8 is norm-preserving; it is not a rotation.** It is a nonlinear,
   input-dependent self-map of the sphere of radius `‖h‖`. The paper's own prose
   (*"a pure rotation in the residual stream"*) is loose, and §6 repeated it.
   `‖h′‖ = ‖h‖` is exact and is all that is claimed by the algebra.
2. **"Exactly invertible" is not a differentia either.** Given `v̂` and `α`,
   Eq. 8 inverts in closed form: `‖h‖ = ‖h′‖` is known, so `h = λ u + α v̂` with
   `u = h′/‖h′‖` and `λ` the positive root of
   `λ² + 2λα⟨u,v̂⟩ + α²‖v̂‖² = ‖h′‖²`. `notes-9.md` §8 and `plan-9.md` §6.1 rest
   part of the novelty claim on invertibility; **that part should be dropped.**
3. **What does survive is a clean object-level distinction, and it is sharper
   than "isometry vs congruence".**

> | | GUARD-IT | a γ-patch |
> |---|---|---|
> | acts on | the **state** `h` | the **bilinear form** `W_QK` (a congruence, `plan-9.md` §4.1) |
> | what changes | where this token is | how *every* pair of tokens is compared |
> | selection | **content**: a similarity gate over forget-corpus embeddings | **geometry**: a subspace and an anisotropy |
> | scope | one token, when gated | the whole block, always |
> | norm | preserved exactly | not the invariant in question |
>
> **State versus operator is the claim to make.** Not isometry-versus-congruence,
> which mis-describes Eq. 8, and not invertibility, which both have.

**And a `notes-10.md` §5 point lands here.** GUARD-IT's clusters are **document
clusters in a sentence-embedding space** — §5's *direction/feature* sense, one
step further removed. They are not token clusters in a residual stream, and no
theorem in this phase's literature is about them. The paper also reports
(§2.1.1) that it *"evaluated alternative clustering algorithms and observed no
consistent differences in downstream unlearning quality"* — a useful contrast
to `status-10.md` §3's reproducibility floor, and **not** a rebuttal of it: a
downstream task can be insensitive to a partition that a per-layer statistic is
not.

Two further details worth keeping: the most effective intervention layers sit
**earlier than mid-stack, around the first quartile** (they argue late steering
leaves no residual depth to re-integrate) — which is where Phase 10's cluster
story lives; and the gate means **utility is untouched on non-matching inputs**,
which is the structural reason it beats gradient methods rather than a tuning
win.

### 14.2 *Unlearning Isn't Deletion* — a taxonomy and a four-diagnostic panel

Xu, Yue, Liu, Ye, Zheng, Hu, Du, Hu. **ICML 2026** (PMLR 306), v3, 16 May 2026.
Code: `github.com/XiaoyuXU1/Representational_Analysis_Tools`.

**Definition 2.1** — with `Δ_u(T) = E(θ_0,T) − E(θ_u,T)` the drop after
unlearning and `Δ_r(T) = E(θ_0,T) − E(θ_r,T)` the change after a **controlled
relearning** phase (budget matched to `|D_f|`), forgetting is *catastrophic* if
both `Δ_u(T_r)` and `Δ_u(T_f)` ≫ 0, and *reversible* if `Δ_r(T_f) ≈ 0`. Four
regimes; **irreversible non-catastrophic is the ideal and they observe it only
once**, as a special case.

The panel, all per layer, on forget / retain / **unrelated** probe sets:
**PCA similarity** (cosine between PC1 directions), **PCA shift** (drift in the
original's top-2 PC plane), **linear CKA**, **FIM diagonal**, and **mean PCA
distance** as the scalar summary.

**Four things this does for the project, and one of them is not about Phase 9.**

1. **§6's reading is confirmed and is a citation, not a threat.** Task-level
   metrics *"can be misleading, as models can appear to forget while their
   original behavior is easily restored through minimal fine-tuning"*. Any
   Phase 9 forgetting result must carry a **relearning arm** or it measures the
   thing this paper says is routinely mismeasured. `notes-9.md` §2's *"collapse
   is a release operation, not a deletion one"* now has an evaluation framework
   behind it.
2. **They reach for a panel for the reason `CLAIM-C` found the hard way.**
   *"Relying on PCA similarity alone can obscure subtle effects; employing both
   avoids overlooking fine-grained distinctions."* `CLAIM-C`'s gate returned
   INSUFFICIENT because its six metrics disagreed (`PROJECT.md` §3.41) —
   **disagreement among complementary diagnostics is what this paper designs
   for**, and it uses **linear CKA** as one of the four, which is `cka_prev`,
   the metric that scored 0/8. Worth reading before the metric set is ever
   revisited.
3. **A closed-form sensitivity bound, and it belongs in `tools/math_checks/`.**
   Via Davis–Kahan, `cos∠(c_i^orig, c_i^upd) ≈ 1 − O(‖E_i‖/(λ_{1,i} − λ_{2,i}))`
   — a PC1-direction readout is trustworthy **only in proportion to the
   eigengap**. This project computes eigengaps everywhere and has never gated a
   spectral readout on one. `CKA ≈ 1 − O(‖ΔK̃‖_*/‖K̃‖_*)` and
   `F̄ = F_0 − O((1/P)Σ‖E_i‖)` are the companions.
4. **The warning that lands on `notes-10.md` §4.5 caveat 4.** *"Small
   perturbations near the logits can distort task-level metrics despite intact
   features, hence leading to misleading assessments."* A lens readout is a
   logit-space readout. **The failure mode the project fears with the J-lens has
   been characterised, and the defence is exactly the causal column §3.1
   already demands.**

---

## 15. Queue, updated

Discharged this pass: **`2411.04990` [R]**, **`2501.10573` [R]**,
**`2605.12765` [R]**, **`2505.16831` [R]**, **`2601.02932` [R]**.

Still open, and neither is now blocking a construction:

1. **`2303.06562`** ContraNorm **[S]** — before any Phase 9 spreading arm is
   designed. §8.
2. **`2607.15495`** the J-lens paper **[S]** — only its companion code has been
   read (§1). Its arXiv id should be added to `p2_eigenspectra/lens_band.py` and
   `archive/p5_single_mstate_analysis/status-5.md` regardless.
3. **Still unasked, and §4 named both**: has anyone clustered tokens **in a lens
   basis**; has the parking prediction been checked **empirically** on a real
   model. §11 makes the second one cheap enough that the answer matters more,
   not less — `github.com/anthropics/jacobian-lens` and the two repos named
   above are reachable and are the fastest route.
