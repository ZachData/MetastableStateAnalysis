<!-- p10_cluster_function/lit-10.md -->
# Phase 10 — LITERATURE (partial scan, 2026-09-20)

**Status: two passes, still not the full scan.** The first pass (2026-09-20,
§1) settled one factual question that arrived with the phase — *does a fitted
Jacobian lens exist for a Pythia model, or must one be trained?* The second pass
(§5–§9, same day) worked the topic list in `notes-10.md` §12 and
`plan-9.md` §12. **Trigger 1 is still not discharged** — every scholarly host is
blocked, so nothing below §1 was read as primary text, and §10 lists what a
machine with arXiv access must still do.

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

In priority order. The first two change constructions.

1. **`2411.04990`** — the exact parking statement (what are the cars, the
   street, a Rényi centre, a strong Rényi centre), whether the count law carries
   an `n` dependence, and the gradient-flow claim verbatim. **Two phases depend
   on this one paper and neither has read it.**
2. **`2501.10573`** — the geometry/loss correlation and the shuffled-token null.
3. **`2605.12765`** and **`2505.16831`** — whether §6's rotation-vs-congruence
   distinction is as clean as it looks.
4. **`2303.06562`** — ContraNorm, before any spreading arm is designed.
5. **`2601.02932`** — the transfer-operator reduction.
6. **`2607.15495`** — the J-lens paper itself; only its companion code has been
   read.
