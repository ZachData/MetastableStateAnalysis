<!-- p10_cluster_function/lit-10.md -->
# Phase 10 — LITERATURE (partial scan, 2026-09-20)

**Status: one question answered properly, the rest are leads.** This is not the
full `CLAUDE.md` trigger-1 scan. It was run to settle a single factual question
that arrived with the phase — *does a fitted Jacobian lens exist for a Pythia
model, or must one be trained?* — because the answer changes which instruments
Phase 10 is built on. The remaining searches `notes-10.md` §12 names are **not**
run, and trigger 1 is **not** discharged.

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
