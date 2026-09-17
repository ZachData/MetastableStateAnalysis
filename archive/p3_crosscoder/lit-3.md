<!-- archive/p3_crosscoder/lit-3.md -->
# Phase 3 — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**This phase is archived** (`archive/README.md`). The file exists because
`FROZEN.md` carries a reintroduction trigger, and a trigger should be evaluated
against what the field has done since, not against what the field had done in 2026-05.

**The headline: Phase 3's null — crosscoder decoder directions align with `V` at
chance (0.484 / 0.501) — has a mechanical explanation in the dictionary-learning
literature, and it is not "no geometric structure". It is that the sparse objective
manufactures near-orthogonality. Phase 4 guessed this and called it "sparsity was the
confound"; the literature says it more strongly.**

---

## 1. The mechanical explanation for Phase 3's null

**[S]**, from the SAE-geometry pass:

> "For sparse coding, the matrix `Dᵀ D` (where `D` is the learned decoder matrix) is
> **approximately an N×N identity matrix after normalization**, meaning the model has
> learned a set of basis vectors where each column is **nearly orthogonal to all
> others**, indicating that the features are independent."

A dictionary whose columns are near-orthogonal *by construction* will align with any
fixed low-dimensional subspace — `U_pos`, `U_neg`, `V`'s top eigenvectors — **at
chance**. That is what 0.484 / 0.501 is. **Phase 3's null is close to a
theorem about its own instrument**, which puts it in the same class as Phase 2b's
withdrawn `rotation_neutral` (`lit-2b.md` §3): a measurement whose readout is
constrained by the intervention that produced it.

**This does not retract the finding**; it explains it, and it makes Phase 4's Track 3
result (dense low-rank AE recovers alignment: 33 bottleneck directions on
V-attractive for ALBERT against 0 for GPT-2) the *informative* arm of the pair. See
`archive/p4_mstate_features/lit-4.md`.

---

## 2. What the field has that Phase 3 did not

- **Crosscoders as a named architecture with a purpose Phase 3 would have wanted.**
  **[S]**: "sparse dictionary learning architectures designed to discover, track, and
  interpret **latent feature evolution across models, checkpoints, and tasks**,
  learning a unified sparse, interpretable feature space aligned across different
  models via paired encoder/decoder weights." **Across checkpoints** is the axis this
  project acquired after Phase 3 was archived.
- *Decoder-Preserving Sparse Autoencoders: Which Readouts Survive Sparse
  Compression?*, **2607.17425** **[N]** — the title is Phase 3's question in the
  field's vocabulary.
- *The Geometry of Concepts: Sparse Autoencoder Feature Structure* **[S]** — "crystals
  whose faces are parallelograms or trapezoids"; SAE dictionaries have measurable
  geometry, which is the premise Phase 3's null was taken to deny.
- *Same Concept, Different Directions: Cross-Modal Feature Heterogeneity in Sparse
  Autoencoders*, **2606.29888** **[N]**; *Empirical Insights into Feature Geometry in
  Sparse Autoencoders* (LessWrong) **[N]**.

---

## 3. The reintroduction trigger, re-evaluated

`FROZEN.md` holds the trigger; this file does not change it, but supplies the state
of the world it should be read against:

1. **The null is explained, not overturned.** Re-running the same measurement on a
   newer SAE would reproduce it for the same mechanical reason. **Do not reintroduce
   Phase 3 to re-test alignment at chance.**
2. **The question that is now live is the one `design-7.md` already wrote down** —
   *does the alignment gap between sparse and dense-low-rank dictionaries track how
   attractor-organised the layer is, on a checkpoint axis?* That is a prediction with
   a falsifier, and the crosscoder literature's own "across checkpoints" framing says
   the instrument for it exists.
3. **The standing rule is unchanged and binding**: no SAE/LRAE features in any
   measurement path (`core/DESIGN_dual_reading.md`). Under §2's framing the SAE is the
   object of study, which that rule never prohibited.
4. `archive/p3_crosscoder/steering.py` already implements a steering intervention with
   a merge-event readout and a recorded null. `design-7.md` says new steering work
   should be written having read it. **That instruction is now more valuable, not
   less**, given `p5b_manifold_steering/lit-5b.md`'s finding that the steering
   literature moved substantially in 2026.

---

## 4. Verification queue

1. The `Dᵀ D ≈ I` claim — find its actual source. It is the load-bearing sentence in
   this file and it came from a search summary.
2. **2607.17425** — *Decoder-Preserving Sparse Autoencoders*.
3. The crosscoder architecture's canonical reference (Anthropic circuits update or
   equivalent), for the across-checkpoints framing.
4. *The Geometry of Concepts* — what structure SAE dictionaries do have, so that
   "aligns at chance with `V`" is stated against the right alternative.

## 5. Search log (2026-09-16)

- `crosscoder sparse autoencoder decoder directions alignment weight matrix eigenvectors geometry features 2026`
- `superposition features low-variance directions residual stream not top principal components privileged basis interpretability`
