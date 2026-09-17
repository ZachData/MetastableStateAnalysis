<!-- archive/p4_mstate_features/lit-4.md -->
# Phase 4 — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; scholarly hosts are blocked by this session's egress proxy.
Marks: **[S]** search-engine summary read, **[N]** title and id only.

**This phase is archived.** Read with `archive/p3_crosscoder/lit-3.md`; they are one
result in two halves.

**The headline: Phase 4's Track 3 result is the informative half of the Phase 3/4
pair, and it is a direct empirical statement about a trade-off the dictionary-learning
literature is actively arguing over. It is the most reusable thing in the archive.**

---

## 1. The result, restated so its value is visible

Phase 3: sparse crosscoder decoder directions align with `V` at **chance**
(0.484 / 0.501, two models).
Phase 4 Track 3: **removing the sparsity penalty** — a dense low-rank autoencoder —
recovers the alignment, **33 bottleneck directions on V-attractive for ALBERT against
0 for GPT-2**.

`status-4.md`'s reading: *"sparsity was the confound, not absence of geometric
structure."*

**The literature supplies the mechanism** (`lit-3.md` §1): a sparse dictionary's
decoder columns are pushed toward mutual orthogonality (`Dᵀ D ≈ I` after
normalisation) **[S]**, so alignment with any fixed subspace is chance by
construction. Dropping the penalty removes that constraint and lets the dictionary
follow the activation geometry instead.

So the pair is not "SAEs failed, dense AEs worked". It is:

> **A sparsity penalty and a geometric-structure probe are in direct conflict, and the
> conflict is measurable: same data, same target subspace, 0 vs 33 directions.**

That is a clean, small, falsifiable statement with a controlled comparison behind it,
and the searches found no paper making it in that form.

---

## 2. Where the field is on this

- *Decoder-Preserving Sparse Autoencoders: Which Readouts Survive Sparse
  Compression?*, **2607.17425** **[N]** — the closest title found. If it asks which
  readouts survive the sparse constraint, it is asking Phase 4's question.
- *Disentangling Dense Embeddings with Sparse Autoencoders*, **2408.00657** **[N]**.
- *Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders*,
  **2411.13117** **[N]**.
- *Empirical Insights into Feature Geometry in Sparse Autoencoders* (LessWrong) **[N]**.
- The **ALBERT-vs-GPT-2 asymmetry** (33 vs 0) has no neighbour found. Phase 2's Regime
  A/B split (`lit-2.md`) puts ALBERT in Regime A (locally detectable, attention-
  dominant, shared weights). **Whether the AE result and the regime split are the same
  distinction has never been checked and is a one-line cross-tabulation.**

---

## 3. Directions to grow, if the phase is ever reintroduced

1. **Re-run the sparse-vs-dense contrast on a checkpoint axis.** `design-7.md`'s
   prediction — the alignment gap should *widen* as attractor structure forms — needs
   exactly Phase 4's two arms and Phase 1's trajectory. Neither existed together when
   Phase 4 ran.
2. **Cross the 33-vs-0 asymmetry against Regime A/B.** Free, and it would tell us
   whether the dictionary result is about architecture (shared weights) or about
   dynamics (locally detectable V-repulsion).
3. **State the trade-off as a benchmark.** "Fraction of dictionary directions
   recovering a known weight-space subspace, as a function of sparsity penalty" is a
   one-axis sweep that the SAE literature could use and that this project has already
   run at two endpoints. The sweep between them was never run.
4. **Rebuild, do not lift** (`archive/README.md` rule 2). `low_rank_ae.py` predates
   the particle schema.

---

## 4. Verification queue

1. **2607.17425** — decides whether direction 3 is occupied.
2. The `Dᵀ D ≈ I` source (shared with `lit-3.md`).
3. **2408.00657** — dense-vs-sparse framing.

## 5. Search log (2026-09-16)

- `crosscoder sparse autoencoder decoder directions alignment weight matrix eigenvectors geometry features 2026`
