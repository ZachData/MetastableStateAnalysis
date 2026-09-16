<!-- p6_subspace/lit-6.md -->
# Phase 6 — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; scholarly hosts are blocked by this session's egress proxy.
Marks: **[S]** search-engine summary read, **[N]** title and id only.

**Phase 6 is being rebuilt live** (`subspace_geometry.py`, `r2_r4_null.py`, with
tests), against the archived ALBERT-only partial run. Its stated premise has been
withdrawn (`math-6.md` §0) and its hypothesis survives independently.

**The headline: Phase 6 asks what the rotational subspace *does*. The field's answer,
for a different rotation, is *position* — and Pythia's RoPE puts a large, known,
analysable rotation in the QK circuit. The phase has been asking about `A` in the OV
circuit while the biggest rotation in the model sits one circuit over, unexamined.**

---

## 1. The division-of-labour hypothesis, against what is known

`math-6.md` §1: Phase 2b answered *does rotation drive clustering?* (no — and
`lit-2b.md` shows that "no" was an orthogonal-invariance identity), and left open
**what the rotational subspace carries.** The searches return one dominant answer for
rotational structure in transformers:

**[S]** — "In models using **Rotary Positional Embedding (RoPE)**, positional
information is not added initially but is **injected during attention via rotation**.
Constant-frequency rotations operate on query and key subspaces."

And the S/A framing that `lit-2b.md` §1 found: `A ∈ 𝔰𝔬(d)`, learning `d/2`
simultaneous rotation planes each with its own angular frequency **[S]**.

**Those two are the same mathematical object in two different circuits** — RoPE's
fixed, hand-designed rotation in QK, and OV's learned antisymmetric part. Nothing
found compares them. That comparison is Phase 6's opening, and it is also the sister
project's **φ question** (`design-8.md`: *does the antisymmetric fraction differ
between the QK and OV halves?*), which `design-8.md` calls "the genuine conceptual
overlap" and which is unanswered.

---

## 2. Adjacent work

- *Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network
  Topology*, **2605.14258** **[N]** — the title is Phase 6's subject.
- *Fingerprint, Not Blueprint: How Positional Schemes Set the Default Spectral Algebra
  of Attention*, **2607.06621** **[N]** — **positional scheme determines spectral
  algebra.** If true, `math-6.md`'s channel decomposition inherits a RoPE-specific
  prior that an ALBERT-era design could not have known about.
- *Self-Attention Dynamics with Rotary Position Embeddings: Twisted States and Explicit
  Consensus Rates on the Sphere*, **2607.24502** **[N]** — RoPE inside the
  particle-dynamics framework this project uses. **The most on-target unread paper for
  this phase.**
- Jane Street, *Using group theory to explore the space of positional encodings for
  attention* **[N]** — group-theoretic treatment of the same rotations.
- **[S]**: "principal component analysis of the residual stream reveals that individual
  attention heads and MLPs inhabit **low-dimensional subspaces with highly specialized
  principal axes**" — the division-of-labour hypothesis, stated positively, by someone
  else.

---

## 3. The rebuild's own design choice is the defensible one

`subspace_geometry.py` reports **alignment relative to chance** rather than raw
alignment, and `r2_r4_null.py` uses a random-subspace null stronger than the archived
version's. Against `lit-7d.md` §4.5 and §3.12-V3 — ambient participation ratio **22 of
1024**, which makes `k/d` useless as a chance value — that choice is not a refinement,
it is a correctness requirement. **Any paper in this area using an isotropic baseline
is exposed**, and *Measuring Affinity between Attention-Head Weight Subspaces via the
Projection Kernel* (**2601.10266** **[N]**) is the one to check first.

---

## 4. Directions to grow

1. **Answer the φ question.** Compute the antisymmetric fraction for the QK and OV
   halves, per head, per layer, per checkpoint. Weights-only, free, and it is the
   phase's hypothesis in its cheapest testable form. It also closes an open item the
   sister project handed over.
2. **Read 2607.24502 before designing anything else.** RoPE in the particle framework
   is Phase 6's question with the theory already attached.
3. **Test 2607.06621's claim directly.** If positional scheme sets the spectral
   algebra, then `core/qk_offset_null.py`'s **rotary-only null (N1)** is exactly the
   control that separates "learned rotation" from "RoPE's rotation" — and that null is
   already built, for Phase 7. Point it at Phase 6's channels.
4. **Do not re-run the archived 0/6 predictions as stated.** `math-6.md` records
   0 of 6 tested predictions passing and 6 of 12 never run, on one model, with a
   withdrawn premise. The rebuild should re-derive its predictions from §1's
   comparison, not inherit them.
5. **The unresolved LDA-alignment inversion** (0.887 imaginary vs 0.067 real
   repulsive, ALBERT) is the archive's one live anomaly with two explanations. If the
   rotational channel carries position, a third explanation becomes available and is
   testable on a RoPE model: **the imaginary channel is linearly decodable because
   position is linearly decodable.**

## 5. Verification queue

1. **2607.24502** — RoPE + sphere dynamics. **Top priority for this phase.**
2. **2607.06621** — positional scheme → spectral algebra.
3. **2605.14258** — residual-stream spectral geometry.
4. **2601.10266** — projection-kernel affinity, for the chance-relative baseline.
5. The `M = S + A` / `𝔰𝔬(d)` source (shared with `lit-2b.md` §5.1).

## 6. Search log (2026-09-16)

- `rotational subspace attention carries positional information division of labour real imaginary channel residual stream function`
- `complex eigenvalues attention weight matrices non-normal rotation antisymmetric component transformer analysis`
