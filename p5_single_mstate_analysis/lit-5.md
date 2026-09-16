<!-- p5_single_mstate_analysis/lit-5.md -->
# Phase 5 — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; scholarly hosts are blocked by this session's egress proxy.
Marks: **[S]** search-engine summary read, **[N]** title and id only.

**Phase 5's code is archived** (`archive/p5_single_mstate_analysis/`); the study notes
`math-5.md` and `math-5c.md` live here. This file covers the *method*, because the
method is what the project kept.

**The headline: "reconstruct one object end to end and cross-reference every framework
against it" is the dominant method of mechanistic interpretability, and it has a
well-known failure mode that the field named in 2026 — `n = 1` circuits do not
generalise across inputs or models. Phase 5's real contribution is not the narrative;
it is §9, the failure-mode taxonomy it produced by being the place where the project
learned to name silent defects.**

---

## 1. The method's literature

The single-object deep dive is the field's standard form:

- *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2
  small* **[N]** — the canonical instance.
- *Copy Suppression: Comprehensively Understanding an Attention Head*, **2310.04625**
  **[S]** — one head, end to end, with a weights-based account explaining 76.9 % of its
  effect. This is Phase 5's form done well, and it is worth reading as a template for
  what "comprehensively understanding" is held to.
- *Sparse Feature Circuits* **[N]**; *Does circuit analysis interpretability scale?
  Evidence from multiple choice capabilities in Chinchilla* **[N]**.

**And the critique, which is recent and directly about `n = 1`:**

- *Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit
  Discovery*, **2606.06267** **[N]** — the title says the problem: which circuit you
  find depends on the inputs you probe with and the granularity you evaluate at.
- *Pattern Selectivity is Not Task-Causal Structure*, **2606.05378** **[S]** — **no two
  (task, model) pairs share the same primary causal screen** (see `lit-8.md` §1.1).
- *Quantifying LLM Attention-Head Stability*, **2602.16740** **[S]** — mid-layer heads
  are the least stable across instances.

Phase 5's own selection discipline anticipated part of this. `select_cluster.py` scores
six gates, and `math-5.md` §1.1 records the stability check — each model's runner-up
shares the same prompt and scores within 0.3 points. **That is an
input-variation robustness check, run in advance, on a phase the field would now
criticise for `n = 1`.** It is worth saying out loud.

---

## 2. What Phase 5 actually contributed, and it is not the narrative

`math-5.md` §0 is blunt: three of six investigation groups blocked by producer/consumer
mismatches, an effective-β estimator carrying a known indexing bug, and a
cluster-selection score with **two of six criteria silently contributing zero** —
including the merge criterion at weight 3.0 of 9.0.

The cause is recorded exactly: `load_phase1_run` reads the wrong one of Phase 1's two
event schemas and normalises it into a shape with `layer_from` as a string and no
`merges` key, so every consumer asking "which merge did this trajectory participate
in" gets `None` **silently, forever**.

**That is a reusable finding about interpretability pipelines**, and the searches found
no paper on it. The class is:

> **A scoring function whose terms can silently evaluate to zero will still return a
> ranked list, and the ranking will look like a selection.**

It is the same shape as `lit-2b.md` §3 (an intervention invariant under its own
readout) and `lit-1.md` §3.1 (a metric confounded by one token's norm). **Three
instances from three different phases is a paper about instrumentation, and this
project is unusually well placed to write it** — see `docs/LITERATURE.md` §4.

---

## 3. Directions to grow

1. **Do not reintroduce Phase 5 as a narrative.** The field's 2026 critique
   (**2606.06267**, **2606.05378**) is aimed precisely at single-object accounts, and
   the project's own 7d finding (a *set* of six functionally redundant heads, not a
   circuit) is the same critique from the inside.
2. **Reintroduce the per-particle table instead.** `math-5.md` §8's v2 reframe —
   the object of study is the per-particle table, with clustering as an annotation —
   is what `p7_motifs` built on, and it survives everything in §1.
3. **Write §9 up.** The failure-mode taxonomy is the phase's asset.
4. **If a single-object study is wanted again, hold it to 2310.04625's standard:**
   a weights-based account with a stated fraction of the effect explained. Phase 5
   produced no such number.

## 4. Verification queue

1. **2606.06267** — the input-variation result, in full.
2. **2310.04625** — the "76.9 % of impact explained" methodology.
3. **2606.05378** — see `lit-7d.md` §4.1.

## 5. Search log (2026-09-16)

- `universality of circuits across model scales cross-model comparison attention heads same circuit different sizes scaling interpretability`
- `copy suppression head negative eigenvalues OV circuit McDougall anti-copying attention head GPT-2 10.7`
