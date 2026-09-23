<!-- p7e_consolidation/lit-7e.md -->
# Phase 7e — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**Extends** `archive/docs/literature_scan_2026-09-10.md` §1 (the SVD-ordering question), which
was the sharpest part of that scan and is the sharpest part of this phase.

**The headline: the compression literature says SVD ordering is *suboptimal* and has
said so since 2022. `L11H14` is *anti-optimal* — bottom-`r` beats top-`r` at every
rank and rank-1 recovery is negative. Nothing found reports that, and the gap between
"suboptimal" and "anti-optimal" is where this phase's contribution lives. The
consolidation half, by contrast, sits next to a large and old literature.**

---

## 1. The two literatures this phase touches

### 1.1 SVD ordering vs causal importance — populated since 2022

- **FWSVD**, *Language model compression with weighted low-rank factorization*,
  arXiv **2207.00112**, ICLR 2022 **[S]**. Verbatim from the summary: *"SVD minimizes
  the squared error toward reconstructing the original matrix **without gauging the
  importance of the parameters**, potentially giving a larger reconstruction error
  for those who affect the task accuracy more … the optimization objective of SVD is
  **not aligned with the trained model's task accuracy**."* Fisher-weighting the
  reconstruction metric fixes it.
- Follow-ups, all **[N]** unless marked: **GFWSVD** *Generalized Fisher-Weighted SVD*
  **2505.17974** **[S]** (diagonal-FIM is too crude; Kronecker-factored corrections);
  **SVD-LLM V2**; **WSVD 2604.02570**; **QSVD 2510.16292**; **CARE 2603.17946**;
  *Low-Rank Prehab: Preparing Neural Networks for SVD Compression* **2512.01980**.

**What that literature establishes:** the Eckart–Young objective is the wrong
objective for function preservation, and re-weighting it helps. **What it does not
appear to establish:** that the ordering ever *inverts* — that keeping the smallest
singular directions beats keeping the largest, at every rank, and that rank-1
truncation is worse than deleting the component outright.

**Three differences that have to hold for 7e's result to be a contribution**, and each
is checkable only by reading:

| | compression literature | 7e |
|---|---|---|
| unit | whole weight matrix / whole model | **one attention head's OV** |
| readout | downstream task accuracy after compression | **causal-ablation recovery of the head's own `dNLL` effect** |
| baseline | uncompressed model | **matched-norm random directions** |
| finding | suboptimal, non-monotone under magnitude truncation | **anti-optimal: bottom-`r` > random > top-`r` at every rank; negative at `r = 1`** |

The **matched-norm random control** is the one the compression literature has no
reason to run, and it is what makes "anti-optimal" a measurement rather than a
description. `L11H14`'s curve sits *below* that control at nearly every rank.

### 1.2 Head pruning and merging — old, large, and adjacent to the consolidation half

**[S]** from the head-merging pass: head importance computed "by accumulating the
absolute value of attention output elements of each head"; *Analyzing Multi-Head
Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned*
(Voita et al.) **[N]**; *Automatic Channel Pruning for Multi-Head Attention*
**2405.20867** **[N]** (a **reweight module compensates for information loss** after
channel removal); *Merging Text Transformer Models from Different Initializations*
**2403.00986** **[N]** ("maintaining head structure in the permutation while allowing
different head correspondences … is the most optimal permutation"); *Pruning via
Merging: Compressing LLMs via Manifold Alignment Based Layer Merging* **2406.16330**
**[N]**; *Ensembling Pruned Attention Heads* **2510.18358** **[N]**.

**"Remove a head and re-fit a survivor to compensate" is structurally what
pruning-with-reweighting does.** 7e must not present the intervention as novel. What
is different, and must be defended on these grounds:

- The objective is **`min ‖resid_full − resid_(B ablated, A refit)‖_F` subject to
  `rank ≤ D_HEAD`** — matching the *residual stream*, not the task loss. Pruning
  re-weights to recover accuracy; 7e re-fits to reproduce an internal state.
- The survivor must do it **through its own QK pattern at its own layer**, which the
  pruning literature never asks because it prunes within a layer.
- **The falsifier is the interaction matrix going flat** (`design-7e.md` check 3),
  not the loss staying low. That is the claim's real content and it is a 7d-derived
  test with no counterpart in the pruning literature.
- The purpose is **interpretability**, not efficiency — and `design-7e.md` already
  states the limit correctly ("success shows the redundancy was *removable*, not that
  the original was secretly simple").

---

## 2. Finding by finding

| 7e finding | Nearest prior work | Verdict |
|---|---|---|
| SVD order is a poor proxy for causal importance | **2207.00112** **[S]** and its whole follow-up line | **NOT NEW (2022)** |
| Non-monotone degradation under magnitude truncation | QSVD / CARE / AdaSVD **[N]** (2026-09-10 scan) | **NOT NEW** |
| **Anti-optimality: bottom-`r` > matched-random > top-`r` at every rank; `r = 1` recovery negative (−0.096)** | Nothing found | **Looks NEW**, conditional on §1.1's three differences surviving a read |
| Five of six members effectively low-rank (`r*` = 1, 1, 2, 12, 24 of 64); `L7H8` recovers 97 % from one direction | Rank-1-ness of induction/copying heads is folklore and partly published; the 2026-09-10 scan says **do not build a paper on "induction heads are rank-1"** | **NOT NEW** |
| **The contrast** — a rank-1, gain-ordered member and a full-rank, anti-ordered member doing the same job, causally substitutable while geometrically orthogonal | Nothing found | **Looks NEW**, and the 2026-09-10 scan ranked it third among survivors. This scan agrees |
| Consolidation by re-fitting a survivor's OV | Pruning-with-reweighting (§1.2) | **NOT NEW as a technique; the objective and the falsifier are ours** |
| The ambient-budget result — the set writes **355 ambient directions** to reach 90 % of joint effect against an ambient PR of **22**, i.e. into **private low-variance bandwidth** | Nothing found. It is the inverse of the "features live in the top variance directions" assumption most subspace methods make | **Looks NEW, and under-exploited.** See §4.3 |
| Energy is not usefulness (73–78 % of joint energy fits in 64 dims but rank ≠ function) | Implicit in FWSVD's whole premise | **Same insight, independently reached** |

---

## 3. What survives

1. **Anti-optimality**, if §1.1's three differences hold. It is a *strengthening* of a
   known result, which is a normal and publishable kind of contribution — but only if
   stated as a strengthening, with FWSVD cited, not as a discovery.
2. **The heterogeneity contrast.** Two members, same job, opposite rank structure,
   opposite SVD-ordering behaviour, causally substitutable, geometrically orthogonal.
   Nothing found has this.
3. **The ambient-budget inversion.** §4.3.
4. **The methodological hold** — *no `r*` from an `svd` basis should be quoted for a
   head not checked with `--bottom`* — which reaches outside this phase into
   `induction_rank_sweep`'s `r*` construction and into anyone else's.

---

## 4. Directions to grow

### 4.1 Run the `schur` vs `svd` basis comparison. It is free and it is still unrun.

`PROJECT.md`'s resume block and `design-7e.md` both name it: the `schur` basis orders
by **eigenvalue** rather than gain and **carries a sign**, so it may not inherit the
defect. Weights-only, no forward passes. Until it is done, a stated hold sits on an
instrument the project uses. **Cheapest item in this file and it has been cheap for a
week.**

### 4.2 Put `L11H14` against a Fisher-weighted ordering

FWSVD's fix is to re-weight the reconstruction metric by parameter importance. If a
Fisher-weighted (or gradient-weighted) ordering of `L11H14`'s OV **recovers the
correct order**, then `L11H14` is an extreme instance of FWSVD's phenomenon and the
result is "a case where the 2022 correction is not merely helpful but necessary".
If it **does not** — if even the importance-weighted ordering is anti-informative —
then the head is outside what that literature explains, and *that* is the paper.
**This is the experiment that decides how big the finding is.** Requires gradients,
which is the only non-free item here.

### 4.3 The private-bandwidth result deserves its own measurement

355 ambient directions to reach 90 % of joint effect, against an ambient participation
ratio of 22. Stated as: **the induction set writes mostly into directions the residual
stream barely uses.** The searches found no counterpart, and it bears directly on two
live literatures — superposition (features in low-variance directions) and
subspace-overlap methods (which use isotropic or top-PC baselines and would call these
directions noise). `design-7e.md` treats it as a failed prediction that the phase
survived; it should be treated as a finding. **The follow-up is causal:** truncate the
joint effect to the ambient top-`k` and to the effect's own top-`k`, and show the
`dNLL` curves separate. That is `useful_rank.py` with a different basis.

### 4.4 Consolidation: position it against pruning from the start

Write the intervention as *"pruning-with-reweighting, but the objective is residual-
stream reconstruction and the success criterion is that the interaction matrix goes
flat"*. `design-7e.md`'s check 3 is the novel part and it should lead.

---

## 5. Verification queue

1. **2207.00112** (FWSVD) — does it or any follow-up report an *inversion* rather than
   suboptimality? Decides whether §3.1 is a contribution.
2. **2505.17974** (GFWSVD) — the Kronecker correction; the natural comparator for §4.2.
3. **2512.01980** (*Low-Rank Prehab*) **[N]** — the title suggests preparing a network
   so SVD works better, i.e. the failure mode is known enough to pre-empt.
4. **Voita et al.**, *Specialized Heads Do the Heavy Lifting* — the canonical head
   pruning citation 7e must carry.
5. **2405.20867** — the reweight module, against our re-fit objective.
6. **2601.10266** (projection kernel) — for the orthogonality claim about `L11H14`.
7. Superposition literature — **searched, inconclusive.** The pass returned the
   canonical sources (*Toy Models of Superposition*, **transformer-circuits.pub/2022**
   **[S]**; *Sparse Autoencoders Find Highly Interpretable Features* **2309.08600**
   **[N]**) and one directly relevant fact **[S]**: *"residual stream basis directions
   are not found to be any more interpretable than random directions"*, with only a
   "weak coordinate privilege" in the residual stream. That says nothing about
   **variance** ordering. **The specific question — do useful directions concentrate
   in high-variance directions? — was not answered by any result found**, so §4.3
   stands as plausibly unoccupied but unproven. The SAE "dark matter" / low-norm-
   feature line is the place to look next and was not reached.

---

## 6. Search log (2026-09-16)

- `SVD truncation singular value magnitude poor proxy importance Fisher weighted low-rank factorization non-monotone degradation attention head`
- `attention head merging pruning consolidate multiple heads into one refit OV weights model editing redundant heads`
- `superposition features low-variance directions residual stream not top principal components privileged basis interpretability`
- (inherited) `OV circuit eigenvalue spectrum attention head copying score negative eigenvalues positive eigenvalues induction mechanistic interpretability`
