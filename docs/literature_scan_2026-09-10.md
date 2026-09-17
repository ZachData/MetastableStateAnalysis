<!-- docs/literature_scan_2026-09-10.md -->
# Literature scan, 2026-09-10 — what of §3.12-V is actually new

> **SUPERSEDED 2026-09-12 by `p8_scale_ladder/literature-8.md`, which fetched
> every id below and found three of this file's characterisations wrong.** Read
> that file, not this one, for any decision. Kept for provenance, and because
> its own verify-before-citing warning was the right instinct.
>
> **Corrections, in brief.** (1) No id here was confabulated — all nine resolve.
> (2) **The SVD follow-up chain does not support §1**: AdaSVD, QSVD and
> especially CARE (which is GQA→MLA conversion) say nothing about truncation
> ordering, so the anti-ordered member has *less* cover than §1 concluded, not
> more. (3) **`2607.01940` is adjacent, not a scoop** — no pairwise matrix, no
> geometry, no training or scale axis — so §103's "read this first" is wrong;
> read `2407.10827` instead, which this scan missed and which asks the phase's
> core question directly.

**Status: LEADS, NOT READINGS.** Four web searches were run and their result
summaries recorded. **No paper below has been read.** Every arXiv id here must be
verified before it is cited, argued against, or used to drop a line of work — a
search summary is not a source. This file exists so the next session starts from
"check these" rather than from "we may be first".

**Why it was run.** §3.12-V produced results that looked novel enough to build a
paper on, and the user asked the right question before that started: *just
because we found it doesn't mean other people haven't stumbled upon it.*

## Verdict, bluntly

**Three of the four headline findings sit in populated territory, and one of them
has a 2022 paper stating the general form.** The distinctive residue is narrower
than it looked this morning, and it is methodological more than phenomenal.

### 1. "SVD ordering is a poor proxy for causal importance" — **NOT NEW (2022)**

`FWSVD` / *Language model compression with weighted low-rank factorization*
(arXiv **2207.00112**, ICLR 2022) states the general form directly: SVD minimises
reconstruction error without gauging parameter importance, so its objective is
**not aligned with task accuracy**; Fisher-weighting the reconstruction error
gives *worse* reconstruction and *better* accuracy. Follow-ups (`AdaSVD`
2502.01403, `CARE` 2603.17946, `QSVD` 2510.16292) report **non-monotone
degradation when truncating singular values by magnitude** — the same shape as
`L11H14`'s curve.

**What may survive:** ours is *stronger than "poor proxy"*. FWSVD says
Eckart–Young is **suboptimal**; §3.12-V4 measured it as **anti-optimal** — bottom-`r`
beat top-`r` at every rank and rank-1 recovery was **negative**, worse than
deleting the head. And ours is a **causal-ablation recovery** measure on a single
circuit, not task accuracy after compression. Whether that gap is a paper or a
paragraph depends on what 2207.00112 and its follow-ups already show.

### 2. Redundancy / super-additive co-ablation — **NOT NEW, and the sign matches**

*The Hydra Effect: Emergent Self-repair in Language Model Computations*
(arXiv **2307.15771**, 2023) is the canonical result: ablate a head and others
grow in importance. **This predicts exactly the super-additivity 7d measured** —
if backups compensate for a single ablation, removing the pair together must cost
more than the sum of the singles. **44/45 super-additive cells is the self-repair
signature, not a new phenomenon.**

Worse for novelty: *Conditional Co-Ablation: Recovering Self-Repair Backups in
Transformer Circuits* (arXiv **2607.01940**, 2026) appears to be structurally the
same instrument as `pairwise_interaction_matrix.py`. **Read this one first** — it
is the closest thing to a direct scoop and it is three months old.

### 3. Developmental trajectories, decay, and turnover — **NOT NEW, and recent**

- Induction heads emerging near **step 1000 of 143000** in Pythia is reported;
  §3.12-U's window is consistent with published work, not ahead of it.
- *Which Attention Heads Matter for In-Context Learning?* (arXiv **2502.14010**)
  reports **FV heads appearing near step 16000** and, critically, that **a head's
  induction score falls as another score rises**. That is §3.12-U's
  "`L5H2`'s induction score falls twenty-fold while its causal effect goes
  +0.01 → +4.97" from the other side.
- *When Do Attention Circuits Form?* (arXiv **2606.02378**, 2026) tracks
  developmental trajectories across three 1B-class models including Pythia-1B at
  log-spaced revisions. **This is the ladder's territory and it is four months
  old.** Note it uses **Pythia-1B** — one of our reserved rungs.
- Reported: **"Pythia's decay-out heads go inert by the end (clean causal
  turnover)"** — heads decaying and being replaced. §3.12-V2's fan-out and
  §3.12-U's "four of six decay to 5–22 % of peak" are the same story.

### 4. Head-subspace geometry — **contested, needs reading**

*Measuring Affinity between Attention-Head Weight Subspaces via the Projection
Kernel* (arXiv **2601.10266**, 2026) is directly adjacent to §3.12-V2/V3. Whether
it handles the **anisotropy correction** — §3.12-V3 measured an ambient
participation ratio of **22 of 1024**, which makes the isotropic `k/d` baseline
useless — is the thing to check, and is where our method may still be ahead.

## What still looks distinctive after the scan

Ranked by how likely it is to survive contact with the actual papers:

1. **Membership defined causally across all 384 heads, with the demonstration
   that structural proxies fail.** §3.12-R (`‖OV‖_F` r² = 0.001, relation
   inverted), §3.12-G6 (no spectral field predicts causal effect), §3.12-V1
   (δ-cosine explains 7–9 % of interaction over 45 pairs). Most of the literature
   selects heads by induction score or attention pattern. **This is
   methodological and it is the strongest remaining card.**
2. **The measured-null discipline.** §3.12-V3's anisotropy result kills the
   isotropic baseline that an adjacent literature appears to use, and §3.12-V5's
   changing-membership artifact is a trap any developmental study of a *forming*
   set will hit. Both are transferable warnings.
3. **Heterogeneity within one functional set** — a rank-1, perfectly gain-ordered
   member (`L7H8`, 97 % from one direction) and a full-rank, anti-ordered one
   (`L11H14`) doing *the same job* and remaining causally substitutable while
   becoming geometrically orthogonal. The *contrast* is more interesting than
   either half.
4. **Anti-optimality (negative recovery)** as a strengthening of FWSVD's
   suboptimality — conditional on §1's check.

## What this changes about the plan

- **Do not build a paper on "induction heads are rank-1" or on "the redundancy
  set is super-additive."** Both are known.
- **Read 2607.01940 and 2606.02378 before the next measurement.** They are the
  two nearest neighbours and both are 2026.
- **The scale-ladder plan survives** — it was always about turning `n = 1` into a
  population claim, and 2606.02378 having done it for other properties is
  evidence the frame is right, not that it is taken. But **it uses Pythia-1B**,
  so check what it measured there before assuming that rung is untouched by the
  field.
- **The `lora_ind` merge case is unaffected**: its value was the dense onset axis
  and the φ question, neither of which this scan touched.
