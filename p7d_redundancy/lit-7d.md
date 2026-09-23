<!-- p7d_redundancy/lit-7d.md -->
# Phase 7d — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**Supersedes and extends** `archive/docs/literature_scan_2026-09-10.md` §2 and §3 for this
phase. That scan's verdict — super-additive co-ablation is the self-repair signature,
not a new phenomenon — **is confirmed and sharpened here**, and the nearest neighbour
is confirmed to exist and to be exactly what the scan feared.

---

## 1. The three papers that own this territory

### 1.1 *The Hydra Effect* — arXiv **2307.15771** (2023)

**[S]** Knock out an attention layer and another increases its effect to compensate;
the Hydra effect plus late-MLP downregulation **restores ~70 % of the logit reduction
at middle layers**; demonstrated on Chinchilla 7B **trained without dropout**, which
rules out dropout as the cause; corroborates earlier GPT-2-Small "backup heads" work.

**This predicts 7d's §3.12-S sign directly.** If a backup compensates for a single
ablation, removing the pair together must cost more than the sum of the singles.
**44/45 super-additive cells is what self-repair looks like**, and it is not a new
phenomenon. The 2026-09-10 scan said this; nothing found since softens it.

### 1.2 *Copy Suppression* — arXiv **2310.04625**, BlackboxNLP 2024

**[S]** Supplies a **mechanism** for self-repair that 7d can test rather than assume:
"if an initial overconfident copier is ablated, then there is nothing to suppress."
The negative-diagonal OV structure is weights-visible (84.70 % of tokens; 76.9 % of
L10H7's impact explained).

**This is the most useful thing in this file that is not a threat.** It converts "the
set is redundant" into a falsifiable *why*: if 7d's super-additivity is copy-suppression
self-repair, then the compensating heads should have **negative-signed OV structure on
the suppressed tokens** — a weight-space prediction, checkable for free, on heads whose
causal effects 7d already has. See §4.2.

### 1.3 *Conditional Co-Ablation (CoAx)* — arXiv **2607.01940**, July 2026

**[S]** **This is the near-scoop the 2026-09-10 scan named, and it is real.** Verbatim
from the summary:

> introduces Conditional Co-Ablation (CoAx), a **label-free, output-grounded score
> that measures how much each remaining unit's ablation effect grows once a primary
> set has been removed** … first-order ablation scoring becomes misleading when a
> transformer self-repairs … recasts this as **conditional circuit completion**, with
> CoAx exposing the **second-order interaction that single-unit scores discard**.

Authors: Zhiren Gong, Zihao Zeng, Chau Yuen, Wei Yang Bryan Lim. Submitted 2 July 2026.

`pairwise_interaction_matrix.py` computes the same second-order object.
**Read this before the next 7d measurement, not after.**

**What is plausibly still ours, stated as hypotheses to check against the paper, not
as claims:**

1. **The decomposition of the interaction.** 7d reports that **74–81 % of the
   interaction is the product of the two heads' own magnitudes** (r² 0.74–0.81) while
   **δ-cosine explains only 7–9 %** across 45 pairs. That is a statement about *what
   the interaction is made of*. CoAx, as described, is a **score** — it ranks units
   for recovery. A score and a variance decomposition are different deliverables.
2. **The full matrix over an independently-established causal set**, rather than a
   ranking from a seed set. 7d's membership came from a **384-head causal sweep**, not
   from a discovery heuristic.
3. **The checkpoint axis.** Nothing in the CoAx summary is developmental.
4. **The ceiling discipline.** `design-7d.md`'s rule — any interaction measured where
   the joint arm approaches `ln V = 10.83` is uninterpretable at raw `dNLL`, and the
   compression **biases toward apparent sub-additivity** — is a correctness condition
   on exactly CoAx's second-order quantity. If CoAx does not handle it, that is a
   methods note with our numbers behind it.

---

## 2. Finding by finding

| 7d finding | Nearest prior work | Verdict |
|---|---|---|
| Super-additive co-ablation across the set (44/45 cells) | **2307.15771** **[S]** | **NOT NEW.** It is the self-repair signature |
| Second-order / conditional ablation as the right instrument | **2607.01940** **[S]** | **NOT NEW as of July 2026** |
| Structural proxies fail: `‖OV‖_F` r² = 0.001 and **inverted** (§3.12-R); no spectral field predicts causal effect (§3.12-G6) | Nothing found reporting a *negative* result of this form. Head-importance-by-weight-norm is standard in the **pruning** literature (Voita et al.; *Automatic Channel Pruning for Multi-Head Attention* **2405.20867** **[N]** — "head importance … by accumulating the absolute value of attention output elements") | **Looks new, and it contradicts a standard practice.** The pruning literature's importance scores are the proxies 7d measured as failing |
| **The behavioural proxy fails too, and is inverted for `L5H2`** (induction score falls 20× while causal effect goes +0.01 → +4.97) | **2502.14010** *Which Attention Heads Matter for In-Context Learning?* **[S]** (2026-09-10 scan) reports a head's induction score **falling as another score rises**; **2606.02378** **[S]** identifies heads with a **selectivity screen** | **PARTIAL, and it is our sharpest live disagreement.** 2606.02378 selects induction heads *by behavioural selectivity*, which §3.12-U says is the wrong instrument. See §4.1 |
| Weight-space overlap and function-space overlap come apart (write subspaces at chance 0.222 vs 0.250 while residual effects 87 % aligned) | *Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel*, **2601.10266** **[N]** (2026-09-10 scan); and **[S]** from the 2d pass: "gauge-invariant bilinear form comparison, centered Gram matrix analysis (kernel PCA) … for measuring **head independence**" | **Contested. Read 2601.10266.** Our anisotropy correction (ambient PR 22 of 1024) may be the part that survives |
| One set, no block structure, magnitude-dominated, direction-decoupled | Nothing found | **Looks new**, conditional on §1.3 |
| Formation of five of six members inside `(512, 2000]` | Induction onset near step 1000 in Pythia is published (**2606.02378** **[S]** and others) | **NOT NEW as a window.** The *per-member ordering* is the open part — invariant 3 in `design-8.md` |

---

## 3. The state of the phase's novelty, bluntly

Of 7d's five questions, **Q1 (how many members) and Q2 (when) sit in populated
territory**, Q3 (as early as possible) is answered by the published onset, and the
distinctive residue is:

- **Q4/Q5 — same way or different ways; one class or several** — the heterogeneity
  result, which is 7e's territory and where the anti-ordered member lives.
- **The negative methodological results**: structural proxies fail, behavioural
  proxies fail and invert, and causal membership is the only defensible definition.

`archive/docs/literature_scan_2026-09-10.md` ranked that second item first among survivors.
**This scan agrees and raises it**, because §4.1 turns it from "our method is better"
into "a June 2026 paper's head-selection method is measurably wrong on our model."

---

## 4. Directions to grow

### 4.1 The selectivity-screen disagreement — the strongest card in the phase

**2606.02378** **[S]** identifies induction / prev-token / BOS-attractor heads by a
**participation-ratio spectral signal plus a capability-specific selectivity screen**,
across 10 revisions on three 1B-class models, and reports emergence curves from it.
§3.12-U measured, on pythia-410m, that **`L5H2`'s induction score falls twenty-fold
across the interval in which its causal effect rises from +0.01 to +4.97.**

> **A screen that selects by induction score would drop `L5H2` at precisely the
> checkpoint where it becomes the set's largest causal member.**

That is a concrete, falsifiable, checkable claim about a published method, and 7d has
the data to make it.

**But the thesis is partly taken, and this must be said before the card is played.**
*Pattern Selectivity is Not Task-Causal Structure: A Cross-Architecture Mechanistic
Study of Composed-Task Circuits in 1B-Class Language Models*, arXiv **2606.05378**
**[S]**, June 2026 — **same group as 2606.02378** (same correspondence address, same
`skydancerosel/spectral-probe-circuits` repository). It "tests whether identifying
attention-head circuits by **task-pattern selectivity** and verifying through **causal
ablation** produces consistent mechanistic claims across model families", and finds
that **no two (task, model) pairs share the same primary causal screen at comparable
effect size**; the IOI circuit's primary head type differs across GPT-2 small, Pythia
1B, OLMo 1B and OLMoE.

So "pattern selectivity is not task-causal structure" is **published, in those words,
by the authors of the paper §4.1 proposes to contradict.** What remains ours is the
*axis*, and it is a real difference:

| | 2606.05378 | §3.12-U / this proposal |
|---|---|---|
| dissociation shown across | **models** (four tasks × three models) | **training time**, within one model |
| ground truth | causal ablation on a final checkpoint | causal ablation at **every checkpoint** |
| claim | the circuit selectivity finds is not portable | **selectivity inverts against causal effect during formation**, worst exactly when the circuit is forming |
| unit | primary head per (task, model) | a **redundancy set** of six members |

**The developmental inversion is the surviving half and it is sharper than the
cross-model one**, because it names *when* the screen fails and predicts that every
developmental study using a selectivity screen — including 2606.02378, by the same
group — reads its onset curves off a misidentified head set in the formation window.
That is still worth doing. It is no longer a novel thesis; it is new evidence for an
existing thesis on an axis its authors did not use. The experiment is: **run their screen's logic (selectivity, and
PR of head output) on our 384-head causal sweep, and report the confusion matrix
against causal membership, per checkpoint.** Every input is on disk.

If it holds, the result is a methods paper with a clear thesis: *behavioural screens
misidentify members of a redundancy set during formation, and the error is
systematic — worst exactly when the circuit is forming, which is when developmental
studies look.* That is more valuable than any of 7d's phenomenological findings, and
none of the scoops touch it.

**Check first:** whether their screen uses induction score alone or a conjunction that
`L5H2` would still pass. The PR half may rescue it, which would be a different and
also interesting result.

### 4.2 Test copy suppression as the mechanism for our super-additivity

§1.2. Weights-only, free, and it turns a known phenomenon into a specific mechanism
claim about *our* set. If the compensating heads show the negative-diagonal signature,
7d's redundancy is copy-suppression self-repair and should be written that way; if
they do not, we have super-additivity **without** the field's standard mechanism, which
is a more interesting result than the super-additivity itself.

### 4.3 Bring the ceiling discipline to CoAx's quantity

§1.3 item 4. If **2607.01940** scores second-order interactions on a readout with a
`ln V` ceiling and does not correct for it, its scores are biased toward sub-additivity
in exactly the regime where self-repair is strongest. `design-7d.md` already has the
argument and §3.12-M5 has the measurement. **This is a short, sharp, verifiable
contribution and it requires reading one paper.**

### 4.4 Check the formation-point equation

**2511.16893** **[S]** predicts the IH formation step from batch size and context size
(see `lit-7.md` §4.1). 7d has the measured window. Free external adjudication.

### 4.5 Measured nulls, not isotropic ones — keep saying it

§3.12-V3's ambient participation ratio of **22 of 1024** makes `k/d` useless as a
chance baseline for subspace overlap. Whether **2601.10266** handles this is the open
question the 2026-09-10 scan already flagged and it is still open.

---

## 5. Verification queue

1. **2607.01940** (CoAx) — score vs decomposition; seed-set vs full matrix; ceiling
   handling; any developmental axis. **Highest priority in the project.**
2. **2606.02378** — the exact selectivity screen. Decides §4.1.
2b. **2606.05378** — *Pattern Selectivity is Not Task-Causal Structure*. **Read with
   2606.02378; same group.** Decides how much of §4.1 is left.
3. **2310.04625** (*Copy Suppression*) — the self-repair mechanism in full. Decides §4.2.
4. **2307.15771** (*Hydra*) — the ~70 % restoration figure and how it was measured, for
   a like-for-like comparison against our 2.2× parts-sum.
5. **2601.10266** (projection kernel) — the anisotropy question.
6. **2502.14010** — the falling-induction-score-while-other-rises result, for §4.1's
   framing.
7. **2606.09607** *Closure-Validated Circuit Discovery in Attention Heads:
   Co-activation Proposes, Ablation Disposes* **[N]** — surfaced under the 2d search;
   the title is 7d's methodological split verbatim.
8. **2608.03629** *Cross-Layer Interaction under Weight-Space Ablation* **[N]**;
   **2608.22007** *The Communication Map of a Transformer* **[N]**.

---

## 6. Search log (2026-09-16)

- `Conditional Co-Ablation Recovering Self-Repair Backups Transformer Circuits 2607.01940`
- `Hydra effect self-repair backup attention heads redundancy ablation compensation language models 2026`
- `"When Do Attention Circuits Form" developmental trajectories attention sink emergence 1B-class architectures Pythia OLMo induction heads`
- `copy suppression head negative eigenvalues OV circuit McDougall anti-copying attention head GPT-2 10.7`
- `attention head merging pruning consolidate multiple heads into one refit OV weights model editing redundant heads`
- `universality of circuits across model scales cross-model comparison attention heads same circuit different sizes scaling interpretability`
- `"Pattern Selectivity is Not Task-Causal Structure" composed-task circuits 1B-class cross-architecture ablation selectivity`
