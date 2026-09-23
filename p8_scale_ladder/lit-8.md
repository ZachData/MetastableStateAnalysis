<!-- p8_scale_ladder/lit-8.md -->
# Phase 8 — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**Supersedes** `archive/docs/literature_scan_2026-09-10.md` §3 for this phase, which said the
ladder's frame survives but flagged that 2606.02378 uses Pythia-1B. **It does, and so
does a second paper by the same group.** The rung policy needs an explicit amendment
and it is proposed in §3.

**The headline: the ladder's *question* is occupied — at least three 2026 papers ask
whether circuits are the same across models — and the answers so far are mostly
negative, which is the best possible news for a phase whose deliverable is "which of
these six invariants survive?".**

---

## 1. Who is on the ladder already

### 1.1 The group that is closest: `skydancerosel/spectral-probe-circuits`

Two papers, same correspondence address, same repository, both June 2026, both on
**Pythia 1B / OLMo 1B / OLMoE 1B-7B**:

- **2606.02378** *When Do Attention Circuits Form? Developmental Trajectories of
  Capability and Attention-Sink Emergence Across Three 1B-Class Architectures* **[S]**.
  10 log-spaced revisions per model, 30 runs. Identifies **induction, previous-token
  and BOS-attractor heads** with a **participation-ratio spectral signal** plus an
  **all-head capability-specific selectivity screen**. Findings given in the summary:
  - **Layers 0 and 1 produce zero BOS-classified heads at every revision in every
    model** — an architectural floor, not a learned outcome.
  - The whole-model BOS-attractor fraction has **three different emergence shapes** —
    gradual ramp in Pythia 1B (Pile, dense), **sharp phase transition in OLMo 1B**
    (7 % → 70 % between adjacent checkpoints), gradual ramp in OLMoE.
  - **In DCLM models, induction-circuit formation precedes BOS-attractor formation by
    10–20× in tokens.** "Capability-circuit formation and attention-sink formation are
    two transitions, not one."
- **2606.05378** *Pattern Selectivity is Not Task-Causal Structure* **[S]**. Four
  composed tasks × three models; **no two (task, model) pairs share the same primary
  causal screen at comparable effect size**; the IOI circuit's primary head type
  differs across GPT-2 small, Pythia 1B, OLMo 1B, OLMoE.

**This is the ladder, built, on one of our reserved rungs, by a group with public
code.** It is not a reason to stop; it is a reason to be precise about what is left.

### 1.2 The rest of the cross-model line

- *Architecture, Not Scale: Circuit Localization in Large Language Models*,
  **2605.08853** **[S]** — "the attention architecture matters more than parameter
  count, with **grouped query attention producing circuits that are far more
  concentrated and mechanistically stable** than standard multi-head attention at
  comparable scales."
- *Quantifying LLM Attention-Head Stability: Implications for Circuit Universality*,
  **2602.16740** **[S]** — "**middle-layer heads are the least stable yet the most
  representationally distinct**; deeper models exhibit stronger mid-depth divergence."
- *Mechanistic Analysis of Universality: Numerical Comparison Circuits Across
  Transformer Architectures*, ACL SRW / ICLR 2026 **[S]** — within-family consistency
  (Qwen), cross-family divergence; "task behaviour similarities **do not imply
  mechanistic universality**."
- *Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit
  Discovery*, **2606.06267** **[N]**.
- *Does circuit analysis interpretability scale? Evidence from multiple choice
  capabilities in Chinchilla* **[N]** — the older precedent.

---

## 2. Invariant by invariant

`design-8.md` lists six. Here is where each stands against the field.

| # | Invariant | Status |
|---|---|---|
| 1 | **Membership is a small, heavy-tailed set** (~4 substantial of 384; median head moves the readout by 0.001) | **Adjacent work exists** (head-importance distributions in the pruning literature, `lit-7e.md` §1.2), but the *shape-not-count* framing with a per-rung measured null is not found. **Keep** |
| 2 | **Members form in one narrow window coinciding with induction onset** | **Occupied.** 2606.02378 **[S]** has onset curves per model; **2511.16893** **[S]** has an *equation* predicting the formation point from batch size and context size. **Demote from invariant to calibration check** — and then use it as an external adjudication (§4.1) |
| 3 | **Ordering vs window — cascade or recruitment** | **The most valuable one, and still open.** No source found reports the *order* in which individual members of a redundancy set form, across independently-seeded models. 2606.02378 reports whole-model fractions and head classes, not member ordering. `design-8.md` already calls this "the one 7d question that always needed a second model" — **the literature agrees by omission** |
| 4 | **Born aligned, then fanning out** (CKA 0.693 vs null 0.128 at birth, peak at 5000, −56 % by 143000 while delta norms grow) | Nothing found. **Keep, and it is the second-best card** |
| 5 | **A low-rank majority with at least one full-rank anti-ordered member** | Rank-1-ness: **NOT NEW** (`lit-7e.md`). Anti-ordering: **looks new**. `design-8.md` calls #5 "the sharpest and most falsifiable"; that remains true **only for the anti-ordered half** |
| 6 | **One set, magnitude-dominated, direction-decoupled** | **Threatened by 2607.01940 (CoAx)** — see `lit-7d.md` §1.3. The *decomposition* (r² 0.74–0.81 magnitude vs 0.07–0.09 direction) may survive; the second-order instrument does not |

---

## 3. The rung policy needs an amendment, and here is the argument

`design-8.md`'s policy reserves **pythia-1b** on the grounds that it is "not in the
registry, never measured". That is a statement about **this project**, and it is still
true. But **1b is no longer unmeasured by the field** — 2606.02378 and 2606.05378 both
run it on the induction axis, with public code.

That changes what an adjudication on 1b *means*, in two opposite directions:

**Against reserving it.** A registered prediction adjudicated on 1b now lands on a
model whose induction-head inventory is partly published. If our prediction's content
is recoverable from their figures, the adjudication is weaker than it looks — the
spent-artifact rule exists to stop exactly this, and it does not currently notice
external spending.

**For reserving it — and this is stronger.** An external, independent measurement of
the same rung by a different group with different instruments is **the best
adjudication site the project has ever had**. A prediction registered *before* reading
their results, adjudicated on our own 1b measurement, and then cross-checked against
their published curves, is a three-way comparison: our prediction, our measurement,
their measurement. `claims/adjudications/` holds zero entries against 39
registrations; this is a route to a strong one.

**Proposed amendment, for the user to accept or reject — it is a human call and
`design-8.md` already says so for the related P-I7 question:**

> Add a rule 4 to the rung policy: **a rung may be externally spent.** Before
> registering a prediction against a reserved rung, record which external
> measurements of that rung exist and what they report, in the registration itself.
> Reserve the rung against *our* measurement as before; reserve the *registration*
> against reading the external results first.

**1.4b is not affected** — no source found measures Pythia-1.4b on the induction axis.
It is now the cleaner of the two reserved rungs and should be preferred for the first
registration.

---

## 4. Directions to grow

### 4.1 Use the formation-point equation as a free cross-rung prediction

**2511.16893** **[S]**: a simple equation in **batch size and context size** predicts
the induction-head formation point. Pythia's batch size and context length are
**constant across the entire suite** — same data order, same schedule, same 2048-token
context. So the equation predicts **the same formation step at every rung**.

That is a sharp, testable, external prediction and it is the cleanest thing the ladder
could run:

> **If formation step is constant across rungs and the equation says it should be, the
> invariant-2 window is a data-schedule fact, not a scale fact.** If it moves with
> scale, the equation is wrong for this suite — which is also a result, and a bigger
> one.

70m has already run (`status-8.md`); 410m is in 7d. **Two rungs of evidence exist
right now, with no new compute.** Do this before the next sweep.

### 4.2 Report invariant 3 (ordering) as the phase's headline

§2 says it is the one nobody has. It is also the one `design-8.md` sequences fourth,
against the sister project's ordered cascade
(3.6 → 3.1 → 4.6 → 4.7 → 3.0 → 3.5). **Promote it.** Cascade-vs-recruitment is a
mechanism question with a binary answer, two independent draws are already available
(published-checkpoint 70m and the dense retrain bracket), and the labelling discipline
for keeping them separate is already written down.

### 4.3 Turn "architecture, not scale" into a control, not a threat

**2605.08853** **[S]** says attention architecture dominates parameter count, and that
GQA concentrates circuits relative to MHA. **The whole Pythia suite is MHA.** So the
ladder is, by construction, a **scale sweep with architecture held fixed** — which is
precisely the control 2605.08853 says is missing from cross-family comparisons. This
is an argument *for* the Pythia-only decision that `design-8.md` currently justifies
only on convenience grounds. Write it down that way.

The corollary is a limit to state honestly: **any invariant that survives this ladder
is demonstrated for MHA at Pythia's data order, and nothing more.** Cross-family
generalisation is a separate claim the ladder cannot make.

### 4.4 Mid-depth instability is a prediction about where our invariants should break

**2602.16740** **[S]**: middle-layer heads are the least stable across instances, and
divergence grows with depth. 7d's members sit at layers 5, 7, 8, 8, 11, 12 of 24 —
**the middle**. So the field predicts that the *identity* of our members should be the
least portable thing about them, while the *set-level* statistics are more likely to
carry. That is exactly `design-8.md`'s "nothing transfers by head name", arrived at
independently — and it is now a citable expectation rather than a methodological
caution. It also predicts which invariants should survive: **4 and 6 (set-level)
before 3 (ordering, identity-dependent)**.

### 4.5 Separate the two transitions before measuring either

2606.02378 **[S]** found capability-circuit formation and attention-sink formation to
be **two transitions separated by 10–20× in tokens** on DCLM models. Pythia is Pile,
where the summary reports a gradual BOS ramp. **Our invariant-2 window is defined by
induction onset; if a sink transition sits nearby on Pythia, any set-level mean over
the window will mix them.** Measuring the BOS-attractor fraction alongside member
formation on 70m and 410m is cheap and it de-confounds the phase's second invariant.

---

## 5. Verification queue

1. **2606.02378** — the selectivity screen, the PR signal, the Pythia-1B curves, and
   the exact revision list. Needed by `lit-7d.md` §4.1 as well. **Top priority.**
2. **2606.05378** — the cross-model dissociation; how much of our developmental
   version is left.
3. **2511.16893** — the formation-point equation. Decides §4.1.
4. **2605.08853** — the architecture-over-scale claim, for §4.3's framing.
5. **2602.16740** — mid-depth instability, for §4.4.
6. The `skydancerosel/spectral-probe-circuits` repository — **code, not a paper**, and
   the fastest way to find out exactly what their screen does. GitHub may be reachable
   where arXiv is not.
7. *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling* —
   for the batch size / context length constants §4.1 needs.

---

## 6. Search log (2026-09-16)

- `induction head formation Pythia checkpoints developmental interpretability arXiv`
- `"When Do Attention Circuits Form" developmental trajectories attention sink emergence 1B-class architectures Pythia OLMo induction heads`
- `universality of circuits across model scales cross-model comparison attention heads same circuit different sizes scaling interpretability`
- `"Pattern Selectivity is Not Task-Causal Structure" composed-task circuits 1B-class cross-architecture ablation selectivity`
- `"Predicting the Formation of Induction Heads" batch size context size bigram repetition frequency Aoyama Wilcox`

Surfaced, not pursued: **2601.21996** *Mechanistic Data Attribution: Tracing the
Training Origins of Interpretable LLM Units* **[N]** — causally modulates the
emergence of interpretable heads in Pythia by adding/removing training samples, which
is an intervention on the formation axis this phase only observes; **2510.12071**
*Influence Dynamics and Stagewise Data Attribution* **[N]**; **2604.13694** *Weight
Patching* **[N]**; **2606.06267** *Many Circuits, One Mechanism* **[N]**.
