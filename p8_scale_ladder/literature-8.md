<!-- p8_scale_ladder/literature-8.md -->
# Phase 8 — LITERATURE: what is already published, and what it hands us

**Status: READINGS, not leads.** Every arXiv id in §1–§5 was **fetched** and its
abstract read; the ones that were not are quarantined in §6.2 and must not be
cited until they are. This is the standard
`docs/literature_scan_2026-09-10.md` set for itself and could not meet — that
file recorded nine ids from four search summaries with **no paper opened**, and
three of its characterisations turn out to be wrong (§4).

**Run 2026-09-12**, under `CLAUDE.md`'s trigger 1: a phase is open and
`design-8.md`'s six invariants are not yet registered, so this is the last
moment the literature can still change what gets built.

**The good news first, because the previous scan's framing was pessimistic and
partly wrong: none of the nine ids was confabulated, the feared "direct scoop"
is not one, and the invariant the scan talked us out of is the one with the
least cover.**

---

## 1. The phase's headline question is answered, and the replacement is better

**`2407.10827` — *LLM Circuit Analyses Are Consistent Across Training and
Scale* (2024).** Decoder-only models **70M–2.8B over 300B tokens** — the Pythia
ladder's exact span. Task abilities and their functional components emerge at
similar token counts across scale, and — the sentence that matters —
**components "may be implemented by different attention heads over time, [but]
the overarching algorithm that they implement remains."**

`design-8.md`'s core question is *"do the structural signatures 7d/7e found
reappear at other scales"*. **That question has been asked and answered
affirmatively, two years ago, on our ladder.** It cannot be the phase's
headline. This displaces `2607.01940` as the must-read before the next
measurement.

**What it does not do is account for the substitution.** It establishes that
heads turn over while the algorithm persists, and says nothing about what the
turnover looks like structurally — no rank, no subspace geometry, no
alignment trajectory. **Invariant 4 is exactly that missing account, and it is
the one card that is both unclaimed and replicates (§2).** So the reframing:

> **Not** "do induction signatures recur across scale" — settled.
> **But** "when the implementing heads turn over while the algorithm persists,
> what is the geometry of the substitution?"

That is a narrower claim, it inherits `2407.10827` as support rather than
competition, and the two surviving invariants answer it directly.

---

## 2. The six invariants, scored against the literature

| # | Invariant | Verdict |
|---|---|---|
| 1 | Membership is a small, heavy-tailed set | **Replication.** Induction circuits are reported small (3–11 heads, 124M→1B); head importance is repeatedly power-law (70–90 % of BERT heads removable). The *shape* is known. |
| 2 | Members form in one narrow window at induction onset | **Replication.** Circuits reach final size within the first 6–25B tokens; `2502.14010` puts induction heads at ~step 1000/143000 on Pythia — our own window. |
| 3 | Ordering vs window (cascade vs recruitment) | **Partially anticipated.** `2407.10827` establishes *that* heads turn over with the algorithm intact. The cross-seed **ordering** test is still unrun by anyone found here. |
| 4 | Born aligned, then fanning out | **OPEN, and it replicates — the load-bearing card.** Nearest is `2602.16740` (attention-head stability: mid-depth heads least stable, deeper models diverge more), which is **cross-seed representational** similarity — not within-model developmental alignment of a causally-defined set whose effect sizes *grow* as alignment decays. |
| 5 | Low-rank majority + ≥1 full-rank anti-ordered member | **OPEN in the literature, but it does NOT replicate.** Nothing found in either search lane, and §4 shows the scan's claimed cover does not exist — so the claim is unclaimed. It also **fails at 70m** (below). |
| 6 | One set, magnitude-dominated, direction-decoupled | **Split.** Super-additive co-ablation *is* the Hydra effect (`2307.15771`) and is not new. The **decomposition** — r²(interaction, `d_a·d_b`) = 0.74–0.81 against r²(interaction, δ-cosine) = 0.07–0.09 — was not found anywhere. |

**Novelty and replication are different axes and must not be collapsed.** This
table scores *novelty only*. `status-8.md` scores replication, and on invariant
5 the two disagree in the most informative possible way:

| | novel? | replicates across rungs? |
|---|---|---|
| invariant 4 | **yes** | **yes** (`status-8.md`: 2 and 4 replicate) |
| invariant 5 | **yes** | **no** — 70m has no low-rank majority; three of its four above-noise heads need the full 64, against 410m's five of six at `r* ≤ 12` summing to 48. The non-replication "survives every check applied to it". |

**So invariant 4 is the phase's load-bearing result** — unclaimed *and* it holds
at a second scale. **Invariant 5 is an unclaimed negative**: nobody else has
made the claim, and our own ladder says it is a property of **pythia-410m, not
of induction**. That is exactly the discrimination the phase was built to make,
and it is publishable as a negative — but it must never be written as though it
were a cross-scale property.

**Read in one line:** invariants 1, 2 and the super-additive half of 6 are
replication; 3 is half-taken; **invariant 4 is the real content, invariant 5 is
a clean unclaimed negative, and 6's magnitude/direction split is untouched
territory.**

---

## 3. The four methodological cards

| Card | Verdict |
|---|---|
| Measured rather than isotropic nulls | **GENUINELY OPEN — strongest card by a clear margin** |
| Probe out-of-distribution artifacts | **Appears open**; cleanest small contribution |
| Causal membership, structural proxies fail | **Partially covered**; only the *inversion* is unclaimed |
| Ablation-mode artifacts | **Phenomenon TAKEN**; residue is the scaling only |

### 3.1 Measured nulls — open, and it is a correction to a published method

**`2601.10266` — *Measuring Affinity between Attention-Head Weight Subspaces
via the Projection Kernel* (Yamagiwa, Takase, Shimodaira, Jan 2026).** Abstract
**and full text** read. Its informativeness measure is a KL divergence against
"a reference distribution derived from **random orthogonal subspaces**",
derived analytically. A full-text search returns **no** mention of anisotropy,
isotropy, participation ratio, or effective dimension, and **no**
acknowledgement that a full-ambient random baseline overstates significance
when representations are effectively low-dimensional.

That is our nearest methodological neighbour using **exactly the baseline that
§3.12-V3's ambient participation ratio of 22 of 1024 invalidates.** This is
stronger than "unclaimed" — it is a live correction to a published method.

**Check before writing it that way:** they work at `d = 768` (GPT-2 scale) and
our anisotropy is measured on pythia-410m at `d = 1024`. **Confirm the
participation ratio transfers to their setting** — if it does not, the critique
narrows to our own scale.

### 3.2 Probe OOD — open, with a twist worth knowing

The repeated-random-token protocol is the field standard (`2209.11895`,
prefix-matching over ~50 random tokens). **The twist: the field treats the
OOD-ness as a deliberate feature** — random tokens avoid confounds from natural
token statistics — and already calls prefix-matching a proxy. What was **not**
found is our actual failure mode: at small scale and late training the language
prior beats the copy mechanism, reading as an induction collapse that repeated
natural text (99.2 % top-1) shows is not there. Scale- and training-dependent,
so it sits naturally in the ladder frame.

### 3.3 Structural proxies fail — reframe before leaning on it

The general principle is established in two literatures: attention-as-
explanation (Jain & Wallace 2019, Serrano & Smith 2019 — **leads, §6.2**) and
pruning, where magnitude's poorness as an importance proxy is *why*
activation-aware and gradient-based scores exist.

The useful find is where the spectral literature **stops**: **`2602.13524` —
*Singular Vectors of Attention Heads Align with Features* (Feb 2026)**
establishes spectral↔feature alignment and **does no causal ablation at all.**
So the spectral-structure literature reaches the edge of causal validation and
halts there.

Also **`2606.09607` — *Closure-Validated Circuit Discovery: Co-activation
Proposes, Ablation Disposes*** (Pythia 1B, OLMo 1B, OLMoE) argues "a cheap
signal is a circuit proposal, not a confirmed circuit", including a proxy whose
real signal fails ablation in the wrong direction. **Same argument as §3.12-R /
§3.12-G6, different evidence** — theirs is co-activation clustering, ours is
that *spectral and norm* fields fail.

**Honest read:** a quantified instance of a principle the field already
believes. **The inverted relation (`‖OV‖_F` r² = 0.001 with the sign backwards)
is the only genuinely unclaimed part.**

### 3.4 Ablation mode — the phenomenon has its own paper now

**`2604.14433` — *Zero-Ablation Overstates Register Content Dependence in DINO
Vision Transformers* (Apr 2026).** An entire paper on §3.15's finding:
zero-ablation's large drops reflect distributional bias, not genuine
dependence; mean- and noise-substitution stay within ~1pp of baseline. **ViTs,
not LMs**, and it makes **no cross-size scaling claim.**

So *"zero-ablation is off-distribution and overstates importance"* is
established and now **citable rather than ours**. The surviving residue is
narrow: **the `1/n_heads` scaling, and the consequence that the bias does not
cancel across rungs.** Caveat-sized — but it lands precisely in the ladder
frame, so it is well-placed rather than wasted.

---

## 4. Corrections owed to the 2026-09-10 scan

**No id was confabulated.** All nine resolve to real papers with essentially
the claimed titles. The concern that motivated re-checking them was wrong, and
that is worth recording as plainly as a hit would have been. Three
*characterisations*, however, do not survive:

1. **The SVD follow-up chain does not say what the scan said.** `2502.01403`
   (AdaSVD) is error-compensation plus layer-wise ratios; `2510.16292` (QSVD)
   is joint-QKV compression for VLMs; **`2603.17946` (CARE) is GQA→MLA
   conversion and has nothing to do with truncation ordering.** **None reports
   non-monotone degradation under magnitude truncation.** Only FWSVD's
   *suboptimality* claim stands against invariant 5's measured
   *anti-optimality*. **The scan talked us down from the invariant that has the
   least cover.**
2. **`2607.01940` (CoAx) is adjacent, not a scoop.** Both search lanes agree
   independently: it is a set-conditional importance score for
   attribution/pruning, primarily GPT-2-small IOI with induction transfer
   across eight models (124M–7B), ROC-AUC 0.33→0.91 for backup recovery. It has
   **no pairwise interaction matrix, no geometric or directional analysis, and
   no training-checkpoint or scale axis.** Downgrade from "read this first".
3. **The reserve protects pythia-1b from us, not from the field — confirmed and
   worse than stated.** Pythia-1B appears in `2606.02378` (verified: 10
   log-spaced revisions × 3 models), in `2606.09607`, and inside
   `2407.10827`'s range.

The scan's conclusion 3 (developmental trajectories are populated) **survives
and sharpens**: `2502.14010` (Yin & Steinhardt, ICML 2025) is confirmed in
detail — Pythia, induction heads ~step 1000, FV heads ~step 16000, induction
scores showing a "sharp initial rise followed by plateau or slight decline" as
FV rises, and **many FV heads start as induction heads.** Nuance worth keeping:
**"slight decline" is not §3.12-U's twenty-fold fall.**

---

## 5. What the literature hands us — six extensions

The point of the scan, per the user: *if someone derived something similar,
learning from them is more powerful than not knowing.* Ranked by value.

1. **Run `2502.14010`'s function-vector score on our members.** They report
   induction heads *becoming* FV heads, with induction score declining as FV
   rises. §3.12-U measured `L5H2`'s induction score falling twenty-fold **while
   its causal effect went +0.01 → +4.97** — which may be exactly an
   induction→FV transition seen from the causal side, and would explain a
   result currently filed as a puzzle. **Concrete, cheap, and the single most
   valuable thing this scan produced.**
2. **Apply `2601.10266`'s projection kernel with our measured null.** Their
   method is better instrumented than ours for subspace affinity; our null is
   better than theirs. Combining them is a real contribution to both and a
   natural collaboration or citation-with-correction (subject to §3.1's
   transfer check).
3. **Run CoAx's conditional score along our checkpoint axis.** `2607.01940` has
   the set-conditional instrument and **no developmental axis**; we have 19
   checkpoints and the runners. Neither project has the product, and it is a
   measurement rather than a reinterpretation.
4. **Use `2606.02378`'s Pythia-1B revisions as a published baseline.** If its
   10 log-spaced revisions are public, the reserved rung stops being virgin
   territory and becomes **a rung with a published trajectory to adjudicate
   against** — which is better for `claims/adjudications/` than a cold start.
   Check availability before registering a 1b prediction.
5. **Cite `2407.10827` as the frame, not the competitor.** Its
   turnover-with-invariant-algorithm result is the phenomenon invariants 4/5
   supply a mechanism for. Stating that relationship up front is stronger than
   discovering it in review.
6. **Note the architecture threat, and that the ladder is clean on it.**
   `2605.08853` (*Architecture, Not Scale*, Pythia + Qwen2.5) argues GQA
   produces far more concentrated and mechanistically stable circuits than MHA
   at comparable scale. **Pythia is all-MHA, so our ladder isolates scale
   cleanly — a strength to state explicitly.** The threat is that architecture
   may be the more interesting axis, which is a future arm, not a fix.

---

## 6. Verification ledger

### 6.1 Fetched — citable

`2407.10827`, `2607.01940`, `2606.02378`, `2606.09607`, `2605.08853`,
`2604.14433`, `2602.16740`, `2602.13524`, `2602.17532`, `2601.10266`,
`2601.04398`, `2603.17946`, `2510.16292`, `2502.14010`, `2502.01403`,
`2307.15771`, `2209.11895`, `2207.00112`, `2606.08292`.

### 6.2 Leads — NOT fetched, do not cite

`2605.24059` (Spectral Probe-Circuits, same author as `2606.02378`),
`2306.11695` (Wanda), `2602.04491` (Greedy-Gnorm), and the pre-2020
attention-as-explanation papers (Jain & Wallace 2019, Serrano & Smith 2019).
These are the same standing the whole 2026-09-10 file had, and they get the
same treatment: verify before use.

### 6.3 The uncovered window

**Nothing was found published after 2026-09-10**, the date of the last scan;
the newest hits are Jun–Aug 2026. **Treat that window as thin rather than
clear** — absence here is weak evidence, since the search was not designed to
sweep a six-week window exhaustively.

---

## 7. What this changes about the plan

* **Reframe the phase around the substitution's geometry** (§1). "Signatures
  recur across scale" is `2407.10827`'s result, not ours.
* **Sequence invariant 4 first**, not invariant 1. `design-8.md` opens with
  invariant 1 on 70m because it is cheapest; the literature says it is also the
  most replicative. **Invariant 4 is the only card that is both unclaimed and
  replicates**, so it is where the next rung's effort belongs.
* **Do not drop invariant 5, and do not oversell it either** (§2, §4.1). The
  old scan's SVD cover for it does not exist, so the claim is unclaimed — but
  our own 70m rung says it is a **410m property, not an induction property**.
  Write it as the negative it is.
* **Register the methodological cards in the order §3 ranks them**, not the
  order the old scan did: measured nulls first, probe-OOD second, and the
  proxy-failure card only in its narrow *inverted-relation* form.
* **Stop treating `2607.01940` as urgent.** Read `2407.10827` instead.
* **Before any 1b work**, check whether `2606.02378`'s revisions are public
  (§5.4). It changes what registering a 1b prediction is worth.
