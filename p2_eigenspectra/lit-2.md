<!-- p2_eigenspectra/lit-2.md -->
# Phase 2 — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**The headline: the eigenvalue sign structure of the OV circuit is not an
under-examined quantity — it is the canonical copying-head statistic from the 2021
transformer-circuits framework, and the negative-eigenvalue case has its own
well-known paper. Phase 2 has been measuring a famous object under a different name
and for a different purpose, and has never cited either source.**

---

## 1. The prior work Phase 2 should have been citing from the start

### 1.1 Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021)

`transformer-circuits.pub/2021/framework` **[S]**. The summary is unambiguous:

> Elhage et al. (2021) propose using the **number of positive real eigenvalues of the
> full OV circuit matrix `W_E W_OV W_U`** as a summary statistic for detecting
> **copying heads**. […] the fraction of positive eigenvalues for each head can be
> calculated as `Σᵢ λᵢ / Σᵢ |λᵢ|`.

Phase 2's core object is the **mixed-sign eigenspectrum of composed OV**, split into
`U_pos` / `U_neg` projectors and read as attractive/repulsive in the particle
dynamics. **That is the same spectrum.** The differences are real and worth stating
precisely, because they are the whole of Phase 2's remaining distinctiveness:

| | Elhage et al. | Phase 2 |
|---|---|---|
| matrix | `W_E W_OV W_U` — **token space** | composed OV — **residual space** |
| reduction | count / fraction of positive real eigenvalues | signed **subspace projectors**, then project Δx onto them |
| purpose | classify a head as copying | explain **energy-monotonicity violations** in a particle flow |
| axis | fixed trained model | 27 Pythia checkpoints |

**The "attractive/repulsive" split is the "copying/anti-copying" split in a different
basis.** Saying so out loud is a gain, not a loss: it means every Phase 2 statement
about repulsive subspace mass has a mechinterp translation, and it means Phase 7's
translation table has one entry already grounded.

### 1.2 McDougall et al., *Copy Suppression* — arXiv **2310.04625**, BlackboxNLP 2024

**[S]** GPT-2 L10H7's OV circuit is "dominated by negative values on the diagonal when
projected to the token unembedding basis", with **84.70 % of tokens having strong
negative self-connections**, and the weights-based account explains **76.9 %** of the
head's impact.

This is the repulsive case, studied as a single head, with a causal budget attached.
Two consequences:

1. **Phase 2's repulsive subspace has a named occupant.** Any claim that repulsive
   structure is exotic must survive the fact that a whole head of it is documented.
2. **Copy suppression is a self-repair mechanism** — the summary states it directly:
   "if an initial overconfident copier is ablated, then there is nothing to suppress."
   That is a mechanism for `p7d_redundancy`'s super-additivity and it is
   weights-visible. **Cross-link: `p7d_redundancy/lit-7d.md` §2.**

### 1.3 Weight-matrix spectra generally

- *The Singular Value Decompositions of Transformer Weight Matrices*, AlignmentForum
  **[N]** — the SVD-interpretability line Phase 7e's `useful_rank.py` is in tension
  with.
- *Spectral Selection in Symmetric Self-Attention Dynamics*, **2604.26085** **[N]**.
- *Fingerprint, Not Blueprint: How Positional Schemes Set the Default Spectral
  Algebra of Attention*, **2607.06621** **[N]** — appeared under both the Phase 1 and
  Phase 2 searches; positional scheme → spectral algebra. Pythia is RoPE, so this
  bears on every spectral quantity in this phase.

---

## 2. Finding by finding

| Phase 2 finding | Nearest prior work | Verdict |
|---|---|---|
| OV eigenvalue sign structure is the object of interest | Elhage et al. 2021 **[S]** | **NOT NEW.** Canonical since 2021 |
| ~98 % of OV eigen-dimensions are complex | Elhage et al. **[S]** ("attention heads can have complex eigenvalues … indicating amplifying effects along with rotation"); *Self-Attention as a Kernel Machine* **[S]** | **NOT NEW.** See `lit-2b.md` §1 |
| Negative/repulsive spectral mass matters causally | *Copy Suppression* **2310.04625** **[S]** | **NOT NEW** |
| **Two regimes: locally-detectable vs globally-coherent-FFN-mediated** (Regime A / B), with a continuous V-score combining three tests | Nothing found | **Looks new** |
| **Rescaled frame `z = e^{−tV}x` as a causal intervention in representation space**, scored by whether violations disappear | Nothing found. The nearest published intervention family is ablation/patching, which removes a component rather than re-coordinatising the space | **Looks new**, and is the phase's most distinctive instrument |
| `ov_norm_partial_rho` — controlling for OV-norm spikes before attributing displacement to the repulsive subspace | Nothing found under that name, but it is the same confound `p7d_redundancy` §3.12-R quantified (`‖OV‖_F` explains 0.1 % of causal effect) | **Ours, and it recurs** |
| Classifier reachability collapse on a parallel-residual architecture (`mixed_or_unattributed` means something different on Pythia) | Not a literature question | Internal |

---

## 3. What survives, and the one thing that is in danger

### 3.1 Survives: the rescaled frame

Nothing found does a **global change of coordinates** as a causal test. The field's
causal vocabulary is ablation, patching, and steering — all of which **remove or
inject**. `e^{−tV}` neither removes nor injects; it asks whether a *different frame*
makes the phenomenon disappear, which is a test of whether the phenomenon is
coordinate-dependent. That is a genuinely different move and it is Phase 2's.

**But it is inert on Pythia** (`status-2.md`, and `math-2.md` §10 per `math-5b.md`'s
note). An instrument that is distinctive and non-functional is worth nothing; fixing
it is worth more than any new measurement in this phase.

### 3.2 In danger: everything stated in token-space terms

If a Phase 2 claim can be restated as "this head copies / anti-copies", it is
Elhage 2021 or McDougall 2023 and must be cited as such. The claims that cannot be so
restated are the ones about **displacement geometry in residual space at violation
layers** — those are the keepers.

---

## 4. Directions to grow

**Corrected 2026-09-16, after this file's first draft.** Directions 1 and 2 below
originally proposed computing the Elhage statistic and putting it on the checkpoint
axis. **Both were already done**, in `PROJECT.md` §3.12-O
(`tools/run/copying_score_sweep.py`, all 384 heads, eight checkpoints, weights only,
with the `ov_factors` transpose trap checked rather than assumed). §3.12-N4 had
already identified the statistic and named it. The corrected directions are below;
the error was mine and it is recorded rather than quietly fixed.

1. **Cross the copying score against `frac_repulsive`.** This is the part §3.12-O did
   **not** do, and §3.12-N4 says why it matters: the copying score is **token-basis**
   (`W_E W_OV W_U`), `frac_repulsive` and the `U_pos`/`U_neg` projectors are
   **residual-basis**, the two differ by the vocabulary round-trip `W_U W_E`, and
   **nothing guarantees they share a sign**. O crossed the score against the *causal*
   readout and found them running in opposite directions. **Both columns are on disk.**
   If they agree, Phase 2's projectors get independent validation and a translation
   into the field's vocabulary. If they disagree, it is a dissociation **between two
   weight-space measures** — a third instance of the project's recurring pattern
   (§3.12-R, §3.12-S) and the sharper outcome, because it would mean "repulsive" and
   "anti-copying" are not the same claim. **Cheapest high-value item here.**
2. **Carry `L11H14` through both.** It is the **top copier at 143000 (+0.723)**
   (§3.12-O2) *and* 7e's full-rank anti-ordered outlier. The head whose singular
   directions do not order its causal usefulness is the one the token-basis measure
   ranks first. Whatever direction 1 finds, this head is where it will be sharpest.
3. **Diagnose the rescaled frame on Pythia before anything else.** §3.1.
4. **Check the RoPE spectral-algebra paper (2607.06621).** If positional scheme sets
   the default spectral algebra, then "98 % complex" may be a RoPE fact rather than a
   learned fact, and the random-weight control for it is a rotary-only null — which
   `core/qk_offset_null.py` already implements for the QK side.
5. **Re-run the two-regime (A/B) split on Pythia natively.** `decompose.py` is frozen;
   the parallel residual makes `Δx = attn_out + ffn_out` exact. The A/B distinction
   was the organising frame for Phases 3–5 and it has never been tested on the
   architecture where the decomposition is clean. `design-2.md` already says this;
   the literature adds no reason to delay it.

---

## 5. Verification queue

1. **`transformer-circuits.pub/2021/framework`** — the exact definition of the
   positive-eigenvalue statistic and the basis it is computed in. Confirms §1.1's
   table, and confirms that §3.12-O's `copying_score_sweep.py` implements the same
   statistic.
2. **2310.04625** — how much of L10H7's effect the negative diagonal explains, and
   the self-repair argument in full (needed by `lit-7d.md` too).
3. **2607.06621** — whether the complex fraction is positional-scheme-determined.
4. **2604.26085** — "spectral selection" in symmetric self-attention dynamics; how it
   relates to `U_pos`/`U_neg`.
5. The SVD-of-weight-matrices AlignmentForum post — for `lit-7e.md`'s anti-ordering
   result as much as for this phase.

---

## 6. Search log (2026-09-16)

- `OV circuit eigenvalue spectrum attention head copying score negative eigenvalues positive eigenvalues induction mechanistic interpretability`
- `complex eigenvalues attention weight matrices non-normal rotation antisymmetric component transformer analysis`
- `copy suppression head negative eigenvalues OV circuit McDougall anti-copying attention head GPT-2 10.7`

Surfaced, not pursued: **2405.00208** *A Primer on the Inner Workings of
Transformer-based Language Models* **[N]** (a survey — likely the fastest way to find
what else this phase has re-derived); **2501.18666** *Structure Development in
List-Sorting Transformers* **[N]**; **2510.25013** *Emergence of Minimal Circuits for
Indirect Object Identification in Attention-Only Transformers* **[N]**.
