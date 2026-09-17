<!-- p2d_operator_activation/lit-2d.md -->
# Phase 2d — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**The headline: Phase 2d's D1 — "which heads are outside the gradient-flow
hypotheses, and do the violations localise there?" — was the right question, and as
of 2026 it has a theorem written for the multi-head case that states the hypotheses
exactly as D1 measures them. 2d stops being an extension of a single-head survey and
becomes the empirical arm of a specific recent result. That is an upgrade.**

---

## 1. The result D1 is now testing

*Gradient Flow Structure and Quantitative Dynamics of Multi-Head Self-Attention*,
arXiv **2605.04279** **[S]**. From the summary:

> interprets transformer self-attention as a **gradient flow on the unit sphere** …
> **Score Symmetry (Condition 3):** the score matrix is symmetric, `M_h = M_hᵀ` for
> all `h` … **Value Alignment (Condition 4):** `W^{V,h} = M_h` for all `h`, which
> identifies the aggregation with the energy gradient … **Under score symmetry and
> value alignment, the total energy is non-decreasing along both flat and sphere
> dynamics.**

Set that beside `design-2d.md`:

> §3.4 makes (SA) a gradient flow in the reweighted metric … **only when `QᵀK` is
> symmetric and `V = QᵀK`**. Heads meeting both must show monotone `E_β`; heads far
> outside carry no guarantee at all. … Measured per head: asymmetry
> `‖Skew(M_h)‖_F / ‖M_h‖_F`, and the signed Frobenius cosine between `W_OV` and `M_h`.

**These are the same two conditions**, and `asymmetry` and `align` are the natural
continuous relaxations of Conditions 3 and 4. The differences worth keeping:

- 2605.04279 is **multi-head**; the survey condition 2d cites is single-head. Since
  2d measures per head and aggregates, the multi-head statement is the correct
  referent and the aggregation question (`design-2d.md`'s "what this cannot settle")
  may be answerable from the paper's own energy functional rather than by choosing
  mean/min/max.
- The paper names a **"radial shadow obstruction"** — geometric interference between
  heads — which the summary flags as a challenge the paper addresses. **This is a
  named obstruction to exactly the aggregation 2d does not know how to do.** Read it.

### 1.1 And one condition that may be prior to both

`lit-1.md` §1.3 records **2411.04990** *Clustering in Causal Attention Masking* **[S]**:
the causally-masked system **cannot be interpreted as a mean-field gradient flow**.
Pythia is decoder-only. If that is right, then on our model the gradient-flow
structure is absent *before* symmetry or value-alignment is asked about, and D1's
framing — "which heads are outside the hypotheses" — has a prior answer: **all of
them, by the mask.**

**This does not kill D1**; it reframes it. The measurable question becomes whether
asymmetry and misalignment still *predict violation locations* even though no head
can satisfy the sufficient condition. A predictor that works without its theorem is a
weaker but still real result — and one nobody has reported.

**Ordering consequence:** read 2411.04990 before running D1, not after.

---

## 2. Finding by finding

| Phase 2d sub-experiment | Nearest prior work | Verdict |
|---|---|---|
| **D1** — per-head asymmetry and `V`-vs-`M` alignment as predictors of energy violation | **2605.04279** **[S]** states the conditions; **2411.04990** **[S]** says the mask voids the structure | **Question NOT NEW; the measurement is.** No source found measures these quantities on a trained model's heads |
| D1's `repulsive_aligned` regime (`align ≈ −1`, where violation is *predicted*) | The `V = −I_d` case is in 2312.10794 §3.2/§9.1 **[S]**; *Copy Suppression* **2310.04625** **[S]** is the empirical instance | **Well-grounded; cite both** |
| Alignment reported against `M` and `Sym(M)` separately | Nothing found | **Looks new**, and it is the distinction Condition 3 forces |
| **D2** — operator-conditioned rank `PR_M = (tr MC)² / tr(MᵀCMC)`, pairing operator spectrum against activation covariance | Closer than first thought. A targeted pass **[S]** found (a) PR of a head's **per-example attention output** used as "a measure of content-dependent computation … in prior probe-circuit work", and (b) "attention-to-target selectivity **and participation ratio** … to measure when attention head classes emerge during training" — i.e. **2606.02378**'s instrument (see `lit-8.md`). Also **[S]**: "gauge-invariant bilinear form comparison, centered Gram matrix analysis (kernel PCA), and variance-explained interpretation for measuring **head independence**" | **PARTIAL.** PR-of-output is in use; the **operator-conditioned** pairing `tr(MᵀCMC) = Σₐᵇ λₐλᵇ|⟨uₐ, Muᵇ⟩|²` was not found. The gap is narrower than `design-2d.md` assumes and must be verified before the claim is made |
| `coupling_efficiency = PR_M / PR_C` | Nothing found | **Looks new** |
| **D3** — Table 1's limit geometry (trimodality normal to `φ₁`) as a testable prediction, with the `⟨Qφ₁, Kφ₁⟩ > 0` half restored and a rescaling scan over `t` | Nothing found testing Table 1 empirically | **Looks new.** Also the most likely to return a null |
| **D4** — the model's own `E_β^{(h)}` vs the identity-weight proxy, counting violations that **disappear** under the head's own operator | **2605.04279** **[S]** supplies a multi-head energy functional; using it as a *corrected baseline for previously-counted violations* is not something the searches found | **Looks new in use** |

---

## 3. What survives

**D2 is the phase's strongest and least-occupied idea.** The field measures operator
spectra (Elhage, the SVD line) and activation spectra (effective rank, RankMe,
intrinsic dimension, matrix-based entropy — see `lit-1.md` §2) **separately** — but the
targeted pass weakened this. Participation ratio **of a head's output activations**
is already a working instrument in the developmental-interpretability line
(`lit-8.md`, **2606.02378**), and someone is comparing **bilinear forms** across heads
for independence. `design-2d.md`'s sentence — *"Phase 2 has the left factor, Phase 1
has the right, and nobody has computed the product"* — is now **conditional on one
distinction**: PR-of-output measures how spread a head's *output* is; `PR_M` measures
how the *operator* pairs against the *input* covariance, `tr(MᵀCMC) = Σₐᵇ
λₐλᵇ |⟨uₐ, M uᵇ⟩|²`. Those are different quantities and only the second can
separate "the head is selective" from "the cloud is low-rank". **Establish that
distinction against 2606.02378 before claiming the product is uncomputed.**

The reading it buys — *heads with large `‖M‖` and small `PR_M` are strong operators
pointed where the tokens are not* — is a **selectivity measure that is neither a
weight norm nor an activation statistic**, and §3.12-R's result (`‖OV‖_F` explains
0.1 % of causal effect, backwards) is precisely the failure a pairing quantity is
supposed to fix. **D2 is the one instrument in the project that could rescue a
structural proxy for causal effect**, and 7d gave up on structural proxies without
having tried it.

That connection is not in any current doc and is the most actionable thing in this
file. See `p7d_redundancy/lit-7d.md` §4.

---

## 4. Directions to grow

1. **Run D2 against 7d's causal sweep.** 7d has causal effect for all 384 heads of
   pythia-410m; D2 is weights + one covariance. **Does `PR_M` (or
   `coupling_efficiency`) predict causal effect where `‖OV‖_F` failed?** If yes, the
   project has a structural screen it currently believes impossible, and the
   claim is sharp: *operator norm fails, operator-activation pairing succeeds.*
   If no, that is a stronger version of §3.12-G6's null and worth stating.
   **Highest-value item in this file, and it needs no new forward passes beyond a
   covariance already on disk.**
2. **Read 2411.04990 before D1** (§1.1). The mask may have answered D1's framing
   question in advance.
3. **Read 2605.04279's "radial shadow obstruction"** for the per-head→per-layer
   aggregation problem `design-2d.md` flags as possibly unresolvable.
4. **Sequencing note stands but its reason has changed.** `design-2d.md` says 2d
   waits on 1c-B because `T_eff ≪ t*` would make the asymptotic energy argument
   non-binding. Add a second reason: if 2411.04990 removes the gradient-flow
   structure entirely, the energy argument is not merely non-binding but
   inapplicable, and D4's "violations that disappear under the head's own energy"
   becomes the *only* well-posed member of the four.
5. **D3 is the one to descope if time is short.** It tests a limit geometry for an
   unobservable `t` on a fixed-depth network, its control arm is more likely to fire
   than its treatment arm, and no external work makes it urgent.

---

## 5. Verification queue

1. **2605.04279** — Conditions 3 and 4 verbatim; whether the energy functional is
   per-head or joint; what the radial shadow obstruction is.
2. **2411.04990** — whether causal masking voids the gradient-flow structure in the
   sense D1 needs.
3. **2312.10794** (Bulletin of the AMS 62(3), 2025) — §3.4 and Table 1 / §9.2 in the
   *published* numbering, since D3 is built on a table number from the preprint.
4. **2310.04625** — for the `repulsive_aligned` regime's empirical anchor.
5. **2606.19249** *Transformer Geometry Observatory TGO-I: Spectral Geometry
   Observatory* **[N]** — a name that suggests a measurement suite overlapping this
   phase wholesale. Unread, and it should not stay unread.

---

## 6. Search log (2026-09-16)

- `"Gradient Flow Structure and Quantitative Dynamics of Multi-Head Self-Attention" symmetric value matrix condition energy`
- `participation ratio operator conditioned covariance trace bilinear pairing attention head selectivity effective rank operator`
- `effective inverse temperature beta attention softmax estimate trained transformer measure integration time depth`
- (inherited) `Clustering in Causal Attention Masking arXiv 2411.04990 decoder-only self-attention dynamics`
- (inherited) `OV circuit eigenvalue spectrum attention head copying score negative eigenvalues positive eigenvalues induction mechanistic interpretability`

Surfaced by the D2-targeted pass, not pursued: **2606.09607** *Closure-Validated
Circuit Discovery in Attention Heads: Co-activation Proposes, Ablation Disposes*
**[N]** (belongs to `lit-7d.md` — co-activation vs ablation is 7d's exact
methodological split); **2606.19249** *Transformer Geometry Observatory TGO-I*
**[N]**; **2603.17946** *CARE: Covariance-Aware and Rank-Enhanced Decomposition*
**[N]** (already in the 2026-09-10 scan, under SVD-vs-importance).
