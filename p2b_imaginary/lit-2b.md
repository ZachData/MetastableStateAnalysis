<!-- p2b_imaginary/lit-2b.md -->
# Phase 2b — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**The headline: the `V = S + A` decomposition, the `so(d)` reading of `A`, the `d/2`
rotation planes, and — critically — the fact that `exp(A)` is orthogonal are all
stated together in at least one public source. That last identity is the one that
forced this phase's `rotation_neutral` withdrawal. The withdrawal was right; the
identity was known.**

---

## 1. The closest neighbour

*Self-Attention as a Kernel Machine: The Geometry of Objects and Relations* **[S]**
(Agus Sudjianto, `agussudjianto.substack.com`). The search summary states, in order:

- "the decomposition `M = S + A` is complete and natural";
- "the antisymmetric component of attention matrices learns `d/2` simultaneous
  **rotation planes**, all from data";
- "the antisymmetric component `A` lives in **𝔰𝔬(d)**, the Lie algebra of the rotation
  group `SO(d)`";
- three properties of `A`: **score reversal**, **self-silence** (zero diagonal), and
  **hidden rotation — `exp(A)` is orthogonal**;
- "antisymmetric matrices in transformer attention can be decomposed through **Schur
  decomposition**".

Every one of those is in `math-2b.md` or `rotational_schur.py`. **The third bullet is
the thing.** `PLAN_2b.md` opens:

> Block 1b's headline — `elim_rotation = 0.0` in 35/35 runs, read as "rotation is
> dynamically neutral" — is an algebraic identity. `A = (V − Vᵀ)/2` is real
> antisymmetric, so `e^{−A}` is orthogonal, and every quantity Block 1b measures is a
> function of `X Xᵀ`, which an orthogonal map preserves exactly.

That is the same sentence, reached the hard way — after 35 runs. **Caveat, and it
matters for how this is written up:** the source is a Substack post, not a
peer-reviewed paper, and it was surfaced by a single search summary. It is a *lead
about a lead*. Verify before treating it as prior art.

**What this does not change:** the withdrawal stands on its own algebra and does not
need external support. What it changes is the framing — "we discovered our result was
an identity" becomes "we rediscovered an identity the field states up front", which
is a weaker claim to novelty and a stronger claim to rigour. `PLAN_2b.md` item 1 is
already correct; this file is the citation it lacked.

---

## 2. Finding by finding

| Phase 2b item | Nearest prior work | Verdict |
|---|---|---|
| `V_eff = S + A` as the natural split | *Self-Attention as a Kernel Machine* **[S]**; standard matrix algebra | **NOT NEW** |
| ~98 % complex eigen-dimensions in OV | Elhage et al. 2021 **[S]** (complex eigenvalues = "amplifying with rotation"); see `lit-2.md` §1.1 | **NOT NEW** |
| Henrici non-normality as a measure of how informative the S/A split is | Nothing found applying Henrici to transformer weights | **Looks new in application.** Standard numerical-analysis quantity, unusual use |
| **`elim_rotation = 0` is forced by orthogonal invariance, not measured** | The `exp(A) ∈ SO(d)` identity is stated in **[S]** above | **NOT NEW as mathematics; the withdrawal is still ours to publish as a methods warning** |
| `elim_full` vs `elim_signed` is the only real contrast, because `e^{−(S+A)} ≠ e^{−S}e^{−A}` unless `S` and `A` commute | Nothing found | **Looks new.** This is the correct replacement test and it is stated nowhere the searches reached |
| Frames renamed by what they **remove**, with `remove_rotation` flagged `is_invariance_control=True` and audited | Not a literature question | Internal, and the right design |
| Rotation in the **readout / weight space** rather than in `X Xᵀ` (PLAN item 14, TODO) | *Fingerprint, Not Blueprint: How Positional Schemes Set the Default Spectral Algebra of Attention*, **2607.06621** **[N]**; *Self-Attention Dynamics with Rotary Position Embeddings: Twisted States and Explicit Consensus Rates on the Sphere*, **2607.24502** **[N]** | **Contested territory — read both before building item 14** |

---

## 3. The trap this phase fell into is a publishable object

`PLAN_2b.md`'s withdrawal is an instance of a general failure mode:

> **A causal test whose readout is invariant under the intervention it applies will
> return a clean null at machine precision, and the null will look like a finding.**

Residual ~1e-15 against a 1e-3 threshold, in 35 of 35 runs, across encoder and decoder
architectures, from 12 to 48 layers — the uniformity that `design-2b.md` read as
"licensing treating it as a closed finding" was the tell. **Perfect cross-architecture
uniformity is evidence of an identity, not of a law.**

The searches found no paper making this point about interpretability interventions.
Given how many ablation and rescaling interventions the field runs, and how many
readouts are Gram-matrix-based (CKA, cosine, kernel alignment, anything on `X Xᵀ`),
**this is a short, useful, transferable methods note**, and the project has the worked
case with the numbers. It is the single most publishable thing in this phase.

---

## 4. Directions to grow

1. **Verify the Substack source, then write the methods note (§3) with it cited as
   the statement of the identity and our 35-run null as the worked failure.** If the
   source does not hold up, the note is stronger, not weaker — the identity is then
   folklore rather than published.
2. **Finish item 14 (the real rotation test) against 2607.24502.** RoPE *is* a
   rotation, applied to QK rather than OV, and a 2026 paper appears to analyse
   self-attention dynamics with RoPE on the sphere ("twisted states"). If rotational
   structure in a transformer has a natural home, RoPE is it, and asking whether OV's
   `A` interacts with RoPE's rotation is a question with a clean falsifier and no
   obvious prior work. **This is the direction with the most room.**
3. **`elim_full` vs `elim_signed` is the measurement.** It is built (PLAN item 3,
   DONE) and its result is not written up anywhere this file could find. Run it, and
   report the commutator `‖[S, A]‖` beside it — the contrast is only informative to
   the extent `S` and `A` fail to commute, and that is a free weights-only number.
4. **The φ question from the sister project** (`design-8.md`: does the antisymmetric
   fraction differ between the QK and OV halves?) is this phase's question asked
   across two circuits. It is listed as the genuine conceptual overlap with
   `Lora_inductionhead` and it is unanswered.

---

## 5. Verification queue

1. **The `M = S + A` / `exp(A) ∈ SO(d)` source** — find whether there is a
   peer-reviewed paper behind the Substack post. Decides §1 and §3's framing.
2. **2607.24502** — RoPE self-attention dynamics on the sphere; decides direction 2.
3. **2607.06621** — positional scheme → spectral algebra; whether "98 % complex" is
   a RoPE artifact.
4. **Elhage et al. 2021** — the complex-eigenvalue discussion, exact wording.
5. **2604.26085** *Spectral Selection in Symmetric Self-Attention Dynamics* **[N]** —
   what "symmetric" buys, i.e. what is lost when `A ≠ 0`.

---

## 6. Search log (2026-09-16)

- `complex eigenvalues attention weight matrices non-normal rotation antisymmetric component transformer analysis`
- `"Gradient Flow Structure and Quantitative Dynamics of Multi-Head Self-Attention" symmetric value matrix condition energy`
- (inherited) `OV circuit eigenvalue spectrum attention head copying score negative eigenvalues positive eigenvalues induction mechanistic interpretability`
