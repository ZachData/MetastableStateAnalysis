<!-- p1c_frames/lit-1c.md -->
# Phase 1c — LITERATURE

**Status: LEADS, NOT READINGS.** Produced 2026-09-16 by web search only. **No paper
below has been read**; `arxiv.org` and every other scholarly host are blocked by this
session's egress proxy. Marks: **[S]** search-engine summary read, **[N]** title and
id only.

**The headline: the "residual block = forward Euler step of an ODE" construction is
standard and has been since 2018, and at least one 2026 paper places that
discretisation on the sphere. What is not standard — and what the searches did not
find anywhere — is *calibrating the step size against the theory's own field and
reporting the residual*. That is Phase 1c's contribution and it survives.**

---

## 1. What the phase rests on, and what is now standard

`design-1c.md` opens by treating "a residual block is a forward-Euler step of the
paper's ODE" as a construction to be justified. It does not need justifying; it is
the field's default reading of a residual network.

- **Continuous-in-Depth Neural Networks**, arXiv **2008.02389** **[S]**: `H_{n+1} =
  H_n + F(H_n)` is forward Euler with `Δτ = 1`; depth `D` is evolution time `T`.
- **ODE Transformer** **[N]**, **A Neural ODE Interpretation of Transformer Layers**
  **[N]**, **Continuous-Depth Transformers with Learned Control Dynamics**,
  **2601.10007** **[N]**.
- **Transformer as an Euler Discretization of Score-based Variational Flow**,
  **2604.23740** **[S]** — and this one is on the sphere: "a Transformer block
  integrates the preceding components into a **forward Euler discretization of the
  spherical SVFlow ODE**, where each layer applies either attention or MoE as the
  vector field and acts as time with layer-wise independent parameters simulating a
  **fixed-step** evolution." **Read this before writing anything about step size.**

**The one sentence in these summaries that is a direct hit on Phase 1c's central
number:** "Learned output scale α bounds the **effective step size** to
`α · Δτ ≈ 0.025`." If somebody has published a measured effective step size for a
transformer block, then `h_calibrated` is not the first such number and §3 of
`design-1c.md` needs a citation and a comparison. **This is the first thing to
check.**

---

## 2. Finding by finding

| Phase 1c construction | Nearest prior work | Verdict |
|---|---|---|
| Residual block as forward-Euler step; depth as integration time | 2008.02389 **[S]** and the ODE-transformer line | **NOT NEW.** Standard since ~2018 |
| Euler discretisation of a **spherical** attention flow specifically | **2604.23740** **[S]** | **NOT NEW, and recent.** Closest neighbour to the construction |
| **Step size measured three ways** (`h_displacement`, `h_calibrated`, `h_attn_only`), with `verdict(robust=False)` when they straddle `t*` | Nothing found. 2604.23740 **[S]** appears to *assume* fixed step; the α ≈ 0.025 number appears to be read off a learned scale, not calibrated against a field | **Looks new**, and it is the phase's best card |
| `T_eff = Σ h_ℓ` compared against `t*` (time for γ_β to reach 0.9) | Nothing found | **Looks new** |
| **Residual** `ip_mean(ℓ) − γ_{β_eff}(T_eff(ℓ))` as the deliverable; the fit is not | Nothing found | **Looks new.** The literature builds ODE transformers; it does not use the ODE as a *null* for a trained one |
| `time_residual` — inverting the null to escape the sigmoid ceiling | Nothing found | **Looks new**, and is a transferable trick |
| The FFN is excluded because the theory has no FFN; Pythia's parallel residual makes the split exact | 2604.23740 **[S]** treats attention **or** MoE as the vector field, i.e. it puts the FFN *inside* the field | **Direct disagreement with a 2026 paper.** Worth stating explicitly — the two choices answer different questions and ours is the frame-correct one for comparing against a no-FFN theorem |
| Per-layer `β_eff`, with run-median fallback and fallback counting | *Quantifying Concentration Phenomena of Mean-Field Transformers in the Low-Temperature Regime*, **2605.10931** **[S]**; *Dynamical Mean-Field Theory of Self-Attention Neural Networks*, **2406.07247** **[S]** | **Adjacent theory exists; the measurement does not.** See §4.1 |
| Anisotropy gap: a second null integrated from the **observed** layer-0 `ip_mean` because Theorem 6.8 assumes orthogonal init | Nothing found | **Looks new**, and it is the right control given `lit-1b.md`'s anisotropy literature |
| Causal vs non-causal field (`sa_field(causal=)`) | **2411.04990** *Clustering in Causal Attention Masking* **[S]** | **The theory now exists.** `design-1c.md` calls `causal=True` "a departure from the theory"; since Nov 2024 it is *the* theory. Update the docstring and the doc |

---

## 3. The one that is not a literature question but is found here anyway

`design-1c.md` §3 establishes that damping the field by 0.3× gives a residual of
−0.0009, i.e. **the residual does not measure how much of the field the network
applies; it measures whether the network moves in a different direction.** No source
found makes this distinction, and it is the difference between a claim about
attenuation and a claim about resistance. Blog 1's "the trained network resists the
collapse its architecture drives" is only defensible in the second sense, and Phase
1c is what makes it so. **That reframing is worth more than the residual number.**

---

## 4. Directions to grow

### 4.1 The critical-β prediction is sitting there unchecked

The search for effective inverse temperature returned this, from the scaling-limits
line **[S]**: *"the critical scale at which selectivity emerges … scales like
`β*_n ≍ n^{2/(d−1)}` for uniform keys on spheres."*

Phase 1c already estimates `β_eff` **per layer, per prompt, at 27 checkpoints**, and
the prompts were chosen to span `n` from 20 to 512 precisely so that length effects
separate from content effects. `d = 1024` is fixed. So:

> **Does the measured `β_eff` sit above or below `β*_n` — and does the crossing point
> move with `n` the way `n^{2/(d−1)}` says?**

This is a published, quantitative, dimensionally-explicit prediction about a quantity
this project already has on disk, across a length range chosen for exactly this kind
of comparison. It costs no forward passes. **It is the highest-value item in this
file.** Caveat before trusting it: the `β_eff` estimator carries the indexing bug
`math-1.md` §3.4 records, and `math-5.md` §0 says Phase 5 inherited it. Fix that
first or the comparison is meaningless.

### 4.2 Run the null in its causal form

`sa_field(causal=True)` is already the default for `T_eff`. What is missing is that
the *null curve* γ_β is still integrated from the unmasked (SA)/(USA) equations of
2312.10794. **2411.04990** **[S]** is the masked theory. If its dynamics admit a
`γ`-analogue, the residual should be computed against that; if they do not, the
honest statement is that no closed-form null exists for a decoder and the residual
is against an inapplicable baseline. Either outcome is worth writing down and
neither is currently written down.

### 4.3 Compare `h_calibrated` against a published effective step size

If 2604.23740's `α · Δτ ≈ 0.025` is a comparable quantity, Phase 1c's calibrated
step on Pythia is a second data point on the same axis from a different architecture
and a different method. Two independent measurements of a transformer's effective
integration step is a small paper on its own.

### 4.4 Take the mean-field-theory papers seriously as competing nulls

**2406.07247** (DMFT of self-attention) **[S]** and **2605.10931** (concentration in
the low-temperature regime) **[S]** both produce predictions about the same
concentration behaviour `γ_β` describes, by different routes. A residual computed
against three nulls that disagree is stronger evidence than one computed against a
null that might be wrong.

---

## 5. Verification queue

1. **2604.23740** — is its spherical Euler discretisation the same object as ours,
   and is `α · Δτ ≈ 0.025` a measured effective step size?
2. **2411.04990** — does the masked theory have a `γ_β` analogue?
3. The `β*_n ≍ n^{2/(d−1)}` result — **find its actual paper**; it came from a
   search summary attributed to the long-context scaling-limits line
   (**2605.08505** **[N]**), and the attribution is not reliable.
4. **2605.10931** — what exactly concentrates, and at what rate.
5. **2406.07247** — whether DMFT predicts metastable lifetimes we could compare
   `T_eff` against.
6. **2008.02389** — for the citation Phase 1c should carry instead of deriving the
   Euler reading from scratch.

---

## 6. Search log (2026-09-16)

- `residual block forward Euler discretization neural ODE transformer depth as time step size measurement`
- `effective inverse temperature beta attention softmax estimate trained transformer measure integration time depth`
- (inherited) `Clustering in Causal Attention Masking arXiv 2411.04990 decoder-only self-attention dynamics`

Surfaced, not pursued: **2306.17759** *The Shaped Transformer: Attention Models in
the Infinite Depth-and-Width Limit* **[N]**; **1911.10305** *Dynamical System
Inspired Adaptive Time Stepping Controller for Residual Network Families* **[N]**;
**2504.18590** *A multilevel approach to accelerate the training of Transformers*
**[N]**; **2601.09775** *The Geometry of Thought: Disclosing the Transformer as a
Tropical Polynomial Circuit* **[N]**.
