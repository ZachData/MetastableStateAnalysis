<!-- trackS_subspace/lit-S2.md -->
# S2, what rotation does: literature scan before its design freezes (2026-10-21)

Trigger (`CLAUDE.md` "Literature scans"): S2 opens (`docs/ROADMAP.md` §7a, order §9 row 6).
Scope: (1) has anyone edited a trained LM's value / OV weights to their symmetric part, or removed
the rotation (complex eigenvalues) from them, and read the effect; (2) what the particle theory
says a symmetric V does to the energy; (3) weight-level spectral surgery and its controls;
(4) what is known about rotation in heads' OV and induction. Nine web searches, six abstract
pages (2026-10-21). Not repeated: `trackS_subspace/lit-S1.md` (shared subspaces, 2607.06621,
2607.09279, copy suppression), `p2b_imaginary/lit-2b.md` (complex fraction),
`trackA_force/lit-C2.md` (energy along training), Phase 10's OV1 / OV1s / OV1d
(`p10_cluster_function/status-10.md` §1.37–1.39, the repo's own weight edits of OV).

**Marks.** [A] the arXiv abstract page was fetched and read this session; [S] a search engine's
summary only. Nothing was read in full.

## 1. Verdict on "nobody edits OV's rotation out and reads the energy and induction"

**Not refuted.** No paper found replaces a trained LM's OV by its symmetric part, or sets its
complex eigenvalues real, and reads either the energy over depth or in-context copying, on any
checkpoint. The searches for "ablate the rotational component" returned only informal posts that
tried and were inconclusive [S]. The nearest:

| work | what it does | bearing on S2 |
|---|---|---|
| Phase 10 OV1 / OV1s / OV1d (this repo, `status-10.md` §1.37–1.39) | every head's OV cut to its attractive part, its sign flipped, a dose at matched centred distance; c3x's groups read against random cuts of matched rank and norm | the harness S2 reuses (LN1 folded, centred frame; the head's constant moved to the bias; read-back check). **Its lesson is S2's first hazard:** arms moved the stream 2–3× as far as their norm-matched controls, so "the sign" was not separated from "moved further" until OV1d matched distance |
| Shihab et al., 2609.26165 [A] | spectral tail surgery on QK products in decoder LMs; component isolation also edits value–output; controls are spectrum-matched Haar draws, not norm-matched | a precedent for weight-level spectral edits read against random-matched controls; QK, not OV's rotation; no training axis that the abstract states |
| `p2b_imaginary/imaginary_ablation.py` (this repo, Block 3) | a post-hoc projection off col(skew OV) | never run and degenerate (the projector is the identity at d = 1024, `status-2b.md`); S2's arm (a) is the weight-space version 2b asked for |

## 2. What the theory says a symmetric V does

| work | statement | bearing |
|---|---|---|
| Geshkovski, Letrouit, Polyanskiy, Rigollet, 2305.05465 [A] (NeurIPS 2023) | the limiting object of the self-attention flow depends on the spectrum of V | V's spectrum is the theory's handle; S2's arm (b) changes only that spectrum's imaginary parts |
| the gradient-flow results (Geshkovski et al. 2023 survey, via [S] summaries; Tomihari, Karakida, 2505.19458 [A]) | the energy is **non-decreasing** along the flow under constraints *(corrected 2026-10-22 from "decreases", read in full at C2 unit 1: `trackC_writeup/lit-C2-unit1.md` §1; no S2 rule moves)*: single head, states on the sphere, **symmetric weights** (V tied to QᵀK; Q = K = V = I in most statements) | a symmetric V alone, with QK untied, is not in these results. `tools/math_checks/s2_ov_arms.py` C6: E₁ (A5's statistic) is non-monotone in 53 of 200 toy attention-only flows with symmetric V and untied QK, monotone in all 200 at Q = K = V = I. **So no theorem predicts "monotone" for arm (a)**; Q1 is an empirical question |
| Tomihari, Karakida, 2505.19458 [A] | without an energy: Jacobians of recurrent self-attention; normalisation suppresses the Jacobian's complex eigenvalues (oscillation); Lyapunov exponents put normalised dynamics near criticality | the closest statement that complex eigenvalues (of a Jacobian, not V) carry oscillation, and that LN damps them. S2 reads V's, through LN1's folded frame |
| Pendharkar, 2605.04279 [A] | multi-head: a natural energy is non-decreasing under a sufficient condition on the score matrices; "radial shadow terms" block per-head monotonicity | Pythia has 16 heads per layer; per-head symmetry need not give a monotone multi-head energy either |
| Bruno, Pasqualotto, Agazzi, 2410.23228 [S] (ICLR 2025) | metastable clustering in the mean-field model as a Wasserstein gradient flow | the gradient-flow setting again; nothing about non-symmetric V found in the abstract |
| Latifi Jebelli, 2607.13295 [A] | attention as a two-body interaction law must be nonlocal and **nonreciprocal** to realise attention digraphs | a theoretical reason the non-symmetric part should matter for function; no experiment |
| Pham et al., 2512.03058 [A] | tokens' convergence or divergence in pre-trained transformers' continuous limit, and rotary vs absolute positions | QK side and positions; nothing on V's rotation in the abstract |

## 3. Rotation, copying and induction at the head level

| work | finding | bearing |
|---|---|---|
| Elhage et al. 2021; copying score Σλ / Σ\|λ\| [S] | complex eigenvalues drive a head's copying score toward 0 | arm (b) raises every head's copying score toward ±1 by construction (Re λ / \|Re λ\|); S2 reads behaviour, not the score |
| Li Hengyu, 2607.06621 [S this session; A in `lit-S1.md`] | QK rotation of previous-token / induction heads emerges with the behaviour on Pythia; altering spectral channels delays formation; induction heads read as symmetric, content-like | QK, not OV; a reason to expect induction to survive an OV de-rotation if its OV is mostly real. S2's R2 reads it |
| Kayyam et al. 2026, QKV variants [S]; 2406.06366 symmetric dot-product attention [S] | tying Q = K (symmetric attention) costs little in BERT, more in causal LMs (one summary: +25 % perplexity, not in the abstract) | symmetry trained in from the start, not edited in after; not S2's question |
| Hoover et al., Energy Transformer, 2302.07253 [S] | an architecture engineered so a global energy decreases | built, not found; no bearing on a trained Pythia beyond "monotone energy is a design constraint elsewhere" |

## What the scan changes in the design

| item | change |
|---|---|
| §2, C6 | Q1 is stated as "is the break the OV's non-symmetric part", not as a test of a theorem; a monotone arm (a) would be an empirical finding, a non-monotone one says nothing against the theory |
| OV1's "moved further" | every S2 cell carries its centred displacement against its controls', and a cell where the arm moves the stream further than every control is not read (design "Readings") |
| 2609.26165 | controls are matched per head in Frobenius norm and in span (the edit's own read and write spaces), not spectrum-matched Haar draws: the question is "this edit, or any edit of this size there" |
| §3 | R2 reads induction behaviourally (7d's repeated-random-token probe), since arm (b) moves the copying score by construction |

## Parked

1. **The gradient-flow statement's exact hypotheses** (why: S2 cites "V tied to QᵀK" from
   summaries, and C6 is a toy; cost: one read of the 2023 survey's §3; could change: the wording
   of "no theorem predicts monotone" only, not any rule). Carry to S2's close. **At S2's close
   (2026-10-22): carried to C2's draft** (`status-S.md` "S2 on one page"), where the energy is
   stated against the theory. **Done at C2 unit 1 (2026-10-22):** QᵀK symmetric and V = QᵀK for
   one head, each Mₕ = W_V,h for many, constant weights, sphere, no MLP; the sign is
   non-decreasing (row above corrected) (`trackC_writeup/lit-C2-unit1.md` §1).
