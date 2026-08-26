<!-- PUBLICATION_IDEAS.md -->
# Publication ideas

Sketches only — three things this repo could carry as a paper or as the next posts in the
Blog 1 sequence, with the outside literature each would sit against. Nothing here is
written, and nothing here commits us to a venue. Blog 1 ("the Transformer functions by
resisting its own architecture") is the on-ramp for all three: it established the
trained-vs-random contrast on GPT-2-large, and each idea below is a different answer to
"and then what?" Read alongside `MATH_INDEX.md` (the open-question register), `claims/CLAIMS.md`
(the six claims), and `MATH_SPECTRAL_OT.md` (the tools argument).

## 1. Metastability at $d \approx 10^3$: the theory measured inside trained transformers

The strongest empirical paper we can write is the one Blog 1 gestured at and `p1c_frames/`
actually instrumented: take Geshkovski et al.'s dynamics seriously as a *quantitative* null and
report how far a real network sits from it. The material is already derived — $T_{\rm eff}$ as
integration time, the $\gamma_\beta$ ODE of Theorems 6.8/6.9 as the concentration prediction, the
vertical and time residuals against it, Wendel's cone condition at layer 0, spherical designs for
the repulsive limit (`math-1c.md` §§2, 3, 7, 8) — and the honest framing is the paper's
best feature rather than a caveat to bury: four of our six registered predictions test statements
the paper itself poses as **open problems**, not theorems (Problem 1 for metastability, Problem 2
for general $Q,K,V$, Problem 5 for Table 1, Remark 3.5 for Sinkhorn — the full map is `math-1.md`
§16). We would be reporting evidence bearing on open conjectures at parameters three orders of
magnitude outside where their supporting numerics live: Figure 3's metastable band vanishes by
$d \approx 512$ and Pythia runs at $d = 1024$, while Theorem 6.9's concentration bound is weakest
exactly mid-trajectory where metastability would live. That gap — not "we replicated the paper" —
is the contribution. Two structural facts about real models push it further and are ours to
report: the models are **causally masked** and use **RoPE**, both of which break the permutation
equivariance of Remark 2.2 that the mean-field limit rests on, which is precisely the setting
Karagodin–Polyanskiy–Rigollet opened up for causal attention; and the headline "trained weights
resist collapse" is only a finding if $T_{\rm eff} \ll t^\ast \approx 4.2$ fails to explain it,
which is registered as P-γ2 and is the first item in the open-question register (`UPDATE_PLAN.md`
§5.2 found the obvious step-size definition understates $T_{\rm eff}$ by ~5.7×, in the direction
that would kill the claim). Every input for that measurement is already on disk.

## 2. There are two spectra, and the field computes the wrong one

The second paper is the methods-and-mechanism one, and it is the most novel thing in the repo.
Eigenvalues of the OV circuit linearize the motion of a *particle in $\mathbb{R}^d$*;
metastability is a property of the *measure*, governed by the spectrum of the Wasserstein
Hessian of $E_\beta$ — near-zero eigenvalues are long-lived states, the sign structure of the
leading eigenvectors *is* the partition into them, and the escape rate over a barrier is
Eyring–Kramers. Every "these are metastable states" claim in this literature, ours included,
currently rests on a clustering algorithm plus a scalar. Three pieces make this writable now
(`MATH_SPECTRAL_OT.md` §§2, 5, 6). First, the **dissipation identity**: $\nabla_{x_i} E_\beta$
has a two-line closed form that is literally an unnormalized attention-weighted average of the
other tokens — the gradient-flow condition stops being a hypothesis tested by proxy and becomes
an identity you can evaluate, per particle, per layer, giving an *exact* attribution of every
energy violation to specific tokens and, on Pythia's parallel residual, an exactly additive
attention/FFN split with no cross term and no clipping. Second, **non-normality**: at
$T_{\rm eff} \ll 1$ the transient is governed by the numerical abscissa $\lambda_{\max}(S)$ and
$\sup_t \lVert e^{-tV}\rVert$, not by $\max \mathrm{Re}\,\lambda$ — so the attractive/repulsive
fraction the field reads off eigenvalues is being read in a regime where it does not govern
(Trefethen & Embree; eigenvalue condition numbers also *derive* the tolerances Phase 2d currently
places by hand). This is weights-only, so it runs across all 27 Pythia checkpoints for the cost
of a Schur decomposition per layer. Third, **the right instrument exists in another field**:
transfer-operator / Markov-state models with implied timescales and PCCA+, standard in molecular
dynamics, of which our Fiedler vector is the $k=2$ special case. Optimal transport is the frame
that makes these one subject rather than three — a coordinate-free $T_{\rm eff}$ as Wasserstein
arc length, the identity-vs-optimal coupling gap as a new observable separating tokens *swapping
places* from genuine motion of the measure, straightness (long path, short net displacement) as
dwelling measured on the measure instead of inferred from HDBSCAN, and Gromov–Wasserstein as the
correct cross-architecture instrument where `linear_cka` needs paired samples in comparable spaces.

## 3. The bridge: mechinterp constructs as particle motifs, and the dimensionality budget

The third is the one that will travel furthest outside the dynamical-systems audience, and it is
`p7_motifs/` plus Phase 5c. The claim (H-BRIDGE) is deliberately not "the two vocabularies can be
translated" — that is unfalsifiable and would produce a glossary — but that the particle reading
makes **differential** predictions, correct where the standard account is silent or says something
else. Induction-head formation is the first study because it has a sharp existing behavioral
metric, a time axis (Olsson et al.'s abrupt formation against our Pythia checkpoint sweep, where
*which comes first* has three informative answers), and a direct bearing on our pre-registered
claim (b) that collapse-resistance emerges at circuit-formation events. Restated as particles, an
induction head is a two-stage `relay` motif in an interaction graph whose edges carry **force**
($A_{ij} \cdot Vx_j$) rather than attention, typed by sign channel, rotational channel, offset
class and pair type — which is what keeps it from being a re-description of attention-pattern
analysis. Paired with this is the resource-allocation result Blog 1 already half-reported: the
~40–50% of tokens HDBSCAN never clusters are not residue. Trained models route attention *toward*
them at 1.6–2×, sign-flipped from random weights, and effective rank plateaus near 200–250 across
models whose $d_{\rm model}$ spans 768–1600 — if the network simply used what training gave it,
rank should scale with $d_{\rm model}$ and it does not. That is a bounded **dimensionality budget**
spent on particles that must stay individuated (induction, n-gram completion, position tracking:
anything needing a *specific* token rather than its cluster's attractor), and it connects the
particle picture directly to rank-collapse/oversmoothing results, to attention sinks, and to the
question of what SAE features are — treated here strictly as an object of study, never an
instrument.

## A possible fourth, shorter piece

There is a methodology post in `POPPER_PLAN.md`, `claims/`, and `MATH_INDEX.md`'s six recurring
failure patterns that would stand on its own: tests that cannot come out the other way, thresholds
not derived from a null, dimension not controlled, clamping that hides the diagnostic,
producer/consumer mismatches that return a plausible empty value, and instruments whose failure
mode looks like the result. Plus what it took to build real nulls — the registered permutation
null for claim (b) that *measured* at 0.32–0.45 under H0 and had to be replaced; the homogeneity
correction for claim (c) and its admissible band; running a gate on an input whose answer is known
before spending forward passes. Most interpretability work does not do this, and the artifacts
that would make the post credible are already committed with timestamps that precede their results.

## Papers to build on

**The core team.** Geshkovski, Letrouit, Polyanskiy & Rigollet, *A mathematical perspective on
Transformers* (arXiv:2312.10794) and *The emergence of clusters in self-attention dynamics*
(NeurIPS 2023); Geshkovski, Koubbi, Polyanskiy & Rigollet, *Dynamic metastability in the
self-attention model* — the direct successor on the phenomenon this repo is named after, and the
first thing to read against Problem 1; Karagodin, Polyanskiy & Rigollet, *Clustering in causal
attention masking* — the causal case, which is the case every model we run is in; Geshkovski,
Rigollet & Ruiz-Balet, *Measure-to-measure interpolation using Transformers*.

**Adjacent dynamics.** Bruno, Pasqualotto & Agazzi on meta-stable clustering in mean-field
transformer models; Sander, Ablin, Blondel & Peyré, *Sinkformers* (doubly-stochastic attention
*is* a Wasserstein gradient flow — Remark 3.5's open question); Castin, Ablin & Peyré, *How smooth
is attention?* and the unified-perspective follow-up; Alcalde, Fantuzzi & Zuazua on hardmax
clustering.

**Collapse and rank.** Dong, Cordonnier & Loukas, *Attention is not all you need* (pure attention
loses rank doubly exponentially with depth) — the closest existing statement of our collapse
result, worth positioning against explicitly; the oversmoothing literature from attention-based
GNNs; entropy/signal-propagation collapse work.

**Tools from other fields.** Trefethen & Embree, *Spectra and Pseudospectra* (non-normality,
transient growth, Kreiss constant); Deuflhard & Weber (PCCA+) and Schütte's transfer-operator
metastability; Bovier–Gayrard–Klein for Eyring–Kramers; Edelman, Kostlan & Shub and the elliptic
law (Sommers et al.) for the random-matrix nulls; Peyré & Cuturi for computational OT.

**Interpretability side.** Olsson et al., *In-context learning and induction heads*; Biderman et
al., *Pythia* (the checkpoint suite the whole transition rests on); the attention-sink literature;
Wurgaft et al. on manifold steering (already used in `math-5b.md`).
