<!-- PUBLICATION_IDEAS.md -->
# Publication ideas

Sketches, not drafts. Six things this repo could carry as a paper or as the next entries in the
Blog 1 sequence, each with what is already on disk, what would have to be run, and the outside
literature it sits against. Blog 1 ("the Transformer functions by resisting its own architecture")
is the on-ramp for all of them: it established the trained-vs-random contrast on GPT-2-large, and
each idea below is a different answer to "and then what?"

Read alongside `MATH_INDEX.md` (the open-question register, ordered by value-per-unit-cost),
`claims/CLAIMS.md` (the six claims and what adjudicates each), `PREDICTIONS.md` (what is
registered and undecided), and `MATH_SPECTRAL_OT.md` (the tools argument).

**On the citations.** Every paper in the reference list was looked up in this session and its
title, authors and arXiv ID confirmed. arxiv.org itself is blocked by this environment's egress
proxy, so the one-line characterizations come from search-result abstracts rather than from
reading the papers. Before any of this goes in a related-work section, the four papers in
"Prior art that has moved" need reading in full — two of them are close enough to our claims
that the reading changes what we write.

---

## Idea 1 — Metastability at $d \approx 10^3$: the theory measured inside trained transformers

The strongest empirical paper we can write is the one Blog 1 gestured at and `p1c_frames/`
actually instrumented: take the Geshkovski et al. dynamics seriously as a *quantitative* null and
report how far a real network sits from it, layer by layer, with the deviation as the object of
study rather than the error bar. The machinery is derived and mostly built — $T_{\rm eff}$ as
integration time, the $\gamma_\beta$ ODE of Theorems 6.8/6.9 as the concentration prediction, the
vertical and time residuals against it, Wendel's theorem for the cone condition at layer 0,
Gegenbauer moments against spherical designs for the repulsive limit (`math-1c.md` §§2, 3, 7, 8).
The paper writes itself as a distance ledger: six named distances between what Pythia computes and
what the idealization assumes (`math-1.md` §3), each with its own instrument, none of them
collapsed into a single "the theory works / doesn't" verdict.

The framing is the paper's best feature and it should be in the abstract, not buried in a
limitations section. Four of our six registered predictions test statements the source paper poses
as **open problems**, not theorems — Problem 1 for metastability itself, Problem 2 for general
$Q,K,V$, Problem 5 for Table 1's generality, Remark 3.5 for the Sinkhorn question — and the full
map of which result lives where is `math-1.md` §16. So the honest claim is *evidence bearing on
open conjectures at parameters three orders of magnitude outside where their supporting numerics
live*: the paper's own Figure 3 says the metastable band vanishes by $d \approx 512$ and we run at
$d = 1024$, while Theorem 6.9's concentration bound is weakest exactly mid-trajectory, which is
where metastability would have to live. That is a stronger justification for the whole enterprise
than "we replicated the paper," and it is one no purely theoretical paper in this line can make.

Two things have to happen before this is submittable, and both are already the top of the
open-question register. First, **P-γ2** — measure $T_{\rm eff}$ against $t^\ast \approx 4.2$.
`UPDATE_PLAN.md` §5.2 found the obvious step-size definition understates $T_{\rm eff}$ by ~5.7×,
in exactly the direction that would turn "trained weights resist collapse" into "the network never
integrates far enough to collapse in the first place," which is a much duller finding. The
prediction is stated so that the outcome that would hurt is the one predicted, every input is on
disk, and it is item 1 in `MATH_INDEX.md`. Second, the **replication gate** (claim (c)) has to
run, and its preconditions are now known in advance: at least 11 of 48 cells must dissent in sign
and at least five prompts must be informative, or the gate cannot return a verdict at all. Worth
saying plainly in the paper that the gate is powered against a contrast with a prompt-specific
signature and unpowered against a uniform one, and that Blog 1's phenomenology is the second kind.

## Idea 2 — There are two spectra, and the field computes the wrong one

This is the most novel argument in the repo and the one with the clearest "you are all measuring
the wrong operator" thesis. Eigenvalues of the OV circuit linearize the motion of *a particle in
$\mathbb{R}^d$*. Metastability is a property of *the measure*: it is governed by the spectrum of
the Wasserstein Hessian of $E_\beta$ at a configuration, where near-zero eigenvalues are slow
directions, the sign structure of the leading eigenvectors *is* the partition into metastable
sets, and the escape rate over a barrier is Eyring–Kramers. Every claim of the form "these are
metastable states" in this literature — ours in Blog 1 included — currently rests on a clustering
algorithm plus a scalar. Naming that gap precisely, and showing what changes when you compute the
second spectrum instead of inferring it, is a whole paper (`MATH_SPECTRAL_OT.md` §6).

Three pieces make it writable now rather than being a research program. **(a) The dissipation
identity.** $\nabla_{x_i} E_\beta$ has a two-line closed form that is literally an unnormalized
attention-weighted average of the other tokens — the paper's gradient-flow condition stops being a
hypothesis tested by proxy and becomes an identity you can evaluate. It gives an *exact*
attribution of every energy violation to specific tokens, and on Pythia's parallel residual an
exactly additive attention/FFN split with no cross term and no clipping, unlike the leave-one-in
scheme in `decompose.py::energy_by_component` whose cross term is not small. `core/dissipation.py`
is built. **(b) Non-normality.** At $T_{\rm eff} \ll 1$ the transient is governed by the numerical
abscissa $\lambda_{\max}(S)$ and $\sup_t \lVert e^{-tV}\rVert$, not by $\max \mathrm{Re}\,\lambda$
— so the attractive/repulsive fractions Phase 2 reads off eigenvalues are being read in a regime
where they do not govern, and eigenvalue condition numbers would *derive* the tolerances Phase 2d
currently places by hand. This is weights-only: it runs across all 27 Pythia checkpoints for the
cost of a Schur decomposition per layer, no forward passes. **(c) The right instrument, imported.**
Transfer-operator / Markov-state models over clusters, implied timescales
$t_i = -1/\log|\lambda_i|$, and PCCA+ — standard in molecular dynamics, and our Fiedler value is
exactly the $k=2$ special case of PCCA+.

Optimal transport is what makes these one subject rather than three, and it also supplies
observables nothing in the project currently has. A coordinate-free $T_{\rm eff}$ as Wasserstein
arc length puts P-γ2 on a defensible footing instead of a convention. The gap between the identity
coupling and the optimal coupling is a free new observable: it separates tokens *swapping places*
— motion that leaves the distribution unchanged — from genuine motion of the measure, and no
existing metric here distinguishes them. Straightness (long path, short net displacement) measures
dwelling on the measure itself rather than inferring it from whether HDBSCAN found clusters, which
is precisely the instrument the metastability claim needs. Gromov–Wasserstein is the correct
cross-architecture comparison where `linear_cka` requires paired samples in comparable spaces —
i.e. it is the right instrument for claim (c). **Positioning caution:** Hengyu Li (2607.09279) has
now done the non-normal/Kreiss argument on the *attention propagator* across GPT-2, Pythia-410m
and Llama-3-8B, under pre-registered criteria, and Wehlitz–Pavliotis–Schütte–Winkelmann
(2601.02932) have done data-driven transfer-operator reduction for particle clustering dynamics in
general. Our angle survives both — the OV circuit rather than the attention matrix, read against a
measured $T_{\rm eff}$; and transformer token clouds rather than generic interacting particles —
but the paper has to say so up front instead of being told so in review.

## Idea 3 — The bridge: mechinterp constructs as particle motifs, and the dimensionality budget

This is the one that travels furthest outside the dynamical-systems audience, and it is
`p7_motifs/` plus Phase 5c. The claim (H-BRIDGE) is deliberately not "the two vocabularies can be
translated" — that is unfalsifiable and produces a glossary — but that the particle reading makes
**differential** predictions, correct where the standard account is silent or says something else.
Restated as particles, an induction head is a two-stage `relay` motif in an interaction graph
whose edges carry **force** ($A_{ij} \cdot Vx_j$) rather than attention, typed along four
independent axes: sign channel through Phase 2's attractive/repulsive projectors, rotational
channel through Phase 2b's Schur split, offset class, and pair type. Typing the edge by force is
what keeps this from being a re-description of attention-pattern analysis, which mechinterp
already does well and which the particle account has no advantage over: two heads with identical
attention patterns and opposite-signed OV circuits produce opposite motion.

Induction-head formation is the right first study for four reasons, and the third is what makes it
a paper rather than a demo. It has a sharp behavioral metric that already exists (mean post-softmax
attention on induction pairs — a number, not a description); it has a time axis, so a motif and a
behavior can be compared not just on whether they co-occur but on **which comes first**, with all
three answers informative; it bears directly on registered claim (b), that collapse-resistance
emerges at circuit-formation events, which `core/changepoint_colocation.py` now adjudicates with a
matched-control null after the registered permutation null was *measured* and found to reject under
H0 at 0.32–0.45; and `core/` is already most of the way there, so the phase's risk sits in its
reasoning rather than in new code. The tautology risk is the phase's central methodological danger
and `design-7.md` states it in full — the motif alphabet is fixed in advance precisely so this
cannot become a motif zoo.

Paired with it is the resource-allocation result Blog 1 already half-reported, and which may be
the more surprising half. The ~40–50% of tokens HDBSCAN never clusters are not residue: trained
models route attention *toward* them at 1.6–2×, sign-flipped from random weights, and effective
rank plateaus near 200–250 across models whose $d_{\rm model}$ spans 768–1600. If the network
simply used what training gave it, rank should scale with $d_{\rm model}$, and it does not. That
is a bounded **dimensionality budget** spent on particles that must stay individuated — induction,
n-gram completion, position tracking, anything needing *a specific token* rather than its cluster's
attractor — and it connects the particle picture directly to rank-collapse and oversmoothing
(Dong–Cordonnier–Loukas), to attention sinks, and to what SAE features are, with SAEs treated
strictly as an object of study and never as an instrument (`core/DESIGN_dual_reading.md`). Before
publishing the budget number, item 3 in the open-question register has to be settled: re-establish
the plateau on *normed* rank and reconcile ~200 against Blog 1's ~250.

## Idea 4 — What temperature is a real transformer running at?

A short, self-contained paper hiding in `core/beta_eff.py` and `core/ln_frame.py`. The entire
theory is parameterized by an inverse temperature $\beta$, and every phase diagram in the
literature — including the Figure 3 that says metastability should not exist at our $d$ — is drawn
in $(d, \beta)$. Nobody has measured what $\beta$ a trained model is actually running at, per
layer, per head, across training. We can: `math-1.md` §3.4 defines $\beta_{\rm eff}$ as one of the
six distances, Phase 1c's §4 dissolves the blocker by bracketing $\beta$ rather than point-
estimating it, and the whole thing needs no forward passes beyond what is cached. The deliverable
is a plot the theory side does not have and cannot produce: the trajectory of a real model through
its own phase diagram, checkpoint by checkpoint.

This lands directly on two recent results from the same MIT group. Chen–Lin–Polyanskiy–Rigollet
(2510.05554) identify a **critical scaling** $\beta_n \asymp \log n$ separating "all tokens collapse
to one direction" from "attention degenerates to identity," and justify the attention scaling used
in YaRN and Qwen. Karagodin–Ge–Polyanskiy–Rigollet (2510.22026) show normalization acts as *speed
regulation* and compare Post-LN, Pre-LN, Peri-LN, nGPT and LN-Scaling on exactly the clustering and
representation-collapse axes we measure. Both are predictions about where a model sits relative to
a critical value, and neither is checked against measured $\beta_{\rm eff}$ in a trained model.
Our prompt suite spans a range of $n$, our checkpoint axis spans training, and Pythia's Pre-LN is
one point in their normalization taxonomy — so "does $\beta_{\rm eff}(n)$ track $\log n$, and does
a trained model sit near criticality or far from it?" is a question we can answer this quarter.

The falsification structure is unusually clean, which matters for a project with our methodology
commitments. If $\beta_{\rm eff}$ sits far below critical, the collapse story is about temperature
rather than about learned resistance, and Blog 1's headline needs restating — the same shape of
threat that P-γ2 poses from the $T_{\rm eff}$ side, and the two together are the honest version of
"is resistance real?" If it sits near critical and moves toward criticality during training, that
is a genuinely new empirical claim about what training optimizes for, and it connects to the
low-temperature concentration results of Alcalde–Bungert–Riedl–Roith (2605.10931). Either way the
paper is short, weights-plus-cached-activations only, and it makes every other paper in this list
easier to write, because $\beta$ is the free parameter all of them currently carry as an
assumption.

## Idea 5 — The assumptions real decoders break: causal masking, RoPE, and the position-0 sink

The mean-field theory rests on exchangeability, and every model we study breaks it twice over.
Remark 2.2's permutation equivariance is destroyed by **causal masking** and by **RoPE**, and
`math-1.md` §2.4 makes the sharper point that with RoPE the attention bilinear is *not*
$x^\top W_Q W_K^\top x$ at all, so the $Q^\top K$ object the theory reasons about does not exist in
the model in the form the theory assumes. On top of that sits the **position-0 sink** (§2.5), a
particle that receives enormous attention and that Phase 6 found can silently collapse the
same-content null onto itself — a confound serious enough that `core/sink_audit.py` exists to keep
it separable. These are usually treated as caveats. They are the paper: a decoder-only transformer
is a *non-exchangeable* interacting particle system with a distinguished particle, and the question
is what of the theory survives that.

The theory side has moved here in the last two years and it moved toward us, which makes this
well-timed rather than speculative. Karagodin–Polyanskiy–Rigollet (2411.04990) modify the dynamics
for causal masking, note explicitly that the result **cannot be interpreted as a mean-field
gradient flow**, and connect metastable states to the Rényi parking problem.
Duerinckx–Geshkovski–Rossi (2605.09213) build a kinetic theory for the causal case with cumulant
expansions adapted to the triangular dependency structure, and derive the **lost-in-the-middle**
phenomenon — a U-shaped retrieval profile with primacy, recency and a unique interior minimum — as
a closed-form consequence. Hao Ye (2607.24502) works out RoPE dynamics on the sphere, finding
twisted states and an interaction energy with derivatives of both signs. Súkeník–López
Amado–Lampert–Mondelli (2605.08453) prove an equivalence between sinks and hard attention switch
and explain why pretrained transformers favour sinks over diagonal patterns.

What we bring is that all three are *measurable* on artifacts we already produce, and none of
those papers measures them in a trained model. `core/qk_offset_null.py` already implements
offset-conditioned nulls, so the causal-geometry claims have an instrument. The lost-in-the-middle
prediction is a positional profile of retrieval, and we have per-particle attention and per-particle
energy attribution at every layer and checkpoint — so we can ask whether the U-shape is visible in
the *particle* observables (does the interior minimum correspond to particles that cluster earlier,
lose individuation, and fall out of the rank budget?) rather than only in end-task retrieval
accuracy. That would be a mechanism for a phenomenon currently established behaviorally and
derived under iid-uniform tokens. And the sink is a natural bridge to Idea 3: on the force-typed
interaction graph the sink is its own motif, `design-7.md` already types it as one, and "the sink
is how a decoder buys anti-collapse" is a claim that both literatures would want adjudicated.

## Idea 6 — How not to fool yourself: nulls, pre-registration, and six failure patterns

There is a methodology paper in `POPPER_PLAN.md`, `claims/`, and `MATH_INDEX.md` that would stand
on its own and would probably be the most-read thing we publish. Its spine is the six recurring
failure patterns, each with a table of real instances from this project: **the test that cannot
come out the other way** (an `elim_rotation = 0.0` null result across 35/35 runs, because every
measured quantity was Gram-based and Gram is orthogonally invariant — the measurement was constant
by construction); **thresholds not derived from a null**; **dimension not controlled**
($\mathbb{E}\lVert P_Uv\rVert^2 = k/d$, so any cross-subspace alignment comparison measures
dimension unless normalized); **clamping that hides the diagnostic**; **producer/consumer
mismatches** that return a plausible empty value which flows into a score; and **instruments whose
failure mode looks like the result**. The one-line test that would have caught the first class is
worth the paper on its own: ask what the output would be on data where the hypothesis is maximally
true and on data where it is maximally false, and if those are the same answer, it is not a test.

The second half is what it actually took to build nulls that hold. Claim (b)'s registered
permutation null was *measured* rather than reasoned about and rejects under H0 at 0.32–0.45;
an enumerated circular shift is honest but reaches 0.103 once both series change early, as
everything does early in training; what replaced it is a matched control series with the pairing
permuted. Claim (c) needed a homogeneity correction because its exchangeable unit — the prompt —
is not independent across draws, and the correction has an admissible band above which the gate
*refuses* rather than corrects, on the derived ground that a perfect result would not survive its
own correction there. Both gates were run on inputs whose answers are known before any forward
passes were spent, which is how we know the preconditions (11 dissenting cells, five informative
prompts) rather than discovering them afterward. Most interpretability work does not do this, and
the artifacts that make the post credible are committed with timestamps preceding their results.

Two framings, and I would write the second. As a *lessons-learned* post it is useful but scolding.
As a *worked example* — here is one project's ledger, here is the null that failed its own
calibration, here is the finding we withdrew (Phase 2b's rotation result, withdrawn as an
orthogonal-invariance identity that was never falsifiable), here is the falsification we withdrew
because the correction turned out larger than the effect it would explain (Phase 6's LDA
inversion), here is what it cost — it is a contribution to how the field works. It also does
something self-serving and legitimate: it establishes why the empirical claims in Ideas 1–5 should
be believed, which for a project making claims about billion-parameter models from eight prompts
is not a small thing to have already banked.

---

## Prior art that has moved, and what it does to us

Read these four in full before writing anything above. Two are close enough to change our framing.

- **Isobe, Inoue & Imaizumi, *Training-Induced Escape from Token Clustering in a Mean-Field
  Formulation of Transformers* (2605.07772).** This is the theoretical statement of Blog 1's
  thesis: after initially following attention-driven clustering, the token distribution *leaves*
  the clustered regime near the final layers, analyzed via an entropy-regularized interaction
  energy, in a noisy mean-field model with a trained parameter-linear FFN. It is a toy model where
  we have real ones, and it trains only the FFN, but "training reshapes the clustering picture" is
  no longer an unclaimed observation. Our contribution reads as the measurement side of their
  theory — which is a *better* position than being unopposed, provided we cite it as such.
- **Hengyu Li, *Transient Reserves, Sink Dampers, and the Failure of Eigenvalue Reasoning in the
  Attention Propagator* (2607.09279).** Pre-registered, non-normal/pseudospectra/Kreiss reasoning
  on the attention propagator across GPT-2, Pythia-410m and Llama-3-8B, with the mask pinning the
  Kreiss constant at $\sqrt{n}$. Overlaps Idea 2(b) directly. Ours is the OV circuit read against a
  measured $T_{\rm eff}$, and the sink connection is shared territory with Idea 5.
- **Wehlitz, Pavliotis, Schütte & Winkelmann, *Data-driven Reduction of Transfer Operators for
  Particle Clustering Dynamics* (2601.02932).** Exactly the method Idea 2(c) proposes — coarse-
  grained transfer operator, Markov states, implied timescales, transition-path analysis — for
  interacting particle systems with clustering, from Schütte's group. Not applied to transformers.
  This is a template to follow and cite, not a collision, but it removes "nobody has done this" as
  a claim.
- **Rigollet, *The Mean-Field Dynamics of Transformers* (2512.01868).** The successor survey to the
  paper Blog 1 is built on, with metastability, normalization effects on contraction speed, and a
  long-context phase transition. Whatever we write should be positioned against this rather than
  against the 2023 paper alone.

Also worth knowing about: Alcalde–Bungert–Riedl–Roith (2605.10931) on low-temperature
concentration; Massucco–Del Grande–Carioni–Brune–Schönlieb (2605.18870) on multi-head architectures
as time-dependent Wasserstein gradient flows; and *Testing the spin-bath view of self-attention*
(2507.00683), which is a sibling in genre — a physics framing tested against GPT-2's actual weights
— and reports a strong *negative* correlation, which is a useful precedent for how to publish that
outcome.

## Sequencing

Idea 4 is the cheapest and unblocks the others by pinning $\beta$; Idea 1 is the flagship but is
gated on P-γ2 and the replication gate; Idea 6 can be written today from artifacts already
committed and makes the rest more credible; Ideas 2, 3 and 5 are each a real quarter of work.
A defensible order is 6 → 4 → 1 → 5 → 3 → 2, with 6 and 4 as blog entries and the rest aimed at
workshops.

---

## References

**The source paper and its successors (Geshkovski / Polyanskiy / Rigollet and co-authors)**

- Geshkovski, Letrouit, Polyanskiy & Rigollet, *A Mathematical Perspective on Transformers*,
  [arXiv:2312.10794](https://arxiv.org/abs/2312.10794) — the paper this project tests.
- Geshkovski, Letrouit, Polyanskiy & Rigollet, *The Emergence of Clusters in Self-Attention
  Dynamics*, [arXiv:2305.05465](https://arxiv.org/abs/2305.05465), NeurIPS 2023.
- Geshkovski, Koubbi, Polyanskiy & Rigollet, *Dynamic Metastability in the Self-Attention Model*,
  [arXiv:2410.06833](https://arxiv.org/abs/2410.06833) — proves dynamic metastability: particles
  stay trapped near multi-cluster configurations for exponentially long before collapsing. The
  direct successor on Problem 1, and the first thing to read against our plateau results.
- Karagodin, Polyanskiy & Rigollet, *Clustering in Causal Attention Masking*,
  [arXiv:2411.04990](https://arxiv.org/abs/2411.04990), NeurIPS 2024 — causal masking; explicitly
  *not* a mean-field gradient flow; Rényi parking problem.
- Karagodin, Ge, Polyanskiy & Rigollet, *Normalization in Attention Dynamics*,
  [arXiv:2510.22026](https://arxiv.org/abs/2510.22026), NeurIPS 2025 — normalization as speed
  regulation across Post-LN / Pre-LN / Mix-LN / Peri-LN / nGPT / LN-Scaling.
- Chen, Lin, Polyanskiy & Rigollet, *Critical Attention Scaling in Long-Context Transformers*,
  [arXiv:2510.05554](https://arxiv.org/abs/2510.05554) — the critical scaling $\beta_n \asymp \log n$.
- Duerinckx, Geshkovski & Rossi, *Kinetic Theory for Transformers and the Lost-in-the-Middle
  Phenomenon*, [arXiv:2605.09213](https://arxiv.org/abs/2605.09213).
- Geshkovski, Rigollet & Ruiz-Balet, *Measure-to-Measure Interpolation Using Transformers*,
  [arXiv:2411.04551](https://arxiv.org/abs/2411.04551).
- Rigollet, *The Mean-Field Dynamics of Transformers*,
  [arXiv:2512.01868](https://arxiv.org/abs/2512.01868).

**Adjacent dynamics and mean-field analysis**

- Bruno, Pasqualotto & Agazzi, *Emergence of Meta-stable Clustering in Mean-field Transformer
  Models*, [arXiv:2410.23228](https://arxiv.org/abs/2410.23228), ICLR 2025.
- Isobe, Inoue & Imaizumi, *Training-Induced Escape from Token Clustering in a Mean-Field
  Formulation of Transformers*, [arXiv:2605.07772](https://arxiv.org/abs/2605.07772).
- Alcalde, Bungert, Riedl & Roith, *Quantifying Concentration Phenomena of Mean-Field Transformers
  in the Low-Temperature Regime*, [arXiv:2605.10931](https://arxiv.org/abs/2605.10931).
- Castin, Ablin, Carrillo & Peyré, *A Unified Perspective on the Dynamics of Deep Transformers*,
  [arXiv:2501.18322](https://arxiv.org/abs/2501.18322) — the Transformer PDE; multi-head, L2,
  Sinkhorn, sigmoid and masked attention in one frame.
- Castin, Ablin & Peyré, *How Smooth Is Attention?*,
  [arXiv:2312.14820](https://arxiv.org/abs/2312.14820), ICML 2024.
- Sander, Ablin, Blondel & Peyré, *Sinkformers: Transformers with Doubly Stochastic Attention*,
  [arXiv:2110.11773](https://arxiv.org/abs/2110.11773) — doubly-stochastic attention *is* a
  Wasserstein gradient flow; the paper's Remark 3.5 open question.
- Massucco, Del Grande, Carioni, Brune & Schönlieb, *Multi-Headed Transformer Architectures as
  Time-dependent Wasserstein Gradient Flows*,
  [arXiv:2605.18870](https://arxiv.org/abs/2605.18870).
- Ye, *Self-Attention Dynamics with Rotary Position Embeddings: Twisted States and Explicit
  Consensus Rates on the Sphere*, [arXiv:2607.24502](https://arxiv.org/abs/2607.24502).

**Collapse, rank, and sinks**

- Dong, Cordonnier & Loukas, *Attention Is Not All You Need: Pure Attention Loses Rank Doubly
  Exponentially with Depth*, [arXiv:2103.03404](https://arxiv.org/abs/2103.03404), ICML 2021 — the
  closest existing statement of our collapse result; position against it explicitly.
- Súkeník, López Amado, Lampert & Mondelli, *Sink vs. Diagonal Patterns as Mechanisms for Attention
  Switch and Oversmoothing Prevention*, [arXiv:2605.08453](https://arxiv.org/abs/2605.08453).
- Xiao, Tian, Chen, Han & Lewis, *Efficient Streaming Language Models with Attention Sinks*,
  [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).
- Gu et al., *When Attention Sink Emerges in Language Models: An Empirical View*,
  [arXiv:2410.10781](https://arxiv.org/abs/2410.10781), ICLR 2025.

**Tools from other fields**

- Trefethen & Embree, *Spectra and Pseudospectra* (Princeton, 2005) — non-normality, transient
  growth, the Kreiss constant, why eigenvalues mislead.
- Li, *Transient Reserves, Sink Dampers, and the Failure of Eigenvalue Reasoning in the Attention
  Propagator*, [arXiv:2607.09279](https://arxiv.org/abs/2607.09279).
- Wehlitz, Pavliotis, Schütte & Winkelmann, *Data-driven Reduction of Transfer Operators for
  Particle Clustering Dynamics*, [arXiv:2601.02932](https://arxiv.org/abs/2601.02932).
- Deuflhard & Weber (PCCA+) and Schütte's transfer-operator metastability; Bovier–Gayrard–Klein for
  Eyring–Kramers; Edelman–Kostlan–Shub and the elliptic law (Sommers et al.) for the random-matrix
  nulls; Peyré & Cuturi for computational OT. *(Classic references — cite from the originals, not
  from this file.)*

**Interpretability side**

- Olsson, Elhage, Nanda et al., *In-Context Learning and Induction Heads*,
  [arXiv:2209.11895](https://arxiv.org/abs/2209.11895) — induction heads form abruptly, at the same
  point as the in-context-learning bump. The time axis Idea 3 races the motif against.
- Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and
  Scaling*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373) — the checkpoint suite the whole
  transition rests on.
- Wurgaft, Rager, Kowal et al., *Manifold Steering Reveals the Shared Geometry of Neural Network
  Representation and Behavior*, [arXiv:2605.05115](https://arxiv.org/abs/2605.05115) — already used
  in `p5b_manifold_steering/math-5b.md`.
- *Testing the Spin-Bath View of Self-Attention: A Hamiltonian Analysis of GPT-2*,
  [arXiv:2507.00683](https://arxiv.org/abs/2507.00683) — a sibling in genre, with a negative result.
