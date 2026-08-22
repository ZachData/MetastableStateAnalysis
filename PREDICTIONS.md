# PREDICTIONS.md — Pythia Transition, Project-Level Falsification Record

This file is distinct from any phase's own falsification table (those live in each phase's
`status-N.md`). This one covers the transition project's own claims — the things the move
from a multi-architecture GPT-2/BERT/ALBERT study to a Pythia checkpoint study is supposed to
show. Written and committed **before the replication gate (execution-order item 6) runs**, so
the timestamp on this file precedes any result it's checked against. Don't edit the
predictions after seeing gate results — if a prediction needs revising, add a dated
addendum below it rather than changing the original text.

## Why these three claims, specifically

The whole point of moving to a checkpoint suite is to ask whether what Blog 1 found on
GPT-2-large — trained weights resisting the collapse the architecture drives, with a
large, informative unclustered population — is (a) something training does or something
architecture already gives you for free, (b) tied to the same circuit-formation events the
literature already anchors checkpoint schedules to, and (c) actually the same phenomenon on
a different architecture rather than a GPT-2-large idiosyncrasy the whole project has been
building on without knowing it. Every one of these has a clean failure mode that would change
what the rest of the plan is even for — which is why they're gated (claim (c) specifically
stops the sweep if it fails) rather than just noted and moved past.

## The three claims

| Claim | Prediction | Failure reading |
|---|---|---|
| (a) Collapse-resistance is learned, not initial | Steps 0 and 8 look "random-like": monotone energy, rank collapse, high stationary Fiedler | Resistance is partly architectural/init-borne; the trained-vs-random contrast (the load-bearing comparison across Phases 1, 2c, and 5c) needs restating for what it actually isolates |
| (b) Resistance emerges at circuit-formation events | The energy-monotonicity break and Fiedler drop co-locate with steps ~512–2,000 (the Pythia-410M pilot, execution-order item 8, tests this directly rather than assuming it) | Clustering dynamics and circuit formation are independent — itself a real result, and it re-anchors the 1.4B checkpoint schedule rather than invalidating the sweep |
| (c) Phenomenology transfers across architecture | `pythia-1.4b-random` (norm-matched to the final trained checkpoint) reproduces `gpt2-large-random` phenomenology; trained checkpoint 143,000 reproduces trained `gpt2-large` phenomenology | The Blog 1 contrast is architecture-dependent, not a general property of trained transformers. **Stop and re-baseline before any checkpoint sweep** — this is the one claim with a hard stop attached, since every downstream phase's Pythia rerun assumes this transfer holds |

## How each claim gets adjudicated, and by what

- **(a)** — the replication gate (item 6): Phase 1 run at Pythia step 0 and step 8, checked
  against the same pass criteria Blog 1 established for gpt2-large-random (monotone energy,
  rank collapse, high stationary Fiedler). This is a cheap-tier check, no expensive-tier
  compute needed.
- **(b)** — the Pythia-410M pilot (item 8): a dense 20–30 checkpoint sweep through the cheap
  tier, explicitly built to test co-location of the energy/Fiedler/effective-rank transitions
  against the circuit-formation-event literature's checkpoint anchors, not to assume the
  anchors are right and just fill in data around them. A failure here is informative on its
  own terms (see table) and directly changes the 1.4B anchor schedule rather than being
  treated as a pilot that "didn't work."
- **(c)** — also the replication gate (item 6), using both the true step-0 init and the
  norm-matched `pythia-1.4b-random` as the two separate objects the plan's two-baseline
  policy requires (see Phase 5c's `design-5c.md` for why these can't be collapsed into one
  "random" condition). **If this fails, no checkpoint-sweep work (items 9–11) proceeds past
  the gate.**

## Status

Not yet adjudicated — the replication gate (item 6) has not run. This file exists to make
sure that when it does, the prediction was on record first.

---

# Phase 1c registered predictions

Committed **before any Phase 1c code is written or run**, for the same reason the three claims
above were: these are re-analyses of artifacts already on disk, which makes it trivially easy
to fit the prediction to the data afterward. Same rule applies — no edits after seeing
results, dated addenda only.

All six follow from re-reading Geshkovski et al. (arXiv:2312.10794v5) against what this
project already measures; the derivations are in `MATH.md` (paper results marked **[P]**, ours
**[U]**). None of them require a forward pass except where noted.

| ID | Prediction | Falsifier | Instrument | Cost |
|---|---|---|---|---|
| **P-γ1** | Layer-wise `ip_mean` on prompts with $n \ll d$ tracks $\gamma_{\beta_{\rm eff}}(T_{\rm eff}(\ell))$ from (6.9) at step 0, and departs from it monotonically with training | The residual at step 0 is already as large as at step 143k | 1c-B | [R+W] |
| **P-γ2** | $T_{\rm eff} \ll t^\ast \approx 4.2$ for all prompts and all checkpoints — the network never integrates far enough to collapse | $T_{\rm eff} \gtrsim 4$ with no collapse observed, i.e. genuine resistance | 1c-A | [R] |
| **P-H1** | Layer-0 tokens lie in an open hemisphere at every checkpoint (Wendel gives probability 1 for $d > n$, which all 8 prompts satisfy) | Infeasible at some checkpoint ⇒ the embedding layer actively breaks the cone condition | 1c-E | [R] |
| **P-S1** | Trained cluster centroids are closer to a spherical $t$-design than step-0 centroids — low-order Gegenbauer moments smaller | No difference between trained and step-0 ⇒ the repulsive-limit story is unsupported | 1c-F | [R] |
| **P-T1** | Heads classified $\lambda_1(V) > 0$ simple show trimodal $\langle \varphi_1, x_i\rangle$ (Table 1, row 2: three parallel hyperplanes) | Unimodal ⇒ Table 1 does not transfer past the $z_i = e^{-tV}x_i$ rescaling | 2d-D3 | [R+W] |
| **P-M1** | Energy-monotonicity violations concentrate in heads far from $Q^\top K$ symmetric and $V = Q^\top K$ | No correlation ⇒ violations are not explained by leaving the gradient-flow regime | 2d-D1 | [W] |

## Why each one is worth pre-committing

**P-γ2 first, because it can invalidate a headline.** Blog 1's claim is that trained weights
resist collapse. That claim silently compares the observed state against $t = \infty$. A
residual block is a forward-Euler step of the paper's ODE with step size $h_\ell =
\|P^\perp_{x_\ell}(\Delta x_\ell)\|/\|x_\ell\|$, so the right comparison is against
$\gamma_\beta(T_{\rm eff})$ with $T_{\rm eff} = \sum_\ell h_\ell$. Integrating (6.9) at
$n = 467$ puts $\gamma_\beta = 0.9$ at $t^\ast \approx 4.2$, essentially independent of
$\beta$ across two decades. If a 24-layer network accumulates $T_{\rm eff} \ll 4$, part of
what we have been calling resistance is depth. The prediction is stated in the direction that
would *hurt* us, deliberately.

**P-γ1 is the residual, not the fit.** The deliverable is not "does the curve match" but the
shape of the gap between observed `ip_mean` and the identity-weight trajectory run for the
observed amount of time. That gap is the part of the layer-wise dynamics learned weights are
responsible for, and it is the only version of "resistance" that is a measured quantity
rather than a comparison against an idealization.

**P-H1 is stated as near-certain on purpose.** Wendel's theorem (Thm 6.7) gives probability 1
for $d > n$, and Lemma 6.4's proof uses *only* positivity of the attention weights — which
softmax guarantees — so its hypothesis is entirely a condition on the initial configuration.
The interesting outcome is therefore the failure or the near-failure: infeasibility, or
feasibility with a margin $\max_w \min_i \langle x_i, w\rangle$ near zero, would mean the
embedding layer is doing something specific to escape a regime that otherwise forces
exponential collapse. Report the margin, not just the boolean.

**P-S1 gives "resistance" a target geometry.** If the trained model sits in the repulsive
regime, §9.1 does not predict a diffuse spread — it predicts a *sharp configuration* (Def.
9.1): few distinct pairwise inner products, and a spherical $(2m{-}1)$-design. Both halves are
checkable on centroids, and the second reduces on the sphere to Gegenbauer moments
$\frac{1}{n^2}\sum_{ij} C_k^\lambda(\langle x_i, x_j\rangle)$ vanishing for $1 \le k \le t$.
This is the first time the project's central empirical claim has a named limit object.

**P-T1 is the most falsifiable prediction available anywhere in the paper**, and it costs a
projection and a histogram. Table 1 (§9.2) maps a classification we *already have* — the sign
and multiplicity of $\lambda_1(V)$ per head, from `p2_eigenspectra` — onto a geometric
statement about activations. Row 2 says a real, simple, positive top eigenvalue predicts
concentration on three parallel hyperplanes normal to $\varphi_1$, i.e. trimodality of
$\langle \varphi_1, x_i\rangle$.

**P-M1 converts a falsification into a localization.** §3.4 makes (SA) a gradient flow in the
reweighted metric only when $Q^\top K$ is symmetric *and* $V = Q^\top K$. Heads meeting both
conditions **must** show monotone $E_\beta$; heads far outside carry no guarantee. So the
right question is not whether the theorem is violated but whether the violations sit where the
hypotheses fail. `p2b_imaginary/rotational_schur.py` already performs the
symmetric/antisymmetric split this needs.

## Adjudication order

P-γ2 and P-γ1 first (1c-A, 1c-B): lowest cost, and the $T_{\rm eff}$ result determines whether
the energy-monotonicity break is even the right thing to attribute, which gates Phase 2d. Then
P-H1, P-S1. P-T1 and P-M1 last, since both need Phase 2 operators.

---

## Addendum — P-T1 amended

**Recorded when the Phase 2d code was written, before any run.**

P-T1 as registered above reads:

> Heads classified $\lambda_1(V) > 0$ simple show trimodal
> $\langle\varphi_1, x_i\rangle$ (Table 1, row 2)

**That is not row 2's hypothesis.** Table 1's second row requires two conditions, and the
registered wording carries only the first:

| | condition | in the registered wording? |
|---|---|---|
| on $V$ | $\lambda_1(V) > 0$, simple | yes |
| on $QK$ | $\langle Q\varphi_1, K\varphi_1\rangle > 0$, i.e. $\varphi_1^\top M_h \varphi_1 > 0$ | **no** |

A head with a positive simple top eigenvalue but a negative QK form is not in row 2 at all.
Testing it against row 2's conclusion would falsify a prediction the paper does not make —
which is structurally the same error as the "Theorem 6.1: higher $d$ → faster convergence"
verdict row this very update cycle retracted. Making it twice, in the same document, is worth
recording rather than quietly fixing.

**Amended statement.** Heads satisfying **both** row-2 conditions show trimodal
$\langle\varphi_1, x_i\rangle$, with the three modes approximately equally spaced (the
prediction is three *parallel* hyperplanes, so spacing regularity is part of it and the
original wording omitted that too).

**Amended falsifier.** Predominantly unimodal among heads meeting both conditions ⇒ Table 1's
geometry does not transfer past the $z_i = e^{-tV}x_i$ rescaling. Before concluding that,
check `rescaled_modality` at several candidate $t$: if trimodality appears at some $t > 0$ and
not at $t = 0$, the structure is real and the rescaling is what hides it, which is a different
conclusion.

**Two adjudication constraints added at the same time, for the same reason — the original
wording did not specify them and either could have been chosen after seeing results:**

1. **A control arm is required.** Report the trimodal rate among row-2 candidates *and* among
   non-candidates. If non-candidates are trimodal at the same rate, trimodality is a property
   of the activations rather than of the classification, and a candidates-only number would
   read as confirmation.
2. **Adjudicate on `stable_n_modes` only** — the mode count that survives a bandwidth scan.
   Any distribution can be made unimodal by over-smoothing and multimodal by under-smoothing,
   so a mode count at a single bandwidth is a choice, not a measurement. `None` (no stable
   count) is a legitimate outcome and must be reported as such rather than resolved.

The `row2_candidate` flag in `p2d_operator_activation/table1_predictions.py` implements the
amended condition; `row2_eigen_only_qk_fails` labels heads that would have been counted under
the original wording, so the size of the error is recoverable from the output.

**What is NOT amended.** The direction of the prediction is unchanged, and no falsifier has
been weakened. If anything the amended version is harder to satisfy: it requires two operator
conditions instead of one, equal spacing, a control arm, and bandwidth stability.

---

## Status

Not yet adjudicated. Phase 1c and Phase 2d code exists and is validated on synthetic data; no
sub-experiment has been run against Pythia artifacts.

---

# Phase 7 registered predictions — induction-head formation as a particle motif

**Recorded 2026-08-22, before any Phase 7 code is written.** Same rule as the Phase 1c block
above and for the same reason: this is an analysis of a phenomenon the mechinterp literature
has already described qualitatively, which makes it trivially easy to fit a motif definition
to the data after seeing it. No edits after results — dated addenda only.

## What is being claimed

Phase 7's thesis is that a named mechinterp phenomenon is a **recurring structure of particle
interactions** — a motif — and not merely describable as one. The first phenomenon is the
induction head, restated as a two-stage `relay` motif: a `prev_token` edge at layer ℓ₁ whose
target particle becomes the source of a `match` edge at layer ℓ₂ > ℓ₁. See
`p7_motifs/design-7.md` for the full alphabet and the interaction-object definition.

All four predictions are adjudicated against the offset nulls in `core/qk_offset_null.py`
(N1 rotary-only, N2 offset-matched, N3 offset-shuffled). **A pass requires clearing N1 and
N2.** No prediction below is evaluable on a prompt that `core/battery_structure.py` flags
degenerate (`uniform`, `empty_null`, `single_offset`, `null_is_sink`) — on those the phase
refuses rather than returning a number.

| ID | Prediction | Falsifier | Instrument | Cost |
|---|---|---|---|---|
| **P-I1** | `relay` motif strength above N1 and N2 first rises in the same checkpoint window as the behavioral induction score | Motif already above nulls at step 0, or absent at step 143,000 despite a high behavioral score | 7-A (formation curve) | [F] |
| **P-I2** | The stage-2 `match` edge is attractive-channel dominant (`U_pos`); the stage-1 `prev_token` edge is offset-driven and channel-neutral | No channel difference between the two stages | 7-B (channel decomposition) | [R+W] |
| **P-I3** | Across heads at a fixed checkpoint, `relay` strength correlates with behavioral induction score — **and does not** among non-induction heads | Non-induction heads carry the motif at the same rate ⇒ the motif is a property of the activations, not of the classification | 7-C (cross-head, control arm) | [R] |
| **P-I4** | `relay_target` particles show `moved_fraction` attributable to the motif's edges, above a matched-magnitude control | Motif present in the attention pattern but moves nothing ⇒ routing artifact, not a dynamical structure | 7-D (event consequence) | [R] |

## Why each one is worth pre-committing

**P-I1's three outcomes are all informative, which is why it is stated as a window and not a
direction.** Motif-before-behavior would mean the geometric structure assembles before the
function it supports is measurable — the interesting case, and the one that would make the
particle account predictive rather than descriptive. Behavior-before-motif would mean the
motif is not what the behavior is made of. Simultaneity is the null-ish case and still
locates a circuit-formation event on the checkpoint axis, which is what `PREDICTIONS.md`
claim (b) needs and currently assumes from the literature rather than measures.

**P-I2 is what makes the two-stage decomposition earn its cost.** If both stages look alike
in the channel decomposition, then `relay` is a more expensive way of computing something a
single-head score already gives, and the alphabet should be simplified rather than defended.

**P-I3 is the prediction that can kill the bridge, and it is stated in the direction that
would hurt.** The control arm is mandatory, not optional: reporting the motif rate only among
induction heads would read as confirmation no matter what the number was. This is the same
error the P-T1 amendment was written to prevent, and it is being pre-empted here rather than
corrected after the fact.

**P-I4 separates a routing claim from a dynamical one.** The particle account's whole
advantage over attention-pattern analysis is that it tracks forces and motion, not
attribution weights. If the motif does not move particles, that advantage is not being used
and the result is a re-description of something mechinterp already measures better.

## Adjudication constraints, fixed now

1. **Effective *n* is the number of heads, not the number of edges.** Edges within a head are
   not independent samples. Any significance computed over edge counts is wrong by orders of
   magnitude, in the direction that manufactures findings.
2. **The tautology check is part of adjudication, not a code-review nicety.** The behavioral
   induction score is "mean attention on induction pairs"; a motif defined as "attentive edge
   on induction pairs" is the same number. Every P-I3 result must state which of the three
   independence sources (two-stage composition, force decomposition, particle event) is
   carrying the association. "None" means the phase measured one thing twice.
3. **N3 is reported but does not gate.** It separates "content and offset jointly required"
   from "either alone suffices" — a distinction about mechanism, not about whether the effect
   is real.
4. **Thresholds are labelled `placed` until derived from an observed distribution**
   (standing rule 6). This includes the `hub` in-degree cutoff and the top-k-by-force
   retention cutoff on the interaction table, neither of which has a calibrated value yet.

## Status

Not yet adjudicated. No Phase 7 code exists as of this entry — that is the point of the
timestamp.
