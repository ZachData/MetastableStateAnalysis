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

### Addendum, 2026-08-24 — what "reproduces" means for claim (c)

Claim (c)'s prediction text above is unchanged and stays unchanged. What it never stated is
the *criterion*: which statistics count as phenomenology, and what counts as reproduction.
That is now fixed, in `p1_mstate_tracking/replication_gate.py` and recorded in
`claims/registry.json` under `CLAIM-C`'s `null_construction` — **before the gate has run**,
which is the only time it can honestly be chosen.

"Reproduces" is read as **sign-concordance of the trained-minus-random contrast**, over the
six per-layer series `CHECKPOINT_METRICS` registers, on a common normalized-depth grid. The
null permutes the trained/random condition label on the pythia side with gpt2-large held
fixed, one sign flip per prompt, enumerated exhaustively.

Three consequences worth having in this file rather than only in the registry:

- **The criterion adjudicates the contrast, not the two absolute reproductions the sentence
  above names.** A pythia pair whose levels both sit far from gpt2-large's, but whose
  *difference* points the same way, passes. Blog 1's result was a contrast, so that is the
  object required to transfer — but it means the criterion is scale-blind, and the absolute
  per-arm distances are reported as a diagnostic that enters no p-value.
- **The two-baseline policy is honoured but asymmetric.** The p-value runs on the norm-matched
  `pythia-1.4b-random`, which is what the sentence above names. The true step-0 init is a
  mandatory sensitivity arm — the code refuses to run without it or a written reason for its
  absence — reported beside the result and kept out of the p-value, because step 0 is claim
  (a)'s object and one dataset must not settle two entries.
- **The hard stop is now three-way.** It fires on both a demonstrated inversion and an
  inconclusive gate — the sweep does not proceed on an unadjudicated gate — but only the
  inversion is recorded as a falsification of claim (c). "The gate was underpowered" and "the
  phenomenology does not transfer" are different findings and the ledger distinguishes them.

**Second addendum, same day — agreement across tools as well as across architectures.**
The criterion now also requires agreement across *metrics*: the cross-architecture test is
re-run once per metric-leave-one-out subset, and the gate requires unanimity in both
directions. Confirmation and falsification each get harder and the inconclusive middle grows;
the hard stop still fires on inconclusive, so only the word *falsified* is reserved for an
inversion that no single metric is carrying. This was recorded before the gate ran, which is
the only time it could be added honestly — `null_construction` freezes at the first
adjudication, and there has been none.

Claims (a) and (b) still have no criterion of their own. `CLAIM-A` and `CLAIM-B` remain
`needs-null`.

**Third addendum, 2026-08-24 — claim (b) now has one, and it is shared with Phase 7's
`P-I1`.** `core/changepoint_colocation.py` reads "co-locate" as: locate each series' change
as the centroid of its change-mass profile on the log-step axis, and ask whether two such
locations sit closer together — and closer to the ~512–2000 window this claim names — than a
matched control population allows. Four things are worth having in this file rather than only
in the registry.

- **The null this claim's entry named does not work, and it was measured rather than
  reasoned about.** "A permutation null over checkpoint order" rejects under H0 at 0.32–0.45,
  because permuting a series scatters its change across every interval and leaves the null
  with far too little variance next to two real, concentrated profiles. Even an enumerated
  circular shift, which is honest, reaches 0.103 once both series change early — as
  everything does early in training. The null used instead is a matched control series: the
  control for a series at one layer is that series at another layer, combined as a
  permutation of the *pairing* between the two series' layers, which holds each series' real
  locations fixed on both sides.
- **The estimator is not `detect_transitions`, and the floor is why.** Dividing by a log-step
  spacing that varies 4.6× across a Pythia sweep puts the null's argmax on the tightest
  interval 44.7% of the time when the value series is permuted against the fixed step grid, so a
  binary "the two top intervals coincide" statistic cannot
  reject at any sensible α. The change-mass centroid that replaced it needs no `n_top`, no
  `min_abs` and no tolerance — the first criterion in this project after CLAIM-C's
  sign-concordance to carry no placed constant at all.
- **The stop rule is three-way and the falsifier's asymmetry is a branch.** CO-LOCATES,
  RE-ANCHORS, INSUFFICIENT. This claim's falsifier says no co-location is *itself a real
  result*, so RE-ANCHORS records a separation positively shown against the controls, not one
  inferred from a failure to reject. Only the `greater` p enters claim (b)'s e-process.
- **The gate will most likely refuse, and the arithmetic was done before the pilot.** The two
  anchor arms have no relabeling available — nothing permutes "unrelated to the literature's
  anchors" — so each needs a reference population, and at α = 0.05 that means 19 control
  series measured on the same sweep at the same layers. A cheap-tier sweep measuring six
  metrics has six. That is a requirement on what the pilot must measure, not a result.

**Fourth addendum, 2026-08-25 — what claim (c)'s gate needs the pilot to measure.**
The gate was run on an input whose answer is known: one model as *both* the reference and
the candidate, so every cell is concordant and the correct verdict is TRANSFERS a priori.
The criterion passed that test — every metric-leave-one-out subset returns exactly the
attainable floor, so the unanimity axis does not bite on a unanimous input. What the sweep
found is an operating range, and it is a precondition on the run rather than a change to
the criterion:

- **At least 11 of the candidate's 48 (prompt × metric) cells must dissent in sign** — about
  23%, on average about 1.8 of the 8 prompts disagreeing on each of the six metrics. Above
  `sign_homogeneity` 0.7708 the homogeneity correction's derived refusal fires on *every*
  input including a perfect one, so neither TRANSFERS nor FAILS-TO-TRANSFER is reachable and
  the hard stop fires unconditionally. (9 cells and 0.8125 when first measured; the band
  tightened later the same day — see the second precondition below.)
- **This costs power, not validity, and it costs it where the effect is most uniform.**
  `sign_homogeneity` measures prompt redundancy under H0 and effect uniformity under H1, and
  the correction cannot separate them. Under independent prompt signs homogeneity sits at
  0.637 and the refusal essentially never fires — 0.0017 of that distribution sits above the
  boundary at eight prompts, 0.026 at six; a contrast pointing the same way on every prompt
  sits at 1.0 and is refused with certainty. More prompts do not move the boundary.
- **How much concordance the gate needs, so the number is known before forward passes are
  spent.** TRANSFERS becomes certain at 38 of 48 concordant cells at homogeneity 0.5833, 38 at
  0.7083 and 43 at 0.7708; falsification needs 14 or fewer. The band between them, where the
  gate says nothing and the hard stop fires, is 27 of the 49 possible counts at 0.7708.

- **And a second precondition, pointing the other way (2026-08-25).** At least five prompts
  must have usable metrics that do NOT split evenly. A prompt whose label flip does not
  change the statistic — every cell dropped, or an even number of usable cells splitting
  exactly half and half, which with six metrics is 3–3 and happens to 20/64 of rows under
  H0 — is enumerated through all 2^n null patterns and never counted, so the floor is
  2^-k in the k prompts that can move. Five is the first k that clears α = 0.05, at every
  prompt count. Before that was checked the gate could be handed a table perfect on four
  prompts and 3–3 on two, return p = 0.0769 — exactly that table's own floor — and report
  it as "not significant".
- **Adding that refusal is what moved the first precondition from 9 cells to 11.** It takes
  no verdict from any individual table — both tails share the floor, so a refused table
  could not have cleared α either way, and P(TRANSFERS) is unchanged to four decimals. What
  it changes is the denominator: every rate in the calibration curve is conditional on the
  gate emitting, and a draw that could never reject is no longer emitted, so the measured H0
  rate among the draws a ledger actually receives is higher and the correction is stronger.
  The band is the price; it is recorded rather than absorbed.

Measured by running the shipped gate, not by reasoning about it; recorded in
`claims/audits/claim_c_dry_run.json` and `POPPER_PLAN.md` §§6j and 6l. The criterion is
unchanged and nothing was adjudicated. §6l did change the gate — it added the
informative-row refusal above, and gave the homogeneity curve a cell-drop dimension so a
run that loses cells is corrected off a table measured on tables that lost cells, or
refused — but neither touches what counts as reproduction.

Recorded before any sweep exists, which is the only time it could be added honestly.

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

## Addendum, 2026-08-30 — `P-I3` has a null, and the registered one was not one

`claims/EVALUABILITY.md` named `P-I3` as the next `needs-null` row that names a matched
control. It is built: `p7_motifs/cross_head_gate.py`, calibrated by
`tools/calibrate_cross_head_association.py` into
`claims/calibration/cross_head_association.json`, `POPPER_PLAN.md` §6s. The registered
wording is unchanged. Four things specific to `P-I3`:

- **The registered null cannot be used, and the reason is the tautology risk this block
  already names.** "Permutation over the head classification" draws which *k* of the *n*
  heads are labelled induction. But an induction head is one whose behavioural induction
  score clears a cutoff, so the classification is a threshold on the very variable the
  prediction correlates against, and exactly **one** of those 1.09e16 label assignments is
  a classification the definition permits. Measured on both readings of the statistic, the
  registered null discriminates a genuine effect from its absence at −0.003 and −0.010 —
  and the slope reading is anti-conservative on plain H0 besides. The null used instead
  compares each induction head against control heads **matched on its own behavioural
  score**, straddled above and below, and permutes the induction label within a matched
  set only. That null enumerates, so the p-value is exact and no draw count enters.

- **Adjudication constraint 2 is now arithmetic rather than a caution — where the
  constraint permits it.** The constraint records that a motif defined as "an attentive
  edge on induction pairs" is the behavioural score wearing a different name, and `P-I1`'s
  gate could only refuse exactly-identical series. Matching on the score is what removes
  the shared component, so when the classification IS the thresholded score no induction
  head can be straddled, no matched set is informative, and the design floor is **1.000**:
  the gate refuses before an edge is counted. The leak it catches is proportional to how
  hard the motif *tracks* the score rather than to a literal identity, which is a larger
  family than the constraint's wording describes. `independence_source` remains required
  and remains the analyst's claim: what the floor discharges is the measurement of one
  quantity twice, not the argument that the motif is independent for the right reason.

- **The `statement` and the `falsifier` name different quantities, and the p-value carries
  the falsifier's.** "Correlates with behavioral induction score" is a within-group
  association; "non-induction heads carry the motif at the same rate" is a level. One
  number cannot carry both, and the association reading is not adjudicable anyway: on ONE
  population with ONE relation the within-group correlations read 0.255 among induction
  heads and 0.673 among the rest — a contrast of −0.417 with no interaction present, in
  the falsifier's own direction. Both correlations are reported beside every result and
  enter no p-value, exactly as `P-I1`'s endpoint counts do.

- **The control's matching key is a registered decision, and it is a trade.** Induction
  heads cluster in a band of layers, and a shared elevation across that band is invisible
  to a control matched on score alone (0.440 at one standard deviation) and removed exactly
  by one drawn from the induction head's own layer (0.011, flat). The author registered
  `"score_and_layer"` on 2026-08-30; it costs roughly a quarter of the informative sets, a
  p-value on 0.657 of runs rather than all of them, and about half the power. Adjudication
  refuses a result computed under the other key.

`P-I3` is reclassified `e-value` in `claims/registry.json`. No falsifier has been weakened
and no threshold has been moved.

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

## Addendum, 2026-08-24 — `P-I1` has a null, and it is `CLAIM-B`'s

`claims/EVALUABILITY.md` had named `P-I1` and `CLAIM-B` as needing the same changepoint
co-location construction and said they should be built together rather than each inventing
one. They were: `core/changepoint_colocation.py` is the construction and
`p7_motifs/formation_gate.py` is this prediction's half. See the third addendum to the
transition-project block above for the estimator, the null, and why the registered wording's
"permutation over checkpoint order" was measured to be invalid.

Four things specific to `P-I1`:

- **The unit is the head, enforced rather than asserted.** Adjudication constraint 1 above
  fixes it; the null permutes which head's motif curve is matched with which head's
  behavioral curve, so an edge-level *n* has no way into the arithmetic.
- **One arm, and no anchor arm.** `P-I1` names no literature anchor — it asks only that the
  two curves rise together — so nothing is invented for it to test. That also makes `P-I1`
  the likelier of the two entries to produce a number, since the anchor arms are what refuse
  on control-set size.
- **The falsifier's second half is a precondition, not a p-value.** "Motif already above
  nulls at step 0, or absent at step 143,000 despite a high behavioral score" is about the
  curve's endpoints and the statistic is about where it rises; one number cannot carry both.
  Both are reported as per-head counts beside the result and enter no p-value. Because the
  series handed in is the above-null *excess*, zero is the null envelope and neither check
  needs a threshold of its own.
- **Constraint 2's tautology risk cannot be discharged by any null here, and now has a number
  behind it.** Two identical series co-locate perfectly, and a tautological pair is
  tautological at every head, so no permutation over the pairing detects it. The gate refuses
  exactly-identical series per head — the degenerate case only — and the measured version of
  the same problem is that a common per-head factor unrelated to the claim produces a
  rejection rate of 1.00 against 0.05. The independence source remains a claim the analyst
  must state in the record, exactly as constraint 2 requires.

## Addendum, 2026-08-25 — `P-ST1` has a construction, and it is the entry that can lose

`p7_motifs/steering_gate.py`. `P-ST1` is the only registered bridge prediction where the
particle and standard accounts make INCOMPATIBLE rather than merely different predictions,
which is why it was built before entries with more apparatus behind them. Five things
specific to it, all fixed before any activation was steered:

- **Two of them were put to the author first, because they change what the prediction
  means.** Steering adds `α·v` to every token, so its whole effect on effective rank is a
  MEAN effect — re-centring after injection annihilates it exactly, as algebra rather than
  simulation. That makes the token cloud's pre-existing mean offset a competitor to the
  injected one: measured, the design as literally worded has no power on a realistic
  residual stream, and at a mean offset of five spreads it rejects more often under H0 than
  under H1. The gate therefore removes the BASELINE mean before injecting and keeps the
  injected offset. Second, α — the injection scale — was a third placed constant this file
  never flagged, and it decides whether the prediction is readable at all: there is a
  plateau at 0.17–0.24 × the population's spread, with the statistic identically zero below
  and direction-independent above. One α is pre-registered and labelled `placed` per
  adjudication constraint 4; the fraction is placed, the scale it multiplies is derived, and
  the α-profile is reported with every result and enters no p-value.
- **"Predominantly" is removed rather than thresholded.** Each arm is drawn uniformly from
  its subspace — 100% by construction, not 60% by cut — so one of the two constants this
  file's `null_construction` flagged no longer exists. Norm matching is likewise by
  construction: both arms of a pair get the same α.
- **The registered null does not hold and was replaced.** Permuting the decomposition label
  across pairs treats *m* pairs as *m* exchangeable units, and every pair at one layer
  shares the tokens and both subspaces; the H0 rejection rate grows from nominal at 8 pairs
  to 0.17–0.22 at 150. What replaces it is a matched-dimension random ORTHOGONAL subspace
  pair, the construction `P6-R2` uses. The registered permutation is still computed and
  reported beside every result, never adjudicated.
- **The registered falsifier is not one an e-process can carry.** "Both arms move effective
  rank the same way, or the effect tracks ‖s‖" describes the NULL in both clauses, and an
  e-process records insufficient evidence and never a null accepted. It maps to
  INSUFFICIENT. The falsification branch is INVERTS — attractive-dominant steering
  demonstrably RAISING effective rank — a reversal positively shown, checked to be a branch
  that can fire.
- **A precondition on the pilot, computed before it runs.** A uniform draw from `U_pos`
  carries only dim(occupied)/dim(`U_pos`) of its energy into the subspace the cloud
  occupies; the per-pair informative rate falls from 1.000 at ratio 1 to 0.000 at ratio 6,
  and the projector audit already records `U_pos` as the un-shrunk bucket. The pilot must
  report dim `U_pos` at the injection layer against the population's effective rank.

Recorded before any activation exists, which is the only time it could be added honestly.

## Addendum, 2026-08-26 — `P-ST1` run on inputs whose answer is known, and a second null retired

`tools/dry_run_p_st1.py` → `claims/audits/p_st1_dry_run.json`;
`POPPER_PLAN.md` §6m. `claims/EVALUABILITY.md` had said every converted row is
owed a run on an input whose answer is known ahead of converting the next one, and
`CLAIM-C` was the first. This is the second, and it changed the gate twice.

- **The null this entry adjudicated does not hold either, and the family that shows
  it is the realistic one.** The matched-dimension random orthogonal pair holds the
  DIMENSIONS fixed and not how much of the population each subspace contains — which is
  what a change in effective rank is driven by. `U_pos` and `U_neg` are cut from the model's
  own OV eigenstructure and a residual stream is orthogonal to neither, so both hold more
  than the k/d a random subspace holds, and such a pair is unusual against random pairs
  whichever arm is called attractive. Measured where both arms are occupied above chance
  and the two are IDENTICAL by construction — so a label swap is a distributional identity
  and INSUFFICIENT is the only correct verdict — it rejects at up to 0.20 against a nominal 0.05, the inflation growing with the pair count. All three H0 families the 2026-08-25 calibration measured put the cloud in a subspace
  ORTHOGONAL to both arms, the one case where that null is exchangeable with the observed
  pair, so nothing could have seen it.
- **What replaces it randomises the SPLIT and not the subspaces**, and that is the first
  time this project's recurring question — what is being randomised — has been answered by
  randomising less. Draw a uniformly random k_pos-dimensional subspace of
  span(`U_pos` + `U_neg`) and take its orthogonal complement inside that union. Dimensions,
  orthogonality, occupancy and the whole spectral relationship to the layer's cloud are held
  exactly fixed, and the observed split is one point of the same Grassmannian the null draws
  from — so exchangeability under H0 is by construction rather than by measurement. It costs no power where the cloud fills the whole arm and costs it as
  dim `U_pos` grows past the dimension the population occupies; power lost that way was
  never power about the decomposition.
- **The reported floor was not the attainable one.** `sum(D)` cannot exceed 2m, and on an
  occupied union many random re-splits already reach 2m and tie it, so the smallest
  expressible p is a fact about the layer: 0.11–0.17 on a perfect input at one pair with 99 draws, where
  `1/(draws+1)` says 0.01. The gate now computes both tails' floors from the null it already
  has and REFUSES when neither reaches alpha — a design that could not have rejected was
  returning "not significant", which is CLAIM-C's §6l defect arriving here. 2m is an upper
  bound on the observation, so the refusal turns away nothing that could have cleared alpha.
- **A run can have only its FALSIFICATION branch reachable.** The two tails' floors are
  computed separately and are not equal, so one can be out of reach while the other is not;
  where the reachable one is INVERTS, the design can return a falsification or nothing. The
  gate does not refuse — one reachable tail is one reachable verdict — and records
  `reachable_tails` instead, because a run that can only produce the branch which enters the
  ledger as a falsification is one a reader must be told about. Two pairs is the smallest
  design that emits at all: at one pair, on an input planted perfectly in either arm, both
  floors sit at 0.11–0.17 and the gate refuses.
- **Each arm's chance-normalized occupancy is now reported and costs no injection to
  compute** — its share of the centred population's energy divided by the k/d a random
  subspace of that dimension holds, which is `POPPER_PLAN.md` §6h's chance normalization with
  the population in place of a single vector. The pilot can read it off the activations and
  the two projectors before spending a sweep, and a reader can see what a TRACKS verdict is
  made of. It does not determine the verdict on its own; the dry run reports how far it goes.

The registered wording is unchanged and no falsifier has been weakened. What changed is the
null the wording's successor named, and it changed in the conservative direction.

## Status

Not yet adjudicated. No Phase 7 *results* exist; `P-I1`'s gate exists as of 2026-08-24 and
`P-ST1`'s as of 2026-08-25, `P-AB1`'s as of 2026-08-27 and `P-I3`'s as of 2026-08-30, and
all of them emit nothing — no checkpoint sweep of motif strength has been run, no head
table of motif rates and behavioural induction scores exists, and no activations or
Phase 2 projectors exist in this repository.

**Addendum, 2026-09-04 — `P-I1` has emitted a p-value, and it is
INSUFFICIENT.** This paragraph is unchanged above because it is now stale for
`P-I1` specifically and not for the others, which this pass did not touch.
`tools/score_p_i1.py`, on the real 19-step sweep, the real behavioural series,
and the relay-count null `PROJECT.md` §3.4 registers: **p = 0.1414**,
attainable floor 0.0005, 116 heads scored (0 skipped — every forming head's
above-null excess still located a rise), verdict **INSUFFICIENT**. Neither
endpoint failure mode fires: 0 heads above-null at step 0, and the 2 heads
absent at step 143000 do not include any with a high behavioural score. Not
adjudicated — `claims/adjudications/` is untouched, and entering this there is
the author's decision, not this run's. `POPPER_PLAN.md` §6v and `PROJECT.md`
§3 carry the construction and the full number set.
