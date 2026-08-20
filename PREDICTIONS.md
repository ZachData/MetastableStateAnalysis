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

# Phase 1d registered predictions

Committed **before any Phase 1d code is written or run**, same rule as above: this is a
re-analysis of artifacts already on disk, and the outcome that would be most convenient — "the
other methods agree with HDBSCAN once tuned, so nothing we have published changes" — is
exactly the one it would be easiest to arrive at by choosing hyperparameters after seeing the
answer. No edits after results; dated addenda only.

## What Phase 1d is for

Every cluster-conditioned claim in this project rests on one clusterer at one setting:
`hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")` on cosine distance
(`p1_mstate_tracking/clustering.py`). Three other partitions of the same tokens are computed
and persisted, and `p1_visualization/cluster_methods.py` measures whether they agree — but
none of the four was ever tuned, and the agreement statistic therefore compares four sets of
library defaults, not four methods. Phase 1d tunes each family against criteria that do not
presuppose HDBSCAN's answer, and asks what a graded, method-independent annotation says that
the categorical label cannot.

| ID | Prediction | Falsifier | Instrument | Cost |
|---|---|---|---|---|
| **P-C1** | At layers where Phase 1 already reports method agreement (counts within ±1), the *tuned* ensemble's consensus strength stays ≥ 0.9 — tuning does not dissolve the agreement the defaults showed | Consensus strength drops under tuning ⇒ the reported cross-method agreement was a coincidence of library defaults, not a property of the geometry, and every "the methods agree" sentence in Phase 1 needs restating | 1d-B | [R] |
| **P-C2** | `min_cluster_size=2` is **not** the stability-optimal HDBSCAN setting: the selected setting is larger, and ARI between the tuned and the shipped HDBSCAN partition is < 0.9 at the majority of mid-depth layers | The shipped default is already stability-optimal ⇒ every existing HDBSCAN-conditioned result stands unchanged and this phase's premise is void — a real and welcome outcome, recorded as such | 1d-A | [R] |
| **P-C3** | The unclustered population is not homogeneous: within HDBSCAN noise, at least 20% of particles sit above the null-calibrated consensus-confidence threshold, i.e. other methods place them in structure HDBSCAN's density criterion refused | Noise-token confidence is indistinguishable from the matched null ⇒ "unclustered" is one population, and Phase 5c's binary clustered/unclustered split is the right object after all | 1d-C | [R] |
| **P-C4** | Graded consensus confidence beats the binary clustered/noise flag at predicting layer-to-layer cluster persistence: ΔAUC > 0 and outside the permutation null's 2σ band | No improvement ⇒ the categorical label loses nothing the ensemble recovers, and the conglomeration is presentation rather than information. **This is the prediction that decides whether the phase produced anything.** | 1d-D | [R] |

## Why each one is worth pre-committing

**P-C2 is stated against our own interest.** The convenient result is that the default was
right all along. The prediction says it was not, which — if it holds — means every downstream
cluster-conditioned number was computed on a partition nobody chose deliberately. Stating it
the other way round would make the phase unfalsifiable: any disagreement could be attributed
to a "badly tuned" alternative.

**P-C1 and P-C2 can both hold, and that combination is the interesting one.** The methods can
agree on the *coarse* partition (P-C1) while HDBSCAN's specific setting is wrong about the
*boundary* (P-C2). Registering them separately keeps that from being resolved by narrative
afterwards.

**P-C3 targets the phase whose object of study this is.** Phase 5c's framing — every particle,
clustering as an annotation — makes "unclustered" a population rather than a residue. If that
population turns out to have a structured sub-part visible to five other methods, the binary
tag is hiding it. The 20% floor is **placed, not calibrated**: there is no distribution to
derive it from before the phase runs, and it is written here rather than chosen later
precisely because it is arbitrary. The threshold it is measured against is calibrated (95th
percentile of the matched-null co-association); only the 20% is placed.

**P-C4 is the phase's own falsification.** Everything else Phase 1d produces is descriptive: a
prettier annotation is not a result. The claim that earns the phase is that the graded
annotation carries information the categorical one does not, and the only honest way to test
that is to have both predict the same held-out thing and compare. If ΔAUC lands inside the
null band, the correct write-up is "the ensemble adds nothing measurable" — and that sentence
is easier to write with the prediction already on record.

## Adjudication constraints, fixed now

1. **One vote per family.** The ensemble takes one tuned partition from each method family.
   Six agglomerative linkages voting against one HDBSCAN is a rigged consensus, and which
   families are included must not be a post-hoc choice.
2. **The comparison partition for P-C2 is the shipped one**, `min_cluster_size=2` with library
   defaults, not a re-run with different defaults.
3. **A family that fails the null gate at a layer does not vote there**, and the fact that it
   abstained is reported alongside the consensus, not silently absorbed into it.
4. **P-C4's persistence target is computed from the consensus partition, not from HDBSCAN**,
   and the same target is used for both predictors. Scoring the graded annotation against a
   target HDBSCAN defined would be a rigged comparison in the other direction.

---

## Status

Not yet adjudicated. Phase 1c, Phase 1d and Phase 2d code exists and is validated on synthetic
data; no sub-experiment has been run against Pythia artifacts.
