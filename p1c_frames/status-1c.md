# Phase 1c — STATUS

**State:** all six sub-experiments implemented and validated on synthetic data and on
configurations with known exact answers, with a driver (`run_1c.py`) and artifact IO
(`p1c_io.py`) that have been run end to end against a synthetic Phase-1 run directory. **Not yet run against Pythia artifacts** — no result
rows below, by design. Predictions P-γ1, P-γ2, P-H1 and P-S1 were registered in
`PREDICTIONS.md` before this code existed.

## Implemented

| Sub-exp | Module | Cost | State |
|---|---|---|---|
| A — effective integration time | `integration_time.py` | [R] | implemented, validated |
| B — the $\gamma_\beta$ null model | `gamma_null.py` | [R+W] | implemented, validated |
| — the closed-form trajectory | `gamma_ode.py` | — | implemented, validated |
| C — cumulant ladder | `moments.py` | [R] | implemented, validated |
| D — frame comparison | `frame_table.py` | [W] | implemented, validated |
| — driver | `run_1c.py` | — | implemented, end-to-end tested |
| — artifact IO | `p1c_io.py` | — | implemented |
| E — hemisphere feasibility | `hemisphere_feasibility.py` | [R] | implemented, validated |
| F — spherical designs | `design_test.py` | [R] | implemented, validated, **wired** |
| — centroids & P-S1 protocol | `centroids.py` | [R] | implemented, validated |
| — $\beta$ envelope | `beta_reduction.py` | [R] | implemented, validated |

## Validation performed

**The ODE reproduces the paper's numbers.** All 28 entries of `MATH.md` §3.2's collapse-time
table — (SA) and (USA), $n \in \{20, 467\}$, $\beta \in \{0.1, 1, 2, 5\}$, thresholds 0.5 and
0.9 — to max absolute deviation **0.005**.

**The step-size estimator recovers a known step.** Driving the true (SA) field forward at an
injected $h = 0.0200$ ($n{=}40$, $d{=}512$, $\beta{=}1$, orthogonal init): `h_calibrated`
recovers 0.0200, and the resulting $T_{\rm eff}$ at $\gamma = 0.9$ reads 3.040 against the
ODE's $t^\ast = 3.015$ (0.8%, consistent with Euler discretization).

**The null is correctly calibrated.** A trajectory that *is* the identity-weight dynamics gives
final residual $+0.0001$. A trajectory perturbed orthogonally to the field gives $-0.0113$
(vertical) / $-0.669$ (time-domain).

**Wendel reproduces the textbook values.** $n{=}3, d{=}2 \to 0.75$; $n{=}4, d{=}2 \to 0.5$;
1.0 whenever $d > n$. Computed in log space — the naive binomial form overflows float64 at
$n = 512$, which is inside our prompt range.

**The cone margin is exact against known geometry.** A $30°$ cone gives margin $\cos 30° =
0.8660$; an antipodal pair and a regular tetrahedron (origin in the hull) both give $0$ with
`feasible=False`, the tetrahedron reporting support size 4.

**The Gegenbauer recurrence matches scipy and detects exact designs.** Agrees with
`scipy.special.gegenbauer` to 1e-10 for $d \in \{3,5,8\}$, $k \le 5$, and gives $P_k(1) = 1$
exactly at $d = 1024$ where scipy's coefficient form overflows. On known designs it recovers
the exact order: the octahedron reads $t = 3$ ($Q_1..Q_3 = 0$, $Q_4 = 0.583$), the icosahedron
$t = 5$ ($Q_6 = 0.44$), and 12 random points $t = 0$.

**The $1/n$ sampling floor is confirmed.** $Q_k$ for i.i.d. uniform points matches $1/n$ to
four decimals at $n \in \{12, 50, 512\}$, $d = 1024$.

**The sink adjudicator separates the two cases.** On a synthetic stack where only three token
norms grow, `corr(raw, norm_pr) = 0.931` against `corr(raw, normed) = 0.091` → SINKS. On a
stack that genuinely loses directional rank at uniform norms, $1.000$ against $0.743$ →
DIRECTIONAL.

## Findings from implementation, before any data

1. **`MATH.md` §8's step-size definition understates $T_{\rm eff}$ by ~5.7×** on the validation
   trajectory, because it omits the $\|\mathcal{X}\|$ denominator and the field runs at ~0.18,
   not its bound of 1. The bias points toward "the network never integrates far enough," which
   is the direction that would make Blog 1's headline an artifact of depth. **P-γ2 is close to
   confirmed-by-construction under that definition.** Three definitions are computed and
   `verdict()` refuses a verdict when they straddle $t^\ast$.

2. **The calibrated step makes the residual rate-invariant.** Damping the field 0.3× gives
   residual $-0.0009$ — correctly, since damping is slower integration, not resistance. The
   residual therefore measures whether the network moves in a *different direction* from the
   identity-weight field, not how much of it it applies. This is a stronger and better-posed
   notion than the update plan specified, and P-γ1's reading should be restated in those terms.

3. **The vertical residual has no dynamic range once the null saturates** ($\gamma > 0.95$).
   The time-domain residual `time_residual_curve` was added for this; on the synthetic pair it
   separates the two cases by 2.5 orders of magnitude more than the vertical one.

4. **The FFN is not in the paper's model at all**, so `h_attn_only` — exact under Pythia's
   parallel residual — is the frame-correct variant and the other two are upper bounds.

5. **The cone condition is nearly vacuous as a boolean; the margin is not.** Wendel gives
   probability 1 for $d > n$, which every prompt satisfies, so P-H1 is close to guaranteed as
   stated. But the margin *shrinks* as $n \to d$: measured on i.i.d. uniform clouds at
   $d = 1024$, the margin is $0.221$ at $n{=}20$ and $0.030$ at $n{=}512$. The reportable
   quantity is the margin and the layer at which it first crosses zero, not the boolean.

6. **$Q_k$ cannot be compared against a fixed tolerance.** For i.i.d. points $E[Q_k] = 1/n$
   exactly, so every large-$n$ configuration looks like a design under an absolute threshold.
   The reported quantity is the ratio $Q_k / Q_k^{\rm random}$ at matched $(n, d)$, and P-S1 is
   adjudicated on the ratio — a raw comparison between checkpoints with different centroid
   counts would be reading the cluster count, not the geometry.

7. **Sharp configurations put their mass at the histogram boundaries.** An interior-only
   local-max scan scored the octahedron — two distinct inner products, the sharpest
   configuration in $\mathbb{R}^3$ — as *unimodal*, because its $-1$ peak sits in bin 0. The
   same scan counted five modes in 200 i.i.d. uniform points. `inner_product_modes` now
   includes boundary bins and requires a strict maximum over a window; it reads 2 modes for the
   octahedron, 3 for the icosahedron (mass 1.00 in both), and 1 for random clouds.

8. **The $\beta = 5$ energy column cannot be dropped.** The cumulant reconstruction is accurate
   to 0.00% / 0.07% / 0.80% at $\beta = 0.1 / 1 / 2$ and **26.6% at $\beta = 5$**; twelve
   moments are needed there, not three. So sub-experiment C settles that three of the four
   energy columns are redundant and the fourth is not.

## The $\beta$ reduction, de-blocked

Open item 1 required deciding the head-to-layer reduction before any residual could be read.
The choice does matter — spread in $\gamma_\beta(T_{\rm eff})$ across $\beta\in[0.5,5]$ at
$T_{\rm eff}=3$ is 0.89 at $n{=}20$, 0.62 at $n{=}128$, 0.26 at $n{=}467$ — so unlike the
clusterer question it cannot be dissolved.

It can be **bracketed**. $\gamma_\beta(t)$ is monotone in $\beta$: verified over 984,246 grid
points per model, (SA) decreasing with **zero** violations and (USA) increasing. The per-head
range therefore brackets the null, and `residual_bracket` reports
$[\text{residual}_{\min}, \text{residual}_{\max}]$ with `sign_unambiguous`. Where the observed
curve is outside the envelope the conclusion holds for every reduction; where it is inside, the
decision matters and the band is the uncertainty — which is the case `run_1c` refuses to paper
over with a default.

Note the two models' envelopes have **swapped endpoints**, since they are monotone in opposite
directions. Using (USA) as a stand-in for (SA) inverts the $\beta$-dependence.

## Open before running

1. **`geometry.json` must carry `beta_eff_per_head`, not just a scalar.** With only a scalar the
   residual is a point estimate whose error bar — the envelope — is unreported. `run_1c` records
   `envelope_note` when this happens rather than leaving it invisible.
2. **Causal vs non-causal field.** Default is causal, which is honest for Pythia and a
   departure from the theory. The non-causal comparison should be run once as a sensitivity
   check, since the masked field is systematically weaker and that inflates `h_calibrated`.
3. **Sublayer streams are not captured on every run.** `h_attn_only` is `nan` without them, and
   it is the frame-correct variant. Check coverage across the 27 checkpoints before treating
   the attention-only column as the primary one.
4. **$t^\ast$ is $n$-dependent and prompts span 20–512 tokens.** At $n{=}20$, $\beta{=}5$, (SA)
   and (USA) differ by a factor of ten (8.30 vs 0.79). Per-prompt $t^\ast$, never a pooled one.

**Plain LayerNorm is exactly sphere projection.** With $\gamma{=}1, \beta_{\rm LN}{=}0$, token
norms come out at $\sqrt d$ with coefficient of variation $3.5\times10^{-8}$. The frame
structurally restores uniform token weights, which is what removes the sink domination D10
identifies.

**The sphere-license adjudicator calibrates correctly.** A constant $\gamma$ and an
ALBERT-like one (sd 0.008 on mean 0.44, cv 0.018) both read LICENSED; cv 0.30 reads 16.6×
ALBERT and NOT LICENSED.

**The LN bias energy floor is real and large.** At $\|\beta_{\rm LN}\| / \|{\rm signal}\| =
0.5$, $\kappa_1$ moves from $+0.0044$ to $+0.2031$ — a 46× inflation of the common mode from a
term that does not depend on the tokens at all — and the floor is **17.9% of $E_{\beta=1}$**.
At ratio 1.86 it is 53.6%.

**Frame choice moves reported quantities materially.** On a cloud with three sink tokens,
`pr_rank` reads 144.7 in the l2 frame and 70.7 in the learned-LN frame; raw effective rank on
the same cloud is 4.99.

## Additional findings from D

9. **The dispersion statistic matters more than the mean.** The paper quotes ALBERT's
   $\gamma$ as mean 0.44, sd 0.008, but a $\gamma$ of all 0.44 and one of all 4.4 both leave
   the manifold a sphere — a uniform rescaling changes nothing. The licensing quantity is the
   coefficient of variation, and `condition_number` ($\max/\min$) is reported alongside
   because that is what bounds the metric distortion.

10. **"Constant across layers" is a second, separate condition.** A model whose $\gamma$ is
    uniform *within* each layer but different *between* layers is on a sphere at every depth
    and on a **different** sphere at each, so cross-layer trajectory metrics — which is what
    all of Phase 1 is — still inherit a rescaling. `sphere_license` reports
    `cross_layer_mean_cv` for this.

11. **Symmetric KL is not a metric**, so the Torgerson Gram is not guaranteed PSD.
    `frame_moments` reports `neg_eigen_mass` rather than clipping: a frame whose Gram carries
    substantial negative mass is not one in which "effective rank" means what it means
    elsewhere. On synthetic Dirichlet distributions it measured 1.7%.

## The clusterer question, settled

Open item 5 said the clusterer choice moves F's random baseline through the centroid count $m$,
and F was left unwired for that reason. **Measured, and it does not hold.** $Q_k/Q_k^{\rm random}$
for i.i.d. uniform configurations at $d = 256$:

| $m$ | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| ratio $Q_1$ | 1.104 | 1.048 | 0.967 | 0.922 | 0.951 | 0.901 |
| ratio $Q_2$ | 1.003 | 1.001 | 1.021 | 0.994 | 0.993 | 0.986 |

Flat at 1 across a 32× range in $m$, and a genuinely sharp configuration stays low at every $m$
— the regular simplex (a spherical 1-design) gives ratio $Q_1 = 0.000$ at $m = 5, 10, 20, 40$.
So **P-S1 can be adjudicated between checkpoints whose clusterings disagree on $m$.** The
clusterer is still fixed per sweep, but it no longer has to be fixed by matching $m$.

Default is **kmeans**, not because it is the best clusterer but because it is the only one whose
*centroids* Phase 1 persists; agglomerative and HDBSCAN persist labels only and are recomputed
from activations. `--f-method` selects the arm and `load_centroids` refuses to fall back — a
sensitivity arm that silently returns the primary arm is not one. If the three arms disagree
about P-S1, that is the result: the design signal would be a property of the clustering rather
than of the geometry.

## The effect-size floor P-S1 was missing

The registered falsifier is "no difference", which carries no threshold. The random ratio has
2σ bands of:

| degree $k$ | 1 | 2 | 3 |
|---|---|---|---|
| band ($m = 8$) | 0.164 | 0.015 | 0.002 |
| band ($m = 32$) | 0.173 | 0.013 | 0.002 |
| band ($m = 128$) | 0.189 | 0.015 | 0.002 |

`adjudicate_p_s1_banded` requires a trained-minus-step-0 improvement larger than the band at the
trained configuration's own $(m,d)$. Without it, three degrees of pure sampling noise yield a
coin-flip's worth of "improvements" and a PARTIAL verdict on nothing — verified: random-vs-random
gives PARTIAL unbanded and "NO DEGREE improves" banded.

**A claim I made and the measurement contradicted.** I wrote that discriminating power is
concentrated at low $k$, on the grounds that the simplex gives $Q_2$ ratio $\approx 0.98$ —
apparently no signal. That reading was wrong. The band at $k=2$ is 0.014, so a deviation of
0.023 is *outside* it, and the simplex registers as improved at both $k=1$ and $k=2$. Higher
degrees are **more** sensitive in relative terms, not less; both the deviation and the noise
shrink with $k$, at different rates, which is exactly why a fixed absolute tolerance would be
wrong in a different direction at every degree. `--f-tmax` defaults to 3 on cost grounds — each
degree needs its own baseline simulation — not because the power is at low $k$.

## Additional open items from C, E

5. **F needs a step-0 comparison run**, which the pilot has, and ideally a norm-matched random
   one, which it does not (claim (c) is still unadjudicated). P-S1 can be adjudicated against
   step 0 alone; the random arm strengthens it and is gated behind the same hard stop as
   everything else.
6. **E should run per layer, not per run.** The reportable object is the depth at which the
   margin first crosses zero. `hemisphere_profile` returns it; nothing yet calls it across the
   27 checkpoints.

7. **`--beta-fallback` has no safe default and the driver refuses to invent one.** $\beta$ is
   a measured property of a trained head (paper footnote 2), not a convention. Runs whose
   `geometry.json` carries no `beta_eff` are skipped with a message rather than defaulted,
   which means open item 1 above is a hard blocker for A and B rather than a refinement.
8. **Old artifacts cannot answer A or C.** `activations.npz` stores unit-norm activations plus
   the `norms` key that reconstructs the raw stream, but `norms` was added later. `raw_states`
   raises on a run without it rather than substituting the unit-norm array, which would
   produce plausible numbers meaning something else. Check coverage before scheduling.

## Not doing here

Phase 2d waits on B's result, since $T_{\rm eff}$ determines whether the energy-monotonicity
break is the right thing to attribute.

F is wired. `--f-method` fixes the clusterer per sweep; run the other two arms once as a
sensitivity check.
