# Phase 1c — STATUS

<!-- phase-card -->
## Card

- **Question:** Does a trained network integrate the identity-weight particle dynamics far enough to reach the collapse time t*, in which direction does it depart from the paper's field (the γ_β residual), and do the geometric preconditions (the cone, the sphere frame, spherical designs) hold on real activations?
- **Inputs:** All six sub-experiments on synthetic and known-answer configurations. Real data for E only: `pythia-410m-step143000`, 8 of the v1 battery's 9 prompts (battery `1e47918ef77a`, without `short_heterogeneous`), 2026-09-19, `claims/audits/p1c_e_hemisphere.json`. Input audit (keys only) over the 19-checkpoint sweep, 152 dirs, `claims/audits/p1c_inputs.json`. A 2026-08-17 pilot attempt skipped every run for want of β
- **Results:**
  - The γ_β ODE reproduces the paper's collapse-time table and the step estimator recovers an injected step — `status-1c.md` "Validation performed"
  - Phase 1's step-size definition understates the effective integration time about 5.7×, which makes `P-gamma2` close to confirmed by construction under it — `status-1c.md` "Findings from implementation, before any data"
  - γ_β is monotone in β, so the head-to-layer reduction becomes a bracket rather than a choice; it survives on the measured β range — `status-1c.md` "The $\beta$ reduction, de-blocked", `status-1c.md` "E-value audit, Phase 1c (2026-09-19)"
  - `P-S1`'s ratio is flat across centroid count m, so clusterings need not match on m; the effect-size floor it lacked is added — `status-1c.md` "The clusterer question, settled", `status-1c.md` "The effect-size floor P-S1 was missing"
  - `P-H1` measured: the cone is feasible at every layer on step 143000, the margin is smallest at layer 0 and grows with depth, well above the i.i.d. reference — `status-1c.md` "E-value audit, Phase 1c (2026-09-19)"
  - A, B and F are blocked by missing artifact keys, not physics; β is derivable with no forward pass, and its scale convention is worth a factor of 8 and undecided — `status-1c.md` "E-value audit, Phase 1c (2026-09-19)"
- **Superseded / wrong:**
  - "kmeans is the only clusterer whose centroids Phase 1 persists" is false on disk: 0/152 directories carry centroids — `status-1c.md` "E-value audit, Phase 1c (2026-09-19)"
  - "Open item 1 is a hard blocker" overstates it: β is derivable from existing artifacts — `status-1c.md` "E-value audit, Phase 1c (2026-09-19)"
  - `lit-1c.md`'s "scholarly hosts are blocked" and its α·Δτ ≈ 0.025 preemption are both false — `status-1c.md` "E-value audit, Phase 1c (2026-09-19)"
  - The Euler-discretisation reading is published, not new — §3.37
  - Validation cites section 3.2 of MATH.md, a file that does not exist — `INDEX.md` "Referenced, not present"
  - The energy-trajectory PNGs bake a wrong citation into three suptitles — `math-1c.md` §11
  - `P-S1`'s "floor is attainable" is a p ≤ α test; at the gate's default 500 draws E ≤ 11.2, so it cannot decide alone. And finding 2's "raw-slope β" is the number `core/beta_eff.py` calls comparable — `docs/PHASE_SYNTHESIS.md` "3.1 Attainable E per adjudicable row", `docs/PHASE_SYNTHESIS.md` "3.2 The remaining decisions, checked against the code first (Parked 7)"
- **Registry:** `P-gamma1`, `P-gamma2` needs-null, blocked (no `beta_eff` in any run dir); `P-H1` measurement, measured on one checkpoint, not adjudicated; `P-S1` e-value, active, dry run only, blocked on inputs (`claims/EXPERIMENTS.md`)
- **Depends on:** 1@d7c7e735e4
- **Feeds:** 2d, 9
- **Open threads:**
  - β's scale convention, to decide before any producer freezes it in an artifact
  - `P-S1`: re-cluster both arms offline at a matched k, or record the gate as unfeedable
  - `h_attn_only`, the frame-correct step, needs sublayer streams no run directory has
  - Causal vs non-causal field, never compared
  - The cone margin's response to a γ patch is closed-form, but only where the minimiser is unique; a runner must report that — §3.50
- **After Phase 10:**
  - Write the β producer (`beta_eff_per_head` into geometry.json) once the convention is fixed; unblocks A and B (free, CPU, one model load per checkpoint)
  - Run E over all 19 checkpoints × the 8 v1 prompts (free)
  - Re-cluster at matched k from `activations.npz` and run `P-S1` (free)
  - `run_1.py --sublayer` reruns for `h_attn_only` (forward pass, 410m)
  - Regenerate the energy-trajectory PNGs (free)
- **Reviewed:** 2026-09-24 · body `798d9ab647`
<!-- /phase-card -->

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-23.

- 2026-09-16 · the Euler-discretisation reading is published · §3.37
- 2026-09-19 · A/B/F blocked by artifact keys; kmeans centroids not persisted; `lit-1c.md`'s preemption lead is false · §3.40
- 2026-09-20 · closed-form cone-margin response to a γ patch; needs a unique minimiser · §3.50
- 2026-09-20 · the cone condition cannot fail for n ≤ d in general position (the same fact as finding 5's Wendel note) · §3.39
- 2026-08-22 · validation cites MATH.md, a file that does not exist · `INDEX.md` "Referenced, not present"
- 2026-08-23 · the energy-trajectory PNGs bake a wrong citation into three suptitles · `math-1c.md` §11
- 2026-09-24 · `P-S1`'s gate defaults to 500 null draws, so E ≤ 11.2 and it cannot reach E ≥ 20 alone (the dry run's "floor is attainable" tests p ≤ α); and finding 2 below calls the ×8 β the non-comparable "raw slope" while `core/beta_eff.py` calls it the comparable one · `docs/PHASE_SYNTHESIS.md` "3.1 Attainable E per adjudicable row", `docs/PHASE_SYNTHESIS.md` "3.2 The remaining decisions, checked against the code first (Parked 7)"

**Registered predictions:** `P-gamma1`, `P-gamma2` (needs-null — bracket and
point-estimate readings with no null built), `P-H1` (measurement — no valid
null exists, by Wendel's theorem; report the margin, never a p) and `P-S1`
(e-value — null in `design_test.py`, calibrated on a known-answer dry run,
**not run against real artifacts**). Nulls and evidence paths are
`claims/registry.json`; the per-phase view is `claims/EXPERIMENTS.md`.

**State:** all six sub-experiments implemented and validated on synthetic data and on
configurations with known exact answers, with a driver (`run_1c.py`) and artifact IO
(`p1c_io.py`) that have been run end to end against a synthetic Phase-1 run directory.
Sub-experiment **E has now been run against Pythia artifacts** (2026-09-19, the audit
below); A, B, C, D and F have not, and on today's artifacts cannot be — see the audit
for what each is missing. Predictions P-γ1, P-γ2, P-H1 and P-S1 were registered in
`PREDICTIONS.md` before this code existed.

## E-value audit, Phase 1c (2026-09-19)

The pass over this phase's four registered predictions (`PROJECT.md` §3.40),
following Phase 1's (`p1_mstate_tracking/status-1.md`). **All four
classifications are correct as registered. Three of the four gates cannot be
fed by any artifact in the tree, and the reasons are artifact keys rather than
physics.** The machine-readable record is `claims/audits/p1c_inputs.json`,
written by `tools/audit_p1c_inputs.py` (+ 7 pure tests); it reads keys and
chooses no statistic, null or clusterer. Run over all **19 pythia-410m
checkpoint directories, 152 model-prompt directories**:

| sub-exp | prediction | registered class | verdict |
|---|---|---|---|
| 1c-A | `P-gamma2` | needs-null | **BLOCKED** — no `beta_eff` anywhere; derivable without a forward pass |
| 1c-B | `P-gamma1` | needs-null | **BLOCKED** — same, and it needs it per head |
| 1c-E | `P-H1` | measurement | **RUNNABLE, and run** — the audit's one number |
| 1c-F | `P-S1` | e-value | **BLOCKED** — three independent artifact reasons, plus the (m, d) refusal |

**1. A and B are blocked on a producer, not on a re-run.** 0/152 `geometry.json`
carry `beta_eff` — open item 1 below, confirmed on disk rather than assumed —
while 152/152 carry `activations` + `norms` (so `raw_states` is satisfied) and
152/152 carry `attentions.npz`. β is therefore **derivable from artifacts
already written plus the cached checkpoint's LN parameters, with no forward
pass**: `ln_frame.frame_for_hidden_state` → `ln_frame.ln_frame_gram` →
`beta_eff.estimate_beta_all_heads`. Demonstrated on `pythia-410m-step143000` /
`wiki_paragraph`: **16/16 heads valid in every one of the 24 blocks**, median
R² 0.18, one model load. So "open item 1 is a hard blocker" overstates the
cost: the blocker is a producer nobody has written.
**The exception is `h_attn_only`**, the frame-correct step-size variant
(finding 4 below): it needs the post-sublayer streams, **0/152 directories have
them**, and they come only from `run_1.py --sublayer` — new forward passes.

**2. β's unit convention is undecided, and it is worth a factor of 8.** With
the model's own `1/sqrt(head_size)` logit scale applied, measured β on that
checkpoint over 384 head-rows is median **0.50**, IQR [0.26, 0.75], range
**[−0.84, 2.19]**, with **0% above 5** and 50% below 0.5. Without it every
number is 8× larger on this model — and `head_size` is 64 on `gpt2-large` and
**128 on `pythia-1.4b`**, so a raw-slope β is not comparable across `CLAIM-C`'s
own arms (`beta_eff.py`'s docstring, problem 3, says exactly this). The
illustrative [0.5, 5] in `beta_reduction.py` is consistent with the *unscaled*
convention. **This is a decision to take before the producer writes anything**,
because γ_β's spread over β is the whole reason the reduction question exists.

**3. The measured β range lies outside the range the monotonicity claim was
verified on — and the claim survives.** "Monotone in β, 984,246 grid points,
zero violations" was measured over β ∈ [0.5, 5]; 50% of measured heads are
below 0.5 and 8% are negative. Re-checked here at t = 3.0 over
β ∈ [−1, 2.2]: (SA) decreasing and (USA) increasing at both n = 20 and n = 467,
zero violations. So `residual_bracket`'s bracketing property holds on the range
the data actually occupies. What changes is the *width*: over the measured
range the (SA) envelope at n = 467, t = 3 is **0.022** against **0.262** over
[0.5, 5], while (USA) is **0.634** against **0.445**. Under the scaled
convention the reduction decision nearly dissolves for (SA) and does not for
(USA) — which is a reason to fix the convention first, not to re-open
"The β reduction, de-blocked" below. *(One checkpoint, one prompt,
whole-sequence index set; a second prompt could move these.)*

**4. `run_1c.py` refuses β-free sub-experiments for want of β — FIXED
2026-09-19.** Run for real with `--subexp E` against the sweep's trained
directory: **8 runs, 8 SKIPs, 0 written**, exit 1, every line "no `beta_eff`
and no `--beta-fallback` given". E uses no β at all, and neither does F; the
`beta_used` finiteness test was applied to every run regardless of `--subexp`.
`requires_beta()` now names the two sub-experiments that integrate
`gamma_beta` (`BETA_SUBEXPERIMENTS = {"A", "B"}`), the driver skips only when
one of those was asked for, and each record carries `beta_required` with
`beta_source` reading `unavailable` rather than a fallback that was never
supplied. Re-run against the same directory with **no** `--beta-fallback`:
**8/8 written, 0 skipped**, and the margins are identical to the fallback run
— which is the check that the fallback never entered E's numbers in finding 6
below. `tests/test_run_1c_beta_gate.py` (5, pure).

**5. `P-S1` cannot be fed by any run directory, for three independent
reasons.**
- **The primary arm's input does not exist.** `centroids.py::load_centroids`
  reads `clusters.npz: kmeans_centroids_L{i}` for kmeans; **0/152 directories
  carry that key** — including the runs made this week. Every directory carries
  `kmeans_labels_L*` and `agglom_mid_labels_L*` and nothing else. So "kmeans is
  the only one whose centroids Phase 1 already persists" ("The clusterer
  decision", `centroids.py`) **is false on disk**; the writer has never
  persisted centroids, and the primary arm's stated advantage over the
  secondary ones is not real.
- **The HDBSCAN arm looks in the wrong file.** `load_centroids` expects
  `hdbscan_labels_L{i}` inside `clusters.npz`; the runner writes
  `hdbscan_labels.json`, and on these runs that file is `{}` because `hdbscan`
  is not installed in the environment the sweep ran in.
- **The agglomerative arm runs, at a cluster count that measures the token
  cloud.** On `step143000` / `wiki_paragraph` it returns m = 209, 207, 187, 83,
  149 at layers 0, 6, 12, 18, 24 — of 467 tokens — with 30–66% of those
  clusters singletons under the default `min_size=1`, which the module's own
  docstring says is "measuring the token cloud, not the cluster structure".
  Kmeans labels at the same layers give m = 9, 5, 6, 2, 2.

And then the gate's **(m, d) refusal** bites: between `step143000` and
`step0`, cluster counts agree on **25 of 175 kmeans layer-rows** and **2 of
175 agglomerative rows** (excluding `repeated_tokens`; 45/200 and 9/200
including it). That is the registry's own "sixth pre-computed requirement" —
both arms clustered to the same count — measured for the first time. **The
cheap route is offline:** centroids can be recomputed from `activations.npz`
at a matched k with no forward passes, which answers the missing-key problem
and the refusal at once.

**6. `P-H1` measured — the phase's first real number, and no p-value.** E run
over all eight prompts of `pythia-410m-step143000` (`hemisphere_profile`,
causal field; β is irrelevant to E, so a `--beta-fallback` was passed purely to
get past the skip in finding 4):

| prompt | min margin | at layer | layer-0 margin | final-layer margin |
|---|---|---|---|---|
| wiki_paragraph | 0.1316 | 0 | 0.1316 | 0.3083 |
| homer_iliad | 0.1334 | 0 | 0.1334 | 0.4122 |
| latex_monograph | 0.1396 | 0 | 0.1396 | 0.3259 |
| sullivan_ballou | 0.1485 | 0 | 0.1485 | 0.3663 |
| camus_letranger | 0.1539 | 0 | 0.1539 | 0.3484 |
| paper_excerpt | 0.1708 | 0 | 0.1708 | 0.3979 |
| hdbscan_code | 0.1727 | 0 | 0.1727 | 0.4010 |
| *repeated_tokens (control)* | *0.5003* | *23* | *0.6536* | *0.6951* |

The cone condition is **feasible at every layer of every prompt** — zero
infeasible layers, so the "layer at which the margin first crosses zero"
(finding 5 below) does not exist on this checkpoint. Two things the table says
that the boolean does not: the **minimum is at layer 0 on all seven**
metastability prompts and the margin *grows* with depth, and the margins sit
**well above** the i.i.d.-uniform reference for these lengths (0.030 at
n = 512, finding 5 below) rather than near it. Nothing is adjudicated and
nothing was written to `claims/` — `P-H1` is a `measurement` row and stays one.
One caveat, stated rather than buried: these artifacts are the checkpoint sweep
that ran while `CLAIM-C`'s hard stop was unrun (`PROJECT.md` §3.36), so the
reading inherits whatever that gate eventually says.

**7. `lit-1c.md`'s one blocking question, answered.** That file is leads-only
because "arxiv.org and every other scholarly host are blocked by this session's
egress proxy". **They are not blocked from this machine**, and its flagged
"first thing to check" — 2604.23740's reported effective step size
`α · Δτ ≈ 0.025`, which would preempt 1c-A's central number — **is not in the
paper.** Its full text (ar5iv) contains no "effective step size" and exactly
one "step size": 0.01, the forward-Euler setting of *their own* synthetic ODE
experiment. The α ≈ 0.025 line came from a search-engine summary, not from
this paper. So 1c-A's calibrated step is not preempted by the nearest
neighbour, and `lit-1c.md` can be upgraded from leads to readings whenever a
session chooses to spend the time.

**Decisions this leaves to the author.** (a) Write the β producer —
`beta_eff_per_head` into `geometry.json` or a side artifact — which unblocks A
and B except for `h_attn_only`; (b) fix β's scale convention first, since (a)
freezes it in an artifact; (c) for `P-S1`, re-cluster both arms offline at a
matched k, or record that the gate stays unfeedable; (d) ~~the one-line
`run_1c.py` β-gate fix~~ **done 2026-09-19, finding 4**; (e) `run_1.py
--sublayer` re-runs, the only item here that costs forward passes, and the only
route to the frame-correct `h_attn_only`.

---

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
   27 checkpoints. *(2026-09-19: `run_1c.py:159` does call it per run, so what was missing was
   the running, not the wiring; done for `step143000`'s eight prompts in the audit above, where
   the crossing depth does not exist because no layer is infeasible. The other 18 checkpoints
   are unrun.)*

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
