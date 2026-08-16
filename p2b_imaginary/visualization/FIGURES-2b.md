# Phase 2b — FIGURES

The figure catalogue for `p2b_imaginary/visualization/`, and the tracker for
building it. Companion to `status-2b.md` (what the phase found, and what it
withdrew), `design-2b.md` (why it was built the way it was) and `PLAN_2b.md`
(what is being rebuilt). This file answers a fourth question: **what does
Phase 2b look like**, and which parts of it can be drawn from the artifacts
the phase actually writes.

Read the status column before citing any figure. Seven quantities this phase
computed and then dropped at serialization time were listed here as data
gaps; **all seven emissions have now landed** in `p2b_imaginary/` (see
[Data gaps](#data-gaps)), so a run made after them draws the complete set.
A run directory made BEFORE them still does not, and those figures skip with
a printed reason rather than failing — the loaders detect each gap from the
artifact rather than assuming it, so both kinds of directory stay readable.
`--list_runs` says which gap is open in a particular directory.

---

## Ground rules

**Artifacts only, never a model, never a weight matrix.** Every figure loads
from a Phase 2b output directory — `phase2b_results.json`, the per-checkpoint
`block1a_rotational_spectrum.json`, and the per-prompt
`block1b_rescaled_comparison.json`. Nothing here imports torch, loads a
checkpoint, opens `ov_weights_*.npz`, or runs a Schur decomposition. That last
one is worth naming explicitly because it is *cheap enough to be tempting*:
Block 1a is weights-only, so a figure module could plausibly re-derive its own
spectrum from the OV npz and would then be a second implementation of the
phase's central quantity, with its own energy convention. This is the same
contract `p1b_hemisphere/visualization` and `p2_eigenspectra/visualization`
work under.

**Analysis logic is imported, never restated.** The verdict vocabulary comes
from `rotational_rescaled.VERDICTS`, the equivalence band from
`EQUIVALENCE_BAND`, the frame keys from `FRAME_KEYS`, the orthogonality
tolerance from `ORTHOGONALITY_TOL`, the tracked statistics and the dated
events from `p2b_report.TRACKED_STATISTICS` / `KNOWN_TRANSITIONS`, and every
trajectory statistic from `p2b_report` itself — `flatness`, `interval_deltas`,
`align_to_transitions`, `co_movement` and `block1b_trajectory` are called, not
reimplemented. A figure that disagrees with `phase2b_summary.txt` is a bug in
this package by construction.

**Draw the refusal, never a zero.** This is Phase 2b's version of Phase 1b's
"draw the continuous quantity, not the label," and it is the rule the phase's
whole rewrite turns on. `elimination_rate` returns `None` with a status for
four distinct refusals, precisely because the pre-rewrite code returned the
float `0.0` for all of them and that value then entered a majority vote. No
figure in this package may plot a refusal at zero. Refusals get their own
marker, their own gray, and their own count beside the panel — `no_violations`
at steps 8–64 is the expected shape of a clean early checkpoint, and it must
not read as "rescaling did nothing."

**The invariance control is drawn as a control.** `remove_rotation` is an
algebraic identity — `e^{−A}` is orthogonal, so the frame reproduces the
original exactly — and reading it as a measurement is how `rotation_neutral`
became a headline. It is drawn in every frame figure, always in the reserved
control style (hatched, gray, labeled "invariance control"), always excluded
from the elimination-rate comparison, and F5 exists only to show the residual
it holds to. Nothing about rotation's dynamical role can be read off it, and
the figures are built so that trying is hard.

**Denominators next to numerators.** `n_transitions_scored` is a first-class
output because it is the denominator; two frames that scored different numbers
of transitions are not comparable at all. Every violation count in this
package is drawn with its scored / gated / NaN companions in the same panel.

**Both energy conventions, always named.** The phase contains three
definitions of "rotational fraction" (per-eigenvalue, legacy per-block,
relative-eigenvalue). Any figure showing one shows which, and S1 and X5 exist
to show the divergence between them rather than let a reader assume the
84–97.5% figure and the corrected one are the same number.

**Log-spaced training axis, step 0 as its own object.** The step axis is
`log10(step+1)` with real checkpoint steps as ticks, and step 0 is drawn in
its own near-black dotted style rather than as the pale end of a colormap —
the convention `p1_mstate_tracking/visualization/checkpoints.py` settled and
that `p2b_report` already computes its interval widths in. See
[the step-axis note](#a-note-on-the-step-axis) for how this package reaches
it without inheriting Phase 1's plotting dependencies.

**A missing input is a skipped figure, not a crash.** Blocks 2, 3 and 4 are
unwired by design (`PLAN_2b.md` items 10–12), nulls and the precision surface
are opt-in (`--with-nulls`, `--with-precision`), Block 1b needs Phase 1
activations that a weights-only (`--blocks 1a`) sweep never touches, per-head
circuits need an OV npz carrying `ov_head{h}_*` arrays, and a
single-checkpoint run has no trajectory. Every figure declares what it needs
and no-ops with a printed reason when it is absent.

**Color by job.** Categorical hues in fixed order (validated for CVD
separation, see `style.py`); one hue light→dark for magnitude; a two-hue
diverging pair with a neutral midpoint for the two genuinely signed
quantities in the phase — the unclipped elimination rate and the
per-interval trajectory delta. Gray is reserved for refusals, controls, and
absent inputs, and is never a data color.

---

## Figure classes

Eight classes, each its own module and each selectable from the CLI
(`--classes spectrum trajectory …`).

| Class | Module | Scope | Figures |
|---|---|---|---|
| `spectrum` | `spectrum.py` | Block 1a — one checkpoint, depth axis | 10 |
| `heads` | `heads.py` | per-head circuits — is the headline about any head? | 4 |
| `frames` | `frames.py` | Block 1b — one (checkpoint, prompt) | 10 |
| `trajectory` | `trajectory.py` | Block 1a across checkpoints | 8 |
| `report` | `report_fig.py` | `p2b_report` — flatness, intervals, dated events | 5 |
| `verdicts` | `verdicts.py` | Block 1b across checkpoints and prompts | 5 |
| `nulls` | `nulls.py` | norm-matched Gaussian null (opt-in) | 4 |
| `curiosities` | `curiosities.py` | exploratory / speculative | 13 |

`spectrum`, `heads` and `nulls`' per-checkpoint half write into
`{out}/{stem}/`; `frames` writes into `{out}/{stem}/{prompt}/`; everything
else writes into `{out}/_cross/`.

Status values: **done** — implemented and exercised against the fixture;
**done — needs `--flag`** — implemented, and drawing it needs a sweep that
opted into an expensive block; **planned** — specified here, not yet built.
Nothing in the catalogue is now blocked on an emission: all seven gaps below
have landed, and a figure needing one skips only against a run directory
older than it.

---

### `spectrum` — Block 1a at one checkpoint, on the depth axis

Reads `block1a.per_layer`. Weights-only, so every one of these is available
from a `--blocks 1a` sweep with no activations anywhere.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| S1 | `complex_fraction_depth` | All three "rotational fraction" definitions on one depth axis: per-eigenvalue energy, the legacy per-block convention, and the dimension fraction. The 84–97.5% headline and its correction in one image, with 1.0 (a Gaussian's value) drawn as the reference the number has to beat to mean anything. | `per_layer` | done |
| S2 | `energy_budget_depth` | Complex and real eigenvalue energy per layer in absolute terms on a log axis, with `ov_frob` beneath. Separates "this layer is 97% rotational because rotation is large" from "…because the real part is tiny" — the fraction alone cannot. | `per_layer` | done |
| S3 | `theta_depth` | `theta_mean` with the min–max band and the median, against π/2. θ is on [0, π] since the rewrite; the old `abs(a)` folded repulsive rotations onto their reflections, so a mean pinned near π/2 is what the *bug* used to produce and what a Gaussian genuinely gives. | `per_layer` | done |
| S4 | `rho_depth` | `rho_mean` ± sd against ρ = 1, with `frac_rho_above_one` as a strip beneath — named so it cannot be mistaken for the dynamical quantity, which is S5's. | `per_layer` | done |
| S5 | `repulsive_depth` | `frac_repulsive_real_part` vs depth against the 0.5 line: the fraction of rotation planes with Re λ < 0, which is the directions `e^{−V}` grows in and the weights-side analogue of Phase 2's `frac_repulsive`. | `per_layer` | done |
| S6 | `henrici_depth` | `henrici_relative` vs depth with the argmax layer marked, and `henrici_absolute_unclamped` beneath with its zero line. A materially negative unclamped value is a block-parse bug, not noise, and the previous version clamped it away silently. | `per_layer` | done |
| S7 | `normality_budget` | `t_frob_sq` split into eigenvalue energy and the Henrici departure, per layer, as fractions. How much of the operator lives in the interaction between Schur blocks rather than in its eigenvalues — i.e. how informative the S/A split is at each depth. | `per_layer` | done |
| S8 | `dims_vs_energy` | `dim_complex_fraction` against `complex_energy_fraction`, one point per layer, with y = x. **`head_circuits`' correction as a figure**: rank deficiency destroys dimension fractions and leaves energy fractions intact, so these two are different questions and the gap between them is the answer. | `per_layer` | done |
| S9 | `plane_spectrum` | Every rotation plane in the checkpoint, in the complex plane, coloured by depth — plus the pooled θ histogram with the observed mean drawn on it. **The spectrum, not a summary of it.** S3's five order statistics are compatible with one tight cluster, with two clusters at either end, and with a uniform smear; if the mean line here sits in a trough, every statistic in the phase built on `theta_mean` describes no plane in the checkpoint. | `planes.npz` | done |
| S10 | `precision_surface` | Complex fraction against the relative tolerance, at float64 and after an fp16 round trip, one line pair per layer, with the shipped 0.01 marked and `precision_verdict`'s per-layer verdict as a strip beneath. **Precision-policy item P2 with a number attached.** A sloping baseline means the headline is a property of the counting rule; a gap between the curves means it is a property of fp16 storage. | `block1a.precision` | done — needs `--with-precision` |

### `frames` — Block 1b at one (checkpoint, prompt)

Reads `block1b_rescaled_comparison.json`. Absent from a `--blocks 1a` sweep,
and absent for any prompt whose Phase 1 run had no activations.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| F8 | `energy_curves` | Interaction energy vs depth, one curve per frame, with the violating transitions marked and the rank gate shaded beneath. **The central Block 1b picture**, and for a long time an undrawable one — a count says how many transitions crossed a threshold, and only the curve says whether the rescaling changed the trajectory's shape at all. | `frames[*].per_layer` | done |
| F9 | `violation_severity` | The relative energy drop at every transition, per frame, against `rel_tol` on a symlog axis. Four marginal violations and one catastrophic one give the same count and the same rate. Unscored transitions are gaps, not zeros. | `counts.rel_drops` | done |
| F10 | `rescaler_growth` | `max |R_cum|` vs depth per frame on a log axis, with the overflow limit and the truncation point marked. Separates a frame that diverged immediately from one that climbed steadily and ran out of depth. The control is flat at 1 by construction — if it is not, F5's identity check is invalid. | `frames[*].r_cum_max_abs` | done |
| F1 | `frame_counts` | Per frame: violations, and beside them the scored / gated / NaN transitions they are a fraction of. The denominator rewrite as a picture — two frames with different scored counts are not comparable, and this is where that is visible before any rate is read. | `frames[*].counts` | done |
| F2 | `elimination_rates` | `elim_full` and `elim_signed` at the reference β, **unclipped** (the negative axis is always drawn, because a negative rate is ALBERT's overcorrection and Phase 2's verification item V2), with the ±`EQUIVALENCE_BAND` band and each refusal drawn as a labeled marker at its own row rather than at zero. | `comparison` | done |
| F3 | `violation_layers_strip` | Which layers violated, one row per frame, on the depth axis, with everything past `n_valid_layers` shaded as not-scored. A frame that truncated at layer 3 and a frame that found no violations after layer 3 look identical in a count; they do not look identical here. | `frames[*].counts.violation_layers` | done |
| F4 | `truncation_ladder` | `n_valid_layers` per frame against the full depth, with `truncation_reason` labeled and `r_cum_max_abs_final` on a log axis beside the overflow limit. The three manufactured-elimination-rate mechanisms, as the one picture that separates them. | `frames[*]` | done |
| F5 | `invariance_control` | The rotation-only frame's orthogonality residual and its worst relative energy difference, against `ORTHOGONALITY_TOL` and the 1e-3 violation tolerance, on a log axis. **status-2b's withdrawal in one image**: the residual sits ~1e-15 against a 1e-3 threshold, so `elim_rotation = 0.0` was forced before any data was read. | `invariance` | done |
| F6 | `sa_decomposition_depth` | `S_frob`, `A_frob` and the Frobenius rotation ratio per layer. The structural claim ("OV is mostly rotation") in the norm that the rescaled frames actually act in, as against S1's spectral one. | `sa_decomp` | done |
| F7 | `phase1_cross_check` | Phase 2b's `original`-frame violation count against Phase 1's own count for the same run, per β. Phase 2b gates on normed effective rank and Phase 1 on raw, so these are *expected* to differ; a large divergence means the gate is doing the work the rescaling is being credited with. | `phase1_cross_check` | done |

### `heads` — is the headline about any actual head?

Reads `block1a_head_circuits.json`. Weights-only, and cheap: per-head `W_OV`
has rank `d_head`, so every spectrum here is a `d_head²` problem rather than
`d_model²` — 16 × 64³ against 1024³ per layer at 410m.

The class exists for one distinction. `ov_total = Σ_h W_OV^h` is the
effective operator only under a counterfactual the model does not satisfy —
that every head shares an attention pattern; the real update is
`Σ_h α^h X W_OV^h`. So "OV is 84–97.5% complex" is a statistic of a matrix
the model never forms, and whether it describes any HEAD is a separate
question. `summed_vs_per_head` reports both and the gap, and does not
adjudicate; neither do these.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| H1 | `summed_vs_per_head` | The summed statistic against the per-head min–max band and mean, per layer, with the gap beneath. Where the summed value falls outside the band it describes no head in that layer — a statement about the counterfactual, not about the model. | `per_layer.summed` / `.per_head` | done |
| H2 | `head_agreement_depth` | `head_agreement` — the fraction of heads within 0.05 of the summed value — vs depth, with the head-to-head sd beneath. High spread with low agreement means the heads differ and the sum describes none of them; low spread with low agreement means they agree with each other and not with the sum. | `per_layer.head_agreement` | done |
| H3 | `head_spectrum_spread` | Every head drawn individually against the one number that stands for all of them. Two clusters of heads, one outlier, and a smooth spread give the same sd and are three different pictures of a layer. | `per_layer` | done |
| H4 | `head_agreement_trajectory` | Mean and worst head agreement vs training step, with the gap and spread beneath. **The interesting version of the question**: falling agreement means the heads are differentiating and the published number describes less of the model as training proceeds. | `head_circuits.summary` | done |

### `trajectory` — Block 1a across checkpoints

The axis the rerun exists for. Needs ≥ 2 checkpoints with Block 1a; a
single-checkpoint directory skips the whole class with that reason.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| T1 | `complex_fraction_trajectory` | The complex energy fraction vs log-step with the across-layer spread as a band, and `range_excess_over_noise` printed on the panel. **`PLAN_2b.md` open question 1's figure.** A 27-point series drawn from pure noise has a range of ~4 standard errors by construction, so the figure draws that expectation rather than letting a wiggle read as a trajectory. | `p2b_report.collect_trajectory` | done |
| T2 | `tracked_statistics_panel` | Every entry in `TRACKED_STATISTICS` as small multiples on a shared step axis, each with its own spread band and flatness verdict. One image, the whole Block 1a trajectory question. | `p2b_report` | done |
| T3 | `depth_step_heatmap` | Layer × step heatmaps for the four per-layer scalars that have a trajectory question attached (complex fraction, Henrici, θ, repulsive fraction). Depth and training on the same picture, which is the only place a *localized* event can appear at all. | `block1a.per_layer` | done |
| T4 | `layer_race` | Every layer's own trajectory as a thin line coloured by depth, with the across-layer mean bold. Distinguishes "the model moved" from "three layers moved and the mean followed them". | `block1a.per_layer` | done |
| T5 | `late_layer_zoom` | Layers 21–23 against every other layer's band, on the step axis, with the 8→16 interval marked. **Phase 1 open item 4 as a figure** — the unexplained collapse was confined to those layers, and this asks whether the OV spectrum does anything there. Skips with a printed reason on a model too shallow to have them. | `block1a.per_layer` | done |
| T6 | `sweep_coverage` | Every step the sweep found, plus `missing_checkpoints` and per-checkpoint failures, as a strip on the step axis. The silent-absence failure made visible: a checkpoint Phase 2 never wrote OV weights for is a gap in the trajectory, not a smooth interpolation. | combined | done |
| T7 | `henrici_hotspot` | Which layer holds the Henrici maximum at each checkpoint, and how sharp that maximum is. If the non-normality hotspot migrates through depth during training, that is a mechanism candidate that no scalar trajectory can show. | `block1a` | done |
| T8 | `angle_modulus_trajectory` | Mean θ and mean ρ vs step, each with its across-layer spread. The two quantities a norm-matched Gaussian null has an actual opinion about — unlike the complex fraction, where the null is ~1.0 by construction. | `block1a` | done |

### `report` — the cross-checkpoint report, drawn

Every figure here calls a `p2b_report` function and draws its return value.
None of them computes a statistic.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| R1 | `flatness_bars` | `range_excess_over_noise` per tracked statistic against the 1.0 line, with `range_in_spreads` beside it as the second, substantive comparison. **The number the phase's first open question turns on**, and the one whose scale was got wrong twice while it was being written (see `PLAN_2b.md`, "A scale error found while demonstrating the report"). | `p2b_report.flatness` | done |
| R2 | `interval_ranking` | Every interval's `delta_in_spreads`, ranked, with the dated `KNOWN_TRANSITIONS` spans marked in place. A large move means little if every interval has one; this is where "2nd of 26" and "17th of 26" become visible. | `p2b_report.interval_deltas` | done |
| R3 | `event_alignment` | Statistic × dated-event heatmap of `delta_in_spreads`, each cell annotated with its `interval_rank`, and `not_bracketed` cells hatched rather than drawn as zero. The phase's rule that an unbracketed span is unanswerable, enforced in the rendering. | `p2b_report.align_to_transitions` | done |
| R4 | `co_movement` | Two trajectories on their shared step grid, plus their interval-to-interval agreement — the less trend-sensitive reading. Defaults to Henrici against the repulsive fraction; `--external` supplies a real series from another phase (Phase 2's `frac_repulsive`) through `p2b_report.external_trajectory`. The caveat is printed on the figure, not just in the docstring. | `p2b_report.co_movement` | done |
| R5 | `known_transitions_map` | The dated events as spans on the step axis with this sweep's checkpoints overlaid, coloured by how many checkpoints fall inside each. Which of Phase 1's and Phase 2's findings this sweep can say anything about at all, before any statistic is read. | `KNOWN_TRANSITIONS` + combined | done |

### `verdicts` — Block 1b across checkpoints and prompts

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| V1 | `verdict_matrix` | Step × prompt heatmap of the overall verdict. The refusal verdicts (`no_violations`, `not_comparable`) take gray slots and the three measurement verdicts take real hues, so a table that is mostly refusals cannot read as a table that is mostly findings. | `interpretation.overall` | done |
| V2 | `verdict_composition` | Verdict composition vs step, stacked, with `n_refused` drawn as its own band. `block1b_trajectory`'s point exactly: a checkpoint where every run refused looks identical to one where every run said "inert" if only the tally is read. | `p2b_report.block1b_trajectory` | done |
| V3 | `elim_trajectory` | `elim_full` and `elim_signed` means vs step with min–max bands and the contributing n annotated at each point. **The phase's remaining question** — `e^{−(S+A)} ≠ e^{−S}e^{−A}` unless S and A commute, so this contrast is the one thing Block 1b measures. | `p2b_report.block1b_trajectory` | done |
| V4 | `refusal_reasons` | Every `elimination_rate` status across the sweep, as counts. How much of the elimination table is a refusal, by kind — and `different_transitions_scored` specifically, which is the gate-divergence mechanism that scales with ‖V‖. | `comparison[*].status` | done |
| V5 | `truncation_map` | Step × prompt count of truncated frames, with invariance-broken runs marked. Truncation is not uniform across prompts (it depends on the trajectory, not just the weights), and a rate computed where the signed frame truncated is the one that comes out at exactly 1.0 for free. | `frames[*].truncated` | done |

### `nulls` — the norm-matched Gaussian control

Only present when the sweep ran `--with-nulls`, which is off by default
because it costs `n_null_draws` extra Schur decompositions per layer. Every
figure here skips with that reason when absent.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| N1 | `null_z_depth` | Observed against the null mean ± sd per layer, one panel per null statistic. The control status-2b caveat (i) says the headline never had. | `per_layer.nulls` | done |
| N2 | `null_percentile_depth` | Percentile within the null per layer, against the 2.5 / 97.5 lines. Reads the same comparison as a rank rather than a z, which is the honest one at 16 draws. | `per_layer.nulls` | done |
| N3 | `null_z_trajectory` | z per statistic vs step. **The interesting version of open question 1**: the raw fraction can be flat while distinguishability from random is not, and only this figure separates them. | `per_layer.nulls` | done |
| N4 | `gaussian_expectation` | Observed vs null mean for each statistic as paired points. A z near zero on the complex fraction is the EXPECTED result and this figure says so on its face; the live comparisons are θ and `henrici_relative`, where a trained matrix sitting *below* the null means training made V more normal. | `per_layer.nulls` | done |

### `curiosities` — the speculative half

Not verdict figures, none in any falsification table. They exist because the
per-layer × per-checkpoint table makes this structure cheap to look at, and
looking is how the next question gets found. Each carries a one-line "what
would be interesting here" note in its own docstring, and a figure that shows
nothing is a result worth having drawn once.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| X1 | `spectrum_fingerprint` | Layer × metric heatmap for one checkpoint, each metric z-scored down its own column. One checkpoint's whole Block 1a table as a single image, for spotting which layers are odd before deciding what to plot. | `block1a.per_layer` | done |
| X2 | `rotation_clock` | Mean θ per layer on polar axes with radius `rho_mean`, coloured by depth. "How fast does each layer rotate, and how hard" as a shape rather than two columns of a table. | `block1a.per_layer` | done |
| X3 | `spectral_annulus` | Where each layer's eigenvalues live in the complex plane. With `planes.npz` present, every plane as a point; without it, the (ρ ± sd) × (θ_min … θ_max) sector its summary statistics imply, labelled on the figure as the reconstruction it is — a sector drawn from four order statistics looks exactly like one drawn from a distribution and is not the same claim. | `planes.npz`, else `per_layer` | done |
| X13 | `theta_ridge` | The rotation-angle distribution at every depth, stacked, on a shared vertical scale with each layer's mean ticked on its baseline. X8 draws θ's coefficient of variation, which cannot tell a wide unimodal spread from two tight clusters far apart; this can. Bimodal ridges mean each layer holds two populations of planes and every mean-based comparison in the phase averages across them. | `planes.npz` | done |
| X4 | `training_ribbon` | Layer × step ribbon of the complex energy fraction with the dated events drawn as rules. The whole phase as one image. | `block1a.per_layer` | done |
| X5 | `convention_divergence` | Per-eigenvalue fraction against the legacy per-block one, per layer and per step, with their ratio beneath. The factor of ~2 between two numbers that shipped under the same name, measured rather than asserted. | `block1a.per_layer` | done |
| X6 | `norm_vs_spectrum` | `ov_frob` against the complex energy fraction over every (layer, checkpoint), coloured by step. Study A's OV spectral-norm confound (partial ρ to −0.71) is a scatter this phase can draw for free, and it is the quantity the rank-gate divergence scales with. | `block1a.per_layer` | done |
| X7 | `depth_coupling` | Correlation across checkpoints between every pair of layers' complex fraction. Do the layers move as one block, in depth-adjacent groups, or independently? Three different mechanisms, one image. | `block1a.per_layer` | done |
| X8 | `theta_coherence` | `theta_std / theta_mean` per layer per step. Where in depth × training the rotation is coherent (one angle) versus scattered (a bag of unrelated planes). | `block1a.per_layer` | done |
| X9 | `spectral_drift_arrows` | Each layer's (dim fraction, energy fraction) position at the first checkpoint and at the last, drawn as an arrow. Whether training moves layers along the y = x line or off it. | `block1a.per_layer` | done |
| X10 | `sweep_cost` | Measured `wall_time_seconds` per checkpoint against `run_2b.estimate_cost`'s prediction. Cheap, and the thing anyone planning the pythia-1.4b sweep actually wants — the estimator is calibrated at d = 1024 and its d³ scaling has never been checked against a real run. | combined + `run_2b.estimate_cost` | done |
| X11 | `violation_depth_density` | Across every Block 1b run, where in depth violations land, per frame. Phase 2's "attribution reorganises while the count stays flat" is a claim about *which* layers; this is the histogram it implies. | `frames[*].counts.violation_layers` | done |
| X12 | `rho_theta_joint` | Every (layer, checkpoint) as a point in (θ_mean, ρ_mean), coloured by step. Where the operator's rotation lives, as a cloud that training moves through. | `block1a.per_layer` | done |

---

## Data gaps

Seven quantities Phase 2b computed and then dropped before writing. **All
seven emissions have landed**, each as a key appended to an existing artifact
or a new sidecar, nothing renamed and nothing rewritten. The loaders still
treat all seven as optional and print a skip reason when absent, because run
directories written before the change do not have them and must stay
readable — and, importantly, the gaps are DETECTED from the artifact rather
than assumed, so both kinds of directory read correctly with no flag to set.

<a id="g1"></a>
**G1 — per-layer energies, effective rank, and IP summaries per frame.**
`p2b_energy.trajectory_scalars` computed them for every frame and
`rotational_rescaled.comparison_to_json` kept only the derived counts, so the
energy curve — the object a "violation" is a feature of — could not be drawn
for any frame, and neither could the gate quantity deciding which transitions
are scored. This was the largest gap in the phase by some margin; F1 and F3
were the counting-level stand-ins for it. *Landed: `frames[*].per_layer`
(`energies` per β, `effective_rank`, `ip_mean`, `ip_mass_near_1`). Drawn by
F8.*

<a id="g2"></a>
**G2 — the rescaler growth curve.** `rescaled_trajectory` returns
`r_cum_max_abs`, recorded deliberately "even when truncation does not fire,
so 'the rescaling was fine' is a measurement rather than the absence of a
flag" — and the serializer reduced it to its maximum, which is the flag
again. *Landed: `frames[*].r_cum_max_abs`. Drawn by F10.*

<a id="g3"></a>
**G3 — per-plane (ρ, θ) lists.** `top_rotation_planes` returns `rhos`,
`thetas`, `signs` and `indices` alongside the `(d, 2)` bases, and
`summary_to_json` dropped the whole `planes` key — correctly for the bases,
which are arrays, but the four scalar lists went with them. So θ and ρ
existed only as mean / sd / min / max / median, and no figure could draw the
distribution. *Landed: `planes.npz` per checkpoint (the split Phase 1b made
for its axes — at d = 1024 a layer holds up to 512 planes and
`phase2b_results.json` is read whole), plus `plane_quantiles` in the JSON for
a reader with only that. Drawn by S9, X3 and X13.*

<a id="g4"></a>
**G4 — per-head circuit results.** `head_circuits.py` was landed and tested
(`PLAN_2b.md` item 19) and was never called by `run_2b.py`, so no artifact
carried `summed_vs_per_head`, `head_agreement`, or any per-head spectrum —
the gap with the most figures behind it, since the phase's headline is a
statistic of `ov_total = Σ_h ov_per_head`, an operator the model never forms.
The per-head arrays were in the OV npz the whole time; nothing read them.
*Landed: `p2b_io.load_ov_data` reads `ov_head{h}_{layer}` (sorted
numerically), and `run_2b.run_head_circuits` writes a
`block1a_head_circuits` subresult. Drawn by the whole `heads` class.*

<a id="g5"></a>
**G5 — the precision surface.** `rotational_schur.precision_surface` wired
`core.precision_policy.analyze_ov_precision` to this phase's own fraction
function and was never called from the runner, so item P2 — "84–97% complex
may be an fp16-storage artifact" — was a caveat in prose with no number.
*Landed: `block1a.precision` under `--with-precision`, off by default because
it costs ~10 dense eigendecompositions per layer. Drawn by S10.*

<a id="g6"></a>
**G6 — nulls covered three statistics.** `frac_repulsive_real_part` — the
quantity with the clearest dynamical reading — had no control, while the
complex fraction, where a Gaussian saturates and z ≈ 0 is foregone, had one.
*Landed: it joins `NULL_STATISTICS`, at no cost, because
`null_comparison_multi` now draws ONE null sample per layer and reads every
statistic off it. The previous per-statistic version multiplied the null's
cost by the number of statistics and scored each against a different random
matrix, so a difference between two z-scores mixed a real difference with two
independent draws.*

<a id="g7"></a>
**G7 — per-transition severity.** `count_violations` returns `rel_drops`,
the relative energy drop at every transition with NaN where unscored;
`comparison_to_json` kept `sum_severity` and `max_severity` and dropped the
array. So "this frame has 4 violations" could be drawn and "these four are
all marginally over 1e-3" versus "one is catastrophic" could not. *Landed:
`counts[β].rel_drops`. Drawn by F9.*

<a id="note-1"></a>
**NOTE-1 — two tracked statistics had no dispersion scale. FIXED.** Not a
dropped emission; a key mismatch, found while drawing R1 and R3.
`p2b_report.collect_trajectory` got each statistic's across-layer spread by
mapping the summary name back to the per-layer column it aggregates, via
`_per_layer_key_for`, which stripped a suffix. Two of the seven entries in
`TRACKED_STATISTICS` did not survive that round trip:

| statistic | the old heuristic gave | the per-layer key actually is |
|---|---|---|
| `complex_energy_fraction_legacy_mean` | `complex_energy_fraction_legacy` | `complex_energy_fraction_legacy_per_block` |
| `theta_mean_across_layers` | `theta` (the `_mean_across_layers` suffix was stripped whole) | `theta_mean` |

Both collected an empty `vals` list, reported `spread = NaN` at every
checkpoint, and carried NaN through `flatness`'s `range_in_spreads`,
`range_in_standard_errors` and `range_excess_over_noise`, and through every
`delta_in_spreads` in `interval_deltas` and `align_to_transitions`. The
trajectories themselves were always fine — `values` and `steps` were correct
— so T1 and T2 drew them normally, and only the spread-relative statistics
were unavailable. That is why nothing failed.

*Fixed in `p2b_imaginary/p2b_report.py`*: the mapping is now the explicit
`SUMMARY_TO_PER_LAYER` table, `resolve_per_layer_key` warns instead of
guessing when a statistic is unregistered, and `collect_trajectory` returns
`spread_status` so a figure can distinguish "no dispersion scale" from "a
dispersion of zero". R1 and R3 still render the missing-scale state
distinctly — a run directory written before the fix has the same NaNs.

One more worth naming even though no figure needs it closed: Block 1b's
activations are in the **`l2_sphere` frame, not the LN frame attention
actually reads** (`frame_spec_for_activations`, and `PLAN_2b.md`'s deferred
item). Every Block 1b figure in this package prints the frame kind in its
subtitle rather than leaving it implicit, because the claim being tested is
about the operator attention applies and the frame is a live caveat on it.

---

## A note on the step axis

The `log10(step+1)` convention, the viridis step colormap and the separate
step-0 style live in `p1_mstate_tracking/visualization/checkpoints.py`, and
this package uses them — but that module imports `.series`, which reaches
`core/plot_utils.py` and so `sklearn`, for figures Phase 2b has no interest
in. `style.py` therefore imports the helpers lazily and falls back to a local
copy of the same four functions when that chain is unavailable. The fallback
is a mirror, not a second convention: the smoke test asserts the two agree
elementwise whenever both are importable, so they cannot drift. The name
grammar itself is not mirrored — `core/model_family.py` is stdlib-only and is
imported directly, which is the reason it was moved there in the first place
(`PLAN_2b.md` item 5).

---

## Running it

```
python -m p2b_imaginary.visualization \
    --p2b_dir results/<phase2b-output-dir> \
    --out     blog_figures/p2b \
    [--classes spectrum heads frames trajectory report verdicts nulls
               curiosities] \
    [--steps 512 3000] [--prompts wiki_paragraph] \
    [--external phase2_frac_repulsive.json] \
    [--list_runs]
```

Figures land in `--out`: `{stem}/` per checkpoint, `{stem}/{prompt}/` per
Block 1b run, and a flat `_cross/` for everything on the training axis.
`--list_runs` prints what was discovered and what each checkpoint is missing,
without drawing anything — the fastest way to find out which gap is biting a
particular directory, and the fastest way to see that a sweep was
`--blocks 1a` before wondering where the `frames` figures went.

Pointing `--p2b_dir` at a pre-rewrite directory raises, through
`p2b_io.refuse_legacy_run_dir` rather than a check restated here: those
numbers were scored with an absolute 1e-6 threshold and a 3.0 rank gate, and
their `elim_rotation` column is an algebraic identity.

To see the whole catalogue without a real sweep:

```
python -m p2b_imaginary.visualization --fixture --out /tmp/p2b_figs
```

which builds a synthetic Phase 2b output directory (`_fixture.py`) at
d = 12 over 6 layers and 6 checkpoints, and draws everything against it. The
weights are random and **no result should ever be read off the numbers** —
but the numbers are produced by the phase's own `run_block_1a`,
`analyze_rotational_rescaling` and `comparison_to_json`, not hand-written
JSON, so every key, every verdict string and every refusal status in the
fixture is one the phase can really emit. A fixture that writes its own JSON
tests the fixture.

---

## Not doing

- **Anything needing weights.** No Schur decomposition, no `expm`, no
  `ov_weights_*.npz`. Block 1a is cheap enough that a figure module could
  re-derive its own spectrum, which is exactly why the rule is written down:
  the phase has had three definitions of "rotational fraction" in one file
  once already.
- **Blocks 2, 3 and 4.** Unwired by design until their maths is redefined
  (`PLAN_2b.md` items 10–12), and Block 3 and Block 4 are degenerate as
  written — a full-rank "imaginary projector" and a curvature that is
  identically 1. There is nothing to draw and drawing it would be worse than
  the gap.
- **Re-deriving a verdict, a rate, or a flatness number.** All of them are
  read from the artifact or computed by calling `p2b_report`. A figure
  disagreeing with `phase2b_summary.txt` is a bug here.
- **Interactive/HTML output.** Every figure is a PNG at 150 dpi on the
  project's `BLOG_STYLE`, matching Phases 1, 1b, 1c and 2. One rendering
  path, one look.
