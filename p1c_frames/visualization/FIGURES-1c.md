# Phase 1c — FIGURES

The figure catalogue for `p1c_frames/visualization/`, and the tracker for building
it. Companion to `status-1c.md` (what the phase found, and what it refuses to
answer) and `design-1c.md` (why it is built this way). This file answers a third
question: **what does Phase 1c look like**, and which parts of it can be drawn
from the artifacts `run_1c` actually writes.

Phase 1c is the phase where a headline stops being a comparison against
$t = \infty$ and becomes a number. Almost every figure here is therefore a
*difference against a null*, not a curve — and the ones that are curves carry
the null drawn behind them.

Read the status column before citing any figure. Four entries need an input the
phase computes and then drops at serialization time, and one whole
sub-experiment (D) is implemented but not wired into the driver, so nothing
writes its block. Those figures skip with a printed reason rather than failing.
`--list_runs` says which directory is which.

---

## Ground rules

**Artifacts only, never a model, never a re-analysis.** Every per-run figure
loads from a Phase 1c output directory (`{run}/p1c.json`, `{run}/p1c_curves.npz`).
Nothing here imports torch, loads weights, reads Phase 1's `activations.npz`, or
recomputes a sub-experiment. Anything that decides what a number *means* — a
verdict, a threshold, an adjudicator — is imported from `p1c_frames` and never
restated, so a figure that disagrees with the phase's own output is a bug here
by construction. Same contract as `p1b_hemisphere/visualization` and
`p2_eigenspectra/visualization`.

**The `theory` class is the one exception, and it is a narrow one.** Those
figures take no run artifacts at all: they draw `gamma_ode.integrate_gamma`,
`hemisphere_feasibility.wendel_probability`, `design_test.gegenbauer_normalized`
and `centroids.random_band` — the phase's own validated functions — over a grid
of arguments. That is a picture of the null model, not a new measurement, and it
is the class you can run before a single Pythia checkpoint exists. **No new math
lives in this package.** If a theory figure needs a quantity `p1c_frames` does
not already export, the quantity belongs in `p1c_frames`, not here.

**Draw the residual, not the fit.** `design-1c.md`: "The residual is the
deliverable. The fit is not." A figure that shows `ip_mean` and $\gamma_\beta$
overlapping prettily is the wrong figure; B1 shades the gap between them and B2
plots that gap on its own zero-centred axis. Where a figure does show the two
curves, the shading is the subject and the curves are the construction.

**Three step definitions, always all three.** status-1c finding 1: the §8
definition understates $T_{\rm eff}$ by ~5.7×, in the direction that would make
our own headline an artifact of depth. No figure in the `integration` class plots
one definition alone, and A4 exists solely to show whether they straddle $t^\ast$
— i.e. whether the answer is a measurement or a choice of units.

**Never clip an unreachable layer.** `time_residual` is `nan` where the observed
`ip_mean` falls below the null's starting point, and those layers are the
*strongest* resistance signal (`gamma_null.py`, and status-1c's note that Phase 1
already found mid-network mass 20× below the embedding floor). B3 draws them as
marked bars on the zero line with a count, never as gaps and never as zeros.

**A band is not an error bar unless it is labelled one.** The β envelope (B5,
B6) is a *bracket over an undecided reduction*, not a confidence interval, and
carries `envelope_verdict`'s own wording in the caption. Where the observed curve
is inside the band, the figure says the reduction decides the sign — which is
exactly what `run_1c` refuses to paper over with a default.

**A skipped sub-experiment is a drawn statement, not an absent file.** Runs
differ in what they could answer: `norms` missing kills A and C, no `beta_eff`
kills A and B, no `clusters.npz` kills F. V5 draws that matrix for the whole
sweep so a hole in the results is visible as a hole rather than as a shorter
list of figures.

**Color by job.** Categorical hues in fixed order (validated for CVD separation,
see `style.py`); one hue light→dark for magnitude; a diverging pair with a
neutral midpoint at exactly zero for the two genuinely signed quantities in the
phase — the vertical residual and the time residual. Gray is reserved for
invalid / not-computed and is never a data color. Every figure encoding a verdict
by color also encodes it as position or a direct label.

---

## Figure classes

Ten classes, each its own module and each selectable from the CLI
(`--classes integration null …`).

| Class | Module | Scope | Figures |
|---|---|---|---|
| `integration` | `integration.py` | A — effective integration time | 6 |
| `null` | `null_model.py` | B — the $\gamma_\beta$ residual | 8 |
| `moments` | `moments_fig.py` | C — cumulant ladder, rank, sinks | 5 |
| `frames` | `frames_fig.py` | D — the four-frame table | 5 |
| `feasibility` | `feasibility.py` | E — Lemma 6.4's cone condition | 4 |
| `designs` | `designs.py` | F — spherical designs | 5 |
| `curiosities` | `curiosities.py` | exploratory / speculative | 11 |
| `theory` | `theory.py` | the null model itself, no artifacts | 7 |
| `crossrun` | `cross_run.py` | model × prompt, and what could run | 6 |
| `checkpoints` | `checkpoints_1c.py` | the training-step axis | 6 |

Status values: **done** — implemented and exercised against the fixture;
**done — needs [Gn]** — implemented, and drawing it needs an input the phase
does not currently persist (see [Data gaps](#data-gaps)); **planned** —
specified here, not yet built.

---

### `integration` — A, how far the network actually integrates

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| A1 | `step_size_depth` | $h_\ell$ under all three definitions vs depth, log-y. The gap between `h_displacement` and `h_calibrated` is finding 1 at every layer rather than as one summary factor. | `A.h_*` | done |
| A2 | `teff_accumulation` | Cumulative $T_{\rm eff}(\ell)$ for all three definitions, with $t^\ast$ as a horizontal line and the crossing layer marked where one exists. **The P-γ2 figure.** | `A.h_*`, `A.t_star` | done |
| A3 | `field_magnitude` | Mean $\|\mathcal{X}\|$ per layer against the paper's bound of 1, with the implied understatement factor $1/\|\mathcal{X}\|$ on a twin axis. Why §8's definition is 5.7× low, layer by layer. | `A.field_mag` | done |
| A4 | `definition_straddle` | The three $T_{\rm eff}$ as a dot plot against $t^\ast$, `verdict.robust` printed as the caption. If the dots land on both sides, the figure says the answer is a definition. | `A.verdict` | done |
| A5 | `calibration_scatter` | `h_calibrated` against `h_displacement`, one point per layer, colored by depth, with $y = x$ and the run's own mean ratio drawn as a second line. | `A.h_*` | done |
| A6 | `attn_vs_full_step` | `h_attn_only` as a fraction of `h_calibrated` vs depth — how much of the block's motion the paper's model actually contains. Skips with a reason when sublayer streams were absent (status-1c open item 3). | `A.h_attn_only` | done |

### `null` — B, the residual

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| B1 | `residual_curve` | `ip_mean` with both nulls (orthogonal-init and observed-matched), the gap shaded. **The phase's central image.** | `B.*` | done |
| B2 | `residual_depth` | Residual and matched residual vs depth on a zero-centred axis, with `anisotropy_gap` annotated — the part of any disagreement that is non-orthogonal embeddings rather than resistance. | `B.residual*` | done |
| B3 | `time_residual` | `t_required` against `T_eff` spent per layer, and their difference beneath, with unreachable layers drawn as marked bars and counted. The measure that keeps its resolution where B2 loses it. | `B.time_domain` | done |
| B4 | `trajectory_in_time` | `ip_mean` against $T_{\rm eff}$ — the trajectory in the ODE's own time coordinate, with the continuous $\gamma_\beta(t)$ curve behind it and layer indices annotated along the path. Depth is a clock, and this is the figure that shows what kind of clock. | `B.*` | done |
| B5 | `beta_envelope` | The β envelope as a band with the observed curve over it, layers outside the band marked. Carries `envelope_verdict`'s wording. | `B.envelope_*` | done — needs per-head β |
| B6 | `residual_bracket` | `[residual_min, residual_max]` vs depth as a band around zero, with `sign_unambiguous` stated. The residual as the interval it actually is. | `B.residual_bracket` | done — needs per-head β |
| B7 | `collapse_fraction` | `time_fraction` and `gamma_fraction` side by side on the $\gamma$ curve they are read off, so a 20% time fraction is never mistaken for 20% of the clustering. | `B.collapse_fraction` | done |
| B8 | `beta_fallback_audit` | Per-layer β used, with fallback layers marked and `n_beta_fallback` stated. A null evaluated at a median β where the regression failed is not the same null. | `B.beta_per_layer` | done |

### `moments` — C, the ladder and the rank

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| C1 | `rank_panel_depth` | `shannon_raw`, `shannon_normed`, `pr_rank`, `norm_pr` vs depth on one axis. status-1's "MinRank → 2.3" and its frame-correct counterpart in one picture. | `C.panels` | done |
| C2 | `sink_ratio_depth` | `sink_ratio` vs depth against 1.0, with `norm_max_over_median` beneath. Near 1 means the raw rank is the norm distribution. | `C.panels` | done |
| C3 | `sink_adjudication` | `shannon_raw` against `norm_pr` and against `shannon_normed`, both correlations annotated, verdict imported from `adjudicate_sink_hypothesis`. The adjudicator as a scatter. | `C.sink_verdict` | done |
| C4 | `moment_identity_error` | Layer × β heatmap of `rel_err_two`, with the 1% contour and the β=5 column called out. Settles which energy columns are redundant (finding 8). | `C.moment_identity` | done |
| C5 | `cumulant_ladder` | $\kappa_1, \kappa_2, \kappa_3$ vs depth with the ladder source per layer, and $1/{\rm PR} = \kappa_2 + \kappa_1^2$ drawn as the identity it is. | ladder | **planned — needs [G2](#g2)** |

### `frames` — D, the four-frame table

Sub-experiment D is implemented (`frame_table.py`) and validated, but `run_1c`
has no `D` branch, so no run directory carries a `D` block — see
[G1](#g1). These figures are written against the shape `frame_table()` returns
and are exercised by the fixture; against a real directory they skip and say why.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| D1 | `frame_ip_mean` | `ip_mean` in all four frames vs depth, with `frame_disagreement.spread` per layer beneath. Whether frame choice is a footnote or a confound. | `D.per_layer` | done — needs [G1](#g1) |
| D2 | `frame_rank` | `pr_rank` per frame vs depth, with the raw effective rank drawn as the un-framed comparison (the 144.7 / 70.7 / 4.99 finding). | `D.per_layer` | done — needs [G1](#g1) |
| D3 | `sphere_license` | Per-layer γ coefficient of variation against ALBERT's 0.018, with the condition number on a twin axis and `cross_layer_mean_cv` stated — both halves of the paper's "constant across layers". | `D.sphere_license` | done — needs [G1](#g1) |
| D4 | `bias_energy_floor` | `energy_floor_frac` per β vs depth, with `bias_norm_ratio` and the $\kappa_1$ shift. A floor that does not depend on the tokens at all. | `D.bias_floor` | done — needs [G1](#g1) |
| D5 | `neg_eigen_mass` | Negative-eigenvalue mass of the Torgerson Gram per layer. Where "effective rank" stops meaning what it means elsewhere. | `D.per_layer` | done — needs [G1](#g1) |

### `feasibility` — E, Lemma 6.4's cone condition

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| E1 | `margin_depth` | The hull margin vs depth with zero, `min_margin_layer`, and the first infeasible layer marked; unconverged layers hatched, since an unconverged optimizer can only overstate the margin. **The reportable object, not the boolean.** | `E.margins`, `E.per_layer` | done |
| E2 | `wendel_reference` | Wendel probability against $n$ at this model's $d$, with this prompt's $n$ marked — P-H1's near-vacuity as a boolean, drawn once so nobody has to take it on trust. | `E.per_layer`, `wendel_probability` | done |
| E3 | `support_and_min_ip` | `support_size` and `min_pairwise_ip` vs depth. When the cone fails, this names how many tokens it took. | `E.per_layer` | done |
| E4 | `margin_shrinkage` | This run's margins against the measured i.i.d. reference (0.221 at n=20, 0.030 at n=512 at d=1024), so a small margin is read against how small a random cloud's would be. | `E`, finding 5 | done |

### `designs` — F, spherical designs

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| F1 | `q_ratio_depth` | $Q_k/Q_k^{\rm random}$ per degree vs depth with the random band shaded and 1.0 drawn. Never the raw $Q_k$ — for i.i.d. points $E[Q_k] = 1/n$ exactly, so a raw curve reads the cluster count. | `F.per_layer` | done |
| F2 | `outside_band_strip` | Layer × degree matrix of `outside_band`. The effect-size floor as a picture. | `F.per_layer` | done |
| F3 | `design_order_depth` | `t_design_vs_random` and `t_design_strict` vs depth, with `n_centroids` beneath — the design order and the m it was measured at. | `F.per_layer` | done |
| F4 | `mode_structure` | `n_modes` and `mass_at_modes` vs depth, plus every mode location as a point on an inner-product axis. Definition 9.1's other half. | `F.per_layer.modes` | done |
| F5 | `sharp_score_depth` | `sharp_score` vs depth with the clusterer arm printed in the subtitle — if the three arms disagree, the signal is a property of the clustering. | `F` | done |

### `curiosities` — the speculative half

Not verdict figures; none is in the falsification table. They exist because a
phase whose unit of work is the whole trajectory makes trajectory-shaped
questions cheap to look at, and looking is how the next question gets found.
Each carries a one-line "what would be interesting here" note in its docstring,
and a figure that shows nothing is a result worth having drawn once.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| X1 | `depth_clock` | Layer index against $T_{\rm eff}$, with a uniform clock as the diagonal. Where in depth the integration budget is actually spent — and whether "layer" is a fair time axis for any Phase 1 figure. | `A` | done |
| X2 | `teff_budget` | Each layer's $h_\ell$ as a stacked contribution to the total, sorted and cumulative. The 80/20 of the network's own clock. | `A` | done |
| X3 | `phase_portrait` | $({\rm ip\_mean}, \Delta{\rm ip\_mean}/\Delta T)$ for the observed trajectory, with the ODE's own $(\gamma, \dot\gamma)$ flow curve behind it. Does the network live on the paper's flow, off it, or on a different one? | `A`, `B` | done |
| X4 | `field_feedback` | $\|\mathcal{X}\|$ against `ip_mean` per layer. The field strengthens as the cloud clusters; this is the loop that makes collapse an instability rather than a drift. | `A`, `B` | done |
| X5 | `residual_barcode` | Layer × run heatmap of the residual, diverging at zero. The whole sweep as one image. | all runs | done |
| X6 | `t_star_landscape` | $t^\ast(n, \beta)$ as a contour field with every run plotted at its own $(n, \beta_{\rm eff})$. status-1c open item 4 — per-prompt $t^\ast$, never a pooled one — as a picture. | all runs, `collapse_time` | done |
| X7 | `sink_gallery` | `norm_max_over_median` against `sink_ratio` per layer, colored by depth. Which layers are sink-dominated, and whether that is a mid-network phenomenon. | `C.panels` | done |
| X8 | `margin_vs_clustering` | Cone margin against `ip_mean` per layer. Does clustering buy hemisphere containment, or cost it? | `E`, `B` | done |
| X9 | `design_vs_residual` | $Q_1$ ratio against the residual, one point per layer per run. Are the sharp configurations the resistant ones? The first place P-S1 and P-γ1 meet. | `F`, `B` | done |
| X10 | `beta_fan` | Per-layer β reductions (mean / median / min / max / attention-weighted) as a fan against the per-head range. What the undecided choice actually spans. | `B.beta_reduction` | done |
| X11 | `run_fingerprint` | Layer × metric heatmap, each metric z-scored down its own column — one run's whole per-layer table as a single fingerprint, for spotting which layers are odd before deciding what to plot. | all per-layer | done |

### `theory` — the null model itself

No run artifacts. Every figure calls a `p1c_frames` function directly, so these
are drawable today, against no data, and are the fastest way to see what the
phase is comparing against.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| T1 | `gamma_family` | $\gamma_\beta(t)$ for a family of β at several $n$, (SA) and (USA) in two panels, $t^\ast$ marked on each. | `integrate_gamma` | done |
| T2 | `collapse_time_table` | $t^\ast$ over $(n, \beta)$ as a heatmap for both models at both thresholds — `MATH.md` §3.2's table, which the phase reproduces to 0.005, as an image. | `collapse_time_table` | done |
| T3 | `beta_monotonicity` | $\gamma_\beta(T)$ against β at fixed $t$ for several $n$: (SA) decreasing, (USA) increasing. Why the envelope's endpoints swap between models. | `integrate_gamma` | done |
| T4 | `wendel_surface` | $P(\text{one hemisphere})$ over $(n, d)$ with our prompt range and $d = 1024$ marked. Where P-H1 is vacuous and where it is not. | `wendel_probability` | done |
| T5 | `gegenbauer_kernels` | Normalized Gegenbauer polynomials $C_k^\lambda$ at $d = 3$ against $d = 1024$. What $Q_k$ weights, and why high $d$ flattens it. | `gegenbauer_normalized` | done |
| T6 | `sigmoid_compression` | The vertical residual's dynamic range against the null level, beside the time residual's. Finding 3, as the argument rather than the anecdote. | `integrate_gamma` | done |
| T7 | `random_band_by_degree` | The measured 2σ band of the $Q_k$ ratio against degree at several $m$ — the effect-size floor P-S1 was missing. Simulated, so `--cheap` reduces the trial count and says so in the caption. | `random_band` | done |

### `crossrun` — the sweep, and what it could answer

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| V1 | `verdict_card` | P-γ2 robustness, the sink verdict, the envelope verdict and P-H1 feasibility as labelled tiles per run, each carrying its supporting number. Verdicts imported, never recomputed. | all runs | done |
| V2 | `teff_across_runs` | $T_{\rm eff}$ under all three definitions per run against that run's own $t^\ast$, sorted. **P-γ2 for the whole sweep in one figure.** | all runs | done |
| V3 | `residual_heatmap` | Model × prompt final residual, and the same for the time residual. | all runs | done |
| V4 | `length_vs_time` | `n_tokens` against $T_{\rm eff}$ and $t^\ast$ on the same axis, since $t^\ast$ is $n$-dependent and prompts span 20–512 tokens. | all runs | done |
| V5 | `availability_matrix` | Run × sub-experiment, colored by whether it ran and carrying the skip reason. `tools/preflight_1c.py`'s table as a figure, drawn from what actually landed. | `skipped`, `available` | done |
| V6 | `beta_audit` | β used per run colored by source (`geometry.json` vs `--beta-fallback`), with the envelope width where per-head betas existed. β is a measured property, not a convention. | all runs | done |

### `checkpoints` — the training-step axis

Grouped by family and plotted against `log10(step+1)` using
`p1_mstate_tracking/visualization/checkpoints.py`'s step-axis, colormap and
family-grouping helpers rather than restating them, so Phase 1c's checkpoint
figures cannot drift from Phase 1's, 1b's and 2's.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| K1 | `residual_vs_step` | \|final residual\| vs log-step with `adjudicate_p_gamma1`'s verdict printed. **The P-γ1 figure**, and the only one in the phase that adjudicates anything across training. | all runs | done |
| K2 | `teff_vs_step` | $T_{\rm eff}$ (three definitions) vs step against $t^\ast$. Does the network's clock change with training, or only what it does with the time? | all runs | done |
| K3 | `residual_depth_by_step` | Layer × step heatmap of the residual, diverging at zero. Depth and training on one picture. | all runs | done |
| K4 | `margin_vs_step` | `min_margin` and the first infeasible layer vs step. P-H1 across training rather than at one checkpoint. | all runs | done |
| K5 | `design_vs_step` | $Q_k$ ratio per degree vs step, with `adjudicate_p_s1_banded` against the step-0 run printed. **The P-S1 figure**, banded — an unbanded version would report noise. | all runs | done |
| K6 | `sink_ratio_vs_step` | `sink_ratio` and both min-ranks vs step. Whether status-1's rank-collapse row survives the frame correction across the whole sweep. | all runs | done |

---

## Data gaps

Quantities the phase computes and then does not persist, or does not compute at
all in the driver. Each is stated as what is absent, what would close it, and
which figures wait on it. **None is closed by this package** — writing them is
work for `p1c_frames/`, not for its visualization folder.

<a id="g1"></a>
**G1 — sub-experiment D is not wired.** `frame_table.py` is implemented and
validated (status-1c's D findings 9–11 come from it), and `run_1c --subexp`
accepts `D`, but `run_one` has no `D` branch: passing it produces nothing. The
whole `frames` class waits on a driver branch writing a `D` block of
`{"per_layer": [frame_table(...) per layer], "sphere_license": …,
"frame_disagreement": …}`. It also needs LN weights, which are a weight read
`[W]` rather than an artifact read, so this is the one gap whose cost is not
zero.

<a id="g2"></a>
**G2 — the cumulant ladder itself is dropped.** `run_1c` calls
`ladder_from_layer` per layer and keeps only `verify_moment_identity`'s output;
$\kappa_1, \kappa_2, \kappa_3$ and the `source` field never reach the artifact.
So C4 can be drawn (the reconstruction error survives) and C5 cannot (the ladder
does not). One line in the driver — appending `**lad` to each `checks` entry —
closes it. Until then the phase reports how *well* the ladder reconstructs
$E_\beta$ without reporting the ladder.

<a id="g3"></a>
**G3 — `h_attn_only` needs sublayer streams and usually does not have them.**
Not a serialization gap: the column is written, as `nan`, whenever
`step_sizes(attn_delta=None)`. It is also the *frame-correct* variant
(design-1c §2 — the paper's model has no FFN), so A6 is the figure most likely
to be blank on exactly the runs it matters for. status-1c open item 3 says to
check coverage across the 27 checkpoints before treating the attention-only
column as primary; A6 draws that coverage question rather than hiding it.

<a id="g4"></a>
**G4 — per-head β is optional and the envelope is what needs it.** Without
`beta_eff_per_head` in `geometry.json` the residual is a point estimate whose
error bar — the envelope — is unreported, and `run_1c` records `envelope_note`
saying so. B5, B6 and X10 skip in that case and print the note verbatim rather
than drawing a band that is not there. This is status-1c open item 1, which is
a hard blocker for reading B at all, not a refinement.

A fifth is worth naming even though no figure waits on it: **the causal/non-causal
choice is recorded per run and never compared**, because the sensitivity arm
(status-1c open item 2) has not been run. When it is, V2 and B1 take a second
series rather than a new figure.

---

## Running it

```
python -m p1c_frames.visualization \
    --p1c_dir results_p1c/ \
    --out     blog_figures/p1c \
    [--classes integration null moments frames feasibility designs \
               curiosities theory crossrun checkpoints] \
    [--models pythia-410m-step143000] [--prompts wiki_paragraph] \
    [--list_runs] [--cheap]
```

Figures land in `--out`, one subdirectory per run for per-run classes and a flat
`_cross/` for the aggregate ones; `theory` writes to `_theory/` and needs no
`--p1c_dir` at all. One bookkeeping note: X5, X6 and X9 belong to `curiosities`
but need every run, so they are drawn during the `crossrun` pass and counted
there — selecting `curiosities` alone gives you the eight per-run ones.

```
python -m p1c_frames.visualization --classes theory --out blog_figures/p1c
```

`--list_runs` prints what was discovered, which sub-experiments each run
carries, and what each is missing — the fastest way to find out which gap above
is biting a particular directory. `--cheap` reduces the simulation count in T7
(the only figure that simulates anything) and says so in its caption.

To see the whole catalogue without a real run:

```
python -m p1c_frames.visualization --fixture --out /tmp/p1c_figs
```

which builds a synthetic Phase 1c output directory (`_fixture.py`) with the same
filenames, keys, dotted-path grammar and array shapes `save_p1c` writes,
including a 6-checkpoint family and one deliberately degraded run. The numbers
are invented and no result should ever be read off them; the shapes are real, so
a figure that breaks against the fixture breaks against a run.

---

## Not doing

- **Anything needing Phase 1's activations.** PCA scatters, cluster overlays,
  sphere renderings and the Gram matrices themselves belong to Phase 1's
  visualization package, which already owns them. This package reads Phase 1c's
  output directory and nothing else — a second package reading a second phase's
  raw artifacts is how two packages start disagreeing about what a layer index
  means.
- **Closing any of the data gaps.** G1 and G2 are one driver branch and one line
  respectively, and both belong in `p1c_frames/`. A visualization package that
  computes the thing it wants to draw is one that can disagree with the phase.
- **Re-deriving a verdict.** `verdict`, `adjudicate_sink_hypothesis`,
  `envelope_verdict`, `adjudicate_p_gamma1` and `adjudicate_p_s1_banded` are
  imported and their strings printed. No figure recomputes a threshold.
- **Interactive/HTML output.** Every figure is a PNG at 150 dpi on the project's
  `BLOG_STYLE`, matching Phases 1, 1b and 2. One rendering path, one look.
- **Pooling across prompts.** $t^\ast$ is $n$-dependent and the prompts span
  20–512 tokens (status-1c open item 4), so every cross-run figure keeps the
  runs separate and draws each against its own $t^\ast$. There is no
  sweep-averaged residual figure in this catalogue, deliberately.
