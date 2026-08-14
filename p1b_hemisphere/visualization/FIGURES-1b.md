# Phase 1b — FIGURES

The figure catalogue for `p1b_hemisphere/visualization/`, and the tracker for
building it. Companion to `status-1b.md` (what the phase found) and
`design-1b.md` (why it is built this way). This file answers a third question:
**what does Phase 1b look like**, and which parts of it can be drawn from the
artifacts the phase actually writes.

Read the status column before citing any figure. Several entries here are
`blocked` not because the figure is hard but because the quantity it draws is
computed and then thrown away at serialization time — those are listed under
[Data gaps](#data-gaps) with the one-line emission fix each needs.

---

## Ground rules

**Artifacts only, never a model.** Every figure loads from a Phase 1b output
directory (`phase1b_{stem}.json`, `phase1b_{stem}_particles.npz`,
`phase1b_cross_run.json`). Nothing here imports torch, loads weights, or
recomputes an analysis. Anything that decides what a number *means* — a regime
threshold, a verdict rule — is imported from `p1b_hemisphere`, never restated,
so a threshold change cannot leave the figures asserting the old one. This is
the same contract `p2_eigenspectra/visualization` works under.

**Draw the continuous quantity, not the label.** status-1b R1 and R3 are both
the same failure in different clothes: a binary regime label was reported where
a continuous, null-referenced quantity was available. Figures follow the
correction — `normalized_margin` over `cone_regime`, `separation_ratio` over
`strong_bipartition`, the Fiedler projection over the hemisphere sign. Regime
strips still appear, but as an annotation band beside the continuous panel,
never as the whole figure.

**Show the null wherever one exists.** Block 3 has matched nulls
(`--n-null`); Block A has `1/sqrt(d)`; `border_vs_noise` has AUC 0.5;
`match_overlap` has `IDENTITY_THRESHOLD`. Each of those is drawn as a reference
line or band in every panel of the quantity it references, so "100% of layers"
is never plotted without the thing it should be compared against.

**A missing input is a skipped figure, not a crash.** Phase 1b runs standalone
off Phase 1 output; Blocks 5/6 need Phase 2 artifacts, `--n-null` is opt-in, and
`hdbscan_labels` only exist when the cross-reference resolved. Every figure
declares what it needs and no-ops with a printed reason when it is absent.

**Color by job.** Categorical hues in fixed order (validated for CVD
separation, see `style.py`); one hue light→dark for magnitude; a two-hue
diverging pair with a neutral midpoint for the one genuinely signed quantity in
the phase, the Fiedler value; gray reserved for invalid/degenerate. Marker
shape carries any identity that color also carries, so no figure is
color-alone.

---

## Figure classes

Eight classes, each its own module and each selectable from the CLI
(`--classes regime cone …`).

| Class | Module | Scope | Figures |
|---|---|---|---|
| `regime` | `regime.py` | Block 0 — bipartition quality | 7 |
| `cone` | `cone.py` | Block 3 — cone-collapse LP | 6 |
| `tracking` | `tracking.py` | Block 1 — identity, rotation, events | 6 |
| `membership` | `membership.py` | Block 2 — per-token, nesting, boundary | 6 |
| `axis` | `axis.py` | Block A — axis identity vs PCA | 4 |
| `curiosities` | `curiosities.py` | exploratory / speculative | 11 |
| `crossrun` | `cross_run.py` | model × prompt aggregation, verdict | 5 |
| `checkpoints` | `checkpoints_1b.py` | training-step axis | 4 |

Status values: **done** — implemented and exercised against the fixture;
**blocked** — implemented or specified but its input is not on disk (see
[Data gaps](#data-gaps)); **planned** — specified here, not yet built.

---

### `regime` — Block 0, what the bipartition actually is

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| R1 | `regime_strip` | Both classifiers as parallel per-layer bands: antipodal (`regime`) above, relative (`regime_relative`) below. The R1 retraction in one image — the top band can be uniformly `collapsed` while the bottom shows `separated`. | `per_layer.regime`, `.regime_relative` | done |
| R2 | `bipartition_quality` | `separation_ratio` vs depth, with within-half IP (both halves) and between-half IP as the two components it is built from. | `per_layer` | done |
| R3 | `centroid_angle` | Centroid angle vs depth against the π/2 line `strong_bipartition` requires. Draws the unreachability rather than asserting it. | `per_layer.centroid_angle` | done |
| R4 | `eigengap_depth` | `bipartition_eigengap` vs depth — how sharply k=2 is preferred, layer by layer. | `per_layer` | done |
| R5 | `hemisphere_balance` | Hemisphere sizes as a stacked band plus `minority_fraction`, with the `collapsed_minority` threshold marked. | `per_layer` | done |
| R6 | `boundary_fraction` | `fiedler_boundary_frac` vs depth — the population sitting near zero on the axis, i.e. the tokens the sign label is least meaningful for. | `per_layer` | done |
| R7 | `asymmetry_depth` | Block 4's `|A−B|/(A+B)` vs depth, with the strong-bipartition-layer mean called out where any exist. | `per_layer.asymmetry` | done |

### `cone` — Block 3, containment as a quantity

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| C1 | `cone_margin_depth` | `normalized_margin` vs depth with the zero line, regime band beneath, and null means overlaid as bands when `--n-null` was run. **The figure status-1b R3 asks for.** | `per_layer`, `cone` summary | done |
| C2 | `cone_regime_strip` | Per-layer regime with escalation markers (`cone_escalated`) — where a reduced-space split was re-solved at full d. | `per_layer` | done |
| C3 | `cone_null_z` | z vs shuffled-dimension and z vs uniform-sphere nulls, as two stacked panels sharing a layer axis (never two y-scales in one). | per-layer null fields | blocked — [G1](#g1) |
| C4 | `cone_binding` | `n_binding` and `d_eff` vs depth — how many tokens hold the witness, in how many effective dimensions. Distinguishes "cone-collapse" from "n < d". | `per_layer` | done |
| C5 | `cone_witness_tokens` | Which tokens are binding, and how often across layers. The cone has a small support; this names it. | `cone.per_layer.binding_tokens` | blocked — [G1](#g1) |
| C6 | `cone_vs_dimension` | Across every run: `normalized_margin` against `n_tokens / d_eff`. If the margin is dimension counting, this is where it shows as a trend rather than a verdict. | all runs | blocked — [G1](#g1) |

### `tracking` — Block 1, does the axis keep its identity

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| T1 | `axis_rotation` | Per-transition rotation and its cumulative sum, two panels. Cumulative is the one that answers "does the axis wander or hold". | `per_layer.axis_rotation` | done |
| T2 | `match_overlap` | Identity persistence per transition against `IDENTITY_THRESHOLD`, imported from the phase rather than hardcoded. | `per_layer.match_overlap` | done |
| T3 | `crossing_counts` | Tokens changing hemisphere per transition, as a fraction of n_tokens so it is comparable across prompts. | `per_layer.crossing_count` | done |
| T4 | `event_timeline` | birth / collapse / swap / shear / drift on the layer axis, one row per type, marker shape per type so identity is not color-alone. Under the antipodal `regime_key` this is expected to be empty — an empty panel that says so is the point (R4). | `events` | done |
| T5 | `crossref_events` | Axis rotation at merge vs off-merge, and crossings at violation vs off-violation layers, as paired bars with n annotated on each. The baseline the old summary omitted. | `summary.crossref_with_phase1` | done |
| T6 | `persistence_length` | Regime persistence length per layer, with the `regime_key` the run used printed in the subtitle — the difference between the foreclosed and reachable vocabularies is the whole content of R4. | Block 1 | blocked — [G2](#g2) |

### `membership` — Block 2, tokens and the boundary

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| M1 | `stability_hist` | Distribution of per-token stability, with the mean and the 0.5 line. | `per_token` | done |
| M2 | `border_vs_stability` | `border_index` against `stability_score`, one point per token. Tests the obvious hypothesis — that unstable tokens are boundary tokens — visually before anyone builds on it. | `per_token` | done |
| M3 | `first_stable_layer_hist` | When tokens settle, with never-stable as an explicit terminal bar rather than a dropped row. | `per_token` | done |
| M4 | `nesting_r_c` | `r_c` distribution over clusters against the nesting poles at 0 and 1, plus fully-nested fraction per layer. | `hdbscan_nesting` | blocked — [G3](#g3) |
| M5 | `border_vs_noise_auc` | AUC vs depth against the 0.5 no-relationship line. **Phase 5c's question as a figure** — is the unclustered population the boundary population. | `border_vs_noise.per_layer` | blocked — [G3](#g3) |
| M6 | `noise_vs_clustered_margin` | Mean \|v\| for HDBSCAN-noise vs clustered tokens per layer — the AUC's two underlying distributions, since an AUC alone hides magnitude. | `border_vs_noise.per_layer` | blocked — [G3](#g3) |

### `axis` — Block A, is the Fiedler axis anything new

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| A1 | `axis_cosines` | \|cos\| to centered PC1, uncentered PC1, and the token mean vs depth, with `1/sqrt(d)` drawn as the chance floor beside every curve. The mean row is the control that must stay at chance by construction. | `axis_identity.per_layer` | done |
| A2 | `redundancy_strip` | Per-layer verdict (`pc1` / `top_pc_block` / `distinct` / `degenerate`) as a band, with the modal verdict annotated. **The Phase 5 caveat as a picture** — if this is `pc1` everywhere, Phase 5 is using PC1 under a more expensive name. | `axis_identity.per_layer` | done |
| A3 | `pc_subspace_fraction` | Fraction of the axis inside the top-k principal subspace, with PC1's explained variance beneath it — the axis can be redundant because it *is* PC1, or because PC1 eats everything. These separate those. | `axis_identity.per_layer` | done |
| A4 | `axis_vs_pc1_scatter` | `cos_axis_pc1` against `pc1_explained_variance` pooled over every layer of every run: is redundancy a property of the axis or of how anisotropic the cloud happens to be? | all runs | done |

### `curiosities` — the speculative half

These are not verdict figures and none of them is in the falsification table.
They exist because the particle table makes per-(layer, token) structure cheap
to look at, and looking is how the next question gets found. Each carries a
one-line "what would be interesting here" note in its own docstring, and a
figure that shows nothing is a result worth having drawn once.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| X1 | `fiedler_barcode` | Layer × token heatmap of the signed Fiedler value, diverging palette centered on zero. The whole run as one image — every other Block 0/1/2 quantity is a projection of this. | particles | done |
| X2 | `hemisphere_ribbon` | Layer × token hemisphere membership with tokens sorted by stability. Laminar bands vs turbulent bands, at a glance. | particles | done |
| X3 | `token_flow` | The 2×2 hemisphere transition matrix per layer pair, drawn as flow bands. Where the mixing actually happens. | particles | done |
| X4 | `position_vs_hemisphere` | Hemisphere membership against sequence position, with position 0 called out. If the split is "the sink vs everything else", it appears here first. | particles | done |
| X5 | `token_class_split` | Hemisphere membership broken down by token surface class — punctuation, leading-space, numeric, subword continuation, alphabetic. A semantics proxy that needs no Phase 2 artifacts, and a cheap preview of whether Block 6 will find anything. | particles | done |
| X6 | `most_volatile_tokens` | The least-stable tokens with their per-layer trajectories as mini-ribbons, labeled with the token string. | particles / `per_token` | done |
| X7 | `cone_opening_polar` | The cone's half-angle vs depth on polar axes — "how tight is the containment", as a shape rather than a number. | `per_layer.normalized_margin` | done |
| X8 | `border_dwellers` | Per layer, the tokens nearest the Fiedler boundary, named. The population M5 measures in aggregate, listed individually. | particles | done |
| X9 | `stability_landscape` | 2D density of (border_index, stability) with the extremes labeled — M2's scatter as a density, for prompts long enough that the scatter saturates. | `per_token` | done |
| X10 | `hemisphere_dwell_histogram` | Layers spent in hemisphere 1 per token. Bimodal means two genuine populations; unimodal at n_layers/2 means the sign is noise. | particles | done |
| X11 | `run_fingerprint` | Layer × metric heatmap, each metric z-scored down its own column — one run's whole per-layer table as a single fingerprint, for spotting which layers are odd before deciding what to plot. | `per_layer` | done |

### `crossrun` — across models and prompts

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| V1 | `verdict_card` | `global_verdict` rendered as labeled status tiles, each carrying its supporting number and sample size. Booleans named for what True means, as `p1b_report` insists. | `cross_run.global_verdict` | done |
| V2 | `model_prompt_heatmap` | One heatmap per aggregated scalar over model × prompt. | per-run summaries | done |
| V3 | `scalar_spread` | Per-model distributions of the aggregated scalars over prompts — a median with spread, not a mean bar. | per-run summaries | done |
| V4 | `event_counts_by_model` | Event-type counts per model, stacked. | `events` | done |
| V5 | `prompt_length_vs_cone` | `n_tokens` against cone-collapse fraction across runs, with `LONG_PROMPT_TOKENS` marked — the verdict's own threshold, drawn on the data it partitions. | per-run | done |

### `checkpoints` — the training-step axis

Grouped by family and plotted against `log10(step+1)`, using
`p1_mstate_tracking/visualization/checkpoints.py`'s step-axis, colormap, and
family-grouping helpers rather than restating them, so Phase 1b's checkpoint
figures cannot drift from Phase 1's and Phase 2's.

| # | Figure | Shows | Source | Status |
|---|---|---|---|---|
| K1 | `checkpoint_scalars` | Each aggregated scalar vs log-step, one panel per scalar, one line per family. | `cross_run.by_checkpoint` | done |
| K2 | `regime_by_step` | Regime-fraction composition vs step, stacked — how the classifier's verdict moves through training. | per-run summaries | done |
| K3 | `checkpoint_depth_heatmap` | Layer × step heatmaps for `normalized_margin` and `separation_ratio`. Depth and training on the same picture. | per-run `per_layer` | done |
| K4 | `axis_settling` | Angle to the final-checkpoint axis vs step per layer, with `axis_settling_step`'s tolerance band and the settling step marked. **The PREDICTIONS.md claim (b) figure**, and the only one in the phase tracking the axis's direction rather than λ₂'s magnitude. | saved axes | blocked — [G4](#g4) |

---

## Data gaps

Four quantities the phase computes and then drops before writing. Each is an
additive emission — a key appended to an existing artifact, nothing renamed,
nothing rewritten — and each unblocks the figures listed above. The loaders
treat all four as optional and print a skip reason when absent, so figures
built before the emission lands do not break after it does, and old runs stay
readable.

<a id="g1"></a>
**G1 — cone per-layer detail.** `_save_run` writes
`cone_collapse_to_json(...)["summary"]` and discards `["per_layer"]`, taking
the null z-scores, percentiles, `d_eff`, and `binding_tokens` with it.
Blocks C3, C5, C6. *(landed: `cone_per_layer` key.)*

<a id="g2"></a>
**G2 — Block 1 persistence.** `persistence_length` is returned by
`analyze_hemisphere_tracking` and never reaches the per-run JSON; only its
derived event counts do. Blocks T6.

<a id="g3"></a>
**G3 — nesting and boundary per-layer.** The per-run summary carries
`hdbscan_nesting_overall` and `border_vs_noise_mean_auc` — one number each —
while `membership_to_json` produced a full per-layer breakdown. Blocks M4, M5,
M6, which is the whole Phase 5c thread. *(landed: `hdbscan_nesting` and
`border_vs_noise` keys.)*

<a id="g4"></a>
**G4 — activation-space axes.** `axis_identity_to_json` drops `axes` with the
comment that they belong in an npz; no npz is written. Without them
`cross_checkpoint_axis_rotation` and `axis_settling_step` have no input from
disk, so the phase's checkpoint headline cannot be drawn. Blocks K4.
*(landed: `phase1b_{stem}_axes.npz`.)*

A fifth is worth naming even though no figure here is blocked on it: **layer 0
is the embedding output, pre-any-LN** (status-1b open blocker 5) and is still
averaged into per-model means. Every depth-axis figure in this package marks
layer 0 rather than dropping it, since dropping it silently would repeat the
mistake in the other direction.

---

## Running it

```
python -m p1b_hemisphere.visualization \
    --p1b_dir results/<phase1b-output-dir> \
    --out     blog_figures/p1b \
    [--classes regime cone tracking membership axis curiosities crossrun checkpoints] \
    [--prompts wiki_paragraph] [--models gpt2-large] \
    [--list_runs]
```

Figures land in `--out`, one subdirectory per run stem for per-run classes and
a flat `_cross/` for the aggregate ones. `--list_runs` prints what was
discovered, and what each run is missing, without drawing anything — the fastest
way to find out which of the gaps above is biting a particular directory.

To see the whole catalogue without a real run:

```
python -m p1b_hemisphere.visualization --fixture --out /tmp/p1b_figs
```

which builds a synthetic Phase 1b output directory (`_fixture.py`) with the
same filenames, keys, and array shapes a real run writes, and draws everything
against it. The numbers are invented and no result should ever be read off
them; the shapes are real, so a figure that breaks against the fixture breaks
against a run.

---

## Not doing

- **Anything needing activations.** PCA scatters colored by hemisphere, cluster
  overlays in activation space, and the 3-D sphere renderings all need the
  arrays Phase 1 saves, not Phase 1b's artifacts. They belong to Phase 1's
  visualization package, which already owns `hdbscan_pca` and
  `projection_comparison`, and adding a second copy here to read a second
  phase's directory is how two packages start disagreeing about what a layer
  index means.
- **Blocks 5 and 6.** Mechanism (axis vs OV/embedding/heads) and semantic MI
  need Phase 2 OV artifacts and are not run, so there is nothing to draw. X5 is
  the deliberately cheap stand-in for Block 6 and is labeled as a proxy in its
  own caption.
- **Interactive/HTML output.** Every figure here is a PNG at 150 dpi on the
  project's `BLOG_STYLE`, matching Phases 1 and 2. One rendering path, one look.
- **Re-deriving verdicts.** No figure recomputes a regime, a threshold, or a
  verdict; they are read from the artifacts or imported from the phase. A
  figure disagreeing with `phase1b_cross_run.md` is a bug in this package by
  construction, which is the only way to keep it that way.
