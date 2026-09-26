<!-- p1d_cluster_ensemble/status-1d.md -->
# Phase 1d — STATUS

## Revived 2026-09-25: the active thread

**Why now (user, 2026-09-25):** Phase 10's experiments are on hold until the
project can say what a cluster is, because every Phase 10 row reads one
HDBSCAN partition and that choice is a confound in all of them. Evidence
that it is: `p10_cluster_function/handoff-10.md` "Parked", first item.

**What was done to revive it.** The code was deleted on 2026-09-23 with its
branch and survived in the local tag `dead/particle-methods-comparison-vpuads`
(`010448c`, 2026-08-20). It was restored from the tag, not rewritten. Three
things had drifted:

| drift | fix |
|---|---|
| `clustering.py` now calls `HDBSCAN(**params)`; the test read inline kwargs | the test reads the `params` literal (`tests/test_phase1d_methods.py`) |
| `PHASE1D` was never on `main`'s `core/artifacts.py` | re-registered, verbatim from the tag |
| the holdout guard (`core/holdout.py`, 2026-09-24) did not exist | `run_1d.py` refuses the 12 held-out prompts; `--v1-only` / `--allow-holdout` |

1d's 112 tests pass, conda `mets` (the full gate: `./scripts/check.sh`).
Paths in the sections below are the tag's: `p1_visualization/` is now
`p1_mstate_tracking/visualization/`.

**What 1d answers, and what it does not.** It tunes seven families per layer
against subsample stability with a whole-pipeline null, and grades each token
by how many tuned families agree (core / halo / contested). That settles
"is HDBSCAN at `min_cluster_size=2` a good choice, and which tokens does the
choice matter for". It does not by itself supply the theory's definitions: the
fixed-scale one (strong Rényi centres at separation `δ`, F13/F14 in
`p10_cluster_function/math-10.md` §7) and the persistence one (tokens staying
together over a window of layers). Tuning was also against subsampling, not
the float-noise re-run that moves HDBSCAN (`p10_cluster_function/status-10.md` §3), so the
first real run should measure that drift too.

### First real run: a smoke test, not a result

**Input:** `pythia-410m-step143000_wiki_paragraph` (v1; Stage 0 pin `64a4087`,
through `data/phase12/stage0_logs/stage0_index.json`), layers 0 / 12 / 18,
467 tokens, `--grid quick`, defaults otherwise (`n_null 20`, `top_m 3`,
`alpha 0.05`, seed 0), conda `mets`, `hdbscan` 0.8.41. **Output kept:**
`data/p1d/smoke_2026-09-25/`. **Re-run:**

    METS_DATA=<main>/data python -m p1d_cluster_ensemble.run_1d --v1-only \
      --results <main>/data/phase12/2026-09-22_21-20-26/pythia-410m-step143000_wiki_paragraph \
      --out <main>/data/p1d/smoke_2026-09-25 --layers 0 12 18 --grid quick

**Cost:** 519 s wall, 6 618 s CPU (~13 cores), 280 MB RSS: about 170 s per
layer, so one run at all 25 layers is about 70 min on the quick grid.

**Two defects before it ran clean.** (1) `separation_score` crashed when a
null draw came back as one cluster per token. (2) After that fix, `calibrate`
dropped such draws as NaN, which silenced the families that behave best on
noise: agglomerative at thresholds 0.05 / 0.25 had 0 of 20 usable draws and
abstained at every layer, and HDBSCAN at L18 kept 18 draws, a p floor of 0.053
above alpha, so it could not pass. **The first version of this section
reported both abstentions as findings; they were the gate's.** Now a
degenerate null draw scores the floor (separation −1, stability 0), and a p
floor above alpha refuses under its own branch (`/challenge-pr` on #98,
finding 1; `selection.py` `DEGENERATE_NULL_*`).

Every array is populated. What the run shows, from one run, so for design
only:

| layer | families admitted | abstained | consensus strength | consensus k | core / halo / contested |
|---|---|---|---|---|---|
| 0 | 6 | gmm | 0.25 | 4 | 3 / 263 / 201 |
| 12 | 7 | — | 0.38 | 5 | 0 / 111 / 356 |
| 18 | 7 | — | 0.36 | 5 | 0 / 237 / 230 |

- **Ranking by subsample stability picks each family's extreme scale.**
  k-means, spherical k-means, spectral and GMM take k = 2 (spectral k = 3 at
  L18), their coarsest option. Agglomerative takes threshold 0.05, its
  finest, where most tokens are singletons and a partition reproduces
  trivially. HDBSCAN takes `min_cluster_size` 5 at L0 / L12 and the shipped 2
  at L18. Co-association averaged over these compares different questions, so
  consensus strength 0.25–0.38 says little about agreement. Stability's pull
  toward trivial scales is a known property (Ben-David, von Luxburg & Pál
  2006, raised in the review). **Before 1d is used to define a cluster, it
  needs a stated scale:** families compared at matched k or matched `δ`, or
  read as levels of one hierarchy. That redesign reopens `design-1d.md`, so it
  triggers a literature scan first (`CLAUDE.md` "Literature scans").
  **Scan done 2026-09-25: `lit-1d.md`.** Stage 1's ranking on raw stability is
  the known defect (von Luxburg 2010 §2: raw instability scales with k
  whatever the data). The standard remedy, normalising by the null, fails
  here: the null mean is 0 at the fine end, and it can move k to the grid's
  other end (`/challenge-pr` on #99). Agglomerative's pick at L12 is k = 452,
  95 % singletons, which the trivial filter lets through: **a defect**
  (`lit-1d.md` option A0). The scan gives the options (§5) and a recommendation
  for the user; `design-1d.md` stays unchanged until the user picks one.
- **A0 fixed 2026-09-25** (see "A0: the singleton bound" below). Agglomerative
  leaves its finest threshold at L12 / L18, not at L0.
- The driver's `P-C1`–`P-C4` lines are unregistered and meaningless on one run
  and a quick grid; they are now printed and stored under
  `verdicts_status: "UNREGISTERED, tier 1: not adjudications"`.
- The scratch agreement read that prompted the revival is in
  `p10_cluster_function/handoff-10.md` "Parked" (first item), not repeated here.

### A0: the singleton bound (2026-09-25)

**Change.** `partition_summary` now calls a partition trivial when more than
`TRIVIAL_SINGLETON_SHARE = 0.5` of all tokens are singletons (placed, a
majority rule; `selection.py`). It is the other end of `TRIVIAL_DOMINANCE`.
HDBSCAN's refusals (-1) are not counted. **Input:** the same run, layers, grid
and defaults as "First real run". **Output:** `data/p1d/smoke_a0_2026-09-25/`.
**Re-run:** the command above with `--out <main>/data/p1d/smoke_a0_2026-09-25`.

| layer | agglomerative before | after | consensus k before → after | ARI(consensus before, after) | core / halo / contested before → after |
|---|---|---|---|---|---|
| 0 | 0.05 (k 215, 31 % singletons) | **unchanged** | 4 → 4 | 1.00 | 3/263/201 → 3/263/201 |
| 12 | 0.05 (k 452, 95 %) | 0.25 (k 291, 43 %, stab 0.95) | 5 → 4 | 0.99 | 0/111/356 → **257**/88/122 |
| 18 | 0.05 (k 434, 89 %) | 0.25 (k 176, 15 %, stab 0.82) | 5 → 4 | 1.00 | 0/237/230 → 26/391/50 |

- **The pick changes where the bound bites (L12, L18), not at L0.** L0's
  pick is token identity: 215 clusters for exactly 215 distinct L0 vectors,
  since layer 0 is the embedding and repeated tokens are identical (checked
  on `activations.npz`; `/challenge-pr` on #100, finding 2). No trivial
  branch can see that. Whether L0 belongs in 1d's readouts is for the user.
  A0 removes the extreme case; it does not give agglomerative a scale. That
  is still D / C (`lit-1d.md` §5).
- A `ward, k = 2` grid point now enters the top 3 (admitted at L12, stability
  0.70; refused at L18). Stage 1 had not reached it before.
- **The consensus partition hardly moves; the grading moves a lot.** At L12,
  core goes from 0 to 257 tokens. The mechanism is in the ensemble's voting,
  and A0 does not remove it (`/challenge-pr` on #100, finding 1). A singleton
  votes *against* grouping on every pair it touches, while an HDBSCAN -1
  abstains. Each family's vote is weighted by its raw stability
  (`selection_weights`), which is highest for fine partitions. The reviewer
  rebuilt the observed ensemble without agglomerative, against the stored
  core threshold (the null thresholds were not recomputed, so this gives the
  direction only): core goes 3 → 178 at L0, 257 → 457 at L12, 26 → 285 at L18.
  **Core / halo / contested counts are not a usable readout** until singleton
  voting and the weights are decided. That is a design question for
  `design-1d.md`, together with D.
- The unregistered `P-C4` line flipped from "CONFIRMED 2/2" to "FALSIFIED 1/2"
  (`P-C3` 0 % → 7 %). These are tier 1 lines, not adjudications. The flip
  shows how little one run on the quick grid supports them.

### Float-noise drift: does tuning reduce it? (2026-09-25)

**Question.** The shipped HDBSCAN partition moves between two sweeps whose
activations differ only by float noise (`p10_cluster_function/status-10.md`
§3). Does 1d's output move less? Tier 1, descriptive, no null.

**Input.** 8 v1 prompts (V1 minus `short_heterogeneous`) × steps 32 / 512 /
143000 × layers 6 / 12 / 18 = **72 layer-records**, each run through 1d twice:
on Stage 0 (`stage0_index.json`, pin `64a4087`, battery `06790b90dcfe`) and on
the pilot (`HDD_1TB/Mets_archive/2026-08-12_05-01-35`). Steps and layers were
fixed before any 1d output was read: 32 has the most baseline drift, 512 and
143000 are the ones Phase 10 reads. L0 is left out (identical in both sweeps).
`--grid quick`, defaults otherwise, seed 0, conda `mets`, code at `ac1e188`.
**Output:** `data/p1d/drift_2026-09-25/` (`stage0/`, `pilot/`, `stage0_rerun/`,
`drift_stage0_vs_pilot.json`, `drift_self.json`). **Re-run:**
`METS_DATA=<main>/data METS_PY=<conda mets python> tools/run/p1d_drift_batch.sh
<main>/data/p1d/drift_2026-09-25` (~9 min per
run, 5.7 h total, skips done runs), then
`python -m tools.run.p1d_drift <out>/stage0 <out>/pilot` (and `<out>/stage0_rerun`
for the self-control). ARI keeps HDBSCAN
noise as a label, as §3 does.

| partition (72 records) | moved | ARI mean | p5 | min |
|---|---|---|---|---|
| shipped HDBSCAN (`mcs=2`) | 13 | 0.950 | 0.601 | 0.240 |
| tuned HDBSCAN | 7 of 70 | 0.969 | 0.886 | 0.223 |
| graph modularity | 6 of 71 | 0.994 | 0.991 | 0.615 |
| k-means | 1 of 71 (the self-control's size) | 1.000 | 1.000 | 0.991 |
| agglomerative, spherical k-means, spectral, GMM | 0 | 1.000 | 1.000 | 1.000 |
| **consensus** | **1** | 0.989 | 1.000 | **0.193** |
| core / halo / contested grade, share of tokens agreeing | 7 | 0.990 | 0.984 | 0.642 |

(A count "of n" leaves out records where the family abstained on both sides.)

- **Every family picks the same parameters on both sweeps, in all 72
  records.** The tuning stage itself does not drift.
- **Tuning HDBSCAN makes it move less often, not less far.** Where the
  shipped partition moved (13 records), tuned HDBSCAN (mostly
  `min_cluster_size` 5) moved in 7 of 11, with p5 0.28 against the shipped
  partition's 0.29.
- **The consensus moves once in 72, against 13 for shipped HDBSCAN. That
  does not show 1d finds a reproducible structure.** The families that never
  move are either coarse (k-means, spherical k-means, spectral, GMM at
  k = 2–5) or agglomerative. Agglomerative is fine, not coarse: k median 162
  over 70 records, and k ≥ 60 in 55 of them, finer than any tuned HDBSCAN pick
  (corrected after `/challenge-pr` on #101, finding 1; the first version said
  every stable family was coarse). But in those 55 records, 30 % of tokens are
  singletons and 74 % of the multi-member clusters are one repeated token
  string (`tools/run/p1d_drift_checks.py --fine`). A partition that groups
  identical strings and leaves the rest alone is stable almost by
  construction. On `repeated_tokens`, where the drift is, every
  non-density family is coarse (k 2–4) or abstains, so scale and method are
  confounded there. A matched-k run on that prompt would separate them
  (Parked).
- **When the consensus moves, it moves a lot, and it sits on a knife edge.**
  At `repeated_tokens` step 32 L6 its ARI is 0.193 (shipped: 0.240): a
  70-token cluster merges into the big one (sizes 193/70/1/1 → 251/12/1/1,
  Mirkin cut 0.465 → 0.439). Swapping the pilot's version in one family at a
  time leaves the consensus identical for every family, HDBSCAN included.
  It takes three changes together: HDBSCAN's moved labels (ARI 0.22), its
  weight (0.388 → 0.471) and graph modularity's weight (0.794 → 0.801); any
  one of them alone changes nothing
  (`tools/run/p1d_drift_checks.py --swap 32 repeated_tokens 6`; corrected
  after finding 4, the first version credited HDBSCAN alone). The weights are
  raw subsample stability (`/challenge-pr` on #100, finding 1), so drift
  enters both through labels and through weights. The one record does not
  show that the weighting makes drift worse; it shows the cut can be crossed
  by small combined shifts.
- **The drift is one prompt.** 9 of the 13 moved records are
  `repeated_tokens`, and so is every consensus or grade move larger than 1d's
  own run-to-run noise. Three grade moves elsewhere (0.996–0.998 agreement,
  `sullivan_ballou` 143000 L12, `wiki_paragraph` 143000 L12, `camus_letranger`
  512 L6) are the size of the self-control's 0.998, which was at exactly
  `wiki_paragraph` 143000 L12. On the other 7 prompts the shipped partition moved
  in 4 of 63 records (ARI ≥ 0.925), and 1d's consensus never moved.
- **The §3 baseline, per prompt** (all 13 shared steps; §3's rule, label
  vectors not identical, over all 25 layers; producer
  `tools/run/p1d_drift_checks.py --baseline`, reading only the stored
  `hdbscan_labels.json`). This is the number the other files point to:

  | | not identical, of all layers | min ARI | p5 ARI, layers ≥ 1 |
  |---|---|---|---|
  | `repeated_tokens` | **309 of 325** | 0.166 | 0.243 |
  | other 7 prompts | 126 of 2 275 | **0.925** | 0.990–1.000 per prompt |
  | all 8 (§3's 2 600) | 435 of 2 600 | 0.166 | |

  Counting ARI < 1 at layers ≥ 1 instead gives 307 of 312 and 118 of 2 184,
  because two `repeated_tokens` pairs differ in label numbering only. The
  mechanism, tested since: float32 distances (next section, `--precision`).
- **1d does not reproduce itself exactly on the same input.** The
  `stage0_rerun/` control (6 records: `wiki_paragraph` 143000,
  `camus_letranger` 32) gave k-means ARI 0.991 once, grade agreement 0.998
  once, and everything else identical. Seeds are fixed per call, so this is
  float summation order, likely threaded k-means. Six records show that
  it exists, not how big it is.

**Answer, for this design.** On the 7 prompts without mass repeats, the
shipped partition's drift is already small (ARI ≥ 0.925), and 1d's
consensus does not move at all. On `repeated_tokens` the drift is large for
every density family, tuned or not, and the consensus carries it once in 9
records. Whether 1d "reduces the drift" can't be told apart from what its
stable families pick (coarse k, or identical strings), so it waits on D / C,
and on the matched-k check on `repeated_tokens`.

### Matched k on `repeated_tokens`, and the float32 defect it led to (2026-09-25)

**Question** (Parked above). On `repeated_tokens`, is the drift a property of
density methods or of the fine scale they pick? **Input:** the 9
`repeated_tokens` records of the drift run (steps 32 / 512 / 143000 × L6 / 12 /
18), both sweeps, stored activations (float32); control: the 9
`wiki_paragraph` records. Code at `c61147e` + this unit's producer, conda
`mets`, `hdbscan` 0.8.41. **Re-run** (~2 min each): `python -m
tools.run.p1d_drift_checks <out>/stage0 <out>/pilot --matched repeated_tokens`,
and `--precision repeated_tokens`.

**Answer: neither. The drift is float32 rounding in the distance step, upstream
of every method** (found by `/challenge-pr` on #102; the first version of this
section named a method-level mechanism, withdrawn). Phase 1's stored labels and
1d's `LayerData` compute cosine distance in float32
(`p1_mstate_tracking/clustering.py` `cluster_tokens`, `pairwise_distances` on
float32 rows; `methods.py` `LayerData.from_normed`). `1 - x·y` loses its
significant digits when `x·y` is near 1, which is where `repeated_tokens`'
tokens sit. Shipped HDBSCAN refit on float64 distances from the same stored
activations (`--precision`):

| step | L6 / 12 / 18: stored A~B | float64 A~B | float64 ~ stored (A) | k stored → float64 |
|---|---|---|---|---|
| 32 | 0.240 / 0.495 / 0.320 | 1 / 1 / 1 | 0.067 / 0.082 / 0.146 | 62→30, 48→27, 53→27 |
| 512 | 0.485 / 0.688 / 0.699 | 1 / 1 / 1 | 0.279 / 0.337 / 0.656 | 50→24, 46→26, 37→25 |
| 143000 | 0.708 / 0.914 / 0.979 | 1 / 1 / 0.982 | 0.728 / 0.772 / 0.778 | 42→39, 43→42, 40→46 |

`wiki_paragraph`: float64 and stored labels are identical (ARI 1 in all 9),
and both sweeps agree (1 in 8, 0.988 in 1, the same in both precisions).

- **The stored `repeated_tokens` partitions are mostly rounding.** At steps
  32 and 512 the float64 partition shares little with the stored one (ARI
  0.07–0.66) and has about half as many clusters. The other six v1 prompts are
  unchecked; `wiki_paragraph`'s stored labels are exact. (All 8 checked
  since: next section.)
- **Matched k (`--matched`) does not separate method from scale, as first
  read.** At Stage 0's HDBSCAN k, k-means and Ward give ARI 1 across sweeps in
  16 of 16 rows, average linkage in 9 (min 0.51), spherical k-means (5 inits)
  in 10 (min 0.62), all on 1d's float32-derived distances. But k-means on one
  sweep, seed 0 against seed 1, agrees at 0.54–1.00 on `repeated_tokens` and
  **0.38–0.75 on `wiki_paragraph`**: at k ≈ 10–60 the k-means partition is
  seed-dependent everywhere, and "never moves" was the fixed seed. Average
  linkage's movement is probably the same float32 distances; not re-run on
  float64.
- **Withdrawn:** that the prompt is "one point" at early steps and its
  clusters name no structure. Float64 still finds 24–30 clusters, and whether
  their spread counts as one point depends on the scale `δ`, which waits on
  β's convention (Blocked 9). The reviewer reports the stored clusters at 32
  / 512 are runs of consecutive positions; not checked here.

**What it changes.** Phase 10 §3's floor (ARI p5 0.347, almost all
`repeated_tokens`) is mostly this defect, not HDBSCAN's sensitivity (routed to
`status-10.md`). The fix (float64 distances in `clustering.py` and 1d's
`LayerData`) changes stored labels, so it is its own unit with a re-run of §3.

**Registry.** `P-C1`–`P-C4` (`predictions-1d.md`) were never registered and
cannot be scored blind on the v1 runs already examined. They return as tier 1,
or get registered fresh against the held-out prompts (the user's call).

### Float64 distances, and Phase 10 §3's floor re-run on them (2026-09-26)

**Fix.** One route for cosine distance, `core.metrics.cosine_distance_matrix`
(float64 rows, `1 - x·y`, clipped, symmetric, zero diagonal), used by
`clustering.py` `cluster_count_sweep` (HDBSCAN, agglomerative, and the k-means
silhouette, now on the precomputed matrix), 1d's `LayerData`,
`backfill_hdbscan.labels_for_activations`, `p1d_drift_checks --precision` and
`p10_hdbscan_planted`. `clustering.json` (per layer and in its HDBSCAN block)
and backfill records now say `distance_dtype: "float64"`;
`backfill_hdbscan.labels_distance_dtype(run_dir)` reads it and returns
`"float32"` for anything unrecorded. The backfill keeps the float32 route only
for `--verify-pilot`, which still replays 3 pilot directories 25/25 identical;
its record says it verified the toolchain on that route, not the float64 labels.
Not changed: k-means still fits on float32 rows; Gram-matrix readers
(`multiscale_nesting`) do not subtract from 1 and are not affected. **Stored
labels are not rewritten**: everything in `data/phase12` and the pilot is
still float32-derived.

**Input.** The 104 (step, prompt) directories both sweeps hold: 8 v1 prompts ×
13 steps, pilot `HDD_1TB/Mets_archive/2026-08-12_05-01-35` against Stage 0
(`data/phase12`), 2 600 layer-records, activations differing by ≤ 7.9e-5
(≤ 2.5e-7 up to step 1000). Code at this PR's head, conda `mets`, hdbscan
0.8.41. **Re-run:** `python tools/run/p10_partition_stability.py --v1-only
[--refit] --out …` (47 s stored, 2 min 24 s refit, 16 cores); records
`data/analysis/p10_partition_stability_{stored,f64}_2026-09-26.json`. The
stored run reproduces §3's numbers exactly. The last column below is the f64
record's per-layer `ari_b_stored_vs_refit` (side b = Stage 0), over L ≥ 1.

| over 2 600 layer-records | stored (float32) | refit (float64) |
|---|---|---|
| label vectors identical | 83.3 % | **99.1 %** (23 differ) |
| ARI mean / p5 / min | 0.933 / 0.347 / 0.166 | 0.9997 / **1.000** / 0.471 |
| ARI noise-dropped p5 / min | 0.585 / 0.327 | 1.000 / 0.477 |
| cluster-count \|Δ\| mean / max | 0.58 / 20 | 0.008 / 9 |

| prompt (325 records each; ARI over L ≥ 1) | stored: not identical, p5, min | float64: not identical, p5, min | Stage 0 stored ~ float64: mean, p5, min |
|---|---|---|---|
| camus_letranger | 10, 1.000, 0.941 | 3, 1.000, 0.988 | 1.000, 1.000, 0.941 |
| hdbscan_code | 4, 1.000, 0.978 | 2, 1.000, 0.979 | 1.000, 1.000, 0.999 |
| homer_iliad | 40, 0.990, 0.929 | 0 | 0.998, 0.990, 0.946 |
| latex_monograph | 25, 0.993, 0.925 | 0 | 0.999, 0.996, 0.920 |
| paper_excerpt | 1, 1.000, 0.961 | 0 | 1.000, 1.000, 0.961 |
| **repeated_tokens** | **309, 0.243, 0.166** | **15, 1.000, 0.471** | **0.230, −0.044, −0.064** |
| sullivan_ballou | 27, 0.994, 0.950 | 1, 1.000, 0.994 | 0.999, 1.000, 0.945 |
| wiki_paragraph | 19, 1.000, 0.957 | 2, 1.000, 0.988 | 1.000, 1.000, 0.969 |

- **The floor was the float32 defect.** On float64 distances the two sweeps
  agree at ARI p5 1.0. Of the 23 records that still differ, 10 are at step
  143000, where the activations themselves differ by 2e-5 to 8e-5 (an input
  difference, not rounding), and 15 are `repeated_tokens` (4 are both). One is far off:
  step 8, `repeated_tokens`, L4, ARI 0.471 (37 vs 46 clusters) on activations
  1.7e-7 apart. So HDBSCAN on `repeated_tokens` still has near-ties at
  float64, rarely. **Not float64 cancellation** (`/challenge-pr` on #103
  asked): on that record, `1 - x·y` in float64 agrees with ½‖x̂ − ŷ‖² from
  direct differences (`scipy` `pdist`) to ≤ 2.2e-15, relative ≤ 2e-8 at the
  smallest distance (1.7e-8); refit on the direct route, labels are identical
  (ARI 1) and A~B stays 0.471.
- **Stage 0's stored `repeated_tokens` partitions are rounding, at every step
  and layer ≥ 1**: ARI to their float64 refit averages 0.230, p5 −0.044.
  The other 7 prompts' stored labels are the float64 labels up to small
  moves (mean ≥ 0.998, min 0.920). The previous section checked 9
  `repeated_tokens` records; this is all 325, and the other six prompts.
- **What it changes.** Every Phase 10 reader that pools `repeated_tokens`
  stored labels includes one prompt in eight whose partition is noise. The
  readers are not re-run here (Phase 10 is on hold); the labels would first
  have to be re-derived at float64 (Parked below). 1d's own drift run
  (72 records, "Float-noise drift") used float32-derived `LayerData`
  distances and is not re-run either.
- **Caveats.** One pair of sweeps, one machine; no null (a floor, as §3).
  ARI treats noise as a cluster (the noise-dropped column agrees).

**Parked** (discoveries, not followed):
1. Re-derive stored labels at float64 (a new `hdbscan_f64.json` per
   directory, or `read_labels` refitting), then re-run Phase 10's
   label readers. Why: `repeated_tokens` rows are noise. Cost: minutes for
   labels, readers an unknown hour or so. Changes: whether any Phase 10 row
   that pools prompts moves; decide when Phase 10 resumes.
2. Re-run 1d's drift (72 records) on float64 `LayerData`. Why: its
   "families never move / HDBSCAN moves" reading was on float32 distances.
   Cost: ~3 h. Changes: whether tuning matters for stability at all, now
   that the input noise is gone.

### Merge tree over scales, and clusters linked across layers (2026-09-26)

**What.** Option D (`lit-1d.md` §5), the item the user ordered first from
1d's "Next": read each layer's full agglomerative hierarchy as cluster
count `k` against distance threshold `delta`; take as that layer's
partition the longest-lived plateau with at least two substantial
clusters (>= `SUBSTANTIAL_CLUSTER_SIZE` = 4 tokens; smaller clusters are
outliers, -1); link neighbouring layers' partitions by token containment
(overlap / smaller cluster's size >= 0.5), and classify every linked
group as stable / merge / split / tangle (`merge_tree.py`, new
sub-experiment F, no prerequisites, no tuning). **Not done:** marking C's
`delta = c*beta_eff^-1/2` on the curve (waits on Blocked 9).

**Revised before merge; the first version's headline is withdrawn.** The
first version took the longest-lived plateau with no size condition and
linked on Jaccard >= 0.1, and reported "splits almost absent, merges at
L0→L2". Both were the method: at layers >= 2 that plateau is one cluster
holding 91–96 % of tokens plus a few outliers, and Jaccard cannot register
a piece leaving a large cluster (1 token of 460 is 1/460), so it records a
birth where there is a split. The L0→L2 "merges" were the pick jumping
from the token-identity partition to the blob. `/challenge-pr` on #104
did not catch this; a second read of the saved outputs did (lesson 6).
Also fixed: identical L0 vectors merge at ~0, which gave `wiki_paragraph`
L0 247 zero-width plateaus of 467 whose `k` no cut reproduces; tied merges
are now applied together (`HEIGHT_TIE_TOL`, placed).

**Input.** All 8 v1 prompts, pythia-410m step143000, all 25 layers, Stage
0: `data/phase12/2026-09-22_21-20-26/` (5 prompts) and
`2026-09-22_21-41-17/` (`camus_letranger`, `hdbscan_code`,
`latex_monograph`; `--v1-only` drops the two held-out prompts there).
Average linkage, min size 4, containment >= 0.5, conda `mets`, code at
this PR's head. **Output:** `data/p1d/merge_tree_2026-09-26/`.
**Re-run** (~3 s each):

    for R in 2026-09-22_21-20-26 2026-09-22_21-41-17; do
      METS_DATA=<main>/data python -m p1d_cluster_ensemble.run_1d --v1-only \
        --results <main>/data/phase12/$R \
        --out <main>/data/p1d/merge_tree_2026-09-26 --subexp F
    done

**A defect found building this: `--subexp F` was not actually cheap.**
`process_layer` ran the full tuning grid (stage A) regardless of
`stages`, because nothing gated it — the only genuinely skippable stage
was B. A `--subexp F` smoke run on real data (L0/12/18) took 2m51s wall
/ 39 min CPU, confirming it was paying for the untuned grid every time.
Fixed: `process_layer` now returns before stage A when `"A" not in
stages`; the same command now runs in 0.9 s, and all 25 layers of 5
prompts in about 3 s. The smoke output taken before the fix was
discarded, not reported.

**What the longest-lived scale is.** "Blob" = the longest-lived
non-trivial plateau has one substantial cluster plus outliers. "Token 0
out" = token 0 is an outlier in the chosen partition (the >= 2-cluster
plateau). Counts over L1–L24.

| prompt | n | distinct L0 vectors (= L0 floor k) | blob layers | token 0 out | chosen partition, substantial clusters L3–24 |
|---|---|---|---|---|---|
| camus_letranger | 465 | 237 | 11/24 | 17/24 | 2–26 |
| hdbscan_code | 242 | 125 | 18/24 | 19/24 | 2–16 |
| homer_iliad | 512 | 273 | 18/24 | 18/24 | 2–4 |
| latex_monograph | 446 | 171 | 20/24 | 19/24 | 2–29 |
| paper_excerpt | 286 | 163 | 20/24 | 20/24 | 2–11 |
| repeated_tokens | 265 | 3 | 8/24 | 5/24 | 2–3 |
| sullivan_ballou | 482 | 232 | 18/24 | 19/24 | 2–25 |
| wiki_paragraph | 467 | 215 | 13/24 | 14/24 | 2–7 |

- **At most layers the longest-lived scale is one cluster plus
  outliers:** 126 of 192 layer-records. Token 0, the attention sink, is
  among the outliers of the chosen partition in 131 of 192.
- **The two-cluster floor moves the extreme; it does not remove it.** At
  L3–L24 the chosen partition has exactly 2 substantial clusters in 112
  of 176 records. Ranking by absolute lifetime still favours the coarsest
  admissible scale, so the scale question D was meant to settle is open.
- L0's floor is token identity in all 8 prompts (floor k = distinct
  vectors; `repeated_tokens` has 3), and is trivial by construction now.

**Merges and splits against depth** (containment >= 0.5; 191 boundaries,
`repeated_tokens` L0→L1 skipped because L0 has no >= 2-cluster plateau):

| band | boundaries | merge | split | tangle | birth | death | stable |
|---|---|---|---|---|---|---|---|
| L0–L7 | 63 | 33 | 21 | 5 | 48 | 28 | 270 |
| L8–L15 | 64 | 16 | 12 | 19 | 7 | 4 | 75 |
| L16–L23 | 64 | 12 | 12 | 23 | 4 | 3 | 61 |

On the same labels, Jaccard >= 0.1 reports 242 births and 353 deaths
against containment's 59 and 35: most of Jaccard's births and deaths are
pieces entering or leaving a larger cluster.

- Early layers have more clusters per partition (up to 30 substantial at
  L0–L2), so the L0–L7 band has more events of every kind; compare kinds
  within a band, not counts across bands.
- A tangle is several clusters on each side sharing tokens: a reshuffle,
  not a coarsening or a refinement.
- **Caveats.** One checkpoint, one linkage (average), min size 4 and
  containment 0.5 placed, tier 1, descriptive, **no null**: a
  size-preserving random relabelling would say which of these rates a
  structureless sequence of partitions also produces.

**Answer, for this design.** Depth does not cleanly separate merges from
splits. Merges outnumber splits only in L0–L7 (33 to 21), where
partitions are still fine; from L8 on they are even (28 to 24) and
tangles are the most common change (42 of 112 non-stable events). The
firmer results are structural: at most layers the longest-lived scale is
one cluster plus outliers, token 0 usually among them, and requiring two
substantial clusters mostly picks exactly two. Which scale a cluster
lives at is still undecided.

**Parked** (discoveries, not followed):
1. Mark C's `delta = c*beta_eff^-1/2` on the merge-tree curve. Why:
   waits on beta's convention (Blocked 9). Cost: cheap once decided.
   Changes: whether the theory's scale sits in a robust plateau.
2. A null for the link counts (random relabelling preserving cluster
   sizes at each layer). Why: tangle and split rates mean nothing
   without one. Cost: minutes. Changes: whether any depth pattern above
   is real.
3. Rank plateaus by something other than absolute lifetime (relative
   lifetime, or a balance condition). Why: the two-cluster pick is the
   new extreme. Cost: small. Changes: which partition F links.
4. Drop token 0 before building the tree. Why: it is an outlier in 131
   of 192 records; whether the blob remains without the sink is the
   question. Cost: minutes. Changes: the blob count.
5. Other checkpoints (steps 32, 512 have the most drift history).

## Deleted and restored (was `FROZEN.md`)

Code deleted 2026-09-23 in a branch cleanup that should have skipped it
(`LESSONS.md` lesson 12); the docs were kept in `archive/p1d_cluster_ensemble/`
from 2026-09-24 and moved back here on 2026-09-25. The rule then (user,
2026-09-24): code may go if its intent stays. `FROZEN.md` asked a rebuild to
"first measure whether tuning reduces *that* drift" (the float-noise one);
that is still the first question.

## As built in August

**State then:** all five sub-experiments implemented and validated on synthetic data with known
answers, with a driver (`run_1d.py`) and artifact IO (`p1d_io.py`) that have been run end to
end against a synthetic Phase-1 run directory. **Not yet run against Pythia artifacts** — no
result rows below, by design. Predictions P-C1, P-C2, P-C3 and P-C4 were written in the
branch's `PREDICTIONS.md` before this code existed (never entered in the registry).

**Cost:** [R] throughout. Reads `activations.npz` and re-clusters; no weights, no forward
pass. Runnable against any existing Phase 1 run directory today.

**Why it exists:** every cluster-conditioned result in this project rests on
`hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")` — a library minimum and a set of
library defaults. Phase 1's three other partitions are equally untuned, so the existing
cross-method agreement statistic compares four sets of defaults rather than four methods. See
`design-1d.md`.

## Implemented

| Sub-exp | Module | Cost | State |
|---|---|---|---|
| A — tune every family per layer | `selection.py` | [R] | implemented, validated |
| — method registry and grids | `methods.py` | — | implemented, validated |
| — Phase 1 constants, read not copied | `constants.py` | — | implemented, validated |
| B — consensus and calibrated confidence | `ensemble.py` | [R] | implemented, validated |
| C — shipped partition and its refusals (P-C2, P-C3) | `comparison.py` | [R] | implemented, validated |
| D — persistence prediction (P-C4) | `comparison.py` | [R] | implemented, validated |
| E — particle-table export | `p1d_io.py` | [R] | implemented, round-trips |
| — driver | `run_1d.py` | — | implemented, end-to-end tested |
| — artifact contract | `core/artifacts.py::PHASE1D` | — | registered, validated against real output |

Seven families: `hdbscan`, `kmeans`, `spherical_kmeans`, `agglomerative` (average / complete /
single at every one of Phase 1's 12 thresholds, plus Ward on k), `spectral`, `gmm`,
`graph_modularity`. The last two of those are implemented here — neither is in sklearn.

## Validation performed

**Every family recovers planted structure.** Three tight caps on $S^{d-1}$ ($d=24$, 15 tokens
each): every one of the seven families has a grid point reaching ARI 1.000 against the planted
labels, and the tuned selection admits all seven.

**The gate separates the three regimes.** Same protocol, 20 null draws, alpha 0.05:

| regime | families admitted |
|---|---|
| three planted caps | 7 of 7 |
| i.i.d. uniform on the sphere | 0–1 of 7 |
| collapsed (all tokens near one direction) | 0 of 7 |

The structureless row is a *rate*, not a bug: the gate is per (family, candidate) with no
multiplicity correction, so 7 families × 2 gated candidates at alpha 0.05 expects ~0.7 false
admissions. Two or more would mean the gate is not working, and the test asserts that bound
rather than perfection.

**Greedy modularity is exact where an exact answer exists.** Two disjoint 10-cliques give
$Q = 0.5$ exactly, split perfectly. On a planted 3-block model ($p_{\rm in}=0.6$,
$p_{\rm out}=0.05$, $n=60$) it recovers the blocks at ARI > 0.95 and within 5% of the planted
partition's own modularity — it is a greedy heuristic with no optimality guarantee, and the
test bounds the shortfall rather than demanding the planted partition be beaten. The modularity
of the returned partition is recomputed by an independent implementation of $Q$, which is the
only way to catch a merge-bookkeeping error that still returns plausible communities.

**AUC matches sklearn including on ties.** The pairwise-concordance AUC used for P-C4 agrees
with `roc_auc_score` to floating point on continuous scores *and* on an all-ties binary
predictor — the case that matters, since the binary clustered/noise flag is nearly all ties and
an implementation resolving them differently would change the comparison the phase exists to
make.

**ΔAUC discriminates in both directions.** A graded score correlated with the target beats an
uninformative binary flag (+0.31 AUC, bootstrap CI [0.21, 0.42]); the same predictor scored
against itself returns exactly 0.000 and the FALSIFIED verdict string.

**The cross-implementation duplication is asserted, not commented.** Co-association under both
noise policies, the singleton relabeling, and the Phase 1 agreement-layer set all agree with
`p1_visualization/cluster_methods.py`'s implementations wherever that package can be imported.
The KMeans trust-gate constants, `DISTANCE_THRESHOLDS` and `K_RANGE` are read out of source
with `ast` rather than copied.

**End to end.** A synthetic Phase 1 run directory (3 layers, 30 tokens, planted caps blurring
with depth, shipped HDBSCAN labels with injected refusals) produces all four verdicts, and
`p1d_results.json`, `p1d_ensemble.npz` and `particle_table.npz` all validate against their
registered `core.artifacts` specs.

## Findings from implementation, before any data

1. **An N-sigma gate on stability discards true structure, because the statistic is bounded.**
   k-means at k=3 on three cleanly planted caps scores stability 1.000 against a null of
   0.648 ± 0.201 — **1.75σ, which fails a 2σ gate**, while exceeding 19 of 20 null draws.
   Spectral clustering on the same data scores 1.00 with two null draws tied at 1.00: a rank
   test failure by ties alone. The decision is made on a rank test, and stability is a floor
   rather than a second significance test, for this reason. This is a real departure from the
   project's N-sigma convention and is flagged wherever the numbers are written.

2. **In the collapsed regime the silhouette is not merely uninformative — it is *worse* than
   its matched null, at a value the shipped trust gate would admit.** A cloud with every token
   near one direction gives k-means at k=2 a silhouette of **0.105**, above
   `cluster_methods.py`'s `KMEANS_SIL_MIN = 0.1`, while the matched null on the same cloud
   scores **0.126** (rank p = 0.95). The shipped gate would call that layer's KMeans k
   trustworthy. Whether this survives on real deep-layer Pythia activations is exactly what
   sub-experiment A measures, but the placed-threshold gate cannot detect the case even in
   principle, and that is a statement about the instrument rather than about the data.

3. **sklearn's HDBSCAN mutates a precomputed distance matrix unless told not to.** With
   `metric="precomputed"` and the default `copy=False`, the input is modified in place; the
   default is scheduled to flip in sklearn 1.10. This phase re-fits ~100 settings against one
   cached distance matrix per layer, so the failure would be silent, cumulative and
   grid-order-dependent. `copy=True` is passed explicitly and a test asserts the matrix is
   unchanged after a fit. **Phase 1 itself uses the `hdbscan` package, which is not affected**
   — but any code that switches backends is.

4. **The two HDBSCAN backends are not interchangeable for P-C2.** `hdbscan` and
   `sklearn.cluster.HDBSCAN` do not agree bit for bit. A P-C2 verdict computed against a
   different implementation than Phase 1 ran would carry an implementation difference inside a
   comparison of settings, so the backend is recorded in every artifact and `hdbscan` is
   preferred when both are installed. **This validation ran on the sklearn backend** (the
   `hdbscan` package is not installed in the environment used); a Pythia run must use the same
   backend Phase 1 used, and the artifact will say which it was.

5. **`n_null` and `alpha` are not independent knobs.** With $n_{\rm null}$ draws the smallest
   attainable p-value is $1/(n_{\rm null}+1)$, so `--n-null 20 --alpha 0.05` leaves exactly one
   passing value (p = 0.048) and any smaller alpha makes every outcome predetermined.
   `select_family` raises rather than running such a sweep. A multiplicity-corrected gate is
   available by passing a smaller alpha, and triples the required draws.

6. **Stability alone would admit i.i.d. points, and the trap is specific.** k-means at k=2 on
   uniform sphere points is highly reproducible — the split it finds is a real property of the
   sample, just not a cluster. This is why the calibration re-runs the *whole pipeline* on each
   null draw rather than scoring the real labels against a null.

## Open items

1. **Not run against Pythia artifacts.** Nothing below the fixture level has been measured. The
   first real run should be a single checkpoint at a moderate `--layer-stride` to establish the
   cost per layer before a sweep is scheduled.
2. **P-C1's scope depends on `clustering.json` being present.** The prediction is registered
   about Phase 1's own agreement layers; `p1d_io.phase1_agreement_layers` reconstructs that set,
   and where the file is absent the verdict falls back to all layers and says so in the verdict
   string. A run adjudicated on the fallback is a weaker test than the registered one.
3. **No figures.** `p1d_ensemble.npz` holds everything a figure needs (co-association matrices,
   per-particle arrays). A `visualization/` submodule matching the other phases' pattern is the
   obvious next step and is not part of this pass.
4. **The consensus is per (run, layer), not tracked across depth or checkpoints.** Consensus
   cluster ids are not aligned between layers — P-C4's persistence target is deliberately
   defined pairwise on co-membership, which needs no alignment. A cross-layer or
   cross-checkpoint chain, the analogue of Phase 1's cluster tracking, is not built.
5. **Promotion to `core/`.** If this phase survives its first run, `co_association`,
   `noise_as_singletons`, `consensus_strength` and the agreement-layer criterion should move to
   `core/` and both this phase and the visualization package should import them. Until then the
   equivalence tests are the mechanism keeping the two copies honest.
6. **`spherical_kmeans` and `graph_modularity` are new code with no external reference
   implementation.** Both are validated against known-answer synthetic cases (planted caps;
   two disjoint cliques at the analytic $Q = 0.5$), which constrains them but is not the same
   as agreeing with an established library.

## Falsification table

Empty by design — nothing has been run. The registered predictions and their falsifiers are in
`PREDICTIONS.md`; the adjudicators (`comparison.adjudicate_p_c1..p_c4`) write their verdicts
into `p1d_results.json` under `verdicts`, and every verdict string names its own prediction id
so a reader cannot mistake which claim was decided.
