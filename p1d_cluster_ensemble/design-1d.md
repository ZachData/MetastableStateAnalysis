<!-- p1d_cluster_ensemble/design-1d.md -->
# Phase 1d — DESIGN

Why this phase is built the way it is. For its current state and what has been validated, see
`status-1d.md`; for the registered predictions, `PREDICTIONS.md` (P-C1..P-C4).

## The problem

Every cluster-conditioned result in this project rests on one line:

```python
hdb = hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")
```

`p1_mstate_tracking/clustering.py`. `min_cluster_size=2` is the library minimum, everything
else is a library default, and nobody chose either — they are what you get when you call the
constructor. Cluster tracking, cluster orthogonality, the merge counts, Phase 5's cluster
selection, Phase 5c's entire object of study (the tokens this call labels `-1`), and Phase 6's
`labels >= 0` masks all inherit that one setting.

Phase 1 does compute three other partitions, and `p1_visualization/cluster_methods.py` already
measures whether they agree. But those three are *also* untuned: KMeans at a
silhouette-selected k, average linkage at whichever threshold happens to sit in the middle of
the sweep, and the sign of the Fiedler vector. So the existing agreement statistic compares
four sets of library defaults. It is a real measurement — four different inductive biases
landing in the same place is evidence — but it is not the measurement people read it as, which
is "the methods agree, therefore the clusters are real".

Two distinct questions follow, and the phase is built to keep them separate:

1. **Is the shipped setting the right one?** A tuning question about HDBSCAN. (P-C2)
2. **Does a conglomeration of tuned methods say something the categorical label cannot?**
   A question about what kind of object a cluster annotation should be. (P-C1, P-C3, P-C4)

## Why a separate directory

The same reason `p1c_frames` is one: a unit of work that is the whole run at once rather than
one forward pass, its own falsification structure, and no forward passes. It is [R] throughout
— it reads `activations.npz` and re-clusters. It could have been a module inside
`p1_mstate_tracking`, but the sweep is expensive enough to be scheduled separately from
anything, and folding it into `analysis_p1`'s per-layer loop would make every future Phase 1
run pay for it.

It is deliberately **not** in the visualization package, even though the nearest existing code
lives there. `cluster_methods.py` reads what Phase 1 wrote; this phase re-runs the clustering.
Those are different cost classes and different kinds of claim.

## Seven families, and why each earns a vote

The consensus is only as good as the diversity of biases entering it. Six centroid methods
would produce a consensus about centroids. The registry is chosen to span:

| family | bias | what it can see that the others cannot |
|---|---|---|
| `hdbscan` | density, with refusal | variable-density clusters; can decline to assign |
| `kmeans` | Euclidean centroids at fixed k | the shipped comparison arm |
| `spherical_kmeans` | cosine centroids at fixed k | the k-means whose objective is the geometry we actually use |
| `agglomerative` | linkage at a distance | scale structure — a plateau across thresholds is a claim HDBSCAN structurally cannot make |
| `spectral` | graph cut on an affinity | clusters that are connected but not compact |
| `gmm` | a likelihood | shape and overlap; the only member scored without a distance |
| `graph_modularity` | community structure | no centre, no radius, no k — the strongest check that clusters are not just blobs |

`spherical_kmeans` and `graph_modularity` are implemented here (neither is in sklearn). The
first is a 40-line fix to a real mismatch: Phase 1 runs Euclidean k-means on L2-normed rows,
where the *assignment* step is equivalent to cosine but the centroid update leaves the sphere.
The second is Clauset-Newman-Moore greedy modularity on a mutual-kNN cosine graph. Mutual
rather than plain kNN because plain kNN forces every token to have degree ≥ k, handing an
isolated token a community regardless of geometry — the graph family needs to be able to make
HDBSCAN's refusal in its own idiom, and a singleton community is how it does.

**Deliberately absent: UMAP-then-cluster.** Available (Phase 1 already optionally imports
`umap-learn`) and excluded, because clustering a 2-D embedding measures the embedding's
inductive bias — neighbour-preserving by construction — and would enter the consensus as a
second vote for whatever the density methods already say.

## What "tuned" means, and why not an internal index

Every standard way to pick a clustering hyperparameter fails on this data, and the project
already knows why. From `cluster_methods.py`:

> K_RANGE starts at 2, so best_k=2 is a floor, not a finding. In the collapsed regime all
> tokens are near-collinear and any 2-way split scores a silhouette of ~0.1-0.3 from geometry
> alone.

That is not specific to silhouette. Every internal index is a ratio of within- to
between-cluster spread, and a collapsed cloud has a perfectly good best split at every k.
Tuning on one would produce methods tuned to the collapse.

Two statistics are used instead, both calibrated against `core.nulls.shuffled_dimension_null`
— same per-dimension marginals, same token count, cross-token geometry destroyed,
re-normalized onto the sphere, **whole pipeline re-run on it**, distance matrix and fit
included:

- **Stability**: mean ARI between two independent 80% subsamples on their overlap
  (Ben-Hur/Elisseeff/Guyon). Two independent draws rather than subsample-versus-full, because
  comparing against the full-set partition rewards a method for being insensitive — the
  full-set partition is one of the two every time.
- **Separation**: silhouette on the same cosine distances. The index just called unusable,
  which it is *against an absolute threshold*. Against a matched baseline it becomes exactly
  the construction UPDATE_PLAN.md §5.7 already forced on $Q_k$: "adjudicated on the ratio to a
  matched random baseline", because $E[Q_k] = 1/n$ makes every large-$n$ configuration look
  like a spherical design under a fixed cutoff. Same disease, same cure.

### The gate is asymmetric, and that is the arguable decision

Separation is the significance test. Stability is a floor (may not be worse than the null's
mean) and the ranking criterion. They are not symmetric because **stability is bounded and its
null saturates**: on three cleanly planted caps, spectral clustering scores a perfect 1.00
while some structureless draws also score 1.00, which is a rank-test failure by ties alone for
a partition that recovers the planted structure exactly. Requiring significance from a
statistic whose null piles up on the ceiling discards true structure — and the expensive error
here is the false negative, because an abstaining family removes an entire inductive bias from
the consensus.

Each still catches what the other cannot: stability alone admits structureless data (k-means at
k=2 on i.i.d. sphere points is highly reproducible — the split is a real property of the
sample, just not a cluster); separation alone admits a partition nobody could reproduce (a
linkage peeling different outliers off each subsample scores a fine silhouette on what it did
assign).

### Rank test, not N-sigma

The decision is made on $p = (1 + \#\{\text{null} \ge \text{observed}\}) / (n_{\rm null} + 1)$,
not on the N-sigma summary this project reports elsewhere. Both statistics are bounded above,
and on data with real structure the observation sits at the bound, where a z-score is
compressed by the null's spread: measured, not hypothesised — k-means at k=3 on three planted
caps scores 1.00 against a null mean of 0.65, which is 1.75σ while exceeding 19 of 20 null
draws. `z_score` is still computed and written into every artifact so these numbers stay
readable next to the project's other null comparisons; it is just not what decides.

The consequence is a hard constraint the driver enforces rather than absorbs: with
$n_{\rm null}$ draws the smallest attainable p is $1/(n_{\rm null}+1)$, so an alpha below that
makes every outcome predetermined. `select_family` raises instead of running such a sweep.

### Two stages, and what that approximation costs

Computing nulls for every grid point costs `n_grid x n_null x n_repeats` fits. Stage 1 ranks
the grid by stability alone; stage 2 computes the gates for the top `top_m` only. A candidate
ranked 4th that would have passed while the top 3 fail is never examined. That is a real
approximation, and `top_m` is written into the artifact rather than the search being described
as exhaustive.

### No multiplicity correction, stated rather than fixed

The gate is applied per (family, candidate, layer) and is not corrected for multiplicity. At
alpha = 0.05 with 7 families and `top_m` gated candidates each, roughly `0.05 * 7 * top_m`
false admissions per layer are expected, and the validation run shows exactly that rate. A
Bonferroni correction is available by passing a smaller `--alpha`, but it is not the default,
because at alpha/3 the required `n_null` triples and the sweep is already the expensive part.
What protects the reading instead: a lone family's vote is weighted by its stability, and the
abstention pattern is reported next to every consensus statistic. **A single family clearing
the gate at one layer is not evidence of structure, and no verdict in this phase treats it as
such.**

## The ensemble

### One vote per family

Registered as an adjudication constraint (PREDICTIONS.md, Phase 1d constraint 1) rather than
left as an implementation detail, because six agglomerative linkages voting against one
HDBSCAN is a rigged consensus and which families are included must not be a post-hoc choice.

### Weighted, and abstaining

Families vote in proportion to the reproducibility of their selected setting — one that cleared
the floor at 0.3 and one that reproduces at 0.95 are not the same evidence — and a family that
failed the gate at a layer does not vote there at all. The number that did vote is carried
alongside every consensus statistic: an agreement among two families is not the same
measurement as an agreement among seven.

### The consensus partition has no k

Average linkage on $(1-C)$, cut at the height minimizing $\sum_{i<j} (C_{ij} - 1[i \sim j])^2$
— the Mirkin/consensus objective against a soft target. Every merge height is a candidate, so
the number of clusters is *derived*. A consensus whose k came from the same family of
assumptions its members made would not be a consensus.

### What the co-association matrix is not

It is not a ground truth. It is an aggregation of biases: if five of seven families assume
clusters are blobs, the consensus finds blobs. What it buys is that no *single* method's bias
can be blamed for a structure that survives it. Every statement this phase makes about $C$ is
an aggregation statement; the only claims calibrated against a null are the per-particle ones.

## The graded annotation — the actual product

The categorical label answers "is this token in a cluster" with a bit. The phase replaces it
with four numbers per particle, exported into `core.particles.ParticleTable`:

- `confidence` — mean co-association with its own consensus cluster minus the best mean with
  any other. A silhouette in *co-association* space, so the units are "fraction of the weighted
  method vote". Near 0 means the methods are split about where this particle goes, which is a
  statement the categorical label cannot make at all.
- `mean_recall` / `min_recall` — the disagreement structure confidence compresses. A low min
  with a high mean says one family specifically dissents; that is a different situation from
  every family being half-right.
- `refusal_fraction` — the fraction of families leaving the particle outside substantial
  structure, where "refused" and "placed in a cluster of two" count the same. This is what
  makes "unclustered" comparable across families that have no noise label; without it every
  non-density method looks like it placed 100% of tokens in structure by construction.

The trichotomy `core` / `halo` / `contested` comes from percentiles of the confidence
distribution measured on matched-null draws of the *same ensemble at the same settings*. The
percentiles (95th, 50th) are placed and labelled as such; the values they produce are
calibrated. **`halo` is the population the binary split has nowhere to put** — particles the
methods mostly agree about, but not at a level structureless data could not reach.

For Phase 5c, whose object of study is the unclustered population, this turns the selector from
`cluster_label < 0` into `population="contested"` or a threshold on a continuous column.

## Why P-C4 is the phase's own falsification

Everything above is descriptive. A tuned method is not a result; a prettier annotation is not a
result. The claim that earns the phase is that the graded annotation carries information the
categorical one does not — and the only honest test is to have both predict the same held-out
thing. That thing is layer-to-layer consensus persistence, computed from the consensus
partition at both layers and never from HDBSCAN, because scoring a graded annotation against a
target one of the two predictors defined would rig the comparison.

ΔAUC is read two ways, and the verdict refuses when they disagree — the same discipline
UPDATE_PLAN.md §5.2 forced on $T_{\rm eff}$, where three definitions of a step size straddled
the threshold. The registered instrument is a paired sign-flip permutation test on the per-pair
concordance differences; the pairs share particles, so its p-value is approximate and
anti-conservative, which is stated in the artifact next to the number rather than hidden. A
particle-level bootstrap respects that dependence and is the conservative reading.

If ΔAUC lands inside the null band, the correct write-up is "the ensemble adds nothing
measurable", and that sentence is easier to write with the prediction already on record.

## Duplication, deliberately incurred

`co_association`, `noise_as_singletons`, `consensus_strength` and the Phase 1 agreement-layer
criterion all exist in `p1_visualization/cluster_methods.py`. They are re-implemented here
because that module lives inside the visualization package, whose `__init__` imports the whole
figure pipeline, and this phase must stay importable in a numpy/scipy/sklearn environment.

The duplication is not left to a comment. `tests/test_phase1d_ensemble.py` asserts the two
implementations agree — co-association under both noise policies, the singleton relabeling, and
the agreement-layer set against `cluster_count_table` — skipping only where the visualization
package cannot be imported. The KMeans trust-gate constants are not copied at all: they are
read out of that module's source with `ast`, the same mechanism `checkpoint_scalars.py` uses
for `ENERGY_VIOLATION_REL_TOL`, so a rename raises at import instead of leaving a stale
literal working. `DISTANCE_THRESHOLDS` and `K_RANGE` are read out of `core/config.py` the same
way, since importing it would drag in torch and transformers.

**If this phase outlives its first run, all of it belongs in `core/`** — the same argument that
produced `core/population.py` after five call sites independently wrote `labels >= 0`.

## Cost

The sweep is quadratic in the wrong places: `n_grid x (1 + 2*n_repeats)` fits per family per
layer, plus `top_m x n_null x (1 + 2*n_null_repeats)` for the gates. On a 24-layer run with a
few hundred tokens the full grid is hours, not minutes. The knobs that move it, in order:
`--layer-stride`, `--grid quick`, `--n-null`, `--top-m`. All are written into the artifact,
because a selection made under `--grid quick` is a different claim from one made over the full
grid.

## What this phase deliberately does not do

- **It does not re-run Phase 1.** Nothing it produces overwrites a Phase 1 artifact, and
  `cluster_label` in its particle table is the *consensus* label with HDBSCAN's carried
  alongside as `extra__hdbscan_label`. A consumer that wants the shipped partition must ask for
  it by name.
- **It does not retro-fit downstream results.** If P-C2 confirms, the consequence is that
  every cluster-conditioned number was computed on a partition nobody chose — deciding what to
  do about that is a scheduling question for whoever reads the verdict, not something this
  phase acts on.
- **It does not adjudicate inside the loop that computes its inputs.** The adjudicators take
  already-computed per-layer results, so the compute step can be rerun without re-deciding
  anything.
- **It does not add figures.** The arrays a figure would need are persisted
  (`p1d_ensemble.npz`); a `visualization/` submodule is the obvious next step and is not part
  of this pass.
