<!-- p1d_cluster_ensemble/design-1d.md -->
# Phase 1d — DESIGN

**Revised 2026-09-30** after Blocked 10 (user: retire the graded readout). The August
design (a tuned seven-family ensemble whose vote grades every token core / halo /
contested) is at `git show 30ccbb4:p1d_cluster_ensemble/design-1d.md`. What survives
from it is in "Kept from August" below. Numbers live in `status-1d.md`; this file
says why the phase is built the way it is. Literature: `lit-1d.md` (§8 for this
revision). Tier 1 throughout: nothing here is registered.

## The question

Phase 10 is on hold until the project can say what a cluster is (user, 2026-09-25).
Every Phase 10 row reads one partition, `HDBSCAN(min_cluster_size=2)` on cosine
distance, chosen by nobody. 1d's job is to replace "whatever that call returns" with
a definition that states its null, its scale, and what it controls for, so that
Phase 10's rows can be re-read on it and their sensitivity to the choice measured.

## What the evidence allows (2026-09-25 to 09-30)

Every row is from `status-1d.md`; v1 = the 8 v1 prompts on 410m.

| piece | verdict | why |
|---|---|---|
| Graded vote (core / halo / contested) | **failed, retired** | dropping one family replaces the trained core set in 21–23 of 24 records under all 9 vote rules ("Vote rules") |
| Consensus partition | **failed** | its k moves 3–10 with the vote rule; `abstain_small` makes it k-means ("Vote rules") |
| Per-family stability ranking | picks extremes | k = 2 for centroid families, finest threshold for agglomerative ("First real run"); normalising by the null moves k to the other end (`lit-1d.md` row 1a) |
| Per-family null gate | held | the gate itself works once degenerate draws score the floor ("First real run") |
| Merge tree (option D) | held as a readout | the longest-lived scale is one blob + outliers in 126 of 192 records; the ≥ 2-cluster pick is Gaussian-typical ("Merge tree", "Gaussian null") |
| Merge-tree lifetime `mt_life` | **fails its control** | untrained step 0, deduped, beats the Gaussian in 59 of 168 records (calibration 1); unexplained (Parked 11) |
| Matched-covariance Gaussian null, deduped | held at v1 length | step 0 at its calibration on `ci2`, `nn1`, `hdb_k`, `hdb_noise`. Off nominal past ~1000 tokens (Parked: "drifts with n") |
| HDBSCAN groups vs that null (`hdb_k`) | **the cleanest local signal** | step143000 20 groups vs 6 (centred), 109 of 168 records past the tail, calibration 1, step 0 2 (calibration 3); at length 28–31 of 32 per band, calibration 0. *Weaker than it reads (2026-10-01, `/challenge-pr` on #122, `status-1d.md` "Admission"):* step 0 is above its null mean in 160 of 168 records (calibration 85) and passed the tail test only on a null twice as wide; ~40 % of the shipped groups counted are tie artefacts |
| Global 2-means excess (`ci2`) | late only | L17–24, 42 of 56 records (calibration 5); not readable at length (calibration fires 24 of 32) |
| Position | a confound, partly measured | 30 % of deduped nearest neighbours within 3 positions (8 % at step 0); no residual null keeps position (Parked) |
| Attention communities vs null B | weak, late | a trained excess only at L17–23 ("Attention communities") |
| Theory's scale `δ = cβ^{-1/2}` | waits | β's convention is Blocked 9 |

Two things follow. A cluster definition has to be **per group**, not per token
(the token grading is what failed), and it has to start from the readout that
passed the step-0 control, which today is HDBSCAN's group count and not the merge
tree's lifetime. The per-group statistic below has **not** been through that
control yet; the step-0 stop rule in "Outcomes" covers it.

## Decision: the graded readout is retired (Blocked 10, user, 2026-09-30)

Retired as 1d's product: `confidence`, `mean_recall` / `min_recall`,
`refusal_fraction`, the core / halo / contested trichotomy, and the consensus
partition as a cluster definition. With them go the three predictions that read
them, `P-C1` (consensus strength), `P-C3` (noise tokens above the confidence
threshold) and `P-C4` (graded vs binary on persistence): `predictions-1d.md`
addendum 2026-09-30. They were never registered and the v1 runs have been seen.

Not retired: the code. `ensemble.py` and `vote_rules.py` stay because the
`vote_rules` result is re-runnable only with them, and `run_1d.py` still writes the
columns (labelled tier 1). Removing them from `run_1d`'s output and from
`core.particles` is a cleanup unit, not this one (lesson 12: intent archived here).

The alternative the user did not take, a matched-scale vote, stays open in one
sentence: in a planted-caps toy matched scale passes, but the one near-matched slice
of real data still sways, so there was no evidence it would pass on Pythia.

## The proposed definition: a group that beats its covariance

**At layer ℓ of run r, a cluster is an HDBSCAN group of deduped tokens whose
per-group excess density exceeds what the tokens' own Gaussian produces anywhere.**

| choice | value | why, and what was rejected |
|---|---|---|
| tokens | first occurrence of each string (`--dedupe-strings`) | all-token results are token identity: step 0 beats the null harder than step143000. Later occurrences get label −1, "not tested"; a nearest-group assignment for them is a separate, flagged column if Phase 10 needs one. `repeated_tokens` (3 strings) drops out, so the batch is 7 prompts |
| frame | centred (shared mean direction projected out, renormed); raw reported beside | raw cosine is dominated by the mean direction and rogue coordinates (`lit-1d.md` §7 row 3). **Chosen after seeing the data:** centred is where `hdb_k`'s excess was largest (20 vs 6 groups; raw 5 vs 3.2), so raw is reported with equal weight |
| algorithm | HDBSCAN, float64 cosine, `min_cluster_size` 2, EOM selection, **on the level-set tree**: every mutual-reachability edge of one weight merged at once (`admit.level_set_hdbscan`). The shipped call's labels, tie artefacts and ARI to these are written beside every record | *Changed while building (2026-10-01, `status-1d.md` "Admission"; was: the shipped call, so Phase 10's comparison is direct).* hdbscan's binary tree orders tied edges by processing order, and with mutual reachability ties are the rule, so its groups and their `S_C` depend on unrelated rows: the two-group invariance check failed on it (3.6 %), and it glued a stray row to a planted cap. Handed hdbscan's own tree, the level-set code reproduces hdbscan exactly, so tie merging is the only difference. Phase 10's comparison becomes shipped vs level-set labels, measured per record. `min_cluster_size` 4 (`SUBSTANTIAL_CLUSTER_SIZE`) is the sensitivity arm |
| per-group statistic | **`S_C / |C|`**: the group's stability (Σ over members of λ_p − λ_birth, λ = 1 / mutual-reachability distance) over its size, from the level-set tree; sensitivity arm: log-lifetime `log(λ_death / λ_birth)` | depends only on the group's own branch of the tree. **Rejected: `cluster_persistence_`** (`/challenge-pr` on #121, finding 1, verified): hdbscan divides `S_C / |C|` by the tree's largest λ, set by the tightest group anywhere in the layer. A planted 6-point group scored 0.75 alone, 0.13 and 0.012 when an unrelated tighter group was added; since trained tokens have far closer nearest neighbours than their null, real groups would be scored down and the null would win. Also rejected: merge-tree lifetime (fails step 0) |
| null | matched-covariance Gaussian in the same frame, same n, renormed (`gaussian_null.py`), 200 draws | the null 1d has calibrated; SigClust's (`lit-1d.md` §7) |
| threshold | the 95th percentile, over draws, of **each draw's maximum** of the statistic | a max statistic: under the null, P(any group admitted) ≤ 0.05 per record, whatever the group count. Rejected: a per-group percentile (`lit-1d.md` §8 row 1), which admits ~5 % of the dozens of groups a Gaussian draw makes |
| counts | admitted groups per record, against the same rule run on `--calibrate` inputs (each layer replaced by a draw of its own Gaussian) | the null is not at nominal level everywhere; 1d reads every result against a calibration on its own inputs (`gaussian_null_report.py` refuses any other) |
| labels | released per (step, frame, band L1–8 / 9–16 / 17–24) only if that band's calibration admits in ≤ 10 % of its records (2α, placed); otherwise the band's labels are withheld and say why | Phase 10 needs labels, not counts. A band whose null over-admits gives labels with an unknown error rate: refuse rather than degrade |
| control | step 0, same prompts and layers | an untrained model has token identity (removed by dedup) and position through the causal mask, and no learned content |
| scope | v1 length (242–482 tokens) | the deduped null drifts with n past ~1000 tokens; long prompts wait on that check |

**Why HDBSCAN and not a new method.** The aim is a definition, not a better
clusterer. Keeping the shipped algorithm and adding a null makes the change to
Phase 10 one thing (a group must beat its covariance) instead of several, and
HDBSCAN's group count is the readout that has already passed the controls.

**Why a max statistic, and what it costs.** It controls the per-record error
without a top-down stop, which matters here: SHC (`lit-1d.md` §8 row 2) stops at a
Gaussian-typical root, and at L1–16 the root is Gaussian-typical while the local
groups are not. The cost is power: the densest Gaussian group in each draw (likely
a close pair) sets the bar for every real group, of any size. Size bands (a max
per band) are the first thing to try if larger groups never pass.

**Placed, not calibrated:** α = 0.05 per record; 200 draws; `min_cluster_size`
2 / 4; the 2α label-release bound; the 1000-token scope edge. Each is written
into the artifact.

**Not yet measured:** how far real layers sit above their null draws in plain
nearest-neighbour density is known (`nn1`, every record); how far the per-group
statistic sits is not. `S_C / |C|` is in units of 1/distance, so it rewards tight
groups, including pairs; whether it admits mostly pairs is the first thing the
first record's output should show.

## Checks reported beside each admitted group (no votes)

| check | what it answers | built? |
|---|---|---|
| Position: span, share of member pairs within 3 positions, contiguous-run flag, each against random groups of the same size drawn from the kept positions | is the group a stretch of text rather than a content group. Kept first occurrences sit early in the prompt, so the baseline is the kept positions, not all positions | the nearest-neighbour version is in `gaussian_null`; per group, no |
| Subsample stability, cluster-wise: mean best-match Jaccard over 80 % subsamples (Hennig 2007) | does the group come back | `selection.py` has partition-level stability only |
| Recovery by the tuned families at the group's own k | is it an HDBSCAN artefact | families yes; matching per group no |
| Cross-layer persistence: containment links (`merge_tree.py`) | does it last over a window of layers (the theory's persistence reading) | linker yes |
| Attention community overlap, read against null B | does the model's attention treat it as a unit | null B yes (`attention_null.py`) |
| Theory scale: does `δ = cβ^{-1/2}` fall inside the group's height interval | is the group at the scale the theory names | waits on Blocked 9 |

A group that fails a check is still admitted; the check is reported with it. Making
any check a gate is a later decision, made on its own numbers.

## Outcomes, written before the run

From `hdb_k`'s excess, expected (not registered): admitted groups at step143000
in most records at L9–24, few at L1–8; step 0 at its calibration; a large share of
admitted groups positional. The outcomes that would change the plan:

| outcome | reading | consequence |
|---|---|---|
| step143000 admits no more than its calibration | HDBSCAN's group-count excess is many weak groups, none individually beyond the Gaussian | try size bands; if still nothing, at v1 length a cluster is not distinguishable from its covariance group by group, and Phase 10 has nothing to condition on |
| step 0 admits beyond its calibration | the control fails, as `mt_life` did | stop; find why (position?) before any trained reading |
| admitted groups are mostly positional | the groups are text stretches | a residual null that keeps position (Parked, ~7 h) before calling anything content |
| admitted, non-positional, recovered by other families | a cluster in the sense Phase 10 needs | Phase 10 re-reads its rows on the admitted labels (the user decides when Phase 10 resumes) |

## Build order

Each is its own unit and PR.

1. **Admission** (`p1d_cluster_ensemble/admit.py`, new sub-experiment; **built and
   run 2026-10-01**, `status-1d.md` "Admission"). Synthetic
   first: planted caps in a Gaussian background are admitted; a pure Gaussian input
   admits in ≤ 5 % of records; **a looser planted group's statistic and verdict do
   not change when an unrelated tighter group is added** (the defect that ruled out
   `cluster_persistence_`). Then 7 v1 prompts × step143000 / step 0 × L1–24, real
   and `--calibrate`, centred and raw, `min_cluster_size` 2 and 4. Open the first
   record's output before the batch. Cost: the deduped Gaussian null took 6 + 4 min
   at 14 workers, so under an hour.
2. **Checks** on the admitted groups, the table above minus the theory scale.
3. **The theory scale**, when Blocked 9 is decided.
4. **Phase 10 re-read** on the admitted labels: the user's call, since Phase 10 is on hold.

Prerequisites not on this list, Parked in `status-1d.md`: the Gaussian null's drift
with n (needed before long prompts), Parked 11 (step 0's lifetime excess; needed
before the merge tree's lifetime is read again), a position-keeping residual null.

## Kept from August: the tuned families and their gate

Still in the code, still used for `P-C2` (is `min_cluster_size=2` the
stability-optimal HDBSCAN setting; descriptive, unregistered) and for the recovery
check above. The August text (git pointer at the top) has the full argument.

**Seven families, one bias each:** `hdbscan` (density, can refuse), `kmeans`
(Euclidean centroids), `spherical_kmeans` (cosine centroids; implemented here),
`agglomerative` (linkage at a distance), `spectral` (graph cut), `gmm` (likelihood),
`graph_modularity` (Clauset–Newman–Moore on a mutual-kNN cosine graph; mutual so an
isolated token can stay a singleton). UMAP-then-cluster is excluded: it would vote
twice for what density methods already say.

**Tuning is not on an internal index.** Every within/between ratio has a good split
at every k on a collapsed cloud. Instead, two statistics against the
shuffled-dimension null with the whole pipeline re-run on each draw: **stability**
(mean ARI of two independent 80 % subsamples on their overlap) as a floor and the
ranking, and **separation** (cosine silhouette) as the significance test. The gate is
asymmetric because stability's null piles up at the ceiling. Decided on the rank p
`(1 + #{null ≥ obs}) / (n_null + 1)`, not on z, since both statistics are bounded and
z is compressed at the bound; `select_family` refuses an alpha below `1/(n_null+1)`.
Two stages (rank on stability, gate the top `top_m`), stated as an approximation; no
multiplicity correction, stated. A degenerate null draw scores the floor; a partition
over 50 % singletons is trivial (A0).

**What changed since August:** ranking on raw stability picks each family's extreme
scale, so a family's selected setting is not a scale claim (table above), and the
shuffled-dimension null is beaten by position and token identity alone (#106, #108),
so a passed gate is not evidence of content. The gate is a filter on the families,
not a cluster definition.

## Duplication, deliberately incurred

`co_association`, `noise_as_singletons`, `consensus_strength` and the agreement-layer
criterion are also in `p1_mstate_tracking/visualization/cluster_methods.py`.
`tests/test_phase1d_ensemble.py` asserts the two agree; constants are read from that
module's source with `ast`. With the vote retired, promoting these to `core/` is no
longer planned; the copies go with the cleanup unit.

## What this phase does not do

- **It does not re-run Phase 1 or rewrite stored labels.** Admission writes its own
  labels beside the shipped ones.
- **It does not re-read Phase 10.** That is build step 4, and the user's call.
- **It does not register anything.** The v1 runs have been seen; a registered
  version would need runs nobody has looked at (the 12 held-out prompts are the
  user's to release).
- **It does not make a check a gate** until that check has its own numbers.
