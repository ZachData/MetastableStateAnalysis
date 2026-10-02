<!-- p1d_cluster_ensemble/design-1d.md -->
# Phase 1d — DESIGN

**Revised 2026-10-02** after Blocked 11⁗ (user: stop modelling nulls; intervene, use
the architecture as the null, one scale axis): section "The programme" below fixes the
rules for units 1–4 before any of them runs; literature `lit-1d.md` §10. The
per-group admission against a Gaussian ("The proposed definition") is no longer the
route to a definition; it stays as a reported column.
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

## The programme (Blocked 11⁗; rules fixed 2026-10-02, before any run)

**Why.** Four constructed nulls (Gaussian, cut at 8 / 32, two position nulls) each
removed one non-learned source and met the next (`status-1d.md` "Blocked 11⁗
decided"). Two questions were being asked of one null; the programme separates them:

| question | answered by | not by |
|---|---|---|
| **is it learned?** | unit 2: the same statistic on the same prompt through many random inits of the same architecture | a model of the residual (the four nulls) |
| **is it content, not position?** | unit 1: move the passage and see whether the group moves with it | a position-keeping null |
| **is it a cluster, and at what scale?** | unit 3: one family on a continuous scale, robust plateaus, subsampling stability | seven families voting at seven scales |
| **can the tool see what it should?** | unit 4: inputs with known answers | — |

Inputs throughout: the 7 deduped v1 prompts on 410m (not `repeated_tokens`; the 12 v2
prompts stay held out), step 0 and step143000, L1–24, both frames (centred primary, raw
beside), `min_cluster_size` 2 primary and 4 as the arm wherever HDBSCAN is used. Every
number below marked **placed** is chosen, not derived, and is written into the artifact.

### Token rules, once, for every unit and every method

| rule | value | why (`lit-1d.md` §10) |
|---|---|---|
| T1. position 0 | **excluded from every cloud in every condition** (step 0, trained, inits, seeds, every preamble length), before the frame, centring, any null fit, any graph; its norm ratio and its nearest token are written beside each record | the sink is a position, not a token (row 3); its massive activation is 20–50× the median at L8–20 in 7 of 7 trained prompts (row 4a) and is most of the position null's noise (`status-1d.md` "Position-keeping null"). Excluded at step 0 too (1.3×), so every comparison is on one token set |
| T2. other massive tokens | a token whose norm exceeds **10×** its layer's median at any of L2–20 in **any** run being compared is excluded from **all** of them, and listed (position, string, maximum ratio) | the first `\n` is a second massive token in 2 of 7 trained prompts, 18–41×; the bulk is ≤ ~3× and step 0 ≤ 1.3× (row 4a). Taking the union over runs keeps one token set per prompt. 10× is **placed** in the gap |
| T3. duplicates | first occurrence of each string, as admission (`--dedupe-strings`) | unchanged: token identity |
| T4. attention-based graphs | the columns of T1–T2 tokens dropped and rows renormalised | `lit-1d.md` §2.1; repo practice (`core/sink_audit.py`) |
| T5. where a rule cannot apply | the record refuses and says why; nothing is read on a cloud that still holds a T1–T2 token | refuse rather than degrade |

T1 changes step 0's opening group (under the smooth null, all 97 admitted centred
step-0 groups held position 0). That is intended: what is left of the opening without the sink is what units 1
and 2 test.

### Unit 1: move the text

| choice | value | why, and what was rejected |
|---|---|---|
| passage | each of the 7 v1 texts, unchanged | the token sets already read |
| preamble | the first `P` tokens of the **continuation** (text after the v1 passage) of the first three of `wiki_paragraph`, `sullivan_ballou`, `hdbscan_code`, `latex_monograph`'s long prompts in that order, skipping the passage's own (`p1d_cluster_ensemble/long_prompts/`) | committed, rule-fixed text (`long_prompts.py`), unrelated to the passage, long enough for 1000 tokens; three preambles so no one preamble's content is the result. Rejected: a repeated filler token (atypical; the sink can vanish on repeated tokens, `lit-1d.md` §10 row 3) |
| P | 0, 50, 300, 1000 tokens | from the 11⁗ table; ≤ 1512 tokens with the passage, inside 2048 |
| join | **`<|endoftext|>`** (primary), `\n\n` (arm) | Pythia trains on packed documents joined by EOD with attention across them (row 8), so the passage after EOD is in distribution. The arm asks whether one context changes it |
| cloud | the passage's tokens only, indexed by passage offset; T1–T3 on the **whole** sequence's positions (the passage's first token is not excluded when P > 0) | the same passage tokens at every P. A whole-sequence cloud (preamble + passage) is reported beside, to see where the opening went |
| clustering | admission's level-set HDBSCAN groups (`admit.layer_groups`) on each cloud; no null draws | this unit asks whether groups move, not whether they are significant (unit 2) |
| readout per (passage, layer, frame) | (a) for each group at P = 0, its best-Jaccard group at each P > 0 and preamble; (b) per passage token, the cosine of its state at P against P = 0 (SHAPE's readout); (c) in the whole-sequence cloud, the group holding the earliest kept position | (a) is the question; (b) separates "the group moved" from "every state moved"; (c) is the first check |
| moves with the text | Jaccard ≥ 0.5 at every P > 0 in ≥ 2 of 3 preambles, primary join | **placed** (0.5 as the identity control's recall; 2 of 3 as a majority) |
| position-bound | a group at P = 0 holding passage offsets < 8 that fails "moves", while (c) finds a group at the preamble's opening | the opening reading of #123 |

**First check (step 0, before any trained cell is read).** Step 0's opening group
(passage offsets < 8, after T1) is position-bound in most passages, and (c) finds a
group at the preamble's start. **If step 0's opening moves with the text instead**, the
opening is not position: stop and report before reading step143000.

**Outcomes** (not registered):

| outcome | reading | consequence |
|---|---|---|
| step 0 position-bound, trained groups mostly move | trained groups are content | unit 2 asks whether they are learned |
| trained groups mostly position-bound too | the trained opening is still the mask's | trained groups that move are the candidates; the rest are labelled positional, not dropped |
| at P = 0 the group exists only with the sink (gone under T1 at every P) | the opening was the sink | say so; step 0's control failures were T1's token |

**Cost.** 7 passages × 4 P × 3 preambles × 2 joins × 2 steps = 336 forward passes, at
most 1512 tokens, hidden states only (no attention stored): under an hour on CPU.
Readouts are computed in-process; activations are kept only for P ∈ {0, 1000}, the
first preamble, the EOD join (≤ 25 × 1.5k × 1024 float32 ≈ 150 MB each, ~2 GB on
`HDD_1TB`). Estimates, not measured.

### Unit 2: the architecture as the null

| choice | value | why, and what was rejected |
|---|---|---|
| init draws | (i) PolyPythias `pythia-410m-seed{1..9}` `step0` plus `pythia-410m` `step0` (seed 0): 10 real inits; (ii) **100 re-inits** of the same config: N(0, 0.019764) for the embeddings, QKV, `dense_h_to_4h` and the unembedding, N(0, 0.002604) for `attention.dense` and `dense_4h_to_h`, biases 0, LayerNorm (1, 0), seeds 0–99, never written to disk | (i) is Pythia's own init (row 1). (ii) gives a rank resolution (i) cannot (p ≥ 1/11). **Rejected: transformers' `init_weights()`**: σ 0.02 for every Linear, 7.7× Pythia's on the output projections (row 2a). 100 is **placed** |
| statistics | per cloud: HDBSCAN group count (`hdb_k`), median nearest-neighbour cosine (`nn1`), 2-means excess (`ci2`); per group: `S_C / |C|` as admission | the readouts 1d already has numbers for |
| rule, per cloud | rank p = (1 + #{inits ≥ obs}) / (N + 1) over (ii), and `z` = (obs − mean) / SD over (ii), per (prompt, layer, frame) | the same form as every 1d null |
| rule, per group | admitted if `S_C / |C|` exceeds the 95th percentile, over the inits, of each init cloud's **maximum** `S_C / |C|` | admission's max statistic, with the Gaussian replaced by the architecture |
| also reported | the Gaussian admission verdict (`admit_record`) as a column | "beyond its covariance" is still a fact about a group; it is no longer the definition |
| replication | PolyPythias `step143000`, seeds 1–9. A seed-0 group replicates in seed s if its best-Jaccard admitted group there has Jaccard ≥ 0.5; it **replicates** if it does in ≥ 6 of 9 seeds. A per-cloud excess replicates if rank p ≤ 0.05 in ≥ 8 of 10 seeds. Seeds 3, 4 are kept and flagged (outliers, row 1) | **placed**. Token sets match across seeds (same tokenizer, same prompt) |

**First check (before any trained cell is read).** The 10 real step-0 clouds must look
like re-inits: pooled over 7 prompts × 24 layers × 3 statistics × 2 frames, the share
of real seeds whose rank falls in the outer 10 % of the re-inits is ≤ 20 % (2×
nominal, **placed**). If it fails, the re-init does not match Pythia's: the null falls
back to the 10 real inits (coarser), and the mismatch is reported. Step 0 is then a
control that passes by construction, as 11⁗ intended.

**Outcomes** (not registered):

| outcome | reading | consequence |
|---|---|---|
| trained beyond the inits, replicating across seeds, in a band | learned structure there | the candidate definition: unit 1's moved groups ∩ unit 2's admitted |
| trained beyond the inits but not replicating | learned, seed-specific | reported; not a definition Phase 10 can use across seeds |
| trained within the inits | at this resolution, training does not make these groups | 1d says so; the per-group question waits on unit 3's scale |

**Cost.** ~18 checkpoints into the HF cache (~1 GB each; 72 GB free on 2026-09-29).
Forward passes: 7 × (100 + 10 + 9) ≈ 830 at ≤ 512 tokens, minutes. HDBSCAN on each
cloud × 24 layers × 2 frames × 2 sizes: ~80 k level-set fits; at #122's rate (1 344
records with 200 draws each, ~270 k fits, in ~10 min) under 10 min. Estimates.

### Unit 3: one family on a continuous scale

| choice | value | why, and what was rejected |
|---|---|---|
| family | **average-linkage merge tree on cosine distance** (`merge_tree.layer_merge_tree`), cut at each δ of a grid | built; deterministic; its scale is a cosine distance, the units of the theory's `δ = cβ^{-1/2}`. **Rejected as primary: Markov stability** on a CkNN graph (`lit-1d.md` §10 row 9): its scale is Markov time, it adds a graph parameter and Louvain randomness. It is the named alternative if the merge tree finds no plateau |
| grid | δ ∈ 40 log-spaced cosine distances from 0.01 to 1.5 | **placed**; spans nearest-neighbour distances to near-orthogonal |
| per δ | (a) Hennig's cluster-wise stability: mean best-match Jaccard over 50 subsamples of 80 %; (b) the substantial cluster count (≥ `SUBSTANTIAL_CLUSTER_SIZE` tokens) against unit 2's re-inits at the same δ, as rank p; (c) for passage partitions, the Jaccard between P = 0 and P = 1000 (unit 1) | (a) "is it a cluster", (b) "is it learned", (c) "is it content", each at every scale |
| robust scale | ≥ 3 consecutive grid points with the same substantial count, mean cluster-wise stability ≥ 0.75, and the count's rank p ≤ 0.05 against the re-inits | **placed**; 0.75 is Hennig's (2007) "stable" bound, not his 0.85 "highly stable" |
| other families | compared to the merge tree only at matched scale: a family's partition against the merge-tree cut with the nearest substantial count | the seven-family consensus is retired as a definition |
| theory scale | `δ = cβ^{-1/2}` marked on the grid when Blocked 9 is decided | waits |

**First check.** On unit 4's multi-scale synthetic (below), the planted plateaus are
found at both planted scales (ARI ≥ 0.8 to the planted labels, **placed**) and no
plateau is found on a single Gaussian of the same covariance. Built in unit 3, before
any real input.

### Unit 4: positive controls

| control | what | pass |
|---|---|---|
| multi-scale synthetic | n = 400 on S^1023: 3 groups of 3 sub-groups (von Mises–Fisher, two planted angular spreads), 25 % background; then the opening mechanism applied: each row mixed with the mean of the rows before it (`identity_sim`'s β = 0 dynamics at small `t`) | unit 3 finds both planted scales; the opening group forms; re-ordering the rows moves it and leaves the planted groups (unit 1's logic) |
| designed-content prompts | 3 new prompts, frozen in a file committed **before** any forward pass on them: a list interleaving three categories, prose interleaved with code, a narrative that repeats a few named entities | at step143000, groups that hold one category / one entity, admitted by unit 2 and moving under unit 1; at step 0, not. A prediction, not registered |

The designed prompts are new text, not v2: whether they join a battery is the user's.

### Order, and what is superseded

Units run 1 → 2 → 3 → 4, each its own PR; unit 3 builds the synthetic its first check
needs. The cut, the two position nulls and the per-group Gaussian admission stay as
code and as rows in `status-1d.md` "Where 1d stands"; none is the route to a
definition. "Build order" below is the pre-11⁗ plan, kept for its history.

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
| Position | a confound, partly measured | 30 % of deduped nearest neighbours within 3 positions (8 % at step 0); no residual null keeps position (Parked). *2026-10-01 (`status-1d.md` "Position"):* step 0's admitted groups are the prompt's opening (near-uniform attention at init); with positions < 32 left out step 0 admits nothing (cut chosen on the control), though its group count keeps a small excess. Positional tilts are common in trained groups at L9–24 (~45 % flagged, admitted or not) but mostly slight (10–18 % mostly near pairs) |
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
| tokens | first occurrence of each string (`--dedupe-strings`) | all-token results are token identity: step 0 beats the null harder than step143000. Later occurrences get label −1, "not tested"; a nearest-group assignment for them is a separate, flagged column if Phase 10 needs one. `repeated_tokens` (3 strings) drops out, so the batch is 7 prompts. **Tried and out (2026-10-01, Blocked 11′):** leaving out absolute positions < 32 (`admit run --min-position 32`), the cut under which step 0 admits nothing on v1. 32 was chosen after seeing 8, on the control itself, so it was kept only if step 0 passed at it on the long prompts (rule committed before the run). It failed: on the long runs the opening re-forms at the first kept token (`status-1d.md` "The M = 32 cut on the long prompts"). What replaces it is Blocked 11″ |
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
| Position: span, share of member pairs within 3 positions, contiguous-run flag, each against random groups of the same size drawn from the kept positions | is the group a stretch of text rather than a content group. Kept first occurrences sit early in the prompt, so the baseline is the kept positions, not all positions | yes, `position_check.py` (2026-10-01) |
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
| admitted groups are mostly positional | the groups are text stretches | a residual null that keeps position before calling anything content (chosen 2026-10-01, Blocked 11″; form and cost in `status-1d.md` "Blocked 11″ decided") |
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
   Position row built and run 2026-10-01 (`status-1d.md` "Position"); the other rows are not.
3. **The theory scale**, when Blocked 9 is decided.
4. **Phase 10 re-read** on the admitted labels: the user's call, since Phase 10 is on hold.

1b. **Identity-weights positive control** (user, 2026-10-01; section below). It
   runs before any trained group is read, beside Blocked 11′.

Prerequisites not on this list, Parked in `status-1d.md`: the Gaussian null's drift
with n (needed before long prompts), Parked 11 (step 0's lifetime excess; needed
before the merge tree's lifetime is read again), a position-keeping residual null.

## Identity-weights positive control (designed 2026-10-01, before any run)

Literature: `lit-1d.md` §9. Admission has only been run on Pythia, where there is no
ground truth. Two questions need a case where there is one.

1. **Power.** Does the definition admit the clusters that the theory's own dynamics
   make? A definition that cannot see those is not measuring the theory's object.
2. **Mechanism.** #123 read step 0's admitted groups as the prompt's opening, because
   near-uniform attention makes early positions share the first tokens' values. The
   theory's causal dynamics at Pythia's β, with nothing else in them, either make that
   opening cluster or they do not.

**What is simulated** (`p1d_cluster_ensemble/identity_sim.py`):

| choice | value | why, and what was rejected |
|---|---|---|
| equation | `2411.04990`'s (CSA) with `Q = K = V = I`: `ẋ_k = P_{x_k}( Σ_{j≤k} e^{β⟨x_k,x_j⟩} x_j / Z_k )`, self included, on the unit sphere. One head, the same weights at every time, no MLP, no RoPE, no LayerNorm | the case Thm 4.1 covers (`lit-1d.md` §9 row 1). Rejected: `2605.09213`'s model (no self term, ALiBi, no softmax partition; §9 row 3), which is a different equation |
| mask | **causal** (primary) and **full** (`j` over all tokens: Geshkovski et al.'s (SA)) | full has a closed form (`p1c_frames/gamma_ode.py`), and it is the control for the mask: an opening cluster under causal and not under full comes from the mask |
| start | Phase 1's L0 rows (the embedding output, unit rows of `activations.npz`), deduped by first occurrence as in admission. 7 v1 prompts × step143000 / step 0, 125–273 tokens | L0 carries no position (Pythia's position is RoPE, inside attention), so any positional structure in a trajectory comes from the mask alone. Step 0's L0 is a random embedding with near-orthogonal rows, close to the iid start `2605.09213` analyses |
| coordinates | an orthonormal basis of the start rows' span (`span_coordinates`), so `n ≤ 273` dimensions instead of 1024 | exact, not an approximation: every velocity is a combination of the `x_j`, so the trajectory never leaves the span. Tested below |
| β | 0, 0.2, 0.43, 1, 2, 3.46, 5.57, 8, 16, 64 | 0.43 (= 3.46 ÷ 8) and 3.46 [1.55, 5.57] are Blocked 9's two conventions (`status-1d.md` "β refit"). 0.2 and 8 bracket them. 0 is uniform attention over the prefix, the mechanism #123 named, with nothing else in it. 16 and 64 put the theory's `δ = 4β^{-1/2}` (1.0, 0.5 rad) below the typical angle between tokens, so several centres can exist. At real β, `δ` is 2.2 to 6.1 rad: one centre, `x₁` (§9 row 2). Without 16 and 64 the positive control has nothing to recover |
| time | `t` ∈ {0, 0.5, 1, 2, 4, 8, 16}, the same for every β | (6.9) puts γ = 0.9 at `t* ≈ 4.2` for `n = 467`, nearly free of β (`gamma_ode.collapse_time_table`), so the grid runs from 8× below `t*` to 4× above it. Each snapshot also stores (6.9)'s `t_0.5` and `t_0.9` at its own `n` and β. Mapping `t` to Pythia's depth needs `T_eff`, which these runs never measured (Phase 1c), so no snapshot is called "layer ℓ" |
| integrator | RK4 with rows renormalised every step; `dt` halved until no snapshot's Gram moves by more than 1e-6 | the field's Lipschitz constant grows with β, so one `dt` does not fit the whole grid. Same rule as `integrate_gamma_converged` |
| float floor | a snapshot whose smallest pairwise `1 − cos` is below 1e-9 is not admitted, and the record says why | admission's distance route reads float32 rows (`LayerData.from_normed`). Collapsed pairs below that resolution become ties at 0 with an infinite λ. Refuse rather than degrade. **Placed** |

**Tests before any real input:**

1. Full mask, orthogonal starts (`n` ∈ {2, 5, 20}, `d ≥ n`), β ∈ {0, 1, 5}: every
   pairwise inner product equals (6.9)'s γ(t) from `gamma_ode.integrate_gamma` to 1e-6,
   and all pairs stay equal.
2. Causal, `n = 2`: `γ_causal(t) = γ_(6.9)(t/2)` at `n = 2`, because only the second
   token moves, so the pair is (6.9) at half speed. A `sympy` check in
   `tools/math_checks/` covers the right-hand side. It does not prove the integrator
   right; test 1 does that.
3. Thm 4.1: under the causal mask, `x₁` never moves (to 1e-12), and with a small random
   start and long `t`, every token's cosine to `x₁(0)` approaches 1 (β ∈ {0, 1, 8}).
4. The span reduction: integrating in the span and in `R^d` gives the same Gram to 1e-10.

**The theory's clusters (ground truth, fixed before the run).** At each snapshot,
these are the connected components of the graph that joins tokens with
`1 − cos(x_i(t), x_j(t)) ≤ η`, keeping components of ≥ 2 tokens. η = 1e-3; the
sensitivity arms are 1e-2 and 1e-4. **Placed:** η is not derived, and the run reports
how the count moves with it.

**Readouts per snapshot:**

| readout | what |
|---|---|
| admission | `admit_record` unchanged on the snapshot: both frames, `min_cluster_size` 2 / 4, 200 draws; and again with `calibrate` |
| recovery | for each theory cluster of ≥ `min_cluster_size` tokens, its best Jaccard to an admitted group. Recall = the share of such clusters with Jaccard ≥ 0.5. Precision = the share of admitted groups with Jaccard ≥ 0.5 to some theory cluster. ARI between admitted labels (unadmitted = noise) and theory clusters (singletons = noise). The 0.5 is **placed** |
| opening | the theory cluster that holds position 0: its size, and its members' ranks among kept positions. `cos(x_k(t), x₁(t))` by position. The admitted groups that hold position 0 |
| position | #123's `position_check` on admitted groups, against random same-size groups from the kept positions. **This needs #123 merged**; until then this readout waits |
| step 0's real groups | for step 0 at β ∈ {0, 0.43, 3.46}: the Jaccard of the simulated opening cluster, and of admitted groups that hold position 0, to step 0's real admitted groups that hold position 0 at L1–24 (`data/p1d/admit_2026-10-01`) |

**Outcomes, written before the run** (not registered):

| outcome | reading | consequence |
|---|---|---|
| at β ∈ {16, 64}, where theory clusters of ≥ 4 tokens exist, recall ≥ 0.5 in most such snapshots, and calibration admits at its usual rate | the definition sees the theory's clusters | it has passed a positive control it could have failed |
| theory clusters exist, recall < 0.5 | the max statistic is too strict for the theory's own clusters (the cost named in "Why a max statistic") | size bands, as planned there |
| no theory clusters other than the opening at any β ≤ 64 by `t = 16` | the grid misses the multi-cluster regime at `d = n` | there is no positive control. Say so; do not read admission's silence as a pass |
| at β ∈ {0.43, 3.46}, causal, early `t`: one theory cluster holding position 0, drawn from the earliest positions, absent under the full mask | the mask-only dynamics cluster the opening at real β. Step 0's opening groups are the theory's first cluster (Thm 4.1's `x₁`), not an HDBSCAN artefact | Blocked 11′'s cut (drop positions < 32) removes the theory's own prediction. The alternative is to keep the opening as a labelled cluster. The user's call |
| the opening does not cluster first, or clusters under the full mask too | step 0's opening needs more than the mask (LN, MLP, the untrained weights) | #123's attention-uniformity reading is incomplete |
| step 0's simulated opening matches its real groups (Jaccard ≥ 0.5 at some `t` for most prompts) | the untrained model's opening is the mask's dynamics on its embeddings | the same as the row above, with a token-level match |

**Cost.** 14 inputs × 10 β × 2 masks = 280 trajectories, and 14 × (1 + 10 × 2 × 6) =
1 694 snapshots × 2 frames, real and calibration, so about 6 800 admission records.
The admission batch did 1 344 records in about 10 min at 14 workers, so this is
about 1 h. Open the first record before the batch.

**What it cannot show.** Pythia is outside the theory: RoPE, MLP, 24 untied layers,
16 heads. A pass says the definition sees the clusters of the case the theory
covers. It does not say Pythia's admitted groups are those clusters. Time here is
not depth.

**Run 2026-10-01** (`status-1d.md` "Identity-weights positive control"). Against the rows
above: the positive control was read once and failed at the group step (raw, several
theory clusters, calibration 0: recall 0.39); elsewhere the theory makes one global
cluster and the centred calibration fires. The time grid was wrong for β ≥ 8: the claim
that `t*` is nearly free of β holds only up to β ≈ 3.5 (`inf` at 16 and 64), so the regime
this design aimed at was never reached. The opening row is met for an *admitted group*,
not for a theory cluster as written: causal, β ≤ 1, `t` 1–2, 5–7 of 7 step-0 prompts,
calibration 0, none at β = 3.46 (the full mask cannot make a positional group from
position-free rows, so that control is trivially passed). The token match to step 0's
real groups holds, but a first-|g|-tokens baseline does as well.

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
