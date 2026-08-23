# Phase 4 — FROZEN

**Frozen 2026-08-22.** `low_rank_ae.py` (Track 3) is frozen-for-deletion alongside
Phase 3's crosscoder, under the same trigger. Full detail in `status-4.md` and
`design-4.md`, which stay authoritative.

## Read this before filing Phase 4 under "SAE work that gave nothing"

It is the easy summary and it is wrong. Phase 3 was a global null. **Phase 4 Track 3
was not.** `status-4.md` flags it explicitly as "the one frozen module with a positive
result." Freezing it is a scoping decision — it is not current work — not a verdict on
the finding.

## What was run

Three tracks against **albert-xlarge-v2** and **gpt2-large**, both complete,
last verified 2026-05-04:

- **Track 1** — do Phase 3's crosscoder features track HDBSCAN clusters?
- **Track 2** — direct geometric / linear-probe separability, no dictionary at all.
- **Track 3** — a **low-rank autoencoder with no sparsity penalty**: a linear
  bottleneck of rank *r*, with *r* set to the HDBSCAN cluster count at each plateau
  layer, no activation function in the bottleneck. Built specifically as the
  controlled contrast against Phase 3: same activations, same alignment test,
  sparsity removed.

## Results

| Model | Track 1 | Track 2 | Track 3 | Overall |
|---|---|---|---|---|
| albert-xlarge-v2 | `crosscoder_tracks_clusters` | `strong_linear_separability` | **`v_alignment_recovered`** | metastable_features_detected |
| gpt2-large | `crosscoder_tracks_clusters` | `strong_linear_separability` | `v_alignment_still_null` | metastable_features_detected |

| Metric | ALBERT | GPT-2 |
|---|---|---|
| Feature–cluster NMI (max) | 1.000 | 0.857 |
| Chorus ARI (max) | 0.001 | 0.000 |
| Linear probe accuracy (mean) | 1.000 | 0.300 (layer-averaged; high within plateau windows) |
| LRAE/CC MSE ratio (mean) | 0.797 | 0.788 |
| **LRAE bottleneck → V attractive dirs** | **33** | **0** |
| LRAE bottleneck → V repulsive dirs | 3 | 0 |

**The headline, in the phase's own words: "sparsity was the confound, not absence of
geometric structure."** Phase 3's sparse dictionary aligned with V at chance
(0.484 / 0.501); drop the sparsity penalty and ALBERT's bottleneck directions land on
V-attractive directions 33 times over. GPT-2 stays null even without the constraint,
which `status-4.md` reads as Regime B (FFN-mediated) not concentrating metastable
signal into a low-rank V-aligned subspace.

Two caveats it records against itself, both kept: Track 1's positive NMI is confounded
by a low crosscoder fire rate (only 12–28 of 2048 ALBERT features, 12 of 5120 GPT-2
features have detectable plateaus), and Track 2's GPT-2 mean probe accuracy of 0.30 is
depressed by layers outside plateau windows — read the max alongside the mean.

## Reintroduction trigger

Identical to Phase 3's, and still a conjunction:

> Activation caches exist at **≥4 checkpoints** *and* a specific particle-dynamics
> question **requires a dictionary**.

## Why this result matters to future work even while frozen

It converts "do dictionary methods find the particle structure?" from an open question
into a sharper one with a direction already measured: **the sparse prior is what fails,
on two models, against a dense low-rank control that succeeds on one of them.**

Neither frozen phase had a checkpoint axis. A future phase does, which makes the
sparse-vs-dense alignment gap testable *as a function of training step* rather than at
a single endpoint — and that is a prediction with a falsifier, not a re-run. Whatever
takes that up should cite these numbers as the prior and should not need to retrain
anything here to do it.
