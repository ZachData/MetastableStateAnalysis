<!-- docs/AXES.md -->
# The measurement grid: what axes exist, what is populated, and what has never been asked of them

**`INDEX.md` maps phases to directories. This file maps *questions to data*.**
It exists because the project keeps rediscovering that an expensive-sounding
question is already answerable from artifacts on disk — `attentions.npz` in
152/152 directories, the transport half of `core/dissipation.py`, the per-head
attention entropies nothing reads — and because the opposite also happens: a
cheap-sounding question turns out to need a producer nobody has written.

Written 2026-09-20 out of Phase 10 (`PROJECT.md` §3.49). **Not a plan and not a
priority list** — `INDEX.md`'s Current priority block is that. This is the map
you consult before deciding something costs a forward pass.

---

## 1. The seven axes

Every measurement in this project is a cell in the same grid.

| axis | values | who varies it |
|---|---|---|
| **model** | gpt2-large, pythia-70m, pythia-410m, pythia-1.4b, + a norm-matched random twin each | Phase 8's ladder; `CLAIM-C`'s arms |
| **checkpoint** | 19 revisions for 70m and 410m, step0 → step143000; sparse for 1.4b | Phase 1's sweep; the whole developmental programme |
| **prompt** | 21 in the battery (v2, 2026-09-19); **8 in the Phase-1 sweep**; `n ∈ [20, 512]` | `core/config.py::PROMPTS` |
| **layer** | 6 (70m), 24 (410m, 1.4b), 36 (gpt2-large) | everything |
| **head** | 8 (70m), 16 (410m, 1.4b), 20 (gpt2-large) | Phase 2, 7, 7d, 7e |
| **token / position** | per-particle; the unit `core/particles.py` is keyed on | Phase 5, 5c, 10 |
| **frame** | sphere · LN · functional · (prospective) J-lens | `core/ln_frame.py`, `core/functional_distance.py`, `p10_cluster_function/lit-10.md` |

Three more that behave like axes and are easy to forget:

- **sub-layer channel** — attention vs FFN, **exact** on Pythia's parallel
  residual (`dissipation_by_channel`). **0/152 directories carry the sublayer
  streams**; they need `run_1.py --sublayer`, i.e. new forward passes.
- **ablation mode** — zero vs mean. Not a detail: zero's off-distribution bias
  scales as `1/n_heads`, so it distorts at 8 heads/layer (70m, 1b) and barely
  touches 16 (410m, 1.4b).
- **trained vs random** — the twin, and the contrast most of Blog 1 rests on.

---

## 2. What is on disk

From `PROJECT.md` §1 and `claims/audits/p1c_inputs.json` (which enumerated keys
across all 19 `pythia-410m` run directories and 152 model-prompt directories).

### 2.1 Weights

| model | on disk | notes |
|---|---|---|
| `pythia-410m` | 51 GB, the pilot schedule | the only model with a complete Phase-1 artifact set |
| `pythia-70m` | 5.0 GB, **19 revisions, step0 → step143000** | 6 layers, `d = 512`, 8 heads. **Cheapest full sweep available** |
| `pythia-1.4b` | 11 GB fp32; revisions `step143000`, `step0`, `main` | **RESERVED rung** |
| `gpt2-large` | 3.1 GB | `CLAIM-C` arms |
| `albert-xlarge-v2` | **not here** | Phase 6's ALBERT results are historical |

Everything runs with `HF_HUB_OFFLINE=1`.

### 2.2 Artifacts — the 410m sweep, 19 checkpoints × 8 prompts

| artifact | coverage | what it unlocks |
|---|---|---|
| `attentions.npz` `(n_layers, n_heads, n, n)` | **152/152** | **per-head attention routing, per checkpoint. Largely unexploited** — see `p10_cluster_function/attention-10.md` |
| `activations` + `norms` | 152/152 | every geometric statistic; the raw stream |
| `hdbscan_labels.json` | **EMPTY in 152/152** | nothing. **Measured 2026-09-20**, no exceptions: every directory was written while `hdbscan` was missing from this machine, so the file is `{}` and `clustering.json`'s hdbscan block reads `{"n_clusters": null, "nesting_summary": "HDBSCAN not available"}`. This entry previously read "written by the runner" |
| `hdbscan_backfill.json` | **152/152**, written 2026-09-20 | **the cluster partition**, re-derived from `activations.npz` by `tools/run/backfill_hdbscan.py` — no forward pass, 69 s for the sweep, verified bit-identical against 3 pilot directories × 25 layers. Read it through `backfill_hdbscan.read_labels`, which owns the precedence. Mean 47.1 clusters/layer, mean noise fraction 0.389 |
| `kmeans_labels_L*`, `agglom_mid_labels_L*` | 152/152 | alternative partitions, for frame agreement |
| `attention_entropy_per_head` | stored per layer by `p1_io.py` | **the row-side attention statistic. Nothing has ever read it against cluster structure** |
| Phase 2 eigenspectra + `sym_*` / `schur_*` projectors | 19 dirs | the operator channels |
| `kmeans_centroids_L*` | **0/152** | `P-S1`'s primary arm |
| `beta_eff`, `beta_eff_per_head` | **0/152** | `P-gamma1`, `P-gamma2` — **derivable with no forward pass**, §4 |
| sublayer streams | **0/152** | the frame-correct `h_attn_only`; needs new passes |

Plus, git-ignored and irreplaceable: the 27-step pilot sweep on `HDD_1TB` (exists
nowhere else), and the 355 GB Blog-1 activation cache, deliberately left alone.

### 2.3 `pythia-410m` step0 and step1 are the same weights

**Measured 2026-09-20**, and it is an upstream fact rather than a bug here. The
HuggingFace revisions `step0` and `step1` resolve to different commits and
different blob hashes, but **all 292 tensors are bit-identical**, and the
sweep's activations for the two are bit-identical in turn (`step0` vs `step2`
differs at 2.4e-07, so the comparison is not saturated).

So the 19-revision checkpoint axis carries **18 distinct points at 410m**, and
a per-checkpoint average over the sweep double-counts one of them. It is a
small effect — two adjacent points at the extreme early end — but it is free to
correct and impossible to notice from the directory names.

Not yet checked at 70m. Anything reading the checkpoint axis as 19 independent
points should check.

### 2.4 The prompt axis is narrower than the battery

**The battery holds 21 prompts. The Phase-1 sweep ran 8** (19 × 8 = 152). So
**13 prompts have never been through Phase 1**, and widening that axis is the
one axis extension that is *cheap per unit of information*: it needs forward
passes, but it needs no new code, no new registered ground, and no reserved
rung.

> It matters most for one specific test. **The Rényi-parking prediction is a
> cluster count *as a function of `n`*** (`lit-1.md` §4 item 1, rated the
> project's best cheap experiment). Its independent variable is prompt length.
> Eight prompts is eight points; twenty-one is twenty-one, over a wider `n`
> range. **The cheapest way to strengthen the project's best adjudication
> candidate is to run more prompts through Phase 1's clustering at a handful of
> checkpoints.**

---

## 3. What each axis buys

| axis | what only it can answer | cost to widen |
|---|---|---|
| **checkpoint** | **when.** Turns a fact into a trajectory, and it is the only axis that can co-locate one event with another | free at 410m (already run); forward passes elsewhere |
| **model / rung** | whether `n = 1` becomes a population claim | high, and **threshold-bound** (§5) |
| **head** | turns "this layer does X" into "these heads do X" — the only form that joins to 7d/7e's causal catalogue | **free** where `attentions.npz` exists |
| **token** | the particle table; parked vs pinned vs carrier | free |
| **layer** (normalised) | whether a phenomenon sits at an absolute depth or a fraction of it | free, but needs ≥2 depths |
| **prompt** | `n`-dependence — the parking law's variable — and content-sensitivity | forward passes only |
| **frame** | the arbiter when two views disagree | a lens fit, or a forward pass with an LM head |

---

## 4. Producers that do not exist, and what each unblocks

The e-value audit's finding, generalised: **the instruments are in better shape
than the inputs, and the gap is usually a producer nobody wrote, not a
computation nobody can afford.**

| producer | cost | unblocks |
|---|---|---|
| **`beta_eff` writer** | **no forward pass.** `attentions.npz` (152/152) + the checkpoint's LN params → `ln_frame.frame_for_hidden_state` → `ln_frame.ln_frame_gram` → `beta_eff.estimate_beta_all_heads`. Demonstrated on one checkpoint: 16/16 heads valid in all 24 blocks, median R² 0.18, one model load | **`P-gamma1` and `P-gamma2`** (two registered predictions), the temperature axis for everything, and every `gamma_beta` comparison Phase 9 would make. **Gated on a decision: β's unit convention, worth a factor of 8** |
| ~~`tools/run/transport.py`~~ | **WRITTEN 2026-09-20** | `w2_optimal` vs `w2_identity`, arc length, straightness, and the per-particle kinematic split. No longer a gap |
| **particle-table populator** | no forward pass | `turnover_decomposition` (validated 2026, awaiting data ever since), `particle_biography`, every per-particle aggregate |
| **operator-derived cluster labels** | no forward pass | `math-6.md` §4 item 4's third frame — the only labelling derived from the operator that generates the dynamics |
| **a J-lens fit** | backward passes, ~100 prompts, ~3 MB per checkpoint at 70m | per-layer functional partitions; `lens_band.py`'s band onsets stop being upper bounds; Phase 5's Group E |
| **`Z_beta,i` per token** | no forward pass | **never looked at at all** (`math-1.md` §15 open q12). The per-token metric; parked vs pinned |

---

## 5. The rules that constrain combining axes

Recorded because each was learned by breaking it.

1. **No absolute threshold transfers between rungs.** Not a membership bar, not
   `r* = 12`, not a noise floor. The first 70m write-up compared "19 of 48 heads
   clear +0.05" against 410m, where the bar had been calibrated on a baseline NLL
   of 0.585 against 70m's 5.725 — *"the same nats mean different things."*
   `compare_rungs.py` holds the threshold-free statistics: participation ratio,
   top-k share, gini, null-relative count.
2. **Full normalised depth, no band restriction.** *"A depth band is a choice
   with as many options as there are bands"* (`POPPER_PLAN.md` §755 item 4).
3. **Ablation mode is named, not defaulted** (§1).
4. **The rung policy: explore on 70m and 410m; 1b and 1.4b are RESERVED.**
   `lit-8.md` proposes a rule 4 — *a rung may be externally spent* — and it is
   **not taken**; it is a human call. 1.4b is the cleaner reserve, since no
   external source measures it on the induction axis.
5. **Prompts on one model are not independent.** They share its weights, so a
   model-wide effect present in every prompt is invisible to the enumeration.
   The prompt is the coarsest unit the design provides.
6. **The mean is not the default.** Exploratory work reports a mean **and** an
   extremum, always; registered work has frozen the choice
   (`PROJECT.md` §3.13). And a z-score against a population containing the point
   scored is capped at `(n−1)/√n` — 3.75 at `n = 16`, which one number has
   already saturated.
7. **Report the margin, not the boolean** (`P-H1`, `math-1c.md` §7.2).
8. **Sinks and norm outliers are audited, not assumed away.** Every share
   against its structural baseline, as an enrichment; the decision rule stated
   before the numbers (`core/sink_audit.py`).
9. **Co-location needs a matched-control null and a falsifier named first.** The
   registered permutation null for this class was measured and found to reject
   under H0 at 0.32–0.45 (`core/changepoint_colocation.py`).
10. **A sub-study on a non-ladder model is labelled ground of its own.** The
    `lora_ind` dense-onset run is **reachability, not developmental**, and it is
    a fork, not `pythia-70m`.

---

## 6. What has never been asked

Ranked by (information × cheapness). Everything above the line reads artifacts
already on disk.

| # | question | cost | where |
|---|---|---|---|
| 1 | **Cluster count vs the Rényi packing law, as a function of `n`**, at 27 checkpoints | free | `lit-1.md` §4.1; rated best-cheap twice, never run |
| 2 | **Where does attention actually go** — per head, per population, paid vs received, and the population×population mass matrix | free | `attention-10.md` §4 |
| 3 | **The attention flip on the checkpoint axis**, and whether its sign crossing co-locates with the four known transitions | free | `attention-10.md` §2.4 |
| 4 | **`Z_beta,i` per token** — the trained per-token metric, and whether high-`Z` is the sink | free | `attention-10.md` §5 |
| 5 | **Transport**: how much displacement is genuine motion of the measure vs tokens swapping places | free | `notes-9.md` §9 |
| 6 | **Turnover**: same particles cycling faster, or different particles clustering later | free | `math-5.md` §8.1 |
| 7 | **Geometric vs functional partition agreement**, per layer | a lens fit or an LM-head pass | `notes-10.md` §4.3 |
| 8 | **Which heads divert attention**, joined to 7d's 384-head causal sweep | free | `attention-10.md` §4.1 |
| 9 | **Token frequency** as a confound for cluster membership *and* for received attention | free | `docs/LITERATURE.md` §6 item 10 |
| 10 | **Per-head attention entropy against cluster structure** — stored since Phase 1, never read | free | §2.2 |
| — | | | |
| 11 | **The 13 unrun prompts through Phase 1's clustering**, to widen the `n` axis | forward passes | §2.3 — strengthens #1 |
| 12 | **A Phase-1 clustering sweep on 70m** across its 19 revisions | forward passes, the cheapest in the project | gives #1–#10 a second rung |
| 13 | **The sublayer streams** (`run_1.py --sublayer`) | forward passes | the frame-correct `h_attn_only`; the exact attn/FFN split on the particle side |
| 14 | **β per head per checkpoint** | a producer, no forward pass | §4; two registered predictions |

**Check before assuming #12 is new:** Phase 8 ran the 48-head catalogue and the
invariants on 70m, which is head-level ablation, **not** a Phase-1 clustering
run. If no clustering sweep exists there, it is the largest cheap artifact the
project could add.

---

## 7. If only four things were done

1. **#1, the parking law.** A published quantitative prediction against a
   measurement already on disk. `claims/adjudications/` holds **zero entries
   against thirty-nine registrations**, and this is the shortest path to the
   first.
2. **#2 + #3, the attention audit and its trajectory** — but **`attention-10.md`
   §6's A0 first**, because the sink and causal-mask baselines decide whether
   there is a finding to plot.
3. **#14, the β producer.** No forward pass, unblocks two registered
   predictions, and supplies the temperature axis every other phase wants.
   Gated on one human decision (β's unit convention).
4. **#12, a 70m clustering sweep.** The only item that turns single-model
   observations into two-rung ones for everything above it.

The first three are re-analysis. The fourth is the cheapest new compute in the
project.
