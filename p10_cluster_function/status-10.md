<!-- p10_cluster_function/status-10.md -->
# Phase 10 — STATUS

<!-- phase-card -->
## Card

- **Question:** What are the particles' metastable clusters made of and what do they do: is a cluster a parking lot that the metric makes cheap to leave but where particles stay, or a category with a functional and causal role, and can the drive that forms clusters be used as an instrument?
- **Inputs:** `pythia-410m` only. The Phase 1 sweep (152 dirs, 19 checkpoints × 8 v1 prompts, battery `1e47918ef77a`), with HDBSCAN labels backfilled from its activations, and the pilot sweep on `HDD_1TB` (243 dirs, 27 checkpoints, native labels) as the second measurement. Seven records at 2 000 permutations: `data/analysis/p10_*.json`. Stage 0 (option B): 20 v2 prompts × 19 checkpoints = 380 runs, battery `06790b90dcfe`, pin `64a4087`, selected only through `data/phase12/stage0_logs/stage0_index.json`; progress lives in `p10_cluster_function/handoff-10.md` §0.3. The 12 new prompts are held out (Registry)
- **Results:**
  - The trained-model attention flip toward unclustered tokens is ~94 % causal mask on 410m: all mask at initialisation, with a learned residual from step ~2000–4000 that persists — `p10_cluster_function/status-10.md` §1.1
  - F0: the earliest member of a density cluster sits late, not early, against the prediction, and the within-cluster restricted null leaves the effect in place. It measured a proxy for the paper's strong Rényi centre, so it is not evidence on the parking account — `p10_cluster_function/status-10.md` §1.2, `p10_cluster_function/lit-10.md` §11.4
  - F1: the identity coupling is the optimal transport plan at 3 630 of 3 648 layer boundaries, so every per-layer displacement on record is the true `W_2`. `swap_fraction` counts repeated tokens, not motion. Clustered particles move less than noise only in a window, steps 32–512 — `p10_cluster_function/status-10.md` §1.3
  - F12: raw `log Z` is almost all position. The sink is the minimum of raw `Z` and the maximum of corrected `Z` at every β tried, so the β-unit convention does not touch it. The raw clustered-minus-noise sign is already present at step 0, because HDBSCAN clusters by density — `p10_cluster_function/status-10.md` §1.4
  - F1 and F12 together read "parked, not pinned" at steps 32–64, a window the trained model passes through. This rests on reading `Z` as a metric, and the density confound is argued from step 0, not controlled — `p10_cluster_function/status-10.md` §1.5
  - F1 and F12 on the pilot's partition: per-step means within 0.005 and 0.008 at the shared steps, and no changed unit flips sign at step 32, the least stable step (50/50, 159/159). Below-baseline `Z` lasts past 512 on both sweeps (to 4000 on WDS, non-monotonically to 11 000 on the pilot), which F12's baseline table had left out. Same activations clustered twice — `p10_cluster_function/status-10.md` §1.13
  - The 410m sweep had no density partition in 152 of 152 dirs, and `pair_agreement`, the only semantic instrument, wrote well-formed zero records. The backfill re-derives the labels, bit-checked against the pilot, and does not rerun the analysis — `p10_cluster_function/status-10.md` §2, `p10_cluster_function/handoff-10.md` §1.2
  - The HDBSCAN partition is not reproducible run to run (ARI 5th percentile 0.347): it moves when activations differ by ~2e-7, one float32 rounding step. Today's pipeline repeats itself bit for bit only because it is deterministic on one machine. A0 and F0 hold on the second sweep; per-layer claims stay exposed — `p10_cluster_function/status-10.md` §3, `p10_cluster_function/status-10.md` §3.1, `p10_cluster_function/status-10.md` §1.11
  - Stage 1 steps 1 and 2 on the pilot agree with Stage 0 to ≤ 0.004 at all 13 shared steps; the handoff's first-look table averaged 9 prompts, not 8. No null — `p10_cluster_function/status-10.md` §1.11
  - The per-layer co-membership and lexical-carry claims hold on the pilot's partition: 3 103 of 3 120 and 383 of 384 per-layer readings agree, all but one disagreement a flip at the ±0.05 floor, and every quoted cell at a shared step (the trained model: 143000 only) is within 0.005. Most cells are bit-identical: 100 of 2 600 run-layers have a different partition. Same activations clustered twice, so this bounds HDBSCAN's instability only. No null — `p10_cluster_function/status-10.md` §1.12
  - 410m's step 0 and step 1 are the same weights, so the checkpoint axis has 18 distinct points — §3.51.3
  - `pair_agreement`'s "ext_semantic" count is mostly a repeat count on Pythia (repeats have cosine 1 at layer 0). Over training the repeat share of mutual-NN pairs falls, and the non-repeat pairs that replace them become similar in the trained embedding. No null — `p10_cluster_function/status-10.md` §1.6
  - Which tokens are clustered is mostly copy count at init and in shallow layers (`min_cluster_size=2`, copies coincide at layer 0), and a moderate effect in the trained model's deep layers. Among unique tokens BPE rank does not predict it; class does, weakly and against trash collection. No null — `p10_cluster_function/status-10.md` §1.7
  - At step 0 layer 0 the partition is what HDBSCAN makes of Gaussian noise with planted duplicates (twins noise at 0.14, singletons clustered at 0.43). At layer 0 the cluster count tracks the prompt's repeated token types at every checkpoint (7-prompt mean ratio 0.92–1.01; per run 0.78–1.24), and ≥ 63 % of clusters at any layer hold a repeat. Phase 1's `max_alive` falls at layer 0 in 92 of 133 runs, mean 57–63 at every step against 59.7 repeated types: on these runs the carrying capacity is mostly a repeat count. `repeated_tokens` is a different mechanism (~50 deep-layer clusters). No null — `p10_cluster_function/status-10.md` §1.8
  - Clustered unique tokens, against step 0: at init, clusters at depth already follow each token's own (random) embedding carried in the residual. Training adds their own class (Δ +0.20 over a random draw, by step 512) and moves them away from copy groups. Own-embedding similarity adds only +0.12, and adjacency a depth-only share. No null — `p10_cluster_function/status-10.md` §1.9
  - At step 512 the network computes that class effect (+0.20 beyond own-embedding similarity; a purely lexical cluster scores ≈ 0). Trained, it is mostly what the embedding already groups: L12 sits on the lexical control, L24 is +0.05 above it. The control was added post hoc. Clustered tokens keep less of their individual layer-0 direction. Context vs per-token feature is not separated. No null — `p10_cluster_function/status-10.md` §1.10
  - Lemma C.1's saturation is a formula for Phase 1's 50–55 carrying capacity — `p10_cluster_function/math-10.md` §5.4, §3.53
- **Superseded / wrong:**
  - F0's reading as a test of the parking account: a density cluster's earliest member is not the paper's strong Rényi centre; F13 is the real row — `p10_cluster_function/lit-10.md` §11.4, `p10_cluster_function/status-10.md` §5.1
  - The Rényi packing law "as a function of `n`": the law is in β and dimension, not `n` — `p10_cluster_function/lit-10.md` §5, `docs/AXES.md` §7
  - §3.53's "no gradient-flow structure": Lemma 5.3 makes the causal dynamics a sequential gradient flow — §3.52.5
  - `handoff-10.md` §0.4's "Stages 1–5 re-run on all 20 prompts for free", which contradicted registering on prompts chosen blind: the 12 are held out on 410m — `p10_cluster_function/handoff-10.md` §0.4, `docs/PHASE_REVIEW.md` "Decisions"
  - §5's order: replaced by §5.1 after the papers were read (F13 first) — `p10_cluster_function/status-10.md` §5.1
  - `handoff-10.md` §1.1's reading that neighbourhoods go "lexical → contextual": on Stage 0's v1 runs they go from repeats to repeats plus embedding-similar tokens — `p10_cluster_function/status-10.md` §1.6
- **Registry:** none, because the phase is pre-design and deliberately unregistered (`claims/EXPERIMENTS.md`). F14 is named as the one to register (`handoff-10.md` "Standing constraints"). The 12 new v2 prompts are held out on 410m as a confirmation set, only partly blind (`CLAIM-C` ran them on 1.4b and gpt2-large); whether F14 is scored on them, on all 20, and in what order is undecided and the user's (`docs/PHASE_REVIEW.md` "Open" 1–4)
- **Depends on:** 1@30c0d54cc5, 5c@e75b33ae46, 7d@19b7d835b7, 7e@0c1071db50, 8@8cc3fb223c, 9@e70efd632b
- **Feeds:** 9
- **Open threads:**
  - F5, the four-signature concordance, is the phase's central test. It needs the J-lens (F2 → F3 → F4), and F2 needs HF access — `p10_cluster_function/status-10.md` §4
  - F13, the strong-Rényi centre scan, which F0 stood in for; it is free and needs no partition. F14 needs it, and so do F15 and F20 — `p10_cluster_function/status-10.md` §5.1
  - `CLAIM-C`'s two HDBSCAN metrics have never been compared against the reproducibility floor. This is the only open item here that bears on a registered prediction — `p10_cluster_function/status-10.md` §5
  - No norm-matched random twin per checkpoint, so F12's density confound and the parked window are argued, not controlled — `p10_cluster_function/status-10.md` §5
  - Is the class grouping computed early in training context, or a per-token feature from an early layer? A context-shuffle test would separate them. The pilot sweep is unread — `p10_cluster_function/status-10.md` §1.10, `p10_cluster_function/handoff-10.md` Parked
  - Does the pilot's 27-checkpoint "50–55" `max_alive` also fall at layer 0? And what sets `repeated_tokens`' ~50 deep-layer clusters? — `p10_cluster_function/status-10.md` §1.8
  - Is 410m spent on the induction axis for any Phase 10 registration that joins 7d's causal sweep? — `docs/PHASE_REVIEW.md` "Parked"
  - Position is a confound in every row here, with three separate corrections and no shared one in `core/` — `p10_cluster_function/handoff-10.md` "Standing constraints on all of it"
  - The `p10_*` readers' default glob now mixes the Phase 1 sweep with Stage 0's v1 dirs. The holdout guard (`core/holdout.py`) removes only the 12. Stage 1's reader selects through Stage 0's index; the older readers still glob — `p10_cluster_function/handoff-10.md` "Parked", `p10_cluster_function/status-10.md` §0
- **After Phase 10:**
  - Rebuild a cluster ensemble (1d's intent) only after measuring whether tuning reduces §3's run-to-run drift (free: the two sweeps' activations)
  - Everything in Stages 1–5 again on all 20 prompts once registrations are frozen (free: Stage 0's dirs)
  - F20, the frozen-centre intervention, as the phase's known-answer dry run (forward pass: 410m or 70m, needs F13)
- **Reviewed:** 2026-09-25 · body `80ff3310c3`
<!-- /phase-card -->

**Registered predictions:** none, and none yet can be. `claims/registry.json` is
untouched and `claims/EXPERIMENTS.md` lists this phase as *pre-design and
deliberately unregistered*. **Nothing below may be quoted as an adjudication.**

**Last verified:** 2026-09-20.
**Overall:** **four rows of `notes-10.md` §8's ladder have RUN on real
checkpoints** — F0, F1, F11-A0 and F12 — plus a producer that unblocked them
and a measurement that was not on the ladder and matters more than two of the
rows. All tier 1, exploratory. Two headline rows were **replicated on a second
sweep with an independently-derived partition** and both hold.

Read `notes-10.md` for what the phase is for, `math-10.md` for the derivations
these rows test, `attention-10.md` for row A0's audit, **`handoff-10.md` for the
cluster-function thread's ordered plan and `questions-10.md` for its
hypotheses**, and `PROJECT.md` §3.51 for the cross-cutting findings (three of which are about the project's
instruments rather than about clusters).

> **2026-09-20, a scoped thread opened out of this one: `handoff-10.md`, and
> its first action is COMPUTE — Stage 0, taking the Phase-1 battery from 8
> prompts to 20.** Eight prompts is the power ceiling under which every e-value
> in this phase sits; the 12 unused v2 prompts were already chosen blind under a
> committed rule, so running them retires no selection risk that has not already
> been retired.**
> The cluster-function question — what clusters are made of, what they do, and
> whether the drive that forms them is usable as an instrument — has its own
> ordered plan in **`handoff-10.md`** and its hypotheses in
> **`questions-10.md`**. It starts at Stage 0, *what is actually in a cluster*,
> and **that stage found a fourth casualty of the HDBSCAN outage §2 records**:
> `pair_agreement` — the project's only semantic instrument — wrote a
> well-formed record of zeros into all 152 WDS directories rather than failing.
> The pilot sweep's copy is populated in 6 066 of 6 075 layer-records and had
> never been reported. First read in `handoff-10.md` §1.1.

> **2026-09-20, later the same day: five papers were READ as primary text**
> (`lit-10.md` §11–§15) — `2411.04990`, `2501.10573`, `2605.12765`,
> `2505.16831`, `2601.02932`. **No number below changes. One interpretation
> does, and it is F0's.** The paper's cluster nucleus is a token separated by
> more than `δ` from *every preceding token* — a geometric acceptance rule on
> positions and distances, with no partition in it — not the earliest member of
> a density cluster. **F0 therefore measured a proxy, and its failure is not
> evidence against the parking account** (`lit-10.md` §11.4). The real row is
> **F13** and it is free. §4 and §5 below are updated accordingly.

**Decision taken 2026-09-20 (user):** F0 runs **exploratory first**, which caps
it at tier 1 — the answer was seen before any wording was frozen. If it is ever
to be the project's first adjudication it needs a fresh axis: 70m, or the
battery prompts that have never been through Phase 1.

**Corrected 2026-09-20: it is 12, not 13** — `core/prompts.py`'s v2 battery is
21 prompts, Phase 1's sweep used 8, and the thirteenth (`short_heterogeneous`,
115 characters) is almost certainly too short to cluster. **Running those 12 is
now the first action of the cluster-function thread** — `handoff-10.md` Stage 0,
which carries the measured storage budget and the four checks. Because they were
chosen blind under a rule committed ahead of the text (`PROJECT.md` §3.42),
running them is not a new selection decision, and **the enlarged battery can
carry a registered prediction where the current one cannot.**

---

## 0. The instruments, and where they live

| what | where | tier |
|---|---|---|
| the holdout guard | `core/holdout.py` — `refuse_held_out`, `add_holdout_args`; every `tools/run/p10_*.py` calls it, and `tests/test_holdout.py` fails any that does not | pure |
| statistics + content-free baselines | `core/parking.py` | pure |
| the arbitrary-dependence e-merger | `core/evalues.py` — `average`, `average_p`, `max_attainable_average_E` | pure |
| tie-tolerant p, restricted null | `core/nulls.py` — `p_from_null_tolerant`, `label_permutation_null_within` | pure |
| the HDBSCAN backfill | `tools/run/backfill_hdbscan.py` | deps |
| row A0 | `tools/run/p10_attention_baseline.py` | — |
| F0 | `tools/run/p10_anchor.py` | — |
| Stage 1 steps 1, 2, co-membership | `tools/run/p10_ext_sem_threshold.py`, `tools/run/p10_token_composition.py`, `tools/run/p10_comembership.py` (read only `stage0_index.json`) | — |
| F1 | `tools/run/transport.py` | — |
| F12 | `tools/run/p10_partition_function.py` | — |
| the reproducibility floor | `tools/run/p10_partition_stability.py` | — |

**Records** under `data/analysis/`, every one at **2 000 permutations** with
`max_attainable_E` **22.37** and `design_can_reject` on its face:
`p10_row_a0.json` (3 646 units), `p10_row_a0_pilot.json` (5 593),
`p10_f0_anchor.json` (3 800), `p10_f0_anchor_pilot.json` (5 835),
`p10_f1_transport.json` (3 648), `p10_f12_z.json` (11 400), and
`p10_partition_stability.json` (2 600 — a floor, **no p-value by design**).

**How to re-run anything here.** From the worktree. Every `tools/run/*.py`
derives `sys.path` from `METS_REPO`, which since 2026-09-24 defaults to the
runner's own checkout (it used to default to the main tree; `docs/PHASE_REVIEW.md`
Parked 2). `METS_DATA` still has to point at the main tree's `data/`:

```bash
METS_REPO=$PWD METS_DATA=/run/media/system/WDS_500/Mets/data \
  /run/media/system/WDS_500/miniforge3/envs/mets/bin/python tools/run/<runner>.py
```

**The conda `mets` env, not `.venv`,** for anything touching HDBSCAN — §2.

**The holdout guard (2026-09-24).** Readers (the four `p10_*` and F1's
`transport.py`) refuse held-out inputs by default. `--v1-only` drops per-prompt ones
and `--allow-holdout` reads them; both are recorded under `"holdout"` in the output
record. It holds out: a run dir for one of the 12 keys (any model; by
`manifest.json` `prompt_key`, else by name), any file named for one, and three
**pooled** kinds that `--v1-only` refuses rather than drops, because they hold v1
values too: a file beside such a run dir (`pair_agreement.json`),
`claim_c_real_run.json`, and `stage0_logs/*.out|*.log`. P-I1's `behavioural.py` refuses
the 12 outright. The other runners over `data/phase12` are exempt, each for a reason
listed in `tests/test_holdout.py`. Dry run on `data/phase12` at 12:40, chunk 2 mid-step512,
reading names and manifest `prompt_key` only: 632 `pythia-410m-*` dirs, 193 held
out, 439 kept. Of the 8 `2026-09-19_*` `CLAIM-C` dirs, the four v2 ones (21 run dirs
each, 48 held out in total) are flagged, and so is each one's `pair_agreement.json`; the four
v1-only ones (9 each) are not. **The records above predate the guard and Stage 0.**
Re-running a reader on the default `--pattern pythia-410m-*` now also globs Stage 0's
v1 dirs beside the Phase 1 sweep (handoff Parked).

---

## 1. What is answered

### 1.1 Row A0 — **the attention flip is ~94 % causal mask**

152 directories, **3 646 layer-units**. `math-10.md` §1's content-free baseline
divided out per layer, with the partition/attention pairing matching
`noise_importance_proxy` exactly (a test plants a signature in one attention
index and checks it lands on the right partition layer).

| | unclustered | clustered | gap |
|---|---|---|---|
| raw, as this project reports it | 1.417× | 0.825× | 0.592 |
| causal-mask-corrected | 1.034× | 0.998× | **0.036** |

**6.2 % of the gap survives**, and the sweep mean hides the finding:

| step | raw gap | corrected gap | position bias |
|---|---|---|---|
| 0–16 | 0.26–0.39 | **0.003–0.004** | 0.061–0.088 |
| 32–512 | 0.10–0.36 | −0.006 to −0.164 | 0.038–0.083 |
| 4000 | 0.949 | **0.245** | 0.035 |
| 143000 | 1.689 | **0.172** | 0.013 |

**Entirely mask at initialisation** — corrected gap 0.004 against a raw 0.26,
exactly as derived. A residual appears at **step ~2000–4000** and persists,
while the position-bias confound *falls* with training. The learned effect is
real in the means and about a sixth the size the uncorrected number implies.

**Verified against an independent reimplementation** written from the closed
form on one layer of `step143000_wiki_paragraph`: raw 1.6685 / 0.4490,
corrected 1.0925 / 0.9238 — exact to four decimals against the runner.

**What may NOT be said.** The 5c number was measured on `gpt2-large` and
ALBERT; this re-measures the same statistic on the 410m sweep rather than
refuting it. And **no arm here is a norm-matched random twin** — step 0 is an
untrained checkpoint, the right baseline for a developmental read but a weaker
control than the `-random` arms, because it shares the initialisation scheme
rather than being matched to a trained model's norms.

### 1.2 F0, the anchor test — **nuclei are LATE, and it is not the confound**

`2411.04990`'s mechanism says early tokens are the nuclei of cluster formation.
Direction fixed in code, before the sweep was read: **"less"**.

- observed **0.3164** against an ordinary-permutation null mean of **0.2077**
- median p **1.00**, fraction below 0.05 **0.18 %**, merged E **0.536**
- the design *could* have rejected at 2 000 draws and did not

**The first version was confounded, and the fix is the result's main
methodological content.** The ordinary permutation is free to move whole
clusters between the early and late halves of a sequence. The sweep has
clustered tokens sitting later than unclustered ones (`position_bias` +0.051),
so a late clustered population produces late cluster minima with nothing said
about nucleation. `label_permutation_null_within` holds the clustered/noise
split **fixed** and shuffles only which clustered token carries which id.

**Both nulls agree.** The restricted null's mean moves only 0.2077 → **0.2311**
against an observed 0.3164, median p 0.9995, E 0.556. The confound accounts for
about a fifth of the gap and the effect survives it.

**AMENDMENT 2026-09-20 — what this row does and does not bear on.** Both
numbers stand; both nulls stand; the confound analysis stands and is still this
row's main methodological content. What does not stand is the inference from
here to the parking account. `2411.04990` §5.1 defines a *strong Rényi centre*
as a token `δ`-separated from **every** preceding token, `δ = cβ^{−1/2}`, and
its Lemma 5.1 remark claims only that the accepted indices `s_j` *"are mostly
small"* and that the early ones among them are near-stationary. **A density
cluster's earliest member is a different object**: it need not be separated
from anything, and a strong centre need not be in any cluster. §6's line "F0 is
not an adjudication of the parking account" was written for a weaker reason and
is true for a stronger one.

### 1.3 F1, transport — **the identity coupling IS optimal, and parking is a window**

3 648 layer boundaries.

**`swap_absorbed_fraction` is 0 to machine precision in 3 630 of 3 648** (mean
5.2e-05, largest 7.0e-02). Displacement across a layer is genuinely motion of
the measure, not tokens changing places, so **every per-layer displacement
number this project has recorded — all identity-coupling — IS the true `W_2`**
rather than the upper bound it was known to be. A validation of a whole class
of existing numbers, with no forward pass.

**`swap_fraction` (0.023 → 0.041, rising with training) is NOT motion.** At
layer 0 a Pythia hidden state is the token embedding alone, so repeated tokens
have identical vectors and swapping their assignments is an exact tie in cost.
Quoting it as transport means quoting repeated tokens. A test fixture pins
this, and pins that the tie argument needs a *small perturbation* — under a
random target the optimal matching genuinely beats identity.

**Straightness falls with training**, 0.151 → **0.122**: net displacement is a
sixth of arc length, and the mature model wanders more per unit of progress.

**The kinematic signature of §3.1 exists and is a window:**

| step | clustered − noise step | median p | merged E |
|---|---|---|---|
| 0–16 | +0.02 to +0.03 | 0.15–0.17 | 5.2–6.9 |
| **32–512** | **−0.29 to −0.49** | **0.0005 (floor)** | **12.9–17.6** |
| 16000–32000 | −0.13 to −0.15 | 0.024–0.039 | 8.7–8.8 |
| 143000 | −0.01 | 0.092 | 7.3 |

### 1.4 F12 — **`math-10.md` §2 confirmed, β-independently; the raw sign is a trap**

11 400 units, three betas.

| | β = 1 | β = 2 | β = 4 |
|---|---|---|---|
| raw `log Z` variance explained by `log(i+1)` | **0.995** | 0.979 | 0.893 |
| percentile of position 0 in **raw** `Z` | **0.0014** | 0.0014 | 0.0014 |
| percentile of position 0 in **corrected** `Z` | **0.9986** | 0.9986 | 0.9986 |

The raw partition function is **99.5 % position** at β = 1. The sink is the
**minimum** of raw `Z` and the **maximum** of corrected `Z`, identical to four
decimals across the grid — so `docs/AXES.md` §4's factor-of-8 convention
question **does not touch this result**. `math-1.md` §1A.6's "high-`Z` = sink"
is right about the corrected quantity and inverted about the raw one.

`position_r2_corrected` of 0.32 is the **known** unit-diagonal residual —
sphere-projected states give `Z_i = e^β + i e^{βγ}`, affine in `i` rather than
proportional to `(i+1)` — not unexplained structure. Both forms are test
fixtures.

**The raw clustered-minus-noise sign is mostly definitional.** It is +0.364
standardised (median p at the floor, 79 % of units below 0.05, E 14.60), which
on the metric reading is *pinned*. But HDBSCAN clusters by density and
`Z_i/(i+1)` IS a density, so a clustered token has high `Z` because that is what
put it in a cluster — and **the effect is +0.465 at step 0, under random
weights, before any training**, with 86 % of units below 0.05.

**Against the step-0 baseline:**

| step | clustered − noise | vs baseline |
|---|---|---|
| 0 (random init) | +0.465 | — |
| **32–64** | **−0.12 to −0.14** | **−0.59 to −0.61** |
| 256–512 | +0.12 to +0.22 | −0.25 to −0.35 |
| 1000–4000 *(rows added 2026-09-25, §1.13)* | +0.34 to +0.37 | −0.09 to −0.13 |
| 8000 | +0.469 | +0.004 |
| 16000–54000 | +0.57 to +0.64 | +0.11 to +0.17 |
| 143000 | +0.676 | +0.21 |

### 1.5 F1 + F12 together — **parked, not pinned, and a window**

At **steps 32–64** clustered particles **move less (F1) AND sit below the
untrained baseline in corrected `Z` (F12)**. Low `Z` on §1A.6's reading means
the metric makes them *cheap* to move — and they do not move.

> **That is the PARKED signature, not the pinned one.** It is a phase the model
> passes through rather than a property of the trained model: by step143000 the
> kinematic difference is −0.01 and the metric difference +0.21 over baseline.

`attention-10.md` §5 named this as the distinction displacement alone cannot
make. It took both rows.

**Where the window ends (2026-09-25, §1.13).** The negative raw sign is 32–64
only. Below-baseline `Z` lasts to 4000 on WDS (back at baseline by 8000) and,
not monotonically, to 11 000 on the pilot; F1's strong difference ends at 512.
Which of these bounds "the window" is not decided here.

**Held as a hazard, not a result.** The density confound is argued from the
step-0 value, not proved — the proper control is a norm-matched random twin per
checkpoint, which the sweep does not carry. And the whole reading rests on
§1A.6's **interpretation** of `Z` as a metric, not on a causal measurement.
**`notes-10.md` §3.1's functional and causal columns are untouched and F5 is
still what discriminates.**

### 1.6 Stage 1 step 1, the `ext_sem_threshold` sweep — **mostly repeats; the rest become embedding-similar**

Tier 1, descriptive, no null. Reader `tools/run/p10_ext_sem_threshold.py
--v1-only`, record `data/analysis/p10_s1_ext_sem_threshold.json`. **Input:**
Stage 0 through `stage0_index.json` (pin `64a4087`, battery `06790b90dcfe`),
8 v1 prompts × 17 steps = 136 runs (16000 and 54000 were not yet indexed;
chunk 2 was running), inputs sha256 `61127896b17d`. Every run reproduced its
stored `n_ext_semantic` at 0.5 exactly, per layer. **Re-run at 152/152 on
2026-09-24** (inputs `c558b210c08f`, the same set as §1.7's re-run): all 2036
summary numbers of the 136-run record are unchanged, 236 are new (steps 16000
and 54000), and the verdicts are the same (frozen dead, self mixed). The table
below has no row for either new step, so it stands.

Means over 8 prompts × 25 layers. "Repeat": the pair is two copies of one token.
Non-repeat columns use the **frozen** frame (layer 0 of step 143000), against
the non-repeat pairs of the same prompt. Base rate of cos > 0.2 in that frame: 0.145.

| step | repeat share | `ext_semantic_fraction` @0.5 (self) | non-repeat: cos > 0.2 | non-repeat: mean percentile | same-cluster among repeats |
|---|---|---|---|---|---|
| 0 | 0.875 | 0.875 | 0.010 | 0.37 | 0.700 |
| 64 | 0.914 | 0.914 | 0.079 | 0.44 | 0.704 |
| 512 | 0.839 | 0.839 | 0.311 | 0.69 | 0.743 |
| 2000 | 0.769 | 0.769 | 0.425 | 0.79 | 0.722 |
| 8000 | 0.733 | 0.747 | 0.477 | 0.80 | 0.703 |
| 143000 | 0.696 | 0.735 | 0.483 | 0.78 | 0.697 |

- **The stored statistic is mostly a repeat count.** Pythia adds no position
  embedding before layer 0, so repeats have cosine exactly 1 in every frame.
  In the self frame, no non-repeat mutual pair passes 0.5 before step 4000,
  so through step 2000 `ext_semantic_fraction` @0.5 *equals* the repeat share.
  After that the cut starts to act: at step 143000 the 0.1 and 0.5 cuts
  differ by 0.15 and self vs frozen by up to 0.03 in between.
- **The repeat share falls 0.875 → 0.696**, mostly between steps 64 and 4000,
  after a rise to 0.914 at step 64. The same-cluster rate among repeats is flat.
- **The non-repeat pairs that replace them become embedding-similar.** Take
  the trained model's layer-0 embedding as the reference. At init, the
  non-repeat mutual pairs sit *below* the typical pair (percentile 0.37). From
  step ~1000 they sit near the 80th percentile, and half have cos > 0.2 against
  a 0.145 base rate. The rise is steps 64–2000. **§1.1's "lexical →
  contextual" is backwards on this measure:** neighbourhoods go from "same
  token" to "same token or a token the trained embedding calls similar".
  Whether that similarity is semantic is step 2's question.
- **The pre-stated verdicts.** Self is "mixed" because of the 0.1 cut, below
  the non-repeat range. Frozen is "dead" because of the all-pairs median cut,
  which sits in the continuous part. Both are in the record. Neither answers
  §1.3's scale question, which the split above answers. The all-pairs quantile
  and rank columns land in the cosine-1 pile when a prompt repeats a lot, so
  the non-repeat block replaces them (`LESSONS.md` lesson 6).
- **Caveats.** No null. Layer 0 is 1 of the 25 layers averaged, and there the
  mutual-NN graph comes from the frame's own Gram. The stored-count gate at 0.5
  could not have failed on the cosines, since none sit near 0.5; it checks the
  tags only (the tokens-vs-pairs check is separate). Re-run when Stage 0
  completes (152 v1 runs). The pilot sweep (§1.1's source): §1.11, which
  agrees to ≤ 0.002 at every shared step.

### 1.7 Stage 1 step 2, the token-composition table — **copy count dominates at init and in shallow layers; among unique tokens rank does nothing**

Tier 1, descriptive, no null. Reader `tools/run/p10_token_composition.py
--v1-only`, record `data/analysis/p10_s1_token_composition.json` (full cells:
feature × level × copies, per layer per step, pooled and prompt-balanced).
**Input:** Stage 0 through `stage0_index.json` (pin `64a4087`, battery
`06790b90dcfe`), 149 v1 runs: 8 prompts × 19 steps less 3 at step 54000
(chunk 2 running). Inputs sha256 `564020cf46e1`, tokenizer.json `c24618a1b3e6`,
native labels only. **Re-run at 152/152 on 2026-09-24** (inputs `c558b210c08f`,
with §1.8's columns): the tables below show no step-54000 row, so they are
unchanged, and so are the verdict counts.

"Clustered" = HDBSCAN label ≠ −1. Rates are per prompt per layer, then averaged.
Copies: *unique* (one copy in the prompt), *first* (first of several), *repeat*
(has an earlier copy; §1.3's column). Rank = token id = BPE merge index + 245, a
frequency proxy.

| step | noise rate | unique | 2 copies | 3–5 copies | 6–20 copies | unique, rank < 1k | unique, rank ≥ 20k | unique word_start | unique punct |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.389 | 0.179 | 0.381 | **1.000** | 0.945 | 0.171 | 0.209 | 0.196 | 0.237 |
| 64 | 0.416 | 0.117 | 0.298 | 0.999 | 0.953 | 0.157 | 0.102 | 0.145 | 0.108 |
| 512 | 0.340 | 0.250 | 0.416 | 0.955 | 0.975 | 0.270 | 0.276 | 0.268 | 0.289 |
| 2000 | 0.355 | 0.277 | 0.489 | 0.900 | 0.882 | 0.242 | 0.310 | 0.321 | 0.290 |
| 8000 | 0.392 | 0.258 | 0.472 | 0.861 | 0.820 | 0.243 | 0.262 | 0.288 | 0.233 |
| 143000 | 0.397 | 0.257 | 0.456 | 0.801 | 0.782 | 0.238 | 0.271 | 0.291 | **0.146** |

(2 copies: the first copy; the later one is within 0.03. 3–5 and 6–20: later copies.)

| step | pre-stated, non-repeat: freq / class contrast | post hoc, unique only: freq / class |
|---|---|---|
| 0 | +0.132 / +0.059 | −0.015 / +0.078 |
| 64 | +0.222 / +0.035 | +0.076 / −0.022 |
| 512 | +0.172 / +0.161 | +0.031 / +0.060 |
| 2000 | +0.088 / +0.095 | −0.037 / +0.007 |
| 8000 | +0.113 / +0.133 | +0.015 / −0.027 |
| 143000 | +0.089 / +0.108 | +0.000 / −0.130 |

By layer (pooled over prompts; added after `/challenge-pr` on #90). Later copies
of 3–5-copy tokens / first copy of a 2-copy token / unique; then the unique-only
contrasts, freq / class:

| step | L0 | L12 | L24 | unique contrasts L0 · L12 · L24 |
|---|---|---|---|---|
| 0 | 1.00 / 0.87 / 0.43 | 1.00 / 0.33 / 0.17 | 1.00 / 0.33 / 0.19 | +0.01/−0.20 · −0.03/+0.09 · −0.02/+0.32 |
| 512 | 1.00 / 0.85 / 0.43 | 0.97 / 0.45 / 0.34 | 0.91 / 0.38 / 0.33 | |
| 2000 | 1.00 / 0.74 / 0.38 | 0.92 / 0.44 / 0.30 | 0.84 / 0.48 / 0.41 | |
| 143000 | 1.00 / 0.74 / 0.34 | 0.80 / 0.45 / 0.26 | **0.70 / 0.48 / 0.44** | +0.03/−0.29 · +0.01/−0.16 · −0.05/−0.05 |

- **Clustered vs noise is mostly copy count at init and in shallow layers.**
  HDBSCAN runs at `min_cluster_size=2` (`p1_mstate_tracking/clustering.py`),
  and copies of one token coincide at layer 0. At step 0 every token with 3–5
  copies is clustered at every layer. Training erodes the copy groups in depth:
  at layer 24 of step 143000 it is 0.70 against 0.44 for unique tokens, and a
  2-copy token is only 0.04 above unique. In the deep layers of the trained
  model, copy count is a moderate effect, not the whole story. The whole-prompt
  noise rate barely moves (0.34–0.42).
- **The pre-stated trash-collection criterion reads "consistent" at 11 of 18
  distinct checkpoints (step 0 = step 1, §3.51.3), but twins carry it.** Its
  non-repeat set keeps first copies, which have cosine-1 twins later in the
  prompt, and high-frequency tokens recur more. Among **unique** tokens the
  verdict is "unclear" at all 18.
- **Among unique tokens, rank does not predict clustering.** The frequency
  contrast is within ±0.08 at every step (0.000 at step 143000) and within
  ±0.05 at L0, L12 and L24. Rank < 1k is the *least* clustered unique bin from
  step 2000 on. This is the solid result.
- **Class does predict, weakly and against trash collection.** Unique
  punctuation is less clustered than word starts late in training (0.146 vs
  0.291 at step 143000), and the class contrast is negative at every layer
  shown there. At step 0 its sign flips with depth (−0.20 at L0, +0.32 at L24).
  The verdict reads "unclear" because the two contrasts disagree, not because
  nothing predicts. Thin: 28 unique punctuation and 3 unique whitespace tokens
  across the 8 prompts. Unique numerals climb to 0.55, the highest class (30
  tokens, 4 prompts).
- **What this does to the semantic question.** It narrows it. The clustered
  unique tokens are the residual to explain, but part of it is there at random
  init (the "unique" column at step 0), so the quantity is trained minus step 0,
  not the trained rate. The next step asks what they cluster *with* (their
  co-members' copies, classes, positions), against step 0 as the baseline. That
  is a new reader, not this table. How much of the copy effect is the
  instrument's: §1.8 (at step 0 layer 0 all of it is).
- **Caveats.** No null. The first two tables average layers.
  The HDBSCAN floor (§3, ARI p5 0.347) applies to every rate. Rank is merge
  order, not corpus frequency. `repeated_tokens` has no unique word starts, so
  the unique contrasts rest on 7 prompts. The pilot sweep agrees to ≤ 0.004
  at every shared step (§1.11).

### 1.8 Two checks on §1.7 — **at step 0 layer 0 the partition is what HDBSCAN makes of noise with duplicates; the carrying capacity is mostly a repeat count at layer 0**

Tier 1, descriptive, no null. Both were parked by `/challenge-pr` on #90.

**(a) Known answer: duplicates planted in Gaussian noise.**
`tools/run/p10_hdbscan_planted.py`, record `data/analysis/p10_hdbscan_planted.json`.
**Input:** 250 background points N(0, I₁₀₂₄) plus 40 groups of 2 exact copies,
10 of 3, 6 of 4 and 4 of 5 (≈ 404 points), seeds 0–19, partitioned by the exact
call `clustering.py` makes (cosine, `min_cluster_size=2`). Conda `mets`,
hdbscan 0.8.41. Step 0's embeddings are a random Gaussian init, so this is close
to the real layer-0 input.

| | planted (20 seeds) | real, step 0 layer 0 (§1.7) |
|---|---|---|
| 2-copy groups: noise | 0.141 (113/800) | 0.13 (26/205) |
| 3–5-copy groups: noise | 0.000 (0/400) | 0.00 |
| singletons clustered | **0.43** (2159/5000 background) | **0.43** (unique tokens) |
| clusters / groups of ≥ 2 | 1.09 (65.3 / 60) | 0.99 (below) |

- **The step 0 layer 0 rates are the instrument's.** Twins labelled noise,
  every 3+-copy group clustered, and 43 % of singletons clustered all come out
  of structureless noise at the rates §1.7 measured.
- **HDBSCAN at `min_cluster_size=2` clusters structureless points.** The known
  answer for the background is 0; it gets 0.43. Of 1307 clusters, 902 hold
  exactly one planted group, 88 hold several, and 317 hold background points
  only (207 of those are pairs). A planted group is its own cluster in only 271
  of 1200 cases; 806 absorb 1–16 background points, and 10 are split between
  labels despite identical coordinates (ties).
- **Pure noise gives no stable count.** The second arm (404 points, nothing
  planted) gives 2–55 clusters per seed, mean 14.7: a few large clusters, not
  a floor near 50. Duplicates are what make the count track the groups.
- **So step 0 is the right baseline for the next reader and cannot be skipped:**
  a clustered unique token at init is a noise point glued to a copy group.

**(b) Cluster count vs repeated token types.** New columns in
`tools/run/p10_token_composition.py` (the §1.7 record). **Input:** the 152 v1 runs,
inputs `c558b210c08f`. Per run per layer: HDBSCAN cluster count, token types
with ≥ 2 copies (a property of the prompt, the same at every checkpoint), and
the share of clusters holding ≥ 2 copies of some token. Means over the 7
prompts other than `repeated_tokens` (59.7 repeated types per prompt):

| step | L0: clusters / repeated types · hold a repeat | L12 | L24 |
|---|---|---|---|
| 0 | 0.99 · 0.89 | 0.85 · 0.91 | 0.83 · 0.87 |
| 512 | 0.98 · 0.87 | 0.75 · 0.83 | 0.72 · 0.82 |
| 2000 | 0.96 · 0.87 | 0.84 · 0.76 | 0.61 · 0.77 |
| 16000 | 0.97 · 0.89 | 0.79 · 0.74 | 0.51 · 0.73 |
| 143000 | 0.93 · 0.89 | 0.79 · 0.74 | 0.59 · 0.70 |

- **At layer 0 the cluster count tracks repeated types at every checkpoint**:
  the 7-prompt mean ratio is 0.92–1.01 over all 19 steps. Per run it is
  0.78–1.24, so "tracks", not "equals". At every layer and step, ≥ 63 % of
  clusters hold a repeat (minimum: step 143000, L19).
- **Training lowers the deep-layer count** (L24: 0.83 at step 0, 0.46–0.59 from
  step 16000). It does not raise it: the trained model merges or dissolves copy
  groups rather than forming clusters without copies.
- **`repeated_tokens` is a different mechanism, not evidence against.** Its
  text has 1 repeated token type; HDBSCAN finds 2 clusters at L0 (ratio 2.00 at
  every step) but 52 at L12 and 55 at L24 at step 0 (43 / 42 at step 143000).
  Its `max_alive` is 41–65 and peaks anywhere from L2 to L22, so it is neither
  fixed nor at the embedding. Something else (position is the first suspect)
  makes ~50 clusters there; pure noise does not (arm 2 of (a)).
- **Phase 1's carrying capacity, on these runs, is mostly a repeat count taken
  at the embedding layer** (measured after `/challenge-pr` on #91 pointed out
  the record already held it). `max_alive` (`cluster_tracking.py:264`) is the
  most clusters at any one layer, from these same labels. Over the 7 prompts it
  falls at L0 (the embedding lookup, before any attention) in 92 of 133 runs,
  and its mean is 57.0–63.0 at every one of the 19 steps, against 59.7 repeated
  types per prompt, which training cannot change. So its invariance across
  training is what a repeat count predicts, and Lemma C.1's formula
  (`math-10.md` §5.4) is not needed to explain it here. **Still open:** the
  "50–55" came from `max_alive` on the pilot's 27 checkpoints, and what it
  averaged over is not recorded; on these runs the 8-prompt mean is 56–63.
  Whether the pilot behaves the same is one pass over its labels.
- **Caveats.** No null. The HDBSCAN floor (§3) applies. Means over 7 prompts
  hide their spread. "Holds a repeat" does not mean "is a copy group": at L0
  only 0.33–0.49 of clusters are a single token type.

### 1.9 What clustered unique tokens cluster with — **trained, their own class and away from copy groups; embedding similarity is mostly there at init**

Tier 1, descriptive, no null. Reader `tools/run/p10_comembership.py --v1-only`,
record `data/analysis/p10_s1_comembership.json`. **Input:** the 152 v1 runs
through `stage0_index.json` (pin `64a4087`, battery `06790b90dcfe`), inputs
`c558b210c08f`, tokenizer.json `c24618a1b3e6`, native labels only. Record
schema 2, after `/challenge-pr` on #92 (below).

Focal token: a clustered unique token (one copy in the prompt) at position > 0.
For each, properties of its co-members against a uniform random draw of the
same size (exact expectations): from the rest of the prompt (pre-stated), and
from its other clustered positions (`_cl`, added). Lift = observed − expected,
per run-layer, then the mean over prompts. The reading, fixed in the docstring
before the first run: Δ = lift(step) − lift(step 0), "above" or "below" step 0
past ±0.05. `repeated_tokens` has 1 unique token, never clustered, so 7
prompts contribute.

**What the review changed.** The first version's headline property, `emb_pct`,
scores co-members in the *trained* layer-0 embedding (the frozen frame). Step
0's clusters come from an unrelated random embedding, so their lift there is
~0 by construction, and the Δ only tracks the embedding moving toward its
final form. Its Δ is now not read. `emb_pct_own` scores them in each run's
**own** layer 0, which has a valid step-0 baseline. At layer 0 it is circular
(the partition was made from that geometry). The first version's reading ("trained, they cluster with tokens the
trained embedding calls similar, emerging over steps 64–2000") was that
artifact, and it is withdrawn.

Lift at layer 0 / 12 / 24 · mean over 25 layers, all-positions draw. Expected at L12: copy_share 0.65, no_copy ≈ 0.13, adjacent 0.02, same_class ≈ 0.51, emb_pct_own 0.50.

| step | same_class | copy_share | no_copy | adjacent | emb_pct_own |
|---|---|---|---|---|---|
| 0 | +0.04 / +0.07 / +0.02 · +0.06 | −0.05 / −0.02 / +0.00 · −0.03 | +0.15 / +0.14 / +0.11 · +0.13 | +0.02 / +0.10 / +0.10 · +0.08 | +0.39 / +0.22 / +0.15 · +0.20 |
| 64 | +0.04 / +0.19 / +0.22 · +0.17 | −0.06 / −0.03 / −0.13 · −0.05 | +0.16 / +0.10 / +0.21 · +0.13 | +0.00 / +0.02 / +0.02 · +0.04 | +0.39 / +0.11 / +0.05 · +0.14 |
| 512 | +0.07 / +0.33 / +0.33 · +0.33 | −0.07 / −0.19 / −0.20 · −0.17 | +0.18 / +0.28 / +0.21 · +0.24 | +0.02 / +0.04 / +0.09 · +0.06 | +0.39 / +0.10 / +0.11 · +0.15 |
| 2000 | +0.29 / +0.31 / +0.28 · +0.32 | −0.13 / −0.23 / −0.17 · −0.21 | +0.23 / +0.35 / +0.26 · +0.32 | −0.01 / +0.26 / +0.17 · +0.17 | +0.43 / +0.34 / +0.21 · +0.32 |
| 16000 | +0.30 / +0.29 / +0.24 · +0.30 | −0.13 / −0.24 / −0.13 · −0.17 | +0.25 / +0.33 / +0.18 · +0.28 | −0.01 / +0.20 / +0.19 · +0.14 | +0.46 / +0.34 / +0.17 · +0.35 |
| 143000 | +0.29 / +0.26 / +0.23 · +0.27 | −0.13 / −0.19 / −0.08 · −0.15 | +0.25 / +0.31 / +0.12 · +0.26 | −0.01 / +0.20 / +0.19 · +0.13 | +0.47 / +0.33 / +0.17 · +0.32 |

Δ at step 143000, layer mean, all-positions draw / clustered draw: same_class
+0.20 / +0.20, copy_share −0.12 / −0.06, no_copy +0.13 / +0.11, adjacent
+0.04 / +0.04 (L12: +0.10 / +0.10), emb_pct_own +0.12 / +0.11 (L12 +0.10, L24
+0.02). Share of unique tokens clustered (layer mean): 0.21 at step 0, 0.30 at
step 143000. Distinct clusters behind the focal tokens: 16–21 per run-layer at
the layer mean, so no single cluster carries a cell.

- **At init, clusters at depth follow each token's own embedding.**
  Step 0's `emb_pct_own` lift is +0.22 at L12 and +0.15 at L24 (L0 is
  circular). The embedding is random, so this is the residual stream carrying
  the token's own vector, not meaning. Copy share is at chance (−0.03), and
  24 % of focal tokens sit in a cluster with no copies, which planted noise
  also makes (§1.8(a): 317 of 1307 clusters are background only).
- **Training adds class, the clearest effect.** `same_class` Δ is +0.20 at
  the layer mean, and the same under the clustered draw, so it is not
  §1.7's class-predicts-clustered effect leaking into the expectation. Per
  prompt it varies widely (+0.03 to +0.60 at L12, step 143000). It moves
  first: +0.19 at L12 by step 64, full size by step 512.
- **Training moves them away from copy groups.** Copy share falls below
  chance (Δ −0.12; −0.06 under the clustered draw, whose pool is copy-heavy),
  and clusters of unique tokens only rise (Δ +0.13 / +0.11).
- **Embedding similarity adds a little.** `emb_pct_own` rises from +0.20 to
  +0.32 at the layer mean (Δ +0.10 at L12, +0.02 at L24). At L12 Δ is
  positive in 6 of 7 prompts (−0.07 in `hdbscan_code`). It dips at steps
  64–512 (+0.10 at L12) and comes back by step 2000, when the embedding
  settles (§1.6). Class and embedding similarity are not separated.
- **Position is a depth-only part.** Adjacent co-members: Δ +0.10 at L12 and
  +0.09 at L24, none at L0, near the floor at the layer mean. At L12, 22 % of
  focal tokens have a neighbour in their cluster against 2 % by chance, so
  most co-members are not neighbours.
- **Caveats.** No null beyond the random draw; no e-value. The ±0.05 floor is
  placed, not calibrated. At trained L24 one cluster holds most of the prompt
  in some runs (mean k 76 at step 143000; `homer_iliad` has 190 focal tokens
  there), so their lifts are near 0 and the L24 column is weaker than it
  looks. Whether trained depth clusters are still the token's own embedding
  (lexical) or contextual is not answered here (§1.10 answers part of it).
  The HDBSCAN floor (§3) applies. Stage 0's v1 runs; on the pilot sweep, §1.12.

### 1.10 §1.9's class effect, lexical or not — **early in training the network computes it; trained, it is mostly what the embedding already groups**

Tier 1, descriptive, no null. Reader `tools/run/p10_lexical_carry.py --v1-only`,
record `data/analysis/p10_s1_lexical_carry.json` (schema 4, after
`/challenge-pr` on #93 and #94). **Input:** as §1.9: 152 v1 runs through
`stage0_index.json`, pin `64a4087`, battery `06790b90dcfe`, inputs
`c558b210c08f`, tokenizer `c24618a1b3e6`. 4 min on 16 cores. Same focal tokens
and all-positions draw as §1.9. Both checks were parked in `handoff-10.md`.
"Class" is §1.7's orthographic category (word start, continuation, punct,
numeric, whitespace, byte fragment), not meaning. Word start vs continuation
depends partly on the previous token.

- **Carry** (`self_pct`): the percentile of cos(x_L[i], x_0[i]) among
  cos(x_L[i], x_0[j]). This removes the common direction. Raw `self_cos` is
  0.05 at L12, trained.
- **Split**: `class_given_emb` is the same-class lift with each co-member
  redrawn from its own-embedding similarity bin, so class is measured beyond
  similarity. `emb_given_class` is the reverse: similarity with each
  co-member redrawn within its own class, so similarity is measured beyond
  class.

**Post hoc, and it decides the reading.** The pre-stated control used
deciles. It cannot work: a purely lexical cluster of the same size at the same
layer (each focal token's k nearest in its own layer 0, `*_knn`) scores +0.21
at trained L12 under it, more than the observed +0.15. At 40 bins that lexical
control is +0.04 at L12. The observed value is read against it, and beside a
class-only cluster's expected score (`*_classonly`: same-class tokens, the
embedding ignored; a reference, not a maximum: one run exceeds it, and per
prompt it runs 0.04–0.44). The pre-stated 10-bin reading ("above step 0",
Δ +0.13 at the layer mean) is recorded but does not settle it. Trained L0, the
first reference used, agrees with the kNN control at 40 bins (+0.03 / +0.03).

`class_given_emb` at 40 bins: observed / lexical control (kNN) / class-only
reference. Step 0 is observed −0.04 to +0.01, kNN −0.04 to −0.01, class-only
0.28–0.30 at every layer. 10 bins, observed / kNN, in brackets:

| step | L12 | L24 | layer mean |
|---|---|---|---|
| 512 | **+0.20** / −0.00 / 0.32 (+0.28 / +0.04) | **+0.20** / −0.00 / 0.30 (+0.27 / +0.05) | +0.20 / +0.00 / 0.31 |
| 2000 | +0.12 / +0.04 / 0.27 (+0.20 / +0.14) | +0.13 / +0.02 / 0.30 (+0.19 / +0.11) | +0.13 / +0.04 / 0.28 |
| 143000 | **+0.04** / +0.04 / 0.26 (+0.15 / +0.21) | **+0.07** / +0.02 / 0.26 (+0.15 / +0.13) | +0.05 / +0.03 / 0.25 |

Carry, focal (clustered unique) / unclustered unique, `self_pct` · `self_top1`,
over the 7 prompts with both groups (`repeated_tokens` has no focal token):

| step | L12 | L24 |
|---|---|---|
| 0 | 0.97 / 0.97 · 0.43 / 0.44 | 0.90 / 0.91 · 0.20 / 0.20 |
| 512 | 0.82 / 0.82 · 0.08 / 0.12 | 0.66 / 0.65 · 0.02 / 0.02 |
| 143000 | 0.84 / 0.88 · 0.13 / 0.20 | 0.61 / 0.72 · 0.00 / 0.05 |

`emb_given_class` Δ at step 143000: +0.07 L12, −0.02 L24, +0.08 mean. `emb_same`
Δ +0.11 L12, `emb_cross` Δ +0.02.

- **Step 512: the network computes the class grouping.** The embedding
  barely groups by class yet: §1.9's same_class lift at L0 is +0.07, and a
  lexical kNN cluster scores +0.04 / +0.05 at L12 / L24 under deciles, against
  +0.21 / +0.13 trained. Depth clusters are +0.20 same-class beyond embedding
  similarity at 40 bins, about two thirds of the class-only reference (0.31). This check cannot
  say whether the source is context or a per-token feature computed after
  layer 0.
- **Trained: mostly what the embedding already groups.** At 40 bins, L12's
  +0.04 is on the lexical control (+0.04). L24's +0.07 is +0.05 above its
  control (+0.02), on the floor. Taken the pre-stated way, each against its
  step-0 value, the L24 excess is +0.04, inside the floor. The class-only
  reference stays 0.26, so the measure could have shown more. Two to three of 7 prompts carry the L24 excess
  (`latex_monograph` +0.23, `hdbscan_code` +0.16, `camus_letranger` +0.08; the
  rest −0.03 to +0.01).
- **Not individual carry, which is a narrower claim.** Clustered unique tokens
  keep less of their *own* layer-0 direction than unclustered ones at trained
  depth (`self_pct` −0.04 at L12, −0.10 at L24; no gap at steps 0 and 512).
  `self_pct` does not look at class, so it cannot rule out tokens carrying a
  direction their class shares in the embedding. That would still be
  lexical, and the kNN control above is consistent with it.
- **Within class, embedding neighbours at mid-depth only.** `emb_given_class`
  Δ is above step 0 at L12 (+0.07) and as step 0 at L24. The embedding
  preference is among same-class co-members (`emb_same` +0.11, `emb_cross`
  +0.02).
- **Caveats.** Binning always leaves some similarity inside a bin, and each
  co-member's own s shrinks a 40-bin lift by ~10 %. "Beyond the own layer-0
  vector" is not "contextual". Position, attention and per-token computed
  features are not separated. A context-shuffle test would separate them
  (Parked in `handoff-10.md`). The floor, no null and HDBSCAN (§3) apply as in
  §1.9. The first version's carry table averaged 7 prompts against 8 and is
  replaced; its "17 %" was 5 % on matched runs.

### 1.11 Stage 1 steps 1 and 2 on the pilot sweep — **the pilot agrees with Stage 0 to ≤ 0.004 at every shared step; its 14 extra steps fill the gaps smoothly**

Tier 1, descriptive, no null. `handoff-10.md` §1.3 step 3. **Inputs:** the pilot
sweep `HDD_1TB/Mets_archive/2026-08-12_05-01-35` (native labels, battery
`1e47918ef77a`, git `3726289`), read with `--run-root` and `--prompts` set to Stage 0's
8 v1 keys: 8 × 27 steps = 216 runs, inputs sha256 `c183a7fcbd9e`, beside Stage 0's
152 v1 runs (`c558b210c08f`, §1.6–§1.7). Records `data/analysis/p10_s1_ext_sem_threshold_pilot.json`,
`p10_s1_token_composition_pilot.json`. Side by side: `tools/run/p10_s1_compare.py`
(prints every row; it reproduces §1.6 and §1.7's Stage 0 columns exactly; its
`--raw` mode produces every label and activation number in this section). Every
pilot run reproduced its stored `n_ext_semantic` at 0.5.

**The inputs are the same, so only the labels can differ.** On the 104 (step,
prompt) runs both sweeps hold (13 steps × 8 prompts), `tokens.txt` is identical
and activations differ by ≤ 2.5e-7 up to step 1000 and ≤ 7.9e-5 at step 143000.
Stage 0's native labels are **bit-identical to the WDS backfill's** in all 3 800
layer-records of the 152 v1 runs, and so are its activations. So "the WDS sweep"
and Stage 0 are one measurement, and the pilot is the only second one. Pilot vs
Stage 0 labels: 2 165 of 2 600 layer-records identical (104/104 at layer 0,
79–90 at every other layer), ARI p5 0.347, min 0.166, which is §3's floor on the
same pairs. `max_alive` is equal in 92 of 104 runs.

| on the 13 shared steps, max \|pilot − Stage 0\| | value |
|---|---|
| §1.6: repeat share, `ext_semantic_fraction` @0.5, non-repeat cos > 0.2, mean percentile | 0.000 |
| §1.6: same-cluster among repeats | 0.002 |
| §1.7: noise rate, unique, 2 / 3–5 / 6–20 copies, unique by rank and class | ≤ 0.003 |
| §1.7: unique freq / class contrast | 0.001 / 0.004 |
| pre-stated verdicts (frozen / self) | dead / mixed in both |

Pilot-only steps (3000, 5000 … 19000, 40000 … 120000), so the 14 not in Stage 0:

| step | repeat share | nr cos > 0.2 | unique clustered | 3–5 copies | unique punct | unique class contrast |
|---|---|---|---|---|---|---|
| 3000 | 0.751 | 0.451 | 0.269 | 0.884 | 0.229 | −0.045 |
| 9000 | 0.730 | 0.468 | 0.243 | 0.847 | 0.179 | −0.067 |
| 19000 | 0.731 | 0.518 | 0.224 | 0.830 | 0.139 | −0.092 |
| 60000 | 0.718 | 0.510 | 0.240 | 0.809 | 0.147 | −0.100 |
| 120000 | 0.695 | 0.487 | 0.247 | 0.793 | 0.144 | −0.116 |

- **§1.6 and §1.7 hold on the independent sweep.** Every Stage 0 number they
  quote is within 0.004 on the pilot, and the pilot's extra steps sit between
  Stage 0's neighbours: the repeat share keeps falling slowly after step 4000
  (0.74 → 0.70), non-repeat similarity is flattest at steps 17 000–40 000 (0.52–0.53),
  and unique punctuation falls from 0.23 at step 3000 to ~0.14 by step 15 000.
  The class contrast goes negative after step 2000 and stays there.
- **What the agreement tests, and what it does not** (after `/challenge-pr`
  on #95). Four of §1.6's five columns never read a label (they read the
  mutual pairs and the embedding), so they could not have differed. The label
  columns (same-cluster among repeats, all of §1.7) average over layers. The
  partitions differ in 17 % of layer-records, but the differences are small
  (mean |Δ n_clusters| ≤ 0.8 per layer), none are at layer 0, and every column
  is a mean over thousands of tokens. §3.1 found the same for A0 and F0. The
  per-layer claims, which are §1.9–§1.10's, are **not** tested here.
- **§1.1's source table is 9 prompts, not 8.** Run over all 9 pilot prompts
  (`short_heterogeneous` included; inputs `1b32908600b1`,
  `p10_s1_ext_sem_threshold_pilot_all9.json`), the reader reproduces every
  §1.1 value: 0.833 / 0.856 / 0.788 / 0.752 / 0.692 / 0.723 / 0.708, and the
  same-cluster column. `handoff-10.md` §1.1 said 8. The 8-prompt pilot starts at
  0.875, as Stage 0 does.
- **The floor is HDBSCAN moving under float noise** (corrected after
  `/challenge-pr` on #95; the first version said the floor was "between the
  pilot and today", which was wrong). At steps 0–1000 the two sweeps'
  activations differ by ≤ 2.5e-7, about one float32 rounding step, and 24–60
  of each step's 200 layer-records still differ. The toolchain did not change:
  the backfill re-clustered the pilot's own activations with today's code and
  matched its labels (§2). That today's pipeline repeats itself bit for bit
  (Stage 0 vs the WDS backfill) shows only that it is deterministic on this
  machine. A change of thread count, library build or hardware would bring
  the floor back.
- **Caveats.** No null. The pilot's 9th prompt is left out of the tables so the
  step means compare. The co-membership and lexical-carry readers (§1.9–§1.10)
  carry the per-layer claims; §1.12 runs them on the pilot.

### 1.12 §1.9–§1.10 on the pilot sweep — **the per-layer claims hold on the second partition: 3 103 of 3 120 and 383 of 384 readings agree**

Tier 1, descriptive, no null. `handoff-10.md` Parked (from `/challenge-pr` on
#95). **Inputs:** the pilot as in §1.11 (216 runs, inputs `c183a7fcbd9e`,
`--run-root` + `--prompts` = the 8 v1 keys), tokenizer `c24618a1b3e6`,
against Stage 0's §1.9–§1.10 records (`c558b210c08f`). Records
`data/analysis/p10_s1_comembership_pilot.json`, `p10_s1_lexical_carry_pilot.json`
(1 min 52 s and 5 min 53 s on 16 cores). Side by side and agreement:
`tools/run/p10_s1_compare.py --per-layer`. A reading is §1.9's "above / below /
as step 0" at ±0.05 on Δ = lift(step) − lift(step 0), counted over the 12
shared steps > 0. For §1.9 it is re-derived from the lifts at all 25 layers and
the mean, because the record's `reading` holds only L0/L12/L24/mean; §1.10's
record holds those four layers only.

| | readings agree | max \|Δ value\|, pilot vs Stage 0 |
|---|---|---|
| §1.9: 5 properties × 2 draws × 26 layers × 12 steps | 3 103 / 3 120 | ≤ 0.021; copy_share and no_copy up to 0.052 |
| §1.10: 8 quantities × 4 layers × 12 steps | 383 / 384 | ≤ 0.020 |

- **The numbers §1.9 and §1.10 quote are the same on the pilot, at the steps
  both sweeps have.** Shared: 0, 1, 2, 4 … 1000 and 143000. The steps §1.9 and
  §1.10 also quote (2000, 16000) are not in the pilot. At steps 0, 64, 512 and
  143000, every quoted cell is within 0.005: same_class lift
  +0.29 / +0.26 / +0.23 at L0 / 12 / 24, `class_given_emb` at 40 bins +0.20 at
  step 512 against kNN ≈ 0, trained L12 +0.04 on +0.04, L24 +0.07 on +0.01–0.02,
  carry `self_pct` 0.84 / 0.88 at L12 and 0.61 / 0.72 at L24. 11 of the 12
  shared steps > 0 are ≤ 1000, so the trained-model claims get one step
  (143000) on the second partition.
- **Most of the agreement is identical cells.** 100 of 2 600 run-layers (13
  steps × 8 prompts × 25 layers; none in `repeated_tokens`) have a different
  partition; the rest are bit-identical. Where one does change, a single prompt's value moves by up to
  0.36 (`latex_monograph`, step 32, L12, `copy_share`). The per-prompt numbers
  §1.9–§1.10 quote hold within 0.027 at 143000 (`/challenge-pr` on #96).
- **The 18 disagreements are threshold flips.** In 17 both Δs are within
  0.011 of ±0.05. The one real gap is step 32 L12 (no_copy Δ −0.071 vs −0.030;
  copy_share_cl and emb_cross flip in the same cell). Step 32 is the least
  stable step (`/challenge-pr` on #96): 29 run-layers differ there, in 16 of 25
  layers, against 15 at step 16, 13 at step 64 and ≤ 8 at every other shared
  step. It is inside §1.9's 32–512 window and the
  32–64 window §1.3 and §1.5 rest on (parked, `handoff-10.md`).
- **§1.10 at all 26 layers** (`/challenge-pr` on #96, the reviewer's own code
  over the records): 7 disagreements in 2 184 cells, all at the floor. The
  table above counts the 4 layers the record's reading holds.
- **The pilot's extra steps fill in §1.10's decline.** `class_given_emb` at 40
  bins, layer mean: 0.20 at step 512, 0.09 at 3000, 0.055 at 13 000–19 000,
  0.05 from 40 000 on; the kNN control is 0.015–0.03 throughout. L24 is
  0.06–0.10 over steps 3000–120 000 against kNN 0.005–0.04, so its "+0.05 over
  the control, on the floor" reads the same on both sweeps.
- **What this shows and what it does not.** It is the same activations
  clustered twice (§1.11), so it measures how far HDBSCAN's float-noise
  instability moves these claims, and here it moves them little. It is not a
  second model, a second prompt set or a null. The ±0.05 floor stays placed,
  not calibrated.

### 1.13 F1 and F12 on the pilot sweep — **the 32–64 window holds on the second partition**

Tier 1, the same statistics and nulls as §1.3–§1.4 (2 000 permutations).
`handoff-10.md` Parked (from `/challenge-pr` on #96). **Inputs:** the pilot's
216 v1 runs, the same set as §1.11 (inputs `c183a7fcbd9e`: 8 v1 keys × 27 steps
of `HDD_1TB/Mets_archive/2026-08-12_05-01-35`, native labels). Git `10e44ea`,
`--v1-only`. `transport.py` and `p10_partition_function.py` take only
`--root`/`--pattern`, so they read a symlink root, which also keeps Stage 0's
dirs out (handoff Parked, "default glob"). The records' `root` is that
throwaway path. Rebuild it and re-run:

```bash
P=/run/media/system/HDD_1TB/Mets_archive/2026-08-12_05-01-35; R=<scratch>/pilot_v1/$(basename $P)
mkdir -p $R; for d in $P/pythia-410m-step*/; do case $d in *short_heterogeneous/) ;; *) ln -sfn ${d%/} $R/;; esac; done
<env as §0> python tools/run/transport.py --root <scratch>/pilot_v1 --v1-only --out data/analysis/p10_f1_transport_pilot.json
<env as §0> python tools/run/p10_partition_function.py --root <scratch>/pilot_v1 --v1-only --out data/analysis/p10_f12_z_pilot.json
```

Records `data/analysis/p10_f1_transport_pilot.json` (5 184 boundaries, 9 min) and
`p10_f12_z_pilot.json` (16 200 units, 21 min, 16 cores), beside §1.3–§1.4's
`p10_f1_transport.json` and `p10_f12_z.json` (152 runs, the same 8 prompts).
Per-unit counts below drop step 1 (step 0's weights, §3.51.3) and count only
units whose value differs between the sweeps: the rest are identical and
cannot disagree.

| on the 12 distinct shared steps | F1 clustered − noise step | F12 clustered − noise |
|---|---|---|
| max \|pilot − WDS\|, per-step mean | 0.005 | 0.008 |
| step 32, WDS / pilot | −0.294 / −0.290 | −0.122 / −0.118 (vs step 0: −0.587 / −0.588) |
| step 64, WDS / pilot | −0.317 / −0.318 | −0.141 / −0.145 (vs step 0: −0.606 / −0.615) |
| units whose value differs, same sign | 361 / 366 (step 32: 50 / 50) | 1 126 / 1 126 (step 32: 159 / 159) |
| largest single-unit move | 0.37 (step 64) | 0.60 (step 64) |

- **§1.3's and §1.5's numbers hold on the second partition.** At step 32, the least
  stable step between the partitions (§1.12), no changed unit flips sign in either
  row. Single units move by up to 0.4–0.6; the per-step means do not.
- **Per-unit p is not compared.** Both runners draw every null from one
  generator seeded once per sweep, so a unit's p depends on how many directories
  ran before it; most p disagreements are in units with bit-identical
  statistics (`/challenge-pr` on #97). Parked in the handoff.
- **Below the step-0 `Z` baseline lasts past 512 on both sweeps**, which §1.4's
  table left out (now added there). WDS: −0.10 / −0.09 / −0.13 at 1000 / 2000 /
  4000, +0.004 at 8000. Pilot: −0.19 at 3000, −0.13 at 5000, −0.04 at 7000,
  +0.04 at 9000, −0.05 at 11 000, then +0.06 to +0.21 from 13 000. F1's weaker
  late dip was in §1.3's 16000–32000 row; the pilot's 13 steps from 3000 to
  100 000 all sit between +0.012 and −0.153, then +0.044 at 120 000 and −0.017
  at 143 000.
- **What this shows.** The same activations clustered twice (§1.11): it bounds
  HDBSCAN's float-noise instability on these two rows, and it is small. It is
  not a second model, a second prompt set, or the missing control.

---

## 2. The blocker that had to be cleared first

**The 410m sweep had no density partition at all.** `hdbscan_labels.json` is
`{}` in **152 of 152** directories and `clustering.json` beside it reads
`"nesting_summary": "HDBSCAN not available"`. The outage `PROJECT.md` §3.41
records was never only `CLAIM-C`'s arms, and `docs/AXES.md` listed the artifact
as present. F0, F5 and F11-A4 all read it.

**A fourth row was hit and nobody noticed — found 2026-09-20, `handoff-10.md`
§0.2.** `pair_agreement` (`pair_hdbscan_agreement`, the project's **only**
semantic instrument: mutual-NN pairs tagged against the embedding Gram) is
computed only when `"labels" in hdb_data`. Through the outage that branch was
never taken, and the `else` wrote a **well-formed record of zeros and nulls**
into all 152 directories rather than failing. A silent zero record looks exactly
like a real one, which is why it survived every check. **The pilot sweep's copy
is populated — 6 066 of 6 075 layer-records — and had never been reported in any
markdown file in this repository.** Same bug class as `docs/AXES.md` listing an
absent artifact as present, and as standing rule 4.

`tools/run/backfill_hdbscan.py` re-derives it from `activations.npz` — no
forward pass, **152 directories in 69 s**. **It does not re-run the analysis**,
by design (item 3 below), so the WDS sweep still carries no semantic record. Mean 47.1 clusters/layer, mean noise
fraction 0.389, in line with the pilot's 45–69 and finding 4's 50–55.

Three properties make it usable rather than merely fast:

1. **It guards on the TOOLCHAIN, not the interpreter path** — the opposite of
   every other runner here, and correct for this one. `clustering.py` records,
   measured, that HDBSCAN's output is a property of the install: conda `mets`
   (py3.10.20, hdbscan 0.8.41, sklearn 1.7.2) reproduces the pilot sweep
   exactly and `.venv` does not.
2. **It re-verifies on every invocation** — `--verify-pilot` replays N pilot
   directories that *do* carry labels and refuses to write unless they come
   back bit-identical. This run: 3 directories × 25 layers, clean.
3. **It writes a separate file and touches nothing.** Filling the canonical one
   would leave `clustering.json` still saying `null` beside it — the shape of
   inconsistency `b55375e` had to un-write. `read_labels` owns the precedence.

---

## 3. The measurement that was not on the ladder

**The HDBSCAN partition is not reproducible run to run** — *confirmed
2026-09-25 (§1.11): up to step 1000 the activations differ by ≤ 2.5e-7, and
24–60 of each step's 200 layer-records still differ. Today's pipeline repeats
itself bit for bit only because it is deterministic on one machine.* Two independent
Phase-1 sweeps cover the same checkpoints and prompts; their `tokens.txt` are
identical in all 104 overlapping directories and their activations differ by at
most **7.9e-05**. Over **2 600 layer-pairs**:

| | value |
|---|---|
| label vectors identical | **83.3 %** |
| ARI median / mean | 1.0 / 0.933 |
| **ARI 5th percentile / minimum** | **0.347 / 0.166** |
| ARI noise-dropped, p05 / min | 0.585 / 0.327 |
| cluster-count \|Δ\| mean / **max** | 0.58 / **20** |

**A measurement-reproducibility floor, not a null** — no hypothesis, no
p-value, by design. No null this project has built accounts for it:
`notes-10.md` §4.4's size-profile null is about ARI's variance under random
*labelling*, a different quantity from its variance under *re-measurement*.

**What it bears on.** `CLAIM-C` reads `cluster_count` and `cluster_membership`
from HDBSCAN and nothing else, and §3.41 scored them 2/8 and 5/8 — the two
weakest of six. **It does NOT follow that the gate's result is noise**: those
arms differ by model and training, not by a re-run, and this measures only the
re-run. What follows is that the comparison has never been made.

### 3.1 Both headline rows were re-run against it, and both hold

| | WDS (backfilled) | pilot (native) |
|---|---|---|
| directories / units, A0 | 152 / 3 646 | 243 / 5 593 |
| **A0** corrected gap, sweep mean | 0.037 | 0.100 |
| **F0** nucleus position | **0.3164** | **0.3157** |
| F0 ordinary / restricted null mean | 0.2077 / 0.2311 | 0.2053 / 0.2220 |
| F0 median p, ordinary / restricted | 1.00 / 0.9995 | 1.00 / 0.9995 |
| F0 merged E, ordinary / restricted | 0.536 / 0.556 | 0.550 / 0.569 |

A0's sweep means differ because the pilot's checkpoint grid is weighted toward
late training, where §1.1 already showed the residual lives.

A0's per-checkpoint corrected gap: 0.004/0.004 at step 0, −0.072/−0.055 at 256,
−0.164/−0.121 at 512, **0.172/0.230 at 143000**. F0 agrees to **three decimal
places** on the statistic itself — and the pilot finds **fewer** clusters per
layer (42.8 against 47.1), so the agreement is not an artifact of a matching
size profile.

**The floor still binds any PER-LAYER claim.** What these checks establish is
that a statistic aggregated over thousands of units is far less exposed to it.

---

## 4. Ladder status

| row | state |
|---|---|
| **F0** anchor test | **RUN**, both sweeps. Fails in the predicted direction |
| **F0b** slope test | not run — needs the `beta_eff` producer (`AXES.md` §4) |
| **F1** transport | **RUN** |
| **F2** verify J-lens artifacts | not run — needs HF access from this machine |
| **F3** fit a J-lens | blocked on F2 |
| **F4** per-layer functional partition + ARI | blocked on F3 |
| **F5** four-signature concordance | **the phase's central test.** Blocked on F4 |
| **F6** `turnover_decomposition` rebuilt | not run. No new experiment — a groupby |
| **F7** centroid substitution | blocked on F4 |
| **F8** operator decomposition of `J_l` | blocked on F3 |
| **F9** MLP object collapse | not run; needs a forward pass |
| **F10** neuron-basis collapse | blocked on F4 |
| **F11** attention audit | **A0 RUN.** A1–A8 not run; A0 gated them and has now cleared |
| **F12** `Z_beta,i` per token | **RUN** |
| **F13** the centre scan (Rényi + strong Rényi, swept in `δ`) | **new 2026-09-20.** Free, unblocked, and the row F0 was a proxy for |
| **F14** observed count vs Lemma C.1 | new. Free; needs F13 |
| **F15** the coverage curve (the paper's Fig. 3, and its own open problem) | new. Free; needs F13 |
| **F16** intrinsic dimension per layer, as the `d_eff` candidate | new. Free |
| **F17** the graded block-shuffle null | new. Free |
| **F18** `V`-spectrum atlas against Table 1 | new. Free |
| **F19** depth-axis merge intervals and splits | new. Free; needs the particle table |
| **F20** frozen-centre intervention (Thm 5.2) | new. **Forward pass**; needs F13 |

---

## 5. What a next session should do, in order

1. **`CLAIM-C`'s two HDBSCAN metrics against the reproducibility floor** (§3).
   **The only open item here that bears on a *registered* prediction**, which
   is why it is first. `tools/run/p10_partition_stability.py` supplies one side
   — the amount a partition moves when the same measurement is re-run — and
   what is missing is the other: how far that gate's four arms actually differ
   on `cluster_count` and `cluster_membership`. §3.41 scored them 2/8 and 5/8,
   the two weakest of six. Free. **This ordering matches `docs/AXES.md` §7.**
2. **F11 rows A1–A8.** A0 gated them and A0 is done. `attention-10.md` §6 rates
   **A2** (the checkpoint axis) and **A4** (the population×population mass
   matrix) highest, and calls A4 a direct H-PARK vs H-CAT test. Free.
3. **F6, `turnover_decomposition`.** Validated on synthetic sweeps in 2026 and
   awaiting real data ever since; a rebuild against `core/particles.py`, not a
   lift (`archive/README.md` rule 2). No forward pass.
4. **A norm-matched random twin per checkpoint**, which is what would turn
   §1.4's step-0 baseline from an argument into a control. Needs forward
   passes, and it is the single thing that would most strengthen §1.5.
5. **Only then F2/F3**, the J-lens, which is what unblocks F4, F5, F7, F8, F10
   — the functional and causal columns, and the phase's central test.

### 5.1 REVISED ORDERING, 2026-09-20, after the papers were read

The list above was written when `2411.04990` was `[S]`. It is now `[R]`
(`lit-10.md` §11), and **one free row moved to the front.**

1. **F13, the centre scan.** Greedy sequential acceptance over token positions,
   **both rules**, **swept in `δ`**, per layer. It needs positions and a
   distance and **not the HDBSCAN partition at all** — so it is the one row in
   this phase immune to §3's reproducibility floor, and it is the test F0 was
   standing in for. Free.
2. **F14 beside it.** `E[#strong centres] = E_{x∼μ}[1/μ(B_δ(x))]` estimated on
   the same cloud, against the observed count. The phase's
   packing-versus-content discriminant **with an exact i.i.d. null and no free
   parameter but `δ`** (`math-10.md` §7.2) — and the `δ` where the two meet
   reads back `c²/β`, turning `PROJECT.md` §3.40's undecided convention into a
   measurement. Free.
3. **`CLAIM-C`'s two HDBSCAN metrics against the reproducibility floor** —
   item 1 above, unchanged, and still the only open item bearing on a
   *registered* prediction. Free.
4. **F11 rows A1–A8, plus the new A9** (are the strong centres the sinks?
   `attention-10.md` §6). A2 and A4 still carry the most information per unit of
   work. Free.
5. **F16 and F17**, because both are cheap and both improve every row after
   them: the manifold `d_eff` (`math-10.md` §7.3), and a graded calibrated null
   to replace the binary controls (`lit-10.md` §12.2).
6. Then F6, the norm-matched random twin, and only then F2/F3.

**F20 is the phase's natural known-answer dry run** — the only experiment here
whose predicted outcome is a theorem — and `claims/EXPERIMENTS.md` records that
two adjudicable gates never had one. It costs a forward pass and it waits on
F13.

**Before any of this becomes `design-10.md`:** `notes-10.md` §12 as amended.
`2411.04990` **has now been read**; what remains unread and could still change a
construction is `2303.06562` (ContraNorm, before any Phase 9 spreading arm) and
`2607.15495` (the J-lens paper itself) — `lit-10.md` §15.

---

## 6. What this phase has NOT established

- **Nothing is registered and nothing is adjudicated.** Tier 1 throughout.
- **H-PARK is not confirmed.** §1.5 reads parked rather than pinned in one
  window, on an interpretation of `Z`, with the density confound argued rather
  than controlled.
- **H-CAT is not refuted.** The functional and causal columns are untouched.
- **The attention flip is not refuted** — it is re-measured on a different
  model from the one that produced the published number, and found to be mostly
  structural there.
- **F0 is not an adjudication of the parking account.** It is a tier-1
  exploratory result that came back against the prediction, on a statistic
  whose wording was never frozen — **and, since 2026-09-20, on a statistic that
  is a proxy for the account's object rather than an instance of it**
  (`lit-10.md` §11.4). F13 is the row that would adjudicate, and it has not run.
- **Nothing in the five papers read on 2026-09-20 is a theorem about Pythia.**
  `2411.04990` ties weights across layers, omits the MLP, and proves its
  meta-stability results at `V = I`, `Q = K = I`, `d = 2`. The correspondence is
  a hypothesis to test on a real model; that is the point of testing it.

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-24 when the card was written.

- 2026-09-20 · 410m's step 0 and step 1 are the same weights, so every count or pooled statistic over checkpoints here counts one checkpoint twice: the unit counts (3 648 = 19 × 8 × 24, 11 400 = 3 × 19 × 8 × 25, A0's 3 646), sweep means and merged E. Min–max ranges such as the "0–16" rows are unchanged · §3.51.3
- 2026-09-23 · the 12 new v2 prompts are held out on 410m, not pooled into Stages 1–5, so the header's "the enlarged battery can carry a registered prediction" holds only for the 12; they are partly seen already via `CLAIM-C` on 1.4b and gpt2-large · `docs/PHASE_REVIEW.md` "Decisions", `p10_cluster_function/handoff-10.md` §0.4
- 2026-09-24 · §0's "`METS_REPO` defaults to the MAIN tree" stopped being true: every runner now defaults to its own checkout (fixed in place) · `docs/PHASE_REVIEW.md` "Parked"
