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
  - The 410m sweep had no density partition in 152 of 152 dirs, and `pair_agreement`, the only semantic instrument, wrote well-formed zero records. The backfill re-derives the labels, bit-checked against the pilot, and does not rerun the analysis — `p10_cluster_function/status-10.md` §2, `p10_cluster_function/handoff-10.md` §1.2
  - The HDBSCAN partition is not reproducible run to run (ARI 5th percentile 0.347). A0 and F0 hold on the second sweep; per-layer claims stay exposed — `p10_cluster_function/status-10.md` §3, `p10_cluster_function/status-10.md` §3.1
  - 410m's step 0 and step 1 are the same weights, so the checkpoint axis has 18 distinct points — §3.51.3
  - First look at `pair_agreement` on the pilot: the chance that a lexically similar pair shares a cluster is flat over training, while the share of mutual-NN pairs that are lexically similar falls between steps 512 and 3000. No null, and the threshold was not swept — `p10_cluster_function/handoff-10.md` §1.1
  - Lemma C.1's saturation is a formula for Phase 1's 50–55 carrying capacity — `p10_cluster_function/math-10.md` §5.4, §3.53
- **Superseded / wrong:**
  - F0's reading as a test of the parking account: a density cluster's earliest member is not the paper's strong Rényi centre; F13 is the real row — `p10_cluster_function/lit-10.md` §11.4, `p10_cluster_function/status-10.md` §5.1
  - The Rényi packing law "as a function of `n`": the law is in β and dimension, not `n` — `p10_cluster_function/lit-10.md` §5, `docs/AXES.md` §7
  - §3.53's "no gradient-flow structure": Lemma 5.3 makes the causal dynamics a sequential gradient flow — §3.52.5
  - `handoff-10.md` §0.4's "Stages 1–5 re-run on all 20 prompts for free", which contradicted registering on prompts chosen blind: the 12 are held out on 410m — `p10_cluster_function/handoff-10.md` §0.4, `docs/PHASE_REVIEW.md` "Decisions"
  - §5's order: replaced by §5.1 after the papers were read (F13 first) — `p10_cluster_function/status-10.md` §5.1
- **Registry:** none, because the phase is pre-design and deliberately unregistered (`claims/EXPERIMENTS.md`). F14 is named as the one to register (`handoff-10.md` "Standing constraints"). The 12 new v2 prompts are held out on 410m as a confirmation set, only partly blind (`CLAIM-C` ran them on 1.4b and gpt2-large); whether F14 is scored on them, on all 20, and in what order is undecided and the user's (`docs/PHASE_REVIEW.md` "Open" 1–4)
- **Depends on:** 1@d881a84e97, 5c@e75b33ae46, 7d@19b7d835b7, 7e@0c1071db50, 8@8cc3fb223c, 9@e70efd632b
- **Feeds:** 9
- **Open threads:**
  - F5, the four-signature concordance, is the phase's central test. It needs the J-lens (F2 → F3 → F4), and F2 needs HF access — `p10_cluster_function/status-10.md` §4
  - F13, the strong-Rényi centre scan, which F0 stood in for; it is free and needs no partition. F14 needs it, and so do F15 and F20 — `p10_cluster_function/status-10.md` §5.1
  - `CLAIM-C`'s two HDBSCAN metrics have never been compared against the reproducibility floor. This is the only open item here that bears on a registered prediction — `p10_cluster_function/status-10.md` §5
  - No norm-matched random twin per checkpoint, so F12's density confound and the parked window are argued, not controlled — `p10_cluster_function/status-10.md` §5
  - Does `pair_agreement`'s decline survive a threshold sweep, and what are clusters made of by token class (Stage 1)? — `p10_cluster_function/handoff-10.md` §1.3
  - Is 410m spent on the induction axis for any Phase 10 registration that joins 7d's causal sweep? — `docs/PHASE_REVIEW.md` "Parked"
  - Position is a confound in every row here, with three separate corrections and no shared one in `core/` — `p10_cluster_function/handoff-10.md` "Standing constraints on all of it"
  - The `p10_*` readers' default glob now mixes the Phase 1 sweep with Stage 0's v1 dirs. The holdout guard (`core/holdout.py`) removes only the 12, so Stage 1's first reader has to select through Stage 0's index — `p10_cluster_function/handoff-10.md` "Parked", `p10_cluster_function/status-10.md` §0
- **After Phase 10:**
  - Rebuild a cluster ensemble (1d's intent) only after measuring whether tuning reduces §3's run-to-run drift (free: the two sweeps' activations)
  - Everything in Stages 1–5 again on all 20 prompts once registrations are frozen (free: Stage 0's dirs)
  - F20, the frozen-centre intervention, as the phase's known-answer dry run (forward pass: 410m or 70m, needs F13)
- **Reviewed:** 2026-09-24 · body `f63187c4b0`
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

**The holdout guard (2026-09-24).** Readers refuse held-out inputs by default.
`--v1-only` drops them and `--allow-holdout` reads them; both are recorded under
`"holdout"` in the output record. It holds out: a run dir for one of the 12 keys (any
model; by `manifest.json` `prompt_key`, else by name), any file named for one,
a pooled file beside such a run dir (`pair_agreement.json`), and
`claim_c_real_run.json`. Dry run on `data/phase12` at 12:40, chunk 2 mid-step512,
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

**Held as a hazard, not a result.** The density confound is argued from the
step-0 value, not proved — the proper control is a norm-matched random twin per
checkpoint, which the sweep does not carry. And the whole reading rests on
§1A.6's **interpretation** of `Z` as a metric, not on a causal measurement.
**`notes-10.md` §3.1's functional and causal columns are untouched and F5 is
still what discriminates.**

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

**The HDBSCAN partition is not reproducible run to run.** Two independent
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
