<!-- p10_cluster_function/status-10.md -->
# Phase 10 — STATUS

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
these rows test, `attention-10.md` for row A0's audit, and `PROJECT.md` §3.51
for the cross-cutting findings (three of which are about the project's
instruments rather than about clusters).

**Decision taken 2026-09-20 (user):** F0 runs **exploratory first**, which caps
it at tier 1 — the answer was seen before any wording was frozen. If it is ever
to be the project's first adjudication it needs a fresh axis: 70m, or the 13
battery prompts that have never been through Phase 1.

---

## 0. The instruments, and where they live

| what | where | tier |
|---|---|---|
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

**How to re-run anything here.** From the worktree, with `METS_REPO` pointed at
it, because every `tools/run/*.py` derives `sys.path` from `METS_REPO` and it
defaults to the MAIN tree:

```bash
METS_REPO=$PWD METS_DATA=/run/media/system/WDS_500/Mets/data \
  /run/media/system/WDS_500/miniforge3/envs/mets/bin/python tools/run/<runner>.py
```

**The conda `mets` env, not `.venv`,** for anything touching HDBSCAN — §2.

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

`tools/run/backfill_hdbscan.py` re-derives it from `activations.npz` — no
forward pass, **152 directories in 69 s**. Mean 47.1 clusters/layer, mean noise
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

**Before any of this becomes `design-10.md`:** `notes-10.md` §12 still holds.
`2411.04990` has not been read as full text by anyone here, and two phases
depend on it.

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
  whose wording was never frozen.
