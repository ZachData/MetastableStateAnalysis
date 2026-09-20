<!-- p10_cluster_function/handoff-10.md -->
# Phase 10 — HANDOFF for the cluster-function thread

**This is a scoped handoff, not the project's.** It carries one thread: *what
are clusters made of, what do they do, and can the drive that forms them be used
as an instrument?* — the hypothesis set in **`questions-10.md`**, opened
2026-09-20 after five papers were read as primary text (`lit-10.md` §11–§15).

> **When this thread is active, start here.** When it closes or is parked, go
> back to **`PROJECT.md`'s "Resume here" block**, which is the main line and
> remains authoritative for everything else — `CLAIM-C`, the e-value audit, the
> registry, disk, and the branch state.

**Last updated:** 2026-09-20.
**Tier:** everything below is **exploratory and unregistered**. `claims/registry.json`
is untouched. Nothing here may be quoted as an adjudication.

---

## The shape of the plan, and why it is in this order

**General first, then particular.** Each stage is meant to leave a *holistic*
picture in place before the next narrows it, so that a later result is read
against a described population rather than against an assumption.

| stage | question | cost | gates |
|---|---|---|---|
| **0** | **What is actually in a cluster?** | free | everything. Do not skip |
| **1** | What does the whole 19 × 24 field look like? | free | which axis any later claim lives on |
| **2** | Is a cluster one anchor plus ballast? | free | the trash-collection question itself |
| **3** | What is the mechanism — clock, or structure? | free | H-WARP vs H-STRUCT; Blog 1's headline |
| **4** | Do the weights predict the clusters? | free | the weight↔activation bridge |
| **5** | What do clusters do for the output? | forward passes | the functional column, H-PARK vs H-CAT |
| **6** | Can the drive be used as an instrument? | forward passes | Phase 9, forgetting |

**Stages 0–4 cost no forward pass.** Read that twice before planning compute.

---

## Stage 0 — What is in a cluster (START HERE)

**The question:** which tokens end up clustered, and which end up noise? Not a
hypothesis — a description. The project has a Jacobian-lens plan, transport
observables, Schur decompositions and an e-value calculus, and the simplest
descriptive fact about its central object has never been tabulated.

### 0.1 What already exists, and it is more than expected

**Corrected while writing this.** The project *does* have a semantic instrument:
`pair_hdbscan_agreement` (`p1_mstate_tracking/clustering.py`) tags mutual
nearest-neighbour pairs against the **embedding Gram** as an external semantic
axis, and reports `ext_semantic_fraction` and `ext_sem_same_cluster_frac` per
layer. It runs inside `analysis_p1.py` and its output is stored in every
`clustering.json`.

> **It has never been reported in any markdown file in this repository**, and it
> is populated in **6 066 of 6 075 layer-records** of the pilot sweep on
> `HDD_1TB`.

**First look, 2026-09-20 — tier 1, exploratory, no null, not registered.**
Mean over 8 prompts × 25 layers per checkpoint, pilot sweep, native labels:

| step | `ext_sem_same_cluster_frac` | `ext_semantic_fraction` |
|---|---|---|
| 0 | 0.688 | **0.833** |
| 64 | 0.631 | 0.856 |
| 512 | 0.678 | 0.788 |
| 1000 | 0.647 | 0.752 |
| **3000** | 0.732 | **0.692** |
| 19000 | 0.638 | 0.723 |
| 143000 | 0.668 | **0.708** |

**Two curves, and only one of them moves.**

- **`ext_sem_same_cluster_frac` is flat** across all 27 checkpoints, 0.62–0.73.
  *Given* that a pair is lexically similar, it co-clusters about two thirds of
  the time, and training does not change that.
- **`ext_semantic_fraction` falls, 0.833 → 0.708**, with the drop concentrated
  between **step 512 and step 3000**.

**The tentative reading** — and it is the first direct evidence on the user's
semantic question:

> **Training does not change how clusters treat lexically-similar tokens. It
> changes which tokens are neighbours at all.** At initialisation, residual-stream
> neighbourhoods are mostly inherited from token identity; by step143000 ~29 % of
> mutual-NN pairs are *not* lexically similar. Neighbourhoods become
> **contextual rather than lexical**, and the transition window is steps
> 512–3000.

**That window is crowded, and the co-location is the interesting part.** A0's
learned attention residual appears at step ~2000–4000; `PROJECT.md` §3.51 puts
the energy break at 256→512 and the Fiedler zero-crossing at 1000→3000; F1/F12's
parked window is 32–512. Whether these are one event or four is not established
by co-occurrence and **must not be asserted from this table.**

**Caveats, all load-bearing.** `ext_semantic` is `emb_gram[i,j] > 0.5` — an
arbitrary cosine threshold, and the decline could be a norm/scale effect on the
Gram rather than a structural change, so **sweep the threshold before believing
it**. Mutual-NN pairs are ~90 per layer: a small, special subpopulation, not the
cloud. The layer axis is collapsed here, which is the mistake Stage 1 exists to
stop. No position control. No null.

### 0.2 The third casualty of the HDBSCAN outage

`pair_agreement` is computed only when `"labels" in hdb_data`. During the outage
(`status-10.md` §2, `PROJECT.md` §3.51.4) that branch was never taken, so the
`else` wrote a **well-formed record of zeros and nulls** into all 152
directories of the WDS sweep rather than failing.

> `status-10.md` §2 lists F0, F5 and F11-A4 as the rows the outage blocked.
> **`pair_agreement` is a fourth, it is the project's only semantic instrument,
> and nothing flagged it** — because a silent zero record looks exactly like a
> real one. The backfill wrote `hdbscan_backfill.json` beside `clustering.json`
> and did not re-run the analysis, by design (`status-10.md` §2 item 3), so the
> WDS sweep still has no semantic record. **The pilot sweep does, and it is what
> the table above is read from.**

This is the same bug class as `docs/AXES.md` listing an absent artifact as
present, and as standing rule 4's *"refuse rather than degrade."*

### 0.3 What to do, in order

1. **Sweep the `ext_sem_threshold`** (currently 0.5) over the pilot sweep and
   check whether the 0.833 → 0.708 decline survives. **Free, and it gates
   everything in §0.1.** If the decline is threshold-dependent it is a scale
   artifact and the reading is dead.
2. **The token-composition table, which still does not exist.** Join
   `tokens.txt` to `hdbscan_labels.json` and report, per layer per checkpoint:
   clustered vs noise composition by **frequency rank**, by whitespace /
   punctuation / subword-continuation / alphabetic class, and by within-prompt
   repetition count. Trash collection predicts clusters dominated by
   high-frequency, low-information tokens. **This is a table, it costs nothing,
   and it either makes the semantic question concrete or retires it.**
3. **Report both sweeps.** The pilot has native labels; the WDS sweep has
   backfilled ones. Agreement across them is the §3.51.4 check that every
   partition-derived claim now owes.
4. **Do not fix `pair_agreement` on the WDS sweep by re-running `analysis_p1.py`**
   without reading `status-10.md` §2 first — the toolchain guard and the
   `read_labels` precedence rule both apply, and filling a canonical file while
   `clustering.json` still says `null` beside it is the shape of inconsistency
   `b55375e` had to un-write.

**Stage 0 is done when** there is a token-composition table with both sweeps and
a threshold sweep behind §0.1's decline.

---

## Stage 1 — The whole field, before any slice of it

**Why now:** A0's sweep mean hid its own finding (`status-10.md` §1.1), and
**that is now a pattern rather than an incident** — F12's raw sign is mostly
definitional until split by checkpoint; F1's kinematic signature is a *window*,
invisible in the mean. Before any new statistic is interpreted, the 19 × 24
field it lives on should be visible.

1. **Plot every statistic already on disk as a checkpoint × layer heatmap.**
   `layer_metrics.csv`, `energies.json`, `spectral.json`, `geometry.json`,
   `sinkhorn.json`, the cluster counts, the noise fraction. No new computation —
   this is a rendering of artifacts that exist in 152 directories.
2. **Mark the four known transitions** (`math-1.md` §13.2) on every panel, plus
   A0's residual onset and §0.1's 512–3000 window. The question the picture
   answers: *how many distinct events are there?*
3. **Never quote a sweep mean again without its checkpoint split.** Worth
   writing into `design-10.md` as a rule when that file exists.

**Stage 1 is done when** one figure sheet shows the whole field and the events
are counted rather than assumed.

---

## Stage 2 — Anchor or ballast: the trash-collection question itself

`questions-10.md` §0.1. **The highest-value free experiment identified in this
pass.**

1. **F13, the centre scan** (`notes-10.md` §8). Greedy sequential acceptance over
   token positions, **both rules** (Rényi and strong Rényi), **swept in `δ`**,
   per layer. Needs positions and a distance; **needs no HDBSCAN partition at
   all**, which makes it the one row in this phase immune to §3.51.4's floor.
   Run it in the geodesic metric **and** in `⟨Qx, Ky⟩` (`core/ln_frame.py`'s
   Gram), per `2411.04990` §5.1's generalisation.
2. **The cross, which is the actual test.** Join the accepted centre set to
   received attention (`attentions.npz`), to `Z_i/(i+1)`, and to cluster
   membership. **Is the centre a sink and the member a ballast?** Reported as
   `attention-10.md` row **A9**.
3. **F15, the coverage curve** — fraction of tokens within `δ` of an accepted
   centre, as a function of depth. The real-model analogue of the paper's
   Figure 3, and **its own §6 open problem** (*does each centre capture `ω(1)`
   particles?*).

**Falsifier:** centres and members receive statistically indistinguishable
attention once position is divided out. That kills H-ANCHOR/BALLAST and leaves
A0's 6 % residual unexplained.

**Stage 2 is done when** the anchor/ballast decomposition has a number and a
direction, on both sweeps.

---

## Stage 3 — Mechanism: is training a clock or a structure?

Both rows are free and they are independent, so they can run in either order.

1. **H-THERMOSTAT** (`questions-10.md` §2). Decompose `Z_k = z_k^{sink} +
   z_k^{rest}` from `attentions.npz`; renormalise rows over `j ≠ sink`; test
   whether the sink's mass share predicts the energy-monotonicity violation rate
   per layer per checkpoint. **Watch the index trap** — the thermostat is the
   sink's contribution to *everyone else's* row sum, not the sink's own `Z_0`
   (`math-10.md` §2).
   **Falsifier:** no correlation across the 18 distinct checkpoints; or the same
   correlation at step 0, which makes it architectural rather than learned.
2. **H-WARP, the curve collapse** (`questions-10.md` §1). Fit one monotone warp
   per checkpoint against `T_eff` / `β_eff` and ask whether the 18 depth-profiles
   fall onto a master curve. **`math-1.md` §15 item 3 has called `T_eff` the
   highest-value unrun quantity at report-only cost since before this phase
   opened.**
   **Falsifier:** no warp collapses them — in which case the residual *is* the
   learned structure and becomes the object, which is the more interesting
   outcome.

**Stage 3 is done when** Blog 1's resistance headline has a candidate mechanism
that is either supported or refused, and the two clocks are separated.

---

## Stage 4 — Do the weights predict the clusters?

`questions-10.md` §3. Free; needs Phase 2's projectors, which are on disk for
all 19 checkpoints.

1. **`d₁ = dim L`** from the OV spectrum per checkpoint (`sym_*` / `schur_*`).
2. **F14: observed strong-centre count against Lemma C.1**, `E_{x∼μ}[1/μ(B_δ(x))]`,
   with `d₁` from step 1 rather than fitted.
3. **F16: kNN intrinsic dimension** (GRIDE / TLE / ESS, `k ≤ 20`) as the
   independent manifold-dimension comparator (`math-10.md` §7.3).
4. **The carrying-capacity join** — does `1/σ^{d₁−1}(B_δ)` land on the measured
   invariant of **50–55 max-alive clusters**? (`docs/readings/2411.04990.md`
   §4.3, `tools/math_checks/parking_center_count.py`.)

> **REGISTER F14 BEFORE LOOKING.** `status-10.md` records the decision that ran
> F0 exploratory and capped it at tier 1. With **39 registrations and zero
> adjudications**, F14 is the second chance at the project's first adjudication:
> a published quantitative prediction, an exact i.i.d. null, one free parameter
> (`δ`), and a rival account that predicts a different answer. `CLAUDE.md`
> trigger 2 applies — the registration is the last moment a literature fact can
> still change the statistic.

**Stage 4 is done when** a weights-only quantity has either predicted an
activation-space count or failed to, on the record, with the wording frozen
first.

---

## Stage 5 — What clusters do for the output

First stage that costs forward passes. Everything above should be read first.

1. **Loss coupling** (`questions-10.md` §4). Per-token surprisal against cluster
   membership, **with position as a covariate**. One forward pass per prompt per
   checkpoint, no backward. **No logits are on disk** — checked.
   The developmental question is the prize: does the clustered/unclustered
   surprisal gap open at the same step as A0's residual and §0.1's window?
2. **Variance decomposition instead of ARI** (`questions-10.md` §5.1).
   Within-cluster vs between-cluster functional spread. **Robust to the
   reproducibility floor in a way F4/F5 as written are not**, and it is the
   phase's central test made cheaper and sturdier at once.
3. **The graded block-shuffle null** (F17), which gives every row above a
   dose–response curve and gives Blog 1 the input-side control it has never had.
4. **Only then F2/F3**, the J-lens, which unblocks F4, F5, F7, F8, F10. Carry
   `notes-10.md` §4.5's four caveats and `2505.16831`'s warning that small
   perturbations near the logits distort task-level metrics while features stay
   intact — a lens readout is a logit-space readout.

---

## Stage 6 — The drive as an instrument

Phase 9's territory; listed here because Stages 0–5 are what would license it.

1. **Switch the lever.** Patch the **attention logits** (that is `β` exactly)
   rather than LayerNorm `γ`, which on Pythia's fused QKV moves the value path
   too (`questions-10.md` §6.3, `docs/readings/2411.04990.md` §1).
2. **The capacity bound.** `δ = cβ^{−1/2}` plus Lemma C.1 gives a closed-form
   ceiling on how much re-parking a budget buys. **No one in unlearning has a
   capacity theorem** (`questions-10.md` §6.1).
3. **Evict and fill** (`questions-10.md` §6.2) — park the target, then occupy
   the space, per Theorem 5.2. The only experiment in this phase whose predicted
   outcome is a theorem, hence **the phase's natural known-answer dry run**,
   which `claims/EXPERIMENTS.md` says two adjudicable gates never had.
4. **A relearning arm is not optional** (`2505.16831`). Without it, a forgetting
   result measures the thing that paper says is routinely mismeasured.

---

## Standing constraints on all of it

- **Tier discipline** (`PROJECT.md` §3.29). Exploratory unless registered first;
  F14 is the one to register.
- **Reserved rungs.** 70m and 410m only. `pythia-1b` and `pythia-1.4b` stay
  reserved until a prediction names them.
- **Position is a confound in every row here**, and three rows have already
  invented three separate corrections. It wants one shared abstraction in
  `core/` (`questions-10.md` §7 item 4).
- **The partition is not reproducible run to run** (§3.51.4). Prefer statistics
  that do not need it (Stage 2), aggregate over thousands of units, and report
  both sweeps.
- **Eight prompts.** The power ceiling on everything (`questions-10.md` §7 item
  2). More prompts is the highest-value compute available and needs no new
  instrument.
- **Nothing in the five papers is a theorem about Pythia** — tied weights, no
  MLP, `V = I`, `Q = K = I`, `d = 2`.

## Returning to the main line

`PROJECT.md`'s **"Resume here"** block. It carries the branch and PR state, the
`CLAIM-C` position, disk, and the project-wide next steps; §3.52 and §3.53 carry
the literature read this thread grew out of, and §3.51 the four rows that ran
before it.
