<!-- p10_cluster_function/handoff-10.md -->
# Phase 10 — HANDOFF for the cluster-function thread

**This is a scoped handoff, not the project's.** It carries one thread: *what
are clusters made of, what do they do, and can the drive that forms them be used
as an instrument?* — the hypothesis set in **`questions-10.md`**, opened
2026-09-20 after five papers were read as primary text (`lit-10.md` §11–§15).

> **When this thread is active, start here.** When it closes or is parked, go
> back to **`STATE.md`**, which is the main line and is authoritative for
> everything else — `CLAIM-C`, the e-value audit, the
> registry, disk, and the branch state.

**Last updated:** 2026-09-24 — Stage 0 at pin `64a4087`: chunk 1 done (144 runs), chunk 2 running since 05:35 (235 of 380 indexed at 09:20), chunk 3 the last. **First action: launch chunk 3 once chunk 2 logs `chunk end` (§0.3).**
**Tier:** everything below is **exploratory and unregistered**. `claims/registry.json`
is untouched. Nothing here may be quoted as an adjudication.

---

## The shape of the plan, and why it is in this order

**General first, then particular.** Each stage is meant to leave a *holistic*
picture in place before the next narrows it, so that a later result is read
against a described population rather than against an assumption.

| stage | question | cost | gates |
|---|---|---|---|
| **0** | **Get more prompts through Phase 1.** 8 → 20 | **forward passes, ~40–74 GB** | the statistical power of every row below |
| **1** | What is actually in a cluster? | free | everything else. Do not skip |
| **2** | What does the whole 19 × 24 field look like? | free | which axis any later claim lives on |
| **3** | Is a cluster one anchor plus ballast? | free | the trash-collection question itself |
| **4** | What is the mechanism — clock, or structure? | free | H-WARP vs H-STRUCT; Blog 1's headline |
| **5** | Do the weights predict the clusters? | free | the weight↔activation bridge |
| **6** | What do clusters do for the output? | forward passes | the functional column, H-PARK vs H-CAT |
| **7** | Can the drive be used as an instrument? | forward passes | Phase 9, forgetting |

**Stage 0 is the only compute-heavy item before Stage 6, and it is first on
purpose.** **Stages 1–5 cost no forward pass**, and every one of them re-runs on
the enlarged battery for free once Stage 0 lands. Read that twice before
planning compute.

---

## Stage 0 — More prompts (DO THIS FIRST)

**Why it is first.** Every e-value in this project merges over units that are
**not independent** — layer-units inside one forward pass share a model, a text
and a prompt. `core/evalues.py`'s `average` merger is valid under arbitrary
dependence and correspondingly **low-powered**, which is why `PROJECT.md` §3.41
records `CLAIM-C` unable to express a p below 0.0661, and every row in this
handoff would inherit the same ceiling.

> **The exchangeable unit is the PROMPT, and there are eight of them.**
> `2501.10573` used **2 244**. Prompt count is what buys independence; token
> count buys within-prompt precision that a dependence-robust merger cannot
> exploit anyway. **This is the binding constraint on every result below, and it
> is the one thing here that compute can fix.**

### 0.1 The work is already licensed — do not invent new prompts

`core/prompts.py` is a **versioned battery** with a deterministic
`PROMPT_BATTERY_HASH` written into every `manifest.json`, plus
`verify_same_battery` to check two runs at analysis time. It already carries
**v2 — 21 prompts, hash `06790b90dcfe`** — extended on 2026-09-19 under **a rule
committed in its own commit, ahead of the text**, precisely so git shows the
rule predates the prompts (`PROJECT.md` §3.42).

**Phase 1's metastability sweep used 8 of them.** The rest have never been
through Phase 1:

| | keys |
|---|---|
| **In the Phase-1 sweep (8)** | `wiki_paragraph`, `sullivan_ballou`, `paper_excerpt`, `homer_iliad`, `hdbscan_code`, `camus_letranger`, `latex_monograph`, `repeated_tokens` (degenerate control) |
| **v2, never run through Phase 1 (12)** | `wiki_photosynthesis`, `wiki_byzantium`, `lincoln_letter_short`, `lincoln_letter_long`, `paper_attention`, `paper_svflow`, `quijote_capitulo`, `moby_loomings`, `sklearn_kmeans_code`, `scipy_linkage_code`, `latex_beamer`, `latex_article` |
| **v1, excluded** | `short_heterogeneous` — **115 characters.** Almost certainly too short to cluster; check before assuming it is usable |

> **Correction to `status-10.md`**, which says *"the 13 battery prompts that have
> never been through Phase 1"*. It is **12**, plus `short_heterogeneous` at 115
> characters, which is probably unusable. The 12 are six genres × two prompts,
> one short-band and one long-band per pair, exactly as the committed rule
> specifies — including the non-English literary prompt (`quijote_capitulo`),
> mirroring v1's `camus_letranger`.

**The selection risk is therefore already retired.** These prompts were chosen
blind, under a written rule, before anyone saw how they behave. **Running them
is not a new selection decision** — it executes one already taken and already
committed. That is worth a great deal: it means the enlarged battery **can carry
a registered prediction**, which the current one cannot without re-making that
argument from scratch.

**8 → 20 usable prompts: a 2.5× increase in the exchangeable unit.**

### 0.2 The budget, measured on disk today

Per run directory, 410m, ~450-token prompt (measured on
`pythia-410m-step1_wiki_paragraph`):

| file | size | scales as |
|---|---|---|
| `plateau_attentions.npz` | **147.6 MB** | n² |
| `attentions.npz` | **147.6 MB** | n² |
| `activations.npz` | 44.3 MB (`(25, n, 1024)` float32) | n |
| the other ten JSON/npz | ~0.6 MB | — |
| **total** | **325 MB** | |

**12 prompts × 19 checkpoints = 228 new directories.**

- At 325 MB: **74 GB**. Free on `WDS_500`: **164 GB**. Fits, leaves ~90 GB.
- Dropping the duplicate below: **~40 GB**. Leaves ~124 GB.

> **`plateau_attentions.npz` is a byte-identical relayout of `attentions.npz` —
> verified across all 24 layers.** One stores `(24,16,n,n)` under a single key;
> the other stores 24 arrays `attn_L0…attn_L23` of shape `(16,n,n)`. Same
> numbers, stored twice, **148 MB per directory**. Across the existing 152
> directories that is **≈ 22 GB recoverable**, and it halves the storage cost of
> every new run.
>
> **Do not delete before checking which readers use which file** — the plateau
> path presumably reads the per-layer layout. The cheap fix is to stop writing
> one for new runs and relayout on load. `PROJECT.md` §5.1's registered-decisions
> rule applies: a disk decision gets recorded before it is taken.

**Attention storage scales as n², so prefer MORE prompts at the current length
(~250–600 tokens) over fewer long ones.** A 1024-token prompt costs ~4× the
attention of a 500-token one for the same single unit of statistical power. This
runs *against* `2501.10573`'s `N ≥ 500` guidance, and the reason is that they
wanted per-prompt intrinsic dimension while this project wants independent
units. **Both are true and they trade off; F16 is the row that will feel it** —
only `homer_iliad` (512 tokens) currently clears their threshold. Measured
lengths: `hdbscan_code` 242, `paper_excerpt` 286, `latex_monograph` 446,
`camus_letranger` 465, `wiki_paragraph` 467, `sullivan_ballou` 482,
`homer_iliad` 512.

**Compute is unmeasured.** No per-run timing for a Phase-1 410m sweep was found
anywhere in the tree. `data/phase12/claim_c_logs/` holds per-prompt timing for
the `CLAIM-C` arms and is the nearest reference. **Time one prompt × one
checkpoint and multiply** rather than launching 228 runs on an estimate.

> **Measured 2026-09-22 — and the paragraph above was wrong about timing.**
> Every `manifest.json` carries `wall_time_seconds`: the 152 WDS directories
> took **median 116 s, 43–400 s, 5.8 h in total**, all `device: cpu`, run from a
> container (`experiment.txt` shows `/mnt/mets/...`) before HDBSCAN was present.
> **Probe:** `wiki_byzantium` × `step143000` in the conda `mets` env with
> `CUDA_VISIBLE_DEVICES=""` (CPU, matching every run on disk) — **203 s in the
> manifest, 3 min 51 s with model load, 3.2 GB RSS, ~4.7 cores, 264 MB on disk
> at 403 tokens.** Kept at `data/phase12/2026-09-22_16-43-47` (log in
> `data/phase12/stage0_logs/timing_probe.log`). Checks 2 and 3 pass on it:
> battery hash `06790b90dcfe`, `hdbscan_labels.json` populated (25 layers,
> 41–49 clusters), `pair_agreement` populated per layer with real mutual pairs.
>
> **Token counts, pythia tokenizer:** the 12 new prompts are 190–614 tokens
> (`latex_article` 614, `scipy_linkage_code` 527); the old 8 are 242–562.
> Scaling the probe by n² for attention and n for activations: **12 new × 19 =
> ~57 GB** (~33 GB without the plateau duplicate); **re-running the old 8 × 19
> under v2 would add ~43 GB**. Either fits in 164 GB **without taking §5.1b's
> duplication decision**, so Stage 0 does not force it.
>
> **Check 1 cannot pass as written.** All 152 WDS directories are battery
> **v1, `1e47918ef77a`**; a new run writes **v2, `06790b90dcfe`**, and this
> module's own docstring says *"a v2 run is comparable only with other v2
> runs"*. What *is* established: **v1's nine texts are byte-identical inside
> v2** — `compute_prompt_battery_hash(v2 minus the twelve, "v1")` reproduces
> `1e47918ef77a` exactly. **Open, for the user:** run only the 12 (8 v1 + 12 v2
> directories, comparability resting on that text identity and on two
> toolchains), or all 20 under v2 in one install (one hash, native partition and
> native `pair_agreement` throughout — which also retires §1.2's zeroed WDS
> record). **Nothing beyond the probe has been launched.**

### 0.3 How to run it

> **Decided 2026-09-22: option B** — all 20 prompts × 19 checkpoints = **380
> runs under v2 from one pinned commit**, in **10-hour chunks** (the machine is
> available in blocks). Driver: **`tools/run/stage0_chunk.py`**. It re-reads
> what is done from disk each time (v2 hash + pinned `git_sha` + populated
> partition + populated `pair_agreement`), so it is resumable with no ledger;
> it plans only what fits the budget, kills the running invocation at the
> deadline, and stops the chunk if any invocation comes back unpopulated.
> **A kill loses the whole invocation** (`pair_agreement.json` is written only
> at its end), so invocations are capped at **5 prompts** (≤ ~20 min lost).
> **Plan at the conservative probe estimate: 3 chunks** (144 + 144 + 92 runs,
> ~25 h with a 1.25× margin); it re-fits from the sweep's own manifests after 5
> runs. The 09-22 probe does **not** count (other commit). **Readers select
> Stage 0's runs through `$METS_RESULTS_DIR/stage0_logs/stage0_index.json`**,
> never by globbing hash + sha: a killed invocation leaves orphaned directories
> that match both (`status` counts them).
>
> ```bash
> # once, after the driver's PR merges: a run tree pinned at that merge commit
> cd /run/media/system/WDS_500/Mets && git fetch -q
> git worktree add --detach ../Mets-stage0 origin/main
> # each chunk -- PIN comes from the run tree itself, never from origin/main
> cd /run/media/system/WDS_500/Mets-stage0 && PIN=$(git rev-parse HEAD)
> export HF_HOME=/run/media/system/WDS_500/Mets/data/hf \
>        METS_RESULTS_DIR=/run/media/system/WDS_500/Mets/data/phase12 \
>        HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
> PY=/run/media/system/WDS_500/miniforge3/envs/mets/bin/python
> $PY -m tools.run.stage0_chunk --pin $PIN plan
> nohup $PY -m tools.run.stage0_chunk --pin $PIN run --budget-hours 10 \
>   > /dev/null 2>&1 &     # log: $METS_RESULTS_DIR/stage0_logs/chunk_*.log
> ```
>
> `../Mets-stage0` is a **run tree, not a task worktree**: detached, never
> edited, removed when Stage 0 is done. Record `$PIN` in `STATE.md` when the
> first chunk starts.
>
> **2026-09-22:** run tree created at `64a4087` (the #68 merge, which carries
> #67), so **`$PIN` = `64a4087`**. Chunk 1 started 21:20 (pid 13350; 144 runs
> in 29 invocations).
>
> **2026-09-23 correction:** chunk 1 was **not** killed. The box suspended
> 21:52:45–05:13:37 (`journalctl`, "PM: suspend exit"); the process froze and
> resumed, and the invocation spanning the suspend finished `rc 0, 5/5`. The
> earlier note read a paused process as a dead one. At 07:30, **65 runs
> indexed** in 13 invocation dirs, all populated (every `hdbscan_labels.json`
> non-empty with real clusters, e.g. step143000 homer_iliad layer 12: labels
> −1…10 over 512 tokens; every `pair_agreement.json` mostly non-zero). The
> `2026-09-22_21-20-26` directories are complete and indexed, not orphans.
> **The budget is awake time**: `_run_chunk` uses `time.monotonic()`, which
> Linux stops during suspend, so the hard stop moved from 07:20 to ≈ 14:41. At
> ~10 min per invocation it should finish all 144 ≈ 10:30. **It did: `chunk
> end` at 10:31:52, 29/29 invocations `rc 0`, 144 runs indexed, every
> `hdbscan_labels.json` and `pair_agreement.json` non-empty.** Runs are faster
> than the probe estimate, so chunk 2's `plan` (which re-fits) may take more
> than 144 and Stage 0 may need 2 chunks, not 3.
>
> **Chunk 2 and later: launch only when no driver is alive.** The log's last
> line is not enough: a driver that dies abnormally writes no stop line, and a
> suspended one looks stalled. Chunk 1 holds no lock, so `pgrep` is the guard
> for it; `flock` guards every chunk from 2 on. `systemd-inhibit` holds off
> suspend only while the command runs (no machine setting changes); drop it
> if you want the box to sleep.
>
> ```bash
> pgrep -f 'python -m tools.run.stage0_chunk' && echo "DRIVER ALIVE: do not launch"
> tail -3 $METS_RESULTS_DIR/stage0_logs/chunk_*.log   # expect a stop line
> $PY -m tools.run.stage0_chunk --pin $PIN plan       # 380 − done left
> nohup systemd-inhibit --what=sleep:idle --why=stage0 \
>   flock -n $METS_RESULTS_DIR/stage0_logs/.driver.lock \
>   $PY -m tools.run.stage0_chunk --pin $PIN run --budget-hours 10 \
>   > /dev/null 2>&1 &
> ```
>
> **2026-09-24: chunk 2 launched 05:35:20** with the block above (driver pid
> 23468). `plan` re-fitted to 125 s per run at 403 tokens: **233 runs in 48
> invocations, ~9.9 h**, leaving 3 runs for chunk 3. The old guard
> `pgrep -af 'tools.run.stage0_chunk'` printed "DRIVER ALIVE" with no driver
> running: run through `bash -c`, it matches its own shell's command line.
> The guard above now matches only the python process.

The original single-run instructions, from the main tree, with the environment from `archive/PROJECT-start-here.md`:

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf METS_RESULTS_DIR=$PWD/data/phase12
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
python -m p1_mstate_tracking.run_1 --models pythia-410m --prompts <key>
```

Four checks, none optional:

1. **`PROMPT_BATTERY_HASH` must match** between the new runs and the existing
   152. `core/prompts.py`'s `verify_same_battery` exists for exactly this. If it
   does not match, the new prompts are not comparable to the old and the point
   is lost.
2. **HDBSCAN must be present.** Otherwise `hdbscan_labels.json` is empty *and*
   `pair_agreement` silently writes zeros — the failure §1.2 documents. Use the
   conda `mets` env (`/run/media/system/WDS_500/miniforge3/envs/mets/bin/python`),
   not `.venv`, for anything touching the partition, and **verify on the first
   directory before launching the rest.**
3. **Time one, then extrapolate** (§0.2).
4. **Never `git add` under `data/`.**

### 0.4 What Stage 0 unblocks, and what it does not

**Unblocks:** the e-value ceiling moves; and — the point — **a registered
prediction becomes possible on a battery whose rows were chosen blind.**

> **Held out (user, 2026-09-23).** The 12 v2 prompts new to the **410m sweep**
> are Phase 10's **confirmation set**. Stage 0 runs all 20, but Stages 1–5 read
> **only the 8 v1 prompts** until the predictions scored on the 12 are in
> `claims/registry.json`. Checking that Stage 0's outputs are *populated* is not
> reading them. **Enforced in code (2026-09-24):** `core/holdout.py`. Every
> runner over `data/phase12` refuses held-out inputs, or is exempt for a stated
> reason; the Phase 10 readers take `--v1-only` (drop them) or `--allow-holdout`,
> and a test fails any runner that is neither (`status-10.md` §0). **One 410m
> value is already in a doc:** §0.3's timing probe records `wiki_byzantium`'s
> HDBSCAN cluster count (41–49 per layer), the quantity F14 would use; weigh it
> under "Open" 1.
> **They are not fully unseen.** `CLAIM-C` ran all 20 on pythia-1.4b and
> gpt2-large, trained and random (`PROJECT.md` §3.46; `data/phase12/2026-09-19_*`).
> `claims/audits/claim_c_real_run.json` carries per-prompt cluster count,
> membership, effective rank and Fiedler for the 12. So they are blind only for
> 410m measures that those runs did not compute. Open for the user: scope,
> 12 vs 20, order, release (`docs/PHASE_REVIEW.md` "Open").

**Does not fix:** twenty is not 2 244. The dependence structure *within* a prompt
is unchanged. And the 12 new prompts share the old eight's checkpoint grid and
model, so they add independence **in text and nothing else**.

**Stage 0 is done when** all 380 runs of option B (§0.3; 228 of them the
12 new prompts) are in `stage0_index.json` with a matching battery hash, a
populated `hdbscan_labels.json`, and a populated `pair_agreement`.

---

## Stage 1 — What is in a cluster

**The question:** which tokens end up clustered, and which end up noise? Not a
hypothesis — a description. The project has a Jacobian-lens plan, transport
observables, Schur decompositions and an e-value calculus, and the simplest
descriptive fact about its central object has never been tabulated.

### 1.1 What already exists, and it is more than expected

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
cloud. The layer axis is collapsed here, which is the mistake Stage 2 exists to
stop. No position control. No null.

### 1.2 The third casualty of the HDBSCAN outage

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

### 1.3 What to do, in order

1. ~~**Sweep the `ext_sem_threshold`**~~ **DONE 2026-09-24 on Stage 0's v1
   runs** (`status-10.md` §1.6). The threshold never acts. "ext_semantic" is
   "the two tokens are the same token" (cosine 1 at layer 0, no position
   embedding), so §1.1's decline is the share of mutual-NN pairs that are repeats,
   0.875 → 0.696. §1.1's "lexical → contextual" survives only as "repeat →
   non-repeat". Re-run at 152 runs once Stage 0 completes.
2. **The token-composition table, which still does not exist.** Carry
   repeat/non-repeat (a token with an earlier copy in the prompt) as a column:
   step 1 shows it drives the only semantic number the project had. Join
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

**Stage 1 is done when** there is a token-composition table with both sweeps and
a threshold sweep behind §1.1's decline.

---

## Stage 2 — The whole field, before any slice of it

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
   A0's residual onset and §1.1's 512–3000 window. The question the picture
   answers: *how many distinct events are there?*
3. **Never quote a sweep mean again without its checkpoint split.** Worth
   writing into `design-10.md` as a rule when that file exists.

**Stage 2 is done when** one figure sheet shows the whole field and the events
are counted rather than assumed.

---

## Stage 3 — Anchor or ballast: the trash-collection question itself

`questions-10.md` §1.1. **The highest-value free experiment identified in this
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

**Stage 3 is done when** the anchor/ballast decomposition has a number and a
direction, on both sweeps.

---

## Stage 4 — Mechanism: is training a clock or a structure?

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

**Stage 4 is done when** Blog 1's resistance headline has a candidate mechanism
that is either supported or refused, and the two clocks are separated.

---

## Stage 5 — Do the weights predict the clusters?

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

**Stage 5 is done when** a weights-only quantity has either predicted an
activation-space count or failed to, on the record, with the wording frozen
first.

---

## Stage 6 — What clusters do for the output

First analysis stage that costs forward passes (Stage 0 aside). Everything above should be read first.

1. **Loss coupling** (`questions-10.md` §4). Per-token surprisal against cluster
   membership, **with position as a covariate**. One forward pass per prompt per
   checkpoint, no backward. **No logits are on disk** — checked.
   The developmental question is the prize: does the clustered/unclustered
   surprisal gap open at the same step as A0's residual and §1.1's window?
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

## Stage 7 — The drive as an instrument

Phase 9's territory; listed here because Stages 1–6 are what would license it.

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
- **Eight prompts.** The power ceiling on exploration, and it stays there:
  Stage 0's 12 new prompts (chosen blind under a committed rule,
  `questions-10.md` §7 item 2) are **held out for confirmation** (§0.4), not
  pooled. Read every exploratory e-value against `PROJECT.md` §3.41's floor.
- **Nothing in the five papers is a theorem about Pythia** — tied weights, no
  MLP, `V = I`, `Q = K = I`, `d = 2`.

## Returning to the main line

**`STATE.md`**. It carries the branch and PR state, the `CLAIM-C` position,
disk, and the project-wide next steps; `PROJECT.md` §3.52 and §3.53 carry
the literature read this thread grew out of, and §3.51 the four rows that ran
before it.

## Parked

- **The `p10_*` readers' default selection now mixes sweeps** (confound, found
  building the holdout guard, 2026-09-24): `--pattern pythia-410m-*` over
  `data/phase12` globs the Phase 1 sweep (152), Stage 0's v1 dirs (135 at 12:40)
  and the `p2_eigenspectra_*` dirs, so one (prompt, step) can appear twice. The
  guard only removes the 12. Why: a re-run of F0/F12/A0 or Stage 1 would pool two
  batteries. Cost: select through `stage0_index.json` (or `core/run_discovery`
  with a battery-hash check) when Stage 1's first reader is written. Changes:
  what every Stage 1 table is computed on. **Done for Stage 1** (2026-09-24):
  `p10_ext_sem_threshold.py` reads only the index. The older `p10_*` readers
  still glob; fix them when they are next re-run.
- **`lit-8.md` never cites `literature-8.md`** (found in the same batch): the
  later leads-only file does not know the earlier fetched readings exist. Why:
  a reader of `lit-8.md` misses 18 verified ids. Cost: one line. Changes:
  nothing measured.
