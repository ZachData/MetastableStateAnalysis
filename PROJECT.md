<!-- PROJECT.md -->
# PROJECT — the living state of this repository

The file to read first, and the one to keep current. It answers: what machine
this runs on, where the work stands, what is blocking, what has been registered
and may not be re-decided, and how to reproduce anything.

**It is not a session diary.** What changed and why lives in `git log`, and the
reasoning behind a construction lives in `POPPER_PLAN.md`'s numbered sections.
This file carries only what a fresh session needs in order to start working,
and every number in it is measured on this machine.

| | |
|---|---|
| Branch | `main` is at `ded8a06` (PR #57 merged). **Two PRs open, stacked**: **#58** `claude/aca-phase-9-planning-rvzw3x` (Phases 9/10 documentation and math checks, no runner) and **#59** `claude/p10-free-rows` **based on it**, carrying Phase 10's four free rows built, tested and RUN. Both from the worktree `../Mets-p10`. **#60** `claude/p10-literature-read` is based on **#59** and merges `origin/claude/aca-phase-9-planning-rvzw3x` into itself to reconcile two concurrent reads of the same paper. **Merge #58, then #59, then #60.** `claude/attention-collapse-augmentation-qsxwg8` is redundant — its one commit is cherry-picked onto #58 — and should be deleted |
| Last updated | **2026-09-20.** **NEXT ACTION IS COMPUTE — §3.54.2b: take the Phase-1 battery from 8 prompts to 20 (`handoff-10.md` Stage 0).** Eight prompts is the exchangeable unit and the power ceiling under which every e-value here sits (§3.41's p ≥ 0.0661 floor); `2501.10573` used 2 244. **No new prompts are to be invented**: `core/prompts.py` already carries v2, 21 prompts, hash `06790b90dcfe`, extended under a rule committed ahead of the text (§3.42), and **12 of them have never been through Phase 1** — so running them is not a new selection decision, and the enlarged battery **can carry a registered prediction where the current one cannot**. (`status-10.md`'s "13" was wrong; the 13th is `short_heterogeneous` at 115 chars.) **Budget measured**: 228 new directories × 325 MB = **74 GB** against **164 GB free**, and **`plateau_attentions.npz` is a verified byte-identical relayout of `attentions.npz`** — 148 MB per directory, **≈22 GB recoverable**, which takes the new-run cost to ~40 GB. Attention scales n², so **more short prompts beat fewer long ones**. Compute is unmeasured: **time one prompt × one checkpoint first.** **§3.54: a scoped thread opened with its own handoff — `p10_cluster_function/handoff-10.md` (ordered plan, **eight** stages, general → particular, Stages 1–5 free) and `questions-10.md` (hypotheses).** Its Stage 1 (*what is actually in a cluster*) found **a fourth casualty of the HDBSCAN outage**: `pair_agreement`, **the project's only semantic instrument**, wrote a well-formed record of zeros into all 152 WDS directories rather than failing — it degraded instead of refusing, which is why it survived the outage, the backfill, the audit and two literature passes. **The pilot sweep's copy is populated in 6 066 of 6 075 layer-records and had never been reported.** First read of it, tier 1: **`ext_sem_same_cluster_frac` is FLAT** across all 27 checkpoints (0.62–0.73) while **`ext_semantic_fraction` falls 0.833 → 0.708, concentrated in steps 512–3000** — tentatively, **training does not change how clusters treat lexically-similar tokens, it changes which tokens are neighbours at all**, and neighbourhoods go from lexical to contextual. That window collides with three other known transitions and the co-location is NOT established. Three hypotheses named project-wide: **H-ANCHOR/BALLAST** (a cluster is one attended anchor plus ignored ballast — HDBSCAN gives membership and never centrality, so this was invisible), **H-THERMOSTAT** (the sink is *how* a trained model resists collapse, by absorbing mass into `Z_k` and lowering coupling — the model already learned a metric intervention, which reframes Phase 9), and **H-WARP** (training is a reparametrisation of the depth clock, separable by curve collapse against `T_eff`). **§3.52: five papers READ as primary text — `2411.04990`, `2501.10573`, `2601.02932`, `2605.12765`, `2505.16831`, the PDFs supplied by the user, every scholarly host still blocked. No measured number changes; three constructions do.** **(a) The parking nuclei are a GEOMETRIC object** — a *strong Rényi centre* is a token separated by more than `δ = cβ^{−1/2}` from **every preceding token**, a greedy sequential rule on positions and distances with no partition in it — so **F0 measured a proxy and its failure is not evidence against the parking account** (§3.52.1). **(b) The count law is exact, distribution-free and dimension-free**: `E[#strong centres] = E_{x∼μ}[1/μ(B_δ(x))]`, no exponent and no β, only `δ` — so §3.50's `d ≫ 1` verdict applies to the power-law asymptotic only, and its *"no 0.7476 in it"* is wrong (Appendix C.4 has it, for `d = 2` ordinary centres). The `δ` where observed meets predicted **reads back `c²/β`**, turning §3.40's undecided convention into a measurement. **(c) The gradient-flow hazard resolves in the project's favour**: the masked system IS a gradient flow, a **sequential** one, `φ̇_k = −(1/Z_k)∂E_k/∂φ_k` — a global ensemble potential is void, the per-token curvature claim survives, and **`1/Z_k` is literally F12's measurement**, making `math-1.md` §1A.6's metric reading of `Z` an equation rather than an interpretation. **(d) `math-10.md` §5.2's `d_eff` list was missing the theoretically-correct entry** — the law wants a *manifold* dimension, so kNN intrinsic dimension (**≈ 7–15**), not effective rank (≈ 225). **(e) Two thirds of Phase 9's novelty differentia fail**: GUARD-IT's update is norm-preserving but **not** a rotation and **is** exactly invertible; what survives is **state versus operator**. **(f) Any Phase 9 forgetting claim now needs a relearning arm** (`2505.16831`). **(g) `2411.04990` §B.3 measures `albert-xlarge-v2`'s `V`-spectra** — prior art for this project's V-attractive/V-repulsive split. **Eight ladder rows added (F13–F20), seven free**, and the next free row is now **F13, the centre scan**, which needs no HDBSCAN partition and is therefore immune to §3.51.4's reproducibility floor (§3.52.4, `status-10.md` §5.1). Documentation only — no code, no records, no registry entry. **§3.53 is a concurrent session's read of the same paper, written the same day and merged here; §3.52.5 says where the two differ — on Lemma 5.3, where §3.52 is the more complete reading. §3.53 adds four things §3.52 does not have: Thm 4.1 makes position 0 a theorem and caps `plan-9.md` §4.3; Lemma C.1's saturation IS the carrying-capacity finding; `d₁ = dim L` makes the `d_eff` test differential; and RMSNorm's diagonal is absorbable into `K, Q, V`, so a γ-patch is NOT a read-side lever.** **§3.51: Phase 10's four free rows RAN on real checkpoints — the first Phase 10 results, all tier 1 and unregistered. `p10_cluster_function/status-10.md` is the phase record; §3.51 carries only what is project-wide.** (1) **Row A0: the attention flip is ~94 % causal mask.** Over 3 646 layer-units the raw 1.417× / 0.825× gap of 0.592 becomes **0.036** once `math-10.md` §1's content-free baseline is divided out — and split by checkpoint, it is **ENTIRELY mask at initialisation** (corrected gap 0.004 at step 0) with a residual appearing only at **step ~2000–4000** and persisting (0.172 at step143000), while the position-bias confound *falls* with training. (2) **F0's anchor test fails in the direction it was predicted to succeed**: nuclei sit at **0.316 against a null 0.208**, median p = 1.00 at alternative 'less' — and it survives the confound, since a null restricted to the clustered population only moves the null mean to 0.231. (3) **F1: the identity coupling is EXACTLY optimal in 99.5 % of 3 648 layer boundaries** (`swap_absorbed_fraction` 0 to machine precision in 3 630 of them), so **every per-layer displacement number this project has recorded is the true `W_2`** rather than the upper bound it was known to be — a validation of a whole class of existing numbers, with no forward pass. Its `swap_fraction` is a duplicate-embedding tie and **not** motion. Straightness *falls* with training, 0.151 → 0.122. (4) **The 410m sweep had NO density partition at all** — `hdbscan_labels.json` is `{}` in **152/152** directories, wider than §3.41 records and listed as present in `docs/AXES.md`; backfilled in 69 s from `activations.npz`, verified bit-identical against the pilot. (5) **Three instrument defects, each of which would have produced a false number**: `core.evalues.combine` is a product and rejects **19.67 %** of the time under the null at 25 dependent units (→ `average`, valid under arbitrary dependence); `p_from_null` returned the **resolution floor** on a degenerate null decided by rounding noise (→ `p_from_null_tolerant`); and the first full A0 run **could not have rejected whatever the data said**, because the largest merged e-value a 400-draw design can produce is 10.01 against a threshold of 20 (→ `max_attainable_average_E`, 1 599 draws is the minimum, runners now at 2 000). (6) **The HDBSCAN partition is NOT reproducible run to run** — over 2 600 layer-pairs from two sweeps whose tokens are identical and whose activations differ by at most **7.9e-05**, **16.7 % of label vectors differ**, ARI's 5th percentile is **0.347** and its minimum **0.166**, and cluster count moves by up to **20**. A measurement-reproducibility floor nobody had measured, and `CLAIM-C` reads two of its six registered metrics from this partition alone (§3.51.4). **Both headline rows clear it**: re-run on the pilot sweep's 243 directories with NATIVE labels, A0's corrected gap agrees to two or three decimals at all 13 shared checkpoints and F0's nucleus statistic to three — 0.3164 against 0.3157 (`status-10.md` §3.1). (7) **`pythia-410m` step0 and step1 are the same weights** — all 292 tensors bit-identical upstream — so the checkpoint axis carries **18 distinct points, not 19**. (8) **F12 confirms `math-10.md` §2 on real data and independently of β's convention**: raw `log Z` is **99.5 %** position at β = 1, the sink is the **minimum** of raw `Z` (percentile 0.0014) and the **maximum** of corrected `Z` (0.9986), identical to four decimals across the β grid. Its raw clustered-minus-noise sign is **mostly definitional** — +0.465 at step 0 under random weights, because HDBSCAN clusters by the density `Z` measures — and against that baseline **the largest real excursion is NEGATIVE at steps 32–64**, in the same window where F1 finds the strongest kinematic signature. Clustered particles there move less *and* sit below the untrained baseline in `Z`: **parked, not pinned, and a window rather than a property of the trained model** (`status-10.md` §1.5). Gate green at **2 616 passed**. **§3.50: the literature scan and the derivations** — four `tools/math_checks/` files (28 checks) and three corrections to statements this repo makes, including that the Rényi-parking law is `Θ(β^((d−1)/2))`, in β and dimension, **not in `n`**. **§3.49: the attention flip audited** and **`docs/AXES.md`** opened. **§3.48: Phase 10 opens**; **§3.47: Phase 9's plan**, parked on it. **2026-09-19:** the **e-value audit is COMPLETE** — five units, thirty-nine registered predictions, **zero e-values** (§3.45's closing table; units §3.36, §3.40, §3.43–§3.45). **`CLAIM-C`'s gate ran three times and refused three different ways** (§3.41, §3.46). **Prompt battery v2**: 9 → 21 (§3.42). Disk: 44 → 188 GB free (§5.2/§5.3). Earlier entries live in their own §3.x sections. |
| Structural map | `INDEX.md` — which phase lives in which directory, and what is archived |
| **Prior work, per phase** | **`docs/LITERATURE.md` — the index; `<phase>/lit-N.md` — the review. Read before writing anything up. Rows 6 and 19–22 are `[R]` as of 2026-09-20 (§3.52); everything else is `[S]` or `[N]`** |
| Method and construction log | `POPPER_PLAN.md` §6a–§6t |
| Pre-registered predictions | `PREDICTIONS.md`, `claims/registry.json` |
| What can carry an e-value | `claims/EVALUABILITY.md` (current state, by phase); `claims/EVALUABILITY_LOG.md` (how each null was built) |

---

## 1. Start here

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf
export METS_RESULTS_DIR=$PWD/data/phase12
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

./scripts/check.sh gate     # 2616 passed / 5 skipped / 47 deselected, ~55 s (2026-09-20)
```

**Two environments, and the difference is load-bearing.** `.venv` is the
project's interpreter and what every runner's guard asserts. **Anything that
runs HDBSCAN must use the conda `mets` env instead** —
`/run/media/system/WDS_500/miniforge3/envs/mets/bin/python` — because it is the
only install that reproduces this project's historical partitions
(`p1_mstate_tracking/clustering.py`'s measured note, `PROJECT.md` §3.51.4).
`tools/run/backfill_hdbscan.py` refuses to run elsewhere rather than writing an
incomparable partition.

If the gate is green the tree is consistent. If it fails on a `sha256` mismatch,
a module carrying a record's hash was edited — see §6.3, it is a chore and not a
bug.

### Resume here (2026-09-20 — Phase 10's four free rows have RUN; the attention flip is mostly the causal mask; two PRs open)

**This block is the handoff.** Everything a session needs to continue is here
or one link away; the sections below it are orientation and history.

**Git. Two PRs are open and the second is stacked on the first.**
`main` is at **`ded8a06`** (PR #57 merged), CI green.
- **PR #58 is OPEN** — `claude/aca-phase-9-planning-rvzw3x` → `main`, from the
  worktree `../Mets-p10`. **Documentation and symbolic checks only**: Phase 9's
  notes and plan (parked), Phase 10's `notes-10`/`lit-10`/`math-10`/
  `attention-10`, `docs/AXES.md`, and four `tools/math_checks/` files.
  4 471 lines, no runner, nothing registered.
- **PR #59 is OPEN and BASED ON #58, so merge #58 first** —
  `claude/p10-free-rows`, **targeted at #58's branch rather than at `main`** so
  its diff is this work alone; GitHub retargets it to `main` when #58 merges.
  Same worktree. **20 commits, 27 files, +5 986/−47**: Phase 10's four free
  rows built, tested and **RUN**; the HDBSCAN backfill that unblocked them;
  the partition-reproducibility measurement; both headline rows replicated on
  the pilot sweep; three fixes to the e-value/null machinery;
  `status-10.md`; and §3.51. Gate green at **2 616 passed**.
- **PR #60 is OPEN and BASED ON #59**, so the stack is #58 → #59 → #60 and it
  merges bottom-up — `claude/p10-literature-read`, same worktree.
  **Documentation only, no code**: five papers read as primary text and the
  eight files they change (§3.52.3). Nothing registered, no runner, no record.
- **`claude/attention-collapse-augmentation-qsxwg8` is redundant** — its one
  commit (`cf5f7ee`, Phase 9's notes) was cherry-picked onto #58's branch.
  Delete it rather than opening a PR.

The user merges from GitHub. After both merge, from the main tree:
`git pull --ff-only && git worktree remove ../Mets-p10`. Only `data/` is
untracked (the HF cache and run directories — never `git add -A` under it).

**`../Mets-claim-c` is a spent worktree** at `6a804c2` on `claude/claim-c-arms`,
a branch merged in PR #57. `git worktree remove ../Mets-claim-c` whenever
convenient; nothing depends on it.

**Run the Phase-10 runners from the worktree with `METS_REPO` pointed at it.**
Every `tools/run/*.py` derives `sys.path` from `METS_REPO`, which defaults to
the MAIN tree — so a runner launched from a worktree silently imports the main
tree's code and fails on anything new. The working invocation is
`METS_REPO=$PWD METS_DATA=/run/media/system/WDS_500/Mets/data <python> tools/run/...`.

**The HDBSCAN backfill must run in the conda `mets` env, not `.venv`** — see
`status-10.md` §2. `/run/media/system/WDS_500/miniforge3/envs/mets/bin/python`. The
script refuses elsewhere rather than writing an incomparable partition.

**A merge mid-session is not a merge of the session.** #56 merged at 18:35 and
captured the branch as it stood at that instant; the seven commits after it sat
unmerged until #57 was opened. Check `git log origin/main..HEAD` in the
worktree before assuming a PR carries what you wrote.

**Phase 10's free rows have run. Records under `data/analysis/`, all at 2 000
permutations with `max_attainable_E` 22.37 on their face:**

| file | row | units |
|---|---|---|
| `p10_row_a0.json` | A0, WDS sweep | 3 646 |
| `p10_row_a0_pilot.json` | A0, **pilot sweep, native labels** | 5 593 |
| `p10_f0_anchor.json` | F0 | 3 800 |
| `p10_f0_anchor_pilot.json` | F0, **pilot sweep, native labels** | 5 835 |
| `p10_f1_transport.json` | F1 | 3 648 |
| `p10_f12_z.json` | F12, three betas | 11 400 |
| `p10_partition_stability.json` | the reproducibility floor — **no p-value, by design** | 2 600 |

All **tier 1, exploratory, unregistered** and **not quotable as
adjudications**; `claims/registry.json` is untouched.

> **A scoped thread is open and has its own handoff: `p10_cluster_function/handoff-10.md`.**
> The cluster-function question — what clusters are made of, what they do, and
> whether the drive that forms them is usable as an instrument (§3.54). **Eight
> stages, general → particular. Stages 1–5 cost no forward pass.**
>
> **START AT STAGE 0, WHICH IS COMPUTE: take the Phase-1 battery from 8 prompts
> to 20** (§3.54.2b). Eight prompts is the power ceiling under which every
> e-value in this phase sits. The 12 unused v2 battery prompts were already
> chosen blind under a rule committed ahead of the text, so running them is not
> a new selection decision — and the enlarged battery can carry a **registered**
> prediction where the current one cannot. 228 directories, ~74 GB against
> 164 GB free, ~40 GB if the verified `plateau_attentions` duplication stops
> being written. **Time one prompt × one checkpoint before launching the rest.** **When that thread is parked, come back to this
> block**, which stays authoritative for the branch state, `CLAIM-C`, the
> registry and disk.
>
> **Read `p10_cluster_function/status-10.md` first** — it is the phase record:
> every number, every caveat, how to re-run each row, the ladder's state, and
> **§5.1, the next-step list as reordered on 2026-09-20.** §3.51 and §3.52 here
> carry only what is project-wide.
>
> **Then read `lit-10.md` §11.** `2411.04990` has been read as primary text at
> last, and it is the paper both Phase 9 and Phase 10 were built on. It changes
> what F0 means, what the count law is, and whether the Hessian framing is
> void (§3.52).

Four headlines: **the attention flip is ~94 % causal mask and entirely mask
before step 2000** (and it holds on a second sweep with an independent
partition, `status-10.md` §3.1); **F0's anchor test fails in the direction it
was predicted to succeed** under both nulls; **the identity coupling is exactly
optimal in 99.5 % of layer boundaries**, so every displacement number on record
is true `W_2`; and **F1 and F12 together read parked rather than pinned, in a
window at steps 32–512 rather than as a property of the trained model**
(`status-10.md` §1.5).

**The free things to do next, REORDERED 2026-09-20 by §3.52** — the full list
is `status-10.md` §5.1:

1. **F13, the centre scan.** Greedy sequential acceptance over token positions,
   both rules (Rényi and strong Rényi), **swept in `δ`**, per layer. It needs
   positions and a distance and **not the HDBSCAN partition at all** — the one
   row in this phase immune to §3.51.4's floor, and the test F0 was standing in
   for. Free.
2. **F14 beside it** — observed count against Lemma C.1's
   `E_{x∼μ}[1/μ(B_δ(x))]`, the packing-versus-content discriminant with an
   exact i.i.d. null and no free parameter but `δ`. Free.
3. **`CLAIM-C`'s two HDBSCAN metrics against the reproducibility floor**
   (§3.51.4) — still the only open item that bears on a *registered*
   prediction. Free.
4. **`attention-10.md`'s rows A1–A8, plus the new A9** (are the strong centres
   the sinks?), which A0 gated and has now cleared. Free.

**Read every `reject: False` in those records against `max_attainable_E`.**
The averaging merger is deliberately low-powered and the first A0 run used a
draw count that made rejection impossible in principle (§3.51.2). Effect sizes,
`median_p` and `frac_below_05` are the informative fields.

**`CLAIM-C`: compute is DONE, and the gate has been run.** All four
required `CLAIM-C` arms were produced, then **re-run on 2026-09-19 with
HDBSCAN present** (61 + 30 + 33 + 17 min): `data/phase12/2026-09-19_10-51-00`
(`gpt2-large`), `_11-52-10` (`gpt2-large-random`), `_12-21-54`
(`pythia-1.4b-step143000`), `_12-54-33` (`pythia-1.4b-random`). **Those four
are the scoreable set**; the 2026-09-17 and earlier-2026-09-19 directories
carry no HDBSCAN metrics and are superseded — the other four metrics are
bit-identical between old and new, checked (`status-1.md`). Only the optional
`step0` sensitivity arm is unproduced.

**The gate returned `INSUFFICIENT`, `hard_stop: true`, `falsified: false`, no
p-value — §3.41.** Not for want of data: every prompt is usable, nothing is
dropped, and all six metrics exist in all four arms. The full six-metric row
cannot express a p below **0.0661** because four of the eight prompts split
their metrics exactly 3–3 and a tied row cannot move the statistic; where a
small p *is* expressible — the six leave-one-out subsets, five metrics each,
all eight rows informative — the p-values are 0.2529–0.9572 for transfer and
0.0778–0.8872 for inversion. Concordance is **23/48 = 47.9%**, the coin. Per
metric: `fiedler_mean` 8/8, `mass_near_1` 7/8, `cluster_membership` 5/8,
`cluster_count` 2/8, `effective_rank` 1/8, `cka_prev` 0/8 — **the metrics
disagree with each other about whether the phenomenology transfers**, which is
what INSUFFICIENT is for. Nothing is adjudicated; the per-metric split is a
diagnostic, and re-picking the metric set from it would be the selection
pre-registration exists to prevent. Only more prompts, chosen blind, can move
the floor (§3.41).

**Disk (git-ignored, so this is the only record of what exists). Cleared and
rearranged 2026-09-19 — see §5.2/§5.3 for the reasoning.**
- `data/hf` (70 GB): `pythia-410m` (51 GB, the pilot schedule), `pythia-70m`
  (5.0 GB, **19 revisions, step0 → step143000**), `pythia-1.4b` (11 GB, fp32,
  revisions `step143000`, `step0` **and `main`**), `gpt2-large` (3.1 GB).
  Everything runs with `HF_HUB_OFFLINE=1`. `albert-xlarge-v2` is **not** here.
- **`CLAIM-C`'s v1-battery arms, scoreable, hashed by
  `claims/audits/claim_c_real_run.json` — do not delete:**
  `data/phase12/2026-09-19_10-51-00` (`gpt2-large`), `_11-52-10`
  (`gpt2-large-random`), `_12-21-54` (`pythia-1.4b-step143000`), `_12-54-33`
  (`pythia-1.4b-random`).
- **`CLAIM-C`'s v2-battery arms** (21 prompts): `data/phase12/2026-09-19_13-40-48`
  onward, one per arm.
- `data/phase12/claim_c_logs/`: per-arm logs with per-prompt timing, plus the
  chain logs `rerun_chain.log`, `rerun_v2_chain.log` and `archive_to_hdd.log`.
- `data/phase12/2026-08-31_*`, `2026-09-01_*`: Phase 1's 19-checkpoint 410m
  sweep and the Phase 2 eigenspectra beside it. **v1 battery**, like everything
  made before 2026-09-19.
- **On HDD_1TB, not on this volume:** `results/2026-08-12_05-01-35` and
  `results/p2_eigenspectra_2026-08-13_05-13-52` are symlinks into
  `/run/media/system/HDD_1TB/Mets_archive/` (§5.2 — 27 pilot steps that exist
  nowhere else). `/run/media/system/HDD_1TB/activation_cache` holds the 355 GB
  Blog-1 cache, **deliberately left alone** (§5.3).
- **Deleted 2026-09-19:** the four superseded `CLAIM-C` arm directories from
  2026-09-17 / early 2026-09-19, and `data/superseded/phase7_float32`.

**A trap that cost the first launch of arm 3.** `_pythia_entry` sets
`tokenizer_revision: None` deliberately, which means the tokenizer loads at
revision `main` — and `main` was not in the 1.4b cache, so `HF_HUB_OFFLINE=1`
failed the arm in seconds with a "couldn't connect" error that names no
revision. pythia-410m had `main` cached, which is why nothing had hit this.
Fixed by fetching the tokenizer at `main` once, online; **any new Pythia size
needs the same** before it can run offline.

**The e-value audit is COMPLETE (§3.36, §3.40, §3.43, §3.44, §3.45).** Five
units, thirty-nine registered predictions, **zero e-values**. One gate has
been run on real checkpoints (`CLAIM-C`, INSUFFICIENT) and one p-value is
now recorded (`P-I1`, INSUFFICIENT, its p not quotable). The per-phase
state, and §3.45's closing table for the pattern:

| phase | state |
|---|---|
| 1 (`CLAIM-A`, `CLAIM-C`) | **audited** (2026-09-17); the gate **RAN** — `INSUFFICIENT`, no p-value, concordance 47.9% (§3.41), `real_run_record` set. The battery is now **v2**, twelve rows added blind to lift the floor (§3.42), and the four arms are re-running on it. `CLAIM-A` stays `needs-null` behind it |
| 1c (`P-gamma1`, `P-gamma2`, `P-H1`, `P-S1`) | **audited (2026-09-19, §3.40).** All four classifications correct; three of four gates cannot be fed by anything on disk. `P-H1` measured for the first time. **Five decisions wait on the author** — `p1c_frames/status-1c.md` "E-value audit", last paragraph |
| 2 / 2d (`CLAIM-B`, `P-T1`, `P-M1`) | **audited (2026-09-19, §3.43).** Phase 2 registers nothing of its own. `P-T1`/`P-M1`: inputs all present and the join verified on real artifacts — blocked only by design (β producer → 1c-B → 2d) and by `P-T1`'s wording, which omits half of Table 1 row 2 and must be amended *before* the gate runs. `CLAIM-B`: needs 19 control series, the sweep has 6; needs 20–30 checkpoints, the sweep has 19 |
| 5b / 6 | **audited (2026-09-19, §3.44).** 19 dormant; `P6-R2` is blocked on `U_A`, a channel no artifact carries and §6 says nothing needs; `P6-R4`'s inputs exist and it is blocked on an unregistered exchangeable unit. Four dormant rows read `e-value` with no gate |
| 7 | **audited (2026-09-19, §3.45) — the audit is COMPLETE.** `P-I1`'s run is now recorded (`real_run_record` set) and the p it was quoted by is not quotable; `P-ST1`/`P-AB1`/`P-I3` built, calibrated, unrun; `P-I5` parked by design |

**Compute is DONE and nothing is running.** The four `CLAIM-C` arms were
re-run on prompt battery v2 in **4 h 51** (127 + 63 + 63 + 38 min, against
~2½ h projected — 21 prompts rather than 9, and v2's twelve are all long):
`data/phase12/2026-09-19_13-40-48`, `_15-47-26`, `_16-50-41`, `_17-53-27`.
**These four are the current scoreable set** and the v1-battery four
(10:51/11:52/12:21/12:54) are superseded but kept, since
`claims/audits/claim_c_real_run.json` has been overwritten with the v2 result
and the v1 one now survives only in `git log`.

**The gate refused a THIRD time, on its own calibration — §3.46.** 20 prompts,
120 cells, 54 concordant, homogeneity 0.875, 240 files hashed: the table is
complete and the floor is no longer binding. What binds is that the
homogeneity correction is tabulated only to **twelve** prompts, because
`tools/calibrate_claim_c_homogeneity.py` was written when eight was the count;
the gate refuses rather than report an uncorrected p, which is right, because
the uncorrected null is already measured anticonservative when the sign-rows
agree — and at 0.875 they largely do. **Extending the table is a calibration
job the gate itself prescribes**, costed on this machine at ~45 min for the
n = 20 row plus ~40 min to regenerate rows 6–12, or ~4 h for a contiguous
6–20. **Decided 2026-09-19 (user): not tonight.**

**Read this block, then §3.45's closing table (the audit's whole result on one
screen), then §3.46, §3.41 and §3.42 (what `CLAIM-C` cost and what it
returned), then the audit units in any order — §3.40, §3.43, §3.44, §3.45 —
then §3.36, §3.38, §3.35, §3.34, §3.33, §3.32, §3.31, §3.30, §3.29, §3.28.** §3.29 is the direction
decision and it demotes everything below it: the programme is the
particle/OT reading and the induction thread is an instance of it that is
now past diminishing returns. §3.34 flagged a check-in: `P-I5`'s control
construction had failed three times running (§3.32–§3.34), the third
failure mechanistically explained, and the user chose to redirect the
session to the alternative already queued — §2.5's isometric path
on `L7H8` — rather than a fourth blind attempt. **§3.35 is that: run for
real, and it found something.** `t=0` and `t=1` have identical singular
values by construction (an exact isometry) but read very differently —
`t=1` (the transpose) stays as broken as the symmetric midpoint rather
than recovering toward baseline, meaning read/write alignment carries
real causal weight beyond the spectral sign alone. `P-I5`'s
control-construction problem (§3.34) is UNRESOLVED, not abandoned — it's
parked, not closed, and is where a future session returns if `P-I5` is
picked back up. §3.28 is the pre-registration scan — it reclassifies
§3.22's novelty and blocks §3.27's registration, so reading §3.20–§3.27
without it will overstate what is new. Everything below this block is
earlier and is kept as background, not as the current state.

**The 2026-09-16 PR stack (#36–#47, plus #48–#53) is MERGED** — `main`
carries all of it as of 2026-09-17 and the branches are deleted. The gate
on the tip is **2368 passed / 5 skipped / 47 deselected** under `.venv`
(smoke tests need `SMOKE_REAL_DEPS=1 pytest -m smoke`). **This machine's
memory watchdog has killed background runs before** (transient — 24 GB+
free either side); long runs go under `setsid nohup python -u …`, log to a
file, and checkpoint per unit — the `P-I5` scripts per step, `run_1.py` per
prompt directory.

**CodeRabbit will not review a PR on its own** — under 10 stars this
repo gets no automatic reviews, so each PR needs its **"🔍 Trigger review"**
checkbox ticked by hand on CodeRabbit's first comment. `gh` works here; see the
git block below.

**Invariant 4 has been reworded (2026-09-13).** `design-8.md`'s item 4 no
longer asserts a fate neither rung has: it now stops the claim at the
mid-training minimum (both rungs diverge from a birth alignment to a minimum —
410m at 143000, 70m at step 32000) and states explicitly that what happens
after is **not** claimed to replicate — 410m ends a locked core plus one
inverted-direction defector, 70m **re-coheres**. §3.19 has the measurements
this rests on; this was the phase's one time-sensitive action (registration
freezes wording, `CLAUDE.md` trigger 2) and it is now closed.

**What is open. READ §3.29 FIRST — it reorders this list and the reason
matters more than the order.** The programme is the particle/OT reading;
induction heads were the instance and are past diminishing returns. Actions 1
and 2 serve the frame; 3–6 are the induction thread's leftovers and are
explicitly *not* the priority.

1. **`P-I5`'s instrument is PARKED, not closed — pick back up here if
   `P-I5` becomes the priority again.** STEP ONE (§3.30) and STEP ONE-B
   (§3.31) are done — with §3.31's statistic corrected on 2026-09-17 to
   the intersection-union test (min-rank did not control the union null a
   conjunction needs). STEP ONE-C through STEP ONE-F (§3.32–§3.34) tried
   two head-comparison controls, both failing to discriminate `L3H6` from
   `L4H6`/`L5H3`, and one `L3H6`-only diagnostic whose null geometric
   result is mechanistically explained (`raw_distance` on the ablated
   layer's own residual stream is structurally insensitive to WHICH
   constant replaces a fully-ablated slice). At the check-in this produced, the user chose
   to redirect to item 2 below instead of a fourth blind construction —
   see §3.35. **Next step here, when resumed:** a geometric readout at a
   LATER layer than the ablated one (downstream of where the ablated
   slice's information would need to propagate through further mixing),
   which the constant-substitution cancellation does not obviously apply
   to. Not attempted yet.
2. **§2.5's isometric path on `L7H8` — RUN (§3.35, 2026-09-16), and it
   found something real.** `t=0` and `t=1` have identical singular values
   (an exact isometry by construction) but `t=1` (the transpose) stays as
   broken as the symmetric midpoint rather than recovering toward
   baseline — read/write alignment carries real causal weight beyond the
   spectral sign. **Next step here:** §2.5.4's second family
   (`M_R = U R Σ V^T`, holding both subspaces fixed and rotating only the
   correspondence) is what would cleanly separate symmetry from
   alignment — this session's result shows alignment matters but not by
   how much relative to sign, which that construction would answer.
3. **Do NOT register §3.27** (§3.28): not differential, and CoAx reached the
   statistic first. Invariant 5 remains a clean *unclaimed negative* and is the
   better registration candidate if one is wanted — but per §3.29 the bar is
   frame-level, and `1b`/`1.4b` are spent only on that.
4. **§3.12-U's `L5H2` puzzle is closed (§3.20–§3.22, 2026-09-13), and the
   self-repair behind it is now measured exhaustively.** *(Closed thread — kept
   for reference, not as an action.)* `L5H2` is a
   previous-token head, uniquely wired into `L7H8`'s read-space
   (composition rank 0/112 at z +5.25, where prev-token heads without the wire
   do nothing), whose ablation breaks `L7H8`'s matching attention — hence
   neither an induction nor an FV score of its own. §3.12-S's super-additivity
   is set-wide: **44 heads above +0.1**, led by `L5H9` +3.36, `L9H5` +2.75,
   `L1H15` +2.35, and **MLP 6 at +6.26, above every head**. Stand-ins split by
   position — upstream ones partly restore `L7H8`'s attention, downstream ones
   move it by exactly zero. **MLP 6 is now opened too (§3.23)**: it forms an
   OR-gate with `L5H2` over `L7H8`'s matching, the repair is *active* (only
   the direction it rotates to works — the one it already had is worth no more
   than noise), and it is **not** recomputing the prev-token signal (cos
   −0.507 to `L5H2`'s own contribution). **§3.24 decodes the direction**: it is
   aimed at the redundancy set's shared key read-space (six of the top nine of
   272 downstream heads are members, on a flat per-layer profile) and is not a
   token signal. **§3.25 closes the per-member attribution** (the rotation
   moves the heads it points at; misses move least). **Open:** why the
   output-side compensator class exists at all (`L11H14` is its extreme, and
   `L5H2`×`L11H14` is the single largest residual against §3.12-V's magnitude
   rule at both checkpoints while being δ-cosine-orthogonal).
5. **70m's `L2H1` row stays ceiling-censored on every probe** (§3.17) — the
   analogue of 410m's headline prev-token × matcher pair. It needs §3.12-M's
   graded KL/λ readout, which is still unbuilt and is now blocking two things.
6. **`--probe` is on three of six runners.** `redundancy_catalog.py`,
   `member_subspace_geometry.py` have it as of 2026-09-12 along with
   `member_formation_curves.py`, `pairwise_interaction_matrix.py` and
   `useful_rank.py`; `ambient_budget.py` is still `wide`-only and its docstring
   still promises a `--text` arm that does not exist.
7. The older threads below (the LoRA tangent, invariant 3's scoping block, the
   step-1000 circuit) are unchanged by today's work.

**Before launching any cross-rung trajectory**, check the cached grids —
`ls data/hf/hub/models--EleutherAI--pythia-*/refs/`. They differ between rungs
and `HF_HUB_OFFLINE=1` turns a missing revision into an `OSError` mid-run
(§3.19). The matched grid is `256,512,1000,2000,4000,8000,16000,32000,143000`.

---

### The earlier resume block (2026-09-16 — every phase now has a literature review; read `docs/LITERATURE.md`)

*Merged 2026-09-17 from `claude/phase-literature-review-ntd772`; its section is §3.37 (renumbered from a colliding §3.16). Current state is the block above.*

**Read `docs/LITERATURE.md` before anything else.** It is new, it is the index to
sixteen new `<phase>/lit-N.md` files, and it changes what several phases should do
next. Full record in **§3.37**. Then `p7d_redundancy/status-7d.md`,
`p7e_consolidation/design-7e.md` and `p8_scale_ladder/status-8.md` for where the
measurements stand, and §3.12-V / §3.15 for the detail.

**The four things from §3.37 that change the plan, stated here so they are not
missed:**

1. **Phase 1's headline developmental arc is scooped** — `2509.23024` (NeurIPS 2025)
   reports the same non-monotonic three-phase trajectory on **Pythia 160M–12B**.
   `status-1.md`'s "the phase's main new object" must be struck or rewritten as a
   replication with a frame correction.
2. **7d's nearest neighbour is real** — `2607.01940` (CoAx, July 2026) computes the
   same second-order conditional-ablation object, and `2606.05378` (June 2026)
   publishes "pattern selectivity is not task-causal structure" as a title. **The
   developmental half of our version survives; the cross-model half does not.**
3. **Two Phase-1 framing statements are stale.** Metastability may no longer be an
   open problem (`2410.06833`), and the theory has had a **causal-mask version since
   Nov 2024** (`2411.04990`) — which Pythia, being decoder-only, should have been
   compared against all along. That one also carries a **Rényi-parking cluster-count
   prediction** nobody has checked on a trained model, and we have the counts on disk.
4. **The rung policy needs a rule 4 — a rung may be *externally* spent.** Pythia-1B is
   measured on the induction axis by two June 2026 papers. `p8_scale_ladder/lit-8.md`
   §3 argues this cuts *in favour* of keeping the reserve, with the registration
   recording what external measurements exist. **This is a human call and has not been
   taken.** **Pythia-1.4b is unaffected and is now the cleaner reserved rung.**

**`docs/LITERATURE.md` §6 lists thirteen cheap experiments this review surfaced**, all
re-analysis or weights-only. **§4 of that file is a paper** — five worked instances,
from five phases, of a null forced by its own instrument — and it needs no compute.

**The constraint the review ran under, because it bounds every claim in it:**
`arxiv.org` and every other scholarly host are **blocked by this session's egress
proxy**. `WebSearch` works; `WebFetch` and `curl` do not. So **nothing was read** —
every id and finding is from a search-engine summary, marked `[S]` or `[N]`.
**Working `docs/LITERATURE.md` §5 from a machine with arXiv access is the
highest-value unblocked task in the project.**

---

### The earlier resume block (2026-09-10 — 7d's open axes are CLOSED; `7e` is live; on a LoRA tangent)

*Still current for the measurement state.*

**Read `p7d_redundancy/status-7d.md` and `p7e_consolidation/design-7e.md`
first — both are current as of this session — then §3.12-V for the detail.**

**What this session did.** Closed 7d's two open axes and opened `7e`. Full
record in **§3.12-V**; the two headlines are that the redundancy set is **one
set with independence dead across 45 pairs**, and that the members are **aligned
at birth and fan out**, losing 56 % of peak alignment *while their signals grow*,
with redundancy surviving the separation. `7e` was created at the user's request
around **`L11H14`** and the **consolidation** question.

**The result most likely to matter elsewhere**, and it is a *hold* on an existing
instrument: **`L11H14` is anti-ordered by SVD** — keeping its smallest singular
directions beats keeping its largest at every rank, and rank-1 truncation is
worse than deleting the head. So **any SVD-ordered rank truncation misleads on
such heads, `induction_rank_sweep`'s `r*` construction included.** Do not quote
an `svd`-basis `r*` for a head not checked with `useful_rank.py --bottom`.
**§3.17 widens this**: there is a third class, *unordered*, where neither end of
the basis beats a random subspace — so "below its matched-norm control" does
**not** by itself mean anti-ordered, and only `--bottom` tells them apart. The
`schur` basis may be immune (eigenvalue-ordered, carries a sign) and the
comparison is **weights-only and free** — it is the cheapest next action.

**CURRENT TANGENT (user-directed, 2026-09-10): the LoRA induction-head project.**
Deliberate context switch, taken at this point *because* of §3.12-V4 — a project
that minimises induction-head rank with LoRA is directly relevant to a finding
that the members are rank-1-to-24 and one of them inverts. Two sources:
**read, and written up in §3.9-A** — start there, not at §3.9, which is
**stale on four points**. Remote `git@github.com:ZachData/Lora_inductionhead.git`
is the source of truth; the desktop clone was 57 commits behind. Headlines:
their **G3 is no longer ambiguous** (optimisation ruled out; reachability
upper-bounds any trained update); their **copying score is a readout failure**
that `tools/run/copying_score_sweep.py` fixes directly; and their re-probe
supplies the **ordered six-head pythia-70m cascade** that §3.14.4-A said it
needed and did not have. §3.9-A lists what to send them and what to take.
Expect to clear context and return here; **this file plus the phase docs
are the return path.**

**DECISION TAKEN 2026-09-10 — `p8_scale_ladder/` is open, and the ladder is the
new organising frame.** The sister project's *concepts and analysis* are
absorbed; its *training half* (EC2 spot, S3, `METRIC_VERSION` CI, the LoRA
fitting) stays upstream — this box has no GPU. The reason for absorbing is that
**everything 7d/7e found is `n = 1`**, and `design-7d.md` already reserved the
slot: a shared signature over independent circuits is a population claim no
single circuit can make. Pythia only, by user decision — the suite is already
wired in, shares one data order and one checkpoint schedule across sizes, and is
convenient enough that other families are not worth the setup cost.

**THE RUNG POLICY — explore low, validate high. Do not violate it by accident.**

| model | status | role |
|---|---|---|
| pythia-70m | untouched on the induction axis | **exploration** |
| pythia-410m | spent (7d/7e, §3.12) | **exploration** |
| pythia-1b | never measured, not in the registry | **RESERVED** |
| pythia-1.4b | Phase-1 phenomenology only (CLAIM-C) | **RESERVED** |

**No induction measurement on 1b or 1.4b until a prediction naming that model is
registered.** This is `check_registry` rule 3 applied *forward* rather than
discovered afterwards, and it is what lets the ladder produce **adjudications** —
of which `claims/adjudications/` holds zero against 39 registrations.

**Amendment proposed 2026-09-16, NOT TAKEN — a rung may be *externally* spent.**
The table's "never measured" is a statement about *this project*. It is no longer
true of the field: **`2606.02378` and `2606.05378` (both June 2026, same group,
public code at `skydancerosel/spectral-probe-circuits`) run pythia-1b on the
induction axis.** `p8_scale_ladder/lit-8.md` §3 argues this cuts **in favour** of
keeping the reserve — an independent measurement of the same rung by another
group turns an adjudication into a three-way comparison — provided the
registration records which external measurements exist and is written before
reading them. **This is a human call and has not been made.** **Pythia-1.4b is
unaffected and is now the cleaner reserved rung; prefer it for the first
registration.**

**1.4b needs no cleanup**, and a proposal to delete its analysis was dropped for
this reason: CLAIM-C measured it on `mass_near_1`, `effective_rank`,
`cluster_membership`, `cluster_count`, `cka_prev`, `fiedler_mean` — **no
induction quantity among them** — so it is already clean on this phase's axis.
Deleting it would have cost a registered claim and bought nothing.

**`P-I7`'s "not yet measured by this project" is NOT binding to the letter**
(user, 2026-09-10): written on a whim, still roughly stands, not to be enforced
rigorously. Under the rung policy it is satisfiable on either reserved rung.
**Still undecided and needing an explicit call:** adjudicate P-I7 on 70m before
exploration touches it, or send 70m to exploration and move P-I7 to 1b.

**The engineering gate on all of it is small and specific:**
`tools/run/induction_rank_sweep.py:83` sets `D_MODEL, D_HEAD, N_HEADS = 1024,
64, 16` at module level and **every 7d/7e runner imports them from there** —
read from `model.config` instead and the whole instrument suite becomes
scale-generic. Then add `PYTHIA_70M_REPO` / `PYTHIA_1B_REPO` to
`core/pythia_registry.py` (410m and 1.4b are already there). **Adding 1b to the
registry is not measuring it.**

**`pyproject.toml` was missing `p7d_redundancy` and `p7e_consolidation`** from
`packages` — the exact failure its own comment warns about ("a NEW phase's
package not being added… p7_motifs was missing while holding the current phase's
code"). Fixed 2026-09-10 along with `p8_scale_ladder`.

**70m is also the TRAINING rung** (user, 2026-09-10) — the only size this box can
train, so it is where **activations** and **custom checkpoints** come from when a
question needs a checkpoint Pythia never published. 1b/1.4b are validation only
and are never trained here.

**The existing 70m retrain is a FORK, not pythia-70m** — beyond §3.9's
"reachability, not development" caveat, its **dataset batching seeds differ**, so
it shares only checkpoint A (step 512) with the published model. That is
**fine for invariant 3** (a differently-seeded draw is exactly what
cascade-vs-recruitment wants) and **disqualifying for anything developmental**.
If a dense *and* faithful axis is wanted, rebuild it on **Pythia's published
batch order** rather than a fresh seed; **do not re-run the old fork protocol.**

**What to keep and drop from the sister project** is enumerated in `design-8.md`
("What to keep from the sister project"). Keep: the dense bracket, the 48-head
re-probe, the ordered cascade, the φ question, the reachability result, the
localization dissociation. Drop: the EC2/S3 infrastructure, the G0–G3/M1–M8 gate
lattice, the broken argmax copying score, the fork retrain protocol. **Record
only** (dead ends, kept so they are not rediscovered): `layer_host_plus_ln_final`
closed 8/8 on spot reclaims; the four negative G3 diagnostics; and Cor 17.2's
gradient-gating mechanism, ruled out because σ_OV is nonzero at all 2048 query
positions.

Read `p8_scale_ladder/design-8.md` for the six candidate invariants and the
sequencing, `status-8.md` for what has run — **the 70m rung, 2026-09-11**, and
§3.15 below for the one result that is not phase-8-local.

**LITERATURE SCAN RUN 2026-09-10 — `docs/literature_scan_2026-09-10.md`. READ IT
BEFORE THE NEXT MEASUREMENT.** It is **leads, not readings** — four searches,
no paper read, every arXiv id needs verifying. Its verdict is that **three of
§3.12-V's four headline findings sit in populated territory**:

- **"SVD ordering is a poor proxy for causal importance" is not new** — `FWSVD`
  (**2207.00112**, ICLR 2022) states the general form, and follow-ups report
  non-monotone degradation when truncating by singular-value magnitude. Ours may
  survive as *stronger* (anti-optimal, **negative** recovery at `r = 1`, versus
  their suboptimal) but that is a narrower claim than it looked.
- **Super-additive co-ablation is the self-repair signature**, not a new
  phenomenon — *The Hydra Effect* (**2307.15771**) predicts exactly the sign 7d
  measured in 44/45 cells. And **2607.01940** (2026) looks structurally like
  `pairwise_interaction_matrix.py`. **Read that one first.**
- **The developmental axis is populated and recent** — **2502.14010** reports a
  head's induction score falling as another rises (§3.12-U from the other side),
  and **2606.02378** (2026) tracks developmental trajectories across three
  1B-class models **including Pythia-1B**, one of our reserved rungs.

**What still looks distinctive**, best card first: **causally-defined membership
across all 384 heads with the demonstration that structural proxies fail**
(§3.12-R, §3.12-G6, §3.12-V1); the **measured-null discipline** (§3.12-V3's
ambient PR of 22/1024 kills the isotropic baseline; §3.12-V5's
changing-membership artifact); the **heterogeneity within one set** (rank-1
gain-ordered `L7H8` beside full-rank anti-ordered `L11H14`, same job, still
substitutable while going orthogonal); and anti-optimality as a strengthening of
FWSVD.

**Consequence for the plan: do NOT build a paper on "induction heads are rank-1"
or "the redundancy set is super-additive".** Both are known. The ladder plan
survives — turning `n = 1` into a population claim was always the point — but
check what **2606.02378** already measured on Pythia-1B before assuming that rung
is untouched by the field. The `lora_ind` merge case is **unaffected**: its value
is the dense onset axis and the φ question, neither of which the scan touched.

**Committed 2026-09-11**: the de-hardcoding, the 70m/1b registry entries, the
`--model` and `--ablation` flags across all six 7d/7e runners, the 70m rung, and
§3.15. Working tree clean at that commit except `data/hf/` (HF cache, not for
version control — note its `.no_exist/` marker files are *not* matched by
`.gitignore`, so never stage `data/` with `git add -A`).

**Results are git-ignored by design.** Every `data/analysis/*.json` this phase
wrote — `redundancy_catalog_pythia-{70m,410m}_*`, `useful_rank_pythia-70m_mean`,
`pairwise_interaction_matrix_pythia-70m_mean`,
`member_subspace_geometry_pythia-70m_mean`, `ambient_budget_pythia-70m_mean`,
`member_formation_curves_pythia-70m_mean`, `p8_rung_comparison.json`, and the
2026-09-12 `freq`-probe arms `pairwise_interaction_matrix_pythia-70m_mean_freq`,
`useful_rank_pythia-70m_mean_freq`,
`useful_rank_pythia-70m_mean_freq_bottom`, `fv_score` (§3.18), and the §3.19
geometry trajectories `member_subspace_geometry_pythia-{410m,70m}_mean` and
`member_subspace_geometry_pythia-70m_mean_freq` — lives
only on this machine. `status-8.md` carries the numbers; the JSON carries the
rows. Re-running is cheap at 70m and ~25 min for a 410m full sweep.

**Machine note:** the geometry and rank runners were **killed for memory twice**
while `falsification/e4_bootstrap.py` (~5 GB, not this project's) was running.
Use `--chunk 2` and `OMP_NUM_THREADS=4` on this box; outputs are written per
step, so `--append` finishes an interrupted grid.

#### The earlier resume block (git state, branches, `P-I7`) — **branch table updated 2026-09-12**

**Git. The backlog is gone.** PRs #26–#32 all merged, `main` is at `32370fd`,
and the four-branch stack (`p-i1-build-null-score`, `spectral-dissipation-infra`,
`induction-programme-stage012`, `p2b-per-head-figure`) is **merged and its
branches deleted**.

**Six remote branches exist**, and two of them must not be deleted:

| branch | state |
|---|---|
| `claude/rescaler-cache-identity-test` | **PR #36**, base `main` — was "+38, PR not opened"; opened 2026-09-12 |
| `claude/probe-ceiling-svd-third-class` | **PR #37**, based on #36 |
| `claude/fv-head-division-of-labour` | **PR #38**, based on #37 |
| `claude/invariant4-set-level` | **PR #39**, based on #38 — the tip |
| `claude/particle-methods-comparison-vpuads` | **KEEP** — all of Phase 1d (`p1d_cluster_ensemble/`), on no other branch |
| `claude/visualize-mets-results-sl2ya5` | **KEEP** — `tools/visualize_latest.py`, on no other branch |

The two `KEEP` branches look deletable and are not: both are from August, ~93
commits behind, one commit ahead. `INDEX.md`'s "In flight on other branches"
section is the record, and `docs/deleted-branches-2026-09-10.md` says it again
next to the 21 branches that *were* deleted (with their SHAs, restorable by
`git push origin <sha>:refs/heads/<name>`).

**The `gh` CLI IS installed and authenticated** (verified 2026-09-12: `gh pr
list`, `gh pr view`, `gh pr create --base <branch>` all work, and PRs #37/#38
were opened with it). *This line used to say "No `gh` CLI — open PRs from the
web compare URL"; that was true when written and is not now.* The compare URL
`https://github.com/ZachData/MetastableStateAnalysis/compare/main...<head>?expand=1`
still works as a fallback.
**The repo's `github_key` is DEAD** — push with the default `~/.ssh/id_ed25519`,
plain `git push`, no `GIT_SSH_COMMAND`.

**CodeRabbit does NOT review this repo automatically** — its own comment says
*"This repository does not receive automatic reviews because it has fewer than
10 stars."* Every PR needs the **"🔍 Trigger review" checkbox** ticked on
CodeRabbit's first comment, by hand. A PR sitting with no review is the
expected state, not a failure.

**PRs are stacked, not piled.** `--base` a PR on the branch below it rather
than on `main` when the work builds on unmerged work: #36 → `main`, #37 → #36's
branch, #38 → #37's branch. Each diff is then the one commit a reviewer
actually has to read, which is what `CLAUDE.md`'s "open PRs at natural
boundaries" is asking for.

*Merge note, in case `PROJECT.md` conflicts again.* Merging `main` into this
branch conflicted in five hunks, all of them this file's handoff header. Every
one resolved to **this branch's side** — `main` carried the superseded
2026-09-08 resume block — and the resolution was checked line-by-line rather
than assumed: the 29 lines only `main` had were all stale handoff text, except
the earlier per-step output families, which were merged back into the
git-ignored-outputs bullet rather than dropped.

**`P-I7` IS REGISTERED** (H-BRIDGE, `needs-null`) — the first registration in a
long while. Content matchers' static QK becomes symmetric, positional matchers
stay at baseline. **It may only be adjudicated on a model this project has NOT
measured**: the 410m artifact is spent under `check_registry` rule 3. pythia-70m
under §3.9's grid is the site, and §5.3 records that its weights are not on disk.
Registry: 39 predictions, `claims/adjudications/` still empty.

**THE LIVE THREAD IS `7d`, the redundancy catalogue (§3.14.2, active).** Its
premise was restated the day it was queued: §3.12-O/P killed the three-stage
reading and §3.12-S killed the serial reading of the pair. The object is a **set
of functionally redundant, structurally distinct heads** holding a
residual-stream regime — not a chain of stages.

**`7d` now has a directory: `p7d_redundancy/`** (2026-09-10). §3.14.1 created
`7a`–`7d` as labels for *this file's* organisation and said they were not phase
directories; that held until `7d` went active and started producing results,
and it was the only live phase without a `status-N.md` while its runners sat in
`tools/run/`, which `INDEX.md` calls one-off scripts. `7a`/`7b`/`7c` are still
labels with no directory. Read `p7d_redundancy/status-7d.md` for what is
answered and `design-7d.md` for why the instrument is causal — those two are now
the fastest way into this thread, and §3.12-S/T/U remain the detail.

**What 7d already knows (do not re-derive):**
- **Q1 is answered** (§3.12-T): the set is **~4 substantial members, ~10 with any
  effect, of 384**; median head moves the readout by **0.001**. Members:
  `L5H2` +1.97, `L7H8` +1.02, **`L12H5` +0.42**, `L8H6` +0.21, `L11H14` +0.19,
  `L8H9` +0.13, `L15H14` +0.09, `L7H1` +0.07, `L9H13` +0.06, `L10H9` +0.05.
  **`L12H5` and `L8H6` were entirely unknown before today.**
- `L5H2` × `L7H8` interact at **+4.15** (joint 2.2× the parts-sum) — redundant,
  not serial — with **87 %-aligned residual effect from chance-level weight
  overlap** (§3.12-S).
- The effect **compounds** down the stack (23× for `L7H8`), and
  norm-proportionality is **ruled out and inverted** (r² = 0.001, §3.12-R).
- **Q2/Q3 are answered** (§3.12-U): members form at **different** times —
  `L5H2`/`L11H14` in `(512, 1000]`, the model's own induction interval;
  `L7H8` **last**, in `(2000, 3000]` — the **interaction is ≈ 0 until `L7H8`
  arrives**, so the redundancy postdates both heads, and four of six members
  **decay to 5–22 % of their peak** while `L7H8` alone rises monotonically.

**THIS IS THE THREAD TO WORK ON.** Stated by the user, 2026-09-10: *"I really
want to work on that. That's what I'm personally most interested in. I want to
see what all these induction heads are doing and to see if there's any
relationship between them and why they're all so independent, but they all seem
to form at the same time."* Those are three questions, they are all live, and
**§3.14.4 turns each into the specific measurement that would answer it.** Read
that section before picking an action. Two of the three are already partly
answered and one of them contains a false premise worth knowing about — the
heads are **not** known to be independent; the only pair ever measured is
strongly *redundant*, and every other pair is simply unmeasured.

**NEXT ACTIONS, in order.** 1 and 2 serve the user's questions directly and are
the reason for the ordering; 3 and 4 were the previous plan and still stand.

1. ~~**Pass 2 of the catalogue — the pairwise interaction matrix.**~~ **DONE
   2026-09-10**, §3.12-V. `p7d_redundancy/pairwise_interaction_matrix.py`, top
   10, 45 cells, steps 16000 and 143000, restore exact, no cell against the
   ceiling. **44/45 positive at 16000, no block structure — one set, and the
   independence premise of §3.14.4-B is dead.** But **74–81 % of the interaction
   is the product of the two heads' own magnitudes**, and δ-cosine explains only
   7–9 % of it: direction and substitutability are **decoupled**, which is
   §3.12-S's single-pair dissociation generalised to 45 pairs.
   The **geometric axis is also done** (`member_subspace_geometry.py`, 13
   checkpoints, with a *measured* null from three near-median control heads):
   members are **born aligned** (step 1000, the only extant pair at cosine 0.857
   and centered CKA 0.693 vs null 0.128), then **fan out** — fixed-15-pair mean
   cosine peaks at step 5000 (+0.744) and falls to +0.327 at 143000 **while
   delta norms grow 671 → 1093**, so it is not a fading-signal artifact. **No
   member's subspace expands** (PR stays in 8–28). And **redundancy survives the
   separation**: `L5H2`×`L11H14` holds interaction +2.18 → +1.53 while its
   cosine goes 0.344 → **0.004**. Full detail: `p7d_redundancy/status-7d.md`.
2. **The step-1000 circuit is a different circuit, and it is now the cheapest
   open question.** §3.12-U found that at step 1000 the mechanism is `L5H2`
   (+4.97) **and `L11H14` (+3.57)**, with `L7H8` absent and every other member
   under +0.06. `L11H14` is the **top copier in the model at 143000** (0.723,
   §3.12-O2) but it does not enter the copying top-5 until step **16000**, and
   `copying_score_sweep.json` stores only the top ten and a named set, so **its
   step-1000 copying score is not on disk** — measuring it is step one, and it is
   a weights-only quantity. Then run the same 2×2 for `L5H2` × `L11H14` across
   1000/2000/3000: serial or redundant? If serial, the three-stage reading
   §3.12-P falsified at step 16000 may be **true early and dismantled later**,
   which is a different claim and a registrable one.
   **A one-step pilot is already on the board and it must not be over-read.** At
   step 1000, `--heads L5H2,L11H14 --pair L5H2,L11H14 --steps 1000` (restore
   exact) gives singles +4.973 and +3.567, joint **+6.859**, interaction
   **−1.682**, δ-cosine +0.857 — *sub*-additive, the serial signature, and the
   opposite sign to the `L7H8` pair. But the joint arm lands at NLL **11.77
   against a uniform ceiling of 10.83**, so it is outside the readout entirely,
   and §3.12-M5 warns that this readout's compression biases independent
   contributions toward exactly this apparent sub-additivity. **The sign here is
   not evidence.** Settling it needs the graded readout (§3.12-M's KL / λ scale)
   before more steps are run at raw `ΔNLL`. Use `--out`: a partial run overwrites
   the six-member curve otherwise.
3. **Structure per member** — mostly **already on disk and unread along this
   axis**: `qk_symmetry_sweep.json`, `ov_per_head_series.json`,
   `copying_score_sweep.json`, `behavioural_series.json`, all 384 heads × the
   checkpoint grid. The catalogue now says which rows to read, and §3.12-U says
   which *steps* matter — the action is in `(512, 4000]`, not at the endpoints.
4. Still unrun, and still the only *designed* intervention that is: the **§2.5
   isometric path** on `L7H8` (`M(t) = γ(t)Σγ(1−t)ᵀ`, exact isometry, rank `k`,
   sweeps 100 % repulsive → attractive → repulsive). **`7e` is a better use of
   it** than `L7H8` alone — see below.

5. **`7e` — `p7e_consolidation/`, opened 2026-09-10 at the user's request.**
   Two objects, one phase: **`L11H14`**, which four independent 7d measurements
   single out (mean cosine to the set **−0.033** at the *third-largest* delta
   norm, against +0.389 for magnitude-matched `L7H1`; effect subspace PR ~50 vs
   everyone else's 8–28; earliest defector, breaking away between steps 2000 and
   4000); and **consolidation** — can the set be collapsed into a single head,
   removing the redundancy, and is the result more interpretable?
   **The first measurement is already run and it refuted this phase's own
   opening argument** (`ambient_budget.py`): reaching 90 % of the joint effect
   needs **355 ambient directions** against an ambient participation ratio of
   **22**, so the set writes into directions the residual stream barely uses —
   private low-variance bandwidth, not the shared trunk. Only **73–78 %** of the
   joint energy fits one head's rank-64 budget. The phase survives because
   **energy is not usefulness**; the capacity question is causal
   (`dNLL(rank k)`), not geometric. Read `p7e_consolidation/design-7e.md`, which
   also records why naive weight-copying between members is ill-posed — they sit
   in **different layers**, so a transplanted OV would be driven by the wrong
   attention pattern.
   **The causal gate is now run too (`useful_rank.py`), and it splits the set.**
   `r*` — the OV rank recovering 90 % of a head's own causal effect, of 64:
   `L7H8` **1**, `L12H5` **1**, `L8H9` **2**, `L5H2` **12**, `L8H6` **24**,
   `L11H14` **64**. **`L7H8` recovers 97 % of its effect from a single
   direction.** The five low-rank members sum to **40, inside one head's
   budget** — consolidation of the aligned core is viable. **`L11H14` is the
   exception on a fifth independent axis**: its recovery curve sits *below* the
   matched-norm random control at nearly every rank, and at `r = 1` recovery is
   **negative** — the Eckart-Young optimal truncation is worse than deleting the
   head. Singular-value magnitude **anti-orders** its causal usefulness, which
   is §3.12-R and §3.12-G6 reproduced *inside a single head*. Its orthogonality
   is load-bearing and it cannot be folded in.
   **Confirmed directly** with `useful_rank.py --bottom`, which keeps the
   *smallest* `r` directions: for `L11H14` the ordering is **bottom-`r` >
   matched-random > top-`r` at every rank** (top-1 **−0.096**, bottom-48
   **+0.846**), while `L7H8` is textbook (top-1 **+0.971**, bottom-48 +0.095).
   **Two classes, not a gradient** — `L7H8`/`L5H2` are gain-ordered, `L11H14`
   differs in kind. **Methodological warning that reaches outside 7e: any
   SVD-ordered rank truncation misleads on `L11H14`-like heads, including
   `induction_rank_sweep`'s `r*` construction.** Its `schur` basis orders by
   eigenvalue and carries a sign so it may be immune; the comparison is
   weights-only. Until then, do not quote an `svd`-basis `r*` for a head not
   checked with `--bottom`.

**Constraints 7d must not lose** (§3.14.2): weight-space overlap is **not**
function-space overlap, so membership is defined causally and every composition
score in §3.12 is blind to it; §3.13's report-both rule, which §3.12-J showed
applies **per training stage**; the **spent-artifact rule** — nothing 7d touches
on 410m can later be registered *and* adjudicated on the same data; and the
particle-dynamics half stays blocked on `dual_reading`'s pairwise field (P-I5).

**Three open code defects** (§3.14.3, all reporting-only, none touching a
p-value): `truncate`'s `random` docstring claims matched-norm and is off 16× at
`r=1`; `target_vs_reference`'s self-inclusive z saturates at 3.75; `ov_factors`
returns the **transpose** of the residual operator with no warning against
directional reads.

**`data/analysis/*.json` are git-ignored** and every one of them is on disk and
regenerable from its `.py` producer in `tools/run/` or `data/analysis/`. The
2026-09-09 batch, fourteen outputs: `induction_diagnostics_7b`,
`induction_abscissa_7b`, `induction_spectral_predicts_7b`,
`induction_composition_whitening`, `qk_symmetry_sweep`,
`induction_position_profile`, `induction_sa_pilot`, `copying_score_sweep`,
`three_stage_mediation`, `what_l7h8_writes`, `norm_proportionality`,
`two_big_heads`, `redundancy_catalog`, `member_formation_curves`. The earlier
per-step families, carried because they are the expensive ones to lose:
`induction_rank_sweep_s*`, `induction_qk_sweep_s*`,
`induction_subspace_characterize_*`, `induction_developmental_series`,
`dissipation_v2_series`, `dissipation_sublayer_series`.
`member_formation_curves.json` is the 23-step, six-member curve of §3.12-U and
costs ~20 min to rebuild: `python -u p7d_redundancy/member_formation_curves.py --top
6 --chunk 4`, then `--append --steps 3000,5000,7000,9000` for the fills.

### The machine

| | |
|---|---|
| Repo | `/run/media/system/WDS_500/Mets` (NVMe, `/dev/nvme0n1p1`, 458 GB) |
| venv | `<repo>/.venv` — Python 3.14.7, torch 2.13.0+cpu, transformers 4.57.6, numpy 2.5.2, scipy 1.18.1 |
| CPU / RAM | 16 cores, 31 GB |
| Free | 95 GB on WDS_500, 440 GB on HDD_1TB |

### The tree

Everything generated lives under the repo, on one root. `data/` is git-ignored
by `*`.

```
Mets/
├── data/                            # all generated bulk
│   ├── hf/                 51 GB    # HF_HOME — 33 mirrored pythia-410m revisions
│   ├── phase12/           118 GB    # METS_RESULTS_DIR — phase 1 and phase 2
│   ├── phase7/            6.1 GB    # the 19 interaction tables
│   ├── analysis/                    # curve.json, formation_series.json
│   ├── logs/
│   └── superseded/phase7_float32/   # 1.1 GB, pre-float64 tables
├── results/               132 GB    # the PILOT grid — §5.2, DO NOT DELETE
└── tools/run/                       # sweep.sh, curve.py — tracked
```

`METS_REPO` and `METS_DATA` are the only two overrides. There is deliberately no
`METS_VOL`: it named a VM scratch volume, which is the class of path that
encodes transient infrastructure and fails silently when the infrastructure
changes. Both run scripts derive everything from `METS_REPO`.

`transformers` is pinned `<5`. On 5.x GPT-NeoX moved rotary parameters into
`config.rope_parameters`, `core/rope.py`'s `rotary_pct` default then fires, and
it reports `rotary_ndims=64` where pythia-410m rotates 16.

### Traps this machine sets

**The gate run from a worktree can test the MAIN tree's code and pass.**
Forty-four modules define `REPO = Path(os.environ.get("METS_REPO",
"/run/media/system/WDS_500/Mets"))` — a hardcoded absolute default — and
twenty-five of them follow it with `sys.path.insert(0, str(REPO))`. Importing
any one of them puts the main tree at the front of `sys.path` for the rest of
the process, so everything imported afterwards resolves there instead of in the
worktree. Whether it bites depends on collection order, which is why it is
invisible most of the time: on 2026-09-19 `./scripts/check.sh gate` in
`../Mets-claim-c` reported `cannot import name 'BETA_SUBEXPERIMENTS' from
'/run/media/system/WDS_500/Mets/p1c_frames/run_1c.py'` — the file being tested
was in the worktree, the file imported was not. **Run the gate from a worktree
as `METS_REPO=$PWD ./scripts/check.sh gate`** (2384 passed that way, a
collection error without it). The real fix is for those modules to resolve
`REPO` from `Path(__file__).resolve().parents[N]` with the environment variable
as an override rather than the other way round, as `tools/score_claim_c.py`
already does; it touches 44 files and has not been done.

**`source .venv/bin/activate` can succeed and give you the wrong interpreter.**
`activate` carries the absolute `VIRTUAL_ENV` recorded at creation. If the repo
has moved, it prepends a directory that does not exist, sets the variable, and
returns 0 — and `python` falls through `PATH` to whatever else is installed (here
a conda env at `miniforge3/envs/mets`, Python 3.10 with a pre-4.45
transformers). This cost a phase-7 checkpoint computed against the wrong library
with nothing in the artifact to record it: the phase-7 manifest stores
`git_sha`, `hf_revision` and `seeds`, but **no library versions**.
`tools/run/sweep.sh` now asserts `sys.prefix` and the torch/transformers
versions rather than trusting activation. Check `sys.prefix`, never
`VIRTUAL_ENV`.

**"No output yet" is not evidence a job died.** The check is `pgrep`, not the
log — and write the pattern as a real ERE, since `pgrep -f "a\|b"` matches
nothing and reports success. Two writers on one temp path can replace a good
table with a corrupt one.

Both of the other traps this repo has hit are now guarded in code with tests:
the phase-1/phase-2 reuse selector identifies each phase by a file only that
phase writes (`tests/test_run_scripts.py`), and `tools/recompress_tables.py` no
longer globs its own temp file (`tests/test_tools_recompress_tables.py`).

---

## 2. Where the work stands

Active work is **Phase 7** — the mechinterp/particle bridge. **The live thread
is the bottom-up induction programme (§3.11)**: the co-location / `relay`-motif
frame (`P-I1`) hit a construction-level circularity and was retired (below,
§3.10, `POPPER_PLAN.md` §6w); it was replaced by isolating one induction head
(`L5H2 → L7H8`) and characterising its circuit directly. As of 2026-09-08
Stages 0–2, the QK half, and a generalisation batch have all run — everything
**exploratory, nothing registered**. **2026-09-09 (§3.12) reopened the design**:
the OV repulsive collapse is a model-wide developmental phase with its floor in
`CLAIM-B`'s own anchor window, §3.11's baseline argument used the wrong
reference class, and the Stage 3 control needs rebuilding. Next action is
§3.12's diagnostics, not a registration. §3.11 and §3.12's dated blocks are the
detail; §§3.6–3.10 below are the retired-frame history, kept as the
construction log.

**Read `docs/LITERATURE.md` before writing any of this up (2026-09-16, §3.37).**
Every phase now has a `lit-N.md`. Three headline results are in populated
territory — Phase 1's developmental arc (`2509.23024`), 7d's second-order
ablation instrument (`2607.01940`), and 7d's "behavioural proxies fail" thesis
(`2606.05378`) — and two of Phase 1's framing statements are stale. What survives
is ranked in that file's §3; the per-claim notes are inline at §3.12-V and
§3.14.4-A below.

`P-I1`, induction-head formation as a two-stage `relay` motif tracked across the
checkpoint axis, ran end to end and scored **INSUFFICIENT** — not
falsified, not validated — at both a 50- and a 100-replicate null (§3.6, §3.7).
The K = 100 rerun (§3.7) settled the sensitivity question: the verdict is
stable but **the p-value is not** (0.14 at K = 50, 0.89 at K = 100, same
observed statistic) — the pairing null is near-degenerate on a 36-head
change-centroid tie coset, so P-I1's arm returns the verdict but cannot
quantify how insufficient. `INDEX.md`'s phase table is still accurate for
everything else.

**Two analysis tracks, run 2026-09-05/06** (`docs/`, §3.8): a co-location panel
putting the Phase 1 / Phase 2 global observables on the P-I1 19-step axis
(`data/analysis/colocation_panel.*`), and the dissipation-identity
decomposition — Tier A (`data/analysis/dissipation_series.json`) and Tier B
(`data/analysis/dissipation_sublayer_series.json`, the exact attn/FFN split +
per-head roll-up). **Tier B's headline**: the per-head attention-dissipation
series breaks §3.7's tie coset and `|d_attn_repulsive|` co-locates with the
behavioural induction rise at a *stable* p ≈ 0.004 (CO-LOCATES) — but this is
**exploratory** (three anchors tried, no differential falsifier registered
first), so it is a hypothesis to register and re-test, not an adjudication.
`claims/adjudications/` stays empty.

**The co-location programme was stopped on 2026-09-07 and NOT registered**
(§3.10, `POPPER_PLAN.md` §6w). `relay` membership requires attractive-channel
edges defined by `U_pos` — the same OV Schur projector whose sign split the
proposed prediction would measure — so the obvious entry would have
"confirmed" at `p_less` = 0.018 a claim its own head-selection presupposes.
Stripping the filter moves into the *other* tautology `find_relays` guards
against, so the circularity is a property of the motif's construction. Work
has pivoted to a bottom-up interventional programme (§3.11).

**Open, in rough priority order** (for the next session):
1. **The bottom-up induction programme** (§3.11) — isolate one induction head,
   find the minimal rank `r*` that carries induction, characterise its shape,
   then perturb it. Replaces the population/co-location frame entirely: no
   projector-defined population, so no §3.10 circularity, and `n = 1` is
   sound because the null randomises over *subspaces* rather than units
   (`claims/EVALUABILITY.md`'s own unused advice). **Stages 0–2 have run,
   plus a generalisation batch** (§3.11, dated blocks). Settled: the
   representational picture — OV core 100 % repulsive, non-normal but not
   outlier, no token-identity copy structure — is **universal** across the top
   8 behavioural induction heads and stable across development; `r*_SVD ≪
   r*_Schur` holds across the whole trained regime. The causal OV→copying
   effect is **concentrated in L7H8** (~10× any other). The **QK half is done**
   and is the spectral opposite — the matcher is a ~12-dim high-eigenvalue
   near-normal invariant subspace (`r*_Schur` 12 < `r*_SVD` 32, SVD below the
   random control). **Pick up at: the four diagnostics of §3.12 block E** —
   §3.11's registration advice is superseded by §3.12. The reference class was
   wrong ("100 % repulsive" is 9/9 against *the induction heads*, but 0.435 →
   0.109 of the 384-head population at matched step), the matched-norm control
   is degenerate, and the drafted Stage 3 entry has three defects decidable before a
   forward pass. Diagnostics first, then the §3.12-D **S-flip** entry, then
   **Stage 3**. `claims/registry.json` unchanged.
2. ~~Violation-restricted subspace split (dissipation v2)~~ **— done
   2026-09-08, `status-2.md` item 5 ~resolved.** The clean per-particle
   version (`v2_attn_pos_*` in `tools/run/dissipation_sublayer.py` →
   `data/analysis/dissipation_v2_series.json`) does **not** reproduce Phase
   2's `frac_repulsive` decay: the energy-weighted repulsive share of the
   positive first-order term at ΔE>0 boundaries stays 0.6–0.9 across the
   trained regime. So `frac_repulsive` (a count with a hard `>0.5` per-
   violation threshold) falls because marginal violations drift across the
   line, not because the energy leaves the repulsive subspace. Nothing
   further for the dissipation runner here.
3. **The pythia-70m dense-onset sub-study** (§3.9) — new registered ground,
   fills the 512 → 1000 gap, would let §3.8's ODE-regime finding be checked at
   4-step resolution. Now also the natural second model for anything §3.11
   produces.
4. **Phase 2's runner needs a manifest** (`_write_run_manifest`) —
   `docs/results_provenance_audit_2026-09-05.md` §3.1.

*(The former item 3, "per-head OV Schur projectors", is **done** — the
machinery already existed in `p2b_imaginary/head_circuits.py` and the sweep is
`data/analysis/ov_per_head_series.json`. See §3.10.)*

**Committed 2026-09-08, on `claude/rescaler-cache-identity-test`, not yet
pushed / no PR.** Commits `b44c3e9`..`HEAD` (10): the `head_spectrum` sign
split, the dissipation runners + v2, `ov_per_head`, the induction programme
Stages 0–2 + generalisation batch, the provenance docs, and the PROJECT /
POPPER / status-2 sync. `./scripts/check.sh gate` was green before the batch
(2270 passed / 5 skipped). The tracked new code is under `tools/run/` and
`data/analysis/*.py`; every `data/analysis/*.json` series is git-ignored (§6)
and reproducible via §7.

**The registered 19-step sweep is complete.** All 19 interaction tables are on
disk under `data/phase7/`.

**The degeneracy that blocked `P-I1` has cleared.** On the twelve-step CLAIM-B
grid every head's change centroid was one number, so the pairing null permuted a
constant and the attainable floor was 1.000. The five registered log-spaced
fills inside (1000, 54000) fixed it:

| `relay_owner` | heads scored | distinct centroids | span (log-step) | sd |
|---|---|---|---|---|
| `tag_writer` | 102 | 68 | 3.8898 – 4.9439 | 0.2813 |
| `matcher` (registered) | 116 | **79** | 4.1604 – 4.9439 | 0.2374 |
| `both` | 122 | 86 | 4.1604 – 4.9439 | 0.2641 |

The relay counts behind it:

| step | relays | ex-`repeated_tokens` | heads (matcher) |
|---|---|---|---|
| 0 – 2000 | 0 | 0 | 0 |
| 4000 | 15,030 | 5,563 | 9 |
| 8000 | 232,568 | 83,659 | 25 |
| 16000 | 509,646 | 216,528 | 46 |
| 32000 | 1,176,478 | 582,796 | 63 |
| 54000 | 2,560,483 | 1,008,553 | 80 |
| 143000 | **2,407,556** | **1,465,052** | **114** |

**Two things that must not be skipped when this is scored.**

*The series is not monotone.* The total relay count FALLS from 54000 to 143000
while heads carrying relays rises 80 → 114 and the ex-`repeated_tokens` count
keeps climbing — the signal spreads across heads and away from the one
combinatorially-loaded prompt while the raw total drops. `change_profile`
rectifies, so that decline lands in `reverse_change_mass` and will inflate
`noise_mass_share_estimate` on a series whose reverse motion is real structure.
That field is documented "REPORTED, NEVER SCORED"; this is the case that earns
the distinction.

*"79 distinct centroids" is not 79 classes.* It is **77 singletons, one class of
three, and one class of thirty-six** — 31% of the heads still put their change in
a single interval. Harmless at 116 heads and not harmless at forty; see §3.2.

---

## 3. `P-I1`: built, run, and scored — INSUFFICIENT (2026-09-04)

**The relay-count null did not exist through 2026-09-03.** `formation_gate`
requires the series to be the excess above a null envelope, and
`core/qk_offset_null.py` computes that for the **QK antisymmetry statistic**,
not for relay counts. `formation_curve.assert_gate_ready` refused the raw
series, correctly. §3.1–§3.4 below is the construction log, kept as it happened
rather than rewritten now that §3.6 has the answer.

`claims/EVALUABILITY.md` prescribes the order — compute the attainable floor,
name what the statistic degenerates on, check what the measurement grid
contributes, and only then build the control. **All three steps before the
control are done.** `POPPER_PLAN.md` §6t is the write-up.

* **Step 2.** Across the 8 battery prompts at step 54000 the raw relay count
  against the prompt's own induction-pair supply runs **r = +0.9958** — 99% of
  the cross-prompt variance is the prompt's combinatorics, not the model's
  circuitry. Excluding `repeated_tokens`, +0.8908. Nothing else is close:
  n_tokens −0.39, n_same_content −0.36, n_distinct_tokens −0.79.
* **Step 3.** §2's table is the answer: the grid contributed the entire previous
  failure and the fills fixed it.
* **Step 1.** `claims/audits/p_i1_attainable_floor.json`. Two findings, below.

### 3.1 The gate cannot score the axis the pipeline builds

`formation_curve_payload` takes its head axis from the **behavioural** series,
dense over all 384 heads (24 × 16), and zero-fills the relay side. But
`paired_colocation_arm` calls `change_profile` on every unit with **no per-unit
skip**, and `change_profile` refuses a series with no rise. 116 heads carry
relays and **268 never do**, so the arm refuses on the first all-zero unit and
`p_value_p_i1` returns **no p-value at all**. On the 116 forming heads the
identical input emits.

The message names none of it — "the series has no rise anywhere in the sweep",
no arm, no head index, no unit count. Pinned as it is, in
`tests/test_p_i1_attainable_floor.py`.

**Fixed, both halves, 2026-09-04 — the author's decision was both, not
either.** Pre-filter the axis to the 116 heads that carry a relay anywhere in
the raw sweep (a static, pre-registrable population — §3.4's null still runs
on the raw series before subtraction, so nothing here depends on the null),
**and** give the arm a per-unit skip for whatever residual heads have zero
above-null EXCESS once §3.4's null is subtracted (a head can be in the 116 and
still have the null absorb its entire signal). `paired_colocation_arm` and
`p_value_p_i1`/`adjudicate_p_i1` now take `skip_no_rise: bool = False` —
default off, CLAIM-B untouched, `p7_motifs/formation_gate.py` — dropping a
unit only on `change_profile`'s "no location to measure" refusal specifically,
reporting the count as `n_skipped_no_rise` on the arm's own record and naming
it in every refusal the arm can still raise afterward. The real scoring call
is `tools/score_p_i1.py`, which pre-filters to the forming axis and passes
`skip_no_rise=True`.

### 3.2 The pairing arm's floor has two halves

Permuting units within a class of equal change locations leaves
`-mean|ca - cb[p]|` exactly unchanged, so every pairing ties a coset of order
`prod(m!)` and no input can express a p below `prod(m!) / n!`. The arm reported
`1 / n_draws` alone. At nine of ten units sharing one location it reported
**0.000500** against an attainable **0.100000** — 200×, above α, emitted with no
refusal. Seven of ten tied is 0.00139 and emits legitimately: **the halves cross
within two units.**

`core.changepoint_colocation.pairing_floor_report` now owns both halves and the
arm refuses on the max. On the real head set the tie half does not bind (116
heads, tie floor 1e-148); it binds on the set a relay-count null *leaves*.

### 3.3 What that constrains — the point of doing it first

A relay-count null turns the series into an above-null excess, and a head whose
excess stops rising leaves the scored set. So the null chooses `n_units`, and
`n_units` with the tie structure chooses the floor.

| survivors | max tied | tie floor there |
|---|---|---|
| 4 | 1 | 0.0417 |
| 6 | 4 | 0.0333 |
| 8 | 6 | 0.0179 |
| 12 | 10 | 0.0076 |
| 19 | 17 | 0.0029 |
| 20 | **19** | 0.0500 |

Not monotone: `k = n − 1` gives exactly `1/n`, so all-but-one-tied clears 0.05
from n = 20 upward and fails at n = 19. Full table in the record.

> **The relay-count null must leave at least four heads with a rising above-null
> excess, and among them no more than k sharing one change location.**

### 3.4 The null — built 2026-09-04, degree-preserving at the head level

The author's decision, walked through and registered rather than started from
the code: degree-preserving at the **head** level, not per particle.
`p7_motifs/relay_count_null.py`. `pair_type` and `offset` are pure facts about
where an edge points, given the prompt's tokenisation; `attractive_frac` /
`repulsive_frac` / `force_magnitude` / `weight` are facts about its force —
independent axes of the same edge-row. So the null is a payload shuffle: for
each (prompt, layer, head), draw `len(group)` DISTINCT positions uniformly at
random from the prompt's full causal pool and reattach each real edge's entire
force-derived payload to it unchanged, recomputing `offset`/`pair_type` from
the new position.

This holds `n_induction` fixed per prompt automatically (the pool and the
induction/strict/same-content candidate sets are properties of the prompt's
tokenisation alone, `PromptNullContext`, identical at every checkpoint and
replicate — no separate bookkeeping needed), and preserves each head's edge
count and its **entire** force distribution exactly, not just an aggregate like
"attractive fraction". Per-particle in/out-degree is NOT held fixed — a
heavier double-edge-swap configuration-model null was considered and not
chosen. The relay count itself, a two-edge composition rather than a single
masked edge, is scored by Monte Carlo — reshuffle, rerun
`find_relays`/`per_head_relay_strength` unchanged, K replicates → mean/sd —
rather than a closed form, to avoid re-deriving the composition's null
distribution by hand. 18 tests, including a planted-relay oracle and
calibration on a structureless table; caching the null's per-prompt grouping
(a ~6.8× speedup, needed to make the real run feasible) also caught a genuine
cross-prompt position-leak bug before it reached the real sweep.

Run over the real 19-step sweep, 50 replicates/checkpoint
(`tools/run/relay_null.py` → `data/analysis/relay_null_series.json`):

| step | raw relays | null mean | excess | excess / null |
|---|---|---|---|---|
| 0 – 2000 | 0 | 0 | 0 | — |
| 4000 | 15,030 | 2,968 | 12,063 | 4.1× |
| 8000 | 232,568 | 27,719 | 204,849 | 8.4× |
| 16000 | 509,646 | 64,176 | 445,473 | 6.9× |
| 32000 | 1,176,478 | 128,647 | 1,047,832 | 8.1× |
| 54000 | 2,560,483 | 346,008 | 2,214,479 | 6.4× |
| 143000 | 2,407,556 | 241,229 | 2,166,327 | 8.9× |

The raw count sits 4–9× the chance level at every formation-window checkpoint
— real excess above what the induction-pair supply and edge counts alone would
produce — and that excess is what §3.6's gate is scored on.

### 3.5 Done: the behavioural arm over the sweep

`tools/run/behavioural.py` → `data/analysis/behavioural_series.json`, run
2026-09-03 (`POPPER_PLAN.md` §6u). Pooled mean post-softmax attention on
induction pairs per (layer, head) per checkpoint, on the same pair set `run_7.py`
types the A side with, tokenisation verified token-for-token against each run's
`tokens.txt`. Cross-prompt convention registered by the author: **mirror the
relay side** — pool the seven non-`repeated_tokens` prompts, carry the
eight-prompt series beside it, never scored. **10,618** pooled pairs (44,809 with
`repeated_tokens`), asserted constant across all 19 steps.

The result: flat at `≈ 1/n_tokens` through step 128 (0 heads elevated), first
rise at 512→1000, sharp climb 2000–8000 — L7H8 peaks **0.0368 at step 4000**,
L6H0 peaks **0.0306 at step 16000**. **Non-monotone in the §2 shape**: leaders
recede (L7H8 → 0.0160, L6H0 → 0.0247 by 143000) while the elevated-head count
runs 0 → 14 (step 8000) → 9. Endpoint precondition clean both ends: step 0 all
384 heads at baseline, step 143000 has 7–9 heads clearly elevated. B leads A by
an interval or two on inspection; the co-location itself needs the gate, which is
blocked on §3.1 and §3.4.

**The floor record is rewired to it** (`schema_version` 2, same session).
`tools/p_i1_attainable_floor.py` arm A now pairs the real relay series against
`series_excl_repeated`, not the synthetic located rise it used before;
`b_side_is_synthetic` is `False` and `--check` verifies the input hash. The
dense-axis refusal is unchanged (`paired_colocation_arm` profiles the A side
first, so the 268 all-zero relay heads decide it regardless of B). What is new:
the 116 forming heads emit **p = 0.420** against the measured B side,
**mean_distance_log_step = 2.02** — on the raw count the two curves do not
co-locate per head, behaviour leading. Not P-I1's test (raw count, not the
above-null excess §3.4's null would produce) and partly a floor effect — the
relay count is structurally zero until step 4000 so its change can't be located
below log-step 3.6 — but it is the number the null now has to move.

### 3.6 The real result: `tools/score_p_i1.py`, p = 0.1414, INSUFFICIENT

`p_value_p_i1` on the real above-null excess series (§3.4) against the real
behavioural series (§3.5), pre-filtered to the 116 forming heads,
`skip_no_rise=True`:

| | |
|---|---|
| p_value | **0.1414** |
| p_reciprocal | 1.0 |
| verdict | **INSUFFICIENT** |
| n_units | 116 (`n_skipped_no_rise` = 0 — every forming head's excess still located a rise) |
| attainable_floor | 0.0005 |
| mean_distance_log_step | 2.018 |

**Barely moved from the raw-series number** (§3.5's 2.02): subtracting the
null rescales the curves' magnitude far more than it moves where each head's
change is located, at least at this replicate count. p dropped from 0.420
(raw) to 0.141 (excess) — real movement, and still nowhere near α = 0.05.

**Both endpoint failure modes are clear**, reported and entering no p-value
per §3.3: 0 of 116 heads are already above-null at step 0, and of the 2 heads
absent at step 143000, 0 had a high behavioural score. Neither disjunct of the
falsifier's second half fires.

**Not adjudicated.** `claims/adjudications/` is untouched —
`tools/score_p_i1.py` deliberately does not call `adjudicate_p_i1(..., adjudicate=
True)`. INSUFFICIENT is not RE-ANCHORS: the design did not fail, and nothing
here falsifies `P-I1`; it means the two curves' rises do not co-locate across
heads more than an arbitrary pairing allows, at the registered sweep and the
50-replicate null.

### 3.7 Answered (2026-09-05): the p-value moves 0.14 → 0.89, and the reason is structural

The K = 100 rerun ran end to end — **3 h 12 m**, matching the estimate — and
wrote `data/analysis/relay_null_series.json` (`n_replicates: 100`). All three
replicate series are now durable in the repo tree:
`data/analysis/relay_null_series_k50.json`, `…_k100.json`,
`relay_null_full_k100.log`.

| | K = 50 | K = 100 |
|---|---|---|
| `p_value` | 0.14143 | **0.89355** |
| `p_reciprocal` | **1.0** | 0.89305 |
| `mean_distance_log_step` | 2.017944 | 2.018009 |
| `n_units` / `n_skipped_no_rise` | 116 / 0 | 116 / 0 |
| `attainable_floor` | 0.0004998 | 0.0004998 |
| verdict | INSUFFICIENT | INSUFFICIENT |

**The p-value moved 6.3×. The observed statistic did not** — `mean_distance_
log_step` changed by 6.5e-5 (0.003%). So this is not the point estimate
shifting; it is the permutation null that the p is read against.

**Mechanism.** 50 more null replicates moved the above-null excess series by
< 0.06 % at every step (0 sign flips, largest single-cell change 91 relays
against millions). That nudged **77 of the 116 heads' change-centroids by
~1e-4 – 1e-3 log-step** — enough to move the observed `-mean|c_a − c_b|`
across the body of the permutation null. The tie structure itself did **not**
change: 79 distinct centroids both times, one class of **36** heads pinned at
log-step 4.9439 (rise entirely in the 54000 → 143000 interval), one class of
3, 77 singletons — exactly §2's "77 singletons, one class of three, and one
class of thirty-six". That 36-head coset dominates the null's spread, so the
permutation distribution of `-mean|c_a − c_b[perm]|` is a near-spike and the
observed value sits at its median. On a near-vertical CDF a 6e-5 shift in x is
a 0.75 shift in F(x). `p_reciprocal` collapsing from exactly 1.000 (pinned to
the atom) to 0.893 is the same tell.

**This realises §3.3's stated precondition as a failure.** "No more than k
heads sharing one change location" — 36 of 116 share one. §3.2's tie *floor*
is 1e-148 here and does not bind, but the tie-driven *atomicity* of the null
still makes the p-value a step function of an input that the replicate count
perturbs at the 4th–5th decimal.

**What is robust, and what is not.** Robust: the verdict (INSUFFICIENT at both
K), the observed mean distance (~2.018 log-step — the two families of rises do
not co-locate per head, nowhere near α), `n_units`, the floor. Not robust:
the p-value itself. **§3.6's p = 0.1414 is not a quotable number, and neither
is 0.8936.** P-I1's pairing arm can return the verdict but cannot quantify how
insufficient the evidence is, on this sweep.

**To get a stable p (not required — the verdict stands):** break the 36-head
coset. Either a denser sweep inside 54000 → 143000 so those heads' rises spread
across more intervals, or a B-anchor that is not centroid-tied the same way —
the per-head attention-dissipation series from
`docs/dissipation_checkpoint_axis_scoping.md` Tier B is the candidate, and it
is defined at every checkpoint rather than structurally zero before step 4000.

---

### 3.8 The two analysis tracks opened 2026-09-05 (context: while K = 100 ran)

Both are in `docs/` and `data/analysis/`; both build only on artifacts already
on disk. Neither adjudicates anything — measurement, with the provenance caveat
from `docs/results_provenance_audit_2026-09-05.md` §3.1 (the Phase 2 OV
projectors carry no `git_sha`; every producing module is unchanged since the
runs, but that is inference not record).

### 3.8.1 Co-location panel — `data/analysis/colocation_panel.{png,csv}`

The Phase 1 / Phase 2 global observables parsed onto the P-I1 19-step axis from
the committed run reports: energy-monotonicity violations (both phases),
`frac_repulsive`, `ov_frac_repulsive`, effective rank (normed + raw), raw λ₂
and the Fiedler deviation, beside the P-I1 behavioural and relay-excess series.
The reading: the transitions fire in a fixed order across one decade —
**128→256** energy break + first behavioural rise; **512** plateau-onset flips
weight→content, `frac_repulsive` hits 1.0, raw λ₂ starts its monotone fall;
**2000–4000** effective rank peaks (44.5) with the behavioural leaders; **4000**
the relay motif first appears (structural zero before). The relay count is the
*lagging* indicator — every other signal turns 3–4 checkpoints earlier and on
more points, which is why §3.7's pairing arm, anchored on the relay side, has
so little to locate against.

### 3.8.2 Dissipation identity, Tier A — `data/analysis/dissipation_series.json`, `dissipation_panel.{png,csv}`

`core/dissipation.py` evaluated per (step, prompt, layer) over the 19 × 7 grid ×
24 layer boundaries, from `activations.npz` (total dX) + the on-disk per-checkpoint
OV Schur projectors. No forward pass. `tools/run/dissipation.py`, ~22 min,
`max_subspace_sum_check = 2e-12` (the split is exact). β ∈ {1.0, 2.0}, sphere
frame. **Caveat, recorded in the artifact:** the subspace split projects the
*total* dX (attn+ffn), not the attention channel — the attention-only and
per-head versions need the Tier B sublayer-capture pass.

Four findings:

1. **The forward-Euler / ODE framing the project rests on is checkpoint- and
   layer-dependent, and mostly does not hold.** The relative linearisation
   residual `|ΔE − Σᵢ⟨Gᵢ,vᵢ⟩| / max(|ΔE|,|Σ⟨G,v⟩|)`:
   - **layer 0 is never in the regime** — residual ≈ 1.0 at every checkpoint
     (step size ~450 in raw coords). Any per-layer dissipation reading must
     drop layer 0.
   - deep layers (4–23) reach residual **~0.17 only at steps 2000–4000** — the
     ODE picture is decent *in the induction-formation window* — and degrade to
     ~0.87 both before (steps 8–256) and after (steps ≥ 16000).
   This is `MATH_SPECTRAL_OT` §5.3(d)'s "a large residual is a finding about the
   project's framing", quantified: it is large outside a narrow window.

2. **Step 512 is a triple co-location in the dissipation view** — same
   checkpoint as the energy break / plateau flip / Fiedler turn. The
   repulsive-subspace share of `|dissipation|` jumps 0.45 → 0.67 and **stays**
   0.63–0.76 for the rest of training; the gradient-flow alignment
   `mean cos(−G, v)` is at its least anti-aligned (≈ −0.01, i.e. motion most
   orthogonal to the energy landscape) with `frac_descending` at its peak 0.55.

3. **The layer motion is weakly anti-aligned with the energy gradient almost
   everywhere** — `mean cos(−G, v)` ∈ [−0.10, −0.01], most negative (−0.10) at
   the step-128–256 energy-break onset, drifting back to −0.08 late. Trained
   layers are, on average, very slightly *ascending* E_β — the activation-side
   reading of the monotone-energy break, and the measured counterpart of Phase
   2d's weights-only D1.

4. **Phase 2 open-item-5's `frac_repulsive` decay (1.00 → 0.56 over steps
   8000–143000) is NOT reproduced** in the total-displacement subspace split —
   the dissipation repulsive-share is flat ~0.64 from step 8000 on. So the
   reorganisation Phase 2 sees is either specific to the *attention* channel
   (Tier B would show it) or to the *violation* mass rather than the
   displacement magnitude. First thing for Tier B to resolve.

### 3.8.3 Dissipation identity, Tier B — `data/analysis/dissipation_sublayer_series.json`, `dissipation_tierB_panel.png`

`tools/run/dissipation_sublayer.py`, run 2026-09-06 (20 min, 133 forward passes
with sublayer capture, `max channel sum_check = 4e-15` — the attn/FFN split is
exact, and the per-head projection `Σ_h dX_attn[h] + bias = dX_attn` holds to
1e-5 at every layer). All 24 layers kept (author's call: fine for L0/L23 to have
unexplained behaviour in analysis, not fine to exclude them from the
measurement). β = 1.0.

1. **The `frac_repulsive` decay is NOT in the attention channel's displacement
   geometry.** The attention channel's repulsive-subspace share of `|dissipation|`
   is near-cancelling at the aggregate (Σ d_attn passes through ~0 several times)
   and does not track Phase 2's violation-based `frac_repulsive`. On the forming
   layers L8–23 it rises to 0.82 at step 2000, dips, returns to 0.85 at
   step 32000, drops to 0.54 at 143000 — a trajectory, but not the smooth
   1.00 → 0.56 decay. The FFN-channel projection nominally tracks it better
   (0.97 → 0.82 over 8000 → 143000) but projecting an FFN output through an OV
   subspace is not physically meaningful. **Net: still not localised to a
   displacement channel — the clean test is a violation-restricted subspace
   split (share of the *positive* first-order term, not of `|dissipation|`),
   which is a v2.**

2. **Step 512 co-location holds in the attention channel specifically** — attn
   `mean cos(−G, v)` crosses positive (+0.03) at step 512, its only positive
   value, against −0.11 to −0.13 through steps 2000–4000.

3. **The per-head co-location test — one stable CO-LOCATES, exploratory.** The
   per-head attention-dissipation series **breaks §3.7's 36-head centroid tie
   coset**: change-centroids are 116 singletons, no ties. Three candidate
   B-anchors were tried post hoc against the behavioural rise, on the same
   `p_value_p_i1` gate, `skip_no_rise=True`:

   | B-anchor (per forming head) | p_value | verdict | mean_dist (log-step) |
   |---|---|---|---|
   | `d_attn_total` (head's attn contribution to first-order ΔE) | 0.333 | INSUFFICIENT | 0.566 |
   | **`|d_attn_repulsive|`** (attn displacement projected on repulsive OV subspace) | **0.0040** | **CO-LOCATES** | 0.576 |
   | `-gfa_cos` (attn anti-alignment with the energy gradient) | 0.587 | INSUFFICIENT | 0.508 |

   The `|d_attn_repulsive|` result is **stable** where §3.7's relay-based p was
   not: deterministic on re-run, seed-stable (10 seeds, p ∈ [0.0015, 0.0045],
   sd 0.0009), head-jackknife-stable (drop any 1 of 116 → p ∈ [0.0005, 0.0065],
   0 flips above α), and not a magnitude tautology (centroid corr with the
   behavioural side r = 0.21; observed mean pairing distance sits at the 0.1st
   percentile of the label-permutation null). Mechanism reading: **as a head
   becomes an induction head, its attention output starts moving the residual
   stream along the repulsive (individuating) OV directions in a way that pushes
   on E_β, and that onset tracks the behavioural onset across all 116 heads.**

   **This is exploratory, not an adjudication.** Three anchors were tried and one
   was significant — multiple comparisons, and no differential falsifier was
   registered before running (POPPER_PLAN §C2's requirement). It is a
   **hypothesis to register and re-test**, e.g. on held-out heads or a second
   model, not a validated P-I1. `claims/adjudications/` stays empty. What it
   does settle: §3.7's "get a stable p by using a non-centroid-tied B-anchor"
   works — the anchor exists and the tie coset is gone.

---

### 3.9 External resource — the pythia-70m dense-onset run (`lora_ind`)

Logged 2026-09-06 from an inspection of a sister project. **Not consumed by any
Mets code yet.** This is a resource note plus where it could plug in.

### What it is

A **full-parameter** training continuation of `pythia-70m` from `step512` to
`step2000`, on an RTX 3080 (6.9 h, torch 2.11.0+cu130, `lora_ind` git
`1ebbe33`). Not LoRA, not that project's M1–M8 protocol — a plain
`GPTNeoXForCausalLM` (`attn_implementation="eager"`, bf16), AdamW (0.9, 0.95),
wd 0.01, grad-clip 1.0, micro-bs 2 × grad-accum 512 = a 1024-sequence batch,
`monology/pile-uncopyrighted` streamed and packed to 2048.

**The load-bearing detail is the LR schedule**: it *continues* Pythia's own —
peak 1e-3, 1430-step linear warmup, cosine to 0.1× over 143k. Step 512 sits
*inside* the warmup at 3.58e-4 and climbs to 1e-3 by ~step 1500, so the onset
window runs at the intended rate, not at peak LR (which is what a
default-configured continuation would do, and would make the transition an
artefact of the harness).

**It is not the Pythia trajectory.** Adam is cold-started at step 512, so it
diverges from published `pythia-70m` from step 1. These checkpoints are "a
clean, correctly-scheduled induction onset in the loss landscape near
`step512`", not "`pythia-70m` steps 528–852". The sister project's own spec
phrases every such result as *reachability, not developmental* — Mets would
inherit that constraint.

### On disk

`/home/iron/Desktop/lora_ind/data/retrain/cb25e3f6c2185c1e/` (67 GB):

| file | content |
|---|---|
| `ckpt/step_NNNNNN.pt` | **249 weight snapshots**, every 4 optimiser steps, 528 → 1860 (one gap, 1184 → 1528). Each a **plain HF `GPTNeoXForCausalLM` state_dict** — fp32, 281 MB, **no optimiser state, no RNG**. Loads with `core.lm_loading.load_causal_lm_from_state_dict`. |
| `probe.jsonl` | 381 rows, step 516 → 2000 every 4 steps: `pms` (prefix-matching score), `icl` (first−second copy NLL), `nll_first/second`, `train_loss`, `lr`. |
| `final/model.safetensors` | the step-2000 model = the sister project's checkpoint **B**. |
| `provenance.json` | `onset_step: 620`. **Its `retained_checkpoints` list (1528–1860) is wrong** — a double-resume reset the buffer bookkeeping. |

**Clean bracket: steps 528 → 852, 82 checkpoints, all present, no gaps**, onset
at 620 (PMS 0.019 @ 592 → 0.051 @ 620 → 0.36 @ 676 → 0.51 @ 700; ICL 0.03 →
1.86; loss declines smoothly, no LR kink). Session 1 ran 516 → 868 correctly;
only steps 856 / 860 are contaminated. Everything 864 → 1860 is from two resumes
that replayed different batches (visible as different `train_loss` at identical
steps) — usable as a rough long tail, not for anything tight.

**State as of 2026-09-06:** the local `lora_ind` repo is **very out of date**,
and a **re-probe is running now** — it re-scores PMS/ICL at `n_eval=512`
(the run used 128, the sister project's §5 wants 512) and raises rather than
logging a 0.0 when attentions come back empty. It emits `onset_bracket.json`,
which is **authoritative** over `provenance.json` for the bracket and the
contaminated steps. Wait for it before consuming the bracket.

### The sister project, in one paragraph

`lora_ind` = "Induction Bandwidth": the minimum-rank weight update that installs
a working induction circuit into a pre-induction `pythia-70m` checkpoint, and
whether the update's antisymmetric fraction φ differs between the
prefix-matching (QK) and copying (OV) halves. **Shared spine with Phase 2 / 2b**:
`M_QK = W_Q W_Kᵀ`, `M_OV = W_Oᵀ W_Vᵀ`, the `S + Λ` split, `φ(M) = ‖Λ‖²/‖M‖²` —
the same operator decomposition `p2b_imaginary` and `core/dual_reading.py` use.
Its gates G0–G2 passed (checkpoint A = step 512, B = step 2000, induction head
**L3H6**, previous-token head **L2H1**). **G3 — the positive control — failed**:
a generous-rank QK-only update did not reach criterion, and the run could not
distinguish "composition rule too restrictive" from "optimisation broken". M1–M8
(the actual rank sweeps) are blocked on it. **So there is no rank-r induction
subspace yet.**

### How / where it could plug into Mets

1. **Dense-onset sub-study — the thing the Pythia mirror cannot give.**
   `p2_eigenspectra/status-2.md` item 4 and §3.7 both note there is no released
   Pythia checkpoint between 512 and 1000. This run has one every 4 steps
   through the onset. It would let the §3.8 findings be checked at real
   resolution instead of 3 sparse points — in particular **§3.8.2 finding 1**
   (the forward-Euler residual "sharpens across 512 → 1000, sharpest at
   2000–4000"): with 4-step spacing the claim becomes testable rather than
   interpolated. `pythia-70m` is 6 layers / d=512 / 8 heads, so a full Phase 1 +
   Phase 2 + dissipation pass over all 82 clean checkpoints × the battery is
   cheap on CPU.

2. **A non-tie-coset behavioural / energy co-location axis.** §3.7's whole
   problem was 36 of 116 heads sharing one change-centroid because the axis is
   too sparse in the formation window. A 4-step axis through the onset spreads
   the centroids by construction — the co-location arm gets a well-conditioned
   null without needing §3.8.3's post-hoc anchor hunt.

3. **The real merge, once `lora_ind` unblocks G3.** When that project produces
   `r*_QK`, `r*_OV` and the φ signatures, Mets can project the rank-r induction
   subspace **in and out** of the residual stream at each dense checkpoint and
   re-measure the particle dynamics — the dissipation identity, the energy
   violations, the relay motif — with vs without the induction contribution.
   That is the concrete form of "isolate what induction does to the particle
   picture", and it is **blocked on `lora_ind` G3**, not on anything here.

### Cost of adoption, stated honestly

- **New registered ground.** Every Mets registered decision is 410m-specific
  (`REGISTERED_P_I1_SWEEP`, `P_I1_RELAY_OWNER`, the CLAIM-B grid, the battery
  tokenisation). A 70m dense run needs its **own** registered grid, forming-head
  axis and battery — no 410m number transfers, and this must be a labelled
  sub-study, not an extension of the 410m sweep.
- **Reachability, not developmental** (above) — any co-location result on this
  run generalises to Pythia only as far as the loss landscape near `step512` is
  representative.
- **No activations saved** — weights + scalar probe trace only. Adoption means
  pointing the Phase 1/2/7 runners at local `.pt` paths and re-running forward
  passes. `core.lm_loading.load_causal_lm_from_state_dict` already exists; a thin
  "checkpoint dir → (model, tokenizer)" adapter and a `REGISTERED_P70M_*` grid
  are the whole lift.
- **torch 2.11 (run) vs 2.13 (this venv)** — irrelevant for loading, the
  snapshots are plain state_dicts.

### 3.9-A The sister project re-read (2026-09-10) — **§3.9 above is stale on four points**

User-directed tangent, taken after §3.12-V because a project that minimises
induction-head **rank** bears directly on a finding that the members are
rank-1-to-24 and one of them **inverts**. Source of truth is now the GitHub
remote `git@github.com:ZachData/Lora_inductionhead.git`; the desktop clone at
`/var/home/iron/Desktop/lora_ind` was **57 commits behind** `origin/main` when
read. Read their `HANDOFF.md`, then `PROJECT.md` §10/§11, then `REVIEW.md`.

**Corrections to §3.9, which was written 2026-09-06:**

1. **G3 is no longer ambiguous.** §3.9 says the run "could not distinguish
   'composition rule too restrictive' from 'optimisation broken'". **Optimisation
   is now ruled out** — eight diagnostics agree, and the reachability graft shows
   `B`'s **own** `W_Q` for the target head does not move `R`, which
   *upper-bounds any trained update*. Four negative diagnostics (broken
   objective, insufficient rank, frozen `W_K`, wrong head) point at a genuine
   capacity/structural limit.
2. **Matching and recovery dissociate.** Grafting blocks 0–2 restores prefix
   matching almost fully (PMS **0.895** vs full graft's 0.964) while `R` stays at
   **0.10**.
3. **The gradient is not gated.** σ_OV is nonzero at all 2048 query positions at
   `A`, so Cor 17.2's condition fails and Prop 17.5's flat-`R(r)` mechanism is
   not what the diagnostics measured.
4. **The re-probe §3.9 says to wait for is done** — 82/82 checkpoints, all 48
   heads at `n_eval=512`, `data/reprobe_merged.{json,png}`.

**What Mets should send them, highest value first.**

- **Their copying score is broken and Mets has the fix.** `src/indbw/probes.py`
  defines it as a **full-vocab argmax hit rate** (`argmax_s(W_U M_OV e_t) == t`).
  It reads **≤ 2.8e-4 for every head at both checkpoints, including `B`'s (3,6),
  which demonstrably does induction** — a known-positive at the floor. They have
  parked it in `REVIEW.md` as a `METRIC_VERSION` human call. **`tools/run/
  copying_score_sweep.py` is the answer**: Elhage's `Σλ / Σ|λ|` over the
  token-to-token circuit, which discriminates on 410m (`L11H14` 0.723). It fixes
  two things at once — it is **continuous** rather than winner-take-all over
  50304 tokens, and it is **transpose-invariant**, which their *directional*
  score is not (Mets' own §3.14.3 defect 3 is exactly this trap, and that runner
  documents it as verified rather than assumed). Cheap via the **64×64**
  reduction: nonzero spec(`C`) = spec(`W_V G W_O`), `G = W_Eᵀ W_U` once per
  checkpoint.
- **§3.12-V4's anti-ordering is a prerequisite check for M1–M8**, not an
  explanation of G3. If their cascade heads are `L11H14`-like, a minimum-rank
  result taken in the SVD/gain basis measures nearly the **opposite** of what it
  intends — bottom-`r` beat top-`r` at every rank, and rank-1 was worse than
  deleting the head. Their whole programme is minimum-rank, so this should run on
  (3,6) and (3,1) **before** M1–M8 unblock. It does **not** explain G3:
  reachability is a more fundamental blocker and does not reduce to a basis
  choice.
- **Their "dissociation puzzle" is Mets' recurring finding.** §3.12-U had
  `L5H2`'s induction score fall **twenty-fold** across the interval its causal
  effect went +0.01 → +4.97; §3.12-O/P found `L7H8` is not a copier at any point;
  §3.12-V1 has δ-cosine explaining 7–9 % of causal interaction across 45 pairs.
  Behavioural and structural proxies systematically fail to predict causal
  effect — their PMS-0.895 / R-0.10 result is an instance, not an anomaly.
- **Two hazards worth forwarding**: the readout **ceiling** (§3.14.4-D — ΔNLL
  near `ln V` is a floor, and it biases interactions toward false
  sub-additivity), and §3.12-V5's **changing-membership artifact**, which is
  live for them because their six heads form at *different* steps inside the
  bracket.

**What Mets gains — and it is the thing §3.14.4-A said it did not have.**

Their re-probe gives an **ordered six-head cascade** through a pythia-70m
induction onset, stride 4, 82 checkpoints, all 48 heads:

| head | first PMS ≥ 0.10 | peak PMS |
|---|---|---|
| **3.6** | 640 | 0.857 |
| **3.1** | 652 | 0.783 |
| 4.6 | 696 | 0.309 |
| 4.7 | 724 | 0.263 |
| 3.0 | 760 | 0.190 |
| 3.5 | 832 | 0.105 |

Prev-token head **2.1 goes 0.389 → 0.947 inside the window**; ICL −0.01 → 6.63;
`R` 0.001 → 0.576. §3.14.4-A asks whether the 410m ordering is **cascade** (order
reproduces) or **recruitment** (window reproduces, order scrambles) and states
the discriminator needs a second, differently-seeded model, with 70m the site and
its weights not on disk. **This is that draw**, and Mets found six ordered
members in 410m against six ordered heads here.

**The caveat is load-bearing and §3.9 already states it**: the retrain
cold-starts Adam at step 512, so it is **reachability, not development** — a
legitimately independent draw of the *ordering*, and not a claim about
`pythia-70m`'s trajectory. Adoption still needs its **own registered grid**
(§3.9's "Cost of adoption"): no 410m number transfers.

**Second gain — resolution.** Every §3.12-V claim in the formation window rests
on 1–2 grid points (alignment peaking at 5000, `L11H14` defecting in (2000,
4000], birth-alignment at 1000). A stride-4 axis converts several of them from
interpolated to measured.

---

### 3.10 Per-head OV, and the circularity that stopped a registration (2026-09-07)

Full narrative in `POPPER_PLAN.md` §6w. What a fresh session needs:

**Nothing was registered.** `claims/registry.json` is unchanged,
`claims/adjudications/` is still empty. Every number below is post-hoc on
artifacts that already existed and may not be adjudicated on them.

**The instrument.** `p2b_imaginary/head_circuits.py` already had the per-head
OV machinery (`head_core`, `head_spectrum`, `sym_antisym_factors`,
`apply_factored`). It gained an energy-weighted sign split —
`repulsive_energy_fraction_core` / `attractive_energy_fraction_core` /
`repulsive_dim_fraction_core`, +7 tests — which is `MATH_SPECTRAL_OT` §3's own
proposed discriminator, energy-weighted rather than bulk-edge-restricted
because a bulk edge is a placed constant and a weighting is not.
`tools/run/ov_per_head.py` → `data/analysis/ov_per_head_series.json`
(45 min, weights only, no forward pass, no model load — every checkpoint's
per-head dense OV is already on disk as `ov_head<h>_layer_<l>`).

**It is calibrated**, which is the strongest null check in the sequence:
chance is 0.500 ± 0.033 (300 matched-shape random heads at the real 1024/64
geometry) and **step 0 measures 0.4982**. Not dominated by one eigenvalue
(top-eigenvalue energy share median 0.057, exceeds 0.5 nowhere in 384 heads).
The 21% of heads pinned at exactly 0.0/1.0 are pinned *structurally* — median
`min|Re λ|/|λ|` = 0.873, none under 1e-2 against an fp32 floor of ~1e-8.

**The trajectory is a real finding regardless of the circularity.** Every head
at chance through step 64; both populations swing to near-total repulsive
dominance by step 1000 (forming 0.989, non-forming 0.948); from step 2000 the
relay-carrying heads reverse to **0.156** while the rest hold at **0.701**.
The divergence opens in the formation window and grows monotonically.

**Two population facts that bear on `P-I1` and `P-I3` directly.**
The 116-head relay axis and the behavioural induction population are nearly
disjoint — **1 of the top 9** behavioural heads is on it, the relay axis sits
in layers 8–23 (mass 21–23) against the behavioural leaders in layers 1–10
(mass 6, 7, 9), and the relay axis's mean peak behavioural score (0.00536) is
*below* the off-axis mean (0.00618). `spearman(max relay excess, peak
behavioural)` = −0.230 (p = 0.013), **but the partial controlling for layer is
+0.004** and the mean within-layer ρ is +0.090 — so it is **no association
within layer**, not an inversion, and the raw negative is two opposite depth
trends multiplying. This is a more direct account of §3.6's INSUFFICIENT than
§3.7's tie coset: `P-I1` pairs two per-head series that are unrelated within
layer, over a population where the behavioural side sits near baseline.

**The summed-OV projector is a fiction, quantified.** The on-disk
`schur_repulse_layer_*` come from `ov_total = sum_h ov_per_head`, and the
summed value lands within 0.05 of only **21.2%** of the heads in its own layer.
So the Tier B per-head anchor (§3.8.3) projects each head's write onto a
subspace of an operator the model never forms.

### 3.11 The bottom-up induction programme — decisions taken before running

Replaces the co-location frame. Isolate one induction head, find the minimal
structure that carries induction, perturb it, then build outward. Machinery is
live: `p2_eigenspectra/head_ablation.py` (per-head OV ablation for GPT-NeoX),
`core/intervention.py` (`run_model_with_hook`, `next_token_kl`),
`head_circuits.py` (factored S/A surgery), `tools/run/behavioural.py`.

**Stages.** 0: target `L7H8` at step 4000 (peak behavioural 0.0368) and locate
its stage-1 prev-token partner. 1: rank sweep on QK and OV separately → `r*`.
2: characterise the `r*` subspace (Schur sign, φ, token-subspace alignment).
3: small variations at matched norm against a matched-norm random control,
joint behavioural + logit + geometry readout (`P-I5`'s registered shape).
4: repeat on the other elevated heads — generalisation *after* mechanism.

**Three decisions, taken now because taking them after seeing a curve would
void the guarantee** (§6l's timing argument):

1. **Both bases, and they answer different questions.** SVD finds `r*` — it
   measures gain, is Eckart–Young optimal for "minimum rank that carries the
   action", and orders unambiguously. Schur characterises what is *in* `r*` —
   it is the only one carrying a **sign**, so the attractive/repulsive
   differential falsifier cannot be posed in the SVD frame at all. Cost is not
   a consideration: 0.40 ms (SVD) and 0.99 ms (real Schur) per 64×64 core,
   ~25 s for the whole 19-step × 384-head grid. **Their disagreement is itself
   a registered outcome**: these cores are strongly non-normal (Henrici median
   **0.450**; rank for 90% of action 36.5 by SVD against 41.5 by eigenvalue
   ordering, median per-head gap 5, max 13), so `r*_SVD ≈ r*_Schur` says the
   induction-relevant part is near-normal and the eigenvalue picture is
   trustworthy, while `r*_SVD ≪ r*_Schur` says induction lives in high-gain
   non-invariant directions and the project's whole attractive/repulsive frame
   is measuring something other than what the head does — `MATH_SPECTRAL_OT`
   §5.3(d)'s "a large residual is a finding about the framing", reached from a
   second direction.
2. **`r*` is derived, not thresholded.** No "induction collapsed" constant.
   Report the score-vs-`r` curve and define `r*` as where it crosses the
   matched-norm control band — the same move that made
   `N_CONTROLS_PER_INDUCTION_HEAD` a frontier rather than a placed number.
3. **Readouts are paired to the operator, and getting this wrong measures
   nothing.** The behavioural induction score is *mean post-softmax attention
   on induction pairs* — a pure QK quantity — so **ablating OV cannot move it
   within the layer**. QK sweep → behavioural score. OV sweep → logit/copying
   effect (`next_token_kl`). Either → particle-geometry delta. A flat OV curve
   read against the attention score would be misread as "OV does not matter".

**Still to register before stage 2 reads anything**: the differential
prediction itself — particle account says the causally-identified induction
subspace is repulsive/individuating, standard account says it is a copier
(attractive, token-aligned). Neither is silent, so INVERTS can fire. Stating
it after seeing the `r*` subspace repeats §6w's mistake one level down.

### What has run (2026-09-07) — Stages 0–1 done, Stage 2 invalid, nothing registered

All exploratory. `claims/registry.json` and `claims/adjudications/` unchanged.

**Stage 0 — the circuit is `L5H2 → L7H8`.** From `attentions.npz` at step 4000,
no forward pass: `L5H2` is an overwhelming previous-token head (mean attention
at offset −1 = **0.895** against a 384-head median of 0.018, ~49×). `L7H8` is
the behavioural leader (§3.5). Textbook two-stage shape: prev-token head in L5,
matcher in L7.

**Stage 1 — OV half only, `tools/run/induction_rank_sweep.py` →
`data/analysis/induction_rank_sweep.json`.** QK half deferred: Pythia rotates
only `rotary_ndims = 16` of 64 head dims, so an `M_QK = W_Q W_Kᵀ` rank
truncation is not a truncation of what the model computes — needs a
RoPE-aware treatment. Readout is the **copying** side (decision 3): second-copy
NLL on repeated uniform-random sequences (`N_REP=96`, 8 seqs) plus KL from the
unablated model, through the full forward. Weights saved/restored around every
measurement; end-of-run restore check exact (`abs_diff` 0.0).

The OV effect on this head is **modest**: full OV ablation (`r=0`) moves the
second-copy NLL only **0.779 → 1.023** (KL 0.043). Fraction of that effect
recovered by a rank-`r` truncation, `(nll₀ − nllᵣ)/(nll₀ − nll_base)`:

| r | SVD | Schur | random control |
|---|---|---|---|
| 1 | **82 %** | 12 % | ~0 % |
| 2 | 85 % | 15 % | ~0 % |
| 6 | 93 % | 75 % | 16 % |
| 16 | ~99 % | 81 % | 32 % |
| 24 | ~99 % | 88 % | 52 % |
| 48 | 100 % | 97 % | 90 % |

**`r*_SVD ≪ r*_Schur`** — SVD rank 1 already carries 82 % and is flat past
r≈6; Schur needs r≈16 to match rank-1 SVD, and beats the random control only
modestly below r≈24. This is the **pre-registered branch of decision 1**:
induction's copying action lives in a **high-gain, non-invariant** direction,
and the attractive/repulsive (eigenvalue-sign) frame is not the natural
description of this head's OV — `MATH_SPECTRAL_OT` §5.3(d) reached from a
second direction.

**Stage 2 — redone 2026-09-08, `tools/run/induction_subspace_characterize.py`
→ `data/analysis/induction_subspace_characterize.json`.** (An earlier inline
Stage 2 was discarded: its copying readout was a direct logit attribution with
no `final_layer_norm` and no control heads.) This version reads copying
**causally through the full forward** — same readout as Stage 1, LN present by
construction — and calibrates every number against all 16 heads of layer 7 and
24 matched-Frobenius-norm random OV operators. Weights save/restore, end
restore check 0.0.

*L7H8's OV is the causally load-bearing half, by a wide margin.* Full OV
ablation moves the second-copy NLL **0.779 → 1.023** (ΔNLL **+0.244**, KL
0.043) — **the largest of all 16 layer-7 heads**, ~10× the layer mean (0.021 ±
0.059). So the mid-Stage-2 doubt ("is OV even the right half") is settled for
this readout: it is.

*`r*_SVD ≪ r*_Schur` holds under the causal readout too.* Rank-1 SVD carries
**82.0 %** of that ΔNLL (rank 1 of 16; layer mean 15 %), rank-1 Schur only
**11.9 %**. The top singular value holds 17.6 % of the OV Frobenius energy
(rank 2 of 16) — one unusually dominant gain direction, which is where the
action is.

*The subspace is entirely repulsive — and that turns out not to be the
description that matters.* Every eigenvalue of the 64×64 core has Re < 0:
`attractive_energy_fraction_core` is **exactly 0.0**, top |λ| has Re −0.165,
the rank-1 SVD mode's own eigenvalue is Re −0.177, the top-16 Schur subspace
is 100 % repulsive. But φ = 0.398 (rank 10/16, *below* the layer mean) and
Henrici = 0.353 (rank 8/16, *less* non-normal than a random operator's 0.71),
so L7H8 is not a spectral outlier in its layer. The `r*_SVD ≪ r*_Schur` result
says the copying action lives in a **high-gain, non-invariant** direction, so
sorting the operator by eigenvalue sign is not sorting it by what it does —
the repulsive sign is *true but not the mechanism*.

*And it is not a token-identity copier.* The direct copy score of the rank-1
mode (LN mean-scale folded, `s_in` 1.36, `s_final` 0.61, descriptive only):
the diagonal is the row-max for **1 token in 4000 — exactly chance** —
diag z-mean 0.03, diag-positive 0.514. No `W_E → W_U` diagonal structure.

**So Stage 2 cuts against both accounts.** The standard account (induction OV =
attractive, token-aligned copier) fails on both counts. The particle account
(repulsive/individuating) has the sign right but the wrong frame — the effect
is carried by an SVD gain direction, not an eigen-mode. What L7H8's OV
actually is, on this evidence: a **functional** copier (ablating its one
high-gain direction measurably degrades repeated-token prediction) that is
neither a **representational** copier (no token diagonal) nor an **eigen-mode**
(SVD, not Schur, is where the rank collapses).

**Caveats.** One head, one checkpoint. The causal ΔNLL / KL are solid; the
"not token-aligned" half leans on the approximate LN-folded copy score. The
per-head SVD/Schur fractions are only interpretable for L7H8 — it is the only
layer-7 head with a non-noise OV effect, so the others' ratios divide by
~0.01. The **QK half is now done** — see block C below.

`POPPER_PLAN.md` §6x is still unwritten — §6w refers forward to it; this §3.11
is currently the only home for the design.

### Generalisation batch + QK half — results (2026-09-08, exploratory, nothing registered)

`data/analysis/induction_developmental_analysis.py` →
`induction_developmental_series.json`. Two questions: does the Stage 2 picture
hold across development, and across the other behavioural induction heads?

**A. L7H8 OV rank sweep across all 19 axis steps.** Through step 2000 the OV
copying effect is ~zero (`ΔOV_nll` ≤ |0.014|; L7H8 cannot do induction yet) —
the rank fractions there are noise. From step 4000 it switches on and grows
monotonically: `ΔOV_nll` **+0.24 → +0.73 → +1.02 → +1.47** (steps
4000/8000/16000/54000), KL 0.04 → 0.57, slight pullback to +1.18 at 143000.
The `r*` story:

| regime | steps | `svd@1` | `svd@2` | `schur@1` | `schur@8` |
|---|---|---|---|---|---|
| formation | 4000–16000 | 0.82–0.96 | 0.85–0.99 | 0.12–0.49 | 0.77–0.93 |
| consolidation | 32000–143000 | **0.68 → 0.20** | 0.85–0.97 | 0.25–0.34 | 0.83–0.89 |

So the copying action is **one SVD direction during formation**, then spreads
to a **second** during consolidation (`r*_SVD` grows 1 → 2). Schur rank-1 is
weak throughout; Schur needs ~8 modes at every step. **`r*_SVD ≪ r*_Schur`
holds across the entire trained regime** — a stable developmental property,
not a snapshot. Caveat: late full-ablation KL is 0.57, so the rank-1-vs-2 gap
there is partly nonlinearity.

**B. Stage 2 on the top 8 behavioural heads** (L7H8, L7H0, L7H3, L7H12, L6H0,
L2H10, L9H9, L9H8, L1H15), each at its peak step, each calibrated against its
own layer.

*Universal — the representational claim generalises.* **Every** head's OV core
is 100 % repulsive (`attractive_energy_fraction_core` = 0.000, top-λ
repulsive, 9 of 9), φ ∈ [0.29, 0.46], Henrici ∈ [0.23, 0.53] (no spectral
outliers), and **none is a token-identity copier** (copy z ≈ 0, diag-is-row-max
at chance). The standard "attractive, token-aligned copier" account fails for
every behavioural induction head, not just L7H8.

*Concentrated — the functional claim does not.* The causal OV→copying effect is
L7H8's at scale (`ΔOV_nll` +0.244) against L9H9 +0.054, L1H15 +0.040, L6H0
+0.021, and the rest ≤ 0.01 or negative (L2H10, L9H8 negative — their OV is
not a copier). The "top behavioural heads" are ranked by attention pattern (a
QK quantity) and are mostly not OV copiers — decision 3, confirmed. Where the
OV effect is real (L7H8, L9H9, L1H15, L6H0) rank-1 SVD carries it (0.68–0.97)
and Schur rank-1 is weaker, so `r*_SVD ≲ r*_Schur` generalises there too;
L1H15 is the near-exception (Schur rank-1 0.76 vs SVD 0.83 — more normal).

**C. The QK half — `tools/run/induction_qk_sweep.py`, L7H8 across 8 steps.**
Rank-truncates the **static** (rows 16–63, non-rotary) QK operator — the
content-match half; RoPE carries the positional part and its 16 dims are left
intact — and reads the induction attention directly (mean post-softmax weight
on the repo's pairs: query `N_REP+t`, key `t`). Full static-QK ablation drops
L7H8's induction attention from **0.92 to 0.02**, so the readout bites.

The matcher is the **spectral opposite of the copier.** Once formed
(step 4000+, induction attn 0.92 → 0.95, stable):

| | `r*` (rank reaching ½ the effect) | low-rank basis vs random |
|---|---|---|
| **OV copier** | `r*_SVD` 1–2 · `r*_Schur` ~8–16 | SVD ≫ random ≫ Schur |
| **QK matcher** | `r*_Schur` **12** · `r*_SVD` **32** | Schur > random > **SVD (below random)** |

So the copy lives in a **rank-1 high-gain non-normal** direction; the match
lives in a **~12-dim high-eigenvalue near-normal invariant subspace**, and its
top-*gain* directions are worse than random. `MATH_SPECTRAL_OT` §5.3(d)'s
"the eigenvalue frame is or isn't the right description" resolves *per half*:
right for the matcher, wrong for the copier. The Schur<SVD pattern for QK
appears exactly when the match forms (step 2000→4000) and holds through 143000.
(Schur reordering is non-monotone above r≈16 on near-degenerate |λ|; the
r ≤ 16 Schur values are clean.)

*Across the other genuine induction heads (L1H15, L9H9, L6H0 — the heads that
both match on the repo-convention pairs and have an OV copy effect; the other
"behavioural leaders" do neither).* **L9H9** shows the same Schur ≫ random > SVD
matcher pattern as L7H8. **L6H0** (weakest matcher, 0.76) leans the same at low
rank but is noisy. **L1H15** is **generic** — SVD ≈ Schur ≈ random, ~40 of 48
dims needed, no spectral structure — and it was also the near-normal outlier on
OV (Schur rank-1 ≈ SVD rank-1). So L1H15's whole circuit is spectrally
unstructured; the QK dissociation holds for 3 of 4 but is not as clean as the
OV representational claim's 9/9.

**Consequence for the registration.** Two things are now population-level
baselines, not distinguishing features: "OV core 100 % repulsive" (every
behavioural induction head) and the **OV/QK spectral dissociation** (a generic
property of a copy+match circuit, plausibly). The Stage 3 differential
prediction has to be about the **rank-1 high-gain OV direction of L7H8**
specifically — under matched-norm perturbation does it act as an individuating
channel, or does copying survive (a copier the token-alignment test missed) —
not about repulsiveness or rank per se. Register that, then Stage 3.

*Superseded in part by §3.12 (2026-09-09): the "repulsiveness is a population
baseline" premise was measured against the wrong reference class, and the
matched-norm control is degenerate. Left unedited — the reasoning is the
construction log.*

### 3.12 The reference-class error, and what it opens (2026-09-09)

Exploratory. `claims/registry.json` and `claims/adjudications/` unchanged.
Nothing here is registered and nothing here is a p-value.

**A. The OV repulsive collapse is a model-wide developmental phase, not an
induction-head property.** Read off `data/analysis/ov_per_head_series.json`
(commit `08175a8`, weights-only, all 384 heads × 19 steps — the file was
computed for §3.10's per-head projector item and its population axis was never
looked at). Median `attractive_energy_fraction_core` over all 384 heads, and
the share of heads whose core is *exactly* 100 % repulsive:

| step | 0–64 | 128 | 256 | 512 | **1000** | **2000** | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median attractive | ~0.504 | 0.481 | 0.385 | 0.094 | **0.000** | **0.000** | 0.001 | 0.026 | 0.066 | 0.134 | 0.225 | 0.392 |
| frac heads exactly 0 | 0.000 | 0.000 | 0.000 | 0.000 | **0.617** | **0.654** | 0.435 | 0.297 | 0.219 | 0.188 | 0.164 | 0.109 |

The whole model slides off the random-matrix baseline into a **fully repulsive
phase whose floor is steps 1000–2000**, then partially recovers. 1030 of 7296
head-steps are exactly 0.0 overall (14.1 %).

*Checked, not assumed:* `attractive + repulsive = 1.000` exactly at every step,
so this is not energy vanishing into an unclassified bucket — the `< 0` / `> 0`
split in `head_spectrum` leaves nothing at zero. And step 0's ~0.504 is the
**random-real-matrix baseline** (eigenvalues symmetric about the imaginary
axis), so the measurement carries its own control and the model demonstrably
moves off it.

**Two things this collides with.** The floor sits *exactly inside* `CLAIM-B`'s
registered anchor window (512–2000) — either a striking convergence or a shared
cause, and which one matters before either is scored. And Phase 2's activation-
level `frac_repulsive` decay (§2 item 2) runs the *other* way over the trained
regime while this weights-only quantity recovers; they are different
measurements and the shapes have never been put side by side.

**B. §3.11's exclusion of repulsiveness rests on a reference-class error.**
"OV core 100 % repulsive, 9 of 9 induction heads" was read as a population
baseline — but the population it was compared against was *the induction heads
themselves*. Against the 384-head population **at matched step**, exactly-0.0
runs 0.435 at step 4000 down to **0.109 at 143000**. `L6H0` peaks at 143000,
where only 11 % of heads are fully repulsive. The 9/9 is not a baseline.

This does **not** resurrect the observational claim as a registrable
differential — 9/9 with no control arm is still not a test. What it removes is
the *reason given* for excluding repulsiveness from Stage 3, which reopens the
**causal** test of it (block D).

**Before anything is built on A:** check the count version
(`repulsive_dim_fraction_core`) against the energy version, in case one dominant
eigenvalue carries it; and name weight decay as the leading alternative
mechanism, though the recovery after step 2000 argues against a pure-decay
story.

**C. The Stage 3 entry was drafted and is not registrable as drafted.**
A draft entry (H-BRIDGE, geometry-vs-copying at the rank-1 OV direction) was in
the session scratchpad, not in the repo. Three defects, all decidable before a
forward pass.

*Three things the gate taught us while writing this section, and they decide
where this discussion lives.* (1) The draft reused the **induction id that
`POPPER_PLAN.md` §6w already spent** on the co-location registration that was
*not* made; the next free one is one higher. (2) `tools/check_registry.py`'s
coverage rule fired the moment that id was named here — working exactly as
intended, since an id in a scanned file with no registry entry is the state the
registry exists to forbid. (3) It did **not** fire on §6w's use, because
`POPPER_PLAN.md` sits in `SCAN_EXCLUDE` beside the three generated files — so
the design narrative can burn an id invisibly to coverage, and no *uniqueness*
check spans the excluded files.

**Consequence, and it is structural rather than a nuisance:** `PROJECT.md`
cannot name an unregistered prediction id at all, so **the id-level design
discussion belongs in `POPPER_PLAN.md` §6x** — which §6w already points forward
to and which is still unwritten. This section is the summary; §6x is where the
candidate entry, its id, and its wording go before registration.

The three defects:

1. **The copy-matched control is probably unbuildable.** Random rank-1
   directions recover ~0 % at matched norm (§3.11 Stage 1), so reaching the
   observed ΔNLL needs hard rescaling, which drives KL up — matching a
   *copying* effect against a *destruction* effect. §3.11's own nonlinearity
   caveat (KL 0.57 late) bites here. This replaced one degenerate null with
   another.
2. **The geometry contrast has a mechanical component with no individuation
   content.** Perturbing `W_OV` along `v` moves every position the head writes
   to by a near-rank-1 update; the matched positions share attention structure
   *by construction*, so they move together and their pairwise distances change
   *less* than unmatched ones — contraction contrast in the wrong direction, for
   algebraic reasons. `P-ST1`'s "steering is a pure mean effect" one level over:
   settle it on paper, not by simulation.
3. **The direction is identified weights-only.** The top singular direction of
   `W_OV` ignores the residual-stream distribution; the 82 % is a causal
   measurement but the identification is not, and Stage 3 perturbs the
   identified object.

Also missing, and every other registry entry has one: a **precondition on the
pilot, computed before it runs**. Expected verdict distribution as drafted is
INSUFFICIENT-dominated, because the direction was *selected* for carrying the
copy effect and is then asked to beat a bar its own selection sets.

**D. The proposal that replaces it: flip the symmetric part.**
For `M = S + A` with `S = (M+Mᵀ)/2`, `A = (M−Mᵀ)/2`: `Re(λ)` is governed by `S`,
and `tr(SᵀA) = 0`, so `‖−S+A‖_F = ‖S+A‖_F`. **`S → −S` flips every eigenvalue's
real part at exactly matched Frobenius norm, by construction** — no rescaling,
no matching search, no tolerance. `p2b_imaginary/head_circuits.py` already does
factored S/A surgery.

- Standard account: copying depends on gain (σ and singular vectors), not on
  eigenvalue sign → flip `S`, **copying survives**.
- Particle account: individuation depends on repulsive character → flip `S`,
  **matched particles collapse and copying degrades**.

Both non-silent, so the falsification branch can fire; the matching problem and
the mechanical-geometry confound both dissolve. And it is the *causal* test of
repulsiveness, which is not a baseline even where the observational version is:
if every induction head is 100 % repulsive and flipping it changes nothing, the
repulsive frame is decoratively true and causally empty — a real registrable
negative. If flipping it destroys induction, the frame is load-bearing. **Either
outcome is informative**, which is the property the drafted entry lacks.

**E. Four absences, all cheap, none producing a p-value.**

1. **The composition has never been measured.** Grep confirms no K/Q/V-
   composition score anywhere in the repo — the only "composition" hits are the
   retired `relay` motif. Induction is `L5H2 → L7H8` and only the *endpoints*
   are characterised. Weights-only, milliseconds, and it yields a developmental
   series: when the **composition** forms against when each half forms. No §6w
   circularity, since composition is defined by weights, not by the projector
   whose sign a prediction would read.
2. **Rotated or spread?** `svd@1` falls 0.68 → 0.20 across consolidation while
   `svd@2` stays high; §3.11 reads this as spreading to a second direction.
   Equally consistent: the direction **rotated** and rank 2 tracks a moving
   target. Distinguish by top-singular-vector overlap between adjacent steps.
3. **The anti-copiers.** `L2H10` and `L9H8` have *negative* `ΔOV_nll` —
   ablating their OV *improves* second-copy prediction — while carrying the
   universal representational description. Stage 2's "representation is not the
   mechanism" in its sharpest available form, already computed.
4. **The 16 rotary dims were never examined.** The QK sweep truncates rows
   16–63 and leaves rotary intact; but `L5H2` is a *pure positional* head and
   "previous token" is positional. Block C describes the static half only.

**Order of work decided here:** the four diagnostics run *before* the entry is
written, because two of them (1 and 3 above, plus the input-whitening check in
C3) can change what a valid control even is. Registering first and diagnosing
after is the §6l timing argument pointed the wrong way — nothing here is a
p-value, so none of it spends the registration.

**F. Diagnostics run (2026-09-09) — `data/analysis/induction_diagnostics_7b.py`
→ `induction_diagnostics_7b.json`.** Weights-only, no model load, no forward
pass, ~1 min 43 s. All four returned, and three of them change the design.

*F0 — the U-shape survives the count check, and gains an ordering.* The
`repulsive_dim_fraction_core` count tells the same story as the energy version
— identical `frac heads exactly 1` at every step (0.617 / 0.654 / 0.435 / …),
so **the collapse is not one dominant eigenvalue**. What the disagreement shows
is *how* it happens: in the ramp, repulsive **energy leads repulsive dimension**
— 0.615 vs 0.562 at step 256, **0.906 vs 0.734 at step 512** — and the two
meet at 1.000 by step 1000. **The large eigenvalues go repulsive first and the
count catches up.** This is exactly the bulk-vs-outlier reading
`ov_per_head.py`'s own docstring said the disagreement was for; nobody had read
it.

*F1 — block D is exact, and feasible.* `(−S+A)` and `−Mᵀ` agree **bit for bit**
(`max|Δ| = 0.000e+00`) on all four heads. The flip preserves every singular
value to `≤1.3e-15`, the Frobenius norm to printed precision, and every
eigenvalue modulus to `≤1.1e-15`, while `attractive_energy_fraction_core` goes
**0.000 → 1.000** for `L7H8`, `L2H10`, `L9H8` (and 0.444 → 0.556 for the mixed
`L5H2`). The factored write-back `W_O' = −W_Vᵀ`, `W_V' = W_Oᵀ` reproduces it at
`1.2e-07` relative — the **fp32 storage floor** the OV artifact already carries
(`s[64]/s[0] = 4.9e-8`), i.e. exact to the data's own precision. So the Stage 3
control needs no rescaling, no matching search, no tolerance, and no new
machinery.

*F2 — "rotated or spread?" is BOTH, at different times, and §3.11 saw only the
second.* Overlap of `L7H8`'s top right-singular vector between adjacent steps,
with the top-2 subspace's principal cosines beside it:

| window | top-1 overlap | top-2 principal cos | reading |
|---|---|---|---|
| 0 → 512 | 1.000 → 0.882 | [0.940, 0.890] | stable |
| **512 → 1000** | **0.188** | **[0.352, 0.122]** | **rotation** |
| **1000 → 2000** | **0.282** | **[0.287, 0.002]** | **rotation** |
| 2000 → 16000 | 0.653 → 0.940 | rising to [0.943, 0.915] | re-forming |
| **16000 → 32000** | **0.490** | **[0.961, 0.951]** | **reordering** |
| 32000 → 143000 | 0.870 → 0.978 | [0.979, 0.974] | stable |

Two distinct events. **512–2000 is a genuine rotation** — the whole plane moves,
second principal cosine reaching **0.002** (orthogonal). **16000–32000 is
reordering inside a stable plane** — top-1 halves while the subspace holds at
[0.961, 0.951], which is `svd@1` falling 0.68 → 0.20 while `svd@2` stays
0.85–0.97, seen from the other side. §3.11's consolidation reading is right; the
earlier event was missed entirely.

**And the rotation window is the repulsive-collapse window is `CLAIM-B`'s
anchor window.** Three independent quantities — a population spectral phase, one
head's OV direction, and a registered literature anchor — all name 512–2000.
Whether that is one event or three is now the question worth asking.

**Design consequence, forced by measurement:** the stable object is the
**top-2 subspace**, not the top-1 direction. The Stage 3 entry must perturb the
plane. "The rank-1 direction" is well defined only inside the formation regime
(4000–16000).

*F3 — the found control is decisive, and it cuts against the spectral frame.*
At step 4000:

| head | attr. frac | complex frac | ‖M‖_F | σ₁ share | σ₁₂ share | participation |
|---|---|---|---|---|---|---|
| `L7H8` copier (ΔOV_nll **+0.244**) | 0.000 | 0.787 | 1.665 | 0.176 | 0.241 | 19.9 |
| `L2H10` anti-copier (**negative**) | 0.000 | 0.881 | 1.287 | 0.166 | 0.226 | 21.5 |
| `L9H8` anti-copier (**negative**) | 0.000 | 0.728 | 1.749 | 0.125 | 0.174 | 30.8 |
| `L5H2` prev-token | **0.444** | 0.896 | 2.568 | 0.062 | 0.118 | 27.5 |

**`L2H10` is spectrally near-indistinguishable from `L7H8`** — same repulsive
fraction (0.000), same gain concentration (0.166 vs 0.176), same participation
ratio (21.5 vs 19.9) — and its causal copying effect has **the opposite sign**.
So the OV spectral description **does not determine the causal role**. That is
§3.11's "the repulsive sign is true but not the mechanism" in its sharpest
available form, obtained from a *found* control rather than a constructed one,
and therefore immune to the matching problem that killed both constructed
designs. `L2H10` is the control head the S-flip experiment should run against.

Separately: `L5H2` is the **only** spectrally mixed head of the four (0.444) and
the least gain-concentrated. The two stages of the circuit are spectrally
unlike, which no one had checked.

**G. The derivation, and the negative it produced (2026-09-09).**
`MATH_SPECTRAL_OT.md` gains **§2.4, "The S/A split as an intervention"**, built
on §§2.1–2.3 rather than re-deriving them. Four results, three of which changed
the design and one of which killed a hypothesis of its own making.

*G1 — the readout Stage 3 depends on is a quadratic form in `S` (§2.4.1).* §2.1
derives the first-order result for `‖x‖`; the particle account is about
*inter-particle* distance. For `δ = x − y` the update is linear, so
`‖δ + Mδ‖² = ‖δ‖² + 2·δᵀSδ + ‖Mδ‖²` — identical form. Hence "individuating" is
literally the statement `δᵀSδ > 0` on the matched differences, `A` contributes
**exactly nothing** at first order, and **§3.12-C2's mechanical confound becomes
computable in closed form and subtractable** rather than something to control
for. `P-I5`'s pairwise readout and `P-ST1`'s effective-rank readout are the same
quadratic form on different arguments.

*G2 — the four sign choices are a group, and block D was under-specified
(§§2.4.2–2.4.3).* `{M, Mᵀ, −M, −Mᵀ} = {S+A, S−A, −S−A, −S+A}` is a complete
**2² factorial** in (sign `S`, sign `A`), and **every arm is an isometry** —
same singular values, same `‖·‖_F`, same eigenvalue moduli, no rescaling, no
draws. But `M = UΣVᵀ ⟹ Mᵀ = VΣUᵀ`, so **transposing swaps the read subspace
with the write subspace**, which for a copier is a change of function. It
resolves exactly: role-swap `⟺ sign(S) ≠ sign(A)`, i.e. the `S×A` interaction.
So **both main effects are clean and the interaction is aliased with the role
swap** — and block D's single `−Mᵀ` arm, run alone, confounds the `S` sign with
the swap completely. **Stage 3 must run all four cells.**

*G3 — the flip's residual is the non-normality (§2.4.4).* Applying `M` and
`−Mᵀ` to the same `δ`, the first-order terms cancel identically and the
difference is `4·δᵀSδ + δᵀ[Mᵀ,M]δ`. So the S-flip inverts the first order
*exactly*, and the whole deviation is a quadratic form in the self-commutator —
a third readout for free, and the one that answers §5.3(d) per head by
intervention rather than by residual.

*G4 — §2.4.5's own prediction, measured and FALSIFIED
(`induction_abscissa_7b.py`, §2.4.6).* Bendixson gives only the inclusion
`Re λ(M) ∈ [λ_min(S), λ_max(S)]`, so two heads can agree on every
eigenvalue-derived field and differ on the numerical abscissa — which would have
explained F3. **It does not.** At step 4000:

| | `attr_frac` | `λ_max(S)` | `λ_min(S)` | `S_pos_E` | `‖[Mᵀ,M]‖` |
|---|---|---|---|---|---|
| `L7H8` (ΔOV_nll **+0.244**) | 0.000 | 0.2644 | −0.4392 | 0.150 | 0.2968 |
| `L2H10` (**negative**) | 0.000 | 0.2238 | −0.3038 | 0.202 | 0.2924 |

The two candidate separators differ by ~15 % and ~35 % **in opposite
directions**, and the commutator agrees to two digits. Across its own layer
`L7H8` ranks **8th of 16** on `λ_max(S)` while carrying ~10× the layer's causal
effect.

**So `L7H8` is unremarkable in the eigenvalue frame, unremarkable in the
symmetric frame, and unremarkable in the gain frame — and it is the head that
does induction.** No weights-only spectral quantity this project computes
identifies the copier. (Population test of that claim:
`induction_spectral_predicts_7b.py`, 112 heads with a measured causal readout.)

This does **not** show the spectral character is causally inert — a static
property failing to *predict* which head copies is not the same as flipping it
failing to *change* what the head does. The G2 factorial is now the thing that
decides it. But it moves the prior hard, and it puts the *representational*
reading of induction in trouble on a third independent front, after the missing
token diagonal and after F3.

*G5 — and the developmental signal is where the structure is.* `L7H8`'s
`λ_max(S)` holds its initialisation value (~0.040) through step 1000, then rises
~20× — 0.104 at 2000, 0.264 at 4000, 0.455, 0.627, peaking **0.729 at 32000**
before falling to 0.385 at 143000. Its `S_pos_E` traces a **U-shape with floor
at step 2000** (0.076). That is the **fifth** quantity to name 512–2000, after
the population repulsive collapse (A), this head's OV plane rotation (F2), the
eigenvalue-frame U-shape (A), and `CLAIM-B`'s registered anchor. The
cross-sectional question has a negative answer; **the developmental one is the
live one**, which is sub-phase 7a rather than 7b.

*G6 — the population test, and it splits the fields cleanly
(`induction_spectral_predicts_7b.py`).* G4 was two heads. Seven
`induction_subspace_characterize*` runs already carry a measured
`full_ablation_delta_nll` for **every head of a layer** at four checkpoints —
112 heads with a causal readout and a dense `W_OV` on disk. Asking whether any
weights-only field rank-correlates with the causal effect: **Spearman ρ is ≈ 0
for all ten fields** (best `|mean ρ|` = 0.133; **no field reaches `|ρ| > 0.5` in
any of the 7 populations**).

**But the correlation was the wrong instrument, and the rank table is the right
one.** Fifteen of sixteen heads per layer carry causal noise, so Spearman
averages the signal away. What matters is where the *causally-top* head sits on
each field (rank 0 = that field's own maximum of 16):

| field | top-head rank across the 7 populations | extreme in |
|---|---|---|
| `sv12_share` | 2, 1, 11, 0, 0, 1, 1 | **6/7 top-3** |
| `lambda_min_S` | 13, 15, 11, 15, 15, 15, 14 | **6/7 bottom-3** |
| `sv1_share` | 2, 2, 12, 0, 0, 1, 3 | 5/7 top-3 |
| `nonnormality` | 2, 2, 6, 0, 0, 1, 3 | 5/7 top-3 |
| `participation_ratio` | 13, 13, 7, 15, 15, 14, 12 | 5/7 bottom-3 |
| `attractive_energy_fraction_core` | 8, 9, 2, 0, 14, 10, 2 | — |
| `sym_pos_energy_fraction` | 10, 8, 1, 0, 15, 14, 13 | — |
| `max_re_lambda` | 11, 9, 2, 0, 14, 14, 13 | — |
| `lambda_max_S` | 7, 7, 3, 0, 5, 8, 5 | — |
| `frobenius` | 8, 6, 1, 2, 13, 13, 5 | — |

Under a uniform-rank null the top group sits at ~2.6e-4 and ~3.5e-3, the bottom
group at 0.13–0.77. **That is arithmetic, not an adjudication** — the
populations share checkpoints, heads within a layer are not independent, and the
fields were not chosen in advance. No p-value is emitted and
`claims/registry.json` is untouched.

**The split is exact and it is the finding.** *Concentration and magnitude*
quantities identify the copier — `σ₁₂` share, `σ₁` share, participation ratio,
non-normality, and the single most-negative `λ(S)`. *Sign-balance* quantities do
not — the attractive fraction, `max Re λ`, `S_pos_E`, and (worst of all ten, at
0.77) `λ_max(S)`, the field §2.4.5 nominated. `‖M‖_F` alone does not either, so
it is concentration rather than size.

**Three consequences.**

1. **`r*_SVD ≪ r*_Schur` — the one Stage 1/2 finding that survived — is now a
   population result**, confirmed across 7 layer-populations and 4 checkpoints
   rather than one head at one step. The copier is the most gain-concentrated
   head in its layer, and that is what marks it.
2. **A pre-run prediction for Stage 3, which the drafted entry lacked.** The
   S-flip preserves *every singular value*, so it preserves gain concentration
   exactly. **Predicted before running: `−Mᵀ` will not destroy copying.**
3. **That sharpens the G2 factorial rather than weakening it.** If concentration
   is the whole story, even `Mᵀ` preserves copying; if the read/write
   *alignment* matters, `Mᵀ` destroys it while `−M` does not. So the informative
   contrast is **`M` vs `Mᵀ`** — alignment at fixed gain — and not the `S` sign
   at all. The factorial computes both for the same four forward passes.

**H. Composition and whitening (2026-09-09,
`tools/run/induction_composition_whitening.py`).** The two checks that needed a
model. Both returned, and both matter.

*H1 — the composition switches on between step 512 and step 1000, and it is the
cleanest circuit-level signal in the programme.* K-composition
`‖W_K^{L7H8} · W_OV^{L5H2}‖_F / (‖W_K‖_F‖W_OV‖_F)`, scored against **all 112
upstream heads** into the same key path, so "elevated" is against the model's
own distribution and no threshold is placed:

| step | K | pop median | **rank / 112** | z | Q | V |
|---|---|---|---|---|---|---|
| 0 – 256 | 0.0311 | 0.0313 | **83** | −0.56 | 0.0311 | 0.0320 |
| 512 | 0.0313 | 0.0313 | **66** | −0.08 | 0.0312 | 0.0320 |
| **1000** | 0.0346 | 0.0314 | **0** | **+5.78** | 0.0331 | 0.0317 |
| 2000 | 0.0502 | 0.0317 | **0** | +6.87 | 0.0486 | 0.0377 |
| 4000 | 0.0551 | 0.0318 | **0** | +5.74 | 0.0541 | 0.0438 |
| 143000 | 0.0778 | 0.0315 | **0** | +6.93 | 0.0778 | 0.0470 |

`L5H2 → L7H8` goes from **below the population median (rank 83 of 112)** to
**the single strongest composition into that key path (rank 0, z ≈ +6)** in one
step interval, and never leaves rank 0 again. **Sixth quantity to name
512–2000**, and the first that is unambiguously about the *circuit* rather than
about spectra.

**The honest caveat, and it is not small: K and Q rise together.** By step 8000
they are equal to three digits (0.0778 both at 143000). V is clearly lower
(0.0470) so the measurement is not purely generic, but **the Q/V arms were
computed without their own population controls**, so this does *not* yet isolate
the K-specific induction story — only that `L5H2`'s output becomes strongly
aligned with `L7H8`'s read subspaces. Scoring Q and V against their own 112-head
populations is the immediate next step and is the same cost.

*H2 — §3.12-C3 confirmed: the weight-identified direction is not the
data-identified one.* At step 4000, comparing the top singular directions of
`W_OV` against those of `W_OV Σ^{1/2}` (pulled back to input space):

- top-1 overlap **0.207**
- top-2 plane principal cosines **[0.293, 0.091]** — very nearly orthogonal planes
- but `σ₁` energy share 0.176 → 0.148 and `σ₁₂` **0.241 → 0.270**

So the *direction* does not survive whitening and the **concentration does**.

**Caveats, both real.** Σ is estimated from only **1536 tokens** for a 1024-dim
space (effective rank 238), so part of the low overlap is estimation noise —
raising `N_SEQS` is cheap and should be done before this is leaned on. And Σ
here comes from the repeated-random-token battery, not natural text; that is
arguably the *right* metric since it is the distribution the causal readout
uses, but it is not the model's operating distribution and the two should be
compared.

**H1-REVISED (2026-09-09, rerun with per-path population controls).** The
caveat was right and it is now settled: **the composition is not K-specific.**
Scoring Q and V against their own 112-head populations:

| step | K rank / z | Q rank / z | V rank / z |
|---|---|---|---|
| 0 – 512 | 83 / −0.56 | 72 / −0.38 | 2 / +1.94 |
| **1000** | **0 / +5.78** | 2 / +3.25 | 21 / +0.81 |
| 2000 | 0 / +6.87 | **0 / +6.40** | 0 / +6.07 |
| 8000 | 0 / +4.94 | **0 / +5.41** | 4 / +2.32 |
| 143000 | 0 / +6.93 | **0 / +6.79** | 6 / +1.54 |

**And the reason is an artifact, identified rather than guessed:** `L7H8`'s
`W_Q` and `W_K` **converge onto the same subspace** over training — mean
principal cosine between their rowspaces **0.213 → 0.829**, top principal cosine
**0.476 → 0.995**. Any operator composing into `K` therefore composes into `Q`,
and the composition score cannot separate the two pathways for this head.

*What survives, and it is most of it.* The **timing** (rank 83 of 112 → rank 0
between 512 and 1000) and the **magnitude** (z ≈ +6, sustained) stand
untouched — those never depended on which read path. What must be dropped is the
phrase "K-composition": the right description is **composition into `L7H8`'s
attention read-space**, which is *one* object because Q and K share it. And the
**V arm does discriminate** — it drifts to rank 4–6 at z ≈ +1.5 while K and Q
hold rank 0 at z ≈ +6 — so the composition is into the **attention** pathway and
not the **value** pathway, which is the induction-shaped result and is the part
that was actually worth having. At the onset step alone (1000) K does lead Q,
rank 0 / z +5.78 against rank 2 / z +3.25; one step and a small gap, recorded
and not leaned on.

**H2-CONFIRMED, by a control that needed no model of the noise.** Three arms
plus a split-half at step 4000:

| arm | tokens | top-1 overlap | plane cos | Σ eff. rank |
|---|---|---|---|---|
| battery | 24,576 | 0.217 | [0.329, 0.088] | 327 |
| battery half 1 | 12,288 | 0.222 | [0.320, 0.088] | — |
| battery half 2 | 12,288 | 0.213 | [0.336, 0.088] | — |
| natural text | 3,185 | 0.266 | [0.270, 0.104] | 281 |

**Split-half: half 1 against half 2 agrees at top-1 0.990 and plane cos
[1.000, 0.778].** The two independent halves agree with *each other* at 0.99
while both disagree with raw at ~0.22 — so **the low overlap is signal, not
estimation noise**, and H2's caveat is discharged. Sixteen times the tokens
moved the answer 0.207 → 0.217, so the original estimate was already sound. And
`Σ(battery)` vs `Σ(natural)` have cosine **0.453** — substantially different
metrics giving the same answer, which is the robustness the caveat asked for.

*H3 — three independent results now converge on the same correction.* G6 says
**concentration** identifies the copier and directional/sign quantities do not.
H2 says concentration **survives whitening** and the direction does not. F2 says
the top-1 direction is not even stable across checkpoints while the top-2
*plane* is. So F2's "perturb the top-2 subspace" fix is **still not enough** —
the object has to be defined in the whitened metric, or, better, **the Stage 3
entry should be about gain concentration rather than about any named
direction.** That is a different prediction from the one §3.11 asked for, and it
is the one the measurements support.

**I. The matcher becomes a similarity kernel, and this is the first quantity
that cleanly identifies the head (2026-09-09).** H1-REVISED's artifact — `W_Q`
and `W_K` converging onto one subspace — is not only an artifact. A head whose
query and key read the *same* subspace computes an attention logit
`qᵀk = xᵀ(W_Qᵀ W_K)y` that is close to a **similarity form**, i.e. the static QK
operator should be becoming **symmetric**. Measured directly, on
`M = W_Q[16:]ᵀ W_K[16:]` (the same static operator §3.11 block C sweeps), as
`‖S‖²_F / ‖M‖²_F`:

| step | `L7H8` sym. fraction | `‖M‖_F` | **layer-7 median** | `L7H8` rank / 16 |
|---|---|---|---|---|
| 0 | 0.5017 | 2.78 | 0.5006 | 2 |
| 512 | 0.5019 | 2.79 | 0.5008 | 1 |
| 1000 | 0.5042 | 2.87 | 0.5034 | 6 |
| **2000** | **0.5742** | 3.50 | 0.5095 | **0** |
| 4000 | 0.7398 | 4.44 | 0.5181 | **0** |
| 8000 | 0.8807 | 5.90 | 0.5200 | **0** |
| 16000 | 0.9311 | 7.85 | 0.5263 | **0** |
| 32000 | 0.9465 | 9.86 | 0.5257 | **0** |
| 143000 | **0.9561** | 7.91 | 0.5203 | **0** |

A random real matrix splits its energy 50/50 between `S` and `A`, and 0.50 is
where every head starts. **`L7H8` goes to 0.956 while its fifteen neighbours
stay at the random baseline (layer median 0.520 at step 143000), and it is rank
0 of 16 at every step from 2000 on.** Takeoff is 1000 → 2000 — the **seventh**
quantity to name that window.

**Three things this is** *(claim 1 below was overstated and is corrected in
block J — left standing because the correction is the point)*.

1. **The first weights-only quantity in the entire programme that cleanly
   identifies the head.** §3.12-G6's negative — no spectral field picks out the
   copier — surveyed the **OV** operator only. On the **QK** side, symmetry
   identifies the matcher decisively: 0.956 against a layer median of 0.520,
   rank 0 of 16, sustained over seven checkpoints.
2. **It explains a previously descriptive finding.** §3.11 block C reports the
   matcher as a "near-normal" invariant subspace with `r*_Schur` 12 < `r*_SVD`
   32. A symmetric operator **is** normal, so near-symmetric *derives*
   near-normal rather than restating it, and it explains why the Schur frame is
   the right one for the matcher and the wrong one for the copier.
3. **"Matching kernel" stops being a metaphor.** The particle account's own
   framing is that induction is "a matching-kernel coupling rather than a
   feature-copying circuit" (`POPPER_PLAN.md` §C2). The static QK half of this
   head *is* a similarity kernel, at 0.956. That is the account's language
   arriving as a measurement — **on the half nobody was testing**.

**What it does not do, stated because the temptation is obvious.** It adjudicates
nothing: `claims/registry.json` is untouched, this is one head in one model, and
the differential prediction §3.11 wanted was about **OV**, where the evidence
still runs the other way. A symmetric *content* operator is also exactly what
one would expect architecturally — RoPE carries the positional asymmetry in the
16 excluded dims and the causal mask carries the rest, so "symmetric content
part plus positional part" is the ordinary way to build a content matcher, and
that reading has to be ruled out before any of this is registered. The obvious
next measurements: the same sweep on the other genuine induction heads
(`L9H9`, `L6H0`, `L1H15` — block C's 3-of-4), and on non-induction heads that
also match, to see whether symmetry tracks *induction* or merely tracks
*matching*.

**J. It tracks MATCHING, not induction — block I's claim 1 is corrected
(2026-09-09, `tools/run/qk_symmetry_sweep.py`, all 384 heads × 19 steps).**
The symmetric fraction reduces to 48×48 algebra — `‖M‖²_F = tr((AᵀA)(BBᵀ))` and
`tr(M²) = tr((BA)²)` for `M = AB` — so no (1024,1024) matrix is ever formed;
the identity is asserted against the direct computation on the first head of
every run rather than trusted.

*The population.* Every head sits at the random baseline (median 0.5005, max
0.5027, **none above 0.7**) through step **512**. First movement at 1000; by
143000 the median is 0.5231 and only **11 of 384** heads exceed 0.7, **one**
exceeds 0.9. So high symmetry is genuinely rare — but it is **not unique**.

*The decisive cell, at 143000 — top-10 symmetry heads and their induction rank
of 384:* `L7H8` 0.956 (rank 2), `L6H0` 0.873 (**0**), `L1H15` 0.840 (5),
`L2H10` 0.835 (**1**), `L7H0` 0.796 (3), `L9H9` 0.738 (7), **`L1H4` 0.734
(40)**, **`L8H13` 0.730 (47)**, `L9H8` 0.727 (4), **`L7H4` 0.708 (91)**.

**Seven of the top ten are top-8 induction heads — and three are not.** At step
4000 the mismatch was starker still (`L10H3`, symmetry rank 5, induction rank
**382 of 384**). So symmetry is close to *necessary* for induction and clearly
not *sufficient*: **block I's "the first quantity that cleanly identifies the
head" was too strong.** What symmetry identifies is the **matching half of the
circuit**, and induction heads are matchers.

*Two controls settle what "matching" means here, and neither was arranged.*

- **`L5H2`, the previous-token head, never becomes symmetric** — 0.500 → 0.504,
  rank 322 → 310 of 384, flat across the entire axis while its induction
  partners climb past 0.8. It is an overwhelming matcher (attention 0.895 at
  offset −1, ~49× the head median) but it matches on **position**. So symmetry
  marks **content** matching specifically, and the architectural reading block I
  flagged is *partly right*: it is about matching, not about induction.
- **`L2H10` matches without copying.** It reaches symmetry rank 3 (0.835) and
  induction rank 1 while its OV copy effect is **negative**. Its trajectory —
  rank 131 at 4000, 23 at 8000, 4 at 16000, 3 thereafter — means the step-4000
  reading ("the anti-copier is not symmetric") **does not survive the full
  axis**, and what replaces it is better: **matching (QK, symmetric) and copying
  (OV) are separable, and this head has one without the other.**

*What the two halves now look like together.* §3.12-G6: on the **OV** side no
sign quantity identifies the copier and only *gain concentration* does, weakly.
Block J: on the **QK** side *symmetry* identifies the matcher, sharply and
rarely (11 of 384). **Induction is both**, and `L2H10` is the existence proof
that they come apart. That is the dissociation §3.11 block C was reaching for,
now on a population footing rather than one head.

*Developmentally — the eighth quantity to name the window.* Flat at baseline
through 512; first movement at 1000; by 2000 `L9H9` leads at rank 0 (0.595)
with `L7H8` at rank 3, and from 4000 `L7H8` takes rank 0 and never gives it up.
And the high-symmetry set **purifies toward induction heads over training** —
the worst induction rank in the symmetry top-10 goes from 382 at step 4000 to 91
at 143000.

*A §3.13 footnote that is not a footnote.* Spearman of symmetry against
induction over all 384 heads is ≈ 0 at every step **except step 1000, where it
is +0.343**. The correlation is the right instrument exactly once — at the onset,
when many heads move slightly together — and the wrong one everywhere after,
when the signal concentrates into a handful. **The right summary changes with
training stage**, which is a sharper version of §3.13 than that section states:
the mean-versus-extremum choice is not only per-quantity, it is per-regime.

**K. What the position profile found instead (2026-09-09).** §3.13.3's own
hypothesis came back negative, but the profile carries two things nobody had
looked for.

*K1 — a mechanism check the readout has never made, and it passes.* At the first
second-copy position `j = 0` the model predicts `ids[N_REP]` from the first copy
alone: **no earlier occurrence of the current token exists yet, so induction
cannot fire there.** Measured, `j = 0` is the **only** position with a negative
effect — ΔNLL **−0.0484**, and it is the run's minimum — so ablating `L7H8`'s OV
slightly *helps* exactly where induction is impossible. Its induction attention
is also the lowest of any position, **0.779** against a mean of 0.910. The
prediction was made from the slice convention before the run and both halves
hold. This is the first end-to-end validation that `second_copy_nll` is
measuring what it is supposed to measure.

*K2 — there is real position structure; it just does not bias the mean.* ΔNLL
runs **−0.048 at j = 0**, peaks at **+0.570 at j = 4**, falls to **+0.390 at
j = 8**, and settles onto a plateau near **+0.22** for the rest of the sequence.
So the copying effect is largest immediately after induction becomes possible
and then decays to about a third of its peak — consistent with the model having
progressively more non-induction evidence as the repeat proceeds, and worth a
look on its own.

*K3 — and the QK/OV dissociation appears on a third axis.* Across the same
positions **induction attention is flat at 0.90–0.93** while **ΔNLL varies
roughly fourfold (0.20 → 0.57)**. The matcher fires essentially uniformly; the
copier's *payoff* does not. That is §3.11 block C's dissociation and §3.12-J's
two-halves picture arriving from the position axis, which neither was derived
from.

**L. The Stage-1 table compares operators of different size (2026-09-09).**
Found while stress-testing the Stage 3 design, and it revises a headline.

`tools/run/induction_rank_sweep.py::truncate` returns, at rank `r`: for `svd` a
genuine rank-`r` truncation (`U[:, :r] * s[:r], Vt[:r]`), and for `schur` and
`random` a **projection of `A`** onto an `r`-dimensional subspace (`A @ P, B`).
**None preserves Frobenius norm, and they lose it at very different rates.**
Measured on `L7H8` at step 4000, energy retained as a fraction of the full OV:

| r | svd | schur | random | svd/rand | svd/schur |
|---|---|---|---|---|---|
| **1** | **0.1755** | **0.0373** | **0.0108** | **16.2×** | **4.7×** |
| 4 | 0.3368 | 0.1398 | 0.0518 | 6.5× | 2.4× |
| 16 | 0.6153 | 0.4039 | 0.2315 | 2.7× | 1.5× |
| 48 | 0.9263 | 0.8017 | 0.7050 | 1.3× | 1.2× |

The `random` branch's own docstring says *"Matched-norm random rank-r control …
so the operator norm scale and the factor structure match the real
truncation."* **At `r = 1` it is off by 16×.** So §3.11's Stage-1 table — the
one reading `r*_SVD ≪ r*_Schur` and "random ≈ 0 %" — **compares operators of
different size at the same rank.**

*Re-read at matched ENERGY instead of matched rank*, using the NLLs already in
`induction_rank_sweep.json` (no new forward passes):

| energy | svd | schur | random |
|---|---|---|---|
| 0.175 (= svd's r=1) | **0.820** (r 1) | **0.599** (r≈4.8) | **0.353** (r≈12.6) |
| 0.25 | 0.847 | 0.760 | 0.427 |
| 0.40 | 0.924 | 0.807 | 0.586 |
| 0.60 | 0.964 | 0.893 | 0.814 |

**The ordering survives — SVD > Schur > random at every matched energy — but the
gaps collapse.** SVD-over-Schur goes from **6.9×** at matched rank (0.820 vs
0.119) to **1.37×** at matched energy. Random is not "≈ 0 %"; at `r = 1` it
holds 1.08 % of the energy, and given 17.55 % it recovers **0.353**.

*What stands and what does not.* The **structural** claim stands: at matched
energy the top singular directions beat structureless ones **2.3×**, so this is
not energy alone. What does not stand is the *magnitude* of `r*_SVD ≪
r*_Schur` — 1.37× is much weaker support for "the copying action lives in a
high-gain non-invariant direction, and the attractive/repulsive frame is the
wrong description" than 6.9× was. Eckart–Young makes "SVD retains most energy at
rank `r`" a **theorem**, so at matched rank part of the gap was never a finding.
**§3.12-G6's population result is unaffected** — it correlates gain
concentration with *which head* is the copier and never uses this comparison.

*And this is §6n's own rule applied to this project's headline for the first
time*: match the control on **the quantity the statistic degenerates on**. It
degenerates on energy; the controls were matched on rank.

**Consequence for Stage 3, and it is a convergence.** Any Stage 3 arm must be
**energy-matched, not rank-matched**. The `{M, Mᵀ, −M, −Mᵀ}` factorial is
*exactly* energy-matched — every arm is an isometry (§2.4.2) — so the design
motivated by the S/A group structure turns out to be the fix for this confound
as well, from a completely independent direction.

*Precondition on the S-flip, computed before it runs.* If `S` carried little of
the OV's energy, "the flip does nothing" would be guaranteed by magnitude rather
than by mechanism. It does not: `‖S‖²/‖M‖²` for `L7H8`'s OV is **0.602** at step
4000 and 0.543 at 143000, so `‖M − (−Mᵀ)‖ = ‖2S‖ ≈ 1.55‖M‖` — **the change is
larger than the operator**. The precondition passes.

*But it also shows the two halves are structurally different in exactly the way
§3.12-J found.* `L7H8`'s **OV** symmetric fraction is 0.602 against a layer
median of 0.584 — at the baseline, carrying no signal. Its **QK** symmetric
fraction is **0.956 against a layer median of 0.520**. The matcher becomes a
similarity kernel; the copier does not, and never does.

**M. The S/A factorial is not a viable Stage 3 design — its own pilot killed it,
and produced a better result (2026-09-09,
`tools/run/induction_sa_pilot.py`).** Step 4000, 16 sequences, restore exact
(`0.0e+00`) on every head. ΔNLL against the unablated model:

| arm | `L7H8` ΔNLL | KL | /ablation | `L9H9` | `L2H10` |
|---|---|---|---|---|---|
| ablation | +0.2420 | 0.093 | — | +0.0752 | −0.0129 |
| scale 0.25 | +0.1412 | 0.036 | | +0.0457 | −0.0096 |
| scale 0.50 | +0.0729 | 0.011 | | +0.0242 | −0.0034 |
| scale 0.75 | +0.0279 | 0.002 | | +0.0094 | −0.0015 |
| **`Mᵀ`** | **+0.2278** | 0.089 | **0.94×** | +0.0928 (1.23×) | −0.0112 |
| **`−M`** | **+1.2409** | **0.969** | **5.13×** | +0.3163 (4.21×) | −0.0076 |
| **`−Mᵀ`** | **+0.4000** | 0.181 | **1.65×** | +0.1113 (1.48×) | −0.0083 |

*M1 — `−M` is off-scale, exactly as predicted.* **5.13×** ablation on `L7H8`,
4.21× on `L9H9`, with KL **0.969** against ablation's 0.093 — an order of
magnitude. Reversing the write is far more destructive than removing it, because
the head actively suppresses the token it used to promote. **So the balanced
main effects — which average `−M` with `−Mᵀ` — are dominated by one catastrophic
arm, and additivity fails.** The prediction was made from the algebra before the
run and both halves hold.

*M2 — and the design cannot answer its own question.* §2.4.3 proved every
pairwise contrast isolating a sign **also swaps the read/write role**. The pilot
now measures what that swap costs on its own: **`Mᵀ` alone destroys copying at
0.94× ablation.** So by the time the transpose has been applied the effect is
already gone, and the `S`-sign contrast (`Mᵀ` → `−Mᵀ`, 0.94× → 1.65×) is
measured **on top of a floor**. The main effects cannot rescue it because of M1.
**Both routes to the `S` sign are blocked, and §3.12-L's rank result shows no
other points exist** — so this is not a design to be repaired. It is retired
before registration, which is what the pilot was for.

*M3 — `G6`'s pre-run prediction is FALSIFIED, and the falsification is the
finding.* G6 predicted `−Mᵀ` would **not** destroy copying, because it preserves
every singular value and hence gain concentration exactly. It destroys it at
**1.65× ablation**. And `Mᵀ` — same singular values, same Frobenius norm, same
eigenvalue moduli, *nothing changed but which subspace reads and which writes* —
destroys it at **0.94×**, as completely as deleting the head.

**So the copier's function is carried by read/write ALIGNMENT, not by its
spectrum and not by its gain profile.** §3.12-G6 showed gain concentration
identifies *which head* is a copier; it does not carry *the copying*. That is
consistent with §3.12-H2 — the top OV direction does not survive whitening
(overlap 0.217) while the concentration does — and it completes that reading:
concentration is a **marker**, alignment is the **mechanism**.

*M4 — `L2H10` is not usable as a control, measured rather than argued.* Its
whole dynamic range is |ΔNLL| ≤ 0.013 and the arm ratios (0.87×, 0.59×, 0.65×)
are noise on that scale. A head whose baseline effect is ~2 % of the target's
cannot calibrate an intervention on the target. The §3.12-J finding that it
matches without copying stands; its use as the Stage 3 control does not.

*M5 — a nonlinearity worth carrying forward.* The scale curve is strongly
**sublinear**: halving `L7H8`'s OV costs only +0.073 against ablation's +0.242,
so **50 % of the operator does 30 % of the damage**. Any future arm reported in
"equivalent λ" is on a compressive scale and must say so.

**N. Literature check before registering (2026-09-09).** Run because §3.12-I/J
and §2.4 are close enough to published work that registering without checking
would risk re-deriving it. Four threads, and they change what should be claimed.

*N1 — QK symmetry is known at the population level, and our result is the
complement rather than a duplicate.* Saponati et al., *"The underlying
structures of self-attention: symmetry, directionality, and emergent dynamics in
Transformer training"* (arXiv 2502.10927), decomposes `W_QK` into symmetric and
skew parts and defines a Frobenius symmetry score. **Their finding is that
bidirectional training induces symmetry while autoregressive training induces
directionality** — decoder-only models score *more directional* than
encoder-only. That is a **median across layers, model-level, with no per-head
breakdown, no head-type analysis, no mention of induction or previous-token
heads, and no exclusion of positional dimensions.**

**This does not contradict §3.12-J; it frames it.** Our population median is
**0.523** — essentially neutral, consistent with their directional aggregate.
What §3.12-J adds is the **tail**: 11 of 384 heads exceed 0.7, one reaches
0.956, and those heads are the content matchers. Their claim is about the
median; ours is about which heads leave it, and why. The RoPE exclusion matters
here too — pythia's positional (directional) information lives in the 16 rotary
dims we remove, so we measure the *content* operator they do not separate.

*N2 — the mechanistic reading of the S/A split exists, and it gives §3.12-J its
name.* *"The Routing and Filtering Structure of Attention"* (arXiv 2605.18826)
splits the pre-softmax score matrix into a symmetric **"filtering"** part
(undirected mutual relevance) and a skew **"routing"** part (directional
transport, purely imaginary eigenvalues), and finds routing removal catastrophic
(699 PPL against a 34.99 baseline). Since `A = X M X^T` for `M = W_Q W_K^T`,
their split of the *scores* is our split of the *weights* conjugated by the
activations — the same decomposition at two levels, which should be said rather
than discovered later. **In their vocabulary §3.12-J reads: `L7H8`'s content
operator becomes almost pure filtering (0.956) with almost no routing** — the
routing being supplied by RoPE and the causal mask, which is exactly the
architectural division block I flagged. They report **no transposition
experiment, no induction-head connection, and no OV analysis.**

*N3 — the induction-head emergence window is established, so §3.12's "eighth
quantity" claim needs restating.* Published checkpoint studies of Pythia put
induction-head emergence at **around step 1000 of 143000**. The 512–2000 window
is therefore **not our finding**, and this document should stop implying
novelty for the window itself. What is ours is the **set of quantities that
co-locate in it** — the population OV repulsive collapse, `L7H8`'s OV plane
rotation, the eigenvalue-frame U-shape, the numerical-abscissa takeoff, the
composition switch-on, and the QK symmetry takeoff — several of which are
weights-only and none of which is the behavioural score the literature dates the
window by.

*N4 — and the actionable one: we have not computed the field's own copying
score.* Elhage et al.'s copying test takes the eigenvalues of the **token-basis**
OV circuit `W_E W_V W_O W_U` and summarises their positiveness as
`sum(lambda) / sum(|lambda|)`; a copier has **positive** eigenvalues.
**This project has never computed that matrix.**
`p2b_imaginary/head_circuits.head_core` returns `W_V W_O` — the `(64,64)` core in
the **residual** basis — and every "100 % repulsive" statement in §3.11 and
§3.12-A is about *that*, not about the token-basis circuit. The two differ by the
vocabulary round-trip `W_U W_E` sandwiched between the factors, and **nothing
guarantees they share a sign**.

It is cheap: with `W_OV = A B`, the nonzero spectrum of `W_E A B W_U` equals that
of the `(64,64)` matrix `B W_U W_E A`, so it costs one `64x64` eigendecomposition
per head — the same price as the core we already compute. **If `L7H8` scores as
a copier on the field-standard measure, then §3.11's "not a token-identity
copier" — which rests on the weaker LN-folded diagonal check the section itself
flags as its softest half — is measuring a different thing, and the
"repulsive/individuating" reading cannot be carried over to it.** This is the
next measurement, before any registration.

**O. It was measured, and it CONFIRMS §3.11 — emphatically (2026-09-09,
`tools/run/copying_score_sweep.py`).** All 384 heads, eight checkpoints, weights
only. A convention trap was checked rather than assumed first: `ov_factors`
returns `OV_h = (W_O W_V)ᵀ`, the **transpose** of the residual operator (verified
at relative error 0.0). Harmless for every quantity read off it so far — all
transpose-invariant — but **not** for a copying score, which is directional, so
this runner uses `W_O W_V` from the model.

*O1 — `L7H8` is not a token-identity copier at any point in training, and it
becomes less of one as its causal effect grows.*

| step | `L7H8` copying score | rank / 384 | `ΔOV_nll` (§3.11-A) |
|---|---|---|---|
| 512 | −0.038 | 324 | ~0 |
| 1000 | −0.050 | 305 | ~0 |
| 2000 | −0.043 | 258 | ~0 |
| 4000 | **+0.062** | 139 | **+0.24** |
| 8000 | +0.020 | 181 | **+0.73** |
| 16000 | −0.050 | 267 | **+1.02** |
| 32000 | −0.115 | 322 | — |
| 143000 | **−0.094** | **328** | +1.18 |

**Its causal OV effect grows roughly sixfold while its copying score stays at or
below zero and its rank falls to 328 of 384.** The two run in opposite
directions. §3.11's conclusion was reached on the LN-folded diagonal check it
called its own softest half; it now stands on the field's instrument, across the
whole axis, and stronger than when it was stated.

*O2 — and the model is full of real copiers, none of them these heads.* Scores
above 0.4 go 0 (step 512) → 10 → 24 → 56 (32000) → 41 (143000), with the maximum
rising to **+0.723**. The top ten at 143000 — `L11H14` 0.723, `L13H5` 0.708,
`L18H8` 0.675, `L12H8` 0.661, `L10H0` 0.649, `L9H0` 0.635, `L17H10`, `L20H15`,
`L11H2`, `L17H6` — sit in **layers 9–20, all downstream of `L7H8`**. So the
measure is not blind: it finds copying where copying is, and reports its absence
at the induction heads.

*O3 — two internal consistency checks nobody arranged.* `L9H8`, whose `ΔOV_nll`
is **negative**, has a **negative** copying score at every step (−0.096 to
−0.109, rank ~305–312). And `L9H9` — the one induction head with a real positive
OV effect besides `L7H8` — is the one induction head that **does** become a
modest copier (+0.206 at 16000, rank ~104), and it sits in layer 9, at the
boundary of the copier band. The measure tracks the causal readout where the two
should agree.

*O4 — LN sensitivity is discharged.* Median raw against final-LN-gain-and-
centring folded, at all eight steps: differences in the **fourth decimal**
(+0.0324 vs +0.0293 at 143000). The softest half of §3.11's copy-score reading
is no longer load-bearing anywhere.

> **What §3.37's review adds to O, and one thing it leaves open (2026-09-16).**
> The measure O computes is **the** field-standard statistic — Elhage et al.
> (2021) propose the positive-eigenvalue fraction of the token-basis OV circuit
> `W_E W_OV W_U`, summarised as `Σλ/Σ|λ|`, as the copying-head detector. So this
> project has **already adopted the field's instrument, on the checkpoint axis,
> for all 384 heads**, which is more than `p2_eigenspectra/lit-2.md`'s first
> draft credited it with — that file proposed computing it as a growth direction
> and has been corrected.
>
> **What is still not crossed** is N4's own distinction: `frac_repulsive` and the
> `U_pos`/`U_neg` projectors are **residual-basis** quantities, the copying score
> is **token-basis**, and the two differ by the vocabulary round-trip `W_U W_E`
> with nothing guaranteeing a shared sign. O crossed the copying score against
> the **causal** readout (`ΔOV_nll`) and found them running in opposite
> directions. **Crossing it against `frac_repulsive` — the Phase 2 quantity the
> "attractive/repulsive" reading is actually built on — has never been done, and
> both columns already exist on disk.** If they agree, Phase 2's projectors get
> an independent validation and a translation into the field's vocabulary. If
> they disagree, that is a third instance of the project's recurring
> weight-space-vs-function-space dissociation (§3.12-R, §3.12-S), this time
> *between two weight-space measures*, and it would mean "repulsive" and
> "anti-copying" are not the same claim.
>
> **One cross-reference O and 7e should share:** `L11H14` is the **top copier at
> 143000 (+0.723)** and the full-rank anti-ordered outlier of §3.12-V4. The head
> whose singular directions do not order its causal usefulness is the head the
> token-basis measure ranks first.

**What this sharpens into, and it is the live question now.** `L7H8` has the
largest OV causal effect in its layer (10× the layer mean, growing sixfold across
training) **and is an anti-copier by the token-identity test**. Both now rest on
solid instruments. So its OV write is causally important for repeated-token
prediction *without* being token-identity copying — which **falsifies the
standard account's central claim for this head**, on the standard account's own
measure.

The natural reading is that the circuit has **three stages, not two**:

    L5H2  (layer 5)     positional matcher, prev-token
    L7H8  (layer 7)     content matcher (QK symmetry 0.956), writes NOT token identity
    L9-L20              the actual token-identity copiers

If that holds, §3.11 has been calling `L7H8` "the copier" because ablating its OV
moves second-copy NLL — but it may be **upstream** of the copier and its ΔNLL
mediated. Testable with machinery already built: composition `L7H8 →` the
downstream copiers' `V` and `K` paths, and whether ablating `L7H8`'s OV
suppresses their contribution. **One cross-reference already points that way:**
`L10H3` is a top-ten copier (0.530 at step 4000) and was §3.12-J's counterexample
— high QK symmetry, induction rank 382 of 384. It copies without matching;
`L7H8` matches without copying. The dissociation now has named heads on both
sides.

**P. The three-stage reading is FALSIFIED, on both tests (2026-09-09,
`tools/run/three_stage_mediation.py`).** Step 16000, 16 sequences, restore exact.
`ΔNLL(ablate L7H8 alone) = +1.1069` against a baseline of 0.5335.

*P1 — mediation: no sub-additivity anywhere, and three copiers are
**super**-additive.* `I = ΔNLL(both) − ΔNLL(L7H8) − ΔNLL(C)`:

| head | kind | ΔNLL(C) | ΔNLL(both) | **I** | reading |
|---|---|---|---|---|---|
| `L13H5` | copier | −0.0000 | +1.1259 | +0.019 | independent |
| `L12H8` | copier | +0.0104 | +1.1331 | +0.016 | independent |
| `L9H0` | copier | +0.0320 | +1.1810 | **+0.042** | super-additive |
| `L10H0` | copier | +0.0452 | +1.2703 | **+0.118** | super-additive |
| **`L11H14`** | copier | **+0.1869** | **+2.2753** | **+0.982** | **super-additive** |
| `L11H10` | ctrl | +0.0004 | +1.1252 | +0.018 | independent |
| `L11H0` | ctrl | +0.0157 | +1.1324 | +0.010 | independent |
| `L9H15` | ctrl | +0.0037 | +1.1117 | +0.001 | independent |

A serial circuit predicts `I < 0` — once the upstream stage is gone there is less
for the downstream one to do. **Every measured `I` is ≥ 0.** The clearest case
inverts the prediction outright: `L11H14` alone costs +0.187 and `L7H8` alone
costs +1.107, but **both together cost +2.275**. With `L7H8` intact, `L11H14`
barely matters; with `L7H8` gone, removing it costs five times as much. That is
**redundancy between parallel paths that partially substitute for each other**,
not mediation.

*And the sign is conservative.* §3.12-M5 measured the readout as **compressive**
(half the operator does 30 % of the damage), and a compressive readout pushes
genuinely independent contributions toward *apparent sub-additivity*. Observing
super-additivity against that bias strengthens the reading rather than weakening
it.

*P2 — composition: the prediction fails, and the controls are why we know.*
`L7H8`'s OV into each copier's read paths, ranked against **every head in the
layers below it**:

| head | kind | Q rank (z) | K rank (z) | V rank (z) |
|---|---|---|---|---|
| `L9H0` | copier | **4** (+2.81) | 63 (−0.06) | 8 (+1.39) |
| `L11H14` | copier | 6 (+2.67) | 142 (−0.54) | 15 (+1.07) |
| `L10H0` | copier | 9 (+1.85) | 99 (−0.31) | 73 (+0.25) |
| `L13H5` | copier | 54 (+0.37) | 59 (+0.39) | 10 (+1.59) |
| `L12H8` | copier | 33 (+0.86) | 58 (+0.32) | 102 (−0.12) |
| **`L11H0`** | **ctrl** | **0 (+3.02)** | 61 (+0.21) | 48 (+0.54) |
| **`L9H15`** | **ctrl** | 39 (+0.33) | 52 (+0.02) | **4 (+2.83)** |
| `L11H10` | ctrl | 113 (−0.51) | 77 (−0.09) | 69 (+0.04) |

The three-stage prediction was **elevated V-composition** — the copier copies
what `L7H8` wrote. Some copiers are modestly elevated on V (ranks 8–15), **but
the control `L9H15` ranks 4th at z +2.83, above every copier.** Q looks elevated
for copiers until the control `L11H0` ranks **0th at z +3.02**, above every
copier. K is flat everywhere. **On both paths a non-copier control beats the
copiers, so composition supplies no evidence for a specific `L7H8` → copier
pathway.** Without the controls, "V rank 10, z +1.59" would have been read as
support; this is the §3.12-H1 lesson (a population control *per path*) paying for
itself a second time.

*P3 — what survives, and it is a sharper puzzle than before.* `L7H8`'s OV write
is **causally enormous** (+1.107 at step 16000, ~2× the baseline NLL), is **not
token-identity copying** (§3.12-O), and **does not route through the heads that
do token-identity copying** (P1, P2). Three explanations remain, and the first is
a gap in this test rather than a hypothesis:

1. **It writes to MLPs, which were never measured.** Every composition score in
   §3.12 is head→head. MLPs are the majority of the parameters and the obvious
   place for a non-token-identity signal to be read. **This is the next
   measurement.**
2. It acts on the unembedding directly but not by token identity — boosting a
   *class*, or suppressing alternatives.
3. It acts on residual-stream geometry rather than any single readable
   direction — which is the particle account's own claim, and the one §3.14.2's
   case-study programme is built to examine.

**Q. Five probes: three negative, and the surviving one corrects a framing
(2026-09-09, `tools/run/what_l7h8_writes.py`, step 16000).** Ordered by what
could reframe the question rather than by convenience; MLPs deliberately last.

*Q1 — the offset. My off-by-one worry was WRONG, and that deepens the puzzle.*
`induction_candidates` documents two conventions and the repo uses the
non-standard one (`ids[key-1] == ids[query-1]`, pairing query `N_REP+j` with key
`j`, the **same-token** position) rather than the Anthropic one
(`ids[key-1] == ids[query]`, pairing with `j+1`, the **successor**). Measured
attention from query `N_REP+j`:

| offset | j−2 | j−1 | **j** | j+1 | j+2 |
|---|---|---|---|---|---|
| `L7H8` | 0.0012 | 0.0013 | **0.9339** | 0.0000 | 0.0000 |

**93.4 % at exactly `j`, and nothing at `j+1`.** So the repo's convention
correctly describes this head. But copying from the same-token position returns
the **current** token when the answer is the **successor** — so a token-identity
copier here would be actively wrong, which is consistent with §3.12-O's near-zero
score and makes "what does it write" harder, not easier.

*Q2 — the effect is a large GLOBAL logit shift, not a targeted promotion.*
`ΔNLL = −(Δlogit_correct − Δlogsumexp)` exactly. Ablating `L7H8`'s OV gives
`Δlogit_correct = −5.02` **and** `Δlogsumexp = −3.91` — every logit falls by
about four nats and the correct one by five, netting +1.11. This is not a
delicate promotion of one token; it is the removal of a large component of the
residual.

*Q3 — and the control says that is specific, not generic — while correcting a
framing this document has carried.* Ablating each head's OV, against a baseline
logsumexp of 16.05 and final residual norm of 52.95:

| head | ΔNLL | Δlogit(correct) | Δlogsumexp | Δ‖resid‖ |
|---|---|---|---|---|
| **`L5H2`** (prev-token) | **+2.227** | −5.89 | −3.67 | **−5.92** |
| `L7H8` | +1.107 | −5.02 | −3.91 | **−8.38** |
| `L11H14` (top copier) | +0.187 | −1.02 | −0.83 | −0.40 |
| `L13H5` (top copier) | −0.000 | −0.02 | −0.02 | −0.06 |
| `L2H10`, `L0H0`, `L20H7` | ≤ +0.008 | ≤ 0.10 | ≤ 0.10 | ≤ 0.19 |

Two things. **The global shift is not generic** — ordinary heads move logsumexp
by under 0.1 where these two move it by ~4, and they remove 11–16 % of the final
residual norm against under 0.4 % for the rest. And **`L5H2` has TWICE `L7H8`'s
effect**. §3.11's "largest of all 16 layer-7 heads" is true and has been read
too broadly: `L7H8` is not the largest OV effect in the circuit, its own
prev-token partner is. Every "the causally load-bearing half" statement needs
that qualifier.

**Caveat CLOSED (2026-09-09, `data/analysis/norm_proportionality.json`), and
it inverts rather than merely fails.** 38 heads sampled across the full `‖OV‖_F`
range (0.4–13.5), each ablated and measured:

- Spearman `‖OV‖_F` vs `|Δ‖resid‖|` = **−0.241** (slightly *negative*)
- Spearman `‖OV‖_F` vs `ΔNLL` = **+0.087** (nothing)
- linear fit **r² = 0.001** — operator norm explains **one tenth of one percent**
  of the variance

| head | `‖OV‖_F` | `Δ‖resid‖` | predicted by norm | **excess** |
|---|---|---|---|---|
| `L11H4` | **13.470** | −0.151 | 0.588 | −0.44 |
| `L16H2` | 8.639 | +0.002 | 0.415 | −0.41 |
| `L11H14` | 8.080 | −0.405 | 0.430 | −0.03 |
| **`L5H2`** | 5.336 | **−5.924** | 0.506 | **+5.42** |
| **`L7H8`** | **4.792** | **−8.382** | 0.520 | **+7.86** |

`L11H4` carries **2.8× `L7H8`'s norm** and has an effect **50× smaller**. Heads
with larger operators do less. So "these two heads are just big" is not merely
unsupported — the relationship runs the wrong way, and the residual-geometry
reading survives on its own.

*Q4 — QK against OV on the same readout.* Ablating the static QK costs +0.616,
**56 % of the OV ablation's +1.107**. §3.11 only ever read attention for the QK
half (0.92 → 0.02); on NLL, attending correctly is worth a bit over half of what
the head is worth in total.

*Q5 — the composed circuit is MORE anti-copying, not less.* §3.12-O's score uses
the raw embedding as the OV's input, but the residual at the attended position
has already been written by `L5H2`. The composed path
`W_U OV(L7H8) OV(L5H2) W_Eᵀ` scores **−0.0997** against the direct
`W_U OV(L7H8) W_Eᵀ` at **−0.0496** — twice as negative. So it is not copying the
prev-token signal either.

*Q6 — the MLPs are negative, which is why they were not run first.* `L7H8`'s OV
into each downstream MLP's input projection, ranked against every head below that
layer: the best is layer 10 at **rank 44 of 159, z +0.57**, layer 9 at rank 47 of
143 (z +0.31), and **every other layer is at or below the population median**
(z −0.16 to −0.73). No elevated MLP pathway anywhere.

**Where this leaves it.** Of the three §3.12-P explanations, (1) MLPs is
**negative** (Q6) and (2) unembedding-without-token-identity is **negative in
the targeted sense** (Q2: the shift is global, not selective). What survives is
**(3), residual-stream geometry** — and it now has a measurement behind it rather
than being the leftover option: `L5H2` and `L7H8` each remove a tenth or more of
the final residual norm, two orders of magnitude more than an ordinary head,
while shifting every logit by ~4 nats. That is the particle account's own claim
arrived at by eliminating the alternatives on their own instruments, and it is
exactly what §3.14.2's case-study programme was queued to examine.

---

**S. `L5H2` x `L7H8`: the induction circuit does not behave like a circuit
(2026-09-09, `p7d_redundancy/two_big_heads.py`, step 16000, restore exact).**
§3.12-P ran the ablation interaction for `L7H8` against downstream *copiers* and
never against **the pair that is supposed to BE the circuit**. Closing that
omission overturns the serial reading.

*S1 — the interaction is larger than either effect.*

| | ΔNLL |
|---|---|
| `L5H2` alone | **+2.2271** |
| `L7H8` alone | +1.1069 |
| sum of parts | +3.3340 |
| **both** | **+7.4844** |
| **interaction** | **+4.1505** |

Removing both costs **2.2x the sum of removing each**, and the interaction
exceeds either individual effect. A serial two-stage circuit predicts
**sub**-additivity — take away the prev-token head and the matcher has less to
match on, so ablating it too should cost *less* than it did alone. The result is
the opposite and it is not marginal. As in §3.12-P the sign is conservative:
§3.12-M5's compressive readout biases independent contributions toward apparent
sub-additivity.

*S2 — and they converge on the same EFFECT without sharing WEIGHTS.*

| overlap | measured | chance (64-dim in R^1024) |
|---|---|---|
| write subspaces (`col W_O`) | **0.222** | 0.250 |
| read subspaces (`row W_V`) | 0.316 | 0.250 |
| **residual-delta cosine** | **+0.868** | — |

Their write subspaces overlap **at or below chance**, yet the residual changes
their ablations produce are **87 % aligned**. So the redundancy in S1 is not two
heads writing the same directions — it is two structurally distinct operators
arriving at the same functional effect through the network. **Weight-space
overlap and function-space overlap come apart**, which is exactly what a
composition score (a weight-space measure) cannot see, and it explains why
§3.12-P's composition probes found no pathway while the ablations shout.

*S3 — the effect compounds down the stack rather than carrying forward.*
Per-layer residual norm under ablation: divergence begins at the head's own
layer (index 6 for `L5H2`, index 8 for `L7H8`) and then **grows**. `L7H8`'s gap
runs −0.36 at index 8 to **−8.38** at the output — a **23x amplification**. A
write that merely added a vector would carry a roughly constant offset forward.
This is the "change of regime rather than a big write" signature §3.12-Q could
not distinguish with a final-layer measurement alone. (Non-monotone detail worth
keeping: ablating **both** *raises* the norm above baseline at index 22, 88.23
against 86.62, before collapsing to 39.53 at the output.)

*Caveat, stated because the joint arm is extreme.* `ΔNLL(both) = +7.48` on a
0.53 baseline puts the model at NLL ≈ 8.0 against a uniform ceiling of
`ln 50304 = 10.8`. Not saturated, but far outside the regime the readout was
calibrated in, so the *magnitude* of the interaction should be read as "large
and super-additive" rather than as a calibrated number.

**What this does to the picture.** The two heads are **functionally redundant
and structurally distinct**, and together they set up a residual-stream regime
that amplifies down the stack and that either one alone can partly maintain.
That is a stronger form of §3.12-Q's geometry reading than the norm evidence
alone supported, and it is the first result in §3.12 that is about the *circuit*
rather than about a head.

---

**T. The redundancy catalogue, pass 1 — the set is not a pair (2026-09-09,
`p7d_redundancy/redundancy_catalog.py`).** Single-head OV ablation `ΔNLL` for **all
384 heads** at step 16000, 8 sequences, restore exact. Answers §3.14.2's Q1.

| | |
|---|---|
| median | **+0.00107** |
| p99 | +0.1935 |
| max / min | +1.9662 / −0.0522 |
| above +0.05 | **10 heads** |
| above +0.2 | **4 heads** |
| above +1.0 | **2 heads** |
| below −0.05 | 1 head |

**Top of the tail:** `L5H2` **+1.966**, `L7H8` **+1.019**, **`L12H5` +0.420**,
`L8H6` +0.212, `L11H14` +0.190, `L8H9` +0.131, `L15H14` +0.087, `L7H1` +0.067,
`L9H13` +0.064, `L10H9` +0.050. Most negative: `L10H7` −0.052.

*Three things this settles or opens.*

1. **Q1 answered: ~4 substantial members, ~10 with any effect, out of 384.** The
   distribution is brutally heavy-tailed — the median head moves the readout by
   **0.001** — so "the redundancy set" is a real, small, identifiable object
   rather than a gradient.
2. **`L12H5` (+0.420) was entirely unknown.** It is the third-largest OV effect
   in the model, four times `L11H14`'s, and nothing in §3.11–§3.12 has ever
   named it. `L8H6` (+0.212) likewise. The 38-head sample of §3.12-R missed both,
   which is precisely why the full sweep was run instead of a proxy.
3. **The members are spread across depth** — layers 5, 7, 8, 8, 11, 12, 15 — not
   clustered in the layer band where the token-identity copiers live (9–20,
   §3.12-O). Membership and copying remain different properties.

*Batch-size note, carried so the numbers are comparable.* This screen used 8
sequences for speed; §3.12-S used 16. `L5H2` reads +1.966 here against +2.227
there and `L7H8` +1.019 against +1.107 — about 10 % lower throughout, with the
ordering unchanged. Pass 2 should fix the sequence count before any pairwise
interaction is compared against §3.12-S's.

*What pass 2 needs, now that membership exists:* the pairwise interaction matrix
over the top members (`n(n-1)/2` arms — 6 for the top 4, 45 for the top 10), and
the per-checkpoint formation curves for each. Both were undefined before this
sweep and are now specified.

---

**U. The formation curves — the members did not form together, and the
redundancy formed after both of them (2026-09-09,
`p7d_redundancy/member_formation_curves.py`, 16 sequences, restore exact at all 23
steps).** OV-ablation `ΔNLL` per checkpoint for the top six catalogue members,
plus the joint `L5H2`+`L7H8` arm and the residual-delta cosine, on the registered
19-step grid with four fills (3000, 5000, 7000, 9000) added to date the
interaction. Answers §3.14.2's Q2 and Q3. `L7H8` reproduces §3.11-A throughout
(+0.242 at 4000 against its +0.24, +0.747 at 8000 against +0.73, +1.107 at 16000
against +1.02 at 8 sequences), and step 16000 reproduces §3.12-S to four decimals
— so the instrument is the same one, extended along the training axis.

*U1 — `L5H2` forms in `(512, 1000]`, alone, and `L7H8` does not exist yet.*

| step | baseline NLL | `L5H2` | `L7H8` | `L12H5` | `L8H6` | `L11H14` | `L8H9` | joint | **interaction** | δ-cosine |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 12.63 | +0.01 | +0.00 | +0.00 | −0.00 | −0.00 | +0.00 | +0.01 | +0.00 | −0.01 |
| 1000 | **4.91** | **+4.97** | −0.01 | −0.01 | +0.05 | **+3.57** | +0.02 | +4.93 | −0.03 | −0.17 |
| 2000 | 1.56 | **+8.43** | −0.01 | +0.86 | +0.44 | +2.44 | +0.27 | +8.44 | +0.02 | +0.02 |
| 3000 | 1.06 | +7.94 | +0.07 | +0.99 | +0.70 | +1.34 | +0.60 | +8.40 | **+0.39** | +0.26 |
| 4000 | 0.72 | +6.22 | +0.24 | +0.94 | +0.61 | +0.68 | +0.40 | +8.18 | **+1.72** | **+0.83** |
| 8000 | 0.67 | +3.68 | +0.75 | +0.66 | +0.28 | +0.38 | +0.28 | +7.74 | +3.32 | +0.85 |
| 16000 | 0.53 | +2.23 | +1.11 | +0.43 | +0.22 | +0.19 | +0.13 | +7.48 | +4.15 | +0.87 |
| 54000 | 0.52 | +1.55 | +1.54 | +0.11 | +0.16 | +0.14 | +0.04 | +6.61 | +3.52 | +0.92 |
| 143000 | 0.61 | +1.26 | +1.22 | +0.11 | +0.15 | +0.17 | −0.01 | +5.91 | +3.43 | +0.87 |

Nothing moves before step 512 — every member reads |ΔNLL| < 0.02 for the first
eleven checkpoints. Then, **in the single interval where the model acquires
induction at all** (second-copy NLL 12.63 → 4.91), `L5H2` goes from +0.01 to
+4.97 and `L11H14` from −0.00 to +3.57. Pythia publishes no checkpoint between
512 and 1000, so `(512, 1000]` is **the finest interval this axis can resolve**:
Q3's "did they form at the earliest point the network could" is **yes for
`L5H2`**, and the §3.12-N3 literature anchor near step 1000 is confirmed for it.

*U2 — and **no** for `L7H8`, which forms in `(2000, 3000]`.* It is the last
member to appear, three to six times later than the rest, and it is the **only
one of the six that rises monotonically** and the only one still near its peak at
143000 (79 % of it, against 15 % for `L5H2`, 11 % for `L12H5`, **5 % for
`L11H14`**). Two-stage circuit stories have the prev-token head feeding the
matcher; the ordering is right, but `L5H2` does not *wait* for a matcher — it
carries induction by itself for over two thousand steps, and `L7H8` arrives into
a mechanism that is already working and already decaying.

*U3 — the redundancy is acquired, and dating the heads would have dated it wrong
by 2000 steps.* The interaction is **≈ 0 while both heads have effects** — −0.03
at 1000, +0.02 at 2000, when `L5H2` is at its maximum — then +0.39 (3000), +1.72
(4000), +4.15 (16000), +4.17 (32000). It tracks `L7H8`'s arrival, not `L5H2`'s.
This is the arm no single-head curve can supply, and it is the direct answer to
§3.14.2's Q2: **the members formed at different times, and the property that
makes them a set formed later than either.**

*U4 — the alignment is `L7H8`'s entry condition, not a converged endpoint.* The
residual-delta cosine that §3.12-S measured at +0.868 does not climb gradually:
−0.17 (1000) → +0.01 (2000) → +0.26 (3000) → **+0.83 (4000)** → +0.91 (7000),
flat thereafter. At step 3000 `L7H8`'s own effect is only **+0.069** — barely
present — and the pair is already 26 % aligned. `L7H8` appears already pointed at
the effect `L5H2` was producing.

*U5 — the pair's total causal load is roughly conserved while its distribution
is not.* The joint arm runs 8.44 (2000) → 8.18 (4000) → 7.48 (16000) → 5.91
(143000), a 30 % decline, while `L5H2`'s share of it falls **85 %** and `L7H8`
rises from nothing to parity (they cross at step 54000: +1.549 against +1.537).
The set redistributes work it does not shed.

*U6 — and the behavioural instrument is blind to all of it, `L5H2` inverted.*
Against `behavioural_series.json` on the same grid: `L5H2`'s induction score
**falls twenty-fold**, 0.0046 → 0.0002, across exactly the interval where its
causal effect goes +0.01 → +4.97, and never recovers. `L7H8`'s peaks at step
4000 (0.0368) and then declines by half while its causal effect keeps rising to
54000. §3.12-R and §3.12-G6 ruled out weights-only predictors of causal effect;
this extends the same failure to the **attention-pattern** proxy along the
developmental axis, and it is why §3.14.2 specified the causal instrument.

*Caveat, and it is a real limit on the magnitudes.* At steps 1000–3000 both the
single-`L5H2` arm and the joint arm sit **within 0.83–1.4 nats of the uniform
ceiling** `ln 50304 = 10.83` (NLL 9.99 and 10.00 at step 2000). The readout is
saturated there, so `L5H2`'s early peaks are **floors rather than calibrated
values**, the measured decay is if anything shallower than the true one, and —
importantly — **the interaction is compressed downward at exactly the steps where
U3 reads it as zero.** Two instruments carry U3's date without that confound:
`L7H8`'s own single-head curve, which is nowhere near ceiling, and the δ-cosine,
which is a geometric measure of the residual and independent of the readout.
Both put the transition in `(2000, 4000]`. A calibrated magnitude for the early
interaction needs a graded readout (§3.12-M's KL / λ scale), not this one.

---

### 3.12-V The matrix, the geometry, and the rank (2026-09-10)

> **Literature status (added 2026-09-16, §3.37).** Detail in
> `p7d_redundancy/lit-7d.md` and `p7e_consolidation/lit-7e.md`. Per result:
>
> | | verdict |
> |---|---|
> | **V1** super-additivity across the set | **NOT NEW.** It is the self-repair signature — *The Hydra Effect*, `2307.15771` |
> | **V1** the second-order interaction *instrument* | **NOT NEW as of July 2026** — *Conditional Co-Ablation (CoAx)*, `2607.01940`. Plausibly still ours: the **variance decomposition** (74–81 % magnitude vs 7–9 % direction), the **full matrix over a causally-established set** rather than a ranking from a seed set, the **checkpoint axis**, and the **ceiling discipline** |
> | **V2** born aligned, then fanning out | **No neighbour found.** Keep |
> | **V3** the ambient stream is ~20-dimensional, which kills the isotropic `k/d` baseline | **No neighbour found, and it is the project's rank-1 survivor.** The adjacent subspace literature appears to use the baseline this measurement voids — check `2601.10266` |
> | **V4** SVD order is a poor proxy for causal importance | **NOT NEW (2022)** — FWSVD, `2207.00112`, states the general form |
> | **V4** `L11H14` is *anti*-ordered (bottom-`r` > matched-random > top-`r`, negative at `r = 1`) | **Looks new.** The compression literature reports *suboptimality*, not inversion. Conditional on a read, and on three differences surviving it: our unit is one head's OV, our readout is causal-ablation recovery, and our baseline is **matched-norm random** — which that literature has no reason to run |
> | **V5** the changing-membership artifact | **No neighbour found.** Transferable |
>
> **Do not write V1's super-additivity or V4's SVD result up as findings.** Write
> V4 as a *strengthening* of FWSVD, with FWSVD cited.

Three runs closing 7d's two open axes and opening 7e. All restore checks exact
(`0.0e+00`). Producers: `p7d_redundancy/pairwise_interaction_matrix.py`,
`p7d_redundancy/member_subspace_geometry.py`, `p7e_consolidation/useful_rank.py`.
Outputs: `pairwise_interaction_matrix.json`, `member_subspace_geometry{,_weighted}.json`,
`ambient_budget.json`, `useful_rank{,_bottom}.json` — all git-ignored, all
regenerable from the commands in `p7d_redundancy/status-7d.md`.

*V1 — one set, and independence is dead.* Top 10, 45 cells, 16 sequences, steps
16000 and 143000, **no cell within 2 nats of the ceiling**. `L5H2`×`L7H8`
reproduces §3.12-S at **+4.1505** vs +4.151.

| | 16000 | 143000 |
|---|---|---|
| positive cells | **44/45** | 38/45 |
| super-additive (> +0.02) | 38 | 28 |
| r²(interaction, `d_a·d_b`) | 0.739 | 0.807 |
| r²(interaction, δ-cosine) | 0.094 | 0.073 |

No block structure at either step — **one** redundancy set. But **74–81 % of the
interaction is magnitude**, and δ-cosine explains 7–9 %: direction and
substitutability are **decoupled**, §3.12-S's single-pair dissociation now at
45 pairs. A tails-only "opposite directions are more redundant" pattern **does
not survive the full sample** (ρ = −0.167, p = 0.27) and must not be quoted.

*V2 — born aligned, then fanning out, with a **measured** null.* Six members
plus three near-median control heads (`L5H5`, `L8H3`, `L12H15`), 13 checkpoints.
Fixed 15-pair set, from step 4000 (first checkpoint with all six formed):

| step | mean δ-cos | null | min pair | max pair | mean ‖δ‖ |
|---|---|---|---|---|---|
| 4000 | +0.722 | −0.196 | +0.322 | +0.967 | 670.6 |
| 5000 | **+0.744** | −0.170 | +0.358 | +0.964 | 643.2 |
| 16000 | +0.603 | +0.004 | +0.131 | +0.882 | 657.1 |
| 54000 | +0.432 | −0.076 | −0.155 | +0.917 | 820.8 |
| 143000 | **+0.327** | −0.044 | **−0.189** | +0.874 | **1093.1** |

Alignment peaks at **step 5000** and loses 56 % by 143000 **while delta norms
grow 671 → 1093** — not a fading-signal artifact. The **max pair stays pinned at
0.87–0.97 while the min falls to −0.19**: a locked core with heads peeling off,
not a uniform drift. At step 1000 the only extant pair is already at δ-cos 0.857
and **centered CKA 0.693 against a null of 0.128** — **aligned at birth, no
private-subspace phase**. **No member's subspace expands** (PR stays 8–28).
And **redundancy survives the separation**: `L5H2`×`L11H14` holds interaction
+2.18 → +1.53 while δ-cos goes 0.344 → **0.004**.

*V3 — the ambient stream is ~20-dimensional, and it broke two measures.* The
baseline residual's own participation ratio is **22.0 of 1024** at step 16000 and
**8.4 at 143000** (on this probe — copied positions of repeated *random-token*
sequences; natural text would be far higher). Consequences: `coverage_by_others`
**saturates to 1.000 and is unusable** (pooled bases span 742–819 dims), and the
isotropic `k/d_model` chance value is not a baseline. **Quote `cka` /
`cka_centered`, never the unweighted `subspace_*_cos` alone.**

*V4 — useful rank splits the set, and `L11H14` inverts.* `r*` = smallest OV rank
recovering 90 % of a head's own causal effect, of `D_HEAD` = 64, against
matched-norm random controls:

| member | effect | `r*` | control `r*` | top-1 | bottom-48 |
|---|---|---|---|---|---|
| `L7H8` | +1.107 | **1** | 48 | **+0.971** | +0.095 |
| `L12H5` | +0.431 | **1** | 48 | — | — |
| `L8H9` | +0.127 | **2** | 48 | — | — |
| `L5H2` | +2.227 | **12** | 24 | +0.123 | +0.616 |
| `L8H6` | +0.219 | **24** | 48 | — | — |
| `L11H14` | +0.187 | **64** | *never* | **−0.096** | **+0.846** |

**`L7H8` recovers 97 % of its causal effect from one direction.** The five
low-rank members sum to **40, inside one head's 64-dim budget** — 7e's
consolidation of the aligned core is viable, and `ambient_budget.py`'s energy
criterion (only 73–78 % of the joint effect inside 64 ambient dims, needing
**355** for 90 %) was the wrong currency.

**`L11H14` is anti-ordered**: bottom-`r` > matched-random > top-`r` **at every
rank**, and at `r = 1` recovery is **negative** — the Eckart-Young optimal
truncation is worse than deleting the head. §3.12-R and §3.12-G6 reproduced
*inside one head's spectrum*. **Two classes, not a gradient.**

**Methodological hold, reaching outside 7e: any SVD-ordered rank truncation
misleads on `L11H14`-like heads, including `induction_rank_sweep`'s entire `r*`
construction.** Its `schur` basis orders by eigenvalue and carries a sign so it
may be immune; the comparison is weights-only and free. **Until it is run, do
not quote an `svd`-basis `r*` for a head not checked with `--bottom`.**

*V5 — the artifact that caught this session three times.* **Every set-level mean
whose membership changes as heads form manufactures a rising trend.** It bit
`union_ratio` (0.94 → 0.16, almost entirely arithmetic), the set-level mean
cosine ("peaks at step 1000" — an average of *one* pair), and centered CKA
("0.197 → 0.637", where 0.197 was one real pair averaged with fourteen
non-existent ones — a spurious three-phase "convergence" was written up and
withdrawn on this). **Read trajectories over a fixed pair set, or print `n`.**
§3.13's report-both rule does not protect against this on its own.

---

## 3.14 How this work is organised, and what is queued (2026-09-09)

### 3.14.1 Three objects, not one thread

§§3.11–3.12 have been running as a single "induction programme", and they are
not one. Three research objects are tangled in them, with **different nulls,
different exchangeable units, and different failure modes**, and separating them
is the largest structural improvement available:

| | object | exchangeable unit | state |
|---|---|---|---|
| **7a** | population spectral development — the OV repulsive collapse (§3.12-A), its relation to `CLAIM-B`'s window, the Phase-2 `frac_repulsive` comparison | head, with a shared-model factor | biggest signal, hardest null |
| **7b** | circuit mechanism at a fixed step — rank sweeps, S/A interventions, composition, alignment | the perturbation draw, so `n = 1` is sound | most of §3.12; closest to registrable |
| **7c** | circuit formation — the backward search from a behavioural anchor, the pythia-70m study (§3.9) | checkpoint / window — where co-location circularity lives | scoped, not started |

Splitting them says immediately what can be registered when, and stops 7a's hard
null blocking 7b's clean one. Note that the labels are for *this document's*
organisation; they are not new phase directories, and §3's numbering is
unchanged (`INDEX.md`'s rule: do not rename directories).

### 3.14.2 `7d` — the redundancy catalogue. **PROMOTED TO ACTIVE 2026-09-09**

Queued earlier the same day as a case-study programme following a circuit's
three stages. **Two results overturned that framing within hours and the
programme is better for it**, so the premise is restated rather than inherited:

- **§3.12-O/P killed the three-stage reading.** `L7H8` is not a token-identity
  copier, the real copiers sit in layers 9–20, and the mediation test found **no
  sub-additivity anywhere** — `L7H8` does not route through them.
- **§3.12-S killed the serial reading of the pair itself.** `L5H2` × `L7H8`
  interact at **+4.151** (joint 2.2× the parts-sum), and they converge on an
  **87 %-aligned residual effect from chance-level weight overlap**.

So the object is **not a chain of stages**. It is a **set of functionally
redundant, structurally distinct heads** that jointly hold a residual-stream
regime, and the programme's unit is that set and its classes.

#### The five questions, as asked

1. **How many members are there?** `L5H2` and `L7H8` may be a pair or the two
   visible members of a set. §3.12-R's 38-head sample cannot say.
2. **Did they form at the same time?** Or at different checkpoints?
3. **Did they form at the earliest point the network *could*?** The literature
   puts induction-head emergence near step 1000 in Pythia (§3.12-N3), so
   "as early as possible" is a checkable claim, not a figure of speech.
4. **Did they form the same way — same structure, same job?** `L5H2` is
   spectrally mixed (attractive fraction 0.444, the only one of four), the least
   gain-concentrated, and its QK symmetry **never leaves baseline**; `L7H8` is
   100 % repulsive with QK symmetry reaching 0.956. Two very different operators
   producing an 87 %-aligned effect.
5. **Are they one class or several?** With a catalogue in hand, do members
   cluster by structure, by formation time, or not at all?

#### What each question needs

**Q1 — membership.** `p7d_redundancy/redundancy_catalog.py`, launched 2026-09-09:
single-head OV ablation `ΔNLL` for **all 384 heads** at step 16000. A full sweep
rather than a proxy screen **because no weights-only quantity predicts causal
effect** — §3.12-R ruled out `‖OV‖_F` (r² = 0.001, relation inverted), §3.12-G6
ruled out every spectral field, and §3.12-S showed weight- and function-space
overlap come apart. A proxy would inherit exactly that failure.

**Q2/Q3 — timing. ANSWERED 2026-09-09, §3.12-U** (`member_formation_curves.py`,
top six members + the joint arm + the δ-cosine, 23 checkpoints, 16 sequences).
**Q2: no — they formed at different times.** `L5H2` and `L11H14` in `(512, 1000]`,
`L12H5` and `L8H9` in `(1000, 2000]`, `L7H8` last in `(2000, 3000]`. **Q3: yes
for `L5H2`, no for `L7H8`.** `L5H2` appears in the same interval the model
acquires induction, which is the finest interval Pythia's grid resolves; `L7H8`
is three to six times later. And the interaction — the arm no single-head curve
can supply — is **≈ 0 until `L7H8` arrives**, so **the redundancy postdates both
heads** and dating it by dating them would have been wrong by 2000 steps. Four
of the six members then **decay** to 5–22 % of their peak while `L7H8` alone
rises monotonically. Magnitudes at steps 1000–3000 are ceiling-limited; see
§3.12-U's caveat.

**Q4 — structure.** Largely **already on disk** and unread along this axis:
`qk_symmetry_sweep.json` (384 heads × 19 steps), `ov_per_head_series.json`
(spectral fields, same grid), `copying_score_sweep.json` (8 steps),
`behavioural_series.json`. What is missing is the *causal* side per member, and
the read-vs-write subspace geometry §3.12-S introduced.

**Q5 — classes.** Only answerable after Q1. Cluster members on (formation step,
spectral signature, QK symmetry trajectory, copying score, causal magnitude) and
ask whether the structure is discrete or continuous — reporting **both** a
central-tendency and an extremum view, per §3.13.

#### Constraints this programme inherits, and must not lose

- **Weight-space overlap is not function-space overlap** (§3.12-S). Every
  composition score in §3.12 is a weight-space measure and is blind to the
  redundancy the ablations show. Catalogue membership must be defined causally.
- **§3.13's rule**: report the mean view *and* the extremum view, and never
  choose between them after seeing the data. §3.12-J found the right instrument
  changes with *training stage*, so this applies per checkpoint too.
- **The spent-artifact rule** (`check_registry` rule 3). Everything 7d touches on
  410m is exploratory and **cannot later be registered and adjudicated on the
  same data** — which is exactly why `P-I7` was registered against an unmeasured
  model. Any 7d claim intended for the registry needs its own unseen test site.
- **The particle-dynamics half needs `dual_reading`'s pairwise field**, which
  `P-I5` is blocked on (§3.12-C). Until it exists, 7d can characterise structure
  and timing but not inter-particle geometry.

#### Where the cross-case question lives

The original framing's best part survives: once several circuits are catalogued
— other 410m members, and pythia-70m under §3.9's grid — ask whether their
**dynamics correlate across cases**. A shared signature over independent
circuits is a population claim no single circuit can make, and it is the natural
home for anything §3.12 produced that wants to generalise.

### 3.14.4 The three questions `7d` is actually for (2026-09-10)

Stated by the user when `7d` got its directory: *what are all these induction
heads doing, is there any relationship between them, and why are they all so
independent but seem to form at the same time?* Each is answerable. One rests on
a premise the data does not support, and saying so is the most useful thing this
section does.

#### A. "Why do they all seem to form at the same time?" — **they do, and that is the finding**

§3.12-U reads as *"the members formed at different times"*, and at the
resolution of the checkpoint grid that is correct. **At the scale of training it
is the wrong emphasis, and the user's reading is the better one.** Five of the
six members cross into existence inside `(512, 2000]`:

| window | members |
|---|---|
| `(512, 1000]` | `L5H2`, `L11H14`, `L8H6` |
| `(1000, 2000]` | `L12H5`, `L8H9` |
| `(2000, 3000]` | `L7H8` |

That is **one doubling of training on a 143,000-step axis** — the last 140,000
steps recruit nobody. And the window is not arbitrary: it is exactly where the
model acquires induction at all, second-copy NLL **12.63 → 4.91 → 1.56**. So
both statements are true and they should be stated together: *ordered at the
grid's resolution, simultaneous at the scale of training.*

**The real question is therefore whether formation time has a common cause.**
Two hypotheses, and they are distinguishable:

- **Recruitment.** The induction phase transition is a single event that makes
  many heads useful at once, and formation order is incidental — noise on a
  shared trigger.
- **Cascade.** One head forms first and its presence is what makes the others
  useful, so the order is causal and should be reproducible.

**The measurement that separates them** is not on the 410m axis at all, because
`n = 1` model gives one draw of the ordering. It needs **a second model** —
pythia-70m under §3.9's grid, which §5.3 records as not yet on disk. If the
order `L5H2 → L12H5 → L7H8` (or its analogue) reappears in a differently-seeded
model, that is cascade; if the *window* reappears but the order scrambles, that
is recruitment. **This is also the one 7d question that could carry a
registered prediction**, precisely because 70m is an unmeasured site and the
spent-artifact rule does not bite there.

> **Promoted 2026-09-16 (§3.37).** `p8_scale_ladder/lit-8.md` §2 checked all six
> of Phase 8's invariants against the field. **This is the only one with no
> neighbour found.** The developmental literature reports whole-model head-class
> fractions and emergence curves (`2606.02378`); **nobody reports the order in
> which individual members of a redundancy set form across independently-seeded
> models.** Invariant 2 (the window) is the opposite case — occupied, and
> `2511.16893` supplies an *equation* predicting the formation point from batch
> size and context size, which Pythia holds constant across the whole suite.
> **Promote ordering to the ladder's headline and demote the window to a
> calibration check.** `2602.16740`'s mid-depth-instability result predicts this
> is also the hardest one to reproduce — our members sit at layers 5–12 of 24.

#### B. "Is there any relationship between them?" — **unmeasured, and the premise of independence is not supported**

The heads are **not** known to be independent. Exactly one pair has ever had its
interaction measured, and it is **strongly redundant**: `L5H2` × `L7H8` at
**+4.151**, joint 2.2× the parts-sum (§3.12-S). Every other pair among the ~10
members is simply **unmeasured** — 44 of the 45 cells of the top-10 matrix are
empty. "They are all so independent" is an impression from reading the
single-head column of §3.12-T, which by construction says nothing about pairs.

There is already evidence pointing the other way, and it is in §3.12-U's own
table. Compare the **sum of the six single-head effects** against the **joint
pair arm**:

| step | Σ singles | joint(`L5H2`,`L7H8`) | ratio joint : Σ-of-that-pair |
|---|---|---|---|
| 2000 | **+12.44** | +8.44 | 1.00 |
| 4000 | +9.09 | +8.18 | 1.27 |
| 16000 | +4.30 | +7.48 | 2.24 |
| 143000 | **+2.90** | +5.91 | **2.38** |

**The sum of what each head is individually worth collapses four-fold, while
removing the pair together still costs most of what it ever did.** That is the
signature of a set becoming *more* interchangeable over training, not less — and
it is the opposite of independence. Pass 2 (next action 1) fills the matrix and
settles it.

#### C. "What are all these induction heads doing?" — **partly on disk, and the members are not copiers**

Two things are already known and should not be re-derived. The members are
**spread across layers 5, 7, 8, 8, 11, 12, 15**, *not* clustered in the 9–20
band where the token-identity copiers live (§3.12-O2) — membership and copying
are different properties. And the pair that has been characterised is
**structurally opposite**: `L5H2` is spectrally mixed (attractive fraction
0.444), least gain-concentrated, QK symmetry never leaves baseline; `L7H8` is
100 % repulsive with QK symmetry reaching 0.956. Two very different operators
producing an **87 %-aligned** effect.

What is missing is the same characterisation for `L12H5`, `L8H6`, `L11H14` and
`L8H9`, and it is **mostly on disk already**: `qk_symmetry_sweep.json` (384
heads × 19 steps), `ov_per_head_series.json`, `copying_score_sweep.json`,
`behavioural_series.json`. The catalogue says which rows to read and §3.12-U says
which **steps** matter — the action is in `(512, 4000]`, not at the endpoints,
which is where every one of those files has previously been read. This is next
action 3, it costs no forward passes, and it is the input Q5 (classes) needs.

#### D. The one thing that will waste a day if forgotten

**The readout has a ceiling and the interesting steps are against it.**
`ΔNLL` on the second copy is bounded by uniform prediction, `ln 50304 = 10.83`.
At steps 1000–3000 both the single-`L5H2` arm and the joint arm sit within
**0.83–1.4 nats** of it, and the `L5H2`×`L11H14` pilot lands *past* it at 11.77.

Three consequences, all of which have already nearly bitten:

1. Early magnitudes are **floors**, not measurements.
2. Interactions are **compressed toward zero** there — so §3.12-U's "interaction
   ≈ 0 before `L7H8` arrives" is safe as a *date* only because two
   ceiling-immune instruments agree with it (`L7H8`'s own curve, and the
   δ-cosine, which is geometric).
3. The compression biases interactions toward apparent **sub**-additivity
   (§3.12-M5) — which is the serial-circuit signature. **A serial-looking result
   measured near the ceiling is an artifact until proven otherwise.**

So: run the pairwise matrix at **step 16000 or later**, where there is 2.8+ nats
of headroom, and treat any early-checkpoint interaction as needing the graded
readout (§3.12-M's KL / λ scale) first.

### 3.14.3 Defects found this session, not yet fixed

All three are **reporting or documentation** defects. None touches a p-value,
`claims/registry.json` is unchanged and `claims/adjudications/` is empty, so
nothing registered is affected — but each is live in code a reader would trust.

1. **`induction_rank_sweep.truncate`'s `random` branch docstring is false**
   (§3.12-L). It claims *"Matched-norm random rank-r control … the operator norm
   scale and the factor structure match the real truncation."* At `r = 1` the
   energy ratio to the `svd` arm is **16×**. Either the docstring changes or the
   branch rescales; the §3.11 Stage-1 table must be read against §3.12-L's
   energy-matched version either way.
2. **`target_vs_reference` computes a self-inclusive z** (§3.13.1), which is
   capped at `(n−1)/√n = 3.75` for `n = 16`. `L7H8` at step 8000 reads **3.74** —
   saturated, against a leave-one-out value of **50.2**. Fix is leave-one-out.
3. **`ov_factors` returns `OV_h = (W_O W_V)ᵀ`**, the transpose of the
   residual-stream operator (§3.12-O, verified at relative error 0.0). Harmless
   for every quantity read off it so far — all transpose-invariant — and the
   docstring does state the convention, but nothing warns that a **directional**
   read (any circuit with `W_E` on one side and `W_U` on the other) must not use
   it. A one-line warning would have saved a careful check.

---

## 3.15 The ablation mode is not a detail (2026-09-11)

Phase 8's first rung ran on pythia-70m; the per-invariant numbers are in
`p8_scale_ladder/status-8.md` and are not repeated here. **One result from it is
not phase-8-local and changes how every ablation number in this file should be
read.**

**Zero-ablation's off-distribution bias scales as `1/n_heads`.** Setting a head's
output to zero does not only remove its function, it puts the residual stream
somewhere the downstream layers never saw. One head is **1/16** of a pythia-410m
layer and **1/8** of a pythia-70m one, so that component does not cancel in a
cross-rung comparison. Measured, all heads, step 16000, 8 sequences:

| rung | mode | participation ratio of \|dNLL\| | Gini | top-5 identity |
|---|---|---|---|---|
| 70m | zero-ish (`ov`) | 5.05 | 0.677 | `L2H1`, `L0H6`, `L0H5`, `L0H0`, `L1H4` |
| 70m | `mean` | **2.12** | 0.779 | `L2H1`, `L0H3`, `L3H1`, `L3H6`, `L3H5` |
| 410m | zero-ish (`ov`) | 1.71 | 0.828 | `L5H2`, `L7H8`, `L12H5`, `L8H6`, `L11H14` |
| 410m | `mean` | 1.47 | 0.859 | unchanged |

**410m is mode-invariant and 70m is not.** Confirmed on the invariant runners
too, 2026-09-11: at 410m, `useful_rank`'s `r*` under `mean` reproduces the
canonical `ov` values for five of six members, ambient top-64 energy reads 0.729
against 0.732, and the `L5H2`x`L7H8` interaction ratio reads 2.22 against
§3.12-S's 2.2x. The 410m body of work is not an artifact of its ablation mode.

At the catalogue level the same asymmetry: at 70m the top-5 changes
identity, layer-0 heads dropping out and the real induction cascade coming in.
`L0H6` goes −3.34 → −0.22, `L0H0` +2.70 → **+0.27**. So 410m's existing results
are safe on this axis, and any *small* model's are not. **`pythia-1b` is also 8
heads/layer**, so a prediction registered against that reserved rung should name
mean-ablation rather than spend the rung on the distorting instrument.

**`write_ov`'s zero path is a bias-ablation, not a zero-ablation.** It zeroes
`W_V`'s weight and leaves its bias, so the head keeps writing a constant —
measured 0.1007 at every position at 70m `L3H6` against an unablated 2.1–3.2.
Every 7d/7e/8 number to date is one. It is **immaterial**: true zero-ablation
through an activation hook reproduces the `ov` NLL *bitwise*. Recorded because
the docstrings say "the head's OV removed entirely" and that is not what happens.

**The probe's token distribution is a third, and it is not an instrument limit
but a scope limit on the readout.** `p8_scale_ladder/probe_distribution.py`:
uniform-random ids from `[1000, 40000)` are so far out of distribution that
first-copy true-token median rank is ~23000 of 50304 at **every** rung, so the
second-copy number inherits a baseline that is already worse than uniform.
pythia-70m at step 143000 reads second-copy NLL **14.25**, past `ln 50304`, and
looks like an induction collapse — but on **repeated natural text the same
checkpoint copies at 99.2 % top-1**, identical to its own step-16000 number and
to 410m's, and its natural-text NLL is 3.68. Induction is intact; the language
prior has won on the OOD arm, and it wins harder as training strengthens the
prior (`wide` top-1 0.366 → 0.242 from step 16000 → 143000, `text` unchanged).
**410m is nearly arm-independent** (0.954 `wide` / 0.989 `text`): it has the
capacity for both, and 70m must trade off. Consequences: anything read on this
probe at a *small* model's *late* checkpoints measures prior-versus-probe
conflict as well as induction; `freq` ids from `[1000, 5000)` keep the induction
dynamic range (ICL median gap 10.41 against `wide`'s 3.76) while cutting
above-ceiling positions 42 % → 8 %, and are strictly the better probe — as an
ADDED arm, since every existing 7d/7e/8 number is on `wide`. And §3.13 bites
hardest here yet: mean 8.097 against median **1.428** on the same `freq` rows.

**The hedge was tested against the finding it threatened, and the finding
survives (2026-09-11).** `--probe {wide,freq}` now selects the token range
(`PROBE_ARMS` in `induction_rank_sweep.py`; `wide` stays the default, so no
existing number changes meaning). 70m's end-of-training column was re-measured
on `freq`, where step 143000 is **not** degenerate — baseline NLL 8.13, below
the ceiling, against `wide`'s 14.25. Prediction stated before running: if
"70m never separates" were a probe artifact, the late cosine should fall
toward 410m's decoherence.

| step | 1000 | 2000 | 16000 | 143000 |
|---|---|---|---|---|
| 70m `mean`, `wide` | 0.934 | 0.928 | 0.709 | **0.685** |
| 70m `mean`, `freq` | 0.936 | 0.951 | 0.559 | **0.750** |
| 410m `mean`, `wide` | 0.888 | 0.887 | 0.340 | **−0.009** |

It does not: on `freq` the endpoint is if anything *higher* (0.750 vs.
`wide`'s 0.685). **Invariant 4's second half genuinely fails to replicate**:
410m decoheres to orthogonality and 70m does not, and that is about the
rungs, not the instrument. Step 16000 moves more (0.709 → 0.559) than the
endpoint does, so there is real probe sensitivity mid-trajectory — none of it
in the direction that would explain the result away. The scope limit above
still stands for anything else read on `wide` at a small model's late
checkpoints; it just does not bite this claim.

**Two instrument limits found in the same pass.** (1) `useful_rank`'s `r = 64`
float32 refactorisation residue is harmless while `|d0|` is large and fatal once
it is not — at 70m `L0H2` the residue is **56 % of that head's own
mean-ablation `d0`**, so `recovery` never approaches 1 and `r*` is meaningless.
`useful_rank` needs `|d0| >>` that residue, and mean-ablation shrinks `d0`, so
the two requirements pull against each other. (2) **No absolute threshold
transfers between rungs** — `design-8.md` says so and the first 70m write-up
imported the `+0.05` bar anyway, producing "40 % of heads matter at 70m vs
1–3 % at 410m". Against each rung's own bulk the same counts are **14.6–16.7 %
at 70m and 15.1–15.6 % at 410m**, i.e. indistinguishable.
`p8_scale_ladder/compare_rungs.py` holds the threshold-free statistics
(participation ratio, Gini, top-k share) and is what cross-rung claims should
be read from.

---

## 3.16 The verified literature scan, and the phase-8 reframe (2026-09-12)

Full record in **`p8_scale_ladder/literature-8.md`**; this is the part that is
not phase-8-local. Run under `CLAUDE.md`'s trigger 1 — a phase open, its
invariants not yet registered, so the last moment the literature could still
change what gets built. **Every id was fetched**, which is what
`docs/literature_scan_2026-09-10.md` could not claim and is now marked
superseded for.

**The phase's headline question is already answered.** `2407.10827`, *LLM
Circuit Analyses Are Consistent Across Training and Scale* (2024), covers
**70M–2.8B over 300B tokens** — our ladder's span — and reports that components
"may be implemented by different attention heads over time, [but] the
overarching algorithm that they implement remains". So "do induction signatures
recur across scale" is settled. **What it does not do is account for the
substitution structurally** — no rank, no subspace geometry, no alignment
trajectory — and **invariant 4 is exactly that account**. The phase reframes
from *whether* signatures recur to **what the substitution looks like
geometrically**, which inherits that paper as support rather than competition.

**Scored against the literature: invariants 1, 2 and the super-additive half of
6 are replication; 3 is half-anticipated; 4, 5 and 6's magnitude/direction split
are the real content.** This inverts `design-8.md`'s sequencing, which opened
with invariant 1 because it is cheapest.

**Novelty and replication are orthogonal, and invariant 5 is where they
disagree.** The literature scan scores novelty; `status-8.md` scores
replication. Invariant 4 is **unclaimed and replicates** — the load-bearing
card, and where the next rung's effort belongs. Invariant 5 is **unclaimed and
fails to replicate**: 70m has no low-rank majority (three of four above-noise
heads need the full 64, against 410m's five of six at `r* ≤ 12` summing to 48),
and that non-replication survived every check. So it is a property of
**pythia-410m, not of induction** — a legitimate negative, and exactly the
discrimination the ladder exists to make, but it must never be written as a
cross-scale property. **Conflating the two axes is the trap this section
exists to prevent**, and a first draft of it fell in.

**Two corrections that matter beyond phase 8.**

1. **A search summary is not a source, and it cut the wrong way.** The
   2026-09-10 scan concluded invariant 5's anti-ordering was covered by an SVD
   compression chain. Fetched, it is not: AdaSVD is error-compensation, QSVD is
   VLM QKV compression, and **CARE is GQA→MLA conversion with nothing to do
   with truncation ordering.** Only FWSVD's *suboptimality* stands against our
   measured *anti-optimality*. **An unverified scan talked us out of the
   finding with the least cover** — the failure mode is not just citing a
   hallucination, it is dropping real work on a bad reading.
2. **Nothing was confabulated.** All nine ids resolved. Recorded as plainly as
   a hit would have been, because the opposite expectation is what motivated
   the check.

**The methodological cards, re-ranked.** The strongest is **measured rather
than isotropic nulls**, and it is a live correction to a published method:
`2601.10266`'s projection kernel scores affinity against a **random orthogonal
subspace** baseline, with no mention of anisotropy in its full text — exactly
the baseline §3.12-V3's ambient participation ratio of **22 of 1024**
invalidates. *Check that the anisotropy transfers to their `d = 768` setting
before writing it as a critique.* Conversely **the ablation-mode phenomenon is
no longer ours**: `2604.14433` is a whole paper on zero-ablation's
distributional bias (in ViTs), so §3.15's residue narrows to the `1/n_heads`
scaling and its non-cancellation across rungs.

**The single most valuable thing the scan produced is an experiment.**
`2502.14010` (Yin & Steinhardt, ICML 2025) reports induction heads *becoming*
function-vector heads, induction score declining as FV score rises. §3.12-U
measured `L5H2`'s induction score falling twenty-fold **while its causal effect
went +0.01 → +4.97** — plausibly that same transition seen from the causal
side, and currently filed as a puzzle. **Running an FV score on our members is
cheap and would resolve it.**

> **RUN 2026-09-12 — see §3.18. The hypothesis is refuted for `L5H2`, and the
> experiment produced a better result than it was designed for: the set
> divides the two roles between different members rather than transitioning
> between them.** §3.12-U's puzzle stands as a puzzle.
>
> **UPDATE 2026-09-13 — see §3.20. Closed on the mechanism**: `L5H2` is a
> previous-token head whose ablation demonstrably breaks `L7H8`'s own
> matching attention; it has neither score because neither is its job. The
> joint-ablation super-additivity (§3.12-S) is a separate, still-open
> question.

---

## 3.54 A scoped thread, with its own handoff: what clusters are made of (2026-09-20)

**`p10_cluster_function/handoff-10.md`** is the ordered plan and
**`questions-10.md`** the hypotheses. Opened out of the §3.52 read, on the
user's question: *how does the algorithmic drive toward cluster formation
interact with the structure SGD imposes, and can that drive be used for
forgetting instead of SGD?* **Eight stages, general → particular. Stage 0 is
compute and is first (§3.54.2b); Stages 1–5 cost no forward pass**, and every
one of them re-runs on the enlarged battery for free once Stage 0 lands. Everything in it is exploratory and unregistered.

This section carries only the two things that are project-wide.

### 3.54.1 A fourth casualty of the HDBSCAN outage, and it is the semantic instrument

`status-10.md` §2 records that `hdbscan_labels.json` was empty in 152/152
directories and names F0, F5 and F11-A4 as the rows that blocked on it. **There
is a fourth, and it is worse, because it did not block — it degraded.**

`pair_agreement` (`pair_hdbscan_agreement`, `p1_mstate_tracking/clustering.py`)
tags mutual nearest-neighbour pairs against the **embedding Gram** as an
external semantic axis and reports `ext_semantic_fraction` and
`ext_sem_same_cluster_frac` per layer. **It is this project's only semantic
instrument.** It runs only when `"labels" in hdb_data`; through the outage that
branch was never taken, and the `else` wrote a **well-formed record of zeros and
nulls** into all 152 directories.

> **A silent zero record looks exactly like a real one.** That is why it
> survived the outage, the backfill, the audit and two literature passes. Same
> bug class as `docs/AXES.md` listing an absent artifact as present, and the
> direct instance of standing rule 4's *"refuse rather than degrade."* The
> backfill deliberately does not re-run the analysis (`status-10.md` §2 item 3),
> so **the WDS sweep still carries no semantic record.**

**And the pilot sweep's copy is populated — 6 066 of 6 075 layer-records — and
has never been reported in any markdown file here.** It has been on `HDD_1TB`
since 2026-08-12.

### 3.54.2 First read of it, and it is the first direct evidence on the semantic question

**Tier 1, exploratory, no null, not registered.** Mean over 8 prompts × 25
layers per checkpoint, pilot sweep, native labels (`handoff-10.md` §1.1 has the
table).

- **`ext_sem_same_cluster_frac` is flat** across all 27 checkpoints, 0.62–0.73.
  *Given* that a pair is lexically similar, it co-clusters about two thirds of
  the time, and **training does not change that.**
- **`ext_semantic_fraction` falls, 0.833 → 0.708**, with the drop concentrated
  between **step 512 and step 3000**.

> **The tentative reading: training does not change how clusters treat
> lexically-similar tokens — it changes which tokens are neighbours at all.**
> At initialisation, residual-stream neighbourhoods are largely inherited from
> token identity; by step143000 ~29 % of mutual-NN pairs are not lexically
> similar. Neighbourhoods become **contextual rather than lexical.**

**That window is crowded and the co-location must not be asserted as one event.**
A0's learned attention residual appears at step ~2000–4000; §3.51 puts the energy
break at 256→512 and the Fiedler zero-crossing at 1000→3000; F1/F12's parked
window is 32–512. Whether these are one transition or four is exactly what
`handoff-10.md` Stage 1 exists to answer, and co-occurrence does not settle it.

**Caveats, all load-bearing.** `ext_semantic` is `emb_gram[i,j] > 0.5` — an
arbitrary cosine threshold, and the decline could be a norm/scale effect rather
than a structural change, so **the threshold sweep gates the reading.** Mutual-NN
pairs are ~90 per layer: a small, special subpopulation. The layer axis is
collapsed. No position control. No null.

### 3.54.2b FIRST ACTION: the Phase-1 battery goes from 8 prompts to 20

**Decided 2026-09-20 (user).** `handoff-10.md` **Stage 0**, and it is the only
compute-heavy item before Stage 6.

**Why first.** Every e-value in this project merges over units that are not
independent — layer-units inside one forward pass share a model, a text and a
prompt. The `average` merger is valid under arbitrary dependence and
correspondingly low-powered, which is why §3.41 records `CLAIM-C` unable to
express a p below 0.0661. **The exchangeable unit is the prompt, and there are
eight.** `2501.10573` used 2 244. **Prompt count is the binding constraint on
every result in this phase and the one thing compute can fix.**

**No new prompts are to be invented, and that is the point.** `core/prompts.py`
carries battery **v2 — 21 prompts, hash `06790b90dcfe`** — extended 2026-09-19
under **a rule committed in its own commit ahead of the text** (§3.42). Phase 1's
metastability sweep used **8** of them. **Twelve have never been through Phase
1**, and because they were chosen blind under a written rule, **running them is
not a new selection decision.** That is what makes the enlarged battery able to
carry a **registered** prediction where the current one cannot.

> **Correction to `status-10.md`, which said "13".** It is **12**; the
> thirteenth is `short_heterogeneous` at **115 characters**, almost certainly
> too short to cluster. The 12 are six genres × two prompts, one short-band and
> one long-band per pair, exactly as the committed rule specifies.

**8 → 20 usable prompts — 2.5× the exchangeable unit.**

**Budget, measured on disk 2026-09-20.** Per run directory at ~450 tokens:
`plateau_attentions.npz` **147.6 MB** + `attentions.npz` **147.6 MB** +
`activations.npz` 44.3 MB + ten small files = **325 MB**, of which **91 % is
attention**. 12 × 19 = **228 directories = 74 GB** against **164 GB free**.

> **`plateau_attentions.npz` is a byte-identical relayout of `attentions.npz`,
> verified across all 24 layers** — `(24,16,n,n)` under one key versus 24 arrays
> `attn_L0…attn_L23` of `(16,n,n)`. Same numbers stored twice, **148 MB per
> directory, ≈ 22 GB across the existing 152.** Halving it takes the new-run
> cost to ~40 GB. **Do not delete before checking which readers use which
> layout**, and record the decision before taking it — §5.1's rule.

**Attention scales as n², so more short prompts beat fewer long ones** for this
project's purpose. That runs *against* `2501.10573`'s `N ≥ 500` guidance, because
they wanted per-prompt intrinsic dimension and this project wants independent
units; both are true and they trade off. **F16 is the row that will feel it** —
of the current eight only `homer_iliad` (512 tokens) clears their threshold.

**Compute is unmeasured** — no per-run timing for a Phase-1 410m sweep exists in
the tree. **Time one prompt × one checkpoint and multiply** before launching 228
runs.

**Four checks, none optional** (`handoff-10.md` §0.3): matching
`PROMPT_BATTERY_HASH` via `verify_same_battery`; HDBSCAN present, in the conda
`mets` env, **verified on the first directory** — otherwise both
`hdbscan_labels.json` and `pair_agreement` silently degrade, per §3.54.1; time
one first; never `git add` under `data/`.

### 3.54.3 The three hypotheses worth naming project-wide

All invented 2026-09-20, all conjecture, none literature-checked
(`questions-10.md` §8).

- **H-ANCHOR/BALLAST** — a cluster is one attended *anchor* plus many ignored
  *members*, which reconciles Blog 1's attention flip with the theory's claim
  that centres are attractors. **HDBSCAN gives membership and never centrality**,
  so this has been structurally invisible to every measurement here. F13 supplies
  centrality for free.
- **H-THERMOSTAT** — the attention sink is *how* a trained transformer resists
  collapse: it absorbs mass into the normaliser `Z_k`, lowering effective
  pairwise coupling among the rest. **The model has already learned a metric
  intervention, and the sink is it** — which reframes Phase 9 as doing
  deliberately what training discovered. Free from `attentions.npz`.
- **H-WARP** — training acts on the clustering dynamics as a **monotone
  reparametrisation of the depth clock**, not as a change of structure. Separable
  by curve collapse against `T_eff`, which `math-1.md` §15 item 3 has called the
  highest-value unrun quantity at report-only cost since before this phase
  opened. **The confound is the answer.**

### 3.54.4 Two process consequences

1. **Register F14 before looking.** `status-10.md` records the decision that ran
   F0 exploratory and capped it at tier 1. With **39 registrations and zero
   adjudications**, F14 is the second chance at the first adjudication.
2. **The power ceiling is eight prompts.** `2501.10573` used 2 244. Layer-units
   within one forward pass are not independent samples; the `average` merger is
   valid under arbitrary dependence and correspondingly low-powered. **More
   prompts is the highest-value compute available and it needs no new
   instrument.**

## 3.53 `2411.04990` read in full: five changes, and the carrying-capacity finding gets a formula (2026-09-20, concurrent session)

> **Renumbered on merge, and read with §3.52.** This section was written in a
> concurrent session against PR #58's branch, where §3.51 was still free; PR
> #59 had already taken that number. **§3.52 is a companion read of the same
> paper plus four others, done the same day** — the two agree on everything
> below **except item 3's gradient-flow verdict**, where §3.52.1 and
> `lit-10.md` §11.5 are the later and more complete reading: Lemma 5.3 says
> the causal dynamics **is** a *sequential* gradient flow. Four findings here
> are **not** in §3.52 and are the reason this section is kept whole: Theorem
> 4.1's consequence for `plan-9.md` §4.3, Lemma C.1 as the carrying-capacity
> formula, `d₁ = dim L`, and the RMSNorm absorption.

**`docs/readings/2411.04990.md`** — the reading note, marked **[R]**, from the
PDF supplied by the user. This was the top item in the verification queue and
`lit-10.md` §10's own description of it: *"two phases depend on this one paper
and neither has read it."* New check file
`tools/math_checks/parking_center_count.py` (8/8); **36 checks across five files,
all passing.** Nothing run on real artifacts; nothing registered.

**New convention: `docs/readings/<arxiv-id>.md` for papers read as primary text,
marked `[R]`.** `docs/LITERATURE.md` §0.1 introduced the mark; this is the first
file to earn it. The PDF itself is not committed — the note is the greppable,
diffable artifact and the binary is 1.7 MB.

### 1. Theorem 4.1 — position 0 is a theorem, not a confound

With `V = Id` and **arbitrary `Q, K`**, for almost every initial configuration
the causal dynamics converge to a single cluster and the limit is **`x₁(0)`** —
the first token's initial position. §3 of the paper: *"the first token is
evolving fully autonomously without the influence of others."*

Strictly weaker hypotheses than the unmasked results (which need `QᵀK = V` or
`QᵀK = Id`). **Consequences here:** §3.49's structural attention tilt and
§3.50's `Z` reversal are two faces of the same autonomy; and **no QK-side
intervention, γ included, can prevent collapse under `V = Id`** — stronger than
`plan-9.md` §4.3's hemisphere ceiling, and it needs no hypothesis on the
configuration at all. `plan-9.md` §4.3 is amended.

Table 1 is the conjectured atlas of final configurations, keyed on `λ_max(V)`
and its eigenspace `L`. **`plan-9.md` §4.7's sign prediction is that table**,
which is a conjecture there — so testing it on a trained model is a
contribution, and the atlas says what to expect in five cases rather than two.

### 2. The `d_eff` regression is the paper's own open conjecture

§3.50 derived a log-log slope regression as a rescue from the `d = 1024`
problem. The paper states it, §5 p. 6: *"we conjecture that the number of
meta-stable clusters should rather be `β^((d₁−1)/2)`, where the ambient dimension
`d` is replaced by the **effective dimension** `d₁` … a rigorous proof of this
dimension-reduction remains an **open problem**."*

**So the regression is a direct empirical attack on a named open problem, not a
workaround.** And `d₁` is not free: the paper identifies it as **`dim L`, the top
eigenspace of `V`** — which Phase 2's `sym_*` / `schur_*` projectors already
compute for all 19 checkpoints. **Predict `d₁` from the OV spectrum, then test the
slope against it**, which makes it differential rather than exploratory.

### 3. Lemma C.1 — the carrying-capacity finding, with a formula

> average number of strong Rényi centres = **`1/σ_{d−1}(B_δ) ~ 1/δ^{d−1}`**

proved for **any spherically symmetric measure in any dimension** (ordinary
Rényi centres are much harder above `d = 2`, where the classical `c·2π/δ` with
`c ≈ 0.75` applies — **that is where the 0.7476 constant actually lives**, and it
is not in anything this project measures). With `δ = cβ^{-1/2}` this is exactly
the `Θ(β^((d−1)/2))` frequency, so the paper's two statements are one.

**It is a limit over an infinite sequence, so the count SATURATES in `n`.**

> **That is Phase 1's unexplained finding.** Max simultaneously-alive clusters
> **invariant at 50–55 across all 27 checkpoints** while lifespan falls 7.0 → 4.5
> and births rise 113 → 164 — a fixed capacity with rising turnover, which is
> the shape a saturating parking count predicts. `lit-1.md` grades it *"Looks
> new"*; it has a formula.

**And inverting it bears on β's undecided convention.** Solving
`1/σ_{d_eff−1}(B_δ) = 52.5` and setting `c = δ√β` at the measured medians:
under the **scaled** convention (β = 0.50) no `d_eff` up to 22 reaches Lemma
5.1's required `c > 1`; under the **unscaled** convention (β = 4.0) every
`d_eff ≥ 5` does. **First evidence in this project bearing on the factor-of-8
decision §3.40 flagged as undecided.** Held loosely and the check says so: an
i.i.d. isotropic hypothesis token embeddings do not satisfy, HDBSCAN clusters
are neither centre type by definition, and the count is asymptotic in `n` while
`n ∈ [20, 512]`.

*(Recorded, not resolved: App. C.4 prints the `d = 3` count as
`(3 sin²(δ/2))⁻¹`; the cap area gives `1/sin²(δ/2)`. Constant in `δ`, so it moves
an intercept and not an exponent — but do not quote an absolute `d = 3` count.)*

### 4. A γ-patch is not a read-side lever

§2 of the paper: a trainable RMSNorm diagonal *"can be equivalently achieved by
**multiplying `K, Q, V` matrices by `D`**."* Pythia's `input_layernorm` feeds
`W_Q`, `W_K` **and** `W_V`, so **a γ-patch moves the attention pattern and the
displacement together.** `notes-9.md` §7's three insertion points are **not
separable by γ**, and **`plan-9.md` §4.7's sign-differential test cannot be run
with a γ patch alone** — it needs a `W_V`-side arm. This narrows the lever and
names it: a γ-patch is a **simultaneous QK-congruence and OV-rescale**.

### 5. Timescales, and F0 fully specified

Quasi-stationarity holds for `T_j·s_j < e^{c²/2 − c⁴/(24β)}·ε` (Lemma 5.1);
final collapse is at `t = exp(Ω(√β))`; and *"the time parameter in our dynamics
corresponds to network depth."* Phase 1c's `T_eff` is in the same units, so
**`lit-1.md` §4 item 4 — filed as blocked on reading the paper — is unblocked.**
Note the `s_j`: **a centre's stationary lifetime falls with its own token
index**, testable against the measured lifespan fall 7.0 → 4.5.

**F0's definitions are now exact.** Rényi centres are separated from previous
**centres** (they capture more clustering but move and merge); **strong** Rényi
centres from **all** previous particles (visually stationary but do not explain
all clusters). `δ = cβ^{-1/2}` because attraction is maximal at order `β^{-1/2}`;
the figures use `c = 4` and Lemma 5.1 needs `c > 1`. Arrival order **is token
order**. And the separation *"extends naturally to distances induced by
`⟨Qx, Ky⟩`"* — which is `core/ln_frame.py`'s Gram, **the frame attention actually
reads**. Report **both** centre types; the paper says they behave differently.

### 6. What the paper says it cannot do, and the pointers it opens

Its own §6: Theorem 5.2 gives **no bound on convergence time**, so
quasi-stationarity does not yet prove meta-stable clustering; a complete theory
*"would require demonstrating that each Rényi center captures `Ω(n)` particles in
`O(1)` time"*, and *"even the weaker claim of capturing `ω(1)` particles remains
unproven."* Practical simplifications: **tied weights across layers** (Pythia is
not tied) and **the MLP omitted** — *"Incorporating the MLP dynamics … remains a
significant open challenge"*, which is exactly Phase 10 §7's question.

Five new pointers, one consequential: **Castin, Ablin & Peyré 2024** introduce
*"a clever reparametrization that allows them to recast causal attention as
mean-field dynamics."* **If that restores mean-field structure it may restore the
gradient-flow framing `plan-9.md` §5.1a wrote off** — read before treating that
hazard as final. Also **Cowsik et al. 2024** (a more realistic architecture
*including MLP layers*, with accurate final-configuration predictions),
**`2410.23228`** (Bruno et al.), **Geshkovski et al. 2024a**, and Agrachev &
Letrouit `2404.08289`.

## 3.52 Five papers READ as primary text — and the one the project most depended on changes three constructions (2026-09-20)

**The user supplied PDFs for five of the six papers in
`p10_cluster_function/lit-10.md` §10's verification queue.** Every scholarly
host is still blocked from a session, so this is the first time any of them has
been read rather than searched. The full reads are `lit-10.md` §11–§15 and the
derivations are `math-10.md` §7; this section carries only what is
project-wide. **No measured number anywhere in this repository changes.** Three
constructions do, and one registered-adjacent framing is un-blocked.

| paper | grade before | grade now | what it moved |
|---|---|---|---|
| **2411.04990** *Clustering in Causal Attention Masking* | `[S]` | **`[R]`** | F0, the count law, the gradient-flow hazard, and a prior-art claim |
| **2501.10573** *The Intrinsic Dimension of Prompts…* | `[S]`, wrong title | **`[R]`** | the `d_eff` candidate list; a graded null |
| **2601.02932** *Data-driven Reduction of Transfer Operators…* | `[N]` | **`[R]`** | `plan-9.md` §5.1's architecture, and one thing not to copy |
| **2605.12765** *GUARD-IT* | `[S]` | **`[R]`** | two thirds of Phase 9's novelty differentia |
| **2505.16831** *Unlearning Isn't Deletion* | `[S]` | **`[R]`** | a required arm and a four-diagnostic panel |

Still unread and still able to change a construction: **2303.06562**
(ContraNorm) and **2607.15495** (the J-lens paper itself).

### 3.52.1 `2411.04990`: the parking correspondence, and F0 measured a proxy

**The correspondence, exactly.** The street is `S^{d−1}` with geodesic distance;
**arrival order is token position**, which is why the analogy exists only under
a causal mask; the car length is `δ = cβ^{−1/2}`, the range beyond which the
attractive force decays. A **Rényi centre** is a token separated by more than
`δ` from every previously accepted centre; a **strong Rényi centre** is one
separated by more than `δ` from **every preceding token**. Both are greedy
sequential acceptance rules on positions and distances. **Neither mentions
clusters.**

> **F0 therefore does not test this claim.** `tools/run/p10_anchor.py` computes
> the mean normalised position of each HDBSCAN cluster's **earliest member** —
> a defensible proxy, and a different object: a cluster's earliest member need
> not be `δ`-separated from anything, and a strong centre need not be in any
> cluster. **§3.51's headline that "F0 fails in the direction it was predicted
> to succeed" stands as a fact about that statistic and is not evidence against
> the parking account.** `status-10.md` §6 already said F0 was not an
> adjudication; it is true for a stronger reason than the one given.

**The count law is better than the scan believed, and §3.50's correction
over-shot.** §3.50 records the law as `Θ(β^((d−1)/2))` with *"no 0.7476 in
it"*. The scaling is right. The constant claim is wrong — Appendix C.4 gives
`c·2π/δ` with `c ≈ 0.75` for `d = 2` ordinary centres as `δ → 0`
(Dvoretzky–Robbins 1964). **And the `d ≫ 1` verdict applies only to the
power-law asymptotic.** Lemma C.1's exact form,

```
    E[ #strong Rényi centres ]  =  E_{x∼μ}[ 1 / μ(B_δ(x)) ]
```

is a reciprocal local-density average: **distribution-free, dimension-free, no
exponent, no `β`** — only `δ`, which is a distance and can be swept. The paper
says so explicitly, and says the analogous computation for *ordinary* centres
remains open above `d = 2`. So the phase's quantitative test is
observed-count-versus-(C.1), not a log-log slope, and **the i.i.d. assumption in
(C.1) is the discriminant rather than a defect** — the gap between observed and
predicted *is* how far a token cloud departs from exchangeable, which is
`notes-10.md` §3.2's packing-versus-content question with an exact null
attached. A bonus: since `δ = cβ^{−1/2}`, the `δ` at which they meet **reads
back `c²/β`**, turning §3.40's undecided factor-of-8 convention into a
measurement.

**The gradient-flow hazard resolves in the project's favour.** `notes-10.md`
§10.1 and `plan-9.md` §5.1a both recorded, `[S]`-grade, that the masked system
*"cannot be interpreted as a mean-field gradient flow"*, and put the
Wasserstein-Hessian framing at risk. The paper says that **and then says what
replaces it** (Lemma 5.3): the causal dynamics is a **sequential gradient
flow**,

```
    φ̇_k = − ( 1 / Z_k(φ_1,…,φ_k) ) · ∂E_k(φ_1,…,φ_k)/∂φ_k
```

— a different energy per particle, causally ordered. **What is void is a single
global potential for the ensemble** (so Łojasiewicz, and "the eigenvector sign
structure *is* the partition", do not transfer). **What survives is the
curvature claim in per-token form**, about `∂²E_k/∂φ_k²`. And `1/Z_k` is
**literally the prefactor on particle `k`'s own gradient** — which is F12's
measurement, and makes `math-1.md` §1A.6's reading of `Z` as a metric an
equation rather than an interpretation. `status-10.md` §1.5's
parked-versus-pinned argument inherits that upgrade.

**Prior art this project should not claim as its own.** §B.3 of the paper
measures the **`V`-matrix spectra of `albert-xlarge-v2`** — the model of Phases
1–6 — finds most heads have real `λ_max`, finds heads 1 and 6 with **negative**
`λ_max`, and notes Gaussian initialisation puts the spectrum far from the left
half-plane, so a negative `λ_max` is learned. That is the V-attractive /
V-repulsive split, published, on this project's model. Table 1 additionally maps
`(sign λ_max, multiplicity)` to five predicted final configurations, which is a
free join against the project's head catalogue.

**Every transfer is bounded by four simplifications the paper names**: tied
weights across layers, **no MLP** (*"a significant open challenge"*), and for
the meta-stability results `V = I`, `Q = K = I`, `d = 2`. **Nothing here is a
theorem about Pythia.**

### 3.52.2 The other four, in one paragraph each

**`2501.10573`** — note the **v2 title change**; `lit-10.md` §9 records the v1
title. It measures per-layer, per-prompt **intrinsic dimension** with three kNN
estimators on four models including Pythia 6.9B, and finds `ρ(log ID, surprisal)
≈ 0.6–0.8` at `p < 0.01`. Two things for this project. **The `d_eff` candidate
list in `math-10.md` §5.2 was missing the theoretically-correct entry**: the
parking law wants a *manifold* dimension, effective rank and participation ratio
are functionals of a covariance spectrum, and a kNN ID estimator gives **≈ 7–15
rather than ≈ 225** — the difference between a predicted slope that is
measurable and one that is not. And its **graded block-shuffle null** (`b_S =
N/4^S`, six levels, unigram-preserving, calibrated by BLEU and BERTScore) is a
dose–response curve where this project has binary controls.

**`2601.02932`** — `plan-9.md` §5.1's construction, built: Perron–Frobenius →
concentrations → coarse partition → Diffusion Maps → Ulam → implied timescales,
PCCA+, MFPT, transition-path theory. **The state is the empirical measure, not
the particle**, which `cluster_tracking.py` is not. It independently reaches
§3.29's programme — a **translation-invariant Wasserstein** metric is necessary
once cluster centres drift, where `L²` fails — and `core/dissipation.py` already
has that machinery, unrun. **The one thing not to copy is its
reversibility-constrained estimator**: their system is a reversible gradient
diffusion and a depth dynamics is not. Their own §5.1.1 names the alternative —
singular values, or the **real Schur decomposition** — which is this
repository's existing instrument, and which closes `math-10.md` §6's thread 4.

**`2605.12765` (GUARD-IT)** — occupies the genus Phase 9 claimed, and **two
thirds of the differentia `lit-10.md` §6 drew do not survive the paper**. Its
Eq. 8, `h′ = (h − αv̂)·‖h‖/‖h − αv̂‖`, is **norm-preserving but not a
rotation** (a nonlinear input-dependent self-map of a sphere), and it **is**
exactly invertible in closed form — so neither "isometry versus congruence" nor
"exactly invertible" separates the two. **What survives is cleaner: state
versus operator.** GUARD-IT moves `h`; a γ-patch is a congruence on `W_QK`,
changing how every *pair* is compared. GUARD-IT selects by **content** (a
similarity gate, and it does nothing when the gate is empty); a γ-patch selects
by **geometry** and always applies. `notes-9.md` §8 is rewritten to make that
claim.

**`2505.16831` (ICML 2026)** — *"models can appear to forget while their
original behavior is easily restored through minimal fine-tuning."* **Any Phase
9 forgetting result now needs a relearning arm**, or it measures the thing this
paper says is routinely mismeasured; `notes-9.md` §2's *"collapse is a release
operation, not a deletion one"* is the same claim and should cite it. Its
four-diagnostic panel — PCA similarity, PCA shift, **linear CKA**, FIM diagonal,
run on forget / retain / *unrelated* probe sets — is built for exactly the
situation `CLAIM-C` hit in §3.41, where six metrics disagreed; one of its four
is `cka_prev`, the metric that scored 0/8. It also carries a **Davis–Kahan
bound**, `cos∠(c^orig, c^upd) ≈ 1 − O(‖E‖/(λ₁−λ₂))`, which gates a
PC-direction readout on the **eigengap** — this project computes eigengaps
everywhere and has never gated a spectral readout on one. A
`tools/math_checks/` item.

### 3.52.3 What this changes in the tree

Documentation only; no code, no records, no registry entry.

- **`p10_cluster_function/lit-10.md`** — §11–§15, the five reads, marked `[R]`.
- **`p10_cluster_function/math-10.md`** — new **§7**; §5 marked partly
  superseded; §6 threads 1 and 4 marked answered. **§7 has no symbolic check
  file yet and says so** — its two closed forms (the finite-`n` saturation
  identity, the `δ → c²/β` inversion) are the kind `CLAUDE.md` says should get
  one.
- **`p10_cluster_function/notes-10.md`** — §3.2, §5, §6, §10.1 and §12 amended;
  **eight ladder rows added (F13–F20)**, seven of them free.
- **`p10_cluster_function/status-10.md`** — F0's reading amended, the ladder
  extended, and **§5.1, a revised next-step ordering**.
- **`p10_cluster_function/attention-10.md`** — §5 upgraded from interpretation
  to equation; row **A9** added.
- **`p9_metric_intervention/plan-9.md`** — §5.1a resolved, **§5.1b** added, two
  readout rows added.
- **`p9_metric_intervention/notes-9.md`** — §8's Hessian claim corrected to its
  per-token form and its novelty claim rewritten.
- **`p1_mstate_tracking/lit-1.md`** §4 item 1 and **`docs/LITERATURE.md`** rows
  6 and 19–22 corrected.

### 3.52.4 The next free row is no longer the one §3.51 named

`status-10.md` §5.1 has the revised order. The change: **F13, the centre scan**
— greedy sequential acceptance over token positions, both rules, swept in `δ`,
per layer — moves to the front. It needs positions and a distance and **not the
HDBSCAN partition at all**, which makes it the one row in this phase **immune
to §3.51.4's reproducibility floor**, and it is the test F0 was standing in for.
**F14** (observed count versus Lemma C.1) sits beside it. `CLAIM-C`'s two
HDBSCAN metrics against the floor is now third, unchanged and still the only
open item bearing on a *registered* prediction.

### 3.52.5 Two sessions read the same paper on the same day, and they differ in one place

`2411.04990` was read twice on 2026-09-20 by concurrent sessions: **§3.53 /
`docs/readings/2411.04990.md`**, and **§3.52 / `lit-10.md` §11 / `math-10.md`
§7**. Recorded here because a reader will otherwise find two `[R]` accounts and
no statement of how they relate.

**They agree on everything except the gradient-flow verdict.** §3.53 item 3
quotes §4 of the paper — *"our system (CSA) does not have a gradient-flow
structure and thus techniques of Łojasiewicz are not applicable"* — and
concludes that `plan-9.md` §5.1a's hazard is **confirmed**, with hierarchy
replacing the lost structure.

> **That is the paper's §4 and not its last word.** §5.2 and **Lemma 5.3** say:
> *"Since our dynamical system is not a gradient flow, the classical Łojasiewicz
> convergence theorem does not apply. Instead, we establish convergence by
> observing that the causal dynamics (both with and without frozen tokens) is,
> in fact, a **sequential gradient flow, where each particle minimizes a
> slightly different energy**."* — with the flow written out,
> `φ̇_k = −(1/Z_k(φ_1,…,φ_k)) ∂E_k/∂φ_k`, and the `E_k` given explicitly in
> App. C.3.
>
> **So the hazard is narrowed rather than confirmed.** A single global potential
> for the ensemble is void; a per-particle energy and a genuine gradient flow in
> it are present. §3.52.1, `lit-10.md` §11.5, `math-10.md` §7.5 and
> `plan-9.md` §5.1a carry the resolved version, and it is the one to use.

**What §3.53 has that §3.52 does not**, and none of it is contradicted:
Theorem 4.1's consequence for `plan-9.md` §4.3 (no QK-side intervention can
prevent collapse under `V = I_d`); **Lemma C.1's saturation as the formula for
the 50–55 carrying-capacity invariant**, which is the better join of the two;
`d₁ = dim L` making the `d_eff` test differential against Phase 2's projectors;
**RMSNorm's diagonal absorbable into `K, Q, V`**, so a γ-patch is not a
read-side lever and `plan-9.md` §4.7 needs a `W_V` arm; Lemma 5.1's `T_j·s_j`
bound; the `⟨Qx, Ky⟩` generalisation of the separation; explicit timescales; and
the Castin–Ablin–Peyré pointer, which may restore mean-field structure and
should be read before the hazard above is treated as final.

**What §3.52 has that §3.53 does not**: the other four papers, Lemma 5.3, the
observation that **F0 measured a proxy**, the re-correction of the 0.7476 claim,
and the kNN intrinsic dimension as the manifold `d_eff`.

**A process note, since this is the second time.** `CLAUDE.md`'s concurrent-work
hazard is real: two sessions sharing this tree did the same read an hour apart,
and only the merge surfaced it. The cheap prevention is the convention §3.53
introduces — **`docs/readings/<arxiv-id>.md`, one file per paper read** — which
makes a duplicate read a merge conflict rather than two divergent accounts.

## 3.51 Phase 10's free rows RAN — and three of the findings are about this project's instruments, not about clusters (2026-09-20)

**`p10_cluster_function/status-10.md` is the phase record and the place to
start.** Four rows of `notes-10.md` §8's ladder ran on real checkpoints — F0,
F1, F11-A0 and F12 — plus a producer that unblocked them and a measurement that
was not on the ladder. All **tier 1, exploratory, unregistered**;
`claims/registry.json` is untouched and none of it is quotable as an
adjudication. Seven records under `data/analysis/`, each at 2 000 permutations
with `max_attainable_E` 22.37 on its face.

This section carries only what is **project-wide**. The per-row numbers,
caveats and re-run instructions are in `status-10.md`.

### 3.51.1 The four rows, in one table

| row | finding | detail |
|---|---|---|
| **F11-A0** | the attention flip is **~94 % causal mask** — raw gap 0.592, corrected **0.036** — and **100 % mask at initialisation**, with a learned residual only from step ~2000 | `status-10.md` §1.1 |
| **F0** | the anchor test **fails in the direction it was predicted to succeed**: nuclei at 0.3164 against a null mean 0.2077, median p 1.00, and it survives a null restricted to the clustered population | §1.2 |
| **F1** | the identity coupling is **exactly optimal in 99.5 %** of 3 648 boundaries, so **every per-layer displacement number this project has recorded is true `W_2`**, not the upper bound it was known to be | §1.3 |
| **F12** | `math-10.md` §2 confirmed **β-independently**: raw `log Z` is 99.5 % position, the sink is the *minimum* of raw `Z` and the *maximum* of corrected `Z` | §1.4 |

**Two blockers had to be cleared.** The 410m sweep carried **no density
partition at all** — `hdbscan_labels.json` empty in **152/152** directories,
wider than §3.41 records and listed as present in `docs/AXES.md` — and the
partition turns out not to be reproducible run to run (§3.51.4 below).

**F1 and F12 together read parked rather than pinned**, in a window at steps
32–512 rather than as a property of the trained model, and held as a hazard
rather than a result: the density confound is argued from a step-0 baseline
rather than controlled, and the reading rests on `math-1.md` §1A.6's
*interpretation* of `Z` as a metric. `status-10.md` §1.5 and §6.

**Both headline rows were replicated on the pilot sweep** — different forward
pass, **native** labels rather than the backfill, nine prompts against eight,
27 checkpoints against 19. A0's corrected gap agrees to two or three decimals
at all 13 shared checkpoints; F0's statistic agrees to three, **0.3164 against
0.3157**, with the pilot finding *fewer* clusters per layer (42.8 against
47.1). `status-10.md` §3.1.

### 3.51.2 Three instrument defects, each of which would have produced a false number

Found while wiring e-values through the rows, and each is fixed with a test
that fails on the old behaviour.

1. **`core.evalues.combine` is a PRODUCT, valid only under conditional
   calibration** (`EProcess`'s own WARNING). Phase 10's units share a model, a
   text and a forward pass, so the product is invalid over them. Measured cost:
   at 25 perfectly dependent units the product rejects **19.67 % of the time
   under the null** against a nominal 5 %. `average` / `average_p` are the
   arithmetic-mean merger, valid under **arbitrary dependence** by linearity
   alone (Vovk & Wang 2021), with the price stated: the mean cannot exceed its
   largest input.
2. **`p_from_null` compares with an exact `>=` and was decided by rounding
   noise on a degenerate null.** Where the mask correction explains a layer
   completely, the corrected value is the same at every token, no permutation
   can move the enrichment, and the honest p is 1. It returned **0.0025, the
   resolution floor** — a perfectly explained layer reading as the strongest
   possible evidence — because ~1.0 differs from ~1.0 in the sixteenth digit.
   `p_from_null_tolerant` counts near-ties conservatively and flags
   `degenerate_null`.
3. **The first full A0 run could not have rejected whatever the data said.**
   The mean merger cannot exceed its largest input and a Monte-Carlo p cannot
   go below `1/(n+1)`, so the largest merged e-value a permutation design can
   EVER produce is `calibrate(1/(n+1))` — **10.01 at 400 draws, against a
   threshold of 20.** It reported `reject: False` at all 19 checkpoints and
   that number said nothing. `core.evalues.max_attainable_average_E` computes
   it, **1 599 draws is the exact minimum**, every runner now runs at 2 000 and
   records `max_attainable_E` and `design_can_reject` on the face of its
   artifact. This is `p_from_null`'s own "should I draw more?" versus "could
   this design have rejected?" applied to the merger.

**Read every `reject: False` in these records against `max_attainable_E`.**
Even at 2 000 draws the averaging merger is deliberately low-powered: rejection
needs nearly every unit at the floor. The informative fields are the effect
sizes, `median_p` and `frac_below_05`.

### 3.51.3 `pythia-410m` step0 and step1 are the same weights

**All 292 tensors bit-identical upstream**, despite different HF revisions and
different blob hashes, and the sweep's activations for the two are bit-identical
in turn — while step0 vs step2 differs at 2.4e-07, so the comparison is not
saturated. Not a runner bug.

**The 19-revision checkpoint axis carries 18 distinct points at 410m**, and
every per-checkpoint average over the sweep double-counts one. Recorded in
`docs/AXES.md` §2.3. **Not yet checked at 70m.**

### 3.51.4 The HDBSCAN partition is not reproducible run to run, and nothing had measured it

Found while cross-checking the backfill, and it is the most consequential thing
in this section because four rows and one registered gate read the partition.

**The natural experiment.** Two independent Phase-1 sweeps cover the same
checkpoints and prompts: the 2026-08-12 pilot on `HDD_1TB` with native labels,
and the 2026-08-31/09-01 sweep whose partition was backfilled. **104
model-prompt directories overlap, 2 600 layer-pairs.** Their `tokens.txt` are
identical in every case and their configs match; their activations differ by at
most **7.9e-05** — ordinary run-to-run float non-determinism, not a difference
in what was computed. So it is the same question asked twice, differing only by
numerical noise.

**What comes back** (`data/analysis/p10_partition_stability.json`):

| | value |
|---|---|
| label vectors identical | **83.3 %** |
| ARI, median | 1.0 |
| ARI, mean | 0.933 |
| **ARI, 5th percentile** | **0.347** |
| **ARI, minimum** | **0.166** |
| ARI with noise dropped, p05 / min | 0.585 / 0.327 |
| cluster-count \|Δ\|, mean / **max** | 0.58 / **20** |
| noise-fraction \|Δ\|, mean / max | 0.003 / 0.132 |

**Usually stable, and in about one layer in six it is not.** In the tail the
two partitions are essentially unrelated — ARI 0.17, cluster count off by 20 —
from activations agreeing to five decimal places.

**This is a measurement-reproducibility floor, not a null.** There is no
hypothesis under test and no p-value, deliberately. It is the amount by which a
partition-derived quantity can differ between two runs that asked the same
question, and **no null this project has built accounts for it**:
`notes-10.md` §4.4's size-profile null is about the ARI's variance under random
labelling, which is a different quantity from its variance under re-measurement.

**What it bears on, and what it does not settle.**
`CLAIM-C` reads **`cluster_count` and `cluster_membership` from HDBSCAN and
from nothing else** (`replication_gate.py`), and §3.41's gate run scored those
two at 2/8 and 5/8 — the two weakest of six. That is now a number that has to
be read against a floor nobody knew the size of. **It does not follow that the
gate's result is noise**: the arms there differ by model and by training, not
by a re-run, and this section measures only the re-run. What follows is that
the comparison has never been made, and `tools/run/p10_partition_stability.py`
is what would make it.

The same caution applies inward. Row A0 and F0 both read the partition, and
both report per-layer statistics aggregated over 3 600+ units — an effect
carried by the aggregate is much less exposed to a per-layer floor than a
statement about one layer would be, but neither row has been re-run against a
second partition to check.

**Done for BOTH headline rows — see `status-10.md` §3.1, and they hold.** A0's corrected
gap agrees to two or three decimals at all 13 shared checkpoints; F0's nucleus
statistic agrees to three (0.3164 against 0.3157). The floor still binds any
per-layer claim; what the checks establish is that a statistic aggregated over
thousands of units is far less exposed to it.

## 3.50 The scan, and the math: four check files, three corrections, and a law that is not in `n` (2026-09-20)

`p10_cluster_function/math-10.md` (the derivations), `lit-10.md` §§5–10 (the
second scan pass), and four new `tools/math_checks/` files — **28 checks, all
passing**, each stating on its face what it does not prove. Nothing run on real
artifacts; nothing registered.

| file | checks | subject |
|---|---|---|
| `causal_mask_attention_baseline.py` | 9 | the mask's structural tilt; `Z_beta,i` |
| `cone_margin_gamma_gradient.py` | 5 | the γ-patch derivative of the cone margin |
| `ari_size_profile_null.py` | 6 | what a size-profile null buys |
| `parking_scaling_slope.py` | 8 | the parking law's convention-free slope |

### 1. The attention flip sits on a structural tilt three orders of magnitude wide

Content-free — attention uniform within the causal triangle — position `j` is
visible to `n − j` queries and

```
    received(j) = H_n − H_j ,   Σ_j received(j) = n
```

so **the layer mean is exactly 1 and `received(j)` *is* the "× layer average"
quantity `noise_importance_proxy.py` reports.** Baseline and measurement are
directly comparable with no rescaling. At the battery's `n = 264`: **6.155×** at
position 0, **0.691×** at the median, **0.0038×** at the last token — about
**1 600×**, before content.

**The observed 1.6× / 0.5× is a factor of 3.2 inside that, and is reproducible
with zero content**: the baseline already equals 1.6× at position ≈ **53** and
0.5× at ≈ **160**. The two values also pin the split to `f = 5/11 ≈ 45 %`, which
`status-5c` independently reports — an internal-consistency check, not evidence
of content. **So `attention-10.md`'s row A0 is not tidying: until the mask
baseline is divided out the flip is not known to measure anything.**

Free corollary: content-free row entropy is exactly `log(i+1)`, layer mean
`log(n!)/n` = **4.590 nats** at `n = 264`. `attention_entropy_per_head` is stored
for the whole sweep and **the deviation from `log(i+1)` is the content**; the raw
value is mostly position.

### 2. `Z_beta,i` reverses under masking, and `math-1.md` §1A.6 is an unmasked statement

In the concentration regime: **unmasked** `Z_i = n·e^{βγ}`, position-independent —
so a spread is content and "a high-`Z` token (a sink) is expensive to move"
reads fine. **Masked**, row `i` sums over `j ≤ i`, so `Z_i = (i+1)·e^{βγ}` and
**position 0 is the minimum.** Meanwhile `received(j)` decreases by exactly
`1/(j+1)` per step.

> **The two mask baselines are anti-aligned. The sink is simultaneously the
> largest received attention and the smallest `Z` — on the metric reading, the
> *cheapest* token to move.** §1A.6's identification should be labelled as
> holding for the unmasked model. **Measure `Z_i/(i+1)`.**

This sharpens the paid/received square rather than weakening it: the sink is a
**specific corner** (low `Z`, high received), distinct from parked (low both) and
carrier (high both).

### 3. The cone margin's response to a γ patch, in closed form

`plan-9.md` §4.4 asserted this was free to compute. It is, and here it is. With
`Γ → Γ + εuuᵀ`, by Danskin at a unique minimiser `λ*`:

```
    d(m²)/dε |_{ε=0}  =  2 · (uᵀ X̂ᵀ λ*) · (uᵀ c(λ*))
```

checked against a full re-solve by central differences at `(6,4)` and `(9,5)`,
relative error `< 1e-9`. **It reads the binding set and nothing else** — `λ*` is
supported on the binding tokens, so a patch aimed where no binding token sits has
**zero first-order effect however large its `D`**. Second order is `s(λ*)²‖u‖² ≥ 0`
at fixed `λ*`. Danskin needs a **unique** minimiser, and the degenerate case is
exactly the near-zero-margin configuration `math-1c.md` §7.2 calls informative —
**a runner must report whether `λ*` was unique.**

### 4. Correction: the ARI is already centred

`plan-9.md` §2.3 and `notes-10.md` §4.4 asked for a size-profile null to remove a
bias. **The adjusted Rand index already subtracts exactly that expectation**, so
`E[ARI] = 0` under that null by construction — Monte Carlo at three regimes,
including 200 + 8×8, all within 4 SE of zero.

**The instruction survives, its reason changes: the null is for the variance.**
Null 95th percentiles: **+0.009** balanced, **+0.092** one giant cluster,
**+0.002** giant-vs-balanced — a **57× range**. So **a fixed ARI threshold is not
comparable across layers, checkpoints or models**, which is `compare_rungs.py`'s
no-absolute-threshold rule one level up, on the axis where HDBSCAN's size profile
is what moves. And a hazard no adjustment touches: **HDBSCAN noise is not a
cluster**, `ignore_noise` changes what `N` means, and 40–50 % of tokens are noise.

### 5. Correction: the parking law is in β and `d`, not in `n` — and the test improves

`lit-1.md` §4 item 1 described the prediction as *"a density constant (the Rényi
constant ≈ 0.7476) and hence an expected number of occupied cells as a function of
n"*, and two reviews rate checking it the project's best cheap experiment.
**Both halves are wrong.** Two independent search summaries **[S]** give the
frequency of Rényi and strong-Rényi centers as **`Θ(β^((d−1)/2))`** (confirmed at
`β^(1/2)` for `d = 2`) — a law in **β and dimension**, with no 0.7476 in it. At
`d = 1024` the exponent is **511.5** and the prediction is unusable: the `d ≫ 1`
problem `design-1.md` already records for Figure 3, inherited. **Registering F0 on
the old reading would have frozen the wrong statistic** — `CLAUDE.md` trigger 2,
firing at trigger-1 time. `lit-1.md` and `docs/LITERATURE.md` §6 are corrected in
place.

What replaces it is better:

- **The anchor test, free today.** The mechanism the paper supplies is that
  **early tokens act as nuclei** for cluster formation. A per-token,
  position-indexed prediction, checkable against `hdbscan_labels.json` plus
  positions with **no β, no convention decision and no reading of the paper**.
  This is now F0.
- **The slope test.** Fit `log count ~ a·log β + b·log n`. A constant rescale of β
  moves the **intercept, not the slope**, so `a` is **invariant to β's undecided
  unit convention** — checked symbolically and by recovery on synthetic data
  (fits on β and 8β agree to `1e-9`). **A test that looked blocked on that
  decision is not.** And `d_eff = 2a + 1` **measures** the effective dimension the
  clustering behaves as: the candidates on disk (ambient 1024, rank plateau ~225,
  participation ratio 22) predict slopes five orders of magnitude apart. One
  failure mode, checked: `head_size` is 64 on gpt2-large and 128 on pythia-1.4b,
  so a **cross-model regression on raw β mixes conventions** and biases the
  slope — per-model regressions, or fix the convention first.

**And the collision worth keeping.** Parking says early tokens are nuclei; §1 says
the mask makes early tokens attention-rich. Two consequences of one mask, and
Phase 10 currently treats one as a finding and the other as a confound.
**Position is not a nuisance variable in this phase — it is the mechanism the
theory names.**

### 6. What the scan did to Phase 9

- **Unlearning: the genus is occupied, the differentia survives narrowly.**
  `2605.12765` (GUARD-IT) is training-free, gradient-free, *"entirely in
  activation space"*, executing unlearning as *"a controlled geometric
  transformation"* — **pure rotations preserving the activation norm.** A rotation
  is an **isometry**; a γ-patch is a **congruence** (§4.1's `W_QK → Γ'W_QKΓ'`),
  which is the one thing an isometry is not. Rewrite the claim to that, not to
  the category. And `2505.16831` (*Unlearning Isn't Deletion*) is **support** for
  `notes-9.md` §2's release-not-deletion rule — with the consequence that any
  Phase 9 unlearning result must be tested for reversibility.
- **Spreading is a comparison, not a discovery.** `2303.06562` **ContraNorm** is
  a normalisation-layer modification that spreads representations apart — the
  closest published object to a Phase 9 metric intervention. Read before
  `design-9.md`. Also `2410.07799`, and `2602.09297` (*Laplacian Heads*, the
  deliberate-smoothing direction, i.e. the cluster-forming arm from the other
  side).
- **§6.2 still looks open.** Every self-repair route the field names is an
  **ablation** route; nothing surfaced measures repair against a non-removing
  intervention. `[S]` only, so it stays queued rather than registered.
- **The gradient-flow hazard is confirmed twice.** *"This modification translates
  into an interacting particle system that cannot be interpreted as a mean-field
  gradient flow."* §3.48's `plan-9.md` §5.1a stands.

### 7. Two neighbours, and a queue

- **`2501.10573`, *The Geometry of Tokens in Internal Representations*** — **the
  closest neighbour Phase 10 has.** Empirical measure, *"the mean-field
  interacting picture"*, intrinsic dimension / neighbourhood overlap / cosine
  similarity per layer, a measured **correlation between token geometry and
  next-token cross-entropy**, and a **shuffled-token control** this project
  should consider adopting as a null.
- **`2601.02932`, *Data-driven Reduction of Transfer Operators for Particle
  Clustering Dynamics*** **[N]** — the title is `plan-9.md` §5.1's construction.
  Searches for PCCA+ / implied timescales on **transformer representations**
  returned molecular dynamics, behaviour and climate and **nothing on
  transformers**, so that move looks open.

`lit-10.md` §10 is the reading queue, in priority order. **`2411.04990` is
first — two phases depend on that one paper and neither has read it.** A thread
not pulled and worth flagging: standard MSM spectral theory assumes
**reversibility**, and a causal transition matrix is not reversible;
`t_i = −1/log|λ_i|` still reads, but PCCA+'s sign-structure argument may not
(`math-10.md` §6 item 4).

## 3.49 The attention flip, audited — and the measurement grid nobody had drawn (2026-09-20)

`p10_cluster_function/attention-10.md` and **`docs/AXES.md`**. Pre-design,
nothing registered.

### The flip, and the reading of the random case is right

`p1_mstate_tracking/visualization/noise_importance_proxy.py` — **live, not
archived** (`notes-10.md` §3.1 said archived; corrected). Per layer it computes
attention received per token, diagonal zeroed, summed over heads and queries,
**divided by the layer mean**. Trained gpt2-large: unclustered **~1.6×**,
clustered **~0.5×**; ALBERT-base **>2× / ~0.5×**; random: **near parity**.

**Because the statistic is a per-token ratio, population size is already divided
out** — so "near parity" under random weights means attention is roughly uniform
and mass therefore follows population, which is precisely the numbers-game
reading. Multiplying back: 1.6× on ~45 % of tokens is ~72 % of the mass, >2× is
~90 %, which is `math-1.md` §13.1's independently stated *"≈90 % of attention
mass on ≈50 % of tokens by late layers"*. **Two instruments, one number.**

### Four things never checked, two of them structural

1. **Position 0 is the sink and is unclustered by construction.** NeoX prepends
   no BOS, so position 0 carries a norm **one to two orders above the bulk**
   (§2.5 of `math-1.md`) — exactly what HDBSCAN calls noise.
   `_received_attention` zeroes only the diagonal. **`core/sink_audit.py` exists
   to decide this class of question** — enrichment against a structural
   baseline, three-outcome rule stated before the numbers — **and has never been
   pointed at the flip.**
2. **The causal mask gives early tokens a mechanical `1/j` advantage.** A token
   at position `j` is visible to `n − j` queries; `received` sums over queries
   without normalising by how many could have attended. **If the unclustered
   population skews early — and position 0 is the extreme case — part of the
   flip is the mask.** `sinkhorn.py` already builds the mask-only uniform
   baseline for a per-head Fiedler purpose and classifies **on the deviation**;
   the same object, one population level up. **This is the sharpest of the four
   and nothing in the flip's measurement carries it.**
3. **It is a mean.** §3.13 is this project's own section on that: exploratory
   work reports a mean **and** an extremum, always. 1.6× over ~120 tokens is
   equally consistent with a broad shift and with one token at 40×.
   `compare_rungs.py`'s participation ratio / top-k share / gini are the
   threshold-free shape statistics, already written.
4. **Never run on Pythia, never on a checkpoint axis.** And
   `claims/audits/p1c_inputs.json` says **152/152 directories carry
   `attentions.npz`** — the full `(n_layers, n_heads, n, n)` tensor, 19
   checkpoints × 8 prompts. **The flip's developmental curve costs no forward
   pass and has been available since 2026-09-01.** Four known transitions to
   co-locate against — under `changepoint_colocation`'s matched-control null,
   with the falsifier named first, because the registered permutation null for
   this class was measured and rejects under H0 at 0.32–0.45.

Already settled: **punctuation is ruled out** (same clustered/unclustered ratio
under random weights). Named and unchecked: **token frequency**
(`docs/LITERATURE.md` §6 item 10) — a genuine alternative explanation for the
whole finding.

### What the tensor answers that the scalar cannot

- **Which heads divert.** Summed over heads today; per-head is free, and at 410m
  it joins to 7d's 384-head causal sweep and to `attention_entropy_per_head`,
  **stored since Phase 1 and never read against cluster structure**.
- **The population×population mass matrix.** The scalar flip is the column sums;
  the off-diagonal is what discriminates. **`H-PARK` predicts low, undifferentiated
  clustered→clustered attention; `H-CAT` predicts high and structured.** 5c
  already has the *inner-product* version of this decomposition — within-cluster
  cohesion high and flat, the energy plateau carried entirely by within-cluster
  pairs — and **high cohesion with low mutual attention is the signature of
  parking.** Nobody has put the two side by side and they are the same shape.
- **Attention paid vs received.** The flip is a column statistic; nothing here
  has looked at the row side by population. The 2×2 separates **sink** (receives
  much, pays nothing), **parked** (inert both ways) and **carrier**
  (individuated and in use) — three things the received-only statistic cannot
  tell apart.

### `Z_beta,i`: a trained per-token metric nobody has examined

`math-1.md` §1A.6: *"the partition function is not noise to be normalized away —
it is a metric... a high-`Z` token (a sink) is one the metric makes expensive to
move"*, and **"nothing in this project has looked at `Z_beta,i` as a per-token
quantity at all"** (§15, open question 12).

**Phase 9's lever `Γ` is a per-channel metric; `Z_beta,i` is a per-token one** —
together the two cheap metric levers the architecture already contains, and `Z`
is measurable from artifacts on disk with no intervention. It **separates the two
kinds of stationary**: a *parked* particle is still because nothing pushes it, a
*pinned* one because the metric makes it expensive to move. That resolves the
sink confound with a measurement rather than an exclusion rule — strictly better,
since excluding position 0 discards the particle whose behaviour is most
informative. **Caution carried:** `Z_beta,i` is particle `i`'s **row** normaliser
while a sink is a **column** phenomenon; §1A.6's identification is asserted, not
measured, and the paid/received 2×2 is its test.

### `docs/AXES.md` — the map from questions to data

New file, project-wide, referenced from `INDEX.md`. `INDEX.md` maps phases to
directories; **this maps questions to data**, because the project keeps
rediscovering that an expensive-sounding question is already answerable and that
a cheap-sounding one needs a producer nobody wrote.

Seven axes (model, checkpoint, prompt, layer, head, token, frame) plus three that
behave like axes and get forgotten (sub-layer channel, ablation mode, the random
twin). What is populated, what each axis buys, and **six producers that do not
exist** — chief among them the **`beta_eff` writer**, which needs *no forward
pass* (`attentions.npz` + LN params → `ln_frame` → `beta_eff`, demonstrated at
16/16 heads in all 24 blocks) and unblocks **two registered predictions** plus
every `gamma_beta` comparison Phase 9 would make. It is gated on one human
decision: **β's unit convention, worth a factor of 8.**

**The prompt axis is narrower than the battery.** 21 prompts in `core/config.py`;
**the Phase-1 sweep ran 8**. So 13 have never been through Phase 1 — and it
matters for one test specifically: **the Rényi-parking prediction is a cluster
count as a function of `n`**, so prompt length is its independent variable.
**Eight points against twenty-one, over a wider `n` range, is the cheapest way to
strengthen the project's best adjudication candidate.**

Ten rules for combining axes are collected in one place for the first time — no
absolute threshold transfers between rungs; normalised depth, no band; ablation
mode named not defaulted; the rung policy and its untaken rule 4; prompts on one
model are not independent; mean-and-extremum; margin not boolean; sinks audited
not assumed away; co-location needs a matched control and a falsifier first; a
non-ladder sub-study is its own ground.

**Fourteen questions never asked**, ten of them free. If only four were done:
the parking law (§3.48); the attention audit and its trajectory, **A0 first**;
the β producer; and a Phase-1 clustering sweep on **70m** — 6 layers, `d = 512`,
19 revisions already on disk, the cheapest new compute in the project and the
only item that gives everything above it a second rung. **Check before assuming
it is new:** Phase 8 ran the head catalogue and invariants at 70m, which is
head-level ablation, not a clustering run.

Nothing has been run. `claims/registry.json` untouched.

## 3.48 Phase 10 opens: what clusters are and what they do, and the instrument that makes it answerable (2026-09-20)

`p10_cluster_function/notes-10.md` and `lit-10.md`. **Pre-design** — nothing
frozen, no `P-*` id, `claims/registry.json` untouched. `CLAUDE.md` trigger 1 is
**partially** discharged: `lit-10.md` settles one question properly and names what
it did not scan.

**Why it is a phase and not a subphase of 9.** §3.47's finding was that Phase 9
cannot start at the intervention. The prerequisite — what a cluster is and does —
has different instruments (a lens, a partition, a packing prediction, a particle
table, against a metric patch and a KL readout), a different literature (the
global-workspace/lens line and the causal-mask theory, against steering and
unlearning), and `design-5c.md` already recorded the reason not to bundle two
questions into one phase. **Phase 9 is parked, not closed**; `notes-10.md` §9
states what each owes the other.

### The hypothesis, made falsifiable

The user's framing: **a cluster may be trash collection** — a compressed cleanup,
one representation standing for one thing, with many particles put into it to keep
them stationary. Seven results already on disk point that way and had never been
read as one argument (`notes-10.md` §2): ~50 % of tokens clustered at any layer;
**unclustered tokens absorb ≈90 % of attention mass by late layers**; trained
models route 1.6–2× toward unclustered tokens and ~0.5× toward clustered ones,
**sign-flipped under random weights**; **carrying capacity invariant at 50–55
max-alive across all 27 checkpoints while lifespan falls 7.0 → 4.5 and births rise
113 → 164**; effective rank plateauing at 200–250 across `d_model` 768–1600; and
the energy plateau carried **entirely by within-cluster pairs**.

Formalised as **four signatures of a parked particle — kinematic, attentional,
functional, causal — and the hypothesis is that they coincide** (`H-PARK`). The
rival (`H-CAT`) says they dissociate: clustered particles are quiet but
load-bearing. **Findings 1–7 establish only the first two signatures; the entire
discrimination lives in the two columns nobody has measured.**

The quantitative arm is **`2411.04990`'s Rényi-parking correspondence**, which
predicts a cluster count as a function of `n`. If a cluster is a parking space its
count is set by packing and is largely content-insensitive; if it is a computed
category the count tracks content. Phase 1 holds counts per layer, per prompt
length, at 27 checkpoints, on disk, and **two independent reviews already rate this
the project's best cheap experiment** (`lit-1.md` §4 item 1: *"do this one
first"*). `claims/adjudications/` holds zero entries against thirty-nine
registrations.

### The instrument: the Jacobian lens, and it was already being cited here

`lit-10.md` §1. **Gurnee et al. 2026, *Verbalizable Representations Form a Global
Workspace in Language Models*** — arXiv **2607.15495** **[S]**,
`transformer-circuits.pub/2026/workspace`. **This is the same paper
`p2_eigenspectra/lens_band.py` and `archive/p5_single_mstate_analysis/status-5.md`
have cited since July, without an arXiv id.** Companion code
`github.com/anthropics/jacobian-lens`, Apache 2.0, **read as primary text**:

```
    lens_l(h) = unembed( J_l @ h ),   J_l = E[ ∂h_final / ∂h_l ]
```

expectation over prompts, source positions and all target positions; the paper's
lenses use 1000 sequences of 128 tokens and quality **saturates by ~100 prompts**;
fitting cost is the model's own backward pass and parallelises via
`JacobianLens.merge`. Pre-fitted lenses for **38 open models** at
`neuronpedia/jacobian-lens`, reported to include **`pythia-70m-deduped`**, one
`[d_model, d_model]` fp16 matrix per layer — all **[S]**, since `huggingface.co` is
blocked from here; **verify on the research machine before building on it.**

Three things it unlocks, and one it does not:

1. **The functional partition becomes measurable per layer.** §3.47 named the
   geometric-vs-functional ARI the most informative unrun number in the tree, and
   assumed only a final-layer LM head. `core/functional_distance.py`'s docstring
   already names "per-layer decoded distributions" as its input, so a J-lens
   readout drops into that slot **with nothing to change**.
2. **`J_l` is a `d × d` operator defined by function, not by weights.** Every
   operator instrument here — the Schur decomposition, the S/A split, `φ`, the
   attracting/repelling projectors — has only ever been pointed at `M_OV` and
   `M_QK`. **§2.4.6's standing negative is that no weights-only spectral quantity
   identifies the copier.** An averaged Jacobian is not weights-only. Running the
   existing decompositions on it is an import, and it is exploratory.
3. **`lens_band.py`'s stated deviation becomes a choice.** Its header says the
   logit lens is used "because no averaged Jacobian has been trained for these
   checkpoints and training one is deliberately out of scope", and that its
   detected band **onset is an upper bound** in consequence. Fitting one is now a
   licensed ~100-prompt procedure; at `d = 512`, 6 layers, a `pythia-70m` lens is
   ~3 MB per checkpoint. Likewise `status-5.md`'s blocker-4 note lists three
   routes for Group E and **the J-lens is the one its own source recommends** —
   the skip-to-output pathology it warns about is specific to correlationally
   trained affine translators, which an averaged Jacobian is not.

**What it does not unlock: the ladder's own model.** `pythia-70m-deduped` is a
different training run from `pythia-70m`, every registered 70m decision names the
latter, and the published lens carries **no checkpoint axis** — which is this
project's whole object. Borrow it as a separate labelled model, or fit our own per
checkpoint; those are different pieces of work and `notes-10.md` §4.5 keeps them
apart. A lens also remains a readout, not ground truth: structural proxies have
failed against causal ground truth twice here.

### The hazard this turned up, and it lands on Phase 9

**`2411.04990`'s first claim is that the causally-masked system cannot be
interpreted as a mean-field gradient flow.** Pythia is causal. `docs/LITERATURE.md`
row 6 already asks whether that voids Phase 2d's framing; **it bears on Phase 9 the
same way and nobody had said so.** `notes-9.md` §8 and `plan-9.md` §5.1 are built
on the **Wasserstein Hessian of `E_beta`** — "stretch a subspace" restated as
"change the curvature of a near-zero eigendirection" — and that restatement
presupposes the structure the masked theory says is absent.

- **At risk:** the Hessian framing, and any claim that a metric patch moves a
  curvature.
- **Survives:** the transfer-operator / implied-timescale / PCCA+ readout, which
  needs a *transition* structure rather than a gradient-flow one and which
  `cluster_tracking.py` already half-computes. **This is now a reason to prefer
  the timescale readout on its own merits.**
- **Survives:** everything algebraic in `plan-9.md` §4. Lemma 6.4 is proved from
  positivity of `a_ij` alone, so it is mask-agnostic; so are the congruence and
  the cone margin.

**`[S]`-grade. Read `2411.04990` before retracting anything** — it is the top item
in the queue, because two phases depend on that one paper and neither has read it.
`plan-9.md` carries the amendment inline as §5.1a, with §2.1a for the lens.

### Three distinctions the notes draw that the project had not

1. **Token clusters versus direction clusters.** A particle is a token; an SAE
   feature or a neuron is a direction. "Form a cluster around a feature" means
   *cluster the particles the direction selects*, which is a different experiment
   from clustering the directions. The J-lens is the bridge — `J_l` maps any
   residual direction into vocabulary space, so a token's state and a feature's
   direction read in the same units for the first time here.
2. **The intervention taxonomy** — subtractive/additive × weights/metric — places
   every intervention the project has run and shows the empty cell (additive ×
   metric = Phase 9). It also shows a second gap: **there is no subtractive metric
   intervention either**, and it is the natural matched control for every active
   `Γ` patch. And writing `mu_cond` into MLP 6's slot (§3.23) is already an
   additive weights intervention that worked — the closest precedent in the
   project to what Phase 9 proposes.
3. **The MLP's object is a direction, and you cannot form a token cluster inside
   an MLP.** §3.23 already found the object: `mu_cond` restores `L7H8`'s matching
   (0.643 / 0.567) while `mu_clean` restores nothing (0.038 / 0.016, on a par with
   zero and with a norm-matched random constant), at `cos = 0.837`, so the effect
   is entirely in the orthogonal component. Being position-independent it **cannot
   change the pairwise coupling** — so attention-side and MLP-side augmentation are
   *different operations*, not one operation at two sites: attention changes who
   couples to whom, the MLP changes the geometry the coupling is computed in, and
   **the MLP arm's effect is second-order and delayed by a layer.** A patch
   measured at its own layer will look like it did nothing.

### The ladder, and a method finding

Eleven rows (`notes-10.md` §8). **F0 (Rényi parking) and F1 (the transport
observables that still no runner calls) are free and unblocked today**; F2 is a
directory listing on the research machine; F6 (`turnover_decomposition`) is a
rebuild of an instrument validated in 2026 against synthetic sweeps and **awaiting
the real sweep ever since** — and it is the test that separates "the same particles
cycle faster" from "different particles cluster later", which cluster-level
statistics cannot.

**Method finding, recorded in `docs/LITERATURE.md` §0.1:** `github.com` and
`raw.githubusercontent.com` are reachable from a cloud session while `arxiv.org`,
`transformer-circuits.pub`, `huggingface.co` and `neuronpedia.org` are not. §5 item
18 guessed this; it is now measured. **A paper's companion repository is primary
text even when the paper is not**, and a new mark **[R]** distinguishes what was
actually read from `[S]`. Every future scan should try the companion repo first.

Nothing has been run. `claims/registry.json` untouched.

## 3.47 Phase 9 gets a plan, and the plan's finding is that the phase cannot start at the intervention (2026-09-20)

`p9_metric_intervention/plan-9.md`, the continuation of §3.39's `notes-9.md`.
**Still pre-design** — no construction frozen, no `P-*` id, `claims/registry.json`
untouched, and `CLAUDE.md` trigger 1 **not** discharged (§12 of the plan adds four
searches to `notes-9.md` §11, and the first of them — self-repair measured against a
non-ablation intervention — is the one that decides whether §6.2 is a question at all).

**The user's framing was: use the metric to cause or remove clusters, as a tool —
for forgetting, for isolating a fact or a mechanism, for spreading a region out so
it is more independent. The plan takes the prerequisite the framing names
seriously: nothing here establishes what a cluster does.** Three labellings of the
same tokens exist as code — geometric (HDBSCAN, run everywhere), functional
(`core/functional_distance.py`, pairwise KL on decoded next-token distributions,
**never run**), and mechanistic (`S`-projections, `math-6.md` §4 item 4,
**never built**). `frame_agreement` was written to score their agreement and has
never been called on a real run. **The ARI between the geometric and the functional
partition is the most informative unrun number in this tree**, it costs one forward
pass per prompt with an LM head and a matmul, and until it exists "cluster" names an
algorithm's output rather than an object.

Four accounts of what a cluster is are enumerated with what separates them
(discarded individuation / a computed category / an epiphenomenon of concentration
/ a capacity ledger), all four consistent with everything on disk, and the
epiphenomenon outcome is registered as legible rather than as a failure mode —
`math-5c.md` §4's discipline.

**What the mathematics gives, written down before any run:**

- A gamma-patch is a **two-sided congruence** `W_QK -> Γ' W_QK Γ'`, shared by every
  head reading that LayerNorm and **symmetric in query and key**. §2.5.6 is the
  warning: the transpose preserves the spectrum and copying stays broken, so
  read/write *asymmetry* carries causal weight a congruence cannot express. That is
  an argument for a write-side arm, not against the phase.
- **On Pythia the MLP can be excluded exactly.** Parallel residual means two LNs per
  block reading the same input: `input_layernorm` is attention's metric,
  `post_attention_layernorm` is the MLP's. "Augment the space in the MLP or in
  attention" is a clean three-arm factorial here, free, where most architectures
  give one blurred knob.
- **Lemma 6.4 survives every gamma** — only positivity of `a_ij` is used, and softmax
  is positive for any `Γ`. No metric deformation prevents collapse-from-a-hemisphere;
  resistance must come from `V` or from outside the paper's model.
- **The cone margin under a candidate patch is free to compute.** `m` needs only `G`,
  and the patched points are `Γ'x̂ + b` from activations already on disk. But the
  lemma's rate bound is `α' ≥ (1-α)/(2n e^{2β})` — `n` and `β`, **not `m`** — so the
  margin governs whether the guarantee applies, not how fast, and the plan says not
  to claim otherwise.
- **The best differential prediction**: raising mutual attention within a set
  converges it if its displacement lies in the attracting subspace and diverges it if
  in the repelling one. Same intervention, opposite outcome, predicted from the
  spectrum — against attention-pattern interpretability, which predicts one sign
  both times. It is simultaneously a test of whether Phase 2's projectors carry any
  causal information, which §2.4.6 gives real reason to doubt.

**The single question the plan would build the phase around** (§6.2): self-repair has
only ever been measured against interventions that *remove a component*. §3.29 records
MLP 6's repair as *"an external field... what it changes is the cost geometry the
coupling is computed in."* A metric patch **is** a change of cost geometry — so
intervention and repair would be in the same channel, which ablation cannot test.
Either answer is publishable and the matched control is obvious.

**Two stale statements found and recorded rather than dropped:**

1. **`archive/p5c_unclustered/status-5c.md` blocker 3 is stale.** "No model in the
   registry has an LM head" — `core/lm_loading.py` exists, is registry-consistent and
   revision-pinned, has two live callers (`p2_eigenspectra/vocab_projection.py`,
   `p7d_redundancy/backup_sweep_full.py`), and supplies
   `load_causal_lm_from_state_dict` for the random twin. **Phase 5c's Group D — the
   force-collapse / force-disperse battery with matched controls, which is exactly
   the experiment Phase 9 wants — is unblocked on this axis and has been for a
   while.** Its other three blockers are not checked here.
2. **Phase 6's LDA inversion has been explained and three documents have not caught
   up.** `p6_subspace/subspace_geometry.py`'s header records
   `dim(U_A)/dim(U_neg) = 24.9` against an observed alignment ratio of **13.2**
   (`claims/audits/p6_projector_labels.json`) — the dimension correction is nearly
   twice the effect it would explain, and the live null holds dimension fixed by
   construction anyway. `math-6.md` §7.2's "until one of these is done, the inversion
   is not evidence", `status-6.md`'s summary and `archive/README.md` rule 3's Phase 6
   bullet all still read as though it were live evidence. **The inversion should stop
   being quoted as a negative on whether cluster identity is encoded.**

Also carried: `P6-R2` is blocked more deeply than `P6-R4` — its second arm is the
antisymmetric subspace and **no such projector exists in any of the 19
`p2_eigenspectra_*` directories**, the rotational channel having been deliberately
not measured. Phase 9 must not assume one exists.

**The ladder (§8) is fourteen rows with costs and dependencies. Three are free and
unblocked today** — the transport observables that no runner calls (§3.39's cheapest
open action, still open), `frame_table.py`'s gamma calibration (sub-experiment D,
written and never run, and it bounds what an in-distribution patch even means), and
a paragraph carrying finding 2 above into the documents that are behind. One more is
a decision rather than a run: `P6-R4`'s exchangeable unit. **Everything involving a
gamma_beta prediction is gated on β's undecided unit convention, which is worth a
factor of 8** (§3.40 finding 2) — named as a blocker to raise, not to route around.

Nothing has been run. `claims/registry.json` untouched. The branch
`claude/attention-collapse-augmentation-qsxwg8` is now redundant: its one commit is
cherry-picked here and it can be deleted.

## 3.46 The v2 battery ran, and `CLAIM-C`'s gate refused on its own calibration (2026-09-19)

Detail in `p1_mstate_tracking/status-1.md`, "The v2 battery ran too". All four
arms re-run on the 21-prompt battery in **4 h 51** — 127 + 63 + 63 + 38 min
against ~2½ h projected, because v2's twelve additions are all long where v1
carried a 115-character prompt and the repeated-token control, and HDBSCAN on a
precomputed distance matrix scales badly with token count.

**The table is now complete and the floor is no longer binding**: 20 prompts,
120 cells, 54 concordant, sign homogeneity 0.875, 240 artifact files hashed.
**And the gate refused for a third distinct reason** — `INSUFFICIENT`,
`hard_stop: true`, no p:

    no homogeneity correction is available, and the correction is what enters
    the e-value: no calibration curve is tabulated for 20 prompts
    (tabulated: [6, 7, 8, 9, 10, 11, 12])

`tools/calibrate_claim_c_homogeneity.py` tabulates to twelve because its own
comment calls that "generous against the eight metastability prompts".
**Extending the battery invalidated the assumption the calibration was built
on** — a coupling nothing in the battery-v2 reasoning (§3.42) anticipated,
because the floor arithmetic and the correction table live in different files
and only the first was checked. The gate refusing here is the right behaviour:
reporting the uncorrected p would assert a Type-I guarantee on a null already
measured to be anticonservative when the prompt sign-rows agree, and at 0.875
they largely do.

**The three refusals are a sequence, not a repetition.** Arms absent → a metric
dead in every arm because HDBSCAN was never installed → the floor unreachable
at eight prompts → the correction untabulated at twenty. Each was fixed and
revealed the next, and **not one of them was a fact about the phenomenology.**
That is worth carrying out of today: the instrument had four independent
defects between it and a number, and they were only ever visible one at a time.

**The fix is prescribed by the gate** ("extend `N_PROMPTS_TABULATED` and
regenerate rather than running uncorrected") and is calibration, not
measurement. Costed on this machine from a 400-draw probe: **~45 min for the
n = 20 row**, ~40 min to regenerate rows 6–12, ~3 h more for a contiguous
13–20. Per-count seeds are `seed + 1000 * i` on the index in the tuple, so
appending after 12 leaves 6–12's seeds untouched and a regeneration must
reproduce them byte-identically — if it does not, that is drift in the gate,
not in this run. **Decided (2026-09-19, user): not tonight.** The refusal is
the committed record; `real_run_record` still points at it.

## 3.45 The e-value audit, Phase 7 — the last unit: `P-I1`'s record written, and a number four documents should not have been quoting (2026-09-19)

Detail in `p7_motifs/status-7.md`. Nine rows, all active — the largest live
phase and the only one whose gates have met real artifacts.

**`P-I1`'s record now exists.** `tools/score_p_i1.py` printed its result and
returned, so the only p-value this project had ever produced against real
artifacts lived in stdout and in prose, with both inputs under git-ignored
`data/`. It now writes `claims/audits/p_i1_real_run.json` — full result, sweep,
seed, replicate count, sha256 of each input — on `score_claim_c.py`'s
record-either-way convention, and `real_run_record` is set. Reproduced today:
**INSUFFICIENT, 116 forming heads, 0 skipped, floor 0.00050, mean distance
2.018 log-step.** `check_registry` refused the entry until the record was
git-tracked ("an artifact only on this machine is not evidence a later reader
can check") — §3.36's rule catching a real case on its first outing.

**Four documents were quoting a number §3.7 says is not quotable.**
`EVALUABILITY.md`, `status-7.md`, `PREDICTIONS.md` and §3.36 all carried
`p = 0.1414` as *the* `P-I1` result. §3.7 measured it at two null sizes —
**0.14143 at K = 50, 0.89355 at K = 100, same verdict** — because 36 heads
share one coset of the relay axis, and said so in terms: "§3.6's p = 0.1414 is
not a quotable number, and neither is 0.8936." All four are annotated rather
than deleted. What they should have carried is the verdict, the mean distance
and the floor, which are robust. **A p-value that moves by 0.75 with the
replicate count is a statement about K.**

**`P-I5` is the registry's one internal contradiction and it is deliberate** —
`needs-null` while naming a gate and carrying a `real_run_record`, so
`check_registry` warns that the gate cannot reach a claim's e-process. That is
wanted: §3.31–§3.34 built the statistic, found the control does not
discriminate, and parked the row with its instrument visible rather than
quietly downgrading it. The warning is a flag, not a defect.

**The four `needs-null` rows without gates are the honest ones** — `P-SA1`,
`P-I2`, `P-I4`, `P-I7` claim nothing the tree does not have, which is the
opposite of 5b/6's four dormant `e-value` rows (§3.44).

---

### The audit, finished: five units, thirty-nine predictions, zero e-values

| phase | rows | what stops them |
|---|---|---|
| 1 | 3 — `CLAIM-A`, `CLAIM-B`, `CLAIM-C` | `CLAIM-C` RAN, three times, refusing three different ways (§3.41, §3.46); `CLAIM-B` cannot reach α on its instrument (§3.43); `CLAIM-A` queued behind `CLAIM-C` |
| 1c | 4 — `P-gamma1/2`, `P-H1`, `P-S1` | inputs no run directory carries — no `beta_eff`, no persisted centroids (§3.40); `P-H1` measured |
| 2d | 2 — `P-T1`, `P-M1` | inputs all present and the join verified; blocked by a design chain and by `P-T1`'s wording (§3.43) |
| 5b | 9 | all dormant; four of them read `e-value` with no gate (§3.44) |
| 6 | 12 | ten dormant. `P6-R2` needs a channel nothing measures; `P6-R4` needs an exchangeable unit nobody registered (§3.44) |
| 7 | 9 | four built, calibrated and unrun; `P-I1` now recorded; `P-I5` parked by design (§3.45) |
| **total** | **39** | **zero e-values** |

*(`CLAIM-B` sits in Phase 1 by the registry's `phase` field but is
instrumented by Phase 2's sweep, which is why §3.43 audits it; Phase 2 itself
registers nothing.)*

**The pattern, stated once.** Across all five units the instruments are in
better shape than the inputs, and wherever the inputs exist the blocker is a
decision nobody has taken: β's scale convention, `P-T1`'s wording, `P6-R2`/`R4`'s
exchangeable unit, `P-S1`'s matched-k clustering. **None of those costs a
forward pass.** The compute this project has spent recently — five `CLAIM-C`
arms twice over — bought one INSUFFICIENT; the decisions above are what the
remaining thirty-four rows are actually waiting on.

**What the audit changed, rather than observed:** `tools/audit_p1c_inputs.py`
and its record; `tools/score_claim_c.py`'s four arms re-run under a declared
HDBSCAN; the `run_1c` β-gate fix; `p_i1_real_run.json` and its scorer;
`p6_subspace/status-6.md`, which did not exist; prompt battery v2; and four
corrections to statements the tree contradicted (§6's rotational-channel claim,
`P-I1`'s p in four documents, `centroids.py`'s persisted-centroids premise,
`lit-1c.md`'s blocked-hosts premise).

## 3.44 The e-value audit, Phases 5b / 6: a prediction whose second arm was never measured (2026-09-19)

Detail in `p6_subspace/status-6.md` (new — the live instrument had no status
file; the frozen phase's history stays in `archive/`). Twenty-one registered
rows, **nineteen dormant, two active** (`P6-R2`, `P6-R4`).

**`P6-R2` is blocked on a channel the project decided not to measure.** Its
statement compares alignment with the real repulsive subspace `U_neg`
*against* the imaginary subspace `U_A`, and `LayerChannels` needs four bases
per layer — `u_pos`, `u_neg`, `u_a`, `u_s`. Across **all 19**
`p2_eigenspectra_*` directories the projector artifacts carry exactly four
array kinds: `{schur, sym} × {attract, repulse}`. That is the **sign channel,
twice**; there is **no antisymmetric, imaginary or rotational subspace
anywhere on disk**. Which is consistent rather than surprising — every Phase 7
manifest writes `rotational_channel: "absent"` and `real_frac`/`imag_frac` are
NaN in every row of every table. §6 examined that absence and concluded no
registered prediction needs the rotational channel. **`P6-R2` is the
counter-example**, and §6 is corrected below. Reviving it means producing the
antisymmetric projectors (`p7_io.rotational_channel_from_blocks` is the named,
unwired seam) or amending the prediction — the second being a registry
amendment.

**Both active rows lack a registered exchangeable unit, and the gate refuses
rather than defaulting.** `EXCHANGEABLE_UNITS = ("model", "layer")`;
`_check_unit` raises because "the two units differ by orders of magnitude in
the p they produce, and a default would silently pick one". The registry names
neither. Same class as `P-T1`'s wording (§3.43): **a registration gap to close
before the gate runs**, since a unit chosen afterwards cannot be told apart
from a unit chosen for its p-value.

**`P6-R4` is the one row in these two phases whose inputs exist today.**
"S-only projection preserves cluster membership" projects onto `u_s`, which
`sym_*` supplies; it does not need `U_A`. Its entry's note — "no p-value is
emitted, because this repository holds no run artifacts and because no
exchangeable unit is registered" — is **half stale**: written 2026-08-24, it
predates the 2026-09-01 sweep that put 19 checkpoints of per-layer projectors
on disk beside the Phase 1 activations and cluster labels an LDA direction
needs. The unit is now the only thing between `P6-R4` and a p-value.

**Four dormant rows are classified `e-value` with no gate and no calibration
record** — `P6-I1`, `P6-I2`, `P5b-B1`, `P5b-C1`. Legal under
`EVALUABILITY.md`'s "a valid null exists **or is directly constructible**", and
all four carry the same `dormant_reason`. But "directly constructible" is the
entire claim, and with the instrument frozen nobody can check it: the
classification is unfalsifiable while the row is dormant, and it inflates the
count of rows that look one step from an e-value. `check_registry` warns when a
row names a gate but reads `needs-null` (`P-I5`); it does not warn on the
converse. Recommended, not done: require a gate for `e-value` unless `dormant`,
or date a note on each of the four naming the construction meant.

**Nothing was scored,** and `P6-R4` in particular was not run: registering its
unit during an audit, with the artifacts to hand, is the ordering the audit
exists to protect.

**The audit is now four phases in, with one left (phase 7).** Thirty-five
registered predictions examined, **zero e-values**, and the pattern across all
four units is the same: the instruments are in better shape than the inputs,
and where the inputs exist the blocker is a decision nobody has taken —
β's convention, `P-T1`'s wording, `P6-R2`/`R4`'s unit.

## 3.43 The e-value audit, Phase 2 / 2d: the first gates whose inputs are all present, and a claim that cannot reach its own floor (2026-09-19)

Detail in `p2d_operator_activation/status-2d.md`, "E-value audit, Phase 2 /
2d". **Phase 2 registers nothing of its own** — `EXPERIMENTS.md` calls it a
measurement programme supplying the artifacts other phases adjudicate on, and
that is right. The unit is therefore `P-T1`, `P-M1` and `CLAIM-B`, the last of
which the registry places in Phase 1 but whose instrument is Phase 2's sweep.

**`P-T1` and `P-M1` are the first registered gates in this audit whose inputs
are all present.** 19 of 19 `p2_eigenspectra_*` directories carry the per-head
`wq_head*` / `wk_head*` / `ov_head*` arrays `load_operators` refuses without,
and the join was run for real on `step143000`: **24 operator layers paired,
16 heads, d_head 64, no refusal**, with both guard warnings firing correctly
(the 24-vs-25 layer count, and `RAW FRAME` because LN parameters were not
supplied — a primary measurement must resolve the frame).

**What blocks 2d is therefore not evidence but two things already written
down.** `design-2d.md` holds the phase behind Phase 1c-B, whose $T_{\rm eff}$
decides whether the energy-monotonicity break is the right thing to attribute
— and 1c-B is blocked on the β producer that does not exist (§3.40). The chain
is **β producer → 1c-B → 2d, and no link in it costs a forward pass.** The
second is a registration defect, the audit's only live one: **`P-T1`'s
registered wording omits half of Table 1's row 2**, which requires
$\langle Q\varphi_1, K\varphi_1\rangle > 0$ as well as $\lambda_1(V) > 0$
simple. The amendment has to land *before* the gate runs; afterwards it cannot
be told apart from fitting the wording to the result.

**`CLAIM-B` cannot reach its own floor, and its gate said so before any data
existed.** Both refusals in `p_value_claim_b`'s docstring, checked against the
sweep on disk: the anchor arms need **19 control series** at α = 0.05 and the
sweep measures **6** metrics; and the registered instrument names a **20–30
checkpoint** cheap-tier sweep while the 410m sweep has **19**. Structurally
worse than either: on this grid a series with **no** located change lands
*inside* the 512–2000 window, so a null reads as co-location — the same
"cannot come down on one side" shape as `CLAIM-C`'s tied rows (§3.41),
arriving by a different route.

**Nothing was scored.** 2d's block is its author's design decision and running
the gates to see what they say is the peek the block exists to prevent;
`CLAIM-B`'s refusal follows from the grid and the control count, which are
properties of the instrument rather than the outcome. No `score_claim_b`
plumbing was written: a scorer that can only ever return "the floor is
unreachable" is a worse artifact than the table in `status-2d.md`.

**The audit's running tally, three phases in.** Fourteen registered
predictions examined; **zero e-values**; one gate run (`CLAIM-C`, INSUFFICIENT,
§3.41); two gates feedable but design-blocked (`P-T1`, `P-M1`); one gate whose
design cannot reach α on its instrument (`CLAIM-B`); three rows blocked on
inputs no run directory carries (`P-gamma1`, `P-gamma2`, `P-S1`); one
measurement row measured (`P-H1`). **Remaining: phases 5b / 6 (19 dormant,
`P6-R2`/`R4` rebuilt and unrun) and phase 7 (four rows built and calibrated,
`P-I1` scored but unrecorded, `P-I5` parked).**

## 3.42 Prompt battery v2: twelve rows added to lift `CLAIM-C`'s floor, chosen blind (2026-09-19)

§3.41's gate stalled on a floor, not on a finding: four of eight rows tied
3–3, a tied row cannot move the statistic, and the smallest p the table could
express was 0.0661. The battery is therefore extended from nine prompts to
twenty-one — `PROMPT_BATTERY_VERSION` `v1` → `v2`, hash `1e47918ef77a` →
`06790b90dcfe`, `CLAIM-C`'s unit count 8 → 20.

**The rule was committed before the text existed** (`33c8710`, then `16a9d3a`),
and that ordering is the point: the prompt is the exchangeable unit, so
prompts chosen after seeing how they behave are selection and a gate built
from them reports nothing. The rule is in `core/prompts.py`'s docstring — six
genres mirroring v1's, two prompts each, one in v1's short band (1000–1600
characters) and one in its long band (1800–2400), sources fixed per genre in
advance, each passage the first meeting its band from a predetermined anchor,
and **nothing may be added, dropped or edited once a model has been run on
v2**. All twelve landed in band; none was run through a model before
inclusion.

| genre | short band | long band |
|---|---|---|
| encyclopedic | `wiki_photosynthesis` 1064 | `wiki_byzantium` 1950 |
| historical letter | `lincoln_letter_short` 1007 | `lincoln_letter_long` 1841 |
| technical exposition | `paper_attention` 1121 | `paper_svflow` 1816 |
| literary narrative | `quijote_capitulo` 1023 | `moby_loomings` 1840 |
| source code | `sklearn_kmeans_code` 1086 | `scipy_linkage_code` 1922 |
| structured markup | `latex_beamer` 1362 | `latex_article` 1901 |

Sources: Wikipedia; Project Gutenberg (Lincoln's *Speeches & Letters* 14721,
*Don Quijote* 2000, *Moby-Dick* 2701); arXiv open access (1706.03762,
2604.23740); BSD-licensed source already installed in the venv
(scikit-learn, scipy); composed LaTeX, as v1's `latex_monograph` is.

**What this buys, in design arithmetic and nothing else** (`attainable_p`, no
data): at 8 prompts with 4 informative rows the floor was 0.06615; at 20
prompts it is 0.0625 with 4 informative, **0.0156 with 6**, 0.00098 with 10.
So the floor clears α = 0.05 as soon as six of the twenty rows can move,
against four of eight observed on v1.

**A floor that can express a small p is not a small p.** v1's six leave-one-out
subsets were never floor-bound and still returned 0.2529–0.9572 against a
concordance of 47.9%. The twelve rows make rejection *possible*; they do not
make it likely, and they were chosen blind precisely so that whatever comes
back counts either way.

**The cost, stated rather than discovered later.** v1 and v2 batteries hash
differently and `verify_same_battery` will refuse to compare across them —
correctly. **Every run on disk before 2026-09-19 is v1**, including Phase 1's
19-checkpoint sweep, so a v2 run is comparable only with other v2 runs. The
four `CLAIM-C` arms are being re-run on v2 for that reason; nothing else has
been, and nothing else needs to be until it is compared with a v2 run.

## 3.41 `CLAIM-C`'s gate runs for the first time, and returns INSUFFICIENT at chance concordance (2026-09-19)

Detail in `p1_mstate_tracking/status-1.md`, "The gate ran, for the first
time". The record is `claims/audits/claim_c_real_run.json`, and
`real_run_record` in the registry now points at it — the gate has been run on
real checkpoints, which is what that field is for, even though it emitted no p
for the conjunction.

**How it got here, one line each.** The four arms were produced (2026-09-17
and 2026-09-19); the gate refused on a metric that HDBSCAN alone produces;
HDBSCAN turned out never to have been installed in this machine's `.venv` and
never to have been declared a dependency; the writer was forging zeros for its
sibling metric, so half the damage read as a valid measurement; all four arms
were re-run with it present in 2 h 20.

**The verdict: `INSUFFICIENT`, `hard_stop: true`, `falsified: false`,
`p_value: null`** — on a table where every prompt is usable and nothing is
dropped. Two reasons, and they say different things:

1. **The full six-metric row cannot express a small enough p.** Six metrics is
   an even number of cells per prompt, and four of the eight prompts split
   exactly 3–3; a tied row contributes the same number to the observed sum and
   to all 256 null patterns, so it is enumerated without ever being counted.
   Four movable rows put the smallest expressible p at **0.0661**, above
   α = 0.05, against the design's own floor of 0.0078 over eight prompts.
   Every leave-one-out subset has five metrics — an odd count, so no row can
   tie — and all eight rows are informative there.
2. **Where a small p is expressible, the data is not near one.** The six
   leave-one-out p-values run 0.2529–0.9572 in the transfer direction and
   0.0778–0.8872 in the inversion direction; nothing clears α in either tail.
   Overall concordance is **23/48 = 47.9%**, which is the coin.

**Per-metric concordance over the eight prompts is the substance:**
`fiedler_mean` **8/8**, `mass_near_1` **7/8**, `cluster_membership` 5/8,
`cluster_count` 2/8, `effective_rank` **1/8**, `cka_prev` **0/8**. The
registered question — does the trained-minus-random contrast transfer from
`gpt2-large` to `pythia-1.4b` — has no single answer on this data: two metrics
transfer almost perfectly, two invert almost perfectly, two sit in between. A
conjunction demanding unanimity across leave-one-out subsets reports
INSUFFICIENT on exactly that shape, which is the instrument working rather
than failing.

**Nothing is adjudicated.** `claims/adjudications/` is untouched and
`--adjudicate` was not passed. The per-metric table is a diagnostic and **must
not be used to re-pick the metric set** — choosing metrics after seeing which
ones transfer is the selection pre-registration exists to prevent. The set
changes by registry amendment or not at all.

**What the hard stop now means.** It fires on INSUFFICIENT, not only on
falsification, so formally "items 9–11 do not proceed" — and §3.36 already
recorded that they proceeded anyway, the whole checkpoint sweep included. The
honest reading is not "the phenomenology fails to transfer", since the
inversion tail does not clear α either, but **"at eight prompts this design
cannot tell, and the six metrics disagree among themselves about which way it
would go."**

**The one thing that could change it is more prompts, chosen blind.** The
floor is set by how many rows can move, so extending the battery (a
`PROMPT_BATTERY_VERSION` bump in `core/prompts.py`) is what lifts the full-set
row off 0.0661 — at the observed rate of four informative rows per eight
prompts, roughly twelve more would be needed, selected without reference to
their contrasts. A looser α would not do it; the gate's own message says
"needs prompts that come down on one side, not a different threshold". And the
leave-one-out rows say those prompts would have to behave very differently
from these eight to move a concordance sitting at the coin.

## 3.40 The e-value audit, Phase 1c: every classification is right and three of the four gates have nothing to eat (2026-09-19)

Detail in `p1c_frames/status-1c.md` "E-value audit, Phase 1c"; the
machine-readable record is `claims/audits/p1c_inputs.json`, written by
`tools/audit_p1c_inputs.py` (+ 7 pure tests) over **all 19 pythia-410m
checkpoint directories, 152 model-prompt directories**. Phase 9's notes
branch numbers itself §3.39, so this is §3.40.

**The classifications hold.** `P-gamma1` and `P-gamma2` are `needs-null` and
have no null; `P-H1` is `measurement` and no valid null exists for it;
`P-S1` is `e-value` with a built, calibrated gate. Nothing needed
reclassifying — which is the audit's least interesting result and worth
saying, because Phase 1's did.

**What the audit found instead is that the inputs are missing, and that this
is cheaper to fix than it looks.**

- **No `geometry.json` in the tree carries `beta_eff`** (0/152), so `P-gamma1`
  and `P-gamma2` have no β to evaluate a null at — `status-1c.md`'s open item
  1, confirmed on disk. But 152/152 carry `attentions.npz` and
  `activations` + `norms`, and β is **derivable from those plus the cached
  checkpoint's LN parameters with no forward pass** —
  `ln_frame.frame_for_hidden_state` → `ln_frame_gram` →
  `beta_eff.estimate_beta_all_heads`, demonstrated at 16/16 valid heads in
  each of 24 blocks, median R² 0.18. The blocker is a producer nobody wrote,
  not a re-run. The one genuine compute item in the phase is `h_attn_only`,
  the frame-correct step-size variant, which needs `run_1.py --sublayer`
  streams that **0/152** directories have.
- **β's unit convention is undecided and worth a factor of 8.** Scaled by the
  model's own `1/sqrt(head_size)`, measured β is median 0.50, IQR
  [0.26, 0.75], range [−0.84, 2.19] over 384 head-rows; unscaled it is 8×
  that on 410m, and `head_size` is 64 on `gpt2-large` against 128 on
  `pythia-1.4b`, so raw slopes are not comparable across `CLAIM-C`'s own
  arms. Decide it before the producer freezes it into an artifact.
- **The measured β range falls outside the interval `beta_reduction.py`'s
  monotonicity was verified on, and monotonicity survives.** Re-checked over
  β ∈ [−1, 2.2] at t = 3: (SA) decreasing, (USA) increasing, zero violations
  at n = 20 and n = 467. The envelope *width* changes a lot — at n = 467,
  t = 3 the (SA) envelope is 0.022 over the measured range against 0.262 over
  the illustrative [0.5, 5] — so the β-reduction question nearly dissolves for
  (SA) and does not for (USA).
- **`P-S1` cannot be fed by any run directory, for three independent
  reasons**, of which the sharpest is that **no run directory carries
  `kmeans_centroids_L*`** — the key `centroids.py::load_centroids` reads for
  the primary arm. The clusterer decision was taken on the grounds that kmeans
  is "the only one whose centroids Phase 1 already persists"; the writer has
  never persisted centroids, including in this week's runs. The HDBSCAN arm
  reads a key in `clusters.npz` that the runner writes to a different file;
  the agglomerative arm works but at m = 83–209 per layer with 30–66%
  singletons. And the gate's (m, d) refusal bites hard: `step143000` and
  `step0` agree on cluster count in **25 of 175 kmeans layer-rows** (2 of 175
  agglomerative). Recomputing centroids offline at a matched k fixes both
  without a forward pass.
- **`run_1c.py` refuses β-free sub-experiments for want of β.** Asked for
  `--subexp E` alone, which uses no β, it skipped all 8 runs and wrote
  nothing. One line at `run_1c.py:257`; found, deliberately not fixed here.

**`P-H1` is measured, for the first time** (`claims/audits/p1c_e_hemisphere.json`).
On `pythia-410m-step143000`, all eight prompts: the cone condition is feasible
at **every layer of every prompt**, the minimum margin is **at layer 0** on all
seven metastability prompts (0.132–0.173) and grows with depth (0.31–0.41 at
the last layer), and those margins sit well above the i.i.d.-uniform reference
for these lengths (0.030 at n = 512). The collapse control `repeated_tokens`
sits at 0.50–0.70. No p-value, no e-value, nothing written to
`claims/adjudications/` — `P-H1` is a `measurement` row by construction. It
reads the checkpoint sweep that ran while `CLAIM-C`'s hard stop was unrun
(§3.36), and inherits whatever that gate eventually says.

**One literature question closed on the way.** `lit-1c.md` is leads-only
because its session's egress proxy blocked every scholarly host. **This
machine can reach them**, and the file's flagged "first thing to check" —
2604.23740's `α · Δτ ≈ 0.025`, which would have preempted 1c-A's central
number — **is not in that paper**: its full text has no "effective step size"
and one "step size", 0.01, belonging to its own synthetic ODE experiment. 1c-A
is not preempted by its nearest neighbour, and the rest of `lit-1c.md` can be
upgraded from leads to readings whenever a session spends the time.

**Five decisions wait on the author**, listed at the end of `status-1c.md`'s
audit section: the β producer, β's scale convention, `P-S1`'s matched-k
re-clustering, the one-line `run_1c.py` fix, and whether to spend forward
passes on `--sublayer` streams.
## 3.39 Phase 9 opens as notes only: intervening on the metric rather than the weights (2026-09-18)

`p9_metric_intervention/notes-9.md`. **Pre-design, nothing frozen, nothing
registered** — `CLAUDE.md` trigger 1 (the scan that runs before a phase's
constructions freeze) has **not** been discharged, and `notes-9.md` §11 names
the searches it still needs.

The phase's object: every intervention this project has run acts on *weights*
(ablation, the §2.4.2 sign factorial, §2.5's isometric path, rank truncation).
Phase 9's would act on **the geometry the dynamics is read in**. `math-1c.md`
§6.2 is why it is implementable without new plumbing — `ln_plain` is *exactly*
sphere projection at constant norm `sqrt(d)`, and what takes the stream off the
sphere is LN's **learned diagonal**, which attention reads directly. The
proposed generalisation is `diag(gamma) -> diag(gamma) + U D U^T`: rank-`k`
anisotropic, one layer, read-side. `MATH_SPECTRAL_OT.md` §6's second spectrum is
the frame that makes it more than a knob — "stretch a subspace" becomes "change
the curvature of a near-zero eigendirection of the Wasserstein Hessian", which
predicts a shift in the implied timescale rather than merely producing one.

**Two stale statements were found while writing it, and are recorded rather than
silently dropped.**

1. **`MATH_SPECTRAL_OT.md` §5.2's "Nothing in the repository computes it" is no
   longer true.** `core/dissipation.py` implements `energy_gradient` and the
   full first-order identity, and §3.8.2 already reports it **run** — Tier A and
   Tier B over the 19 x 7 grid. The gradient is available as a scoring function
   today.
2. **The transport half of that same module has never been run.**
   `w2_identity`, `w2_optimal`, `sliced_w2`, `wasserstein_arc_length` and
   `straightness` are implemented and tested, and **no runner in `tools/run/`
   calls any of them** (verified by grep: only the module and its test mention
   them). No `W_2`, arc length or straightness appears in `PROJECT.md` or
   `docs/`. **That is the cheapest open action in the phase** — no forward pass,
   scipy already a dependency — and the identity-vs-optimal gap is the only
   instrument here that separates *tokens swapping places* from *genuine motion
   of the measure*.

Also derived and worth carrying out of the phase: the cone condition cannot fail
for `n <= d` in general position (Caratheodory; `math-1c.md` §7.2's Wendel note
is the same fact from the other side), so the hemisphere lever is **gated on
context length** — unreachable at `d = 1024` with the current prompt grid,
reachable on pythia-70m (`d = 512`, 2048 window). And Lemma 6.4's "only
positivity of `a_ij` is used" has a corollary the phase leans on: **the QK
circuit is powerless against collapse-from-a-hemisphere; resistance must come
from `V` or from outside the paper's model.**

Nothing has been run. `claims/registry.json` untouched.

## 3.38 Review corrections after the stack merged: `P-I5`'s statistic, five overclaims, and a red CI (2026-09-17)

CodeRabbit's pass over the merged stack, worked through finding by finding
and verified against the code before anything changed. One is
substantive, the rest are wording — but wording that overclaimed.

**`P-I5`'s statistic was wrong for the claim it serves (§3.31 correction).**
The min-rank (Tippett) combination controls the COMPLETE null only. The
prediction is a conjunction, so its null is the union, and on the
falsifier's own configuration min-rank rejected at 0.17 — which §3.31 had
been calling "what a correctly-calibrated test should do", i.e. a Type-I
rate misread as a power trade. Replaced by `intersection_union_pvalue`
(max of the two axes' exact sign-flip p's; `CLAIM-C`'s device), calibrated
on all three arms of the union null at 0.002 / 0.039 / 0.033
(`p_i5_joint_null.json` schema 2). The real-activation scripts report both
statistics; `L3H6` re-scores 0.0234 → 0.0312; the 2026-09-16
negative-control records lack per-prompt deltas and cannot be re-scored
(the scripts now store them). Registry: dated addendum in `P-I5`'s notes,
`gate` repointed; still `needs-null`, still parked.

**Overclaims corrected in §3.32–§3.35, `POPPER_PLAN.md` §6x–§6za and the
modules' docstrings.** (1) The random-vs-random draw was one observation,
not a validation of the pipeline — only its two-sided reading means
anything (0.031 min-rank, 0.156 intersection-union), and "rules out a
pipeline defect" is withdrawn. (2) "Mean-ablation of essentially any head
beats isotropic noise" is a claim about `L4H6` and `L5H3`, the two heads
measured. (3) `delta_logit` under the constant-substitution diagnostic is
negative on seven of eight prompts, not "every single prompt". (4) That
diagnostic runs on `L3H6` alone — it is a readout observation, not a third
failure to discriminate; the two head comparisons are §3.33's isotropic
control and §3.34's other-head direction. (5) The hook replaces the head's
64-dim output slice BEFORE `attention.dense` mixes heads, so the projected
constant reaches every residual dimension; the cancellation argument
survives (same vector at both positions) but "the other 448 dimensions
untouched" was wrong. (6) §3.35's endpoint is `(M_r)^T`, the transpose of
the rank-r truncation at r = 1, 8, 16, and the alignment conclusion is
rank-specific; restoration is checked to tolerance (rel error < 1e-10 on
the OV product, NLL within 1e-6), not "exactly" or "bit-for-bit".

**Code.** `core/isometric_path.py` has one `MIN_SIGMA` threshold shared by
`polar_frame` (squared, against eigenvalues) and `check_refusal`, so
preflight cannot pass what `build_M_t` refuses; the sweep runner caps
"full rank" at `d_head` and rejects ranks outside `1..d_head`; the
calibration tool's exception handler now covers only the optional import
and tokenizer load, so a measurement defect stops calibration instead of
reading as an absent cache; a test's observed statistics come from the
all-positive enumeration row rather than `sum()`, avoiding a one-ULP
mismatch. **CI on `main` had been red since #51**: forty runner scripts
raise `SystemExit` at import unless the interpreter is this machine's
`.venv`, and a pure test imports one of them. The guard is now
script-time only (`__name__ == "__main__"`), which keeps the wrong-env
protection for runs; the pure tier passes under the system interpreter.
Two markdown chores: `POPPER_PLAN.md` §6x–§6za headings on one line, and
the merge's duplicate `## 6y` renumbered `6zb`.

## 3.36 The e-value audit, phase by phase: the registry now carries evidence paths, and `EVALUABILITY.md` holds current state only (2026-09-17)

The user asked for a pass over every phase checking that its e-values are
valid and logged. The first fact the pass turned up is that **there are no
e-values**: `claims/FALSIFICATION.md` is empty, no prediction has been
adjudicated, and the one p-value ever produced against real artifacts
(`P-I1`, §3; its p is K-dependent and not quotable, §3.7) had no committed record at the time — it lived in git-ignored
`data/analysis/relay_null_series.json` and in prose. So the audit is of
*classification and evidence*, not numbers, and the tooling was changed so
that audit is mechanical rather than a re-read of a 1,400-line file.

**Four changes, one PR.**

1. **`claims/registry.json` carries four new fields per entry.** `phase`
   (INDEX.md's directory family — the registry was not organised by phase and
   the `instrument` prose was the only way to map it) and three evidence
   fields, `null_module`, `calibration_record`, `real_run_record` *(`null_module`
   was replaced by the finer `gate` on merge, see the addendum below)*, each a
   **git-tracked repo-relative path or null**: the live module emitting the
   p-value, the known-answer artifact that checked it, the committed record of
   a run on real checkpoints. Paths rather than booleans because a path can be
   checked — `tools/check_registry.py` fails when one is missing or untracked,
   when a `real_run_record` is set with no `null_module`, or when a
   `measurement` entry names a null. `tests/test_check_registry_evidence.py`
   (6, pure) covers the rules. Counts on the tip: **12 nulls built, 12
   calibrated, 1 run on real artifacts (`P-I5`, exploratory), 0 adjudicated.**
2. **`EVALUABILITY.md` has a generated "By phase" section** — one table per
   phase with those paths as cells. Phases 1b, 2b, 7d, 7e and 8 have no rows:
   exploratory by design, nothing there may carry an e-value.
3. **The construction diary moved out.** `EVALUABILITY.md` was ~150 lines of
   generated table and ~1,250 lines of thirteen dated passes, each ending in a
   "What the pilot must produce, after N" table that superseded the previous
   one — the session-diary failure mode `CLAUDE.md` names for this file, and
   the thirteen tables had been read as a queue wrongly three times by the
   file's own account. Those sections are now `claims/EVALUABILITY_LOG.md`,
   verbatim (checked line-by-line: only three lines of the original are absent
   from the pair, and those are the ones deliberately reworded). What stayed:
   the three states, the recurring patterns, a new **"The order to build a
   null in"** distilling the thirteen passes' four defect kinds and the
   floor-first prescription, and a single **"What is next"** that is replaced
   rather than appended.
4. **Every live `status-N.md` opens with a "Registered predictions" block** —
   the phase's ids, their state, and how far each null has got — so the phase
   and the registry know about each other. Phase 1's status file had not
   mentioned `CLAIM-C` or `replication_gate.py` at all, though the gate lives
   in that directory.

**What the by-phase view says, read in order.** Phase 1: `CLAIM-C`'s gate
is built and calibrated, never run on the Pythia sweep that Phase 1 is
"Complete" on. Phase 1c: never run against Pythia at all; `P-S1` is the only
row with a null. Phase 2 / 2d: `CLAIM-B`, `P-T1`, `P-M1` built and
calibrated, unrun — and `P-T1`/`P-M1` share an instrument under one claim,
which their product would not show. Phases 5b / 6: 19 dormant, `P6-R2`/`R4`
rebuilt and unrun. Phase 7: four e-value rows built and calibrated, `P-I1`
scored but unrecorded, `P-I5` parked. `claims/registry.json`'s frozen fields
are untouched.

**Phase 1, audited (same day; PR #47).** Detail in
`p1_mstate_tracking/status-1.md` "E-value audit". Two findings, one decision each:

- **`CLAIM-C` cannot run on anything in the tree.** Its gate was built and
  calibrated but had no plumbing from a run directory; `tools/score_claim_c.py`
  (+ 4 pure tests) now assembles the five registered arms and writes a record
  either way. Against all 19 Phase 1 run directories it refuses — the tree
  holds pythia-410m only; `gpt2-large`, `gpt2-large-random`,
  `pythia-1.4b-step143000`, `pythia-1.4b-random` have never been run
  (`claims/audits/claim_c_real_run.json`). Cost to produce them: two HF
  downloads (offline mode currently forbids) and five `run_1.py` model runs,
  ~6–8 h. **The hard stop was bypassed de facto** — Phase 2's rerun and
  Phase 7's sweep are the work it gated. Decision: spend the compute, or
  record that the gate is unrun and the sweeps stand as ungated.
- **`CLAIM-A` has no null and its 410m step-0/8 reading has been seen**, so a
  null must be calibrated on known-answer inputs and run on pythia-1.4b steps
  0/8 (unseen; the same download). Recommended and NOT built: intersection-
  union max over three per-criterion one-sided p's (the prediction is a
  conjunction; `CLAIM-C`'s own device), unit = prompt, floor 0.0078 at eight
  prompts. Decision: build it now, or leave `needs-null` and move to Phase 1c.

**Decisions taken (2026-09-17, user; detail in `status-1.md` "Decisions
taken").** `CLAIM-C`: spend the compute, one arm per `run_1.py` invocation
(arms are atomic to the scorer; a kill costs one arm). `CLAIM-A`: leave
`needs-null`, queued behind `CLAIM-C`'s verdict. **Two of five arms produced
in a 3 h window:** `gpt2-large` (65 min, `data/phase12/2026-09-17_14-51-40`)
and `gpt2-large-random` (33 min, `2026-09-17_16-04-54`); the 6–8 h estimate
for all five was high for these and about right for the 1.4b ones (~2 h each,
one per 3 h window). `claims/audits/claim_c_real_run.json` now refuses only
on `pythia-1.4b-step143000` / `-random`; all three 1.4b revisions are cached
in `data/hf`, so the remaining arms run offline (commands in `status-1.md`).
The arms' contrast has not been read. **Defect:** `gpt2-large-random`'s
`MODEL_CONFIGS` entry named no base repo, so `load_model` asked the Hub for
`gpt2-large-random` and the arm had never run through this code path;
`hf_repo` added to it and to `albert-base-v2-random`, with
`tests/test_random_controls_name_base.py` (smoke) guarding the invariant.

**Next unit: Phase 1c** (`P-gamma1`, `P-gamma2`, `P-H1`, `P-S1`) — *done, and so is the rest of the audit; see §3.45.* The three
1.4b arms are compute to schedule, not a unit of reading work.

**Addendum, 2026-09-17 — reconciled with the parallel phase-map (§3.37).**
A session on 2026-09-16 had built the same join independently on
`claude/phase-literature-review-ntd772`: `phase`, `experiment` and a `gate`
(`module:function`, resolved textually by `check_registry`) per entry, and
`claims/EXPERIMENTS.md` as the generated phase → prediction → gate view —
alongside sixteen per-phase literature reviews. Merged rather than chosen
between: **their `phase` wins** (`CLAIM-B` is Phase 1's item-8 checkpoint
pilot, not Phase 2 — this file had it wrong), **their `gate` replaces
`null_module`** (finer, and `P-S1`'s now points at `centroids:p_value_p_s1`
rather than the baseline module), `P-I5`'s built-but-invalid gate stays named
(a warning, by design), my `calibration_record` / `real_run_record` survive
and are now columns in `EXPERIMENTS.md`'s map. **One per-phase view:**
`EXPERIMENTS.md`; `EVALUABILITY.md`'s "By phase" section is a pointer to it.
Their section collided with an existing §3.16 and is renumbered **§3.37**.
Their `check_registry` also warns that `H-BUDGET` is declared in `CLAIMS.md`
with no prediction naming it — true today, and now visible.

## 3.35 §2.5's isometric path on `L7H8`, run for real: a genuine asymmetry between `t=0` and `t=1` (2026-09-16)

Following the check-in at §3.34's close, the next session was pointed at
the alternative already queued: §2.5's isometric path
(`MATH_SPECTRAL_OT.md` §2.5), the only *designed* particle intervention
in this project and unrun until now. `tools/run/isometric_path_sweep.py`
+ `core/isometric_path.py` (pure math, split out so it's testable without
torch) + `data/analysis/isometric_path_L7H8_step4000.json`. Full
derivation in `MATH_SPECTRAL_OT.md` §2.5.6; short version here.

**What it is.** `L7H8`'s OV operator `M = U Σ V^T` (thin SVD). A path
`γ(t)` from `U` to `V` (closed-form polar retraction of the chord) gives
`M(t) = γ(t) Σ γ(1-t)^T` — an EXACT ISOMETRY at every `t` (same singular
values, same rank, same energy as `M`), with `M(0) = M`, `M(1) = M^T`,
`M(1/2)` symmetric PSD. Sweeps `L7H8`'s 100%-repulsive spectrum through
100%-attractive and back, at matched magnitude — the causal test the
naive arithmetic-mean pilot (§3.12-M) could not give, because a
rank-`k` head cannot write its own symmetric part.

**Full rank refuses; truncated ranks don't.** `k=64` fails exactly at
`t=0.5` — `L7H8`'s tail singular values are numerically zero (consistent
with §3.11's own `r*=1`), so `Y(0.5)` is singular in those directions.
Verified directly: rank ≤ 32 clears the refusal comfortably. Ran at ranks
1, 8, 16 (no single truncation is obviously "the" fair one here) rather
than picking one.

**The finding: `t=0` and `t=1` are NOT symmetric, though they have
IDENTICAL singular values by construction.** Second-copy NLL (baseline
0.78) stays near baseline through `t~0.3`, rises to `t=0.5` (0.91-0.93),
peaks around `t~0.8` (~1.05), and **stays elevated through `t=1`**
(1.04-1.06) rather than recovering — all three ranks (1, 8, 16) agree.
Cross-checked against an independently-built `(M_r)^T` — the transpose of
the rank-r truncation, since full rank is refused — bypassing the path
construction entirely: equal NLL to the printed precision. **At each
measured rank, the read/write alignment — which subspace is read vs.
written — is doing real causal work, not merely the spectral sign.**
`(M_r)^T` preserves the truncated spectrum (100% repulsive, same singular
values) while swapping that alignment, and copying stays broken. The
conclusion is rank-specific; it is not measured at full `d_head`.

This is exactly the ambiguity `MATH_SPECTRAL_OT.md` §2.5.3 flagged as
unresolved by this one path ("shows WHETHER the spectral character is
load-bearing, not WHICH of the two [symmetry, alignment] carries it") —
now measured, not just named. It rules out the naive "sign alone
explains it" reading; separating symmetry from alignment cleanly needs
§2.5.4's second family (`M_R = U R Σ V^T`, holding both subspaces fixed,
rotating only the correspondence) — named, not built here.

Every run: written-back OV product within relative error `1e-10` of the
original (measured `0.00e+00`) and the re-measured second-copy NLL within
`1e-6` of the first baseline, at every rank — tolerance checks on the
product and one scalar, not exact weight equality. Exploratory — no `P-*`
id names this curve, `claims/registry.json` is untouched.

---

## 3.34 `P-I5`'s joint null, part four: two more constructions, two more failures — the problem moved from the control to the readout (2026-09-16)

§3.33 named the fix (draw the control from real structure, not isotropic
noise) but didn't build it. This builds it, and a second, sharper
diagnostic. The first is a head comparison (all three heads) and fails
to discriminate `L3H6` from uninvolved heads; the second is run on `L3H6`
alone and is a readout observation, not a discrimination test, with a
mechanistic cause. Full derivation `POPPER_PLAN.md` §6za. (p-values below
are min-rank; see §3.31's 2026-09-17 correction.)

**Other-head-direction control (magnitude-matched, structured): still
fails.** `L4H6` p = 0.0078, `L5H3` p = 0.0156, `L3H6` p = 0.0234 — same
pattern as the retired isotropic control.

**Constant-substitution diagnostic (`L3H6` only): `delta_geometric` ≈ 0
for every prompt, mechanistically, not as a null finding.** The hook
replaces the head's 64-dim output slice in the input to `attention.dense`,
before head mixing, so the substituted constant is projected through
`W_O` and can reach every residual dimension — but substituting ANY
constant, the head's own mean or a donor's, makes that slice IDENTICAL at
both members of a matched (query, key) pair, so its projected contribution
is the same vector at both and cancels exactly in their pairwise distance
regardless of which constant was used. `raw_distance` on the
ablated layer's own residual stream cannot see "the right constant" vs
"a wrong constant" under this intervention type. (§3.33's and this
session's magnitude-matched constructions displace each position
differently rather than unifying them, so they don't hit this exact
cancellation — which is why they show *a* signal, just not a specific
one.) `delta_logit` under this diagnostic runs the WRONG way for `P-I5`:
negative on seven of eight prompts, negative on average — the donor swap
is MORE disruptive than the target's own mean on all but one prompt —
plausibly foreignness, not induction-relevance.

**The open problem now has two parts.** What control isolates directional
relevance (§3.33's framing), and whether `raw_distance` on the ablated
layer's OWN residual stream has the sensitivity this test needs at all.
A geometric readout at a LATER layer — downstream of where the ablated
information would need to propagate through further mixing — is the next
candidate, named not built. `claims/registry.json`'s `P-I5` entry is
unchanged.

---

## 3.33 `P-I5`'s joint null, part three: the validation found a real problem — the control does not discriminate (2026-09-16)

§3.32's closing section named five gaps before its `L3H6` reading
(p = 0.0234) could be trusted. Closing them found the headline result:
**the pipeline does not discriminate `L3H6` from heads with no
relationship to induction — §3.32's reading is not yet evidence for
`P-I5`.** Full derivation in `POPPER_PLAN.md` §6z.

**Negative controls fail.** `L4H6` ("below the +0.05 print threshold" in
`status-8.md`'s cascade table, layer 4 not layer 3) gives `joint_rank_pvalue`
**p = 0.0039 — the exact floor**, more extreme than `L3H6`'s own 0.0234.
`L5H3` (arbitrary, final layer, named nowhere as induction-related) gives
**p = 0.0391**, same order of magnitude. Two heads with nothing to do with
induction pass the gate `L3H6` passes.

**One random-vs-random draw (neither arm real; seed 100 vs 200), reported
as what it is.** Both arms random means no predeclared direction, so only
the two-sided reading counts: min-rank 0.031, intersection-union 0.156
re-scored; the one-sided 0.930 says nothing either way. One draw is one
observation from the null, not a calibration of the real-activation
pipeline — that needs repeated seed-pair draws, not run. (Corrected
2026-09-17 in review; an earlier wording read the one-sided value as
clearing the statistic and the pipeline.) What the two controls measured
show: mean-ablation of `L4H6` and of `L5H3` each beats an ISOTROPIC random
direction of matched magnitude as decisively as `L3H6`'s does — plausibly
because a real, trained direction is structured, and concentration of
measure in `d_head = 64` dimensions makes uniform random noise generically
near-orthogonal to whatever a downstream reading is sensitive to. Whether
that holds for heads generally is not measured.
"Matched-magnitude random-direction ablation" (the registry's own phrase)
matches the norm but not the thing that actually needs to be null:
directional relevance, not mere directionality.

**Seed sensitivity, checkpoint replication (`step64000`), and a
`cosine_distance` cross-check all landed as expected — stable, but that
answers "is the reading stable," not "is it specific," and specificity is
what's missing.**

**Next step, named not built:** redesign the control so the random
direction is unstructured RELATIVE TO WHAT THE READING IS SENSITIVE TO —
e.g. drawn from other heads' own output directions, not a fresh Gaussian
draw. `claims/registry.json`'s `P-I5` entry stays unchanged; a power
analysis is moot until the control measures what it claims to.

---

## 3.32 `P-I5`'s joint null, part two: the control, real activations, a first reading — exploratory, not an adjudication (2026-09-16)

`p7_motifs/p_i5_ablation.py` builds §3.31's missing piece: the
matched-magnitude random-direction ablation control, on real cached
`pythia-70m` activations. Full derivation in `POPPER_PLAN.md` §6y; short
version here.

**Target: `L3H6` at `step143000`** — `status-8.md`'s own table names it
pythia-70m's induction/matcher head (the `L7H8` analogue), mean-ablated per
the resume block's 8-heads-per-layer rule, via
`tools/run/induction_rank_sweep.py::ablate_heads` (reused, not
reimplemented). **The control, defined here for the first time in this
project:** same per-position displacement magnitude as the real
mean-ablation, one fixed random direction per (prompt, ablation site).
Geometric readout: `pairwise_geometric_reading`'s `raw_distance` on the
matched (query, key) pair, read right after the ablated layer. Logit
readout: `core.intervention.next_token_kl` at position `query - 1` (where
the copied token's own prediction lives, per HuggingFace's `logits[i]`
predicts `token[i+1]`).

**First reading, all 8 informative prompts: `joint_rank_pvalue`
`p = 0.0234`** (floor `0.0039` — not floor-saturated). 7/8 prompts positive
on the geometric axis, 6/8 on the logit axis. *Re-scored 2026-09-17 under
the intersection-union statistic that replaced min-rank (§3.31's
correction): **p = 0.0312**, geometric axis 0.0039, logit axis 0.0312 —
the logit axis binds.*

**Explicitly a first look, not an adjudication.** `claims/registry.json`'s
`P-I5` entry is unchanged; `claims/adjudications/` stays empty. Missing
before this could be trusted the way `P-AB1`/`P-I3`'s finished gates are:
a positive/negative control pair, more than one checkpoint, a sensitivity
check on the random draw, a power analysis, and a cross-check against
`cosine_distance`/a projector-restricted reading. `tests/
test_p_i5_ablation_smoke.py` (7 tests) is run for real against the cached
checkpoint, not left unverified.

---

## 3.31 `P-I5`'s joint null, part one: the floor, a construction that did not hold, and its fix (2026-09-16)

STEP ONE-B from §3.30's close, taken as far as it goes without touching a
real activation. Full derivation in `POPPER_PLAN.md` §6x and
`p7_motifs/p_i5_gate.py`'s own docstring; this is the short version.

**The obvious joint statistic (an AND-corner: both axes' sign-flip sums
beat the observed) does not hold.** Run on synthetic data under a TRUE
joint H0 — both `delta_geometric` and `delta_logit` pure independent
noise, zero effect on either — it rejects at 0.18–0.21 against a nominal
0.05 across n = 6, 8, 10, 12 (`claims/calibration/p_i5_joint_null.json`).
Found the way P-ST1 and P6-R2/R4's retired nulls were: by running it on
inputs whose answer is known, not by inspection. Mechanism: intersecting
two independently-derived "at least this extreme" sets does not preserve
the single-dimension exchangeability argument that makes a plain sign-flip
test valid.

**The fix: rank by the weaker axis, not the intersection.** A
Tippett-style minimum-rank statistic — `min(rank_geometric, rank_logit)`
per sign pattern, ranks via `scipy.stats.rankdata` — is a genuine scalar
function of the sign pattern, so the standard exchangeability argument
applies directly. Calibrates correctly: 0.040–0.049 at nominal 0.05 across
the same four n. On the falsifier's own configuration (real logit effect,
no geometric effect), it also does what the AND-corner was meant to: the
logit axis alone rejects at 0.79, the joint statistic on the same draws
rejects at 0.17.

**Corrected 2026-09-17 (review): that 0.17 is an uncontrolled Type-I
rate, and min-rank is not `P-I5`'s statistic.** `P-I5` is a conjunction
(both effects required; either missing falsifies), so its null is the
UNION of the two axes' nulls and the falsifier's configuration is a point
in it — a Tippett minimum is the right combination for "at least one
effect" and the wrong one for "both". The statistic is now the
**intersection-union test** (`intersection_union_pvalue`: each axis's own
exact sign-flip p over the joint pattern space, reported as their max;
valid under the union null with no correction — `CLAIM-C`'s own device).
Calibrated on all three arms of the union null at n = 8: **0.002
(complete) / 0.039 (geometry-null, logit effect) / 0.033 (logit-null,
geometric effect)**, where min-rank reads 0.171 on the second
(`p_i5_joint_null.json` schema 2). The real-activation p-values in
§3.32–§3.34 were computed with min-rank; `L3H6`'s re-scores 0.0234 →
0.0312 from its stored deltas, the negative-control records lack deltas
and cannot be re-scored without a rerun (the scripts now store them and
report both statistics). `claims/registry.json`'s `P-I5` entry is still
`needs-null` and unchanged. Full derivation `POPPER_PLAN.md` §6x.

**The measurement grid, real, not synthetic: `core.battery_structure.
induction_candidates` against `core.config.PROMPTS` under the actual
cached pythia-70m tokenizer.** Per-prompt matched-pair counts run 1 to
2873, with `repeated_tokens` at 34,191 (cross-validates against the figure
`p7_motifs/motif_alphabet.py` already quotes for that prompt).
`repeated_tokens` excluded per the `P_I1_DOMINANT_PROMPT` convention.
**Design choice, put to the author before the control is built (matching
how `P-AB1`'s and `P-I3`'s own matching-unit choices were registered): the
exchangeable unit is the PROMPT** (pairs inside one prompt share a forward
pass and would share one random-direction draw), giving **n = 8** and
floor `1/256`.

**What is still not built:** the matched-magnitude random-direction
ablation itself — wiring `core/intervention.py`, running it on the cached
checkpoints, and scoring with `pairwise_geometric_reading` /
`core/intervention.py::next_token_kl` to get real
`(delta_geometric, delta_logit)` arrays. Nothing here has touched a real
activation; `claims/registry.json`'s `P-I5` entry is unchanged.

---

## 3.30 The pairwise geometric field lands; `P-I5`'s gate is next (2026-09-16)

§3.29's STEP ONE is done: `core/dual_reading.py` has
`pairwise_geometric_reading(vector_a, vector_b, projectors)`, the field
`P-I5`'s `null_construction` named as its blocker ("every current geometric
field is per-point and this needs a pairwise one"). Schema written first in
`core/DESIGN_dual_reading.md` per that doc's own rule, then implemented,
then tested — `tests/test_dual_reading.py` gained 8 cases (27 pass total in
the module), gate green.

**Design: substitution, not new machinery.** Every existing per-point field
is `_squared_norm_frac` of a vector against a subspace; the pairwise field
is the same function applied to the displacement `vector_a - vector_b`
instead — `raw_distance`, `cosine_similarity`, `cosine_distance`, and four
subspace fractions (`distance_attractive_frac`, `_repulsive_frac`,
`_real_frac`, `_imag_frac`). Cross-checked directly against
`geometric_reading(a - b, ...)` rather than trusting a re-derivation
(`test_matches_squared_norm_frac_of_difference`). Kept separate from
`dual_reading()`/`geometric_reading()` (would make the combined
entry-point's signature ambiguous about which mode it's in), not
vectorized over many pairs (the caller loops over matched positions, same
division of labour as `effective_rank_contribution` and
`member_subspace_geometry.py`'s `cell()`), and not wired into
`ParticleTable` (pairwise has no single row to attach to).

**What this does NOT do, so it isn't mistaken for more than it is.** `P-I5`
is still `needs-null` in `claims/registry.json` — untouched here. This
lands the *reading*, not the permutation-null gate over the
matched-magnitude random-direction ablation arm on the joint (geometric
delta, logit delta) statistic. That gate is the next action, and now has
every primitive it needs: `core/intervention.py` (both the ablation and
the logit half — `next_token_kl` — corrected 2026-09-16: this line and
this section's own "So step one is" paragraph below previously
misattributed `next_token_kl` to `core/functional_distance.py`, a
different pairwise-KL primitive for clustering), and this extension (the
geometric half). Building it is the first thing that would put the
project's first adjudication in reach.

---

## 3.29 DIRECTION (2026-09-13, user) — the programme is the particle/OT reading; induction heads are an instance of it

**Read this before choosing an action.** Stated by the user at the close of the
session, and it reframes everything above it.

**The goal is the particle interpretation of transformers**, via the
mathematical perspective and optimal transport, applied to mechanisms the
network actually builds. Induction heads are the *current instance*, not the
object. The same pass — **why did this form, what were the mechanisms of its
forming, what happened after it formed** — is to be run on SAE feature
decompositions, on the clusters themselves, and on other objects, building a
developmental trail from the random-matrix start. The standing hazard is
**depth in one place against sparseness everywhere else**, and this session is
an example of it.

**Honest accounting of §3.20–§3.28.** Nine commits, essentially all head-level
causal mechanism on one model, three of them instrument corrections. **None of
it is expressed in particle or OT language.** It is good plumbing and it drifted
from the programme. Past the point of diminishing returns on this one object:
the next session's job is to make the *second* object cheap, not the first one
more complete.

**The particle/OT translation of what was found**, recorded so the work is not
lost when the frame changes. An attention head **is** a transport plan —
row-stochastic, queries onto keys — which is what the softmax produces, not a
metaphor:

- **The relay is a translation.** `L5H2` at 0.95 on offset −1 is transport by
  fixed displacement, the simplest OT object available.
- **The matcher is a near-Monge map.** `L7H8` at 0.934 on offset `j`
  concentrates nearly all mass on one target per query — an almost
  deterministic map, not a diffuse plan.
- **MLP 6's repair is an external field, not an interaction.** It is
  position-independent, so it cannot change the pairwise coupling; what it
  changes is the **cost geometry** the coupling is computed in (it projects
  into the keys' read subspace, §3.24). §3.23's result restates as: the field
  must point the right way in that geometry, and a field of equal magnitude in
  the wrong direction is **worse than no field** (8.955 vs 8.002 nats).
- **Invariant 4 is already the cluster story.** "Born aligned, then diverging to
  a mid-training minimum" (§3.19), with 70m re-cohering, is cluster formation
  and dissolution measured inside one causally-defined set. No new phase needed
  for that link.

**Instrument note, cheap and worth doing: W₁ over TV.** §3.25–§3.27 measure
attention change with total-variation distance. TV says *the plan changed*;
**Wasserstein-1 on the position axis says the plan changed by moving mass this
far**, which for induction is the whole mechanism — mass leaking to `j ± 1`
versus scattering uniformly are different events and TV cannot separate them.
Same attention tensors, different reduction, and it puts the readout in the
frame's own language so later objects inherit it. It would also have separated
70m's `L0H3` from a real matcher more sharply than TV did.

**Registration discipline, decided here.** The near-miss in §3.27/§3.28 was
registering an **instance-level** claim ("this relay supports this set"). No
account on the table predicts otherwise, so the null does no work and a
reserved rung buys a fact rather than a discrimination. **What the particle/OT
frame can predict differentially is *shape*** — the trajectory of a transport
plan's concentration, the attractive/repulsive split at formation, the
dissipation signature at onset. Three tiers, held strictly:

1. **Exploratory, labelled, no p-value.** Surveys, catalogues, "what is this
   object". Most work lives here; the repo already does this well.
2. **Registered frame-level predictions.** Rare, differential, with a null a
   rival account could beat. Reserved rungs (1b, 1.4b) spent only here.
3. **An e-process across objects.** One frame-level prediction carried across
   induction heads → SAE features → clusters, e-values multiplied — anytime
   valid, and it solves the multiplicity problem that 40 separate entries
   creates. `claims/EVALUABILITY.md` is the authority on what may enter the
   product and is **generated from the registry, never hand-edited**. §4's κ/α
   analysis points the same way: a single-factor claim needs p ≤ 6.25e-4 to
   validate at the e-threshold, 80× stricter than the gates check — so few,
   sharp, multiplied.

**39 registrations, 0 adjudications.** The marginal value of a 40th is low and
of the first adjudication is high. **Do not register §3.27** — not primarily
because CoAx reached the statistic first (§3.28), but because it is not
differential.

### The next action, and it is already registered: unblock `P-I5`

`P-I5` is the particle frame's own differential test and it is sitting in the
registry unbuilt. Its statement: *"Ablating an induction head changes the
pairwise-distance distribution among the particles it couples (the matched
positions), not only the logit at the copied token."* Its falsifier is the
frame itself: *"a large logit effect with a pairwise-distance change
indistinguishable from the matched control: the head moves the readout without
moving the particles, **and the transport reading is wrong**."* Adjudicated on
the **joint** outcome — a geometric effect with no logit effect falsifies it
equally, in the other direction.

It is `needs-null`, and the registry states the blocker exactly:

> Permutation null over the matched-magnitude random-direction ablation arm, on
> a two-dimensional statistic (geometric delta, logit delta). The joint form
> matters: two separate one-dimensional tests would let the prediction be
> scored a partial pass in the configuration it is designed to rule out.
> **REQUIRES an extension to `core/dual_reading.py` — every current geometric
> field is per-point and this needs a pairwise one.**

**So step one is: give `core/dual_reading.py` a pairwise geometric field.**
Verified 2026-09-13: the module has no pairwise function, and
`core/intervention.py` (the ablation, and `next_token_kl` for the logit
readout — corrected 2026-09-16: previously misattributed the logit readout
to `core/functional_distance.py`, a different pairwise-KL primitive built
for clustering, not this) already exists. That single extension unblocks the null, which
unblocks the adjudication, which would be the project's **first** — and it is
the bridge that makes every later object (SAE features, clusters) expressible
in the same language instead of needing its own bespoke plumbing.

**Second, and now better set up than it was:** §2.5's isometric path on `L7H8`
(`M(t) = γ(t)Σγ(1−t)ᵀ`, exact isometry, repulsive → attractive → repulsive) is
the only *designed* particle intervention and is still unrun. This session made
it more readable, not less: `L7H8`'s matching is now known to depend on the
relay plus MLP 6's background field, so the sweep has a mechanism to be read
against rather than a black box.

---

## 3.28 The pre-registration scan: §3.22's statistic is published, and the MLP result is in the gap (2026-09-13)

Full record in **`docs/literature_scan_2026-09-13.md`**; every id was fetched
and read as full text, two of them by pulling the PDF and extracting locally
because the abstract pages answered none of the questions that mattered. Run
under `CLAUDE.md` **trigger 2** — an entry was about to be proposed off §3.27
and registration freezes the wording, the statistic and the null. **It changed
them, which is the whole reason the trigger exists.**

**`2607.01940`, *Conditional Co-Ablation* (Gong et al., NTU, 2 Jul 2026) — the
paper §3.16 flagged as "read that one first" and nobody read.** Its Definition 1
is §3.22's statistic: the conditional ablation effect of a unit given an ablated
set, scored as **the growth of that effect once the primary set is removed**.
Ours is `dNLL(S+u) − dNLL(S) − dNLL(u)`; theirs is the same second-order object
in a Fisher energy over logits. Its abstract is §3.22's headline — first-order
scores *"become misleading when a transformer self-repairs: a dormant backup can
take over, muting the primary's measured effect while the backup itself appears
irrelevant on the intact model"* — which is exactly §3.22's `L4H9` (solo
**−0.001**, marginal **+1.018**), presented there as a finding rather than a
reproduction. And it runs **induction across eight models in six families,
Pythia-410M included**, with attribution factors from 2.1× (Pythia-160M) to
**12× (Pythia-410M)**.

**`2402.15390`, Rushing & Nanda (ICML 2024).** MLP participation in self-repair
is published: *"changes in the final LayerNorm scaling factor and sparse sets of
neurons implementing Anti-Erasure"*, on gpt2-small/medium/large and
pythia-160m/**410m**/1b.

**So four things are reclassified as replication**: §3.22's interaction
statistic; its misleading-first-order-scores framing; cross-scale recovery of
induction backups; and MLPs participating in self-repair at all.

**What survives is sharper for sitting in a gap the paper names itself.** CoAx
is **head-level by construction** — *"the signal is instantiated primarily at
attention-head granularity"*, head-level CoAx *"does not recover"* the
MLP-dominated case, its FFN probe is *"preliminary"*, and *"a full FFN-level
treatment of strongly MLP-mediated self-repair"* is *"left to future work."*
That is precisely where §3.23–§3.25 sit:

1. **An MLP as the *dominant* backup for one head** — MLP 6 at **+6.26**, above
   every head including `L7H8`'s +4.07, on a solo effect of +0.14.
2. **The mechanism is an active rotation of a constant direction** — `mu_clean`
   restores nothing (0.038, on a par with zero and with noise), `mu_cond`
   restores 0.643. Neither paper characterises a compensator's residual-stream
   write at all, and **the published mechanism is a rival hypothesis §3.23
   already rejects three ways** (identical residual norms under zero vs mean
   with attention 0.045 vs 0.643; norm-matched random reproducing zero; MLP 5
   removing as much norm for a fraction of the damage). The direction result is
   not LayerNorm scaling measured differently.
3. **The rotation is aimed at the set's shared key read-space and moves the
   heads it points at** (§3.24, §3.25) — no residual geometry in either paper.
4. **The attention/TV readouts and the ceiling-immunity argument**, which is what
   made the 70m port possible where §3.17's ceiling censors every ΔNLL cell.
   Both papers are output-grounded.

**And one non-hit recorded as a non-hit.** A search summary attributed
*"many-to-many wiring between previous-token and induction heads"* to
`2604.01094`; the phrase is not in its abstract and the full text was not
obtained, so it is logged as **unresolved, not as support**. Quoting it would
have repeated the 2026-09-10 failure §3.16 exists to prevent.

**Consequence for the registration, and it is a human call.** §3.27's candidate
— relay support concentrated on the causally-defined set, replicating across
rungs — is adjacent to CoAx's induction transfer. Not identical (ours is the
relay's *support* over downstream heads; theirs is *backups of* an ablated
primary) but close enough that **registering it as worded would claim novelty
this scan does not support.** The card with literature cover is the
**MLP-mediated, direction-carried repair**, and it needs its own scan of
FFN/neuron-level interpretability before anything is frozen. Nothing has been
written to `claims/registry.json`.

---

## 3.27 What DOES replicate: the relay supports the causally-defined set, at both rungs (2026-09-13)

> **NOVELTY QUALIFIED 2026-09-13 by §3.28.** The cross-rung measurement stands.
> Its status as a *registration candidate* does not: CoAx (`2607.01940`) reports
> induction-backup recovery transferring across eight models including
> Pythia-410M, so "backup structure replicates across scale" is substantially
> weaker as a novelty claim than this section presents it. **Do not register as
> worded.**

`p7d_redundancy/relay_support_profile.py`, new. §3.26 closed by admitting the
replication question as posed "does not quite have a subject" at 70m, because
70m does not do relay-backed matching. This is the better-posed version, and it
is the **first positive replication in this thread beyond the relay itself**.

**The question.** 70m's `L2H1` moves the readout by **+6.17 on a 5.73
baseline**, so *something* depends on it. Naming that something is the 70m
analogue of §3.20 — and asking it at both rungs on one instrument gives a
comparison §3.26's table could not, since that had to be assembled from two.

**The instrument needed a measured null, and its absence was caught by the
positive control.** Raw TV between clean and relay-ablated put `L7H8` at **rank
191 of 288**, with TV 0.1887 *below* the 0.2400 population median — even though
§3.20 measured exactly this ablation dropping `L7H8` 0.938 → 0.751. The numbers
agree (0.189 vs 0.190); the *comparison* fails, because ablating the relay
moves every downstream head by ~0.24 and a specific 0.19 does not stand out.
Calibrating each head against **its own sensitivity to generic ablation** (four
heads drawn outside the catalogue, §3.12-V3's discipline) fixes it: `L7H8` goes
to **rank 7** at 74x its own null, and `L6H0` to rank 8 at 67x. **Raw TV finds
a collapse and misses a targeted shift** — which is why
`mlp_backup_attention_scan.py` passed on raw TV (there `L7H8` goes to 0.045)
and this did not.

**What replicates:**

| | pythia-410m (`L5H2`) | pythia-70m (`L2H1`) |
|---|---|---|
| downstream heads | 288 | 24 |
| carry half the above-median support | **5** (1.7 %) | **2** (8.3 %) |
| **top-5 by support that are catalogue top-10** | **5 of 5** | **5 of 5** |
| Spearman(catalogue ΔNLL, support) | +0.312 (p = 6.2e-08) | **+0.727** (p = 5.7e-05) |

**The relay's support is concentrated, and it lands on the causally-defined
set — five of five at both rungs.** That is a structural invariant stated over
a causally-defined population rather than over head names (`design-8.md`:
"nothing transfers by head name"), and it is a candidate for the phase's
registrable list.

**What does not: how tightly support tracks causal magnitude.** At 70m the
support ranking *is* nearly the catalogue ranking — `L3H6` (+2.150) → `L3H1`
(+1.889) → `L3H5` (+1.248) → `L3H0` (+0.644) in that exact order, ρ = **+0.73**.
At 410m the two are only loosely related, ρ = **+0.31**: `L8H6` takes the top
support slot at **632x its null** on a causal effect of +0.212, while `L7H8` —
five times its causal effect at +1.019 — sits seventh. So at 70m one relay
supports a small set in order of how much each member matters; at 410m support
and importance come apart. That is the same decoupling §3.12-V found between
direction and substitutability, and it fits §3.24's finding that the 410m set
shares a read-space, so support spreads across the set rather than tracking any
member's own weight.

**Caveats, named.** 24 downstream heads at 70m against 288 at 410m, so the
concentration *shares* (8.3 % vs 1.7 %) are not comparable and only the counts
and the ranking are; the 70m Spearman is over 24 units. The rungs were read on
different probes (`freq` at 70m per §3.17, `wide` at 410m), which the ratio
statistic mitigates by calibrating each head within its own rung but does not
erase. Heads within a layer are not independent (§3.12-G6), so both p-values are
descriptive. Exploratory; `claims/registry.json` unchanged; both rungs spent
under `check_registry` rule 3.

---

## 3.26 The chain at 70m: the relay replicates, the circuit around it does not (2026-09-13)

`p7d_redundancy/mlp_backup_attention_scan.py` (new) plus
`prev_token_profile.py --model pythia-70m`. Detail in
`p8_scale_ladder/status-8.md` ("The self-repair chain at 70m"). §3.20–§3.25 are
`n = 1` on pythia-410m, which is spent forever under `check_registry` rule 3, so
the ladder is the only route by which any of it becomes registrable. 70m is the
other exploration rung and is free (`design-8.md`'s rung policy; 1b/1.4b stay
reserved and untouched).

**The blocker first, and the instrument built to get past it.** §3.17 records
that 70m's `L2H1` costs **+6.88 on a 2.67 baseline**, so it censors its own
cells against `ln 50304` *even on the `freq` probe* — every ΔNLL interaction in
§3.22/§3.23 is unavailable at that rung. Attention is immune: a distribution
over keys is well defined however badly the model is doing. That is what made
this port possible at all, and it is the first time the ceiling blocker §3.17
logged has been worked around rather than waited on.

**1. The relay replicates, cleanly.** 70m's `L2H1` carries prev-token attention
**0.950** (rank 1 of 48, next is 0.347) *and* the largest causal effect in the
catalogue (**+6.17**) — the same double signature as 410m's `L5H2` (0.970,
rank 1 of 384, +1.97). Two rungs, same object: a dominant previous-token head
that is also the most causally load-bearing head in the model.

**2. The matcher does not, and the ordering is inverted.** 70m's only strong
same-token matcher is **`L0H3` at 0.906 / 0.957** — as high as `L7H8`'s 0.934 —
but it sits in **layer 0**, *upstream* of the relay, and its catalogue effect is
**−0.839**: ablating it *improves* the readout. A layer-0 head cannot compose;
it matches on the embedding directly. So 70m solves same-token matching off the
embeddings at the bottom of the network while 410m does it seven layers up
through a relay, and **the relay → matcher ordering that defines the 410m
circuit is reversed at 70m and cannot exist there.** The offset convention is
not the explanation: §3.12's Q1 already measured `L7H8` at 93.4 % on exactly `j`
with nothing at `j+1`, so the instrument describes "attends to an earlier copy
of my own token" correctly at both rungs. And **no causally-important 70m head
does induction attention at all** — every head above +1.0 in the catalogue
scores under 0.024.

**3. The MLP backup does not replicate in its structural form.** At 410m the
dominant backup is MLP 6, *the first MLP that can see the relay's output* under
parallel residual, at **d_mean +0.1099** with its argmax head independently
landing on `L7H8` — 2.1x the next MLP. At 70m, on the same validated
instrument, step 16000:

| | 410m (relay `L5H2`, layer 5) | 70m (relay `L2H1`, layer 2) |
|---|---|---|
| parallel MLP (cannot see the relay) | MLP 5, +0.0377 | **MLP 2, +0.0944 / +0.0528** |
| first MLP that can see it | **MLP 6, +0.1099** | MLP 3, **−0.0153 / −0.0083** |
| separation over next best | 2.1x | 1.4–1.8x |

**The ordering inverts.** At 70m the winner is MLP 2, which is *parallel* to
`L2H1` and so architecturally cannot be responding to its output at all, while
MLP 3 — the structural analogue of MLP 6 — is **negative on both probes** at
step 16000. Its targeted consumer is consistently `L3H6`, a catalogue member
(+2.15), so 70m does have a targeted backup pathway; it is simply not the one
410m uses.

**Read 16000, not 143000.** At the endpoint the 70m readout is degenerate on
*both* probes for this instrument — TV saturates (max 0.90–0.98, and every MLP
0–2 near ceiling), which is §3.15's late-`wide` degeneracy showing up in
attention rather than only in NLL, and `freq` does not rescue it. Step 16000 is
the checkpoint that carries the claim, and both probes agree there.

**The honest limitation.** 70m has **6 layers against 410m's 24**, so
"parallel to the relay" and "the first sublayer that can see it" are one layer
apart in a network a quarter as deep, and the analogy is geometrically strained
in a way no amount of care fixes. This is evidence that the *specific*
MLP-6 structure is 410m's, not induction's — the same verdict §3.16/§3.17
reached for invariant 5 and `L11H14` — but it is **not** evidence that
relay-backed matching is absent at 70m, because 70m does not do relay-backed
matching in the first place (point 2). The replication question as posed does
not quite have a subject at this rung.

**A methodological finding that cost three iterations and is worth more than
the port.** The new instrument failed its own 410m positive control twice
before passing, and both failures would have produced a confident wrong answer
about 70m:

1. **Mean-ablation is blind by construction to a backup carried by the mean.**
   §3.15 makes `mean` the control for `zero`'s off-distribution bias, and that
   is right when the signal is in the variation. MLP 6's signal *is* its mean
   (§3.23), so mean-ablating it preserves the very thing under test and the
   scan put MLP 6 at **−0.0160**. **§3.15's rule needs this caveat attached:
   `mean` is the conservative control only when the mechanism is not itself the
   mean.**
2. **Averaging TV over all query positions halved a second-copy-only effect**
   and pushed `L7H8` below a layer-23 head, so the max statistic nominated the
   wrong MLP. Restricting to second-copy queries, matching `induction_scores`'
   own scope, fixed it.

Only after both fixes does the control reproduce MLP 6 *and* identify `L7H8` as
its target without being told. **A new instrument that has not reproduced a
known answer is not evidence about a new rung**, and this is the cleanest
example this project has of that principle paying for itself.

Exploratory; no p-value; `claims/registry.json` unchanged; both rungs spent
under `check_registry` rule 3.

---

## 3.25 The geometry predicts the function: the rotation moves the heads it points at (2026-09-13)

`p7d_redundancy/rotation_per_head_effect.py`, new. §3.24 showed MLP 6's repair
direction is aimed at the redundancy set's key read-spaces and flagged the gap
it could not close: **that is a weight-space measure**, and §3.12-S is this
project's own finding that weight-space overlap is not function-space overlap.
This closes it.

**The minimal pair.** Two states differing by exactly the rotation and nothing
else, both holding a *constant* in MLP 6's slot:

    REF  = `L5H2` ablated, MLP 6 := mu_cond    (rotation present; NLL 2.859)
    TEST = `L5H2` ablated, MLP 6 := mu_clean   (rotation absent;  NLL 8.955)

**The readout is attention, not loss, and that is forced.** At NLL 8.955 the
model is a nat from `ln 50304`, so any per-head marginal computed there is
ceiling-contaminated (§3.14.4-D); an attention distribution stays well defined
however badly the model is doing. Per head: **total-variation distance** between
its attention in REF and TEST, averaged over queries and sequences — in [0, 1],
needing no scale calibration, and role-agnostic, which matters because §3.18
found the FV-positive members never exceed an induction score of 0.015 and so
have no induction attention to measure. **Zero check: layers 0–6 return TV
exactly 0.00e+00** at both checkpoints, as causality requires.

**Members move, non-members do not** (layers > 6, n = 272):

| | members (n=9) | non-members (n=263) | Mann-Whitney |
|---|---|---|---|
| step 16000 | **0.2850** | 0.1471 | p = 1.1e-05 |
| step 143000 | **0.2483** | 0.1201 | p = 1.0e-02 |

Within the layers where the effect is largest, members exceed the non-member
**maximum** at both checkpoints — layer 7: members 0.353 / 0.281 against a
non-member max of 0.249 / 0.154; layer 8: 0.321 / 0.323 against 0.265 / 0.225;
layer 10: 0.321 / 0.318 against 0.289 / 0.231.

**And the geometry predicts the function head by head.** TV against `frac_K`
from §3.24, over the same 272 heads:

| | raw Spearman | layer-centred Spearman | mean within-layer | layers positive |
|---|---|---|---|---|
| 16000 | +0.318 (p = 7.9e-08) | **+0.325** (p = 4.2e-08) | +0.285 | 14 / 17 |
| 143000 | +0.223 (p = 2.1e-04) | **+0.337** (p = 1.1e-08) | +0.318 | 15 / 17 |

**Controlling for layer strengthens it rather than explaining it away**, which
is the confound that had to be ruled out: TV rises with depth on its own (per-layer
medians 0.03 → 0.16), so the raw ranking is contaminated by accumulation, and
the relationship survives centring each layer on its own median and holds
*within* individual layers in 14 of 17 and 15 of 17.

**The exceptions are the geometry's own.** `L9H13` — `frac_K` rank #104 / #89,
i.e. not a targeted head — sits **below** its layer's non-member median at both
checkpoints (0.113 vs 0.133; 0.062 vs 0.094). `L12H5` (#36 / #43) is barely
above median at 143000 (0.162 vs 0.141). The members the direction does not
point at are the members that do not move, which is the prediction rather than
a rescue.

**Two things not to quote.** The **raw top-TV ranking at 143000 is dominated by
layers 22–23** (`L23H7` 0.497 and eight more from layers 22–23 above any
member) — pure accumulation, and the reason the layer-controlled statistics are
the ones that carry the claim; at 16000, where accumulation is milder, five of
the top nine are members. And **`L15H14` flips**: above its layer's non-members
at 16000 (0.264 vs 0.113) and below at 143000 (0.044 vs 0.101). Recorded rather
than smoothed.

**Where this leaves the thread.** The chain is now complete end to end and each
link is measured rather than inferred: `L5H2` is a previous-token head wired
into the matcher's read-space (§3.20); removing it is compensated set-wide
(§3.21, §3.22); the dominant compensator is MLP 6, which *actively rotates* a
constant direction (§3.23); that direction is aimed at the redundancy set's
shared key subspace (§3.24); and it moves precisely those heads (this section).
No p-value here is an adjudication — heads within a layer are not independent
(§3.12-G6) and these statistics are descriptive. `claims/registry.json`
unchanged; pythia-410m spent under `check_registry` rule 3.

---

## 3.24 The repair direction decoded: it is aimed at the redundancy set's shared read-space (2026-09-13)

`p7d_redundancy/mlp6_decode_direction.py`, new. §3.23 closed by naming the
decode as the obvious next step; this is it. The object is `mu_cond` — MLP 6's
mean output once `L5H2` is ablated — and in particular the **rotation
component**, `mu_cond` minus its projection onto `mu_clean`, which §3.23 showed
carries the entire causal effect (writing `mu_clean` restores nothing, 0.038;
writing `mu_cond` restores 0.643).

**The LayerNorm guard, first, because it could have invalidated the whole
thing.** GPT-NeoX LayerNorm subtracts the mean across the hidden dimension, so
anything along the uniform vector is deleted before `L7H8` reads it.
`cos(mu, uniform)` is **+0.005 / −0.007** and the share surviving
mean-subtraction is **1.0000**. Nothing here lives in LayerNorm's null
direction.

**1. The rotation is aimed at attention read-space.** `W_K` for one head is
64x1024, so a random direction lands `64/1024 = 0.0625` of its squared norm in
its rowspace; measured over 200 random directions, **0.0617 ± 0.0100**. Taken
through the layer's own LayerNorm gain:

| vector | frac in `L7H8`'s K | frac in Q |
|---|---|---|
| random null | 0.0617 ± 0.0100 | 0.0616 ± 0.0100 |
| `mu_clean` | 0.091 / 0.120 | 0.064 / 0.114 |
| `mu_cond` | 0.220 / 0.237 | 0.144 / 0.197 |
| **rotation component** | **0.456 / 0.499** | 0.323 / 0.398 |

The component that carries the causal effect puts **~half its energy into a
64-of-1024 subspace** — 7–8x chance, ~39 sd above the null — while the
direction MLP 6 already had is ordinary. **K exceeds Q at both checkpoints**
(0.456 vs 0.323; 0.499 vs 0.398), which is the induction-shaped side.

**2. It is not a token signal.** The logit lens through `W_U` returns noise —
`'urn'`, `'il'`, `'ats'`, `'abo'` — with **entirely different token sets at the
two checkpoints**. That is a real negative and it agrees with the geometry: the
direction is position-independent, so it could not carry per-token match
content, and it is pointed at the matcher's machinery rather than at the
vocabulary.

**3. And it is aimed at the SET, not at `L7H8`.** Scoring the rotation into
every one of the **272 heads downstream of MLP 6**, `L7H8` ranks **5th at step
16000 and 4th at 143000** — far above the 0.073 median, but not the target.
The proximity confound is dead on the layer profile, which is flat (per-layer
medians 0.064–0.081 at 16000, spanning layers 7 to 23). What separates heads is
membership, not position:

| | members | non-members |
|---|---|---|
| within layer 7 | `L7H1`, `L7H8` — median **0.581** | median 0.071 |
| within layer 8 | `L8H6`, `L8H9` — median **0.650** | median 0.061, max 0.435 |

Global ranks of 272 at step 16000: **`L7H1` #0, `L8H6` #1, `L8H9` #3, `L7H8`
#5, `L10H9` #6, `L11H14` #8** — six of the top nine are catalogue members, from
a population where members are ~2 %. The same six lead at 143000.

**The exceptions are consistent, which is the best kind.** The members that are
*not* in the targeted group are `L12H5` (#36 / #43), `L9H13` (#104 / #89) and
`L15H14` (#125 / #177) — and `L12H5` is precisely the member that has come out
uncoupled on every previous measurement: it moved `L7H8`'s attention by exactly
**0.0000** (§3.20), and it is the largest *negative* residual against §3.12-V's
magnitude rule (§3.22). A fourth independent instrument putting `L12H5` outside
the same pathway is a consistency check that was not designed in.

**What this says.** The redundancy set is not merely a collection of heads with
interchangeable function — **its members share a read-space, and MLP 6's repair
addresses that shared space rather than any one head.** That is why §3.21/§3.22
found compensation distributed across several members instead of routed to the
matcher: the repair is a broadcast into the set's common input subspace. It
also gives §3.14.2's "a set of heads holding a residual-stream regime" a
concrete geometric referent.

**The function-space check, because §3.12-S's lesson binds here.** A rowspace
projection is a **weight-space** measure and weight-space overlap is not
function-space overlap — the finding that made membership causal in the first
place. The other members cannot be checked on `L7H8`'s instrument (§3.18:
their induction attention never exceeds 0.015, so there is nothing to restore),
so the readout is the **loss**, which aggregates every member's contribution:

| MLP 6's slot, `L5H2` ablated | NLL @ 16000 | NLL @ 143000 |
|---|---|---|
| *baseline (`L5H2` ablated, MLP 6 intact)* | *2.551* | *1.761* |
| zero | 8.002 | 6.701 |
| norm-matched random constant | 8.09–8.99 | 7.99–8.32 |
| **`mu_clean`** | **8.955** | **7.153** |
| **`mu_cond`** | **2.859** | **1.986** |

**One constant vector substitutes for MLP 6's entire position-varying output**,
recovering to within **0.31 / 0.23 nats** of the baseline and closing ~95 % of
the 5.45-nat gap that deleting the MLP opens. And `mu_clean` is **worse than
deleting the MLP outright** (8.955 against 8.002) — the pre-ablation direction
is not merely useless in the ablated state, it is actively wrong, which is what
an operating-point term should look like when set to the wrong point. So the
rotation is load-bearing in function space, not only in geometry.

**The per-member attribution is closed in §3.25**: the rotation moves the
heads its geometry points at, members exceeding their layer's non-member
maximum in layers 7/8/10, with TV-vs-`frac_K` surviving layer control
(Spearman +0.325 / +0.337).
Exploratory; no p-value; `claims/registry.json` unchanged; pythia-410m spent
under `check_registry` rule 3.

---

## 3.23 What MLP 6 is doing: active self-repair, by rotating one direction (2026-09-13)

`mlp_relay_role.py`, `mlp6_content_vs_scale.py`, `mlp6_response.py`, all new.
Detail in `status-7d.md` ("Opening MLP 6"). §3.22 found MLP 6 is `L5H2`'s
largest stand-in (+6.26, above every head) and closed with "nothing here opens
the MLP itself". This opens it.

**The architecture makes MLP 6 the unique candidate, and it is a fact not an
assumption.** pythia-410m has `use_parallel_residual = True` (read off the
loaded config): at every layer the attention and the MLP read the *same*
layernormed residual and both write into the next. So **MLP 5 cannot see
`L5H2`'s output** — it is parallel to it — and **MLP 6 is the first sublayer in
the network that can**, while writing into the residual `L7H8` reads. Two exact
validity checks fell out of measuring it: MLPs **7–23 move `L7H8`'s attention by
exactly 0.0000** (they are at or after the matcher), and MLPs **0–5 have
`cos(μ_clean, μ_cond) = 1.0000` and per-position change exactly 0.0000** when
`L5H2` is ablated. The instrument returns exact zero everywhere causality
requires it.

**1. `L5H2` and MLP 6 are an OR-gate over `L7H8`'s matching.** Either alone is
dispensable; together they are the whole supply:

| state | `L7H8` induction attention (16000 / 143000) |
|---|---|
| clean | 0.938 / 0.947 |
| `L5H2` ablated | 0.751 / 0.796 |
| MLP 6 ablated | 0.876 / — |
| **both** | **0.045 / 0.012** |

That is the attention-level mechanism behind §3.22's +6.26 ΔNLL interaction:
the super-additivity in the loss is the shadow of the matcher going blind.

**2. It is not scale, and that is established rather than assumed.** §3.15's
rule puts the burden on `zero`, so three arms answer it. `zero` and `mean`
leave the residual entering layer 7 at **43.07 vs 43.46** (and 42.83 vs 42.91
at 143000) — indistinguishable — while giving attention 0.045 vs 0.643. A
**norm-matched random constant** reproduces `zero` exactly (0.030–0.050 /
0.011–0.014). And MLP 5 removes as much residual norm as MLP 6 (43.2 vs 43.5)
while doing a fraction of the damage. Three independent ways of saying the
collapse is not an off-distribution norm artifact.

**3. The signal is a specific direction, carried by the mean.** MLP 6's output
is only **25–28 % constant** (‖μ‖ 9.69 against RMS‖out‖ 35.19), yet replacing
the *entire* output with μ alone retains **0.643 / 0.567** of the matcher's
attention. Direction-specificity is what separates MLP 6 from its neighbours —
`mean` minus `random` is **+0.60** for MLP 6, +0.12 for MLP 5, **−0.02** for
MLP 3.

**4. It is ACTIVE self-repair, not pre-existing redundancy — and this is the
result worth having.** Every interaction measured in §3.21/§3.22 is equally
consistent with a component that does exactly the same thing and merely becomes
load-bearing; §3.16 imported "self-repair" from `2307.15771` without anything
here separating the two. MLP 6's output **moves** when the relay is ablated:
its mean rotates to `cos = 0.837` and **grows 21 %** (the only MLP that grows),
with the change concentrated in the mean — its per-position variation changes
by 0.19, the *smallest* of MLPs 6–23. The causal version settles it:

| constant written into MLP 6's slot, `L5H2` ablated | 16000 | 143000 |
|---|---|---|
| zero | 0.045 | 0.012 |
| norm-matched random | 0.030–0.050 | 0.011–0.014 |
| **μ from the CLEAN state** | **0.038** | **0.016** |
| **μ after responding** | **0.643** | **0.567** |

**The direction MLP 6 already had is worth no more than zero or noise; only the
direction it moves to restores the matcher.** The response *is* the mechanism.
And the rotation is small — `cos = 0.837`, about 33° — so the entire difference
between a blind matcher and a working one lives in the component orthogonal to
what MLP 6 was already writing.

**5. It is not standing in by writing where `L5H2` wrote.** `cos(μ_cond,
d_L5H2)` = **−0.507**, the largest magnitude of any MLP and *negative*, against
0.01–0.14 for the unchanged MLPs 0–5. So "MLP 6 recomputes the prev-token
signal" is **not** what the data show — a position-independent constant could
not carry per-token match information in any case. The role is closer to an
enabling or operating-point term for the matcher than to a re-supply of its
content.

**Caveats, none of them hidden.** Every `L5H2` ablation in this thread is `ov`
mode, which §3.15 notes is really a *bias*-ablation; consistent throughout, and
named. The `mean` arms recompute μ inside the conditional state rather than
reusing clean-model means, because a clean mean injected into an ablated pass
is itself an off-distribution constant. `--controls 3` random directions,
agreeing to ~0.02. **What that direction is: decoded in §3.24** — it is
aimed at the redundancy set's shared key read-space (6 of the top 9 of 272
downstream heads are members), and it is not a token signal. Exploratory; no
p-value; `claims/registry.json`
unchanged; pythia-410m spent under `check_registry` rule 3.

---

## 3.22 The self-repair, measured exhaustively — and §3.21 was wrong about who does it (2026-09-13)

> **NOVELTY CORRECTED 2026-09-13 by §3.28, same day.** The measurements stand.
> The *framing* does not: this section's conditional marginal-vs-solo
> interaction is **CoAx** (`2607.01940`, 2 Jul 2026, Definition 1), and its
> headline — that single-unit scores mislead under self-repair because a backup
> looks irrelevant on the intact model — is that paper's central thesis, run on
> Pythia-410M among eight models. Independently derived here, published first.
> Read §3.28 before quoting any novelty claim from this section.

Four runners, all new: `backup_sweep_full.py` (every head's marginal cost once
`L5H2` is gone), `prev_token_profile.py`, `relay_selection_check.py`,
`mlp_backup_check.py`. Detail in `status-7d.md` ("Tying up the self-repair").
§3.21 left three things open and closing them **corrected §3.21 itself twice
and turned up a larger effect than anything in the thread so far.**

**1. §3.21's stand-in list was an artifact of the instrument, and it missed the
biggest ones.** The full 383-head causal sweep at step 16000 (solo, joint,
marginal, interaction per head; **solo column reproduces
`redundancy_catalog.json` bit-for-bit, max |diff| 0.0e+00 over 383 heads**;
restore exact; 0 of 383 ceiling-contaminated) ranks the stand-ins:

| rank | head | interaction | solo | found by §3.21's attention search? |
|---|---|---|---|---|
| 1 | `L7H8` | +4.068 | +1.019 | — (the known pair) |
| 2 | **`L5H9`** | **+3.356** | +0.035 | **no** |
| 3 | **`L9H5`** | **+2.745** | +0.030 | **no** |
| 4 | **`L1H15`** | **+2.351** | +0.040 | **no** |
| 5 | `L11H14` | +1.944 | +0.190 | yes |
| 6 | `L8H6` | +1.464 | +0.212 | yes |
| 15 | `L8H9` | +0.523 | +0.131 | yes |

**The attention search missed all four of the largest and ranked one of its own
top candidates (`L10H7`) at 377 of 383, with a negative interaction.** Its
precision and recall were both poor, and the reason is structural: it searched
for heads whose own *induction-attention* rose, and **a head that backs up
`L5H2` without being an induction head has no such score to rise.** So
§3.21's "three redundancy-set members" is withdrawn as a characterisation of
the stand-in population — it is **44 heads above +0.1**, of which §3.21 named
three. The set-wide claim survives and is strengthened; the *membership* claim
does not. This is §3.19's "a single pair is not a set" one level over, and the
same lesson as §3.12-R/G6/S with a *behavioural* proxy instead of a weights-one.

**2. The MLPs are the largest stand-in in the model, and no one had looked.**
MLP 6's interaction with `L5H2` is **+6.26** — above `L7H8`'s +4.07 and above
every head — while its solo effect is **+0.14**, so on the clean model it looks
irrelevant to induction. It survives every control:

| arm | interaction | headroom |
|---|---|---|
| step 16000, `mean`, `wide` | **+6.26** | 1.87 |
| step 16000, `zero`, `wide` | +5.25 | 2.82 |
| step 16000, `mean`, `freq` | +5.67 | 3.04 |
| step 143000, `mean`, `wide` | +5.14 | 3.67 |
| step 4000, `mean`, `wide` | +1.56 | 2.38 |

argmax over all 24 layers at every checkpoint; MLP 5 is second (+1.43/+1.72)
and everything from layer 12 up is under 0.25. **And it is specific to
`L5H2`**: the same MLP against `L7H8` gives **+0.125** and against `L12H5`
**+0.127** — 50x smaller — and with those sources no MLP anywhere exceeds
+0.25. Architecturally MLP 6 is the sublayer immediately after `L5H2`
(layer 5) and immediately before `L7H8` (layer 7). **MLP 0 is excluded in
every arm**: its solo ablation costs +12.5 and puts the joint arm *past*
`ln V`, so its negative interaction is exactly §3.12-M5's predicted artifact.
This **overturns the impression §3.12-Q6 left** ("no elevated MLP pathway",
best rank 44/159 at z +0.57) — that was a weights-only composition score on a
different question, and the causal interaction finds the model's largest effect
where the proxy found nothing.

**3. Why `L11H14`? Not by supplying what `L5H2` supplies — and the stand-ins
split into two classes by layer position.** The obvious hypothesis was that a
stand-in is a head that can re-supply the prev-token signal. For §3.21's three
it is dead: `L11H14`, `L8H6`, `L8H9` rank **377th, 380th and 372nd of 384** on
prev-token attention, *below* the population median. But the sweep's larger
stand-ins include real prev-token heads (`L5H9` 0.62, `L4H9` 0.81, `L3H1`
0.71), so the hypothesis is not simply wrong either — it is a **threshold**
property, and this is a fresh §3.13 case: across 383 heads the rank
correlation is **zero** (Spearman −0.016, p = 0.75) while the 12 heads above
prev-token 0.3 have median interaction **+0.444 against the other 371's
+0.0007** (Mann-Whitney p < 1e-5). Pearson (+0.322) splits the difference and
is the misleading one. Below the threshold prev-token attention predicts
nothing; above it, it predicts a great deal.

What separates the two classes is **position relative to the matcher**, read
off a conditional arm (`--background L5H2`, new flag) against a conditional
null where 6 generic controls move `L7H8`'s attention by ≤ 0.0038:

| stand-in | layer | Δ `L7H8` attention given `L5H2` gone | ΔNLL interaction |
|---|---|---|---|
| `L5H9` | 5 | **−0.0870** (23x null) | +3.356 |
| `L1H15` | 1 | **−0.0671** (18x null) | +2.351 |
| `L4H9` | 4 | −0.0095 | +1.018 |
| `L9H5` | 9 | **+0.0000** | +2.745 |
| `L11H14` | 11 | **+0.0000** | +1.944 |

**Upstream stand-ins partly restore the matching pathway; downstream ones
cannot and do not.** `L5H9`'s conditional effect is 3.4x its unconditional one
(−0.025 → −0.087) — it matters *more* to `L7H8` once the relay is gone, which
is the literal backup signature. `L9H5` and `L11H14` sit past layer 7, carry
interactions of +2.7 and +1.9, and move `L7H8`'s attention by **exactly
zero**: they compensate at the readout, not in the circuit. So `L11H14` is an
output-side compensator, and "why is it the strongest" is now the narrower
question of why the output-side class exists at all.

**4. What selects `L5H2` as the relay is composition, not its attention
pattern.** Prev-token capacity is common — 13 heads above 0.3 — so it cannot
be what makes one head the relay. Scoring every layer-0..6 head's composition
into `L7H8`'s read-space per head (`relay_selection_check.py`; H1-REVISED
stored only the population summary, so this comparison was previously
unanswerable) and pairing it with an ablation:

| head | prev-token | composition rank / z | Δ `L7H8` attention |
|---|---|---|---|
| `L5H2` | 0.970 (1st) | **0 / +5.25** | **−0.190** |
| `L5H9` | 0.616 (4th) | 1 / +3.64 | −0.025 |
| `L6H0` | 0.005 (381st) | 2 / +3.27 | −0.029 |
| `L6H13` | 0.429 | 3 / +2.85 | −0.017 |
| `L4H9` | 0.805 (2nd) | 23 / +0.28 | **+0.0008** |
| `L3H1` | 0.710 (3rd) | 24 / +0.25 | **+0.0028** |

**Prev-token attention without composition buys nothing** — `L4H9` and `L3H1`
carry 71–83 % of `L5H2`'s prev-token attention and are within the generic
control band. Composition without prev-token attention buys a little
(`L6H0`). `L5H2` is the joint extreme and is **7x the next largest**, so the
account is conjunctive and strongly super-linear; no functional form is
claimed. *Report-both caveat:* Pearson(prev-token, composition) is +0.42 but
Spearman is **+0.12, p = 0.23** — the axes are barely rank-associated and the
Pearson is driven by `L5H2` being extreme on both, so "these are independent
axes" is the conservative reading and "composition tracks prev-token" is not
supported.

**5. Part of §3.21 was already on disk, unread.** `L5H2`×`L11H14` = **+2.1800**
at step 16000 is in `pairwise_interaction_matrix.json` (2026-09-10), and
§3.21's run reproduced it to four decimals. The 45-cell matrix already
contained the whole `L5H2` row, so "which top-10 members are super-additive
with `L5H2`" was answerable without a forward pass. What was genuinely new in
§3.21 was `L10H7`/`L10H15` (not in the top 10), step 4000, and the framing.
Recorded because `CLAUDE.md` opens on this cost and §3.17 logged a prior
instance. **And reading that matrix against its own regression pays off
immediately**: §3.12-V's magnitude rule (r² 0.74 at 16000, 0.81 at 143000,
both reproduced) has **one systematic exception, `L5H2`×`L11H14`** — the
largest positive residual of all 45 cells at *both* checkpoints (+1.37, +0.97)
— while its δ-cosine at 143000 is **0.004**, orthogonal. `L11H14` stands in far
more than either its size or its alignment predicts.

Exploratory throughout; no p-value is claimed as an adjudication (the two
population tests above are descriptive, on non-independent units);
`claims/registry.json` unchanged; pythia-410m spent under `check_registry`
rule 3.

---

## 3.21 The super-additivity is set-wide self-repair, not an `L5H2`×`L7H8` special case (2026-09-13)

> **CORRECTED 2026-09-13 by §3.22, same day.** The set-wide conclusion stands.
> The *membership* claim below does not: the three members named here are
> ranks 5, 6 and 15 of a 44-head stand-in population, and the four largest
> stand-ins — `L5H9`, `L9H5`, `L1H15`, and **MLP 6**, which beats every head —
> were missed because the search below reads induction-attention, which a
> non-induction backup does not have. Read §3.22 before quoting this section.

`p7d_redundancy/l5h2_backup_search.py` and `l5h2_backup_causal_check.py`, new
this session; detail in `status-7d.md` ("Self-repair"). §3.20 closed `L5H2`'s
mechanism but left its companion open: §3.12-S found the joint ablation of
`L5H2` + `L7H8` is super-additive (2.2x the sum of parts), the opposite of
what a direct serial dependency predicts, and named Hydra-effect self-repair
as the untested standing hypothesis.

**Step 1 — a cheap full-model search.** `induction_scores` batches over every
head in one attention-output pass, so scoring all 384 heads' own
induction-attention under baseline and under `L5H2`-ablation costs two
forward passes per checkpoint (~13 s at 8 seqs). Two things fall out that
were not being looked for:

- **`L5H2` feeds more than `L7H8`.** `L6H0` — not a redundancy-set member by
  the causal-ablation criterion, but a strong repeated-random-token induction
  head in its own right (0.75–0.84 through training) — falls **harder** than
  `L7H8` at every checkpoint from 2000 on: delta −0.435 at 16000, **−0.485 at
  143000**, against `L7H8`'s −0.190 / −0.161. `L9H9` and `L9H8` fall too,
  smaller and fading late. `L5H2` is a hub feeding several matchers, not a
  private circuit with one.
- **A consistent riser cluster.** `L10H7`, `L10H15`, and three already-known
  redundancy-set members — `L11H14`, `L8H6`, `L8H9` — show their OWN
  induction-attention rise when `L5H2` is ablated, from step 1000 onward.
  `L10H15` peaks mid-training (+0.118 at 16000); `L10H7` is the dominant riser
  only at the trained endpoint (+0.136 at 143000). Rising attention is not a
  causal claim by itself — it says a head *could* be compensating.

**Step 2 — the causal version, on the risers the search actually produced**
(fixed in advance, not chosen after seeing this run). For each candidate,
`interaction = dNLL(L5H2 + candidate) − dNLL(L5H2 alone) − dNLL(candidate
alone)` — the same quantity §3.12-S computed for `L7H8`, now asked of heads
the attention search surfaced instead of assumed. `L7H8` reproduces to three
decimals at step 16000 (**+4.1505** here against §3.12-S's **+4.151**), which
calibrates the method.

| step | `L11H14` | `L8H6` | `L8H9` | `L10H15` | `L10H7` | `L7H8` (control) |
|---|---|---|---|---|---|---|
| 4000 | **+1.578** | +0.750 | +0.500 | +0.220 | −0.011 | +1.717 |
| 16000 | **+2.180** | +1.613 | +0.543 | +0.154 | −0.220 | +4.151 |
| 143000 | **+1.532** | +0.302 | +0.083 | +0.002 | −0.101 | +3.430 |

**Three of the five candidates show `L7H8`'s own signature.** `L11H14`,
`L8H6` and `L8H9` are all super-additive with `L5H2` at every checkpoint —
`L11H14`'s interaction (+1.53 to +2.18) is **8–11x its own solo effect**
(+0.17 to +0.19), a bigger relative jump than `L7H8`'s own (interaction ~4x
solo). **The super-additivity is a set-wide property, not a private feature
of the `L5H2`×`L7H8` pair** — which is what actually reconciles §3.12-S with
§3.20: `L5H2`'s ablation is compensated for by several members of its own
redundancy set at once, and the pairwise `L5H2`×`L7H8` arm was only ever
seeing one of them.

**`L10H15` fades exactly like the set's other members do.** Positive at 4000
and 16000, vanished by 143000 (+0.002) — the same decay-to-near-zero shape
§3.12-U found for four of the six original members.

**`L10H7` is the dissociation, and it is worth keeping.** Its attention rose
at every checkpoint, but its causal interaction is **negative at every
checkpoint** (−0.01 to −0.22) — sub-additive, the opposite of self-repair.
Rising attention does not imply rising causal usefulness, which is §3.12-R/G6's
lesson (weights-only proxies fail to predict causal effect) reproduced one
level up: here the proxy is a *behavioural* readout (attention), not a
weight one, and it still fails to predict the causal quantity it looks like
it should predict.

**What this does and does not settle.** It gives §3.12-S's super-additivity a
concrete, located mechanism — three named members of the set standing in for
`L5H2` — rather than leaving "self-repair" as an unlocated citation. It does
**not** show these three are the *only* contributors (MLPs are untested; the
384-head search only looked at attention, not at every head's own marginal
ΔNLL), and it does not explain why `L11H14` — already the set's oddest
member on four independent axes (§3.19, §7e) — is also its strongest
stand-in for `L5H2` specifically. Exploratory; no p-value; `claims/registry.json`
unchanged; pythia-410m spent under `check_registry` rule 3.

---

## 3.20 `L5H2`'s puzzle, closed on the mechanism — not on the additivity (2026-09-13)

`p7d_redundancy/upstream_relay_check.py`, new this session; detail in
`status-7d.md` ("The upstream-relay check"). §3.16/§3.18 filed a puzzle:
`L5H2` has the largest single-head causal effect on the readout of any
redundancy-set member (+1.97 at step 16000, §3.12-T) yet scores near zero on
both instruments ever pointed at it — the QK-based induction-attention score
(§3.12-U) and the FV score (§3.18). Both instruments measure `L5H2`'s **own**
behaviour; neither can see a causal role running through a downstream head.
Two facts already on disk pointed exactly there and had never been connected:
`L5H2` is a confirmed previous-token head on its own attention (Stage 0,
2026-09-07: offset −1 = 0.895, ~49x the 384-head median), and its OV composes
into `L7H8`'s Q/K read-space at rank 0 of 112 (z ≈ +6), onset step 512–1000
(§3.12-H1-REVISED, 2026-09-09) — a weights-only quantity that was never tested
for whether it does anything.

**It does.** Ablating `L5H2`'s OV (restore exact) degrades `L7H8`'s **own**
induction-attention score at every one of 6 checkpoints, 60–600x more than any
of 4 generic controls per step: delta −0.0052 (step 1000, floor-limited) to
**−0.3742** (step 4000, `L7H8`'s own formation window) to −0.16 at the trained
endpoint, against controls that never move it by more than 0.0032. **The
sharper control**: `L12H5`, the model's third-largest single-head causal
effect (+0.42), moves `L7H8`'s attention by **exactly 0.0000** at three
checkpoints — mattering a lot for the readout is not sufficient to disturb
`L7H8`; only `L5H2` does, so this is not a "removing something big" artifact.

**This closes the mechanism half**: `L5H2` has no induction score of its own
because it is a genuine previous-token head, not an induction-position
attender, and no FV score because that is not its job either — its causal
weight runs through feeding `L7H8`'s own matching attention, and the
composition already on disk is now shown to be functional rather than only
structural. **It does not close §3.12-S's super-additivity** — a dependency
this direct predicts *sub*-additive joint ablation and S1 found the opposite
(2.2x the sum of the parts). The standing hypothesis, not yet measured, is
that the Hydra-effect self-repair §3.16 already invokes for 44/45 pairwise
cells restores `L7H8`'s effect on the **loss** when `L5H2` alone is ablated,
without restoring `L7H8`'s **attention pattern** — consistent with why an
attention-level probe, not an NLL one, is what could still see this. Finding
the head(s) that do that restoring is the next test if this thread reopens.

---

## 3.19 Invariant 4, measured on the set instead of one pair (2026-09-12)

Detail in `p8_scale_ladder/status-8.md` ("Invariant 4 as a set-level
trajectory") and `p7d_redundancy/status-7d.md`. `literature-8.md` calls
invariant 4 the phase's load-bearing card — the only one both unclaimed and
replicating — and the reframe makes it *the* account of what `2407.10827`'s
head turnover looks like geometrically. Its evidence was **one pair per rung**.
Now it is the set, at both rungs, on the matched instrument.

**Three things that are not phase-8-local.**

**1. A single pair is not a set, and here it inverted the conclusion.**
`status-8.md` concluded "70m never separates" from `L2H1`×`L3H6` going
0.934 → 0.685. That pair turns out to be **70m's most-aligned pair at every
late checkpoint**. On the four-head core the same trajectory reads **+0.922 →
+0.321 by step 16000**, against 410m's core at **+0.814** — so 70m separates
*more*, not less. **The finding was not wrong about its pair; it was wrong to
be a set-level claim.** Any invariant stated over "the members" needs a
set-level statistic with `n`, and §3.12-V5's changing-membership hazard is the
same lesson one level down.

**2. The rank-1 cosine and the subspace CKA can disagree, and which one is
right depends on the head's dimensionality.** `L11H14`'s mean-delta cosine to
the set inverts to **−0.099** while its centered CKA holds **+0.317 against a
measured null of +0.179**. Both are correct: its mean write direction
anti-aligns, its effect subspace still overlaps. The tell is its participation
ratio, ~60 against everyone else's 8–28 — **a rank-1 summary of a
60-dimensional effect is the wrong instrument**, which is `§3.13`'s principle
on the dimension axis rather than the population axis. `status-7d.md` already
said to quote `cka`/`cka_centered` over the unweighted cosine; this is the case
that shows the two genuinely parting company rather than merely differing in
scale. **Never write "`L11H14` is orthogonal to the set" — write "anti-aligned
in mean direction".**

**3. The cached checkpoint grids differ between rungs, and offline mode turns
that into a mid-run crash.** 410m carries earlier phases' log-spaced fills
(3000, 5000, 7000, 9000, 54000); 70m carries the 19-step behavioural grid and
64000. Neither has the other's. With `HF_HUB_OFFLINE=1` a missing revision is
an `OSError` eleven minutes into a run, which is how this was found. **The
intersection — `256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 143000` — is
the only matched grid**, and any cross-rung trajectory must be planned on it
before the job is launched. `ls data/hf/hub/models--EleutherAI--pythia-*/refs/`
is the check.

**What it does to the invariant itself.** *Born aligned* replicates at both
rungs and is solid (+0.888 at 410m, +0.922 at 70m, both at step 1000, neither
with a private-subspace phase). *Then fans out* replicates at both rungs once
measured on the set. **The fate does not replicate, and neither ending is in
the current wording**: 410m ends as a locked core plus one inverted defector;
70m's core reaches its minimum at step 32000 (+0.302) and then **re-coheres**,
+0.580 by 64000 and +0.611 at 143000 — all six core pairs, delta norms growing
through it, and it survives the `freq` re-check that exists precisely because
70m's late `wide` readout is degenerate. `design-8.md`'s invariant 4 stops at
the fan-out. **It must be reworded before registration, and that is cheap now
and impossible afterwards** — which is `CLAUDE.md`'s literature-scan trigger 2
applied to the wording rather than to the citation.

---

## 3.18 The redundancy set divides labour between induction and function-vector roles (2026-09-12)

`p7d_redundancy/fv_score.py`, detail in `p7d_redundancy/status-7d.md` ("The
FV-head experiment"). Six members + six controls, four word-pair tasks, six
checkpoints, **both scores measured in one process off the same weights** —
because the claim under test is about two trajectories' joint shape, and
reading one of them off a stored series would rest the comparison on an
uncontrolled instrument difference.

**§3.16's hypothesis is refuted for `L5H2`.** Its FV score never leaves zero
and ends **negative** (−0.0018 at 143000, below four of the six controls), and
per task it is sign-inconsistent. `L5H2` is not acquiring a function-vector
role as it sheds an induction one. **§3.12-U's puzzle is still open** — the
head with the largest causal effect on the induction readout does neither job
by either score.

**The result that replaced it is the structural one.** Within one
causally-defined redundancy set, **no member does both jobs**:

| | induction score | FV score | role |
|---|---|---|---|
| `L7H8` | **0.021 → 0.947** | ±0.0001 at every step | induction only |
| `L8H9` | ≤ 0.011 | **+0.0152** at 143000 | function-vector only |
| `L8H6`, `L12H5`, `L11H14` | ≤ 0.015 | +0.0012 … +0.0085 | function-vector only |
| `L5H2` | 0.0000 | −0.0018 | **neither**, and the largest causal effect |

The four FV-positive members never exceed an induction score inside the control
band (0.008–0.013), and the one real induction head has no FV score at all. At
step 8000 the **top four FV heads of the twelve scored are all members**, the
fourth beating the best control by 18×.

**Why this is worth more than the hypothesis it replaced.** `2502.14010`
reports induction heads *transitioning* into function-vector heads over
training. A **division of labour** across members of one set is a different
structure, and it is visible here only because §3.14.2 defined membership
**causally** — by `ΔNLL` under ablation — rather than by either score. A
catalogue built on induction score would have contained `L7H8` and missed the
other four; one built on FV score would have done the reverse. **This is the
strongest direct vindication the causal-membership card (§3.16's "best card
first") has had**, and unlike the rest of that card it is a positive result
rather than a demonstration that proxies fail.

**Read against the endpoint at your peril.** The measured null grows: max abs
control 0.0000 → **0.0103** over the grid, so `L8H9` beats the best control 12×
at step 8000 and only 1.5× at 143000. The effect is cleanest **mid-training**.
The FV rise is also confounded with the model simply learning the tasks (the
ICL gap rises over the same interval) — what breaks that confound is the
**controls staying at zero while the gap grows**, which holds cleanly through
step 16000 and weakens at 143000. Both §3.13 views agree in sign and ordering
at every checkpoint, the median uniformly smaller.

**A methodological note that generalises.** The FV instrument validates itself
on the task axis: per-task effect tracks the per-task ICL gap, and the two
tasks pythia-410m cannot do show almost nothing. **An FV score measured on a
task the model has not learned is not a null, it is undefined** — so any FV
work at another rung must carry its per-task ICL gap beside the score. The
runner does.

---

## 3.17 The probe is the ceiling handle, and SVD ordering has a third class (2026-09-12)

Phase-8 detail is in `p8_scale_ladder/status-8.md` ("Invariants 5 and 6 on the
`freq` probe"). Two results from it are not phase-8-local.

**1. `--probe freq` is the instrument for ceiling censorship, not just for
late-checkpoint degeneracy.** §3.15 introduced `freq` as a scope fix for a
*small model's late checkpoints*, where the language prior beats the copy
mechanism. It is more general than that: the probe sets the **baseline NLL**,
and the baseline sets the headroom every joint arm has before it hits
`ln 50304`. At pythia-70m step 16000 the switch takes the baseline **5.73 →
2.67** and the headroom **5.10 → 8.15 nats**, which uncensors two cells of the
pairwise matrix outright — and the three uncensored cells turn out to be **the
three largest interactions in the matrix** (+0.79, +0.74, +0.67). §3.14.4-D
said to run the matrix at a step with headroom; the probe is the *other* knob,
and it is the one that works when the head itself is large. So: **any
`ceiling_contaminated` cell should be retried on `freq` before it is written
off as unmeasurable.** Ported to `pairwise_interaction_matrix.py` and
`useful_rank.py` on 2026-09-12; `wide` stays the default everywhere, so nothing
recorded changes meaning.

Its limit, which is real: a head whose own effect is comparable to the whole
headroom cannot be rescued by any probe. 70m's `L2H1` (+6.88 on a 2.67
baseline) censors all five of its own cells on `freq` too, so the analogue of
§3.12-S's prev-token × matcher pair stays unmeasurable at raw `dNLL` at that
rung and needs §3.12-M's graded readout.

**2. SVD ordering versus causal usefulness has THREE classes, not two.** §3.12-V4
and 7e set up a dichotomy — gain-ordered (`L7H8`: top-1 recovers 0.971) versus
anti-ordered (`L11H14`: bottom-`r` > matched-random > top-`r` at every rank,
top-1 **−0.096**). pythia-70m's `L3H1` is neither. It sits below its
matched-norm control at 11 of 11 ranks, which is how `L11H14` announced itself —
but run with `--bottom`, its top and bottom curves **coincide** (r=8: +0.239 vs
+0.238) and a random subspace beats both:

| class | top-`r` | random | bottom-`r` | exemplar |
|---|---|---|---|---|
| gain-ordered | best | mid | worst | `L7H8`, `L5H2`, 70m `L2H1`/`L3H6` |
| **unordered** | ≈ bottom | **best** | ≈ top | **70m `L3H1`** |
| anti-ordered | worst | mid | **best** | `L11H14` — still the only one |

**This widens §3.15's methodological warning rather than narrowing it.** That
warning said an SVD-ordered `r*` misleads on `L11H14`-like heads. The third
class means *"below its control"* is **not** sufficient to diagnose
anti-ordering — it is consistent with a basis that carries no ordering at all,
and only `--bottom` separates the two. `induction_rank_sweep`'s `r*`
construction is wrong on both classes and for different reasons. **Do not quote
an `svd`-basis `r*` for a head not checked with `--bottom`**, and do not infer
anti-ordering from the control comparison alone.

**`L11H14` remains a pythia-410m singleton.** 70m has no anti-ordered member,
so invariant 5 fails at that rung on both halves (no low-rank majority, no
anti-ordered exception) — which is §3.16's "a property of pythia-410m, not of
induction" in its sharpest form.

*Caveat carried, not buried:* `--controls 2`, and `L3H1`'s top-versus-random
margin is only 0.02–0.08. The robust half is **bottom beats random 0/11**,
where the gain-ordered heads separate by 0.2–0.9.

**And one stale-doc cost, recorded because `CLAUDE.md` opens on it.**
`status-8.md`'s "Reproducing" section claimed *"none of them has been re-read
under `mean`"* for a full day after the re-read landed in the section above it,
and sent a session looking for work that was already done. The nominated
candidate in that same document — `L0H0` as a second anti-ordered head — was
also wrong, and wrong in a diagnosable way: under `mean` its `d0` is −0.09, so
its recovery fractions (+63.7 against a control of +31.9) are the
small-denominator garbage §3.15's second instrument limit predicts. **A
`useful_rank` row whose `|d0|` is near the noise floor is not evidence in
either direction.**

---

## 3.13 When the mean is the wrong instrument (2026-09-09)

`§3.12-G6` found a signal that Spearman could not see: fifteen of sixteen heads
per layer carry causal noise, so a correlation over all sixteen averages the
signal away, while the *rank of the top head* recovers it at ~2.6e-4. That is a
general lesson and this section is where it is logged, because the sites it
applies to are spread across the repo.

**The principle, and its edge.** A mean or correlation is the right summary when
the effect is *distributed* across the population; an extremum — max, argmax,
top-rank — is right when it is *concentrated* in a few members. **Choosing
between them after seeing the data is exactly the selection `claims/registry.json`
exists to forbid.** So the rule is not "use max": it is

- **exploratory work reports both, always** (`§3.12-G6` was legitimate because
  nothing is registered and both were reported);
- **registered work has already frozen the choice**, and the alternative may be
  *reported beside* the result but never swapped in — the same discipline the
  registry already applies to `p_reciprocal`, which is a stop-rule input and
  enters no `E`.

### 3.13.1 The arithmetic sub-case, which is not a judgment call

A z-score computed against a population **that contains the point being
scored** is capped. For a sample of `n`, the largest attainable studentized
deviate is `(n − 1)/√n` — at `n = 16`, **3.75**. The statistic cannot report a
larger effect however large the effect is.

`induction_subspace_characterize`'s `target_vs_reference` computes the target
head's z against `layer_mean` / `layer_sd` **over all 16 heads, the target
included**. Measured on the seven populations already on disk:

| population | top head | ΔOV_nll | z (self-included) | z (leave-one-out) | |
|---|---|---|---|---|---|
| 4000 / L7 | H8 | 0.244 | 3.69 | **20.59** | 5.6× |
| **8000 / L7** | **H8** | **0.725** | **3.74** | **50.17** | **13.4×** |
| 8000 / L9 | H5 | 0.187 | 2.93 | 4.84 | 1.7× |
| 2000 / L9 | H5 | 0.317 | 3.60 | 13.25 | 3.7× |
| 16000 / L1 | H15 | 0.040 | 3.35 | 7.72 | 2.3× |
| 16000 / L6 | H0 | 0.021 | 2.23 | 2.86 | 1.3× |
| 32000 / L2 | H2 | 0.056 | 2.48 | 3.41 | 1.4× |

**`L7H8` at step 8000 measures 3.74 against a ceiling of 3.75 — saturated.** A
reader comparing it to step 4000's 3.69 would conclude the effect barely moved;
the leave-one-out z went **20.6 → 50.2**. The reported number stopped being a
measurement and became the ceiling.

This is a **reporting defect, not a scoring one** — no p-value is computed from
this z and `claims/adjudications/` is empty, so nothing registered is affected.
`core/nulls.py`'s z is *not* subject to it: there `observed` is scored against a
null distribution it is not a member of, and where the identity permutation *is*
included the draw count is large enough that `(P−1)/√P` is far above anything
attainable. **The fix is leave-one-out, and it is a docstring and three lines.**

### 3.13.2 Sites surveyed

*Already handled — recorded so they are not re-litigated.* `tools/run/
behavioural.py` prints `mean` **and** `max` side by side. `P-M1`'s registry
entry computes the mean/min/max head-to-layer aggregates, declares `mean`
primary *in advance* so it cannot be picked after the fact, and **refuses a
p-value when the three disagree in sign**. `CLAIM-B` reports dispersion beside
every centroid precisely so a bimodal profile is visible. `ov_per_head.py`
records the energy split *and* the count because their disagreement is the
reading (`§3.12-F0` finally read it).

*Worth a look, none registered-blocking.* `p7_motifs/motif_stats.py`'s
`mean_ind` / `mean_non` are means over head sets — `P-I3`'s registered statistic
already superseded them with a rank-based matched contrast, so the means are
diagnostics; they should say so. `p2d_operator_activation/run_2d.py`'s
`head_mean` energy series is `P-M1`'s aggregate and inherits that entry's
refusal.

*A frozen entry whose stated reason does not distinguish the two.* `CLAIM-C`
takes `delta = mean over normalized depth`. Its `null_construction` justifies
the choice against a depth band: *"Blog 1 quotes layers 5-30 of gpt2-large, but
a depth band is a choice with as many options as there are bands."* That
argument is sound against a **band** — and **a max over depth places no constant
either**, so it does not distinguish mean from max at all. The registered
wording is frozen and stays; this is recorded because the reasoning has a gap,
not because the entry should change. Blog 1 quoting a band is itself weak
evidence the contrast is depth-concentrated.

### 3.13.3 What this suggests we have overlooked: the position axis

Every readout in `§3.11`–`§3.12` is a mean over **token positions**, and nothing
has ever looked at that axis:

- `behavioural_induction_score` and `induction_attention` are
  `picked.mean()` over `N_REP = 96` second-copy positions × 8 sequences;
- `second_copy_nll` — the readout the **entire** OV rank sweep, the 82 %, and
  every ΔOV_nll in this document rest on — is a mean over the repeated half.

If copying is concentrated at particular positions (later second-copy positions
have more context, so concentration is the expected shape rather than an exotic
one), then the mean dilutes it and the rank sweep's `r*` is being read off a
diluted curve. **This is the same error `§3.12-G6` found, one axis over, in the
measurement everything else depends on.** It is one forward pass to check: emit
the per-position NLL delta instead of its mean and look at the profile. Report
mean and max together, per 3.13's own rule.

**MEASURED 2026-09-09 (`tools/run/induction_position_profile.py`, `L7H8` at
step 4000, 64 sequences — eight times the original readout's sample count).
THE HYPOTHESIS ABOVE IS WRONG, and the negative is worth more than the positive
would have been.**

*The effect is not position-concentrated.* **37 of 96** positions carry half the
ΔNLL mass (uniform would be 48); ninety percent needs **80 of 96**; the top
decile of positions carries **0.178** of the total against 0.100 for uniform.
Mildly above uniform, nowhere near "a few members".

*And `r*` does not move.* Recovered fraction by rank, computed as a **ratio of
sums** over each position set rather than a mean of per-position ratios (the
per-position denominator is near zero where the head does nothing):

| rank | all positions | concentrated half | dilute half | excluding j=0 |
|---|---|---|---|---|
| 1 | **0.824** | 0.791 | 0.858 | 0.822 |
| 2 | 0.850 | 0.832 | 0.869 | 0.849 |
| 4 | 0.871 | 0.866 | 0.877 | 0.870 |
| 8 | 0.945 | 0.943 | 0.947 | 0.944 |
| 16 | 0.971 | 0.972 | 0.971 | 0.971 |
| 64 | 1.001 | 1.001 | 1.000 | 1.001 |

The concentrated and dilute halves differ by at most **0.067** at rank 1 and are
identical to three digits by rank 8. **The 82 % is not an artifact of
averaging**, and `r*_SVD = 1` survives the axis that could have dissolved it.
Restore check exact (`0.000e+00`).

*The methodological reading, which is the point of §3.13.* The principle says
**report both**, not *expect the extremum to win*. Here the mean was the right
instrument and the check confirms a result rather than overturning one. §3.13
is not a licence to prefer extrema — it is a requirement to look, and looking is
cheap.

---

## 3.37 The per-phase literature review (2026-09-16; merged 2026-09-17)

**Every phase now has a `lit-N.md`.** `docs/LITERATURE.md` is the index, the shared
method, the cross-phase findings, and a single ranked reading queue. This section is
the record of what the review changed; the detail lives in the files.

### Why it was run, and what it supersedes

`docs/literature_scan_2026-09-10.md` covered §3.12-V only — the 7d/7e/8 territory. Every
other phase was standing on citations chosen when the phase was designed, some of them
two years old, with no check on what the field did since. The user asked for the same
treatment, per phase. **All four of the 2026-09-10 scan's verdicts are confirmed or
sharpened; none is overturned.**

### The constraint, stated first because it bounds everything

**`arxiv.org`, `semanticscholar.org`, `openreview.net` and every other scholarly host
are blocked by this session's egress proxy.** Not one abstract page could be fetched.
`WebSearch` runs server-side and works; `WebFetch` and `curl` return `EGRESS_BLOCKED`
or a 403 CONNECT.

So **nothing was read.** Every arXiv id, author list, date and finding came from a
search engine's summary of a page — weaker than an abstract, much weaker than a paper.
The marks `[S]` (summary read) and `[N]` (title and id only) appear throughout, and
nothing is marked "read". At least one attribution is flagged as unreliable for
exactly this reason (`p1c_frames/lit-1c.md` §5.3). **This is the same discipline the
2026-09-10 scan set and it is not optional.**

**Working `docs/LITERATURE.md` §5 from a machine with arXiv access is the
highest-value unblocked task in the project.** Item 18 in that queue is a GitHub
repository rather than a paper — GitHub may be reachable where arXiv is not.

### What is scooped

Ranked, with the full list in `docs/LITERATURE.md` §2:

1. **Phase 1's four-transition developmental arc.** `2509.23024`, *Tracing the
   Representation Geometry of Language Models from Pretraining to Post-training*
   **[S]**, NeurIPS 2025 / ICML 2025 — **Pythia 160M–12B and OLMo**, RankMe and αReQ,
   "a consistent non-monotonic sequence of three geometric phases": warmup collapse →
   entropy-seeking expansion (peak n-gram memorisation) → compression-seeking
   consolidation. That is `status-1.md`'s "collapse, recovery, overshoot, slow
   decline", which it calls **"the phase's main new object"**. **Strike or rewrite.**
2. **7d's second-order interaction instrument.** `2607.01940`, *Conditional
   Co-Ablation (CoAx)* **[S]**, 2 July 2026 — "measures how much each remaining unit's
   ablation effect grows once a primary set has been removed". The 2026-09-10 scan
   named it as the nearest scoop; it is.
3. **7d's "behavioural proxies fail" thesis.** `2606.05378`, *Pattern Selectivity is
   Not Task-Causal Structure* **[S]**, June 2026 — by the **same group** as
   `2606.02378`, same repository. The cross-model half of §3.12-U's thesis is
   published; **the developmental half — selectivity inverts against causal effect
   during formation — is not, and is sharper.**
4. **Phase 1's raw-effective-rank defect, as a phenomenon.** `2510.06477` **[S]**,
   ICLR 2026, *proves* massive activations produce representational compression.
5. Phase 1b's cone-collapse (anisotropy literature, NAACL 2021); Phase 2's OV sign
   spectrum (Elhage et al. 2021's copying-head statistic `Σλ/Σ|λ|`); Phase 2b's S/A
   decomposition and the `exp(A)` orthogonality identity; Phase 1c's Euler
   discretisation; 7e's "SVD order is a poor proxy" (FWSVD, ICLR 2022); 7d's
   super-additivity (the Hydra effect, 2023).

### What survives

Also ranked in `docs/LITERATURE.md` §3. The top four:

1. **The measured-null discipline**, and specifically §3.12-V3's anisotropy correction
   — ambient participation ratio **22 of 1024**, which kills the isotropic `k/d`
   baseline the adjacent subspace literature appears to use. The live Phase 6 rebuild
   already depends on it.
2. **Causal membership over every head, with structural *and* behavioural proxies
   demonstrated to fail** — narrowed by `2606.05378` to its developmental half, which
   is the better half.
3. **Force-typed interaction edges** (Phase 7). Every attention-graph paper found uses
   the attention weight as the edge; ours carries `A_ij · V x_j`, typed by sign and
   rotational channel. Novel in composition, from parts the field trusts.
4. **The `γ_β` residual as a null for a trained network** (Phase 1c). The field builds
   ODE transformers; nobody subtracts the ODE from a trained one.

### The cross-phase finding, which no single phase's file could state

Three phases independently produced **a clean, uniform null forced by the instrument**:
2b's `elim_rotation = 0.0` in 35/35 runs (`exp(A)` orthogonal, readout a function of
`X Xᵀ`); Phase 3's chance alignment (the sparse objective drives `Dᵀ D → I`); Phase 5's
two-of-six criteria contributing 0.0 on every model (wrong event schema, silently).
Two more phases have a metric confounded by a single token or a stale threshold
(Phase 1's raw effective rank, D1; Phase 1's Fiedler classification, D2).

> **An intervention whose readout is invariant under it returns a clean null at machine
> precision, and perfect cross-architecture uniformity is the tell. A scoring function
> whose terms can silently evaluate to zero still returns a ranked list, and the
> ranking still looks like a selection.**

**Five worked instances, from five phases, with numbers.** The searches found nobody
making this point about interpretability instruments. It needs no compute, and no
other group is positioned to write it, because no other group keeps its failures in
the repository. `docs/LITERATURE.md` §4.

### Two Phase-1 framing statements are stale rather than scooped

- **Metastability may no longer be an open problem.** `2410.06833`, *Dynamic
  metastability in the self-attention model* **[S]**, Oct 2024, is summarised as
  *proving* that particles remain trapped near a several-cluster configuration for an
  **exponentially long** period. `design-1.md` and `status-1.md` both rest on
  "Problem 1, which the paper poses as open". **What it is proved *for* is the thing
  to check** — a search summary elsewhere says "theoretical results are proven for
  d = 1", and we run at d = 1024. If the proof is d = 1 with identity weights, only
  the word "open" has to change.
- **The theory has had a causal-mask version since Nov 2024.** `2411.04990`,
  *Clustering in Causal Attention Masking* **[S]** (Karagodin, Polyanskiy, Rigollet,
  NeurIPS): the masked system **cannot be interpreted as a mean-field gradient flow**;
  convergence to a single cluster is proved for **arbitrary QK with V = I**; and it
  connects metastable states to the **Rényi parking problem**. `design-1c.md` calls
  `causal=True` "a departure from the theory" — since Nov 2024 it *is* the theory, and
  every Pythia number in this project has been compared against the unmasked one.

The parking link is the most useful single item the review found: **it predicts a
number of clusters as a function of `n`**, and Phase 1 has cluster counts per layer,
per prompt length, at 27 checkpoints, already on disk. Nobody has checked a
parking-derived prediction against a trained transformer, it costs no forward passes,
and `claims/adjudications/` holds zero entries against 39 registrations.

### The rung policy needs a rule 4, and it is a human call

`design-8.md` reserves **pythia-1b** because it is "not in the registry, never
measured". True of *this project*. **It is no longer true of the field:**
`2606.02378` and `2606.05378` both run Pythia-1B on the induction axis, with public
code at `skydancerosel/spectral-probe-circuits`.

**Proposed amendment (`p8_scale_ladder/lit-8.md` §3), not taken:**

> Add a rule 4: **a rung may be externally spent.** Before registering a prediction
> against a reserved rung, record which external measurements of that rung exist and
> what they report, in the registration itself. Reserve the rung against *our*
> measurement as before; reserve the *registration* against reading the external
> results first.

The argument cuts **in favour of keeping the reserve**: an independent measurement of
the same rung by a different group with different instruments turns an adjudication
into a three-way comparison — our prediction, our measurement, theirs. That is the
strongest adjudication site this project has had.

**Pythia-1.4b is unaffected.** No source found measures it on the induction axis, so
it is now the cleaner of the two reserved rungs and should be preferred for the first
registration.

### The thirteen cheap experiments

`docs/LITERATURE.md` §6 lists them in full. All are re-analysis of artifacts on disk or
weights-only, except the last. The five with the best ratio:

1. **The Rényi-parking cluster-count prediction** against 27 checkpoints of counts.
2. **The formation-point equation** from `2511.16893` **[S]** — batch size and context
   size predict the induction-head formation step, and Pythia holds both constant
   across the whole suite, so it predicts **the same step at every rung**. 70m and
   410m are both already measured. Free external adjudication.
3. **`PR_M` / `coupling_efficiency` (Phase 2d D2) against 7d's 384-head causal sweep.**
   The one instrument that could rescue a structural proxy for causal effect after
   `‖OV‖_F` failed at r² = 0.001 and ran backwards. This connection is in no other doc.
4. **Elhage's `Σλ/Σ|λ|` against `frac_repulsive`**, then on the checkpoint axis.
   Validation if they agree; a real dissociation if they do not.
5. **`schur` vs `svd` basis on `L11H14`.** Weights-only, named in the resume block on
   2026-09-10, **still unrun**, and a stated hold sits on `induction_rank_sweep`'s `r*`
   construction until it is done.

### Done in PROJECT.md (2026-09-16, second pass)

The review's findings are now inline at the claims they touch, not only here:

- **§1**, rung-policy table — the *externally spent* amendment, proposed and
  **not taken**.
- **§2** — a pointer at the top of the orientation section.
- **§3.12-O** — a note that O already implements the field-standard statistic,
  what the review leaves uncrossed (`frac_repulsive`, residual basis vs token
  basis), and the `L11H14` cross-reference.
- **§3.12-V** — a per-result literature-status table. V1's super-additivity and
  V4's SVD result must not be written up as findings.
- **§3.14.4-A** — ordering promoted to Phase 8's headline; the window demoted to
  a calibration check.
- **§8** — `docs/LITERATURE.md` added to "where to read next".

**One correction to the review itself, recorded rather than quietly fixed.**
`p2_eigenspectra/lit-2.md`'s first draft proposed computing Elhage's copying
statistic and putting it on the checkpoint axis as growth directions. **Both were
already done** — §3.12-N4 identified the statistic, §3.12-O ran it over 384 heads
at eight checkpoints. That file and `docs/LITERATURE.md` §6 now say so, and the
surviving direction is the narrower one: the token-basis score has never been
crossed against the residual-basis `frac_repulsive`.

### What must change in other documents

Not done in this session beyond the pointers; recorded so it is not lost.

- **`design-1.md`** — the "Problem 1 is open" framing, and the `causal=True`
  "departure from the theory" line in **`design-1c.md`**.
- **`status-1.md`** — "the phase's main new object".
- **`math-5b.md`** — add the id `2605.05115` for Wurgaft et al., and correct
  "approximately isometric" to **"scaled isometry"** wherever a threshold depends on it.
- **`design-8.md`** — the rung policy, if the amendment is accepted.
- **Project-wide** — cite `2312.10794` as *Bulletin of the AMS* **62(3), 2025**, not as
  the preprint, and check whether the published version renumbers the theorems the
  absent `MATH.md` §9 depends on.

### One design change this review proposes outright

**Phase 5b must not run its main arm before an ordered-concept positive control.**
`2605.05115`'s isometry correlations of **≈ 0.999** are on **weekday, month, letter and
age** — totally ordered, low-cardinality, densely sampled concepts, where a
spline-threaded manifold is nearly a 1-D curve. Our substitution puts **unordered
HDBSCAN cluster centroids with layer-varying membership** in their place. A negative
result under that substitution would be uninterpretable. Build one of their four tasks,
run our pipeline on it, and check that our unsupervised centroids recover their
supervised manifold first. **If they do not, the substitution has failed on the easy
case.** (`p5b_manifold_steering/lit-5b.md` §1.1.)

---

## 4. Open, analysed, not yet acted on: the scoring threshold

Investigated 2026-09-03, nothing changed in code. Recorded here because it is
measured and it affects every gate.

Every gate refuses when `attainable_floor > alpha`. The e-process validates at
`E >= 1/alpha`. With κ = 0.5 those are different requirements: for a claim
carrying k factors each at its floor, `p <= (κ·α^(1/k))^(1/(1−κ))`.

| k factors | required p | vs. the α the gates check |
|---|---|---|
| 1 | 6.25e-4 | **80× stricter** |
| 2 | 0.0125 | 4× stricter |
| 4 | 0.0559 | `p ≤ α` suffices |

`H-EMERGE`, `H-TRANSFER` and `H-RESIST` each have **exactly one** active
e-value row. `H-BRIDGE` and `H-OPERATOR` have four each.

**CLAIM-B on a perfect input returns p = 0.05 on all five seeds** — its arms
combine by max and the anchor arm is floored at `1/(n_controls+1)` with the 19
controls its dry run uses. `claims/audits/claim_b_p_i1_dry_run.json` already
carries `floor_equals_alpha: True`. That is e = 2.24 against a threshold of 20:
falsifiable via the RE-ANCHORS branch, **not validatable**, at any data.

`core/evalues.py:216`'s `required_p_for_rejection` already computes the right
number and **no gate calls it**. That is the whole defect in one line.

Best floors across the committed audits as e-value ceilings: `P-T1`/`P-M1` 183,
`CLAIM-C` 22.6, `P-I1` 22.4, `P-ST1` 1.58.

Measured, on 400k H0 replicates at N = 2000: a randomization p is discrete on a
known grid, so `e = (N+1)/(R·H_{N+1})` is a valid e-value directly (E[e] =
1.0005 under H0) and returns **244.7** at rank 1 where `calibrate(p)` returns
22.4. κ registered per prediction from its own floor (κ\* = 1/ln(1/p_floor))
gives 96.8. Neither fixes the tie floor: with heavy ties and the conservative
convention `paired_colocation_arm` correctly uses, E[e] = 0.28 under H0 — the
design cannot produce evidence under any scoring rule.

Nothing here has been implemented. The cheapest structural win is **more factors
per claim**, not a better calibrator.

---

## 5. Registered decisions, and disk that must not be deleted

### 5.1 Registered — do not re-decide from the code

1. **`P-I1`'s grid** — `REGISTERED_P_I1_SWEEP`, 19 steps: `0, 1, 2, 4, 8, 16,
   32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000`.
   A superset of the CLAIM-B sweep. All 19 tables are on disk.
2. **`P_I1_RELAY_OWNER = "matcher"`** — `p7_motifs/formation_gate.py:143`.
3. **Endpoints** — steps 0 and 143000 are in the grid, which is what
   `endpoint_flags` needs.
4. **`P_I1_DOMINANT_PROMPT = "repeated_tokens"`** — kept, carried beside the
   excluding-it series, reported and never scored. It holds **34,191** induction
   pairs against the next prompt's 2,873, because every repeated token pairs with
   every other; its 61% share is a fact about the prompt, not the checkpoint.
5. **`CLAIM-B`'s sweep** — `REGISTERED_CLAIM_B_SWEEP`, chosen 2026-08-28 from the
   computed feasible set (`POPPER_PLAN.md` §6r).
6. **`P-I3`'s matching** — `"score_and_layer"`, registered 2026-08-30 with both
   sides measured (§6s).

### 5.1b UNTAKEN disk decision: `plateau_attentions.npz` duplicates `attentions.npz` (found 2026-09-20)

**Measured, not inferred.** In every Phase-1 run directory both files are
written and they hold the **same numbers in two layouts**:

| file | layout | size at n ≈ 467 |
|---|---|---|
| `attentions.npz` | one key `attentions`, `(24, 16, n, n)` float32 | 147.6 MB |
| `plateau_attentions.npz` | 24 keys `attn_L0…attn_L23`, each `(16, n, n)` | 147.6 MB |

**Verified byte-identical across all 24 layers** on
`pythia-410m-step1_wiki_paragraph`. Together they are **91 % of a 325 MB run
directory**; `activations.npz` is 44.3 MB and the ten other files ~0.6 MB.

- **≈ 22 GB recoverable** across the existing 152 directories.
- **Halves the storage cost of every new run** — §3.54.2b's Stage 0 goes from
  74 GB to ~40 GB.

**NOT ACTED ON, deliberately.** Which reader uses which layout has not been
established, and the plateau path presumably wants the per-layer one. The cheap
fix is to stop *writing* one for new runs and relayout on load, not to delete
what exists. **Per §5.1's rule, this gets recorded before it is taken** — it is
recorded here and it is still open.

### 5.2 `results/`'s 132 GB must be kept — and as of 2026-09-19 it is kept on the HDD

`2026-08-12_05-01-35` (56.5 GB) and `p2_eigenspectra_2026-08-13_05-13-52`
(74.2 GB) each cover **27 steps on the PILOT schedule** — 11000, 13000, 15000,
17000, 19000, 100000, 120000 and so on. Those steps appear in nothing else on
disk, and `core/pythia_registry.py` keeps `PYTHIA_410M_PILOT_STEPS` loadable for
exactly this reason. `p1b_pilot`, `p2b_pilot`, `p2d_pilot` and `phase3` are
small; `phase3` is referenced from `archive/`.

**Both were MOVED to `/run/media/system/HDD_1TB/Mets_archive/` on 2026-09-19
and symlinked back into `results/`**, freeing ~130 GB on WDS_500 without
deleting a step that exists nowhere else. Reads through the symlink work
unchanged; the cost is that `results/` now depends on HDD_1TB being mounted,
and a run that cannot see it will fail to open those paths rather than
silently skip them. The move verified file count AND byte total on both sides
before removing either source — a mismatch would have kept the source and
moved nothing.

**Also cleared that day, after checking each against the committed record:**
the four superseded `CLAIM-C` arm directories from 2026-09-17 and early
2026-09-19 (14.4 GB — the v1-battery arms, replaced first by the HDBSCAN
re-run and then by v2), and `data/superseded/phase7_float32` (1.1 GB, the
pre-float64 Phase 7 tables recorded in
`docs/results_provenance_audit_2026-09-05.md` as moved aside on 2026-09-03).
**Not** the 2026-09-19 10:51/11:52/12:21/12:54 set, which
`claims/audits/claim_c_real_run.json` hashes.

### 5.3 The activation cache: measured before deleting (2026-09-09), re-checked and LEFT ALONE (2026-09-19)

**Decision 2026-09-19 (user): leave the 355 GB cache as it is.** The
re-check below found two of the 2026-09-09 premises now stale and two still
standing, and the two that stand are the ones that decide it.

**Stale now — the tree has changed under them.**
- *"No pythia-70m checkpoints were found on either volume, and the 70m model
  is not mirrored in `HF_HOME`"* — **false today**. `data/hf` holds
  `models--EleutherAI--pythia-70m` (5.0 GB) with **19 revisions cached**,
  step0 through step143000: exactly the sweep schedule. A 70m sweep is
  runnable offline right now.
- *"The caches are probably not regenerable"* — **half false**. `HF_HOME` now
  holds `gpt2-large` (3.1 GB), `pythia-1.4b` (11 GB) and `pythia-70m` (5.0 GB)
  beside `pythia-410m` (51 GB), so the **316 GB `gpt2_large` half is
  regenerable at compute cost**. `albert-xlarge-v2` is still absent, so the
  **39 GB `albert_xlarge_v2` half remains irreplaceable while offline** — that
  is the one-way door.

**Still standing, and decisive.**
- *It frees the wrong drive.* The cache is on HDD_1TB. After 2026-09-19's
  archive move (§5.2) that volume still has ~390 GB free, while WDS_500 is the
  constrained one. A 70m sweep at 1M tokens costs 127 GiB and fits in HDD_1TB's
  existing free space twice over, so replacing the cache buys nothing the
  programme needs.
- *`P6-R2` and `P6-R4` are REGISTERED on albert-xlarge-v2* and `CLAIM-C`'s
  statement names gpt2-large. Deleting their instrument's data without first
  checking what those entries still need is §6's "flattering subset" problem
  arriving through the disk.

The original 2026-09-09 assessment follows, with its arithmetic, which the
re-check did not disturb.

Asked whether to clear the 355 GB activation cache and refill it with
**real** pythia-70m activations. **Measured answer: do not delete — the premise
does not hold.**

*The size arithmetic, verified against the cache that exists.* Cost is
`tokens x d_model x n_states x 2` (float16): `gpt2_large` 13.2M tokens, d=1280,
10 states = **315.8 GiB**; `albert_xlarge_v2` 1.0M, d=2048, 10 states =
**38.5 GiB**; total **354.3 GiB** against `du`'s 355G. The formula is good, so
the projections below are too.

*70m is much smaller — 7.1x — but that is not the binding factor.*

| model | d | states | at 13.2M tokens |
|---|---|---|---|
| pythia-70m | 512 | 7 | **88.4 GiB** |
| pythia-410m | 1024 | 25 | 631.6 GiB |

**Token count dominates.** A 19-checkpoint 70m sweep with every layer costs
1680 GiB at Blog-1's 13.2M tokens, **127 GiB at 1M**, and **25 GiB at 200k** —
and the induction battery this project actually reads is **1,536 tokens**
(8 x 192). The old cache is enormous because it holds 13.2M tokens for Blog-1,
roughly a hundred times more than anything the current programme touches.

*Three reasons not to delete.*

1. **It frees the wrong drive.** `activation_cache` lives on **HDD_1TB, which
   has 440 GB free (50 % used)**. The constrained volume is **WDS_500 at 95 GB
   free (79 %)**. A 19-checkpoint 70m sweep at 1M tokens (127 GiB) fits in
   HDD_1TB's existing free space three times over.
2. **The caches are probably not regenerable.** `HF_HOME` holds **only**
   `models--EleutherAI--pythia-410m` (51 GB). No gpt2-large, no albert, no 70m —
   and `HF_HUB_OFFLINE=1`. Deleting is irreversible without network.
3. **`P6-R2` and `P6-R4` are REGISTERED on albert-xlarge-v2** (their entries cite
   "albert-xlarge-v2's exact shape" and "the 2026-04 ALBERT run"), and
   `CLAIM-C`'s statement names gpt2-large. Deleting their instrument's data
   without checking what those entries still need would be §6's "flattering
   subset" problem arriving through the disk.

*A premise that needs checking first.* **No pythia-70m checkpoints were found on
either volume**, and the 70m *model* is not mirrored in `HF_HOME` either. If
manually-made 70m checkpoints exist they are somewhere not searched; if they do
not, the sweep needs network access before it needs disk.

*If WDS_500 space is the real need*, the target is `data/` — **not** the
activation cache. *(2026-09-19: acted on. `data/superseded/` is gone, the four
superseded `CLAIM-C` arm directories are gone, and `results/`'s two protected
pilot sweeps moved to HDD_1TB under §5.2. `data/hf` has since grown to 70 GB —
gpt2-large, pythia-70m and pythia-1.4b were added for `CLAIM-C` — and
`data/phase12` is the volume's largest single consumer.)*

---

## 6. Untouched, and named so it is not mistaken for done

* `core/precision_policy.py`'s **P2** (Pythia ships fp16; an fp16-epsilon
  perturbation splits a genuinely real eigenvalue pair into a complex one) and
  **item 13** (the forward pass runs under bf16 autocast).
* **`real_frac`/`imag_frac` are NaN in every row of every table** — deliberate
  and correctly recorded (`rotational_channel: "absent"` in the manifest), not a
  silent gap. **Both open questions answered 2026-09-04, nothing changed in
  code.** *(CORRECTED 2026-09-19, §3.44: one registered prediction DOES need
  it — `P6-R2` compares `U_neg` against the imaginary subspace `U_A`, which no
  artifact carries. The sentence below is right about Phase 7's rows and wrong
  as a statement about the registry.)* No registered prediction needs the rotational channel: `P-I2` names
  only the sign channel (`U_pos`), and `P-I1`/`P-I3`/`P-I4` don't reference
  `real_frac`/`imag_frac` at all. And no consumer reads them for a computation —
  grepped across every `.py` file: `run_7.py` writes the NaN, `p7_io.py` is the
  seam that would fill it (`rotational_channel_from_blocks`, unwired), and
  `core/interactions.py` / `core/artifacts.py` only carry the schema and
  validation. None of `motif_stats.py`, `formation_gate.py`,
  `formation_curve.py`, `cross_head_gate.py`, `patching_gate.py`, `events.py` or
  `motif_alphabet.py` touch either column. (`core/dual_reading.py` computes
  fields with the same names but is an unrelated per-particle primitive from an
  earlier phase, not a Phase 7 consumer.) So the columns are exactly what §6
  asked whether they were: schema nothing fills and nothing reads. Left as is —
  removing them would touch `InteractionTable`'s hashed schema for a channel
  Phase 2b's `extract_schur_blocks` could still wire in later, and no registered
  prediction is asking for the removal either. `p7_io.rotational_channel_from_blocks`
  stays the seam if that changes.
* **The phase-7 manifest records no library versions.** §1's first trap is the
  argument for adding them; not done, because it changes the manifest schema and
  every record that hashes it.
* **The in-memory categorical option** (int8 codes for `model`/`prompt_key`/
  `pair_type`, 5.49 GB → 1.89 GB expanded). Compression fixed disk and does
  nothing for RAM.
* **Eleven predictions are adjudicable in principle and
  `claims/adjudications/` is empty.**
* **`data/analysis/` is git-ignored, so every number quoted in a committed
  document is reproducible only by re-running its producer.** `data/` is
  ignored by `*` (§1), which is right for the 118 GB of bulk and wrong for the
  small JSON series and the builder scripts sitting beside them:
  `dissipation_series`, `dissipation_sublayer_series`, `ov_per_head_series`,
  `relay_null_series_k50`/`_k100`, `behavioural_series`, and the five/six
  `build_*.py` / `*_analysis.py` scripts. `POPPER_PLAN.md` §6w, §3.8 and
  `p2_eigenspectra/status-2.md`'s dated section all quote figures whose only
  provenance is a file outside version control. The producers are tracked and
  deterministic, so this is recoverable rather than lost — but it is inference
  from a rerun, not a record, which is the same class of gap
  `docs/results_provenance_audit_2026-09-05.md` §3.1 raises against Phase 2's
  missing manifest. **Fix: un-ignore `data/analysis/*.json` and
  `data/analysis/*.py`** (a `!` rule under the `data/` ignore), or move them to
  a tracked `results/analysis/`. Deferred deliberately — it is a chore, and the
  decision on which of the two shapes to take is not yet made.
* **Process tooling, decided 2026-09-12, not yet built.** CodeRabbit is
  **installed** on the repo (reviews PRs from here on) — that part is done.
  Still open: CI/CD is one gate (`./scripts/check.sh gate`, tier 0 + 1) with no
  tiered required-checks policy on GitHub itself, and there is no TDD
  discipline for new phase work — tests get written to validate a result after
  the fact (or not at all for one-off exploratory scripts), not before the
  code that produces it. Scope not yet decided: whether "tighten TDD" means a
  repo-wide policy or just raising the bar for new `tools/run/` runners going
  forward. Deliberately deferred to its own session rather than mixed into
  research work. **The working agreements that came out of the same decision
  are in `CLAUDE.md`** (new 2026-09-12): update this file as work closes rather
  than at session end, open PRs at boundaries a reviewer can get through, and
  run a literature scan at two triggers only — a phase/subphase opening before
  its `design-N.md` freezes, and before an entry lands in
  `claims/registry.json`.

---

## 7. Reproducing anything

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf METS_RESULTS_DIR=$PWD/data/phase12 HF_HUB_OFFLINE=1

./scripts/check.sh gate     # tier 0 + 1, what gates a merge; ~35 s
./scripts/check.sh all      # adds the deps tier; ~2:15

bash tools/run/sweep.sh     # resumable; all 19 steps present, so it is a no-op
python tools/run/curve.py   # ~2:43, writes data/analysis/curve.json AND
                            # data/analysis/formation_series.json

python3 -m tools.run.behavioural --write    # ~1:06, reads 19x8 attentions.npz,
                                            # writes data/analysis/behavioural_series.json
python3 -m tools.run.behavioural --check    # structural checks on the written series

python3 -m tools.p_i1_attainable_floor --write     # ~0.2 s, needs the series
python3 -m tools.p_i1_attainable_floor --check     # needs no data

METS_NULL_REPLICATES=50 python3 -m tools.run.relay_null
                            # ~1:33 at 50 reps (measured); ~3:00+ at 100 --
                            # scales close to linearly in the replicate count.
                            # Writes data/analysis/relay_null_series.json.
                            # Prints one line per checkpoint as it goes, so a
                            # long run can be judged and killed early; the
                            # output file is only written at the very end, so
                            # killing it mid-run loses nothing already on disk
                            # (the PREVIOUS successful --write, if any, is
                            # untouched until the new run's last line prints).
python3 -m tools.score_p_i1
                            # needs relay_null_series.json and
                            # behavioural_series.json; prints P-I1's p-value

METS_REPO=$PWD METS_DATA=$PWD/data python3 -m tools.run.dissipation
                            # ~22 min. Dissipation identity Tier A per
                            # (step, prompt, layer) from activations.npz +
                            # the on-disk OV Schur projectors, no forward
                            # pass. Writes data/analysis/dissipation_series.json.
METS_REPO=$PWD METS_DATA=$PWD/data python3 -m tools.run.dissipation_sublayer
                            # ~20 min, 133 forward passes (loads pythia-410m
                            # checkpoints). Exact attn/FFN split + per-head
                            # roll-up. Writes dissipation_sublayer_series.json.
# Panels + the per-head co-location test are rebuilt by the scripts in
# data/analysis/ (build_colocation_panel.py, build_dissipation_panel.py,
# build_earlylayer_and_512to1k.py, tierB_panel_and_colocation.py).

METS_REPO=/run/media/system/WDS_500/Mets METS_DATA=$METS_REPO/data \
  python3 -m tools.run.induction_rank_sweep
                            # ~4 min, loads pythia-410m-step4000. Stage 1 of
                            # sec 3.11: OV rank sweep of L7H8 in the SVD, Schur
                            # and random bases. Writes induction_rank_sweep.json.
METS_REPO=/run/media/system/WDS_500/Mets METS_DATA=$METS_REPO/data \
  python3 -m tools.run.induction_subspace_characterize
                            # ~5.5 min. Stage 2: L7H8 OV r* characterisation
                            # (Schur sign, phi, Henrici, rank-1 mode) calibrated
                            # against all 16 layer-7 heads + 24 random OVs.
                            # Copying read causally (final LN present). Writes
                            # induction_subspace_characterize.json.
                            # NB METS_REPO must be the canonical path (matches
                            # sys.prefix); $PWD via the bind mount fails the
                            # interpreter check.
```

### 7.1 `curve.json` is the artifact that gets diffed

Every change to the storage or estimator layer is verified by re-running
`curve.py` and diffing `curve.json` against the pre-change copy. It has come
back **0 differences** three times: after the single-pass rewrite and table
compression, after the 176 GB migration, and after adding the per-head series
dump. That is why the series went into a **second** file — a file whose content
is diffed is not the place to add a key.

### 7.2 Records that carry file hashes

Three records hash `core/changepoint_colocation.py` or
`p7_motifs/formation_gate.py` and must be rewritten whenever either changes. The
gate fails loudly if they are stale, which is the intended behaviour.

```bash
python3 -m tools.dry_run_claim_b_p_i1 --write      # ~4 min
python3 -m tools.claim_b_grid_feasibility --write  # ~3:45
python3 -m tools.p_i1_attainable_floor --write     # ~0.2 s
```

### 7.3 One gotcha

`pythonpath = .` in `pytest.ini` applies to pytest only. A plain
`python script.py` needs `PYTHONPATH` set, which `tools/run/curve.py` does for
itself.

### 7.4 The math is symbolically checked, and `MATH_SPECTRAL_OT.md` is clean

`tools/math_checks/` holds seven `sympy` scripts, **47 checks, all passing as of
2026-09-12**. Each exits non-zero on any failure, so they are drop-in for a CI
tier whenever §6's CI/TDD item gets picked up.

```bash
for f in tools/math_checks/*.py; do python3 "$f"; done   # a few seconds total
```

**Nothing wrong was found in the source.** Every closed-form derivation in
`MATH_SPECTRAL_OT.md` §2.1, §2.4–2.4.5, §2.5.1–2.5.4, §3 and §5.2–5.3(b) holds
exactly, including the two places a sign or transpose error was most likely: the
§2.4.4 sum/difference identities under the `M` vs `−M^T` flip, and §5.2's
"the factor of 2 and the `beta` both cancel" gradient. §5.2 is checked by
**direct symbolic differentiation of `E_beta`**, not by re-deriving the same
algebra, and the chain was verified end to end — `core/metrics.py:117`
implements the energy the doc attributes to it, so this is not a
producer/consumer mismatch of the kind `MATH_INDEX.md`'s pattern 5 names.

**Read the scripts' docstrings for what each does NOT prove.** A general-`n`
matrix identity instantiated at `n = 4` is evidence, not a proof; the §2.5.2
polar-retraction and §2.4.5 Bendixson checks are numeric instances of classical
results rather than derivations of this project's own.

**Why they exist, stated as a hit rate.** Of the six corrections in
`MATH_INDEX.md`'s "Corrections owed to the source", **four were algebraic or
arithmetic claims this class of check catches mechanically** (the Hellinger
range, the Henrici real-vs-complex-Schur gap — re-derived here as exactly
`(b−c)²` per complex-conjugate block — the V-score weights, and
`UPDATE_PLAN.md` §5.6's trace contraction). The other two were structural, and
need a human. So the cheap win is to check a derivation **before** it is written
into a document, not after.

---

## 8. Where to read next

| Question | File |
|---|---|
| What this is, which document answers what, how to install and run the gate | `README.md` |
| Which phase lives where, what is archived | `INDEX.md` |
| **What has been done before, per phase, and where we can still grow** | **`docs/LITERATURE.md`, then `<phase>/lit-N.md`** |
| Why a construction is the way it is | `POPPER_PLAN.md` §6a–§6t |
| What is pre-registered, and its falsifier | `PREDICTIONS.md`, `claims/registry.json` |
| Which predictions can carry an e-value, and the order to build a null in | `claims/EVALUABILITY.md` |
| Phase 7's translation table and motif alphabet | `p7_motifs/design-7.md` |
| A phase's current state | `<phase>/status-N.md` |
| The dissipation-identity run — what it is, Tiers A/B, v2 list | `docs/dissipation_checkpoint_axis_scoping.md`, §3.8 |
| Are the on-disk phase12/phase7 results stale? | `docs/results_provenance_audit_2026-09-05.md` |
| The pythia-70m dense-onset run, and how it could plug in | §3.9 |
| What is already published, and what it hands us | `p8_scale_ladder/literature-8.md`, §3.16 |
| How the working agreements read (handoff cadence, PR size, scan triggers) | `CLAUDE.md` |
| Whether a closed-form derivation has been checked | `tools/math_checks/`, §7.4 |
| What changed and when | `git log` |
