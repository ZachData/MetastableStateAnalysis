<!-- archive/PROJECT-start-here.md -->
# PROJECT.md §1 "Start here" — archived 2026-09-22

Moved verbatim out of `PROJECT.md` (its lines 31–937 at `64a4087`). Frozen: the
"Resume here" blocks, the machine, the tree and the traps as they stood on
2026-09-22. `STATE.md` replaced them as the startup file; the machine and
hazards that still bite are in `STATE.md`, the rest in `LESSONS.md`.
Old → new paths: `archive/MOVED.md`.

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

### Resume here (2026-09-22 — #60 merged; CI's float32 tripwire fixed in its own PR; next is Stage 0, the battery to 20 prompts)

**This block is the handoff.** Everything a session needs to continue is here
or one link away; the sections below it are orientation and history.

**Git (2026-09-22). `main` is at `086bdd1` — PR #60 merged, and with it #59's
four free rows, their runners, `status-10.md` and §3.51–§3.54.** Verified with
`git merge-base --is-ancestor 7717fa3 origin/main`. The superseded remote
branches `claude/p10-free-rows`, `claude/aca-phase-9-planning-rvzw3x` and
`claude/attention-collapse-augmentation-qsxwg8` carry nothing unique and can be
deleted from GitHub.

**CI on `main` was red from #57 to #60; the fix is on
`claude/float32-tripwire-ci`, its own PR.** Two tests asserted that a float32
Schur projector residual is *worse* than `PROJECTOR_TOL = 1e-6`, at `d = 128`.
At that size float32 roundoff is itself ~1e-06, so the assertion was a coin
flip on the host's BLAS: forcing OpenBLAS kernels locally with
`OPENBLAS_CORETYPE` (Prescott / Sandybridge / Haswell / Zen, 1 and 4 threads,
8 seeds) gives **6.3e-07 .. 1.4e-06**, and seed 0 on Prescott/Sandybridge gives
8.34e-07 — the runner's 8.64e-07 is in that range. **Re-derived rather than
marked machine-dependent**: a marker would have switched the tripwire off on exactly
the hosts where it fires and left it a coin flip everywhere else. The test now
builds at **`d = 1024`**, the width the defect was measured at on real weights,
where the same sweep gives float32 **3.3e-06 .. 5.1e-06** (≥ 3.3× clear) and
float64 ~2e-15. The margin is measured, not derived — a BLAS outside those
kernels could still land lower, and the docstring says so.

**The lesson, and it has now cost twice.** A PR targeted at another PR's branch
does not retarget itself safely once the base merges; merging it then lands the
work somewhere that is not `main`. **Target `main` and merge in order**, or
check `git merge-base --is-ancestor <tip> origin/main` before assuming a merged
PR's content actually reached `main`.

The user merges from GitHub. **Folders: the main tree `Mets` plus at most one
task worktree, `../Mets-work`**, removed (`git worktree remove ../Mets-work`)
once its PR merges. `../Mets-p10` and `../Mets-claim-c` were removed on
2026-09-22, both clean and merged. Only `data/` is untracked (the HF cache and
run directories — never `git add -A` under it).

**Run the Phase-10 runners from a worktree with `METS_REPO` pointed at it.**
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

