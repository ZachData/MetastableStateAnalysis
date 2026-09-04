<!-- HANDOFF.md -->
# HANDOFF — session of 2026-09-03

Written to be pasted into, or read at the start of, a fresh session. It records
what changed, what was measured, what is still open, and where everything
lives. Every number below is measured on this machine.

Branch: `claude/rescaler-cache-identity-test`. Nothing is merged and no PR is
open.

**The registered sweep is COMPLETE — all 19 steps.** The question the previous
session left standing is answered, and the answer is in §3.

**Step 1 of EVALUABILITY.md's order — the attainable floor — is now DONE**, and
it found two things before any control existed, which is what the order is for.
§5 has them. `P-I1` is still `needs-null`; the relay-count null is still the
author's decision and is still not built.

**Everything moved.** Paths in the previous handoff are all dead. See §1 first.

---

## 1. The machine, and why every path in the last handoff is wrong

The previous session ran with the repo at `/mnt/mets` and a bulk volume at
`/mnt/vm_storage`. **Neither mount exists.** They are the same two
filesystems, mounted by label:

| | |
|---|---|
| Repo | `/run/media/system/WDS_500/Mets` (NVMe, `/dev/nvme0n1p1`, 458 GB) |
| venv | `<repo>/.venv` — torch 2.13.0+cpu, transformers 4.57.6, numpy 2.5.2, scipy 1.18.1 |
| CPU / RAM | 16 cores, 31 GB |
| Free after this session | 95 GB on WDS_500, 440 GB on HDD_1TB |

**The generated tree now lives under the repo**, on one root, and
`/run/media/system/HDD_1TB/vm_storage` is empty:

```
Mets/
├── data/                          # all generated bulk; git-ignored by `*`
│   ├── hf/                 51 GB  # HF_HOME — 33 mirrored 410M revisions
│   ├── phase12/           118 GB  # METS_RESULTS_DIR — phase 1 and phase 2
│   ├── phase7/            6.1 GB  # the 19 interaction tables
│   ├── analysis/                  # curve.json
│   ├── logs/
│   └── superseded/phase7_float32/   1.1 GB  # pre-float64 tables
├── results/               132 GB  # the PILOT grid — see §6, do not delete
└── tools/run/                     # sweep.sh, curve.py — now TRACKED
```

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf
export METS_RESULTS_DIR=$PWD/data/phase12
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
```

`METS_VOL` **is gone deliberately.** It named a VM's scratch volume, which is
the same class of path as `/mnt/mets`: one that encodes transient
infrastructure and fails silently when the infrastructure changes. Both
scripts now derive everything from `METS_REPO`, with `METS_DATA` as the single
override.

`transformers` is still pinned `<5`: on 5.x GPT-NeoX moved rotary parameters
into `config.rope_parameters` and `core/rope.py`'s `rotary_pct` default then
fires, reporting `rotary_ndims=64` where pythia-410m rotates 16.

---

## 2. Three ways the environment lied, all silent

Each cost real time and none announced itself. They are one family: **a check
that reads the shape of a thing instead of its content.**

### 2.1 `activate` set the variable and gave the wrong interpreter

`.venv/bin/activate` carries the absolute `VIRTUAL_ENV` recorded at creation:
`/mnt/mets/.venv`. After the remount it prepended a directory that does not
exist, set `VIRTUAL_ENV`, and returned 0. `python` then fell through `PATH` to
a conda env at `miniforge3/envs/mets` — **Python 3.10 with a pre-4.45
transformers**, against the pinned 3.14 / 4.57.6.

Phase 7 for step143000 ran under it for two minutes before this was caught.
Had that table landed, **nothing in the artifact would have recorded it**: the
phase-7 manifest stores `git_sha`, `hf_revision` and `seeds`, but no library
versions. One checkpoint of a 19-point curve computed against a different
transformers, with no trace.

Fixed by rewriting the 30 text files in `.venv/bin` that named the old path.
`sweep.sh` now **asserts** `sys.prefix` and the torch/transformers versions
rather than trusting activation, because `sys.prefix` is the only thing that
answers "which interpreter".

### 2.2 The phase-1 reuse search returned a phase-2 directory

Phase 1 and phase 2 both write `${model}_${prompt}` subdirectories into
`METS_RESULTS_DIR`. `find_existing_p1` counted those directories, so when
phase 2 for step143000 already existed and was the newer of the two, it was
handed to `--phase1-dir`. `run_2` started over on the wrong input and did not
error.

Both are now identified by a file only that phase writes — `activations.npz`
per prompt for phase 1, `phase2_verdict.json` per prompt plus top-level
`ov_decomp_${M}.npz` for phase 2 — and **phase 2 is reused when present**, for
the same reason phase 1 is. Confirmed by dry-run: the fixed selectors
reproduce exactly the two paths recorded in step54000's `p1_dir.txt` and
`p2_dir.txt`.

That reuse turned step143000 from a 35-minute run into a 10-minute one; phase
1 and phase 2 for it had both completed on 2026-09-01 before the sweep died.

### 2.3 `recompress_tables.py` ate its own temporary file

An interrupted run leaves `<name>.npz.recompress-tmp.npz` beside the original.
**That name matches the tool's own `*.npz` glob**, and sorted order places it
immediately after the real table — whose successful `os.replace` then consumes
it, so the next `stat()` raises `FileNotFoundError` and kills the batch. It
aborted after `step16` with `step4` untouched and no record of what had been
skipped.

Two changes, both mutation-checked (revert either half and its test fails;
reverting the first reproduces the symptom verbatim):

* `collect()` excludes the tool's own temp suffix, named once as `TMP_SUFFIX`
  because two places need it. Excluded rather than deleted: a leftover temp is
  evidence a run was interrupted, and this tool is not what should decide to
  discard it.
* A file that vanished between listing and rewriting is a reported skip, not
  an exception. Aborting there loses every later file and says nothing about
  which.

`tests/test_tools_recompress_tables.py` is new — the tool had no tests at all.

### 2.4 One near-miss worth recording

Reading an empty log, this session concluded a recompression launch had failed
and started a second one on the same directory. **Two writers to one temp
path** could have replaced a good table with a corrupt one. Nothing was lost —
both were killed and `step4`'s original verified (20 arrays, 19,077,120 finite
weights) — but "no output yet" is not evidence a job died. The check is
`pgrep`, not the log.

---

## 3. The answer: the degeneracy clears

The previous handoff's §6 named the first thing to check when the tables
landed — whether the six interval midpoints inside (1000, 54000) give more
than one distinct change centroid across heads. **They do, decisively.**

| `relay_owner` | heads scored | distinct centroids | span (log-step) | sd |
|---|---|---|---|---|
| `tag_writer` | 102 | **68** | 3.8898 – 4.9439 | 0.2813 |
| `matcher` (registered) | 116 | **79** | 4.1604 – 4.9439 | 0.2374 |
| `both` | 122 | **86** | 4.1604 – 4.9439 | 0.2641 |

Against the twelve-step CLAIM-B grid, where all three owners gave **ONE**
distinct centroid across 68/80/87 heads, because every head's change mass fell
in the single interval (1000, 54000].

**§4 decision 1 does not come back with a denser grid.** The five log-spaced
fills bought exactly what they were registered for.

The relay counts show why:

| step | relays | ex-`repeated_tokens` | heads (matcher) |
|---|---|---|---|
| 0 – 2000 | 0 | 0 | 0 |
| 4000 | 15,030 | 5,563 | 9 |
| 8000 | 232,568 | 83,659 | 25 |
| 16000 | 509,646 | 216,528 | 46 |
| 32000 | 1,176,478 | 582,796 | 63 |
| 54000 | 2,560,483 | 1,008,553 | 80 |
| 143000 | **2,407,556** | **1,465,052** | **114** |

### Two things that must not be skipped when this is scored

**The series is not monotone.** The total relay count FALLS from step 54000 to
143000, while heads carrying relays rises 80 → 114 and the ex-`repeated_tokens`
count keeps climbing. The signal spreads across heads and away from the one
combinatorially-loaded prompt while the raw total drops. `change_profile`
rectifies, so that decline lands in `reverse_change_mass` and will inflate
`noise_mass_share_estimate` on a series whose reverse motion is real structure,
not noise. That field is documented "REPORTED, NEVER SCORED" — this is the
case that earns the distinction.

**"0 refused (no rise)" is a property of the axis, not of the model.**
`tools/run/curve.py` tracks only heads carrying at least one relay somewhere in
the sweep: **116 of the model's 384** (24 layers × 16), all at **layers 8–23**,
which is what `L2 > L1` forces for a stage-2 matcher. On the dense behavioural
axis `formation_curve_payload` actually builds, the other 268 heads are
all-zero and `change_profile` refuses each one. The 79 distinct centroids stand
for the 116 heads that form.

*Corrected 2026-09-03: "refuses each one" is what a per-unit refusal would do
and there is no per-unit branch. The first all-zero unit takes the whole gate
with it and `p_value_p_i1` returns no p-value at all. Measured, in §5.1.*

The real payload still needs `behavioural_induction_score` per step, which
reads phase-1 `attentions.npz` per prompt — a separate computation from the
relay series, and one that only matters once §5's null exists.

---

## 4. Compression, migration, deletion — and what each was verified against

### 4.1 The tables compressed 15–16×, as §8 of the last handoff predicted

| | before | after | ratio |
|---|---|---|---|
| 16 sweep tables | 84.9 GB | 5.4 GB | 15.0–16.0× |
| 5 `phase7_float32` tables | 16.4 GB | 1.1 GB | 15.0–15.7× |

Two workers, ~28 s per table — **not** the ~3 minutes the last handoff
estimated. Parallelism is RAM-bound, not CPU-bound: the tool holds the payload
and the verification copy simultaneously, ~11 GB peak per worker.

### 4.2 The migration was copy-verify-then-delete, never a bare `mv`

It crossed filesystems, so it was a copy anyway, and an interrupted `mv` of
176 GB leaves the source half-deleted with no record of which half. Each tree
was rsynced, then its file count and byte total compared, and only then was the
source removed:

| | files | bytes |
|---|---|---|
| `phase7` | 76 | 6,506,076,204 |
| `superseded/phase7_float32` | 16 | 1,077,868,550 |
| `hf` | 131 | 54,421,658,932 |
| `phase12` | 8,105 | 126,007,651,040 |

`hf` re-verified after the move: **33 `model.safetensors`, and all 33
registered steps resolve under `HF_HUB_OFFLINE=1`, 0 failures.**

### 4.3 45 GB deleted, on evidence

Deleted with the author's explicit approval, after checking coverage rather
than dates:

* **18 directories, 41 GB** — the single-step 2026-08-31 runs that were in
  `Mets/results/`. Every step they cover is in the current 19-step
  `data/phase12` tree, and the current version is the **float64** redo; these
  are the pre-`dd6b0de` float32 first attempts. Same artifacts whose phase-7
  output was already set aside as `stale_float32`.
* **4.2 GB** — two aborted partials in `phase12`
  (`p2_eigenspectra_2026-08-31_10-59-06`, step-2 phase 2 at 2.99 GB against its
  complete twin's 3.98; `2026-09-01_14-32-24`, step-8000 phase 1 at 1.16 GB
  against 2.01) plus two empty directories.

### 4.4 What was NOT deleted, and why

`results/` still holds **132 GB** and must keep it. `2026-08-12_05-01-35`
(56.5 GB) and `p2_eigenspectra_2026-08-13_05-13-52` (74.2 GB) each cover **27
steps on the PILOT schedule** — 11000, 13000, 15000, 17000, 19000, 100000,
120000 and so on. Those steps appear in nothing else on disk, and
`core/pythia_registry.py` keeps `PYTHIA_410M_PILOT_STEPS` loadable for exactly
this reason. `p1b_pilot`, `p2b_pilot`, `p2d_pilot` and `phase3` are small;
`phase3` is referenced from `archive/`.

### 4.5 `curve.py` did four passes where one would do

It called `find_relays(t.filter(prompt_key=p))` per prompt (8 filters, each
allocating a copy of a slice of a 5.5 GB table) and then
`per_head_relay_strength(t, o)` for three owners — and that helper calls
`find_relays(t)` itself. `RelayInstance` carries `prompt_key`,
`(layer_1, head_1)` and `(layer_2, head_2)`, so every number is a projection of
ONE pass.

Combined with compression the run went **~20 min at 26% CPU (I/O-bound) →
2:43 at 89% CPU**, peak RSS 9.6 GB.

### 4.6 The whole chain is verified by output, not by assertion

`curve.json` was captured before any of this and diffed after each stage:

| | differences |
|---|---|
| single-pass rewrite + compressed tables | **0** |
| after migration to the new layout | **0** |

Every relay count, per-head strength and change centroid is bit-identical
across compression, a 176 GB move, and 45 GB of deletion.

---

## 5. The blocker, and step 1 of the order is now done

**The relay-count null still does not exist.** `formation_gate` requires the
series to be the excess above the N1/N2 offset-null envelope, and
`core/qk_offset_null.py` computes N1/N2 for the **QK antisymmetry statistic**,
not for relay counts. `formation_curve.assert_gate_ready` refuses the series,
correctly.

`claims/EVALUABILITY.md` prescribes the order: compute the attainable floor,
name what the statistic degenerates on, check what the measurement grid
contributes, and only then build the control. **All three steps before the
control are now done.**

* **Step 2 (2026-09-01).** Across the 8 battery prompts at step 54000 the raw
  relay count against the prompt's own induction-pair supply runs **r =
  +0.9958** — 99% of the cross-prompt variance is the prompt's combinatorics,
  not the model's circuitry. Excluding `repeated_tokens`, +0.8908. Nothing else
  is close: n_tokens −0.39, n_same_content −0.36, n_distinct_tokens −0.79.
* **Step 3 (2026-09-01).** §3 above is its answer: the grid contributed the
  entire previous failure, and the fills fixed it.
* **Step 1 (2026-09-03).** `tools/p_i1_attainable_floor.py` →
  `claims/audits/p_i1_attainable_floor.json`, `POPPER_PLAN.md` §6t. Two
  findings, below.

### 5.1 The gate cannot score the axis the pipeline builds

`formation_curve_payload` takes its head axis from the **behavioural** series,
which is dense over all 384 heads (24 × 16), and zero-fills the relay side. But
`paired_colocation_arm` calls `change_profile` on every unit with **no per-unit
skip**, and `change_profile` refuses a series with no rise. 116 heads carry
relays and **268 never do**, so the arm refuses on the first all-zero unit and
`p_value_p_i1` returns **no p-value at all**. On the 116 forming heads the
identical input emits — which is what says the refusal is about the axis, not
the data.

§3 of this document said `change_profile` "refuses each one" of the 268. That
is what a per-unit refusal would do and is not what happens: there is no
per-unit branch.

**And the message names none of it** — "the series has no rise anywhere in the
sweep", no arm, no head index, no unit count. Pinned as it is, in
`tests/test_p_i1_attainable_floor.py::test_the_refusal_names_neither_the_unit_
nor_how_many`. Fixing it properly means either reducing the axis to the forming
heads or giving the arm a per-unit skip with a count, and **both change what
P-I1's unit is**, which `PREDICTIONS.md`'s first Phase 7 adjudication constraint
fixes at the head. Author's call, with the measurement beside it.

### 5.2 The floor had two halves and the arm reported the wrong one

`paired_colocation_arm` reported `1 / n_draws` alone. The statistic is
`-mean|ca - cb[p]|`, and permuting units within a class of equal locations
leaves it **exactly** unchanged, so every pairing ties a coset of order
`prod(m!)` and no input can express a p below `prod(m!) / n!`.

Measured on the registered 19-step grid with nine of ten units sharing one
location: reported **0.000500**, attainable **0.100000** — a factor of **200**,
above α, emitted as a p-value with no refusal. Seven of ten tied is 0.00139 and
emits legitimately: **the two halves cross within two heads.**

`p7_motifs/steering_gate.py` has carried both halves since 2026-08-26 with a
test pinning them, and the shared estimator two gates over did not.
`core.changepoint_colocation.pairing_floor_report` now owns both, and the arm
refuses when the max exceeds α with a message saying that raising the draw
count does not fix it. It **adds** a refusal rather than lifting one, and it
costs nothing where it does not fire.

**On the real head set it does not bind.** 116 heads, tie floor 10^-148. But
"79 distinct centroids" is **77 singletons, one class of three and one class of
thirty-six** — 31% of the heads still put their change in one interval — and
the tying subgroup is dominated by the largest class, not by the count of them.

### 5.3 What that constrains, which is the point of doing it first

A relay-count null turns the series into an above-null **excess**, and a head
whose excess stops rising leaves the scored set. So the null chooses `n_units`,
and `n_units` with the tie structure chooses the floor:

| survivors | max tied | tie floor there |
|---|---|---|
| 4 | 1 | 0.0417 |
| 6 | 4 | 0.0333 |
| 8 | 6 | 0.0179 |
| 12 | 10 | 0.0076 |
| 19 | 17 | 0.0029 |
| 20 | **19** | 0.0500 |

The jump at twenty is arithmetic: `k = n − 1` gives exactly `1/n`, so
all-but-one-tied clears 0.05 from n = 20 and fails at 19. Full table in the
record.

> **The relay-count null must leave at least four heads with a rising above-null
> excess, and among them no more than k sharing one change location.**

### 5.4 Unchanged, and still the author's

The constraint from step 2 stands: a relay-count null that does not hold
`n_induction` fixed per prompt is testing whether the prompt has induction
pairs, which is known before the model runs. What survives normalising is not
nothing — relays per induction pair spans 45 to 133 — and that residue is where
a formation signal would have to live.

**§3's non-monotonicity is still new input to this design.** A null built to
explain a monotone rise will not describe a series that peaks between 54000 and
143000 while spreading across 34 more heads.

The obvious shape remains a degree-preserving rewiring within each
(context, layer, head) that keeps each head's edge count and attractive
fraction, randomises which particles the edges connect, and preserves the
induction structure. **That is a design choice for the author, not one to
register from the code. Do not start it from the control.**

**What is also not done:** the behavioural arm. `behavioural_induction_score`
reads phase 1's `attentions.npz` per prompt per checkpoint and has never been
run over the sweep. Every relay-side number in the record is measured on the 19
tables; the B side in it is synthetic, and is sound only because
`paired_colocation_arm` profiles the A side first — which the record checks
rather than assumes.

---

## 6. Registered decisions — unchanged from 2026-09-01

1. **P-I1's grid** — `REGISTERED_P_I1_SWEEP`, 19 steps: `0, 1, 2, 4, 8, 16, 32,
   64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000`. A
   superset of the CLAIM-B sweep. **All 19 tables are now on disk.**
2. **`P_I1_RELAY_OWNER = "matcher"`** — `p7_motifs/formation_gate.py:143`.
3. **Endpoints** — steps 0 and 143000 in the grid.
4. **`P_I1_DOMINANT_PROMPT = "repeated_tokens"`** — kept, carried beside the
   excluding-it series, reported and never scored. It holds **34,191** induction
   pairs against the next prompt's 2,873, because every repeated token pairs
   with every other; its 61% share is a fact about the prompt, not the
   checkpoint.

---

## 7. Untouched, and named so it is not mistaken for done

* `core/precision_policy.py`'s **P2** (Pythia ships fp16; an fp16-epsilon
  perturbation splits a genuinely real eigenvalue pair into a complex one) and
  **item 13** (the forward pass runs under bf16 autocast).
* **`real_frac`/`imag_frac` are NaN in every row of every table** — deliberate
  and correctly recorded (`rotational_channel: "absent"` in the manifest), not
  a silent gap. Now that tables are compressed the storage cost is negligible,
  but the two open questions stand: does any registered prediction need the
  rotational channel, and if none does, are those columns schema no producer
  fills and no consumer reads? `p7_io.rotational_channel_from_blocks` is the
  seam.
* **The phase-7 manifest records no library versions.** §2.1 is the argument
  for adding them; it was not done, because it changes the manifest schema and
  every record that hashes it.
* The in-memory categorical option (int8 codes for `model`/`prompt_key`/
  `pair_type`, 5.49 GB → 1.89 GB expanded) is still not done. Compression
  fixed disk and does nothing for RAM.

---

## 8. Reproducing anything

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf METS_RESULTS_DIR=$PWD/data/phase12 HF_HUB_OFFLINE=1

./scripts/check.sh gate     # 2209 passed / 5 skipped / 30 deselected
./scripts/check.sh all      # adds tier 3

bash tools/run/sweep.sh     # resumable; all 19 steps present, so it is a no-op
python tools/run/curve.py   # ~2:43, writes data/analysis/curve.json AND
                            # data/analysis/formation_series.json

# step 1 of the order. Needs formation_series.json above; ~0.2s after that.
python3 -m tools.p_i1_attainable_floor --write
python3 -m tools.p_i1_attainable_floor --check     # no data needed
```

`curve.py` now writes the per-head series beside the centroids, in a SECOND
file. curve.json is what §4.6 diffs after every change to the storage layer, so
it is not the place to add a key — and it was re-run after this change and
diffed against the pre-change copy: **0 differences**, a third time.

THREE records now carry file hashes of `core/changepoint_colocation.py` or
`p7_motifs/formation_gate.py` and must be rewritten whenever either changes.
`changepoint_colocation.py` DID move this session (§5.2), so all three were
rewritten and the gate is green:

```bash
python3 -m tools.dry_run_claim_b_p_i1 --write      # ~4 min (measured, not ~1)
python3 -m tools.claim_b_grid_feasibility --write  # 224.5 s
python3 -m tools.p_i1_attainable_floor --write     # ~0.2 s, needs the series
```

`pythonpath = .` in `pytest.ini` applies to pytest only — a plain
`python script.py` needs `PYTHONPATH` set, which `tools/run/curve.py` does for
itself.
