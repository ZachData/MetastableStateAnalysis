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
| Branch | `claude/rescaler-cache-identity-test` — nothing merged, no PR open |
| Last updated | 2026-09-03 |
| Structural map | `INDEX.md` — which phase lives in which directory, and what is archived |
| Method and construction log | `POPPER_PLAN.md` §6a–§6t |
| Pre-registered predictions | `PREDICTIONS.md`, `claims/registry.json` |
| What can carry an e-value | `claims/EVALUABILITY.md` |

---

## 1. Start here

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf
export METS_RESULTS_DIR=$PWD/data/phase12
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

./scripts/check.sh gate     # 2228 passed / 5 skipped / 30 deselected, ~35 s
```

If the gate is green the tree is consistent. If it fails on a `sha256` mismatch,
a module carrying a record's hash was edited — see §6.3, it is a chore and not a
bug.

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

Active work is **Phase 7** — the mechinterp/particle bridge — and specifically
`P-I1`, induction-head formation as a two-stage `relay` motif tracked across the
checkpoint axis. `INDEX.md`'s phase table is still accurate for everything else.

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

## 3. The open front: `P-I1`'s relay-count null

**The relay-count null does not exist.** `formation_gate` requires the series to
be the excess above the N1/N2 offset-null envelope, and `core/qk_offset_null.py`
computes N1/N2 for the **QK antisymmetry statistic**, not for relay counts.
`formation_curve.assert_gate_ready` refuses the raw series, correctly.

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
`tests/test_p_i1_attainable_floor.py`. Fixing it properly means either reducing
the axis to the forming heads or giving the arm a per-unit skip with a count
reported, and **both change what `P-I1`'s unit is**, which `PREDICTIONS.md`'s
first Phase 7 adjudication constraint fixes at the head. Author's decision.

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

### 3.4 What is still the author's, and must not be started from the code

The null's shape. The obvious one is a degree-preserving rewiring within each
(context, layer, head) that keeps each head's edge count and attractive
fraction, randomises which particles the edges connect, and preserves the
induction structure. Two constraints on it are already measured: it must hold
`n_induction` fixed per prompt (step 2 — otherwise it tests whether the prompt
has induction pairs, which is known before the model runs), and it must describe
a series that peaks between 54000 and 143000 rather than a monotone rise.
Relays per induction pair spans 45 to 133 across the battery, and that residue is
where a formation signal would have to live.

### 3.5 Also not done: the behavioural arm

`behavioural_induction_score` reads phase-1 `attentions.npz` per prompt per
checkpoint and has never been run over the sweep. Every relay-side number above
is measured on the 19 tables; the B side in the floor record is synthetic, and is
sound only because `paired_colocation_arm` profiles the A side first — which the
record checks rather than assumes.

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

### 5.2 `results/` holds 132 GB and must keep it

`2026-08-12_05-01-35` (56.5 GB) and `p2_eigenspectra_2026-08-13_05-13-52`
(74.2 GB) each cover **27 steps on the PILOT schedule** — 11000, 13000, 15000,
17000, 19000, 100000, 120000 and so on. Those steps appear in nothing else on
disk, and `core/pythia_registry.py` keeps `PYTHIA_410M_PILOT_STEPS` loadable for
exactly this reason. `p1b_pilot`, `p2b_pilot`, `p2d_pilot` and `phase3` are
small; `phase3` is referenced from `archive/`.

---

## 6. Untouched, and named so it is not mistaken for done

* `core/precision_policy.py`'s **P2** (Pythia ships fp16; an fp16-epsilon
  perturbation splits a genuinely real eigenvalue pair into a complex one) and
  **item 13** (the forward pass runs under bf16 autocast).
* **`real_frac`/`imag_frac` are NaN in every row of every table** — deliberate
  and correctly recorded (`rotational_channel: "absent"` in the manifest), not a
  silent gap. Two open questions: does any registered prediction need the
  rotational channel, and if none does, are those columns schema no producer
  fills and no consumer reads? `p7_io.rotational_channel_from_blocks` is the seam.
* **The phase-7 manifest records no library versions.** §1's first trap is the
  argument for adding them; not done, because it changes the manifest schema and
  every record that hashes it.
* **The in-memory categorical option** (int8 codes for `model`/`prompt_key`/
  `pair_type`, 5.49 GB → 1.89 GB expanded). Compression fixed disk and does
  nothing for RAM.
* **Eleven predictions are adjudicable in principle and
  `claims/adjudications/` is empty.**

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

python3 -m tools.p_i1_attainable_floor --write     # ~0.2 s, needs the series
python3 -m tools.p_i1_attainable_floor --check     # needs no data
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

---

## 8. Where to read next

| Question | File |
|---|---|
| Which phase lives where, what is archived | `INDEX.md` |
| Why a construction is the way it is | `POPPER_PLAN.md` §6a–§6t |
| What is pre-registered, and its falsifier | `PREDICTIONS.md`, `claims/registry.json` |
| Which predictions can carry an e-value, and the order to build a null in | `claims/EVALUABILITY.md` |
| Phase 7's translation table and motif alphabet | `p7_motifs/design-7.md` |
| A phase's current state | `<phase>/status-N.md` |
| What changed and when | `git log` |
