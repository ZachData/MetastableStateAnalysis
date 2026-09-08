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
| Last updated | 2026-09-08 |
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

./scripts/check.sh gate     # 2264 passed / 5 skipped / 30 deselected, ~34 s
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

Active work is **Phase 7** — the mechinterp/particle bridge. `P-I1`,
induction-head formation as a two-stage `relay` motif tracked across the
checkpoint axis, has run end to end and scored **INSUFFICIENT** — not
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
   (`claims/EVALUABILITY.md`'s own unused advice). **Stages 0–2 have run**
   (§3.11, dated blocks): the circuit is `L5H2 → L7H8`; the OV copying effect
   is carried ~82 % by a single high-gain SVD direction (`r*_SVD ≪ r*_Schur`);
   the core is 100 % repulsive but not a spectral outlier, and the rank-1 mode
   shows no token-identity copy structure — cutting against both the standard
   and the particle accounts. **Pick up at: register the differential
   prediction** (reshaped by Stage 2 — see §3.11 "Next"), then **Stage 3**
   (matched-norm perturbation of the rank-1 mode). QK half still not done
   (RoPE). `claims/registry.json` unchanged.
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

**Nothing is committed.** All of the above is on the working tree;
`./scripts/check.sh gate` is green (2264 passed / 5 skipped, 34 s). New files:
`tools/run/dissipation.py`, `tools/run/dissipation_sublayer.py`,
`tools/run/ov_per_head.py`, `tools/run/induction_rank_sweep.py`,
`tools/run/induction_subspace_characterize.py`,
`docs/dissipation_checkpoint_axis_scoping.md`,
`docs/results_provenance_audit_2026-09-05.md`; `data/analysis/` holds the new
series JSONs (`dissipation_series`, `dissipation_sublayer_series`,
`ov_per_head_series`, `induction_rank_sweep`,
`induction_subspace_characterize`, `relay_null_series_k50` / `_k100`), four
`.png` panels (+ two `.csv`), and durable copies of the builder scripts.

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

**Not adjudicated.** `claims/adjudications/` is untouched — `tools/
score_p_i1.py` deliberately does not call `adjudicate_p_i1(..., adjudicate=
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
~0.01. **QK half still not done** (RoPE — `rotary_ndims = 16` of 64).

**Next.** Register the differential prediction before Stage 3 reads anything
(§3.11 opening) — the Stage 2 result reshapes it: the live question is no
longer "attractive vs repulsive" but "does the high-gain SVD direction that
carries copying behave as an individuating / repulsive channel under
perturbation, or as a copier the token-alignment test just missed". Then
Stage 3 (matched-norm variations of the rank-1 mode, joint behavioural +
logit + geometry readout). `POPPER_PLAN.md` §6x is still unwritten — §6w
refers forward to it; this §3.11 is currently the only home for the design.

**In flight (2026-09-08, exploratory, nothing registered).** Two generalisation
runs, launched to answer whether the Stage 2 picture is L7H8-specific before
the prediction is written:
- `induction_rank_sweep.py --step` across all 19 axis steps for L7H8 →
  `data/analysis/induction_rank_sweep_s<step>_L7H8.json`. *When* does
  `r*_SVD ≪ r*_Schur` form?
- `induction_subspace_characterize.py --step --layer --head` on the top
  behavioural heads (L7H8, L6H0, L2H10, L9H9, L7H0, L9H8, L7H3, L1H15) each at
  its peak step → `…_s<step>_L<layer>.json`. Is "high-gain SVD direction /
  100 % repulsive core / non-token-aligned" general to induction OV or
  idiosyncratic?
Both scripts now take `--step/--layer/--head`; the no-arg defaults reproduce
the L7H8 @ 4000 files above unchanged.

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
  silent gap. **Both open questions answered 2026-09-04, nothing changed in
  code.** No registered prediction needs the rotational channel: `P-I2` names
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
| The dissipation-identity run — what it is, Tiers A/B, v2 list | `docs/dissipation_checkpoint_axis_scoping.md`, §3.8 |
| Are the on-disk phase12/phase7 results stale? | `docs/results_provenance_audit_2026-09-05.md` |
| The pythia-70m dense-onset run, and how it could plug in | §3.9 |
| What changed and when | `git log` |
