# CI_BASELINE.md — what the suite did before CI existed, and what is still red

POPPER_PLAN.md item A0. Measured 2026-08-23, on the commit that introduced
workstream A. Recorded so that later greenness means something: a CI pipeline
stood up over an unmeasured suite cannot tell "we fixed it" from "it was always
like that".

## Before

The suite did not run at all. `pytest --collect-only` returned **zero tests**,
because `tests/conftest.py:20` imported `torch` at module scope and torch was
not installed. Collection failure in a conftest takes every module in the
directory down with it, so the number was 0 rather than "most of them".

With torch installed, collection still failed on ten modules, from two causes:

* `p4_mstate_features/__init__.py` imported `.analysis`; the module is
  `analysis_p4.py`. The package therefore could not be imported at all, so no
  Phase 4 test had ever run. `tests/test_phase3_analysis.py` had the same
  mismatch against `p3_crosscoder.analysis_p3`.
* `scikit-learn` and `matplotlib` were not declared anywhere, so nothing said
  they were needed.

## After

| tier | what it needs | tests | time | state |
|---|---|---|---|---|
| tier 0 — hygiene + registry | nothing | 5 lint rules, 30 registry entries | ~1 s | green |
| tier 1 — `pure` | numpy, scipy, pytest | 1532 passed, 8 skipped | ~10 s | **green** |
| tier 3 — `deps` | + torch, sklearn, matplotlib | 998 passed, 25 failed, 45 skipped | ~3 min | **25 red** |

Tier assignment was measured, not assigned: a module is `pure` only if its whole
test set passes with torch, transformers, scikit-learn and matplotlib all made
unimportable. 59 of 95 modules qualify.

**That table is a snapshot of the A0 pass, not a live count**, and it is left at
its measured values rather than edited forward — a baseline that gets updated
in place stops being a baseline. The registry held 30 entries then and holds 38
now; `./scripts/check.sh gate` reports the current tier-0 + tier-1 figure, which
was **2205 passed, 5 skipped, 30 deselected** on 2026-09-01, after edge and
particle tables were written compressed (2203 the same day after P-I1's own
sweep grid was registered; 2197 after the pairing null's
invariance guard was made structural; 2194 after the motif join was
confined to one (model, checkpoint, prompt) context; 2183 on
2026-08-31 after Phase 2's
float64 promotion; 2178 the same day after Phase 7's formation curve; 2157
after its driver; 2124 earlier, after the two gates the real tokenizer
found; 2119 when the registry gained CLAIM-B's registered sweep; 2115 on
2026-08-30 after the tier retrofit below, 1894 before it on the same day; 1839
before
P-I3's cross-head construction, 1788 before the
CLAIM-B grid feasibility record, 1651 on 2026-08-26, 1629 before the
P6-R2/R4 dry run, 1589 before
P-ST1's dry run, 1551 before CLAIM-C's cell-drop dimension, 1504 before
P-ST1's steering-sign construction, 1477 before the CLAIM-C dry run, 1397
before the CLAIM-B/P-I1 changepoint construction, 1351 before the Phase 6
revival, 1325 before the projector audit, 1298 at PR #12's merge).

## The 25 red tests

They are all in the `deps` tier, in three modules, and they were red before this
work started — the marker and import fixes did not cause them and do not hide
them. They are listed rather than quarantined, deliberately: skipping or
`xfail`-ing a test to get a green badge is exactly the move this project's own
rules forbid, and a red count that is written down and shrinking is more honest
than a green badge over a suppressed failure.

| module | n | first look |
|---|---|---|
| `tests/test_p2_producer_changes.py` | 5 | profile axis lengths, index alignment against the violation loop, and what the profiles forward. Added in the recent Phase 2 producer work (commit `0d2818d` re-added p3/p4); the tests encode a contract the producer does not currently meet. |
| `tests/test_phase5b_integration.py` | 20 | the whole `TestGroundTruthAligned` class plus the shuffled-control and global-mean-regression cases. Phase 5b has **never been run** (`status-5b.md`: "Not started (execution)"), so this is an integration suite over an unexecuted pipeline. |
| `tests/test_phase6_regression.py` | 5 | the induction-score sentinel (`None` vs the old `0.0` return) and the baseline-ARI sanity checks. |

**Why tier 1 is the gate and tier 3 is not, yet.** A gate that is red on arrival
gets ignored within a week. Tier 1 is green, fast, and covers 1532 tests, so it
can gate every push starting now. Tier 3 runs in the nightly smoke workflow and
reports; it becomes a gate when the count reaches zero.

That is a stopgap with a shape, not an excuse: **each of the three modules is a
chunk** (POPPER_PLAN.md workstream A follow-up), and the count above is the
acceptance test. Phase 5b's 20 are the most interesting of the three, because a
20-failure integration suite over a pipeline that has never been executed is
evidence about the suite as much as about the pipeline — the first question is
whether those tests encode the design or the author's expectation of it.

### The tier retrofit finished, and 221 tests that ran in nothing (2026-08-30)

`-m heavy` collects **zero tests**: the marker is registered at `pytest.ini:28`
but nothing anywhere carries it, in `tests/` or `archive/tests/`, and nothing
applies it dynamically. Consistent with its definition — "needs real run
artifacts on disk" — and with there being no real run artifacts yet.

Counting the other markers is what mattered. Against 2721 collected:

| marker | collected |
|---|---|
| `pure` | 1899 |
| `deps` | 571 |
| `smoke` | 36 |
| `heavy` | 0 |
| **unmarked** | **221** |

`check.sh gate` selects `-m pure` and `check.sh all` adds `-m deps`, so **221
tests, 8% of the suite, ran in no tier** — executed by nothing, locally or in
CI, with nothing failing to say so. Seven modules, five of them Phase 7's:
`test_p7_motif_alphabet`, `test_p7_motif_stats`, `test_p7_interaction_graph`,
`test_p7_events`, `test_p7_io`, plus `test_core_interactions` and
`test_core_dissipation`. The motif alphabet, the interaction graph, event
extraction and the I/O layer underneath the current P-I3 work were all outside
the gate meant to protect them.

The cause is exact and is worth keeping: all seven arrived in `e6d7dba`
(2026-08-22), "Archive phases 3-6; open Phase 7" — one day before `4fb460d`
introduced the tier taxonomy. They predate the tiering pass and were never
swept up, while every Phase 7 module written after `4fb460d`
(`cross_head_gate`, `patching_gate`, `formation_gate`) carries its marker.

All seven are now `pure`, measured under the project's own rule rather than
assigned: they pass with torch, transformers, scikit-learn and matplotlib all
made genuinely unimportable. The gate moves 1894 -> 2115 and deselected 251 ->
30.

`rule_test_tier_markers` in `tools/lint_repo.py` had been reporting exactly
these seven as **warnings** since the taxonomy landed, above a comment reading
"Promote to error once that lands." It is now an error, which is the half that
matters — a warning is what let 221 tests sit unrun for eight sessions while
the rule that knew about them printed every time. Checked against the family it
is meant to catch: with one module's `pytestmark` removed, `check.sh lint`
exits 1.

### Remeasured (2026-08-30): tier 3 is green, and the table above is not

The A0 table is left at its measured values, per the rule stated under it. This
section records what tier 3 does *now*, because "25 red" had become wrong in
three separate ways and a stale red count is read as a to-do list.

Measured on Fedora, Python 3.14.7, with numpy 2.5.2, scipy 1.18.1,
torch 2.13.0+cpu, transformers 5.16.1, scikit-learn 1.9.0, matplotlib 3.11.1,
pytest 9.1.1:

    ./scripts/check.sh all
    tier 3 (-m deps): 569 passed, 2 skipped, 2150 deselected, 0 failed, 145 s

**Zero red.** Taking the three modules in turn:

* `tests/test_p2_producer_changes.py` — **18 tests, all passing**, and it is in
  the deps tier, so those 18 are inside the 569. Its recorded 5 failures do not
  reproduce, and not because anything was fixed: the test file and both modules
  it imports (`p2_eigenspectra/weights.py`, `p2_eigenspectra/head_ov_analysis.py`)
  are byte-identical to `4fb460d`, the commit the A0 figure was measured on —
  the only change to `p2_eigenspectra/` since is an added `math-2.md`. Checked
  the hard way rather than inferred: `4fb460d` checked out into a worktree and
  run against the dependency set above gives 18 passed there too. So the 5 is a
  property of the 2026-08-23 *environment*, not of the code at that commit.
* `tests/test_phase5b_integration.py` — now `archive/tests/`, excluded by
  `pytest.ini`'s `norecursedirs`. Its 20 is not merely uncollected but
  unmeasurable in place: pointed at directly it fails collection with
  `ModuleNotFoundError: No module named 'tests.test_phase5b_io'`, the helper
  having stayed behind when the module was archived.
* `tests/test_phase6_regression.py` — now `archive/tests/`, likewise excluded.
  Run directly it gives **14 failed, 4 passed**, not the 5 recorded.

Two further things the remeasurement turned up, both worth more than the count:

**The A0 section does not add up.** The header says 25 red and the table below
it lists 5 + 20 + 5 = 30. Which of the two was measured cannot be recovered now,
so this is recorded rather than corrected.

**No dependency versions were written down, which is what made the p2 case
ambiguous.** `requirements/base.txt` floats everything (`numpy>=1.24`,
`scipy>=1.10`), so "we fixed it" and "the environment moved" were
indistinguishable from the record alone — the exact confusion this file exists
to prevent, one level up from the test counts it prevents it for. It took a
worktree checkout to settle a question a pinned version list would have
answered. The live example arrived the same day: scipy 1.18 stopped returning an
owndata array from `expm` and turned a tautological assertion in
`tests/test_phase2b_rescaled.py` red, in the *gating* tier, with no commit
involved. Hence the version block above, and hence recording it here as a
standing gap rather than as a one-off.

## Reproducing this

```bash
pip install -r requirements/test.txt
./scripts/check.sh gate          # tier 0 + tier 1 (isolated), ~11 s

pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements/heavy.txt
./scripts/check.sh all           # adds tier 3
```

Note that `download.pytorch.org` is reachable from GitHub Actions but not from
every sandbox; the plain PyPI `torch` wheel is the CUDA build at ~4.9 GB
unpacked and works but is wasteful.

## Why `gate` isolates the pure tier

`./scripts/check.sh gate` runs tier 1 with torch, transformers, scikit-learn and
matplotlib shadowed by packages that raise `ImportError`. That is not paranoia;
it is the difference between the two environments, and it cost two red CI runs.

A developer machine has torch installed. `tests/conftest.py` installs its
MagicMock stub with `sys.modules.setdefault`, so on that machine the stub never
takes effect and `pytest -m pure` exercises the real library throughout. The
pure tier's central claim -- *this passes with no heavy dependencies* -- is then
not tested at all by the command that appears to test it.

### The same gap, one tier up (2026-08-24)

Tier 0's contract is stronger: *no dependencies at all*, which is why CI's lint
job installs nothing and why `tools/lint_repo.py`, `check_registry.py`,
`render_evaluability.py` and `render_falsification.py` each say "standard
library only" in their docstrings.

`python -m core.adjudication --verify` joined tier 0 when the ledger became
self-verifying, and it brought `core/evalues.py` with it. That module's scope
line said "pure numpy + stdlib" and it imported numpy at module scope, so the
lint job went red with `ModuleNotFoundError: No module named 'numpy'` on a
runner that correctly had nothing installed — and passed on every developer
machine, all of which have numpy. Identical in shape to the tier-1 failure
below, one tier up, and undetectable by the command that appeared to test it.

Two fixes, because either alone leaves the hole open. The calibrator and the
e-process are pure `math`; numpy appears in one simulation helper
(`simulate_type_i_error`) whose callers live in the test tier, so its import is
now function-local. And `check.sh lint` now runs tier 0 with numpy, scipy and
the heavy four all shadowed, so the contract is exercised rather than asserted
— `TestTierZeroIsStdlibOnly` in `tests/test_core_evalues.py` is the fast
in-suite guard for the same rule.

The lesson is the one this section already carried, and it is now three for
three: **a tier that claims to run without something has to be run without it.**

### The tier-1 failure that surfaced this pattern

The failure that surfaced it was a good one to have: scipy's array-API dispatch
asks `issubclass(cls, torch.Tensor)` on any path through `scipy.stats`, and a
MagicMock attribute is an instance rather than a class, so it raised

    TypeError: issubclass() arg 2 must be a class, a tuple of classes, or a union

at collection time, in a traceback naming neither torch nor the stub. Invisible
with real torch present. `tests/test_core_evalues.py::TestTorchStubIsScipySafe`
now asserts the stub's `Tensor` is a real class, and `check.sh gate` runs the
tier the way the runner does.

## Tier 1 now depends on committed data artifacts (2026-08-24)

Thirteen of them, as of P-I3's cross-head record (2026-08-30).

`claims/calibration/claim_c_homogeneity.json` is CLAIM-C's homogeneity
calibration curve (`POPPER_PLAN.md` §6g, §6l). It is **generated offline** by
`tools/calibrate_claim_c_homogeneity.py --write` and committed, so the gating
tier reads it rather than recomputing it.

**Its cost changed by a factor of fifty on 2026-08-25 and that is worth knowing
before someone runs it casually.** It was about 50 seconds. Adding the
cell-drop dimension (§6l) crossed the existing bias-shape grid with a drop grid
of one complete-table configuration plus three mechanisms at seven rates, so the
configuration count went from 35 per prompt count to 770 — **about 40 minutes**
for the seven tabulated prompt counts, and the file went from roughly 0.4 MiB to
roughly 3 MiB. It is still the same kind of artifact: generated once, looked at,
committed. It is just no longer something to regenerate while waiting.

**The draw count is per prompt count, not one number**, which surprises people
reading `--write`. Every rate in the curve is conditional on the gate emitting,
so a bin needs a fixed count of *emitted* draws; the informative-row refusal
(§6l) turns away 61% of independent-row H0 draws at six prompts and 1% at
twelve. `trials_for` scales the base count by `1 / P` with *P* in closed form,
capped — 3× at six prompts, 2× at seven, unchanged above. Each curve carries a
`coverage` block naming any drop slab with no measurement, because a slab like
that is one the gate refuses outright, and the first generation of this file had
one at six prompts that no test was failing on.

Two consequences worth knowing before the file surprises someone:

- **If the curve is missing or stale, the gate fails loudly rather than
  quietly.** `p_value_claim_c` refuses to emit a p-value at all without a
  correction, so a lost or ignored file turns into refusals across
  `tests/test_claim_c_homogeneity.py`, not into subtly wrong numbers. That is
  deliberate: the corrected p is what enters H-TRANSFER's e-value, so falling
  back to the uncorrected one would put a p already measured to be
  anticonservative into the ledger.
- **`.gitignore` is a whitelist, and the curve is only tracked because
  `!claims/**/*.json` already un-ignores it.** A curve written anywhere else in
  the tree would be silently untracked, CI would run without it, and the
  failure would look like a code defect. This is the same trap §2 of this file
  records for `pyproject.toml` and the workflows; `git check-ignore -v <path>`
  is still the thing to run.

### The second one: the Phase 6 projector audit

`claims/audits/p6_projector_labels.json` is the record that settles
`status-6.md` item 5 (`POPPER_PLAN.md` §6h). Same pattern, one extra wrinkle
worth stating plainly.

It is generated by `tools/audit_p6_projector_labels.py --write` — about 100
seconds, dominated by sixteen Schur decompositions at `d = 2048`. The wrinkle is
**why** it cannot be a test: producing it requires importing
`archive/p6_subspace/subspace_build.py`, and `archive/README.md` rule 1 is that
nothing under `archive/` is imported by anything live. The tool loads it by file
path, only when run, and only pytest-collected code counts as live — so the rule
holds intact and no exception was carved into it.

`tests/test_p6_projector_audit.py` pins the committed result **and the sha256 of
the audited file**. Without the hash the audit could go on describing a file
that no longer exists in that form, and nothing in the suite would notice,
because nothing live imports it. Regenerating is the fix; editing the hash is
not.

### The third one: the changepoint co-location calibration

`claims/calibration/changepoint_colocation.json` holds the measured H0
rejection rates for `core/changepoint_colocation.py`, the construction shared by
`CLAIM-B` and `P-I1` (`POPPER_PLAN.md` §6i). Generated by
`python3 -m tools.calibrate_changepoint_colocation --write` — about 100 seconds,
dominated by 4500 runs of a 2001-permutation pairing null.

It differs from the other two in what it is *for*. The CLAIM-C curve is **read
at runtime** — the gate applies it to correct the p it reports — so a stale
curve corrupts a number. This one is read by nobody: the pairing null is exact
by exchangeability and needs no correction. What the artifact carries is the
**evidence that the null is the one that works**, including the two rows that
decided the design (nominal under a common early trend, where every
permutation-over-checkpoint-order null inflates) and the row that records its
severe limitation (rejection 1.00 under a shared per-unit factor). Losing it
would not break the gate; it would lose the reason the gate is built the way it
is, which is the thing this project keeps finding it cannot reconstruct later.

`tests/test_changepoint_colocation.py::TestCommittedCalibration` pins it. There
is no `--check` staleness mode and no file hash, because unlike the projector
audit it describes no other file — it describes the module the pure tier already
exercises directly, and the assertions are on the rates themselves.

### The fourth one: CLAIM-C's dry run

`claims/audits/claim_c_dry_run.json` is what CLAIM-C's gate did when it was run
on inputs whose correct verdict is known a priori — a self-comparison and a
power curve (`POPPER_PLAN.md` §6j). Generated by
`python3 -m tools.dry_run_claim_c --write` — about five minutes, dominated by
roughly nineteen thousand end-to-end gate calls at ~15 ms each.

Like the changepoint calibration it is **read by nobody at runtime**: it
corrects no number and the gate behaves identically without it. What it carries
is the gate's measured operating range — the admissible homogeneity band, the
concordance the gate needs, the fact that the derived refusal is tight, and
since 2026-08-25 the measured cost of the informative-row floor — and that is
the thing this project keeps finding it cannot reconstruct later.

Its schema went to 2 on 2026-08-25 for that last section. One field in it is
worth knowing about: `informative_row_floor.costs_no_power` is `None`, never
`True`, when the refusal did not fire anywhere in the sweep, and
`--check` fails on `None` as loudly as on `False`. A sweep with nothing to
re-score would otherwise report success while being incapable of reporting
anything else, which is the audit arm `POPPER_PLAN.md` §6h found reporting PASS
without being able to fail.

It is one of two that hash **two** files — P-ST1's dry run is the other. Every
verdict in it is a joint property of `p1_mstate_tracking/replication_gate.py`
and the committed homogeneity curve, so `tests/test_claim_c_dry_run.py` pins
both sha256s.
`tools/dry_run_claim_c.py --check` is the staleness mode, pinned in the pure
tier. The test module also **re-derives the headline boundary from scratch** in
about half a second rather than only comparing the stored number against itself
— three passes running, this project has found defects in generated artifacts
that no test was failing on, and the cheap half of a five-minute recomputation
is worth having in the gate.

### The fifth one: P-ST1's steering-sign calibration

`claims/calibration/steering_sign.json` is what P-ST1's construction does,
measured on synthetic populations with a planted answer (`POPPER_PLAN.md` §6k,
§6m). Generated by `python3 -m tools.calibrate_steering_sign --write` — **about
forty minutes**, dominated by the subspace nulls, which cost `n_draws ×
n_pairs` effective ranks per gate run.

**Its cost and its shape both changed on 2026-08-26 (§6m), and the reason is
worth knowing.** It was about twenty-five minutes. Every gate run now computes
*two* subspace nulls — the adjudicated re-split of the observed union and the
retired matched-dimension pair, on the same draws so the comparison is paired —
two H0 families were added in which *both* arms are occupied above chance, and
a dedicated `reciprocal_tail` section measures the INVERTS branch at four times
the replicates. Against that, `CALIB_SUBSPACE_DRAWS` dropped from 99 to 49
(floor 0.02, still far under α = 0.05), which is where most of the doubling was
paid for.

**The family list is part of the measurement.** Until §6m every H0 family here
placed the token cloud in a subspace orthogonal to both arms, which leaves both
at chance occupancy — the one circumstance in which the matched-dimension null
was valid. Its failure was therefore invisible for a pass, at up to 0.20 against a nominal 0.05 on the family that was missing. `check_record()` now fails if no
`H0-both-arms` family is present, and fails again if the retired null does
*not* come back anticonservative — an artifact that no longer supports the
retirement is a problem with the retirement, not something to pass over.

It is read by nobody at runtime, like the changepoint calibration: the gate
behaves identically without it. What it carries is the evidence behind four
module constants and one precondition, including the two measurements that
reversed a choice made earlier the same day — that `normed` effective rank
answers differently for `v` and `−v`, and that the injection scale has a
plateau a coarse grid could not see.

It hashes the module it describes, as the projector audit and the CLAIM-C dry
run do, because its rates are rates of one specific construction.
`tests/test_p_st1_steering_gate.py::TestCommittedCalibration` pins it and calls
`check_record()`, which also asserts that each measured section still supports
the constant it decided — so flipping `ER_MODE` back to `normed` fails the gate
rather than leaving the record quietly describing an argument for something
else.

Its assertions are on the DIRECTION each section establishes rather than its
digits. The digits are proportions over a few hundred draws, and pinning
sampling noise to three places would make the gate fail for the wrong reason.

### The ninth and tenth: the last three entries' dry runs

`claims/audits/p_t1_p_m1_dry_run.json` and `claims/audits/p_s1_dry_run.json`
finish `claims/EVALUABILITY.md`'s queue — all nine adjudicable rows have now
been run on inputs whose answer is known (`POPPER_PLAN.md` §6p). Generated by
`python3 -m tools.dry_run_p_t1_p_m1 --write` and
`python3 -m tools.dry_run_p_s1 --write`; both store their own
`elapsed_seconds`, and the P-T1/P-M1 one is the cheapest artifact in the set by
a wide margin.

**Two entries in one record, and the reason is not the one CLAIM-B and P-I1
had.** Those two share a *construction*. P-T1 and P-M1 do not — they are here
together because they had the same defect independently, and because they are
the only pair whose shared instrument feeds a *single* claim, so their e-values
must not be read as two independent factors.

Each record hashes **two** files. `tests/test_p_t1_p_m1_dry_run.py` and
`tests/test_p_s1_dry_run.py` pin both, and each re-derives its closed form from
scratch in milliseconds — the hypergeometric floor and `∏(mult)!/n!` for the
first, and for the second the check that the module's fallback note no longer
claims a rate that stopped describing it. A pinned number only ever compared
against itself is a number nobody has checked.

`p_s1_dry_run.json` also carries a **second implementation** of the gate's
arithmetic, because the module now refuses every input the finding is about and
the arm has to reach that state. It is pinned against `p_value_p_s1` on every
matched pair, where the module still emits — the same guard
`claim_b_p_i1_dry_run.json` uses, and for the same reason `POPPER_PLAN.md` §6g
gives on CLAIM-C's fast path.

### The eighth one: CLAIM-B and P-I1's dry run

`claims/audits/claim_b_p_i1_dry_run.json` is what those two entries did on
inputs whose answer is known, and the measurement that changed CLAIM-B's anchor
arms (`POPPER_PLAN.md` §6o). Generated by
`python3 -m tools.dry_run_claim_b_p_i1 --write`, and it is the cheapest of the
four dry runs by a wide margin — the only one still worth regenerating while
waiting.

**Its generation cost is in the record rather than in this paragraph**, as
`elapsed_seconds`, measured on every write. That is the fix for what happened
one pass earlier: `tools/dry_run_p6_r2_r4.py` stated twenty minutes, a section
was added that took it to thirty-five, and the figure had to be chased across a
module docstring, a `--help` string and this file. A cost the tool measures
cannot drift from the tool.

One dry run covers **two** registry entries, because they share one estimator —
which is also why it hashes two files, the shared construction and P-I1's thin
half over it, and why `tests/test_claim_b_p_i1_dry_run.py` pins both.

Two things about its `check_record()` are worth copying. It fails if the
**finding** stops being in the record: CLAIM-B's anchor arms gained a refusal on
the evidence that the arm's rejection rate on a change-free input equals its
rate on a perfectly anchored one, and an artifact that no longer shows that does
not support the change it is the evidence for. And it fails if the
change-free reference's rank stops being **flat** across the control-family
axis — that flatness is the reason the refusal is built on the arm's ceiling
rather than on that rank, so if it ever moves the condition should be
reconsidered rather than left standing.

The file also carries a second implementation of the anchor arm's ranking,
because the dry run has to reach rates the module now refuses. `module_agreement`
pins it against `anchor_arm` itself on every input where the module emits —
`POPPER_PLAN.md` §6g records a second implementation of a gate's arithmetic as a
real risk on CLAIM-C's fast path, and this is the same check.

### The seventh one: P6-R2 and P6-R4's dry run

`claims/audits/p6_r2_r4_dry_run.json` is what those two entries did on inputs
whose answer is known, and the measurement that changed P6-R2's null
(`POPPER_PLAN.md` §6n). Generated by `python3 -m tools.dry_run_p6_r2_r4 --write`
— about thirty-five minutes, most of it the `precision_check` section, which
re-runs the two ends of the sweep at 600 replicates because 250 cannot separate
0.05 from 0.07.

It hashes **two** files, the null construction and the geometry module that
builds both null families, and `tests/test_p6_r2_r4_dry_run.py` pins both plus
re-derives the mechanism in milliseconds. Its `check_record()` fails in *both*
directions, which is the part worth copying: if the adjudicated null stops
holding, and also if the RETIRED one stops looking anticonservative — an
artifact that no longer supports a retirement is a problem with the retirement
rather than something to pass over.

### The sixth one: P-ST1's dry run

`claims/audits/p_st1_dry_run.json` is what P-ST1's gate did when it was run on
inputs whose correct verdict is known a priori (`POPPER_PLAN.md` §6m), the
treatment `claims/EVALUABILITY.md` says every converted row is owed. Generated
by `python3 -m tools.dry_run_p_st1 --write` — about ten minutes.

It hashes **two** files: the gate, which decides the verdicts, and the
steering-sign calibration, which is where the gate's constants came from.
`tests/test_p_st1_dry_run.py` pins both sha256s and re-derives two of the
headlines from scratch in milliseconds — the planted verdict at one cell, and a
handful of exchangeable draws — because a pinned number only ever compared
against itself is a number nobody has checked. This is the sixth pass running
in which looking at a generated output found something no test was failing on:
here, that the gate's reported floor was not the attainable one.

### The eleventh: P-AB1's growth-exponent calibration

`claims/calibration/patching_exponent.json` holds the measured H0 rates,
refusal behaviour and design arithmetic for `p7_motifs/patching_gate.py`, the
construction `P-AB1` gained on 2026-08-27 (`POPPER_PLAN.md` §6q). Generated by
`python3 -m tools.calibrate_patching_exponent --write`; it stores its own
`elapsed_seconds`.

Like the other two calibrations it is **read by nobody at runtime**: it corrects
no number and the gate behaves identically without it. What it carries is the
evidence for four decisions the gate could not otherwise defend — the sign sum
over a mean of paired differences, the per-arm power-law refusal over the paired
bend contrast that was tried first, the unit question left open rather than
guessed, and the fixed-offset limitation that no unit removes.

**Four of its sections take no replicates at all**, which is unusual for this
set and is the point of building the row in `EVALUABILITY.md`'s prescribed
order: `window_dependence` fits one fixed set of dynamics over several windows,
`sampling_spread` is arithmetic on the window, `grid_arithmetic` is closed-form
over the sign-flip group, and the registered null's invariance is one line of
algebra in the module. Those four are where the design was decided; the sampled
sections confirm it.

`tests/test_p7_patching_gate.py::TestCommittedCalibration` pins it. There is no
`--check` staleness mode and no file hash, for the reason
`changepoint_colocation.json` has neither: it describes the module the pure tier
already exercises directly, its assertions are on the rates themselves, and its
design block is asserted against the module's own constants so a change to
either fails rather than drifts.

### The twelfth: CLAIM-B's grid feasibility record

`claims/calibration/claim_b_grid_feasibility.json` is which checkpoint sweeps
can carry `CLAIM-B`'s anchor arms, enumerated over Pythia's published schedule
before any data exists (`POPPER_PLAN.md` §6r). Generated by
`python3 -m tools.claim_b_grid_feasibility --write`; it stores its own
`elapsed_seconds`, and roughly a fifth of that is the enumeration — 96,127 grids
— with the rest the anchor-arm measurement on the grids it picks.

**It is the first artifact here whose main content is a search rather than a
rate**, and that changes what the staleness check has to do. Two of
`check_record`'s conditions are about the search rather than about a number: the
frontier must not rest on an **artificial** bound of the enumerated family
(Pythia's own release granularity and last checkpoint are real bounds and a grid
resting on those rests on the data), and the record must still say that **no**
grid reaches the reference retention — if one ever does, that is a better answer
than this record gives and the section should be rewritten rather than the check
relaxed.

It hashes `core/changepoint_colocation.py`, as `claim_b_p_i1_dry_run.json` does,
because every number in it is a property of that module's arithmetic. It is read
by nobody at runtime: `anchor_arm` carries `grid_feasibility`'s two grid-only
conditions on every record it emits, and behaves identically without the file.
`tests/test_claim_b_grid_feasibility.py` pins it and **re-derives most of it in
milliseconds** — the change-free spread against a live simulation, the
rectified-normal covariance constant from its own integral, and the fact the
whole design turns on: that the sharp-change reading of a location is not a
bound.

Two things about it are worth copying. Its `closed_forms` section predicts and
measures the same quantity on every row, which is the check that would have
caught the defect §6r records — a reading that scored a grid at retention 1.000
where the measurement returned 0.017, with both numbers in the same smoke run
and nothing comparing them. And `check_record` fails if the sharp reading ever
stops **overstating** retention, because that is the argument for requiring the
change width as an input rather than defaulting it.

`claims/calibration/cross_head_association.json` is `P-I3`'s cross-head
calibration (`POPPER_PLAN.md` §6s), generated by
`tools/calibrate_cross_head_association.py --write` and pinned by
`tests/test_p7_cross_head_gate.py::TestCommittedCalibration`. It stores its own
`elapsed_seconds` and hashes `p7_motifs/cross_head_gate.py`, as the two records
before it hash the module whose arithmetic they are about.

Two things about it are worth copying. Its `check_record` refuses to let the
FINDINGS leave the file, not merely the fields: it fails if the registered
null starts discriminating, if a threshold classification stops flooring the
design at 1.000, if the discarded nearest-by-score matching stops leaking, or
if the registered matching key stops costing power — because a record that no
longer shows the trade is not evidence for a decision presented as one. And its
slope-agreement check is written against **three standard errors of the arm's
own spread** rather than a fixed tolerance: the first version used a constant,
and a constant tolerance on a statistic whose enormous spread that same section
exists to report fires on a healthy artifact at a low replicate count. That is
§6n's `precision_check` failure mode a third time, and the fix is the general
one — a check on a noisy quantity has to scale with the run that produced it.

All thirteen artifacts are tracked only because `.gitignore`'s whitelist already
un-ignores `claims/**/*.json`. None is in tier 0, and none can be: all thirteen
tools import numpy at module scope and tier 0 runs with numpy shadowed.

### The staleness checks

The staleness check for the curve is
`tools/calibrate_claim_c_homogeneity.py --check`, pinned
in the pure tier by `TestCurveIsInStepWithTheGate`. It compares the stored
metric set, tails, subset names, bin edges, levels, drop-bin edges, drop
mechanisms and drop rates against the gate's current constants — so adding a
metric to `CLAIM_C_METRICS` fails the gate instead of leaving the curve
describing a test that no longer exists.

It also compares the stored **alpha** against the registry's, which is newer
than the rest and less obvious. Since §6l the curve depends on alpha twice
over: the gate's derived refusal is `R(h, floor) > alpha`, and the
informative-row refusal reproduced inside the simulation is `floor > alpha`,
which decides *which draws emit* and therefore what every stored rate is
conditional on. A curve measured at one alpha and read at another is a
measurement of a different test, and nothing else in the file would show it. It is not
in tier 0, and cannot be: the tool imports numpy at module scope and tier 0 runs
with numpy shadowed.
