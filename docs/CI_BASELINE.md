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
was **1551 passed, 5 skipped, 251 deselected** on 2026-08-25 (1504 before
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

Five of them, as of P-ST1's steering-sign construction.

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

It is the only one of the five that hashes **two** files. Every verdict in it is
a joint property of `p1_mstate_tracking/replication_gate.py` and the committed
homogeneity curve, so `tests/test_claim_c_dry_run.py` pins both sha256s.
`tools/dry_run_claim_c.py --check` is the staleness mode, pinned in the pure
tier. The test module also **re-derives the headline boundary from scratch** in
about half a second rather than only comparing the stored number against itself
— three passes running, this project has found defects in generated artifacts
that no test was failing on, and the cheap half of a five-minute recomputation
is worth having in the gate.

### The fifth one: P-ST1's steering-sign calibration

`claims/calibration/steering_sign.json` is what P-ST1's construction does,
measured on synthetic populations with a planted answer (`POPPER_PLAN.md` §6k).
Generated by `python3 -m tools.calibrate_steering_sign --write` — about
twenty-five minutes, dominated by the matched-subspace null, which costs
`n_draws × n_pairs` effective ranks per gate run.

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

All five artifacts are tracked only because `.gitignore`'s whitelist already
un-ignores `claims/**/*.json`. None is in tier 0, and none can be: all five
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
