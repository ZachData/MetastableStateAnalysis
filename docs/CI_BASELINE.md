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
./scripts/check.sh gate          # tier 0 + tier 1, ~11 s

pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements/heavy.txt
./scripts/check.sh all           # adds tier 3
```

Note that `download.pytorch.org` is reachable from GitHub Actions but not from
every sandbox; the plain PyPI `torch` wheel is the CUDA build at ~4.9 GB
unpacked and works but is wasteful.
