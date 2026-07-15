"""
tests/test_phase2_smoke.py — Tier 1 smoke test for p2_eigenspectra.

Unlike phase 1's smoke test, this one isn't just "does a forward pass
work" — phase 2 is a consumer of phase 1's saved artifacts (_find_run_dir
globs for phase 1's run directory by reconstructing its stem). Running
this for real, on real files, is what actually checks the artifact
contract holds — the same class of bug already found in Phase 5
(merge_verdict, OV values, Group D all n/a because of producer/consumer
naming mismatches). A mocked phase-1 output would not catch this.

Run with:
    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_phase2_smoke.py -v

Depends on tiny_phase1_dir (tests/conftest.py), which itself depends on
phase 1 actually working — if this test fails, check test_phase1_smoke.py
passes first before debugging here.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.smoke

SMOKE_TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
SMOKE_PROMPT    = "short_heterogeneous"


def test_phase2_produced_output_dir(tiny_phase2_dir):
    assert list(tiny_phase2_dir.glob("p2_eigenspectra_*")), (
        f"no p2_eigenspectra_* directory under {tiny_phase2_dir} — "
        "run_full ran but wrote nothing, or wrote somewhere unexpected"
    )


def test_phase2_ov_weights_written_and_finite(tiny_phase2_dir):
    """
    If _find_run_dir failed to match phase 1's stem, run_full's per-prompt
    loop silently `continue`s (see run_2.py) and all_verdicts would already
    be empty -- caught by the assertion inside the tiny_phase2_dir fixture
    itself. Getting this far means the match succeeded; this checks the
    actual OV decomposition values aren't NaN/Inf.
    """
    ov_files = list(tiny_phase2_dir.rglob("ov_weights_*.npz"))
    assert ov_files, "no ov_weights_*.npz written anywhere under phase2 output"

    for f in ov_files:
        with np.load(f) as data:
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.number):
                    assert np.isfinite(arr).all(), f"{f.name}:{key} has NaN/Inf"


def test_phase2_verdict_json_written(tiny_phase2_dir):
    """Every prompt run should produce a verdict artifact — this is the
    per-prompt output run_one_prompt writes via reporting.save_verdict.
    Exact filename pattern inferred from reporting.py's save_verdict; if
    this fails on a name mismatch rather than a missing directory, that's
    worth checking against the real function before assuming breakage."""
    verdict_files = list(tiny_phase2_dir.rglob("*verdict*"))
    assert verdict_files, (
        f"no verdict-named artifact under {tiny_phase2_dir} — check "
        "reporting.save_verdict's actual output filename"
    )
