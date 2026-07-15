"""
tests/test_phase1_smoke.py — Tier 1 smoke test for p1_mstate_tracking.

Runs the real phase 1 pipeline (model load -> extraction -> analysis ->
save_run) on hf-internal-testing/tiny-random-gpt2 and one short prompt.
This is NOT a unit test against synthetic fixtures like the rest of
tests/test_phase1_*.py — it needs real torch/transformers and a real
(tiny) forward pass, which is why it's marked `smoke` and gated by
conftest.py's SMOKE_REAL_DEPS check rather than running by default.

Run with:
    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_phase1_smoke.py -v

The fixtures this depends on (tiny_phase1_dir, _register_smoke_models) live
in tests/conftest.py alongside every other shared fixture in this suite.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.smoke

# Duplicated from conftest.py rather than imported: tests/ has no __init__.py,
# so it isn't a package, and pytest's conftest.py isn't reliably importable
# as a plain module by name across pytest versions/rootdir configs. These
# are one-line constants — cheaper to keep in sync by eye than to fight the
# import machinery over.
SMOKE_TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
SMOKE_PROMPT    = "short_heterogeneous"


def _run_dir(phase1_root):
    stem = f"{SMOKE_TINY_GPT2.replace('/', '_')}_{SMOKE_PROMPT}"
    return phase1_root / stem


def test_phase1_run_directory_exists(tiny_phase1_dir):
    """The pipeline ran to completion and wrote something to disk."""
    run_dir = _run_dir(tiny_phase1_dir)
    assert run_dir.exists(), (
        f"expected {run_dir} — if this is missing check the stem format "
        "in run_1.py hasn't drifted from '{model.replace(/,_)}_{prompt}'"
    )


def test_phase1_layer_metrics_present(tiny_phase1_dir):
    """layer_metrics.json is the file phase 2's _find_run_dir gates on —
    if it's missing here, phase 2's smoke test will fail for a reason
    that has nothing to do with phase 2."""
    run_dir = _run_dir(tiny_phase1_dir)
    assert (run_dir / "layer_metrics.json").exists()


def test_phase1_npz_artifacts_are_finite(tiny_phase1_dir):
    """Shapes-and-finiteness check, not a correctness check (that's the
    oracle tier). A NaN/Inf here means something in the pipeline broke
    silently on real (if tiny and random) activations."""
    run_dir = _run_dir(tiny_phase1_dir)
    npz_files = list(run_dir.glob("*.npz"))
    assert npz_files, f"no .npz artifacts written to {run_dir}"

    checked_any_numeric = False
    for f in npz_files:
        with np.load(f) as data:
            for key in data.files:
                arr = data[key]
                if np.issubdtype(arr.dtype, np.number):
                    checked_any_numeric = True
                    assert np.isfinite(arr).all(), f"{f.name}:{key} has NaN/Inf"
    assert checked_any_numeric, "no numeric arrays found across any .npz artifact"