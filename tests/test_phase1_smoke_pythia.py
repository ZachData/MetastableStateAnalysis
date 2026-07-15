"""
tests/test_phase1_smoke_pythia.py — Tier 1 smoke test for the GPT-NeoX
(Pythia) branch of p1_mstate_tracking.

Same shape as test_phase1_smoke.py, against
hf-internal-testing/tiny-random-GPTNeoXForCausalLM instead of tiny-gpt2.
This is the test that actually exercises code paths the GPT-2 smoke test
can't reach: the extended _CAUSAL_MODEL_PREFIXES check in analysis.py, and
analyze_value_eigenspectrum's fused-query_key_value split in plots.py.

Run with:
    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_phase1_smoke_pythia.py -v

Needs network once to pull the tiny checkpoint (confirmed to exist on the
Hub). Not runnable in an offline sandbox.

Depends on the tiny_phase1_gptneox_dir fixture, which lives directly in
tests/conftest.py (section 4, "Smoke-test fixtures"), next to
tiny_phase1_dir/tiny_phase2_dir — not in a separate file. A prior pass
shipped that fixture as its own conftest_pythia_addition.py, which pytest
never collects (fixture discovery only reads a file literally named
conftest.py per directory); that's why the fixture showed as "not found"
even though the code existed on disk. Fixed by merging it into the real
conftest.py instead.

conftest.py's own header also documents a separate, real, pre-existing
bug this test will still hit until it's fixed in core/models.py: on a
CUDA-visible machine, extract_activations doesn't move its tokenized
`inputs` onto the model's device before calling it, so every model in
run_1.run_all's loop fails silently and `tiny_phase1_gptneox_dir`'s own
assert fires first. That fix is out of scope for this file — it isn't a
test problem.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.smoke

SMOKE_TINY_GPTNEOX = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
SMOKE_PROMPT       = "short_heterogeneous"


def _run_dir(phase1_root):
    stem = f"{SMOKE_TINY_GPTNEOX.replace('/', '_')}_{SMOKE_PROMPT}"
    return phase1_root / stem


def test_phase1_run_directory_exists(tiny_phase1_gptneox_dir):
    """The pipeline ran to completion and wrote something to disk."""
    run_dir = _run_dir(tiny_phase1_gptneox_dir)
    assert run_dir.exists(), (
        f"expected {run_dir} — check load_model resolved "
        f"{SMOKE_TINY_GPTNEOX} via GPTNeoXModel.from_pretrained without "
        "needing a revision (this checkpoint has no step-numbered branches, "
        "unlike the real pythia-* registry entries)"
    )


def test_phase1_layer_metrics_present(tiny_phase1_gptneox_dir):
    """layer_metrics.json is the file phase 2's _find_run_dir gates on —
    if it's missing here, phase 2's smoke test will fail for a reason that
    has nothing to do with phase 2."""
    run_dir = _run_dir(tiny_phase1_gptneox_dir)
    assert (run_dir / "layer_metrics.json").exists()


def test_phase1_npz_artifacts_are_finite(tiny_phase1_gptneox_dir):
    """Shapes-and-finiteness check, not a correctness check (that's the
    oracle tier). A NaN/Inf here means something in the pipeline broke
    silently on real (if tiny and random) activations."""
    run_dir = _run_dir(tiny_phase1_gptneox_dir)
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


def test_v_spectrum_extracted_via_qkv_split(tiny_phase1_gptneox_dir):
    """
    The one check specific to this branch: run_1.run_all calls
    analyze_value_eigenspectrum once per model, saving to OUTPUT_DIR root
    (not the per-prompt run_dir — see plots.py's
    `save_dir / f"v_eigenspectrum_{model_name}.json"`), which must now take
    the GPT-NeoX elif branch (fused query_key_value split) rather than
    falling through to the empty-dict return every unrecognized
    architecture gets. A non-empty JSON with per-layer entries is the
    signal that branch fired; a missing or empty file means the model
    name didn't match the "pythia"/"gpt-neox"/"gptneox" check (see the
    case-sensitivity note in PATCH.md — this exact checkpoint id is the
    reason that check needed `.lower()`) and the split code never ran.
    """
    import json

    spectrum_path = (
        tiny_phase1_gptneox_dir
        / f"v_eigenspectrum_{SMOKE_TINY_GPTNEOX.replace('/', '_')}.json"
    )
    assert spectrum_path.exists(), (
        "v_eigenspectrum json missing — analyze_value_eigenspectrum likely "
        "returned {} for this model, meaning the GPT-NeoX branch didn't match"
    )
    with open(spectrum_path) as f:
        data = json.load(f)
    assert data.get("layers"), (
        "v_eigenspectrum json has no per-layer entries — the GPT-NeoX "
        "branch matched but extract_v_gptneox likely raised or returned "
        "nothing for every layer"
    )
