"""
tests/test_phase2i_smoke.py — Tier 1 smoke test for p2b_imaginary (run_2i).

Unlike phase 1/1b, this is an offline consumer of phase 1 + phase 2's
saved artifacts (run_2i.load_ov_data / run_2i.find_phase1_runs) — no live
model, no forward pass. Its own value is the same as test_phase2_smoke.py's:
running it for real, against real files, is what actually exercises the
artifact contract rather than assuming it holds.

Depends on tiny_phase1_dir and tiny_phase2_eigenspectra_dir (conftest.py).
model_stem is derived the same way phase 1 and phase 2 name their
directories/files (`model_name.replace("/", "_")`, hyphens preserved) —
NOT the way run_5.py's own default model_stem derivation does it (which
additionally replaces hyphens; see test_phase5_smoke.py's docstring for
where that actually breaks something). run_2i.py's own find_phase1_runs /
load_ov_data use the hyphen-preserving form throughout, so no override is
needed here the way it is for phase 5.

Run with:
    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_phase2i_smoke.py -v
"""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

SMOKE_TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
SMOKE_PROMPT    = "short_heterogeneous"
MODEL_STEM      = SMOKE_TINY_GPT2.replace("/", "_")


@pytest.fixture(scope="session")
def tiny_phase2i_result(tiny_phase1_dir, tiny_phase2_eigenspectra_dir,
                         tmp_path_factory):
    from p2b_imaginary.run_2i import run_model

    save_dir = tmp_path_factory.mktemp("phase2i_smoke")
    result = run_model(
        model_stem=MODEL_STEM,
        phase2_dir=tiny_phase2_eigenspectra_dir,
        phase1_dir=tiny_phase1_dir,
        save_dir=save_dir,
        force_block2=True,   # exercise Block 2 regardless of Block 1b's
                              # data-dependent rotation_contributes verdict
    )
    return result, save_dir


def test_phase2i_no_top_level_error(tiny_phase2i_result):
    result, _ = tiny_phase2i_result
    assert "error" not in result, (
        f"run_model returned a top-level error: {result.get('error')} — "
        "most likely load_ov_data didn't find ov_weights_<stem>.npz; check "
        "MODEL_STEM against what tiny_phase2_eigenspectra_dir actually "
        "wrote"
    )


def test_phase2i_block1a_ran(tiny_phase2i_result):
    result, save_dir = tiny_phase2i_result
    b1a = result.get("block1a", {})
    assert "error" not in b1a, f"Block 1a failed: {b1a.get('error')}"
    assert (save_dir / MODEL_STEM / "sub" / "block1a_rotational_spectrum.json").exists()


def test_phase2i_block1b_found_phase1_activations(tiny_phase2i_result):
    """
    If this fails with "no_phase1_activations", find_phase1_runs didn't
    match phase 1's directory naming for MODEL_STEM — the artifact-contract
    bug class this tier exists to catch, same as Phase 5's Group B/C1/D
    known bugs.
    """
    result, _ = tiny_phase2i_result
    b1b = result.get("block1b", {})
    assert b1b != {"error": "no_phase1_activations"}, (
        f"find_phase1_runs found nothing for stem '{MODEL_STEM}'"
    )


def test_phase2i_block2_ran_when_forced(tiny_phase2i_result):
    result, save_dir = tiny_phase2i_result
    assert result.get("block2_decision") == "run", (
        f"expected Block 2 to run under force_block2=True, got decision="
        f"{result.get('block2_decision')}"
    )
    assert result.get("block2"), "Block 2 ran but produced no per-prompt results"
    assert (save_dir / MODEL_STEM / "sub" / "block2_hemispheric.json").exists()


def test_phase2i_summary_txt_written(tiny_phase2i_result):
    _, save_dir = tiny_phase2i_result
    summary_path = save_dir / MODEL_STEM / "summary.txt"
    assert summary_path.exists()
    assert summary_path.read_text().strip()


def test_phase2i_json_values_finite(tiny_phase2i_result):
    """Spot-check every sub/*.json under the model's output for NaN/Inf,
    the same bar test_phase2_smoke.py holds ov_weights_*.npz to."""
    import json
    _, save_dir = tiny_phase2i_result
    sub_dir = save_dir / MODEL_STEM / "sub"
    json_files = list(sub_dir.glob("*.json"))
    assert json_files, f"no sub-result JSON files under {sub_dir}"

    def _check(obj, path):
        if isinstance(obj, float):
            assert np.isfinite(obj), f"{path}: non-finite value {obj}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _check(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check(v, f"{path}[{i}]")

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        _check(data, jf.name)
