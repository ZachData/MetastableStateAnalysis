"""
tests/test_phase6_smoke.py — Tier 1 smoke test for p6_subspace (run_6.py).

Offline consumer of phase 1 + phase 2's saved artifacts. load_model=False
(no live model) — this gates off the dissociation sub-experiment (needs a
live model per test_phase6_wiring.py::test_dissociation_gated_off_without_load_model)
but runs everything else in the registry: head_classify, qk_decompose,
induction_ov, eigenspace_degeneracy, centroid_velocity, local_contraction,
probe_subspace, write_subspace.

Unlike run_5.py, run_6.py's own OV-weights lookup (_find_p2_weights_path)
already tries the hyphen-preserving form first and falls back to the
underscore form — matching Phase 1/2's actual naming — so no --model-stem-
style override is needed here the way test_phase5_smoke.py needs one.
Phase 1 lookup goes through core.io.find_phase1_run_dir, which is the same
real module reviewed for FIX-B7 (not the shadowed p5io), so this one's
naming resolution could be checked directly rather than inferred.

Run with:
    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_phase6_smoke.py -v
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

SMOKE_TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
SMOKE_PROMPT    = "short_heterogeneous"
MODEL_STEM      = SMOKE_TINY_GPT2.replace("/", "_").replace("-", "_")   # run_one_model's own stem


@pytest.fixture(scope="session")
def tiny_phase6_out_dir(tiny_phase1_dir, tiny_phase2_eigenspectra_dir,
                         tmp_path_factory):
    from p6_subspace.run_6 import run_one_model

    out_root = tmp_path_factory.mktemp("phase6_smoke")
    run_one_model(
        model_name=SMOKE_TINY_GPT2,
        phase1_dir=tiny_phase1_dir,
        phase2_dir=tiny_phase2_eigenspectra_dir,
        out_dir=out_root,
        tracks="all",
        load_model=False,
        prompt_key=SMOKE_PROMPT,
    )
    # run_one_model nests its own output under out_dir / stem (see its body)
    return out_root / MODEL_STEM


def test_phase6_report_written(tiny_phase6_out_dir):
    report_path = tiny_phase6_out_dir / "phase6_report.txt"
    assert report_path.exists(), (
        f"no phase6_report.txt under {tiny_phase6_out_dir} — if "
        "_build_or_load_projectors returned None (no OV weights found), "
        "run_one_model returns silently before ever building ctx or "
        "writing anything; check ov_weights_{MODEL_STEM}.npz path "
        "resolution first"
    )
    assert report_path.read_text().strip()


def test_phase6_sub_results_written(tiny_phase6_out_dir):
    sub_dir = tiny_phase6_out_dir / "sub"
    assert sub_dir.is_dir(), f"no sub/ directory under {tiny_phase6_out_dir}"
    json_files = list(sub_dir.glob("*.json"))
    assert json_files, f"no sub-experiment JSON files under {sub_dir}"


def test_phase6_head_classify_ran(tiny_phase6_out_dir):
    """head_classify has no live-model dependency and no phase-1-activation
    dependency beyond OV weights, so it's the one sub-experiment that
    should reliably run regardless of anything else in the registry."""
    candidates = list((tiny_phase6_out_dir / "sub").glob("*head_classify*"))
    assert candidates, (
        f"no head_classify sub-result found — contents of sub/: "
        f"{[p.name for p in (tiny_phase6_out_dir / 'sub').iterdir()]}"
    )


def test_phase6_dissociation_gated_off(tiny_phase6_out_dir):
    """With load_model=False, dissociation should be skipped rather than
    attempted and failed — confirms the gate itself works, matching
    test_phase6_wiring.py's existing unit-level coverage of the same gate,
    but exercised here through the real CLI path end-to-end."""
    candidates = list((tiny_phase6_out_dir / "sub").glob("*dissociation*"))
    for c in candidates:
        with open(c) as f:
            data = json.load(f)
        assert data.get("applicable") is not True or "error" in data, (
            f"{c.name} looks like dissociation actually ran without a "
            "live model — the load_model gate may not be wired the way "
            "test_phase6_wiring.py expects"
        )


def test_phase6_json_values_finite(tiny_phase6_out_dir):
    sub_dir = tiny_phase6_out_dir / "sub"

    def _check(obj, path):
        if isinstance(obj, float):
            assert np.isfinite(obj), f"{path}: non-finite value {obj}"
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _check(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _check(v, f"{path}[{i}]")

    for jf in sub_dir.glob("*.json"):
        with open(jf) as f:
            data = json.load(f)
        _check(data, jf.name)
