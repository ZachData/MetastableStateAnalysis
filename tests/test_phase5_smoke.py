"""
tests/test_phase5_smoke.py — Tier 1 smoke test for p5_single_mstate_analysis
(run_5.py).

Offline consumer of phase 1 + phase 2's saved artifacts, same as phase 2i —
no live model, no forward pass. Groups run: A, B, C1, C2, G.

Groups deliberately excluded:
  D   needs phase3_ckpt/phase3_cache + phase4_dir — Phase 3 is frozen-for-
      deletion (FROZEN_p3_crosscoder.md) and Phase 4 is out of scope for
      this pass; Group D is also already a known-broken artifact-contract
      bug (status-5.md) with no fix attempted here.
  E,F need a live model (tuned-lens decode / causal ablation) — phase 5's
      own characterization is "no live model needed"; E additionally needs
      a *trained* tuned-lens checkpoint, which doesn't exist for the tiny
      smoke checkpoint regardless.

model-stem override — the reason this test passes --model-stem explicitly
instead of just --models:
  run_5.py's own default stem derivation (in main()) is
      model_name.replace("-", "_").replace("/", "_")
  which replaces BOTH hyphens and slashes. Phase 1 and Phase 2 name their
  directories/files with model_name.replace("/", "_") ONLY — hyphens
  preserved. For "hf-internal-testing/tiny-random-gpt2" these disagree:
      run_5 default : hf_internal_testing_tiny_random_gpt2
      phase1/phase2 : hf-internal-testing_tiny-random-gpt2
  Group C1's OV-weights loader already works around exactly this
  (_run_group_C1's stem_h reversal, discovered while root-causing the
  "OV always n/a" known bug) by trying both forms — but nothing suggests
  the same care was taken wherever the real p5_single_mstate_analysis/io.py
  resolves Phase 1 run directories (that file isn't available to check
  directly here — see FIX-B7's docstring in test_phase5_bugs.py for the
  same access gap). `--model-stem` is a supported CLI override for exactly
  this situation (single-model runs only); using it here sidesteps the
  mismatch rather than depending on it being handled correctly downstream.
  If this test ever needs to run withOUT the override to be a faithful
  reproduction of default CLI usage, that's the mismatch to check first.

Run with:
    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_phase5_smoke.py -v
"""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

SMOKE_TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
SMOKE_PROMPT    = "short_heterogeneous"
MODEL_STEM      = SMOKE_TINY_GPT2.replace("/", "_")   # hyphens preserved — see module docstring
SMOKE_GROUPS    = ["A", "B", "C1", "C2", "G"]


@pytest.fixture(scope="session")
def tiny_phase5_result(tiny_phase1_dir, tiny_phase2_eigenspectra_dir,
                        tmp_path_factory):
    from p5_single_mstate_analysis import run_5

    out_dir = tmp_path_factory.mktemp("phase5_smoke")
    unused_dir = tmp_path_factory.mktemp("phase5_smoke_unused")  # phase2i-dir, phase4-dir: unused by the groups run here

    argv = [
        "--models", SMOKE_TINY_GPT2,
        "--model-stem", MODEL_STEM,
        "--phase1-dir", str(tiny_phase1_dir),
        "--phase2-dir", str(tiny_phase2_eigenspectra_dir),
        "--phase2i-dir", str(unused_dir),
        "--phase4-dir", str(unused_dir),
        "--out", str(out_dir),
        "--groups", *SMOKE_GROUPS,
    ]
    rc = run_5.main(argv)
    return rc, out_dir


def _model_out_dir(out_dir: Path, model_stem: str) -> Path:
    candidates = sorted(out_dir.glob(f"{model_stem}_*"))
    assert candidates, (
        f"no {model_stem}_* directory under {out_dir} — "
        f"contents: {[p.name for p in out_dir.iterdir()]}"
    )
    return candidates[-1]


def test_phase5_main_returns_success(tiny_phase5_result):
    rc, out_dir = tiny_phase5_result
    assert rc == 0, (
        f"run_5.main returned {rc} (failure) — check stdout/stderr from "
        "the run for which group raised; _run_one_model prints "
        "'[<group>] failed: <exc>' per group rather than raising, so a "
        "non-zero rc here means the top-level per-model try/except caught "
        "something outside any single group (e.g. model selection itself)"
    )


def test_phase5_expected_group_outputs_written(tiny_phase5_result):
    """
    Exact filenames from each group's save_* function (cluster_profile.py,
    v_alignment.py, head_contributions.py, ffn_contributions.py,
    sibling_contrast.py) — not a glob, for the same reason phase 1b/2i
    smoke tests use exact names: a silently-wrong filename here is the
    artifact-contract bug class this tier exists to catch.
    """
    _, out_dir = tiny_phase5_result
    model_dir = _model_out_dir(out_dir, MODEL_STEM)
    expected = [
        "group_A_profile_primary.json",
        "group_B_v_alignment_primary.json",
        "group_C1_heads_primary.json",
        "group_C2_ffn_primary.json",
        "group_G_sibling_contrast.json",
        "cluster_report.txt",
        "per_layer_arrays.npz",
    ]
    present = {p.name for p in model_dir.iterdir()}
    missing = [f for f in expected if f not in present]
    assert not missing, f"missing expected outputs under {model_dir}: {missing}"


def test_phase5_report_is_nonempty(tiny_phase5_result):
    _, out_dir = tiny_phase5_result
    model_dir = _model_out_dir(out_dir, MODEL_STEM)
    text = (model_dir / "cluster_report.txt").read_text()
    assert text.strip()
    assert "merge_verdict" in text, (
        "cluster_report.txt should include the Group B merge_verdict "
        "line (report.py's _render_verdicts) regardless of its value — "
        "FIX-B7 makes this a real computed value rather than always n/a, "
        "but even pre-fix the line itself was always present"
    )


def test_phase5_per_layer_arrays_finite(tiny_phase5_result):
    _, out_dir = tiny_phase5_result
    model_dir = _model_out_dir(out_dir, MODEL_STEM)
    with np.load(model_dir / "per_layer_arrays.npz") as data:
        for key in data.files:
            arr = data[key]
            if np.issubdtype(arr.dtype, np.number):
                assert np.isfinite(arr).all(), f"per_layer_arrays.npz:{key} has NaN/Inf"


def test_phase5_group_jsons_have_no_python_exceptions_embedded(tiny_phase5_result):
    """
    Each group's runner wraps its own body in try/except and can write a
    result dict whose value is a caught exception message rather than
    raising — that keeps main() green (rc=0) while silently burying a
    per-group failure. This checks none of the primary result files
    contain that shape.
    """
    import json
    _, out_dir = tiny_phase5_result
    model_dir = _model_out_dir(out_dir, MODEL_STEM)
    for name in ("group_A_profile_primary.json", "group_B_v_alignment_primary.json",
                 "group_C1_heads_primary.json", "group_C2_ffn_primary.json",
                 "group_G_sibling_contrast.json"):
        with open(model_dir / name) as f:
            data = json.load(f)
        assert not (isinstance(data, dict) and set(data.keys()) == {"error"}), (
            f"{name} is a bare error dict: {data}"
        )
