"""
tests/test_phase1b_smoke.py — Tier 1 smoke test for p1b_hemisphere.

Same shape as test_phase1_smoke.py: a real forward pass on the tiny GPT-2
checkpoint, real Blocks 0-4 (bipartition, hemisphere tracking, membership,
cone-collapse LP, asymmetry), real files on disk. No cross-phase artifact
contract to break here (phase1_dir is optional and left unset — see
tiny_phase1b_dir's docstring in conftest.py) — this checks phase 1h stands
on its own, the way phase 1 does.

Run with:
    SMOKE_REAL_DEPS=1 pytest -m smoke tests/test_phase1b_smoke.py -v

Depends on the tiny_phase1b_dir fixture (tests/conftest.py). If this test
fails, check test_phase1_smoke.py passes first — both fixtures share the
same core.models.extract_activations device-mismatch risk documented in
conftest.py's "Smoke-test fixtures" header.
"""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

SMOKE_TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
SMOKE_PROMPT    = "short_heterogeneous"
STEM            = f"{SMOKE_TINY_GPT2.replace('/', '_')}_{SMOKE_PROMPT}"


def test_phase1b_wrote_experiment_manifest(tiny_phase1b_dir):
    assert (tiny_phase1b_dir / "experiment.txt").exists(), (
        f"no experiment.txt under {tiny_phase1b_dir} — _write_manifest "
        "should run unconditionally at the top of run_all, before any "
        "per-model work; if this is missing, run_all likely didn't even "
        "start"
    )


def test_phase1b_per_run_json_and_md_written(tiny_phase1b_dir):
    """
    _save_run writes phase1h_{stem}.json and .md per (model, prompt) —
    stem = f"{model_name.replace('/','_')}_{prompt_key}" (run_1b.py
    _run_standard). Exact filenames, not a glob, since a silent stem
    mismatch here would be exactly the artifact-contract bug class this
    tier exists to catch.
    """
    json_path = tiny_phase1b_dir / f"phase1h_{STEM}.json"
    md_path   = tiny_phase1b_dir / f"phase1h_{STEM}.md"
    assert json_path.exists(), (
        f"{json_path.name} not found under {tiny_phase1b_dir} — "
        f"contents: {[p.name for p in tiny_phase1b_dir.iterdir()]}"
    )
    assert md_path.exists()
    assert md_path.read_text().strip(), "phase1h_*.md written but empty"


def test_phase1b_cross_run_digest_written(tiny_phase1b_dir):
    """_write_cross_run runs whenever all_results is non-empty (run_all's
    own guard) — with one model and one prompt that's still "cross-run"
    in name only, but the digest should still exist."""
    assert (tiny_phase1b_dir / "phase1h_cross_run.json").exists()
    assert (tiny_phase1b_dir / "phase1h_cross_run.md").exists()


def test_phase1b_per_layer_fields_finite(tiny_phase1b_dir):
    """
    Spot-checks the flat per-layer schema (_assemble_per_layer) actually
    has the shape Blocks 0/1/3/4 are supposed to produce, and that nothing
    numeric came back NaN/Inf where a real value was expected.
    """
    json_path = tiny_phase1b_dir / f"phase1h_{STEM}.json"
    with open(json_path) as f:
        data = json.load(f)

    assert data["n_layers"] > 0
    assert data["n_tokens"] > 0
    assert len(data["per_layer"]) == data["n_layers"], (
        "per_layer length should equal n_layers regardless of run_cone "
        "(Block 3 entries are optional per-layer, not optional-layer)"
    )

    numeric_keys_seen = 0
    for layer_entry in data["per_layer"]:
        for key, val in layer_entry.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_keys_seen += 1
                assert np.isfinite(val), (
                    f"per_layer[{key}] = {val} is not finite"
                )
    assert numeric_keys_seen > 0, (
        "no numeric fields found anywhere in per_layer — "
        "_assemble_per_layer's schema may have changed"
    )


def test_phase1b_summary_has_verdict_fields(tiny_phase1b_dir):
    """summary is built by _build_summary from blocks 0/1/2/3/4 — this is
    what the cross-run digest and the one-page synthesis actually read, so
    an empty summary would mean the pipeline ran but produced nothing
    usable."""
    json_path = tiny_phase1b_dir / f"phase1h_{STEM}.json"
    with open(json_path) as f:
        data = json.load(f)
    assert data["summary"], "summary dict is empty"
