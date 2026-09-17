"""
tools/score_claim_c.py: the plumbing from Phase 1 run directories into
CLAIM-C's gate. What is tested is discovery, refusal wording and the record,
not the statistic — that is `test_claim_c_null.py`'s job.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools import score_claim_c as sc

pytestmark = pytest.mark.pure

PROMPTS = ["short_heterogeneous", "wiki_paragraph", "sullivan_ballou",
           "paper_excerpt", "homer_iliad", "hdbscan_code", "camus_letranger",
           "latex_monograph"]


def _write_model(run_dir, model, prompts, rng, n_layers, *, signs):
    """`signs` is None for a random arm, else {prompt: six ±1 effect signs}."""
    for p in prompts:
        d = run_dir / f"{model}_{p}"
        d.mkdir(parents=True)
        sg = np.asarray(signs[p], dtype=float) if signs else np.zeros(6)
        geo = [{"ip_mass_near_1": 0.5 + 0.3 * sg[0] + rng.normal(0, 0.05),
                "effective_rank_normed": 10 + sg[1] + rng.normal(0, 0.1),
                "cka_prev": 0.5 + 0.2 * sg[2] + rng.normal(0, 0.03)}
               for _ in range(n_layers)]
        clu = [{"clustering": {"hdbscan": {"noise_fraction": 0.3 + 0.1 * sg[3] + rng.normal(0, 0.02),
                                           "n_clusters": 4 + int(sg[4]) + int(rng.integers(0, 2))}}}
               for _ in range(n_layers)]
        snk = [{"fiedler_mean": 0.4 + 0.2 * sg[5] + rng.normal(0, 0.03)} for _ in range(n_layers)]
        (d / "geometry.json").write_text(json.dumps({"layers": geo}))
        (d / "clustering.json").write_text(json.dumps({"layers": clu}))
        (d / "sinkhorn.json").write_text(json.dumps({"layers": snk}))


def _sign_tables(rng, prompts):
    """
    Reference signs random per (prompt, metric); candidate = reference with
    ONE metric flipped per prompt. Every row then has swing 4 (never the 3/3
    split that makes a row unable to move the statistic), the candidate's
    pattern varies across prompts (so no identical-pattern refusal), and
    homogeneity stays inside the gate's admissible band. What is under test
    here is the plumbing, so the fixture is built to reach the gate, not to
    exercise its refusals -- `test_claim_c_null.py` does that.
    """
    ref = {p: rng.choice([-1.0, 1.0], size=6) for p in prompts}
    cand = {}
    for i, p in enumerate(prompts):
        c = ref[p].copy()
        c[i % 6] *= -1
        cand[p] = c
    return ref, cand


def _write_all_arms(root, prompts, rng, *, step0: bool, split=False):
    ref, cand = _sign_tables(rng, prompts)
    ref_dir = root / "ref" if split else root
    cand_dir = root / "cand" if split else root
    _write_model(ref_dir, sc.REFERENCE_TRAINED, prompts, rng, 36, signs=ref)
    _write_model(ref_dir, sc.REFERENCE_RANDOM, prompts, rng, 36, signs=None)
    _write_model(cand_dir, sc.CANDIDATE_TRAINED, prompts, rng, 24, signs=cand)
    _write_model(cand_dir, sc.CANDIDATE_RANDOM, prompts, rng, 24, signs=None)
    if step0:
        _write_model(cand_dir, sc.CANDIDATE_STEP0, prompts, rng, 24, signs=None)
    return [ref_dir, cand_dir] if split else [root]


@pytest.fixture
def prompts(monkeypatch):
    monkeypatch.setattr(sc, "metastability_prompts", lambda: list(PROMPTS))
    return PROMPTS


def test_refuses_naming_every_absent_arm(tmp_path, prompts):
    rng = np.random.default_rng(0)
    ref, _ = _sign_tables(rng, prompts)
    _write_model(tmp_path, sc.REFERENCE_TRAINED, prompts, rng, 36, signs=ref)
    rec = sc.run([tmp_path])
    assert rec["result"] is None
    for m in (sc.REFERENCE_RANDOM, sc.CANDIDATE_TRAINED, sc.CANDIDATE_RANDOM):
        assert m in rec["refused"]
    assert sc.CANDIDATE_STEP0 not in rec["refused"], "step 0 is a sensitivity arm, not a hard requirement"
    assert rec["arms"][sc.REFERENCE_TRAINED]["n_prompts"] == len(prompts)
    assert rec["artifact_hashes"], "the arm that was found is hashed even on refusal"


def test_control_prompt_is_never_a_unit(tmp_path, prompts):
    rng = np.random.default_rng(1)
    ref, _ = _sign_tables(rng, prompts + ["repeated_tokens"])
    _write_model(tmp_path, sc.REFERENCE_TRAINED, prompts + ["repeated_tokens"], rng, 36, signs=ref)
    rec = sc.run([tmp_path])
    assert "repeated_tokens" not in rec["arms"][sc.REFERENCE_TRAINED]["prompts"]


def test_arms_from_several_run_dirs_reach_the_gate(tmp_path, prompts):
    dirs = _write_all_arms(tmp_path, prompts, np.random.default_rng(2), step0=False, split=True)
    rec = sc.run(dirs)
    assert rec["refused"] is None, rec["refused"]
    res = rec["result"]
    assert res["n_prompts"] == len(prompts)
    assert res["p_value"] is not None and "verdict" in res
    assert res["adjudication"] is None
    assert res["step0_sensitivity"]["available"] is False
    assert len(rec["artifact_hashes"]) == 4 * len(prompts) * 3


def test_step0_arm_is_picked_up_when_present(tmp_path, prompts):
    dirs = _write_all_arms(tmp_path, prompts, np.random.default_rng(3), step0=True)
    rec = sc.run(dirs)
    assert rec["refused"] is None, rec["refused"]
    assert rec["result"]["step0_sensitivity"]["available"] is True


def test_gate_refusal_is_recorded_as_a_refusal(tmp_path, prompts):
    """A gate that ran and refused is a refusal in the record, with the result kept."""
    rng = np.random.default_rng(4)
    same = {p: np.ones(6) for p in prompts}          # identical sign pattern on every prompt
    _write_model(tmp_path, sc.REFERENCE_TRAINED, prompts, rng, 36, signs=same)
    _write_model(tmp_path, sc.REFERENCE_RANDOM, prompts, rng, 36, signs=None)
    _write_model(tmp_path, sc.CANDIDATE_TRAINED, prompts, rng, 24, signs=same)
    _write_model(tmp_path, sc.CANDIDATE_RANDOM, prompts, rng, 24, signs=None)
    rec = sc.run([tmp_path])
    assert rec["refused"] and "SAME candidate sign pattern" in rec["refused"]
    assert rec["result"] is not None and rec["result"]["p_value"] is None
