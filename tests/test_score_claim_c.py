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


def _write_model(run_dir, model, prompts, rng, n_layers, *, trained: bool):
    for p in prompts:
        d = run_dir / f"{model}_{p}"
        d.mkdir(parents=True)
        base = 1.0 if trained else 0.0
        geo = [{"ip_mass_near_1": base + rng.normal(0, 0.3),
                "effective_rank_normed": 10 - base + rng.normal(0, 0.3),
                "cka_prev": 0.5 + 0.2 * base + rng.normal(0, 0.1)}
               for _ in range(n_layers)]
        clu = [{"clustering": {"hdbscan": {"noise_fraction": 0.3 - 0.1 * base + rng.normal(0, 0.05),
                                           "n_clusters": 3 + int(base) + int(rng.integers(0, 2))}}}
               for _ in range(n_layers)]
        snk = [{"fiedler_mean": 0.4 + 0.2 * base + rng.normal(0, 0.1)} for _ in range(n_layers)]
        (d / "geometry.json").write_text(json.dumps({"layers": geo}))
        (d / "clustering.json").write_text(json.dumps({"layers": clu}))
        (d / "sinkhorn.json").write_text(json.dumps({"layers": snk}))


@pytest.fixture
def prompts(monkeypatch):
    monkeypatch.setattr(sc, "metastability_prompts", lambda: list(PROMPTS))
    return PROMPTS


def test_refuses_naming_every_absent_arm(tmp_path, prompts):
    rng = np.random.default_rng(0)
    _write_model(tmp_path, sc.REFERENCE_TRAINED, prompts, rng, 36, trained=True)
    rec = sc.run([tmp_path])
    assert rec["result"] is None
    for m in (sc.REFERENCE_RANDOM, sc.CANDIDATE_TRAINED, sc.CANDIDATE_RANDOM):
        assert m in rec["refused"]
    assert sc.CANDIDATE_STEP0 not in rec["refused"], "step 0 is a sensitivity arm, not a hard requirement"
    assert rec["arms"][sc.REFERENCE_TRAINED]["n_prompts"] == len(prompts)
    assert rec["artifact_hashes"], "the arm that was found is hashed even on refusal"


def test_control_prompt_is_never_a_unit(tmp_path, prompts):
    rng = np.random.default_rng(1)
    _write_model(tmp_path, sc.REFERENCE_TRAINED, prompts + ["repeated_tokens"], rng, 36, trained=True)
    rec = sc.run([tmp_path])
    assert "repeated_tokens" not in rec["arms"][sc.REFERENCE_TRAINED]["prompts"]


def test_arms_from_several_run_dirs_reach_the_gate(tmp_path, prompts):
    rng = np.random.default_rng(2)
    ref, cand = tmp_path / "ref", tmp_path / "cand"
    _write_model(ref, sc.REFERENCE_TRAINED, prompts, rng, 36, trained=True)
    _write_model(ref, sc.REFERENCE_RANDOM, prompts, rng, 36, trained=False)
    _write_model(cand, sc.CANDIDATE_TRAINED, prompts, rng, 24, trained=True)
    _write_model(cand, sc.CANDIDATE_RANDOM, prompts, rng, 24, trained=False)
    rec = sc.run([ref, cand])
    assert rec["refused"] is None
    res = rec["result"]
    assert res["n_prompts"] == len(prompts)
    assert "verdict" in res and "p_value" in res
    assert res["adjudication"] is None
    assert res["step0_sensitivity"]["available"] is False
    assert len(rec["artifact_hashes"]) == 4 * len(prompts) * 3


def test_step0_arm_is_picked_up_when_present(tmp_path, prompts):
    rng = np.random.default_rng(3)
    for m, t, n in ((sc.REFERENCE_TRAINED, True, 36), (sc.REFERENCE_RANDOM, False, 36),
                    (sc.CANDIDATE_TRAINED, True, 24), (sc.CANDIDATE_RANDOM, False, 24),
                    (sc.CANDIDATE_STEP0, False, 24)):
        _write_model(tmp_path, m, prompts, rng, n, trained=t)
    rec = sc.run([tmp_path])
    assert rec["refused"] is None
    assert rec["result"]["step0_sensitivity"]["available"] is True
