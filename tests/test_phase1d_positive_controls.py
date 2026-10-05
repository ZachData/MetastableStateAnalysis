"""
Unit 4 on the designed prompts (`p1d_cluster_ensemble/positive_controls.py`):
prompt sets and seeds (v1 unchanged), the group route's join and pass on rows
with known answers, and the reader's partition scoring.
"""

import json

import pytest

from p1d_cluster_ensemble import arch_null as an
from p1d_cluster_ensemble import designed_prompts as dp
from p1d_cluster_ensemble import positive_controls as pc
from p1d_cluster_ensemble import scale_real as sr
from p1d_cluster_ensemble.move_text import V1_PASSAGES

pytestmark = pytest.mark.deps


class TestPromptSets:
    def test_v1_indices_unchanged_designed_after(self):
        assert [an.prompt_index(k) for k in V1_PASSAGES] == list(range(7))
        assert [an.prompt_index(k) for k in dp.KEYS] == [7, 8, 9]
        assert an.prompt_keys() == tuple(V1_PASSAGES) and an.prompt_keys("designed") == tuple(dp.KEYS)
        with pytest.raises(ValueError):
            an.prompt_keys("v2")

    def test_v1_cloud_seeds_unchanged(self):
        # The step-0 batch and the trained reading were read with these.
        assert sr.cloud_seed("wiki_paragraph", 1) == 10_001
        assert sr.cloud_seed(V1_PASSAGES[6], 24) == 10_624
        assert sr.cloud_seed(dp.KEYS[0], 1) == 10_701

    def test_load_sets_refuses_another_prompt_set(self, tmp_path):
        f = tmp_path / "s.json"
        comp = [an.label(st, m) for st, m in an.comparison_models("trained")]
        f.write_text(json.dumps({"comparison": comp, "sets": {k: {"kept": [1, 2]} for k in dp.KEYS}}))
        with pytest.raises(SystemExit):
            sr.load_sets(f)
        assert sr.load_sets(f, "designed")[dp.KEYS[0]] == [1, 2]


LABELS = {k: ["a"] * 10 + ["b"] * 10 + [None] * 10 for k in dp.KEYS}


def _g(seed, prompt, members, learned, layer=1, frame="centred", size=2):
    return {"seed": seed, "prompt": prompt, "layer": layer, "frame": frame, "size": size,
            "members": members, "s": 2.0, "bar": 1.5, "admitted": True, "learned": learned}


class TestGroupRoute:
    def test_content_rule_is_unit_1s(self):
        assert pc.group_content([1, 2, 3], LABELS[dp.KEYS[0]]) == "a"
        assert pc.group_content([1, 2], LABELS[dp.KEYS[0]]) is None          # < 3 members of a label
        assert pc.group_content([1, 2, 3, 11], LABELS[dp.KEYS[0]]) == "a"     # 3 / 4 = 0.75
        assert pc.group_content([1, 2, 3, 11, 12], LABELS[dp.KEYS[0]]) is None
        assert pc.group_content([1, 2, 3, 25, 26], LABELS[dp.KEYS[0]]) == "a"  # unlabelled not counted

    def test_moves_needs_jaccard_and_class(self):
        k = dp.KEYS[0]
        u1 = {(k, 1, "centred", 2): [([1, 2, 3, 4], "moves"), ([11, 12, 13], "context-bound")]}
        rows = [_g(0, k, [1, 2, 3], True), _g(0, k, [11, 12, 13], True), _g(0, k, [1, 2, 3, 5, 6, 7, 8, 9], True)]
        pc.annotate(rows, LABELS, u1)
        assert [r["moves"] for r in rows] == [True, False, False]  # J 0.75; class; J 4/9 < 0.5
        assert rows[1]["unit1_class"] == "context-bound"

    def test_pass_needs_a_candidate_and_beats_every_step0_init(self):
        k = dp.KEYS[0]
        u1 = {(k, 1, "centred", 2): [([1, 2, 3], "moves")]}
        trained = [_g(0, k, [1, 2, 3], True), _g(0, k, [11, 12, 13], True)]
        step0 = [_g(sd, k, [1, 2, 3], sd == 4) for sd in range(10)]
        pc.annotate(trained, LABELS, u1)
        pc.annotate(step0, LABELS)
        v = pc.prompt_verdicts(trained, step0, list(range(10)))[k]
        assert (v["candidates"], v["content_learned"], v["step0_max"]) == (1, 2, 1) and v["pass"]
        step0.append(_g(4, k, [11, 12, 13], True))      # one init reaches 2: not exceeded
        pc.annotate(step0, LABELS)
        assert not pc.prompt_verdicts(trained, step0, list(range(10)))[k]["pass"]

    def test_no_candidate_fails_and_refused_counts_as_not_learned(self):
        k = dp.KEYS[1]
        trained = [_g(0, k, [1, 2, 3], None), _g(0, k, [11, 12, 13], True)]
        pc.annotate(trained, LABELS, {})
        v = pc.prompt_verdicts(trained, [], list(range(10)))[k]
        assert v["refused"] and v["candidates"] == 0 and not v["pass"]

    def test_verdict_two_of_three(self):
        assert pc.verdict({"x": {"pass": True}, "y": {"pass": True}, "z": {"pass": False}}).startswith("pass")
        assert pc.verdict({"x": {"pass": True}, "y": {"pass": False}, "z": {"pass": False}}).startswith("fail")


class TestReaderScoring:
    def test_pure_partition(self):
        s = pc.score_partition([[0, 1, 2, 3], [10, 11, 12]], LABELS[dp.KEYS[0]])
        assert s["content"] == ["a", "b"] and s["ari"] == pytest.approx(1.0)

    def test_mixed_partition_has_no_content(self):
        s = pc.score_partition([[0, 1, 10, 11], [2, 12]], LABELS[dp.KEYS[0]])
        assert s["n_content"] == 0 and s["ari"] < 0.5

    def test_one_label_has_no_ari(self):
        assert pc.score_partition([[0, 1, 2]], LABELS[dp.KEYS[0]])["ari"] is None


def test_non_finite_bar_refuses_its_cell_only_when_not_strict():
    def rec(model, s):
        return {"model": model, "prompt": "p", "layers": [{"layer": 1, "frame": "raw", "arms": {
            "4": {"groups": [{"members": [1, 2, 3, 4], "excess": 1.0, "s": s}]}}}]}
    reinits = [f"reinit:{i}" for i in range(5)]
    recs0 = [rec(m, None) for m in reinits]          # every re-init cloud's max s is inf
    with pytest.raises(ValueError):
        an.group_bars(recs0, reinits)
    bars = an.group_bars(recs0, reinits, strict=False)
    assert bars[("p", 1, "raw", 4)] is None
    rows = an.group_rules([rec("init:0", 3.0)], bars, set())
    assert rows[0]["learned"] is None and rows[0]["bar"] is None


def test_unit1_join_refuses_other_designed_prompts(tmp_path):
    d = tmp_path / an.TRAINED_STEP
    d.mkdir()
    for k in dp.KEYS:
        (d / f"{k}.json").write_text(json.dumps({"meta": {"designed_hash": "stale"}, "layers": []}))
    with pytest.raises(SystemExit, match="designed prompts"):
        pc.unit1_groups(tmp_path)


def test_filter_ablation_shares():
    k = dp.KEYS[0]
    rows = [_g(0, k, [1, 2, 3], True), _g(0, k, [21, 22, 23], True), _g(0, k, [11, 12, 13], False)]
    pc.annotate(rows, LABELS, {(k, 1, "centred", 2): [([11, 12, 13], "moves")]})
    v = pc.filter_ablation(rows, [], "centred", 2)[k]
    assert v["share_seed0"]["learned"] == 0.5 and v["share_seed0"]["not_learned"] == 1.0
    assert v["share_seed0"]["moves"] == 1.0 and v["share_seed0"]["not_moves"] == 0.5
    assert v["seed0_counts"] == {"content_moves": 1, "content_learned": 1, "content_both": 0}


def test_recovery_and_clusters():
    s = pc.score_partition([[0, 1, 2, 3], [10, 11, 12]], LABELS[dp.KEYS[0]], 20)
    assert s["recovery"] == pytest.approx(7 / 20)
    import numpy as np
    assert pc._clusters(np.array([0, 0, 1, 2, 2, 2]), 2, [3, 5, 6, 8, 9, 11]) == [[3, 5], [8, 9, 11]]
