"""
`tools/run/p10_label_source.py` — the Phase 10 re-read's label source (R0), and
`move_text.p0_match`, the check that joins unit 1's pass to Stage 0's.

Synthetic clouds with planted groups: the ladder's filters remove exactly the
groups they name, ids carry across columns, every column but c0 lives on the
kept domain, and the source refuses (prompt, layer or read) rather than
degrading when unit 1's groups, match or record do not line up.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble import arch_null as an
from p1d_cluster_ensemble import move_text as mt
from tools.run import p10_label_source as ls


def test_bulk_share_is_arch_nulls():
    assert ls.BULK_SHARE == an.BULK_SHARE


def test_labels_of_keeps_ids_and_refuses_overlap():
    assert ls.labels_of(6, [[0, 1], [3, 4]], [False, True]) == [-1, -1, -1, 1, 1, -1]
    assert ls.labels_of(4, [[0, 1], [2]]) == [0, 0, 1, -1]
    with pytest.raises(ValueError):
        ls.labels_of(4, [[0, 1], [1, 2]])


def test_ladder_keeps_is_cumulative():
    # group 0 bulk (moves), 1 unstable, 2 floor 0, 3 stable not moving, 4 moves
    sizes, n = [30, 3, 3, 3, 3], 100
    classes = ["moves", "unstable", mt.FLOOR_ZERO, "context-bound", "moves"]
    k = ls.ladder_keeps(sizes, classes, n)
    assert k["c2b"] == [False, True, True, True, True]
    assert k["c2"] == [False, False, False, True, True]
    assert k["c3"] == [False, False, False, False, True]


def test_readable():
    assert ls.readable([0] * 10 + [-1] * 10)
    assert not ls.readable([0] * 9 + [-1] * 20)
    assert not ls.readable([0] * 20 + [-1] * 9)


def _planted(n_per=6, n_groups=5, d=16, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_groups, d)) * 5
    Y = np.concatenate([c + 0.05 * rng.normal(size=(n_per, d)) for c in centres])
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def test_p0_match(tmp_path):
    rng = np.random.default_rng(1)
    H = rng.normal(size=(25, 7, 8)).astype(np.float32)
    n = np.linalg.norm(H, axis=-1)
    np.savez(tmp_path / "activations.npz", activations=H / n[..., None], norms=n)
    assert mt.p0_match(H, tmp_path)["ok"]
    H2 = H.copy()
    H2[3, 2] *= 1.001                      # norm off by 1e-3, direction unchanged
    m = mt.p0_match(H2, tmp_path)
    assert not m["ok"] and m["rel_norm_max"] > 1e-4 and m["direction_max"] < 1e-6
    assert not mt.p0_match(H[:, :6], tmp_path)["ok"]


def _fixture(tmp_path, n_tokens=40, kept_drop=(0, 5)):
    """A Stage 0 run dir and a unit 1 record on the same planted cloud, every layer alike."""
    Yk = _planted()                                    # 30 kept tokens, 5 groups of 6 (20 %)
    kept = np.asarray([o for o in range(n_tokens) if o not in kept_drop][:Yk.shape[0]])
    A = np.random.default_rng(2).normal(size=(25, n_tokens, Yk.shape[1]))
    A /= np.linalg.norm(A, axis=-1, keepdims=True)
    A[:, kept] = Yk
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "activations.npz", activations=A.astype(np.float32),
             norms=np.ones((25, n_tokens), dtype=np.float32))
    (run / "hdbscan_labels.json").write_text(json.dumps({str(L): [L % 3] * n_tokens for L in range(25)}))
    layers = []
    for L in ls.LAYERS:
        for frame in mt.FRAMES:
            lay = {"layer": L, "frame": frame, "mcs": {}}
            for mcs in mt.MIN_CLUSTER_SIZES:
                _, g = mt.groups_of(Yk, frame, mcs)
                offs = [kept[x].tolist() for x in g]
                # group 0 moves, group 1 does not, the rest unstable
                groups = [{"offsets": o, "stable": i < 2, "J0": 0.5, "median": 0.9 if i < 2 else 0.1,
                           "opening": False} for i, o in enumerate(offs)]
                bj = [1.0 if i == 0 else 0.0 for i in range(len(offs))]
                conds = {f"{s}|{P}|{j}": {"best_jaccard": list(bj)} for s in ("a", "b") for P in (50, 1000)
                         for j in mt.JOINS}
                lay["mcs"][str(mcs)] = {"groups": groups, "opening_group": None, "c_holds": False,
                                        "conditions": conds}
            layers.append(lay)
    u1 = {"n_passage": n_tokens, "kept_offsets": kept.tolist(), "layers": layers,
          "p0_match": {"ok": True, "run_dir": str(run)}}
    return u1, run, kept


def test_build_prompt_columns(tmp_path):
    u1, run, kept = _fixture(tmp_path)
    rec = ls.build_prompt("step512", "wiki_paragraph", u1, run, None)
    assert not rec["refused_layers"] and set(rec["layers"]) == {str(L) for L in ls.LAYERS}
    cols = rec["layers"]["3"]
    assert set(cols) == set(ls.COLUMNS)
    assert all(len(cols[c]) == 40 for c in ls.ALL_POSITIONS)
    assert all(len(cols[c]) == kept.size for c in ls.COLUMNS if c not in ls.ALL_POSITIONS)
    c2a, c2, c3 = (np.asarray(cols[c]) for c in ("c2a", "c2", "c3"))
    assert len(set(c2a[c2a >= 0])) == 5                      # the 5 planted groups
    assert set(c2[c2 >= 0]) == {0, 1} and set(c3[c3 >= 0]) == {0}
    assert np.array_equal(c3 >= 0, c2a == 0)                  # ids carry across columns
    # shipped HDBSCAN finds the same 5 planted groups
    assert len(set(cols["c1c"]) - {-1}) == 5


def test_build_prompt_refuses(tmp_path):
    u1, run, _ = _fixture(tmp_path)
    bad = json.loads(json.dumps(u1))
    lay = next(x for x in bad["layers"] if x["layer"] == 7 and x["frame"] == "centred")
    g = lay["mcs"]["2"]["groups"]
    g[0]["offsets"], g[1]["offsets"] = g[0]["offsets"] + g[1]["offsets"][:1], g[1]["offsets"][1:]
    rec = ls.build_prompt("step512", "wiki_paragraph", bad, run, None)
    assert set(rec["refused_layers"]) == {"7"} and "7" not in rec["layers"]
    assert "refused" in ls.build_prompt("step512", "x", u1 | {"p0_match": {"ok": False}}, run, None)
    assert "refused" in ls.build_prompt("step512", "x", u1, tmp_path, None)          # another run dir
    assert "refused" in ls.build_prompt("step512", "x", u1 | {"n_passage": 31}, run, None)


def test_learned_join_at_143000(tmp_path):
    u1, run, _ = _fixture(tmp_path)
    lr = {}
    for lay in u1["layers"]:
        if lay["frame"] == "centred":
            lr[("p", lay["layer"])] = {frozenset(g["offsets"]): "learned only" for g in lay["mcs"]["2"]["groups"]}
    rec = ls.build_prompt(ls.LEARNED_STEP, "p", u1, run, lr)
    assert rec["layers"]["1"]["learned"] == {"0": True}
    rec = ls.build_prompt(ls.LEARNED_STEP, "p", u1, run, {})
    assert len(rec["refused_layers"]) == 24


def test_load_column_domain_and_refusals(tmp_path):
    u1, run, kept = _fixture(tmp_path)
    rec = ls.build_prompt("step512", "p", u1, run, None)
    rec["refused_layers"] = {"9": "why"}
    del rec["layers"]["9"]
    (tmp_path / "step512.json").write_text(json.dumps({"prompts": {"p": rec}, "refused": {"q": "no match"}}))
    pos, lab = ls.load_column(tmp_path, "step512", "p", 3, "c3")
    assert pos.tolist() == kept.tolist() and lab.size == kept.size
    pos0, lab0 = ls.load_column(tmp_path, "step512", "p", 3, "c0")
    assert pos0.tolist() == list(range(40))
    for args in (("step512", "p", 9, "c3"), ("step512", "q", 3, "c3"), ("step512", "p", 3, "c9"),
                 ("step64", "p", 3, "c3")):
        with pytest.raises(ls.LabelSourceError):
            ls.load_column(tmp_path, *args)
    with pytest.raises(ls.LabelSourceError):
        ls.load_learned(tmp_path, "step512", "p", 3)
    s = ls.step_summary({"prompts": {"p": rec}, "refused": {"q": "x"}})
    assert s["columns"]["c3"]["groups"] == 23 and s["refused_layers"] == 1


def test_classes_follow_their_groups_when_unit1_order_differs(tmp_path):
    """/challenge-pr on #145, finding 3: a class must stay with its own group, whatever
    order unit 1 lists the groups in."""
    u1, run, kept = _fixture(tmp_path)
    moving = None
    for lay in u1["layers"]:
        for m in lay["mcs"].values():
            if lay["layer"] == 3 and lay["frame"] == "centred" and m is lay["mcs"]["2"]:
                moving = set(m["groups"][0]["offsets"])
            m["groups"].reverse()
            for c in m["conditions"].values():
                c["best_jaccard"].reverse()
    rec = ls.build_prompt("step512", "p", u1, run, None)
    assert not rec["refused_layers"]
    c3 = np.asarray(rec["layers"]["3"]["c3"])
    assert set(kept[c3 >= 0].tolist()) == moving and len(set(c3[c3 >= 0])) == 1


def test_c3_member_mismatches(tmp_path):
    u1, run, kept = _fixture(tmp_path)
    rec = ls.build_prompt("step512", "p", u1, run, None)
    d = {"prompts": {"p": rec}}
    rows = []
    for lay in u1["layers"]:
        if lay["frame"] == "centred":
            g = lay["mcs"]["2"]["groups"]
            rows += [{"prompt": "p", "layer": lay["layer"], "frame": "centred", "size": 2, "bulk": False,
                      "class": "moves" if i == 0 else "unstable", "members": x["offsets"]} for i, x in enumerate(g)]
    assert ls.c3_member_mismatches(d, rows) == []
    rows[0]["class"], rows[1]["class"] = "unstable", "moves"      # the class on the wrong group
    assert ls.c3_member_mismatches(d, rows) == [("p", 1)]


def test_c0f_is_the_shipped_call_on_float64():
    import hdbscan
    from core.metrics import cosine_distance_matrix
    X = _planted()
    want = hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed").fit_predict(cosine_distance_matrix(X))
    assert ls.shipped_f64(X) == want.tolist()


def test_build_resume_refuses_when_a_unit1_record_changed(tmp_path, capsys):
    """CodeRabbit on #145: a step file is reused only if the unit 1 records it read are unchanged."""
    u1, run, kept = _fixture(tmp_path)
    ts = tmp_path / "token_sets.json"
    ts.write_text(json.dumps({"sets": {"wiki_paragraph": {"kept": kept.tolist()}}}))
    u1["meta"] = {"kept_from": {"sha256": ls._sha(ts)}}
    rec = tmp_path / "unit1" / "step512" / "wiki_paragraph.json"
    rec.parent.mkdir(parents=True)
    rec.write_text(json.dumps(u1))
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps({"runs": {"512|wiki_paragraph": str(run)}}))
    rows = tmp_path / "rows.json"
    rows.write_text("{}")
    args = ["--unit1", str(tmp_path / "unit1"), "--token-sets", str(ts), "--rows", str(rows),
            "--index", str(idx), "--steps", "step512", "--out", str(tmp_path / "labels"), "--workers", "1"]
    ls.build(args)                                   # other passages absent: refused, file written
    assert "wiki_paragraph" in json.loads((tmp_path / "labels" / "step512.json").read_text())["prompts"]
    capsys.readouterr()
    ls.build(args)
    assert "already" in capsys.readouterr().out
    u1["n_kept_note"] = "changed"
    rec.write_text(json.dumps(u1))
    assert ls.build(args) == 1
    assert "refusing" in capsys.readouterr().err
