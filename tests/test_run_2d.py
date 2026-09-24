"""
tests/test_run_2d.py — the Phase 2d driver measures and never scores.

`run_2d.py` computes the inputs of two registered gates (`P-T1`, `P-M1`).
Until 2026-09-24 it also adjudicated them and printed `P-T1: <verdict>` per
run, and its D3 records were shaped so that `p_value_p_t1` would have skipped
every head. These tests pin the replacement: gate-ready records, a manifest
that carries the Phase 1 input's battery hash, and no outcome on stdout.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from core.io import write_manifest
from p2d_operator_activation import run_2d
from p2d_operator_activation.gradient_flow_condition import p_value_p_m1
from p2d_operator_activation.table1_predictions import p_value_p_t1

pytestmark = pytest.mark.pure

D, D_HEAD, N_TOK = 8, 2, 150


def _joined(seed=0, n_layers=3, n_heads=4, zero_layer=None):
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n_layers):
        Y = (np.zeros((N_TOK, D)) if i == zero_layer
             else rng.normal(size=(N_TOK, D)))
        heads = [{"head": h,
                  "wq": rng.normal(size=(D, D_HEAD)),
                  "wk": rng.normal(size=(D, D_HEAD)),
                  "ov": rng.normal(size=(D, D))}
                 for h in range(n_heads)]
        pairs.append({"layer": i, "layer_name": f"L{i}", "heads": heads,
                      "Y": Y, "d_head": D_HEAD})
    return {"pairs": pairs, "frame": "raw", "warnings": [],
            "revision": "step0", "n_paired": n_layers,
            "n_heads": n_heads, "d_head": D_HEAD}


def test_d3_records_reach_the_p_t1_gate():
    """Every head is seen by the gate: in an arm, or counted undetermined."""
    res = run_2d.analyse(_joined(), {"D1", "D3"}, [1.0], center_cov=True)
    heads = res["per_head"]
    assert all("stability" in r for r in heads)
    out = p_value_p_t1(heads)
    seen = out["n_candidates"] + out["n_controls"] + out["n_undetermined"]
    assert seen == len(heads)


def test_degenerate_projection_is_counted_not_dropped():
    res = run_2d.analyse(_joined(zero_layer=0), {"D3"}, [1.0], center_cov=True)
    layer0 = [r for r in res["per_head"] if r["layer"] == 0]
    assert layer0 and all(r["stability"]["stable_n_modes"] is None
                          for r in layer0)
    out = p_value_p_t1(res["per_head"])
    assert out["n_undetermined"] >= len(layer0)


def test_d1_records_reach_the_p_m1_gate():
    res = run_2d.analyse(_joined(), {"D1"}, [1.0], center_cov=True)
    out = p_value_p_m1(res["per_head"], [0.0, 1.0, 0.0])
    assert "aggregates" in out and out["aggregates"]


def test_driver_calls_no_adjudicator_or_gate():
    """Scoring is a separate, deliberate step, never a side effect of a run."""
    tree = ast.parse(Path(run_2d.__file__).read_text())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            called.add(f.id if isinstance(f, ast.Name)
                       else f.attr if isinstance(f, ast.Attribute) else "")
    assert not {c for c in called
                if c.startswith(("adjudicate", "p_value"))}


def test_manifest_refuses_without_a_p1_manifest(tmp_path):
    p1 = tmp_path / "p1"
    p1.mkdir()
    with pytest.raises(run_2d.ManifestRefused):
        run_2d.build_manifest(tmp_path / "out", p1_run=p1, model="toy",
                              revision="step0", config={},
                              wall_time_seconds=0.0)


def test_manifest_carries_the_p1_battery(tmp_path):
    p1 = tmp_path / "p1"
    write_manifest(p1, model="toy", prompt_battery_hash="abc123",
                   wall_time_seconds=1.0, hf_revision="step0",
                   checkpoint_step=0, prompt_key="wiki_paragraph")
    m = run_2d.build_manifest(tmp_path / "out", p1_run=p1, model="toy",
                              revision="step0", config={"subexp": ["D1"]},
                              wall_time_seconds=2.0)
    on_disk = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert on_disk == json.loads(json.dumps(m))
    assert m["prompt_battery_hash"] == "abc123"
    assert m["prompt_key"] == "wiki_paragraph"
    assert m["phase"] == "2d" and m["scored"] is False
    assert "git_sha" in m and "git_dirty" in m


def test_main_writes_a_manifest_and_prints_no_outcome(tmp_path, monkeypatch,
                                                      capsys):
    p1 = tmp_path / "toy_wiki_paragraph"
    write_manifest(p1, model="toy", prompt_battery_hash="abc123",
                   wall_time_seconds=1.0, prompt_key="wiki_paragraph")
    joined = _joined()
    run = {"activations": np.zeros((3, N_TOK, D)),
           "energies": {"layers": [{"energies": {"1.0": e}}
                                   for e in (3.0, 2.0, 2.5)]}}
    import p1c_frames.p1c_io as p1c_io
    monkeypatch.setattr(run_2d, "resolve_hf_repo", lambda m, r: m)
    monkeypatch.setattr(run_2d, "load_operators", lambda *a, **k: {})
    monkeypatch.setattr(p1c_io, "load_run", lambda *a, **k: run)
    monkeypatch.setattr(run_2d, "revision_from_run", lambda *a, **k: "step0")
    monkeypatch.setattr(run_2d, "join", lambda *a, **k: joined)

    rc = run_2d.main(["--p2-dir", str(tmp_path), "--model", "toy",
                      "--p1-run", str(p1), "--revision", "step0",
                      "--out", str(tmp_path / "out"), "--raw-frame",
                      "--subexp", "D1", "D3", "--pm1-beta", "1.0"])
    assert rc == 0
    out_dir = tmp_path / "out" / p1.name
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "p2d.json").exists()
    stdout = capsys.readouterr().out
    for word in ("P-T1", "P-M1", "verdict", "CONFIRMED", "FALSIFIED",
                 "p_value", "gradient-flow regime"):
        assert word not in stdout


def test_main_refuses_when_the_p1_run_has_no_manifest(tmp_path, monkeypatch):
    p1 = tmp_path / "bare"
    p1.mkdir()
    import p1c_frames.p1c_io as p1c_io
    monkeypatch.setattr(run_2d, "resolve_hf_repo", lambda m, r: m)
    monkeypatch.setattr(run_2d, "load_operators", lambda *a, **k: {})
    monkeypatch.setattr(p1c_io, "load_run",
                        lambda *a, **k: {"activations": np.zeros((3, N_TOK, D))})
    monkeypatch.setattr(run_2d, "revision_from_run", lambda *a, **k: "step0")
    monkeypatch.setattr(run_2d, "join", lambda *a, **k: _joined())
    rc = run_2d.main(["--p2-dir", str(tmp_path), "--model", "toy",
                      "--p1-run", str(p1), "--revision", "step0",
                      "--out", str(tmp_path / "out"), "--raw-frame"])
    assert rc == 4
    assert not (tmp_path / "out" / p1.name / "p2d.json").exists()
