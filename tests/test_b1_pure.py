"""B1's statistic and verdict (`trackB_circuit/design-B.md` "B1")."""
import numpy as np
import pytest

from trackB_circuit import b1_prev_token as b1

pytestmark = pytest.mark.pure


def test_s_is_log_offset1_over_median_of_2_to_5():
    sig = np.ones((3, 6, 16))
    sig[:, 1, 2] = np.e ** 2          # L5H2: σ₁ = e² × the rest
    sig[:, 2:6, 9] = [1, 2, 4, 8]     # L5H9: σ₁ = 1, median(σ₂…σ₅) = 3
    s = b1.s_stat(sig)
    assert np.isclose(s[2], 2) and np.isclose(s[9], -np.log(3)) and np.isclose(s[0], 0)
    assert b1.heads_beaten(s) == 15


def test_readings_and_the_pass_rule():
    assert [b1.step_reading(n) for n in (7, 6, 5, 4)] == ["on", "on", "leans", "off"]
    on = {s: "on" for s in b1.PASS_STEPS}
    assert b1.verdict({0: "off", **on})["verdict"] == "sees"
    assert b1.verdict({0: "leans", **on})["verdict"] == "does not see"
    assert b1.verdict({0: "off", **on, 2000: "leans"})["verdict"] == "does not see"
    assert b1.verdict({0: "off", **on, 256: "on", 143000: "off"})["verdict"] == "sees"
    assert b1.verdict({0: "off"})["verdict"] is None


def test_a_zero_sigma_is_held_at_the_floor():
    sig = np.ones((2, 6, 16))
    sig[:, 2:6, 2] = 0.0              # a sharp head: exact zeros at k ≥ 2
    sig[:, 1, 2] = 10.0
    s = b1.s_stat(sig, lo=np.full(2, 0.1))
    assert np.isclose(s[2], np.log(100)) and np.all(np.isfinite(s))


def _gate(tmp_path, *, name="2026-10-14c", code="1e78e2be9d94", fail=("128",), arms=(("camus_letranger", 82),),
          g3=True):
    import json
    from trackB_circuit import b0_gate as g0
    d = tmp_path / name
    d.mkdir()
    steps = {}
    for s in g0.STEPS:
        v = {"pass": str(s) not in fail, "G0": True, "G1": True, "G2": True, "G3": True, "G4a": True,
             "G4b_lin": {"rank1_mismatch": 0}}
        if str(s) in fail:
            v["G3"] = g3
            v["G4b_lin"]["rank1_mismatch"] = len(arms)
        steps[str(s)] = v
    (d / "gate.json").write_text(json.dumps({"code": code, "complete": True, "prompts_hash": "h",
                                             "steps": steps}))
    c = str(g0.G4B_C)
    prompts = {}
    for key, j in arms + (("hdbscan_code", 9),):
        mism = (key, j) in arms
        prompts.setdefault(key, {"G4b": {"arms": []}})["G4b"]["arms"].append(
            {"j": j, "lin_rank": 1, "c": {c: {"rank": 2 if mism else 1}}})
    (d / "gpu_step128.json").write_text(json.dumps({"prompts": prompts}))
    return d


def test_check_gate_accepts_only_the_waived_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(b1.g0, "prompts_hash", lambda: "h")
    assert b1.check_gate(_gate(tmp_path))["waived"] is True
    bad = [dict(name="2026-10-15"), dict(code="000000000000"), dict(fail=("128", "256")),
           dict(arms=(("camus_letranger", 83),)), dict(arms=(("camus_letranger", 82), ("homer_iliad", 40))),
           dict(g3=False)]
    for i, kw in enumerate(bad):
        with pytest.raises(b1.B1Error):
            (tmp_path / str(i)).mkdir()
            b1.check_gate(_gate(tmp_path / str(i), **kw))
