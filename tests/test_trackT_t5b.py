"""T5 U1b, the Llama's parts at init (`trackT_timaeus/t5b.py`, design-T5.md "U1b"): pure parts (the
swaps on real transformers: `test_trackT_t5b_smoke.py`)."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports, transformers
from trackT_timaeus import t5, t5b


def test_the_frozen_constants():
    assert t5b.PARTS == ("ln", "par", "gelu", "rot", "mha", "tok") and t5b.SWAPS[-2:] == ("untied", "all")
    assert (t5b.KNOWN_TOL, t5b.SD_TOL, t5b.NULL_MAX, t5b.NEOX_MAX) == (1e-6, 0.02, 0.05, 0.05)
    assert len(t5b.cells()) == 18 and t5b.cell_key("P", None) == "32m-llama-P-nob"
    assert t5b.cell_key("F", "ln") == "32m-llama-F-nob+ln" and t5b.parts_of("all") == t5b.PARTS + ("untied",)


def _med(e0=0.36, bp=0.48, moves=None, removes=None, null=0.0):
    """Medians with the base at (bp, bp + e0); swap k removes ``removes[k]`` of E0 by lowering F."""
    moves, removes = moves or {}, removes or {}
    med = {("P", None): bp, ("F", None): bp + e0}
    for s in t5b.SWAPS:
        mv = moves.get(s, 0.0) + (null if s == "untied" else 0.0)
        med[("P", s)] = bp + mv
        med[("F", s)] = bp + e0 + mv - removes.get(s, 0.0)
    return med


def test_labels():
    r = t5b.u1b_reading(_med(removes={"ln": 0.30, "par": 0.12, "gelu": 0.05, "rot": -0.15, "all": 0.40}))
    assert r["tau"] == 0.02 and r["bar"] == 0.1
    assert r["label"]["ln"] == "carries" and r["label"]["par"] == "partly"
    assert r["label"]["gelu"] == "does not" and r["label"]["rot"] == "amplifies"
    assert r["label"]["all"] == "carries" and r["carries"] == ["ln"] and r["reading"] == "carried by ln"
    assert abs(r["additivity_beside"]["sum_dE_parts"] - 0.32) < 1e-12


def test_none_carries_and_moves_beside():
    r = t5b.u1b_reading(_med(moves={"mha": -0.15}, removes={"mha": 0.0}))
    assert r["reading"] == "no single part" and r["label"]["mha"] == "does not"
    assert r["moves_at_both_inits_beside"]["mha"] and not r["moves_at_both_inits_beside"]["ln"]


def test_null_sets_tau():
    r = t5b.u1b_reading(_med(null=0.07))
    assert abs(r["tau"] - 0.07) < 1e-12 and abs(r["bar"] - 0.14) < 1e-12


def test_partial_rotary_layout():
    f = t5b.partial_inv_freq(64)
    assert f.shape == (32,) and np.all(f[8:] == 0) and f[0] == 1.0 and abs(f[1] - 10000 ** (-2 / 16)) < 1e-15
    src = t5b.neox_to_llama_dims(64)
    assert sorted(src.tolist()) == list(range(64))
    assert src[0] == 0 and src[32] == 8 and src[7] == 7 and src[39] == 15 and src[8] == 16
