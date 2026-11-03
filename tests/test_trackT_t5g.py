"""T5 U4s, the screen at init (`trackT_timaeus/t5g.py`, design-T5.md "U4s"): the cells and the rule
(pure parts; the builds on real transformers: `test_trackT_t5g_smoke.py`)."""
import pytest

pytestmark = pytest.mark.deps  # a1a's imports, transformers
from trackT_timaeus import t5, t5b, t5g


def test_the_cells():
    ns = t5g.names()
    assert len(ns) == 79 and len(set(ns)) == 79 and len(t5g.cells()) == 158
    assert ns[0] is None and t5g.subset_name(t5b.PARTS) == "all" and ns[63] == "all"
    assert set(t5b.SWAPS) <= set(ns)                         # U1b's 8 swaps and its base, the same keys
    assert t5g.cell_key("F", "ln+par") == "32m-llama-F-nob+ln+par"
    assert t5g.parts_of("all") == t5b.PARTS + ("untied",)
    assert t5g.parts_of("rev-mha") == ("ln", "par", "gelu", "rot", "tok", "untied")
    assert t5g.parts_of("rev-all6") == ("untied",) and t5g.parts_of("gate+act") == ("gate", "act")
    assert t5g.split_names() == ["gate", "act", "width", "gate+act", "gate+width", "act+width", "gate+act+width"]
    with pytest.raises(t5g.T5gRefused):
        t5g.parts_of("gelu+gate")
    assert (t5g.KNOWN_TOL, t5g.SD_TOL, t5g.BUILT_TOL, t5g.GELU_WIDTH) == (1e-6, 0.02, 0.05, 1536)


def _med(e0=0.36, bp=0.48, removes=None, null=0.0, rev=None):
    """Base at (bp, bp + e0); cell n removes ``removes[n]`` of E0 by lowering F; reverse cells at
    E_all + ``rev[n]``."""
    removes, rev = removes or {}, rev or {}
    med = {}
    for n in t5g.names():
        mv = null if n == "untied" else 0.0
        med[("P", n)] = bp + mv
        med[("F", n)] = bp + e0 + mv - removes.get(n, 0.0)
    e_all = e0 - removes.get("all", 0.0)
    for n in t5g.rev_names():
        med[("F", n)] = med[("P", n)] + e_all + rev.get(n, 0.0)
    return med


def test_minimal_carrying_sets():
    r = t5g.u4s_reading(_med(removes={"par+gelu": 0.2, "ln+par+gelu": 0.25, "all": 0.4, "rot+mha": 0.19}))
    assert r["bar"] == 0.1 and r["minimal_carrying_sets"] == ["par+gelu", "rot+mha"]
    assert r["screen_label"]["ln+par+gelu"] == "carries" and r["screen_label"]["ln"] == "does not"
    r = t5g.u4s_reading(_med(removes={"all": 0.4, "gelu": 0.15}))
    assert r["minimal_carrying_sets"] == ["all"] and r["reading"] == "only all six together"
    assert r["screen_label"]["gelu"] == "partly"


def test_reverse_labels_and_null():
    r = t5g.u4s_reading(_med(removes={"all": 0.4}, rev={"rev-gelu": 0.2, "rev-par": 0.12, "rev-ln": -0.15,
                                                        "rev-all6": 0.4}))
    assert r["reverse_label"] == {"rev-ln": "opposes", "rev-par": "partly", "rev-gelu": "suffices",
                                  "rev-rot": "does not", "rev-mha": "does not", "rev-tok": "does not"}
    assert r["reverse_reading"] == "suffices alone: gelu" and abs(r["reverse_G"]["rev-all6"] - 0.4) < 1e-12
    r = t5g.u4s_reading(_med(null=0.07))
    assert abs(r["tau"] - 0.07) < 1e-12 and abs(r["bar"] - 0.14) < 1e-12


def test_beside_interactions():
    r = t5g.u4s_reading(_med(removes={"ln": 0.05, "par": 0.02, "ln+par": 0.2, "all": 0.4}))
    v = r["beside"]["rho"]
    assert abs(v["pair_interaction"]["ln+par"] - 0.13) < 1e-12 and abs(v["pair_interaction"]["rot+mha"]) < 1e-12
    assert abs(v["mean_dE_by_size"][6] - 0.4) < 1e-12 and abs(v["share"]["all"] - 0.4 / 0.36) < 1e-12
