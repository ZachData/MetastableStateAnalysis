"""
tests/test_phase1d_candidates.py — the candidates' origin reading and joins
(`p1d_cluster_ensemble/candidates.py`), and `move_text`'s ``--kept-from`` guard,
on hand-built group lists with known answers.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble import candidates as cd
from p1d_cluster_ensemble import move_text as mt

G = [1, 2, 3, 4]


def _layers(present: dict, n: int = 6) -> dict:
    """Layers 0..n-1; at each, G itself, a near copy (Jaccard 0.6) or an unrelated group."""
    out = {}
    for L in range(n):
        kind = present.get(L, "none")
        g = G if kind == "same" else [1, 2, 3, 9, 10] if kind == "near" else [20, 21]
        out[L] = [g, [30, 31, 32]]
    return out


def test_origin_carried_when_present_from_l0():
    o = cd.origin(G, _layers({L: "same" for L in range(6)}), 4)
    assert o == {"origin": 0, "origin_identical": 0, "first_present": 0, "last_forward": 5}
    assert cd.origin_class(o["origin"]) == "carried"


def test_origin_counts_a_near_copy_but_not_identity():
    by = _layers({0: "near", 1: "same", 2: "same", 3: "same"})
    o = cd.origin(G, by, 3)
    assert o["origin"] == 0 and o["origin_identical"] == 1 and o["last_forward"] == 3


def test_origin_needs_contiguity_back_from_its_layer():
    # present at L0, absent at L1-2, back from L3: formed later, though first present at L0
    by = _layers({0: "same", 3: "same", 4: "same"})
    o = cd.origin(G, by, 4)
    assert o["origin"] == 3 and o["first_present"] == 0 and cd.origin_class(3) == "formed later"
    assert cd.origin_class(cd.origin(G, _layers({1: "same", 2: "same"}), 2)["origin"]) == "formed at L1"


def test_origin_refuses_a_group_not_in_its_own_layer():
    with pytest.raises(ValueError):
        cd.origin(G, _layers({0: "same"}), 2)


def test_same_groups_is_order_free_and_exact():
    assert cd.same_groups([[1, 2], [3, 4, 5]], [[5, 4, 3], [2, 1]])
    assert not cd.same_groups([[1, 2], [3, 4, 5]], [[1, 2], [3, 4]])
    assert not cd.same_groups([[1, 2]], [[1, 2], [1, 2]])


def test_status_and_refusal():
    assert cd.status(True, True) == "learned+replicates"
    assert cd.status(True, False) == "learned only"
    assert cd.status(False, True) == "not learned"
    assert cd.status(None, True) == "refused"


def test_outcome_rows():
    mk = lambda o: {"origin": o}
    assert cd.outcome([mk(0)] * (cd.FEW - 1)) == "few candidates"
    assert cd.outcome([mk(0)] * 11 + [mk(1)] * 10) == "most carried"
    assert cd.outcome([mk(1)] * 11 + [mk(5)] * 10) == "most formed at L1"
    assert cd.outcome([mk(3)] * 11 + [mk(0)] * 10) == "most formed later"
    assert cd.outcome([mk(0)] * 10 + [mk(1)] * 10 + [mk(2)] * 10) == "mixed origins"


def test_tables_dedupe_a_candidate_across_layers_and_set_bulk_apart():
    row = dict(prompt="p", frame="centred", size=2, members=[3, 5], tokens=["a", "b"],
               status="learned+replicates", bulk=False, candidate=True, **{"class": "moves"})
    rows = [dict(row, layer=3, origin=1, origin_class="formed at L1"),
            dict(row, layer=4, origin=1, origin_class="formed at L1"),
            dict(row, layer=5, members=list(range(40)), bulk=True, candidate=False, origin=0,
                 origin_class="carried")]
    t = cd.tables(rows)
    assert len(t["distinct"]) == 1 and t["distinct"][0]["layers"] == [3, 4]
    assert t["distinct_by_frame_size"] == {"centred/2": {"formed at L1": 1}}
    assert t["cross"]["centred/2/L1-8"] == {"moves | learned+replicates": 2}
    assert t["cross_bulk"]["centred/2/L1-8"] == {"moves | learned+replicates": 1}
    assert t["origin"]["centred/2/L1-8"]["candidates"] == {"formed at L1": 2}


def test_definition_counts_filter_dedupe_and_drop_bulk():
    row = dict(prompt="p", frame="centred", size=2, members=[3, 5], tokens=["a", "b"],
               status="not learned", bulk=False, candidate=False, origin=2,
               origin_class="formed later", **{"class": "moves"})
    rows = [dict(row, layer=3), dict(row, layer=12, origin=0),
            dict(row, layer=4, members=[7, 8], status="learned only"),
            dict(row, layer=5, members=[9, 10], **{"class": "unstable"}),
            dict(row, layer=6, members=list(range(40)), bulk=True)]
    c = cd.definition_counts(rows)
    assert c["moves"]["centred/2"] == {"records": 3, "records_by_band": {"L1-8": 2, "L9-16": 1},
                                       "distinct": {"carried": 1, "formed later": 1}, "n_distinct": 2}
    assert c["moves+learned"]["centred/2"]["n_distinct"] == 1
    assert c["candidate"]["centred/2"]["records"] == 0
    assert c["all"]["centred/2"]["records"] == 4


def test_forced_kept_takes_the_given_set_and_refuses_an_extra_offset():
    own = np.array([1, 2, 3, 5, 8])
    assert mt.forced_kept(own, [8, 1, 3], "p").tolist() == [1, 3, 8]
    with pytest.raises(SystemExit):
        mt.forced_kept(own, [1, 4], "p")
