"""A1b's rule and its per-step vote (`trackA_force/design-A.md` "A1b, the reading")."""

import pytest

from trackA_force import a1b


def test_cell_rule():
    assert a1b.read_cell(1.0, 2.0, 3.0) == "below"
    assert a1b.read_cell(2.5, 2.0, 3.0) == "straddles"
    assert a1b.read_cell(2.0, 2.0, 3.0) == "straddles"
    assert a1b.read_cell(3.0, 2.0, 3.0) == "not below"
    assert a1b.read_cell(1.0, None, 3.0) == "not read"
    assert a1b.read_cell(1.0, 2.0, float("inf")) == "not read"
    with pytest.raises(ValueError):
        a1b.read_cell(1.0, 3.0, 2.0)


def test_step_vote():
    assert a1b.read_step(["below"] * 6 + ["straddles"]) == "below"
    assert a1b.read_step(["below"] * 5 + ["straddles"] * 2) == "mixed"
    assert a1b.read_step(["not read"] * 7) == "not read"


def test_beta_doubling_brackets_the_table():
    # design-A.md's (6.9) table, g0 = 0, n = 467: t* 4.17 at β 0, 8.70 at 7
    b = a1b.beta_doubling(467, 0.0)
    assert 6.5 < b < 7.5
