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
