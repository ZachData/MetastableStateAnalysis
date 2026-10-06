"""
`tools/run/p10_r5c_gpt2_sink.py` — R5c's reading of 5c's flip with T1–T2 out of the means: the
prompt is the unit, the sign test and the verdict are the ones `design-10.md` "R5c" fixed.
"""
import pytest

from tools.run import p10_r5c_gpt2_sink as r5c

# Tier: numpy only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure


def _rows(gaps, arm="drop"):
    """One prompt's layer rows: ``gaps`` = per-layer raw gap (corrected = raw / 10)."""
    return [{"layer": L, arm: None if g is None else [g, g / 10]} for L, g in enumerate(gaps)]


def test_sign_p_is_the_binomial_tail():
    assert r5c.sign_p(21, 21) == pytest.approx(2 ** -21)
    assert r5c.sign_p(16, 21) == pytest.approx(0.0133, abs=1e-4)
    assert r5c.sign_p(0, 21) == 1.0


def test_a_prompt_is_the_mean_over_its_readable_layers():
    assert r5c.prompt_gap(_rows([0.2, None, 0.4]), "drop", "raw") == pytest.approx(0.3)
    assert r5c.prompt_gap(_rows([0.2, None, 0.4]), "drop", "corrected") == pytest.approx(0.03)
    assert r5c.prompt_gap(_rows([None]), "drop", "raw") is None


def test_the_rule_needs_16_of_21_trained_and_16_of_21_above_random():
    trained = {f"p{i}": _rows([0.5 if i < 16 else -0.1]) for i in range(21)}
    random_ = {f"p{i}": _rows([-0.2]) for i in range(21)}
    out = r5c.read_arm(trained, random_, "drop", "raw")
    assert (out["trained_positive"], out["trained_minus_random_positive"]) == (16, 21)
    assert out["verdict"] == "sign-flipped as 5c stated"
    trained["p0"] = _rows([-0.1])
    assert r5c.read_arm(trained, random_, "drop", "raw")["verdict"] == "does not survive"


def test_a_trained_flip_that_random_matches_does_not_survive():
    """The `all` arm's case: trained positive everywhere, random as large."""
    trained = {f"p{i}": _rows([1.0]) for i in range(21)}
    random_ = {f"p{i}": _rows([1.0 + (0.1 if i % 2 else -0.1)]) for i in range(21)}
    out = r5c.read_arm(trained, random_, "drop", "raw")
    assert out["trained_positive"] == 21 and out["trained_minus_random_positive"] == 11
    assert out["verdict"] == "does not survive"


def test_survives_without_a_reversed_random_arm():
    trained = {f"p{i}": _rows([0.5]) for i in range(21)}
    random_ = {f"p{i}": _rows([0.1]) for i in range(21)}
    assert r5c.read_arm(trained, random_, "drop", "raw")["verdict"] == "survives"


def test_refuses_mismatched_prompts_and_another_prompt_count():
    with pytest.raises(ValueError, match="prompts differ"):
        r5c.read_arm({"a": _rows([1])}, {"b": _rows([1])}, "drop", "raw")
    with pytest.raises(ValueError, match="fixed for 21"):
        r5c.read_arm({"a": _rows([1])}, {"a": _rows([0])}, "drop", "raw")


def test_by_third_pools_units_by_depth():
    rows = {"a": _rows([1.0] * 12 + [2.0] * 12 + [None] * 12)}
    assert r5c.by_third(rows, "drop", "raw") == [1.0, 2.0, None]
