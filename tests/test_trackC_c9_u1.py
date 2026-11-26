"""C9 U1 (`trackC_writeup/c9_u1.py`): the series statistics, the plants and the label."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c9_u1 as c


def _cloud(states, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.05, 0.9, states)
    rest = rng.uniform(0.0, 0.3, states)
    return 0.5 + 0.5 * m + rest, m


def test_series_matches_c2_readings_on_a_25_state_record():
    from trackC_writeup.c2_readings import one_series
    rng = np.random.default_rng(1)
    n = 465
    cos = rng.uniform(0.0, 0.5, 25)
    m = (1 + (n - 1) * cos) / n
    e = 0.5 + 0.5 * m + rng.uniform(0, 0.2, 25)
    r = {"e1": e.tolist(), "cos": cos.tolist(), "n": n, "prompt": "p", "step": 0}
    a, b = one_series(r), c.series(r["e1"], c.m2(r))
    for k in ("d", "m_fall", "e_fall", "same"):
        assert np.array_equal(a[k], b[k])
    assert np.array_equal(a["s"], b["s"], equal_nan=True) and a["net_inner"] == b["net_inner"]


@pytest.mark.parametrize("states", [7, 13, 25])
def test_plants_read_one_and_zero(states):
    e, m = _cloud(states)
    q = c.counts(c.series(*c.plant(e, m, "mean")))
    st = c.stats(q)
    assert q["drops"] > 0 and st["M"] == 1.0 and st["S_any"] == 1.0
    q = c.counts(c.series(*c.plant(e, m, "R")))
    st = c.stats(q)
    assert q["drops"] > 0 and st["M"] == 0.0 and c.label(st["M"], 0.99) == "not"


def test_m_can_fail_where_s_any_is_high():
    # every E₁ fall is R's, with ‖m‖² falling by a hair: S_any 1, M 0
    m = np.linspace(0.5, 0.49, 9)
    rest = np.array([0.3, 0.2, 0.25, 0.1, 0.15, 0.05, 0.08, 0.02, 0.03])
    st = c.stats(c.counts(c.series(0.5 + 0.5 * m + rest, m)))
    assert st["S_any"] == 1.0 and st["M"] == 0.0


def test_label_bands():
    assert c.label(0.95, 0.99) == "carries" and c.label(0.89, 0.99) == "carries"
    assert c.label(0.88, 0.99) == "partly" and c.label(0.5, 0.99) == "partly"
    assert c.label(0.49, 0.99) == "not" and c.label(None, 0.99).startswith("refused")


def test_load_run_refuses_a_missing_prompt(tmp_path):
    import json
    from trackA_force.a1a import V1
    for p in V1[:-1]:
        (tmp_path / f"step0_{p}.json").write_text(json.dumps({"e1": [1, 1], "cos": [0, 0], "n": 3}))
    with pytest.raises(c.Refused):
        c.load_run(tmp_path)


def test_load_run_takes_n_per_prompt_and_refuses_n_changing_across_steps(tmp_path):
    # `/challenge-pr` on #381, finding 5: n is a prompt's token count, one per prompt across steps
    import json
    from trackA_force.a1a import V1
    for st in (0, 2):
        for i, p in enumerate(V1):
            (tmp_path / f"step{st}_{p}.json").write_text(json.dumps({"e1": [1, 1], "cos": [0, 0], "n": 10 + i}))
    steps, _ = c.load_run(tmp_path)                      # n differs by prompt: accepted
    assert steps == [0, 2]
    (tmp_path / f"step2_{V1[0]}.json").write_text(json.dumps({"e1": [1, 1], "cos": [0, 0], "n": 99}))
    with pytest.raises(c.Refused, match="across steps"):
        c.load_run(tmp_path)
