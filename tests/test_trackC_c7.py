"""C7 (i), Isobe et al.'s three phases on 410m (`trackC_writeup/c7_profile.py`, design-C7.md): pure parts."""
import numpy as np
import pytest

pytestmark = pytest.mark.deps  # a1a's imports
from trackC_writeup import c7_profile as c


def test_criterion_is_the_papers_b3_and_sign():
    for a in (0.3, 0.6, 1.0):
        r = c.criterion(c.template(a))
        assert r["full"] and r["window"] == [10, 13] and r["argmin"] in (11, 12)
        assert r["a"] == pytest.approx(a, rel=0.25)  # F_pl and the other tail bias the fit
    # our energies rise with clustering: the template's E1 = -F reads the same once negated
    assert c.criterion(-(-c.template(0.6)))["full"]
    assert not c.criterion(-c.template(0.6))["full"]


def test_criterion_refuses_the_wrong_shapes():
    P = c.plants()
    assert not c.criterion(P["linear"])["full"] and not c.criterion(P["logistic"])["full"]
    assert not c.criterion(P["midstack"])["plateau"]


def test_embedding_blind_spot_and_the_1_23_range():
    emb = c.plants()["embedding"]
    assert not c.criterion(emb)["terminal"]
    r = c.criterion(emb[1:], t0=1)
    assert r["terminal"] and r["T"] == 23 and r["argmax"] == 23


def test_t_esc_in_the_second_half():
    r = c.criterion(c.template(0.6))
    assert r["t_esc"] is not None and r["t_esc"] >= 12


def test_gate_passes_on_this_code():
    assert c.gate_g1()["pass"] and c.gate_g2()["pass"]


def test_split_is_exact_in_shapley_and_signed_at_first_order():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 16)) + 3
    U = 1e-4 * rng.normal(size=X.shape)
    A = 1e-4 * rng.normal(size=X.shape)
    r = c.split(X, X + U + A, U)
    assert r["shap_u"] + r["shap_a"] == pytest.approx(r["delta"], abs=1e-15)
    assert abs(r["remainder"]) < 1e-2 * abs(r["delta"])
    assert c.split(X, None, U).keys() == {"d_u", "speed_u"}


def test_f_label_needs_all_three_blocks():
    good = {b: {"d_u": -2.0, "d_a": 1.0, "shap_u": -2.0, "shap_a": 1.0} for b in c.TERMINAL_BLOCKS}
    assert c.f_label(good) == {"first_order": True, "shapley": True}
    bad = dict(good)
    bad[21] = {"d_u": -0.5, "d_a": 1.0, "shap_u": 0.5, "shap_a": 1.0}
    assert c.f_label(bad) == {"first_order": False, "shapley": False}


def test_layer_form_on_a_planted_diagonal():
    d = 1024
    heads = []
    for h, lam in enumerate((3.0, 1.0)):
        Wq = np.zeros((64, d))
        Wk = np.zeros((64, d))
        Wq[0, h] = Wk[0, h] = np.sqrt(lam * 8)
        heads.append({"Wq": Wq, "Wk": Wk})
    r = c.layer_form(heads)
    assert r["l1"] == pytest.approx(3.0) and r["l2"] == pytest.approx(1.0)
    assert r["gap"] == pytest.approx(2 / 3) and r["share"] == pytest.approx(1.0)
    assert c.check_k3(heads) < 1e-10
