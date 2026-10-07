"""
`p1e_energy_field/u2_block.py` — U2's block arm on synthetic clouds: an exact mean-shift step
reads +1, its reverse −1 (`design-1e.md` "U2's block arm: the rule", first checks).
"""
import numpy as np
import pytest

from p1e_energy_field import u2_block as u2

pytestmark = pytest.mark.pure


def _cloud(n=60, d=16, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _step(U, g, eta):
    V = U + eta * g
    return V / np.linalg.norm(V, axis=1, keepdims=True)


@pytest.mark.parametrize("src", u2.SOURCES)
@pytest.mark.parametrize("sign", [1, -1])
def test_exact_mean_shift_step_reads_plus_or_minus_one(src, sign):
    U = _cloud()
    g = u2.forces(U, 3.5)[src]
    d = u2.tangent(U, _step(U, g, sign * 0.05) - U)
    tgt = np.arange(2, U.shape[0])
    c = u2.cell(d, g, U, tgt, u2.make_perms("synthetic"))
    assert c["A"] == pytest.approx(sign, abs=1e-9)
    assert c["n"] == tgt.size and c["left_out"] == 0
    assert (c["p_hi"] if sign < 0 else c["p_lo"]) == 1.0
    assert sign * c["X"] > 0


def test_force_is_tangent_and_the_sink_alone_has_none():
    U = _cloud()
    f = u2.forces(U, 3.5)
    for g in f.values():
        assert np.abs(np.sum(g * U, axis=1)).max() < 1e-12
    assert np.linalg.norm(f["causal"][0]) < 1e-12          # row 0 feels only itself
    assert np.linalg.norm(f["nosink"][1]) < 1e-12          # row 1 without the sink: only itself


def test_left_out_tokens_are_counted():
    U = _cloud()
    g = u2.forces(U, 3.5)["nosink"]
    d = u2.tangent(U, _step(U, u2.forces(U, 3.5)["causal"], 0.05) - U)
    c = u2.cell(d, g, U, np.arange(1, U.shape[0]), u2.make_perms("x"))
    assert c["left_out"] == 1 and c["n"] == U.shape[0] - 2


def test_null_cos_identity_permutation_is_the_observed_cosine():
    U = _cloud()
    g = u2.forces(U, 3.5)["causal"][5:]
    d = u2.tangent(U, np.random.default_rng(1).normal(size=U.shape))[5:]
    Gh = g / np.linalg.norm(g, axis=1, keepdims=True)
    a, nul = u2.null_cos(d @ Gh.T, d @ U[5:].T, np.sum(d * d, axis=1),
                         np.arange(len(d))[None, :])
    direct = np.sum(d * Gh, axis=1) / np.linalg.norm(d, axis=1)
    assert np.allclose(a, direct) and nul[0] == pytest.approx(a.mean())


def test_shuffled_move_is_projected_onto_the_targets_tangent_space():
    U = _cloud()
    g = u2.forces(U, 3.5)["causal"]
    d = u2.tangent(U, np.random.default_rng(2).normal(size=U.shape))
    t = np.arange(3, 10)
    Gh = g[t] / np.linalg.norm(g[t], axis=1, keepdims=True)
    perm = np.roll(np.arange(t.size), 1)
    _, nul = u2.null_cos(d[t] @ Gh.T, d[t] @ U[t].T, np.sum(d[t] ** 2, axis=1), perm[None, :])
    want = []
    for i in range(t.size):
        dk = u2.tangent(U[t][i:i + 1], d[t][perm[i]][None, :])[0]
        want.append(dk @ Gh[i] / np.linalg.norm(dk))
    assert nul[0] == pytest.approx(np.mean(want))


def test_unit_rows_match_core_ln_frame():
    from core.ln_frame import ln_transform
    X = np.random.default_rng(3).normal(size=(5, 8)) * 7
    w, b = np.linspace(0.5, 2, 8), np.linspace(-0.1, 0.1, 8)
    Y = ln_transform(X, gamma=w, beta=b, eps=1e-5)
    assert np.allclose(u2.unit_rows(X, w, b, 1e-5), Y / np.linalg.norm(Y, axis=1, keepdims=True))


def test_block_23_refused(tmp_path):
    with pytest.raises(ValueError, match="block 23 refused"):
        acts = np.tile(np.eye(4)[None, :1], (25, 3, 1)).astype(np.float32)
        np.savez(tmp_path / "activations.npz", activations=acts, norms=np.ones((25, 3)))
        ln1 = {"w": np.ones((24, 4)), "b": np.zeros((24, 4)), "eps": 1e-5}
        u2.read_run(tmp_path, ln1, {"t": np.array([1, 2])}, "k", blocks=[23])


def test_off_unit_rows_refused(tmp_path):
    acts = np.ones((25, 3, 4), dtype=np.float32)
    np.savez(tmp_path / "activations.npz", activations=acts, norms=np.ones((25, 3)))
    with pytest.raises(ValueError, match="off unit"):
        u2.read_run(tmp_path, {}, {"t": np.array([1])}, "k")


def test_label_rule_and_chance_counts_match_the_design():
    from p1e_energy_field.u2_report import chance, label
    assert label([0.1] * 8) == "ascends" and label([-0.1] * 8) == "descends"
    assert label([0.1] * 7 + [-0.1]) == "leans ascends"
    assert label([-0.1] * 7 + [0.1]) == "leans descends"
    assert label([0.1] * 6 + [-0.1] * 2) == "mixed"
    assert label([0.1] * 6 + [-0.1]) == "leans ascends"       # v1: 6 of 7
    c = chance(8, 54)                                            # design: 0.42 and 3.4
    assert round(c["ascends_or_descends"], 2) == 0.42 and round(c["leans"], 1) == 3.4


def test_isolated_lean_needs_an_adjacent_step_in_the_same_direction():
    from p1e_energy_field.u2_report import isolated
    k = ("t12", "causal", 3.5, 512, "L1-8")
    labs = {k: "leans ascends", ("t12", "causal", 3.5, 256, "L1-8"): "mixed",
            ("t12", "causal", 3.5, 1000, "L1-8"): "leans descends"}
    assert isolated(labs, k)
    labs[("t12", "causal", 3.5, 256, "L1-8")] = "ascends"
    assert not isolated(labs, k)
