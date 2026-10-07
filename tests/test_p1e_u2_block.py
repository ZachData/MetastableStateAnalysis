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


def _concentrated(n=400, d=64, kappa=1.2, seed=4):
    X = np.random.default_rng(seed).normal(size=(n, d))
    X[:, 0] += kappa * np.sqrt(d) / 4
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.mark.parametrize("sign", [1, -1])
def test_an_update_shared_by_every_token_reads_as_the_field_on_the_frozen_statistic(sign):
    """`/challenge-pr` on #158, finding 1: the frozen X and Xt cannot tell a shared update from
    field-following; ``shared_cells``' ``residout`` reading removes it (up to the noise)."""
    rng = np.random.default_rng(7)
    X = 10 * _concentrated() + rng.normal(size=(400, 64))           # residual rows, LN-able
    c = np.zeros(64)
    c[0] = sign * 2.0
    X2 = X + c + 0.05 * rng.normal(size=X.shape)                     # shared update + noise
    w, b = np.ones(64), np.zeros(64)
    frame = lambda Y: u2.unit_rows(Y, w, b, 1e-5)                    # noqa: E731
    U, U2 = frame(X), frame(X2)
    tgt = np.arange(1, 400)
    g = u2.forces(U, 3.5)["causal"]
    frozen = u2.cell(u2.tangent(U, U2 - U), g, U, tgt, u2.make_perms("s"))
    assert sign * frozen["X"] > 0.02 and sign * frozen["Xt"] > 0.02      # the limitation
    rows = {r["source"]: r for r in u2.shared_cells(U, U2, X, X2, frame, tgt, u2.make_perms("s"))}
    assert abs(rows["causal:residout"]["X"]) < 0.2 * abs(frozen["X"])
    assert abs(rows["mean0:residout"]["X"]) < 0.2 * abs(frozen["X"])
    assert sign * rows["causal:resid"]["X"] > 0.02
    assert rows["mean0:frozen"]["X"] == pytest.approx(
        u2.cell(u2.tangent(U, U2 - U), u2.forces(U, 0.0, only=("mean0",))["mean0"], U, tgt,
                u2.make_perms("s"))["X"])


def test_forces_only_computes_what_is_asked():
    U = _cloud()
    assert set(u2.forces(U, 3.5, only=("mean0",))) == {"mean0"}
    full = u2.forces(U, 3.5)
    assert np.allclose(u2.forces(U, 3.5, only=("local",))["local"], full["local"])
    m0 = np.cumsum(U, axis=0) / np.arange(1, len(U) + 1)[:, None]
    assert np.allclose(u2.forces(U, 0.0, only=("causal",))["causal"], u2.tangent(U, m0))


def _ln_cloud(n=400, d=64, seed=8):
    """Residual rows that LN (gain 1, bias 0) maps to themselves: zero-mean, unit variance."""
    U = _concentrated(n, d, seed=seed)
    U = U - U.mean(axis=1, keepdims=True)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    return np.sqrt(d) * U, (lambda Y: u2.unit_rows(Y, np.ones(d), np.zeros(d), 1e-5))


def test_a_field_step_with_uneven_sizes_still_ascends_under_r1out():
    """`/challenge-pr` on #159, finding 1: ``residout`` reverses small movers' moves; ``r1out``
    removes only each token's own component along the shared direction, so it cannot."""
    X, frame = _ln_cloud()
    U = frame(X)
    g = u2.forces(U, 3.5)["causal"]
    s = np.random.default_rng(9).lognormal(sigma=1.5, size=len(U))      # uneven step sizes
    X2 = X + 0.02 * np.sqrt(X.shape[1]) * s[:, None] * g
    tgt = np.arange(1, len(U))
    rows = {r["source"]: r for r in u2.shared_cells(U, frame(X2), X, X2, frame, tgt, u2.make_perms("u"))}
    assert rows["causal:r1out"]["X"] > 0.02
    assert rows["causal:residout"]["X"] < rows["causal:r1out"]["X"]       # the reversal it fixes


def test_a_shared_direction_at_per_token_weights_reads_zero_under_r1out():
    """`/challenge-pr` on #159, finding 2: a sink-like update ``w_i c`` passes ``residout``."""
    X, frame = _ln_cloud()
    rng = np.random.default_rng(10)
    c = rng.normal(size=X.shape[1])
    c -= c.mean()
    w = rng.uniform(0.2, 2.0, size=len(X))
    X2 = X + 0.3 * w[:, None] * c + 0.01 * rng.normal(size=X.shape)
    tgt = np.arange(1, len(X))
    rows = {r["source"]: r for r in u2.shared_cells(frame(X), frame(X2), X, X2, frame, tgt, u2.make_perms("w"))}
    assert abs(rows["causal:r1out"]["X"]) < 0.2 * abs(rows["mean0:frozen"]["X"])
