"""
tests/test_core_dissipation.py — tests for core/dissipation.py.

Self-contained, in the style of tests/test_core_metrics.py: fixtures are
built locally so the file exercises core.dissipation in isolation and runs
wherever numpy/scipy are available.

The governing rule here is status-2d.md's finding 5: *an anchor that only
tests the identity case tests almost nothing about a bilinear form.* So
every additivity anchor has a NON-SYMMETRIC arm — unequal channel deltas,
a genuinely non-normal operator for the subspace split, and clouds that are
not rigid transforms of one another for the transport functions.

Structure:
  1. Tangential geometry — linearity, which is what makes every split exact
  2. The gradient — against finite differences and a closed form
  3. Dissipation — the identity, its residual, and the exact splits
  4. Alignment — gradient descent reads +1, ascent reads -1
  5. Transport — permutation absorption, bounds, arc length, straightness
  6. Degenerate cases — vacuity returns None, not a manufactured 0.0
"""

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core.metrics import interaction_energy, l2_normalize
from core.dissipation import (
    tangential,
    tangential_velocity,
    energy_gradient,
    dissipation,
    dissipation_by_channel,
    dissipation_by_subspace,
    gradient_flow_alignment,
    w2_identity,
    w2_optimal,
    sliced_w2,
    wasserstein_arc_length,
    BETA_OVERFLOW_THRESHOLD,
)

BETA = 2.0


@pytest.fixture
def rng():
    return np.random.default_rng(20260822)


@pytest.fixture
def cloud(rng):
    """A generic, deliberately anisotropic cloud — not a symmetric or
    otherwise special configuration."""
    X = rng.standard_normal((24, 8))
    X[:, 0] *= 3.0            # anisotropy
    X[:8] += np.array([2.0, 0, 0, 0, 0, 0, 0, 0])   # a lump, so it is not uniform
    return l2_normalize(X)


# ---------------------------------------------------------------------------
# 1. Tangential geometry
# ---------------------------------------------------------------------------

def test_tangential_removes_the_radial_component(cloud):
    W = np.random.default_rng(0).standard_normal(cloud.shape)
    T = tangential(cloud, W)
    assert np.allclose(np.sum(T * cloud, axis=-1), 0.0, atol=1e-12)


def test_tangential_is_linear(cloud, rng):
    """The foundation of every exact split in this module. Tested on two
    DIFFERENT random fields, not on a field and its own multiple."""
    A = rng.standard_normal(cloud.shape)
    B = rng.standard_normal(cloud.shape) * 0.3
    assert np.allclose(tangential(cloud, A + B),
                       tangential(cloud, A) + tangential(cloud, B), atol=1e-14)


def test_step_size_matches_phase1c_convention(cloud, rng):
    """||v_i|| must equal Phase 1c's h_l = ||P_perp(dx)|| / ||x||, so this
    module is the rigorous form of that heuristic rather than a rival."""
    X = cloud * rng.uniform(0.5, 4.0, size=(cloud.shape[0], 1))   # non-unit norms
    dX = rng.standard_normal(X.shape) * 0.05
    v = tangential_velocity(X, dX)

    X_hat = l2_normalize(X)
    norms = np.linalg.norm(X, axis=-1)
    expected = np.linalg.norm(tangential(X_hat, dX), axis=-1) / norms
    assert np.allclose(np.linalg.norm(v, axis=-1), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. The gradient
# ---------------------------------------------------------------------------

def test_gradient_matches_finite_difference(cloud, rng):
    """The load-bearing test: the closed form
    dE/dx_i = (1/n^2) sum_j exp(beta <x_i,x_j>) x_j, differentiated against
    core.metrics.interaction_energy itself so the two conventions cannot
    drift."""
    g = energy_gradient(cloud, BETA)["grad"]
    delta = tangential(cloud, rng.standard_normal(cloud.shape))
    delta /= np.linalg.norm(delta)

    h = 1e-6
    e_plus = interaction_energy(cloud + h * delta, BETA)
    e_minus = interaction_energy(cloud - h * delta, BETA)
    numeric = (e_plus - e_minus) / (2 * h)
    analytic = float(np.sum(g * delta))

    assert numeric == pytest.approx(analytic, rel=1e-5, abs=1e-12)


def test_gradient_of_two_antipodal_particles(rng):
    """Closed form on a configuration with a hand-checkable answer.
    x_2 = -x_1, so dE/dx_1 = (1/4)[e^{beta} x_1 + e^{-beta} x_2]
                            = (1/4)(e^{beta} - e^{-beta}) x_1,
    which is purely RADIAL — so the tangential gradient is exactly zero and
    an antipodal pair is a critical point of E_beta on the sphere."""
    u = rng.standard_normal(6)
    u /= np.linalg.norm(u)
    X = np.stack([u, -u])

    raw = energy_gradient(X, BETA, tangential_only=False)["grad"]
    expected = 0.25 * (np.exp(BETA) - np.exp(-BETA)) * u
    assert np.allclose(raw[0], expected, rtol=1e-12)
    assert np.allclose(raw[1], -expected, rtol=1e-12)

    tan = energy_gradient(X, BETA)["grad"]
    assert np.allclose(tan, 0.0, atol=1e-12)


def test_gradient_is_tangential_by_default(cloud):
    g = energy_gradient(cloud, BETA)["grad"]
    assert np.allclose(np.sum(g * cloud, axis=-1), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Dissipation
# ---------------------------------------------------------------------------

def test_first_order_converges_to_actual_quadratically(cloud, rng):
    """The linearisation residual must be O(||dx||^2). Halving the step
    should quarter the residual — this is what licenses reading the
    first-order term as the energy change at all."""
    direction = rng.standard_normal(cloud.shape)
    residuals = []
    for scale in (1e-2, 5e-3, 2.5e-3):
        r = dissipation(cloud, scale * direction, BETA)
        residuals.append(abs(r["residual"]))

    assert residuals[1] / residuals[0] == pytest.approx(0.25, rel=0.15)
    assert residuals[2] / residuals[1] == pytest.approx(0.25, rel=0.15)


def test_per_particle_sums_to_first_order(cloud, rng):
    r = dissipation(cloud, rng.standard_normal(cloud.shape) * 0.01, BETA)
    assert float(r["per_particle"].sum()) == pytest.approx(r["first_order"], rel=1e-12)


def test_rigid_rotation_leaves_energy_exactly_invariant(cloud, rng):
    """E_beta depends only on the Gram matrix, so an orthogonal map of the
    whole cloud changes it by exactly zero — and the first-order term must
    agree. This is the same orthogonal-invariance fact that sank Phase 2b's
    Block 1b, used here deliberately as a control."""
    from scipy.linalg import expm

    d = cloud.shape[1]
    M = rng.standard_normal((d, d))
    # expm of an antisymmetric matrix is EXACTLY orthogonal. `I + eps*A` is
    # only orthogonal to first order, which shows up as a spurious energy
    # change at exactly the tolerance this test needs. Small angle, so the
    # first-order term is meaningful rather than swamped by the second.
    A = M - M.T

    firsts = []
    for theta in (1e-4, 5e-5):
        Q = expm(theta * A)
        assert np.allclose(Q @ Q.T, np.eye(d), atol=1e-14)
        r = dissipation(cloud, cloud @ Q - cloud, BETA)

        # The exact statement: the energy does not move at all.
        assert r["actual_delta_E"] == pytest.approx(0.0, abs=1e-12)
        firsts.append(abs(r["first_order"]))

    # The first-order term is therefore purely the O(theta^2) remainder,
    # exactly cancelled by the residual. Asserting it is "small" would only
    # be asserting that theta is small; asserting it scales as theta^2 is
    # the actual claim, and it is what fails if the tangential projection
    # or the gradient is wrong.
    assert firsts[1] / firsts[0] == pytest.approx(0.25, rel=0.1)


def test_channel_split_is_exact_on_unequal_deltas(cloud, rng):
    """The non-symmetric arm: attn and ffn deltas differ in scale, in
    direction, and in rank. If the split only worked for equal halves this
    would catch it."""
    d = cloud.shape[1]
    dX_attn = rng.standard_normal(cloud.shape) * 0.02
    low_rank = rng.standard_normal((cloud.shape[0], 2)) @ rng.standard_normal((2, d))
    dX_ffn = low_rank * 0.003

    assert not np.allclose(dX_attn, dX_ffn)

    r = dissipation_by_channel(cloud, dX_attn, dX_ffn, BETA)
    assert r["exact"], f"sum_check {r['sum_check']}"
    assert r["attn"] + r["ffn"] == pytest.approx(r["total"], rel=1e-12)

    joint = dissipation(cloud, dX_attn + dX_ffn, BETA)
    assert r["total"] == pytest.approx(joint["first_order"], rel=1e-12)


def test_channel_shares_use_magnitude_so_cancellation_is_visible(cloud, rng):
    """Two channels pushing in opposite directions must not read as a small
    evenly-split total — status-2d.md finding 6, the same cancellation trap
    that made PR_M read absence as coupling."""
    dX = rng.standard_normal(cloud.shape) * 0.02
    r = dissipation_by_channel(cloud, dX, -dX, BETA)
    assert r["total"] == pytest.approx(0.0, abs=1e-15)
    assert r["attn_share"] == pytest.approx(0.5, abs=1e-9)
    assert abs(r["attn"]) > 1e-9      # each channel is individually large


def test_subspace_split_is_exact_for_a_non_normal_operator(cloud, rng):
    """Schur projectors from a matrix that is genuinely non-normal and has
    eigenvalues of both signs — the identity case would prove nothing."""
    from scipy.linalg import schur

    d = cloud.shape[1]
    V = rng.standard_normal((d, d))
    V += np.triu(rng.standard_normal((d, d)), 1) * 2.0   # push it off-normal
    assert np.linalg.norm(V @ V.T - V.T @ V) > 1.0       # confirm non-normal

    T, Z, sdim = schur(V, output="real", sort="rhp")
    assert 0 < sdim < d, "need both signs present for this to be a real test"
    P_a = Z[:, :sdim] @ Z[:, :sdim].T
    P_r = Z[:, sdim:] @ Z[:, sdim:].T
    assert np.allclose(P_a + P_r, np.eye(d), atol=1e-10)

    dX = rng.standard_normal(cloud.shape) * 0.01
    r = dissipation_by_subspace(cloud, dX, BETA, P_a, P_r)
    assert r["sum_check"] < 1e-10
    assert r["attractive"] + r["repulsive"] == pytest.approx(r["total"], rel=1e-10)


def test_signs_are_not_clipped(cloud, rng):
    """status-2b.md's V2 lesson: a positive dissipation (uphill motion) is
    information and must survive to the caller."""
    g = energy_gradient(cloud, BETA)["grad"]
    uphill = dissipation(cloud, g * 0.01, BETA)     # move ALONG the gradient
    assert uphill["first_order"] > 0


# ---------------------------------------------------------------------------
# 4. Alignment
# ---------------------------------------------------------------------------

def test_gradient_descent_reads_plus_one_and_dissipates(cloud):
    """An explicit descent step on E_beta is by construction a Wasserstein
    gradient flow, so alignment must be +1 and the energy must fall."""
    g = energy_gradient(cloud, BETA)["grad"]
    step = -g * 1e-3

    a = gradient_flow_alignment(cloud, step, BETA)
    assert a["mean"] == pytest.approx(1.0, abs=1e-9)
    assert a["frac_descending"] == 1.0

    r = dissipation(cloud, step, BETA)
    assert r["first_order"] < 0
    assert r["actual_delta_E"] < 0


def test_gradient_ascent_reads_minus_one(cloud):
    g = energy_gradient(cloud, BETA)["grad"]
    a = gradient_flow_alignment(cloud, g * 1e-3, BETA)
    assert a["mean"] == pytest.approx(-1.0, abs=1e-9)
    assert a["frac_descending"] == 0.0


def test_alignment_reports_a_distribution_not_only_a_mean(cloud, rng):
    """A mean near zero is produced both by uniformly orthogonal motion and
    by half descending / half ascending. The second case must be visible."""
    g = energy_gradient(cloud, BETA)["grad"]
    signs = np.where(np.arange(cloud.shape[0]) % 2 == 0, 1.0, -1.0)[:, None]
    a = gradient_flow_alignment(cloud, -g * signs * 1e-3, BETA)

    assert a["mean"] == pytest.approx(0.0, abs=1e-6)
    assert a["frac_descending"] == pytest.approx(0.5, abs=1e-9)
    assert a["q10"] == pytest.approx(-1.0, abs=1e-6)
    assert a["q90"] == pytest.approx(1.0, abs=1e-6)
    assert a["std"] > 0.9          # the mean alone would have hidden this


# ---------------------------------------------------------------------------
# 5. Transport
# ---------------------------------------------------------------------------

def test_permutation_is_fully_absorbed_by_the_optimal_coupling(cloud, rng):
    """The observable this module adds: relabelling tokens leaves the
    measure untouched, so true W_2 is zero while the identity coupling
    reports a large distance."""
    perm = rng.permutation(cloud.shape[0])
    assert not np.all(perm == np.arange(cloud.shape[0]))
    Y = cloud[perm]

    r = w2_optimal(cloud, Y)
    assert r["w2"] == pytest.approx(0.0, abs=1e-12)
    assert r["w2_identity"] > 0.1
    assert r["swap_absorbed_fraction"] == pytest.approx(1.0, abs=1e-8)


def test_optimal_never_exceeds_identity_on_generic_clouds(rng):
    """Not a permutation, not a rotation — two independently drawn clouds,
    so the bound is doing real work."""
    for _ in range(5):
        A = l2_normalize(rng.standard_normal((30, 6)))
        B = l2_normalize(rng.standard_normal((30, 6)))
        r = w2_optimal(A, B)
        assert r["w2"] <= r["w2_identity"] + 1e-12
        assert r["w2"] >= 0.0
        assert w2_identity(A, B) == pytest.approx(r["w2_identity"], rel=1e-12)


def test_w2_of_a_measure_with_itself_is_zero(cloud):
    r = w2_optimal(cloud, cloud)
    assert r["w2"] == pytest.approx(0.0, abs=1e-12)
    assert r["swap_absorbed_fraction"] is None or r["swap_absorbed_fraction"] == 0.0


def test_sliced_w2_tracks_true_w2(rng):
    """Sliced W_2 is a different quantity, not an estimator of W_2, so this
    checks the weaker property that actually matters: it ORDERS a near pair
    below a far pair."""
    A = l2_normalize(rng.standard_normal((40, 5)))
    near = l2_normalize(A + rng.standard_normal(A.shape) * 0.05)
    far = l2_normalize(rng.standard_normal((40, 5)))

    g = np.random.default_rng(7)
    s_near = sliced_w2(A, near, n_proj=256, rng=g)["sliced_w2"]
    s_far = sliced_w2(A, far, n_proj=256, rng=g)["sliced_w2"]
    assert s_near < s_far


def test_arc_length_and_straightness(rng):
    """A monotone drift is straight; an out-and-back path has the same arc
    length with near-zero net displacement, which is what dwelling looks
    like on the measure."""
    n, d = 20, 4
    base = l2_normalize(rng.standard_normal((n, d)))
    push = rng.standard_normal((n, d)) * 0.05

    straight = np.stack([l2_normalize(base + k * push) for k in range(6)])
    ks = [0, 1, 2, 3, 2, 1, 0]
    out_and_back = np.stack([l2_normalize(base + k * push) for k in ks])

    a = wasserstein_arc_length(straight)
    b = wasserstein_arc_length(out_and_back)

    assert a["straightness"] == pytest.approx(1.0, abs=0.05)
    assert b["straightness"] == pytest.approx(0.0, abs=1e-6)
    assert b["arc_length_optimal"] > 0.05
    assert a["arc_length_identity"] >= a["arc_length_optimal"] - 1e-12
    assert len(a["per_step"]) == 5


def test_arc_length_refuses_a_one_layer_trajectory(rng):
    one = l2_normalize(rng.standard_normal((1, 10, 4)))
    r = wasserstein_arc_length(one)
    assert r["arc_length_identity"] is None
    assert r["status"] == "trajectory_too_short_for_an_arc_length"


# ---------------------------------------------------------------------------
# 6. Degenerate cases — None, never a manufactured number
# ---------------------------------------------------------------------------

def test_zero_velocity_gives_no_alignment_rather_than_zero(cloud):
    """status-2b.md known-issue 5: a float that reads as a measurement is
    worse than an explicit refusal."""
    a = gradient_flow_alignment(cloud, np.zeros_like(cloud), BETA)
    assert a["mean"] is None
    assert a["n_defined"] == 0
    assert a["status"] == "no_particle_has_a_defined_alignment"


def test_partially_undefined_alignment_is_counted_not_imputed(cloud):
    g = energy_gradient(cloud, BETA)["grad"]
    step = -g * 1e-3
    step[:5] = 0.0
    a = gradient_flow_alignment(cloud, step, BETA)

    assert a["n_undefined"] == 5
    assert a["n_defined"] == cloud.shape[0] - 5
    assert a["mean"] == pytest.approx(1.0, abs=1e-9)   # unaffected by the zeros
    assert np.isnan(a["per_particle"][:5]).all()


def test_overflow_branch_refuses_absolute_energy_but_keeps_attribution(cloud, rng):
    beta = BETA_OVERFLOW_THRESHOLD * 2
    r = dissipation(cloud, rng.standard_normal(cloud.shape) * 1e-3, beta)

    assert r["overflow_guarded"] is True
    assert r["actual_delta_E"] is None
    assert r["residual"] is None
    assert r["status"] == "energy_unrepresentable_at_this_beta"
    assert np.isfinite(r["first_order"])       # attribution survives the shift


def test_shape_mismatch_raises_rather_than_broadcasting(cloud):
    with pytest.raises(ValueError):
        dissipation(cloud, cloud[:, :3], BETA)
    with pytest.raises(ValueError):
        w2_identity(cloud, cloud[:5])
