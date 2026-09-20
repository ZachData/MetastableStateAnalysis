"""
`tools/run/transport.py` — F1, the transport observables.

The two numbers this row exists to produce are easy to get subtly wrong, and
each has a construction with a known answer:

  swap_absorbed_fraction  is 1 when a step is nothing but tokens changing
                          places, and 0 when the identity coupling is already
                          optimal. Both are built here.
  straightness            is 1 on a straight path and near 0 on one that
                          returns to where it started.

There is also a trap worth a test of its own: at the embedding layer, repeated
tokens have IDENTICAL vectors, so swapping their assignments is an exact tie in
cost. `swap_fraction` reads ~0.5 while the optimal coupling absorbs nothing at
all. A reader who took `swap_fraction` for motion of the measure would be
reading duplicate tokens.
"""
import numpy as np
import pytest

from tools.run.transport import (
    N_PERMUTATIONS,
    aggregate,
    measure_boundary,
    measure_directory,
    per_particle_step,
)


def _rng():
    return np.random.default_rng(0)


def _unit(X):
    return np.asarray(X, dtype=np.float64) / np.linalg.norm(X, axis=1, keepdims=True)


# --- the per-particle step -------------------------------------------------

def test_no_motion_is_no_step():
    X = _unit(np.random.default_rng(1).normal(size=(10, 5)))
    assert np.allclose(per_particle_step(X, X), 0.0)


def test_the_step_is_per_particle_not_aggregated():
    X = _unit(np.eye(4)[[0, 1, 2, 3]])
    Y = X.copy()
    Y[2] = _unit(np.array([[1.0, 0.0, 1.0, 0.0]]))[0]
    step = per_particle_step(X, Y)
    assert step.shape == (4,)
    assert step[2] > 0.1
    assert np.allclose(step[[0, 1, 3]], 0.0)


def test_radial_motion_does_not_count_as_a_step():
    """`tangential_velocity` projects out the radial component, which is the
    project's existing convention -- the sphere is the state space."""
    X = _unit(np.random.default_rng(2).normal(size=(8, 6)))
    assert np.allclose(per_particle_step(X, 3.0 * X), 0.0, atol=1e-9)


# --- the coupling gap, the number the row exists for ----------------------

def test_a_pure_permutation_is_absorbed_entirely():
    """Tokens changing places leaves the measure untouched, so the optimal
    coupling sees no distance at all and the identity coupling sees a lot."""
    X = _unit(np.random.default_rng(3).normal(size=(12, 7)))
    Y = X[np.random.default_rng(4).permutation(12)]
    rec = measure_boundary(X, Y, np.full(12, -1), _rng())
    assert rec["w2_optimal"] == pytest.approx(0.0, abs=1e-9)
    assert rec["w2_identity"] > 0.5
    assert rec["swap_absorbed_fraction"] == pytest.approx(1.0)


def test_a_common_translation_is_absorbed_not_at_all():
    """Every point moving the same way IS motion of the measure, and no
    permutation can help."""
    rng = np.random.default_rng(5)
    X = _unit(rng.normal(size=(30, 8)))
    Y = _unit(X + 0.3 * rng.normal(size=(1, 8)))
    rec = measure_boundary(X, Y, np.full(30, -1), _rng())
    assert rec["swap_absorbed_fraction"] == pytest.approx(0.0, abs=0.05)
    assert rec["w2_optimal"] == pytest.approx(rec["w2_identity"], rel=0.05)


def test_duplicate_sources_inflate_swap_fraction_without_absorbing_anything():
    """The embedding-layer trap, and it is a real one: the sweep's layer-0
    boundary reports `swap_fraction` 0.48 with a gain of EXACTLY zero.

    At layer 0 a Pythia hidden state is the token embedding alone, so repeated
    tokens have identical vectors. Their targets at layer 1 differ, but since
    the sources are equal the cost of sending either source to either target is
    equal too — the swaps are exact TIES. The assignment takes them freely,
    `swap_fraction` climbs, and the total cost does not move by a bit.

    Anyone reading `swap_fraction` as motion of the measure would be reading
    duplicate tokens."""
    rng = np.random.default_rng(6)
    base = _unit(rng.normal(size=(4, 8)))
    X = np.repeat(base, 5, axis=0)              # 20 points, 4 distinct
    # A SMALL perturbation, which is what one block of a transformer is. With a
    # random target the optimal matching genuinely beats identity and the tie
    # argument does not apply; the sweep's boundaries are perturbations.
    Y = _unit(X + 0.05 * rng.normal(size=X.shape))
    rec = measure_boundary(X, Y, np.full(20, -1), _rng())
    assert rec["swap_fraction"] > 0.3
    assert rec["swap_absorbed_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert rec["w2_optimal"] == pytest.approx(rec["w2_identity"], abs=1e-12)


def test_no_motion_at_all_leaves_the_absorbed_fraction_undefined():
    """None, not 0: there was no displacement to attribute, and calling that
    'the identity coupling was optimal' would be a claim about a step that
    never happened."""
    X = _unit(np.random.default_rng(60).normal(size=(12, 5)))
    rec = measure_boundary(X, X, np.full(12, -1), _rng())
    assert rec["w2_identity"] == pytest.approx(0.0, abs=1e-9)
    assert rec["swap_absorbed_fraction"] is None


def test_identity_only_omits_the_optimal_fields_rather_than_faking_them():
    X = _unit(np.random.default_rng(7).normal(size=(10, 5)))
    Y = _unit(np.random.default_rng(8).normal(size=(10, 5)))
    rec = measure_boundary(X, Y, np.full(10, -1), _rng(), optimal=False)
    assert "w2_identity" in rec
    assert "w2_optimal" not in rec
    assert "swap_absorbed_fraction" not in rec


def test_the_assignment_array_is_not_kept_in_the_record():
    """It is the largest thing `w2_optimal` produces and nothing reads it.
    152 of them in one JSON is the difference between a file that opens and
    one that does not."""
    X = _unit(np.random.default_rng(9).normal(size=(10, 5)))
    Y = _unit(np.random.default_rng(10).normal(size=(10, 5)))
    rec = measure_boundary(X, Y, np.full(10, -1), _rng())
    assert "assignment" not in rec


# --- the population split --------------------------------------------------

def test_a_population_difference_in_step_size_is_detected():
    """Power on the kinematic signature: move the noise tokens and leave the
    clustered ones still."""
    rng = np.random.default_rng(11)
    n = 120
    lab = np.full(n, -1)
    clustered = rng.choice(n, size=60, replace=False)
    lab[clustered] = rng.integers(0, 8, size=60)
    X = _unit(rng.normal(size=(n, 10)))
    Y = X.copy()
    moving = np.setdiff1d(np.arange(n), clustered)
    Y[moving] = _unit(X[moving] + 0.8 * rng.normal(size=(moving.size, 10)))
    rec = measure_boundary(X, Y, lab, rng)
    assert rec["clustered_minus_noise_step"] < -0.5      # clustered move less
    assert rec["clustered_minus_noise_step_p"] <= 1.0 / (N_PERMUTATIONS + 1) + 1e-12


def test_the_direction_is_two_sided():
    """H-PARK predicts clustered particles move less; nothing predicted the
    other sign, so neither may be chosen now and both must be reportable."""
    rng = np.random.default_rng(12)
    n = 120
    lab = np.full(n, -1)
    clustered = rng.choice(n, size=60, replace=False)
    lab[clustered] = rng.integers(0, 8, size=60)
    X = _unit(rng.normal(size=(n, 10)))
    Y = X.copy()
    Y[clustered] = _unit(X[clustered] + 0.8 * rng.normal(size=(60, 10)))
    rec = measure_boundary(X, Y, lab, rng)
    assert rec["clustered_minus_noise_step"] > 0.5
    assert rec["clustered_minus_noise_step_p"] <= 1.0 / (N_PERMUTATIONS + 1) + 1e-12


def test_a_boundary_with_no_partition_still_reports_transport(tmp_path):
    """The measure-level numbers do not need labels, so a directory without a
    partition must still produce them rather than being skipped."""
    X = _unit(np.random.default_rng(13).normal(size=(10, 5)))
    Y = _unit(np.random.default_rng(14).normal(size=(10, 5)))
    rec = measure_boundary(X, Y, np.full(10, -1), _rng())
    assert rec["w2_identity"] > 0
    assert rec["clustered_minus_noise_step"] is None


# --- the trajectory --------------------------------------------------------

def _write(tmp_path, traj, labels=None, name="pythia-410m-step9_wiki"):
    import json

    d = tmp_path / "2026-01-01_00-00-00" / name
    d.mkdir(parents=True)
    np.savez(d / "activations.npz", activations=np.asarray(traj, dtype=np.float32))
    if labels is not None:
        (d / "hdbscan_labels.json").write_text(json.dumps(labels))
    return d


def test_straightness_is_near_one_on_a_monotone_path(tmp_path):
    """Three points marching in one direction along a great circle.

    Near one and not exactly one, deliberately: `W_2` here uses the CHORDAL
    distance, and chords of an arc are shorter than the arc, so a path that is
    geodesically straight reads slightly bent. 0.975 over this arc. A test
    demanding exactly 1 would be demanding the wrong geometry."""
    t = np.linspace(0.0, 0.9, 3)
    traj = np.stack([_unit(np.stack([np.array([np.cos(a), np.sin(a), 0.0])
                                     for a in (x, x + 0.1, x + 0.2)])) for x in t])
    got = measure_directory(_write(tmp_path, traj), _rng())
    assert got["trajectory"]["straightness"] == pytest.approx(0.975, abs=0.01)


def test_straightness_is_near_zero_on_a_path_that_returns(tmp_path):
    a = _unit(np.random.default_rng(15).normal(size=(6, 4)))
    b = _unit(np.random.default_rng(16).normal(size=(6, 4)))
    got = measure_directory(_write(tmp_path, np.stack([a, b, a])), _rng())
    assert got["trajectory"]["straightness"] == pytest.approx(0.0, abs=1e-6)


def test_the_per_step_list_is_dropped_from_the_record(tmp_path):
    """It duplicates `boundaries` at 24 entries per directory across 152
    directories."""
    a = _unit(np.random.default_rng(17).normal(size=(6, 4)))
    got = measure_directory(_write(tmp_path, np.stack([a, a, a])), _rng())
    assert "per_step" not in got["trajectory"]
    assert len(got["boundaries"]) == 2


def test_a_directory_without_activations_is_skipped_with_a_reason(tmp_path):
    d = tmp_path / "2026-01-01_00-00-00" / "pythia-410m-step0_wiki"
    d.mkdir(parents=True)
    assert measure_directory(d, _rng())["skipped"] == "no activations.npz"


def test_labels_are_matched_to_the_lower_layer_of_each_boundary(tmp_path):
    """Boundary l runs from state l to state l+1, so the partition that
    describes the particles being moved is the one at layer l."""
    rng = np.random.default_rng(18)
    n = 40
    lab_a = np.full(n, -1)
    lab_a[:20] = 0
    lab_b = np.full(n, -1)
    lab_b[20:] = 0
    X = _unit(rng.normal(size=(n, 6)))
    Y = X.copy()
    Y[:20] = _unit(X[:20] + 1.0 * rng.normal(size=(20, 6)))
    d = _write(tmp_path, np.stack([X, Y, Y]),
               labels={"0": lab_a.tolist(), "1": lab_b.tolist()})
    got = measure_directory(d, rng)
    # Boundary 0 uses layer 0's labels, where the MOVING tokens are clustered.
    assert got["boundaries"][0]["clustered_minus_noise_step"] > 0.5


# --- the aggregate ---------------------------------------------------------

def _b(p=0.5, absorbed=0.0, diff=0.1):
    return {"layer": 0, "w2_identity": 0.4, "w2_optimal": 0.4,
            "swap_absorbed_fraction": absorbed, "swap_fraction": 0.02,
            "mean_step": 0.36, "clustered_minus_noise_step": diff,
            "clustered_minus_noise_step_p": p, "degenerate": False}


def _d(ckpt=0, n=2, **kw):
    return {"checkpoint": ckpt, "boundaries": [_b(**kw) for _ in range(n)],
            "trajectory": {"straightness": 0.15, "arc_length_identity": 9.0,
                           "arc_length_optimal": 9.0}}


def test_aggregate_of_nothing_says_so():
    assert aggregate([{"skipped": "x"}]) == {"n_boundaries": 0}


def test_aggregate_merges_with_the_mean_not_the_product():
    from core.evalues import calibrate

    got = aggregate([_d(n=25, p=0.19)])
    assert got["clustered_minus_noise_step"]["E"] == pytest.approx(
        calibrate(0.19), abs=1e-3)
    assert got["clustered_minus_noise_step"]["reject"] is False


def test_trajectory_statistics_are_averaged_over_directories_not_boundaries():
    """One straightness per directory. Averaging it per boundary would weight
    long prompts more, for no reason."""
    got = aggregate([_d(ckpt=0, n=2), _d(ckpt=1, n=20)])
    assert got["mean_straightness"] == pytest.approx(0.15)
    assert got["n_directories"] == 2
    assert got["n_boundaries"] == 22


def test_aggregate_splits_by_checkpoint():
    got = aggregate([_d(ckpt=0, p=0.9), _d(ckpt=143000, p=0.001)])
    assert list(got["by_checkpoint"]) == ["0", "143000"]
    assert got["by_checkpoint"]["143000"]["clustered_minus_noise_step"]["E"] > \
        got["by_checkpoint"]["0"]["clustered_minus_noise_step"]["E"]


def test_none_absorbed_fractions_are_dropped_not_scored_as_zero():
    """`swap_absorbed_fraction` is None when there was no displacement to
    attribute, and counting that as 0 would claim the identity coupling was
    optimal on a step that never happened."""
    d = _d(n=1)
    d["boundaries"].append({**_b(), "swap_absorbed_fraction": None})
    got = aggregate([d])
    assert got["mean_swap_absorbed_fraction"] == pytest.approx(0.0)
    assert got["n_boundaries"] == 2
