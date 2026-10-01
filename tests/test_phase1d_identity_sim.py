"""
tests/test_phase1d_identity_sim.py — the identity-weights simulator
(`p1d_cluster_ensemble/identity_sim.py`, `design-1d.md` "Identity-weights
positive control"): the full mask against `gamma_ode`'s (6.9), the causal pair
at half speed (`tools/math_checks/identity_sim_closed_form.py`), Thm 4.1 of
2411.04990, the span reduction, the float floor, and the three subcommands
end to end on a fake run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.deps

from p1c_frames.gamma_ode import integrate_gamma
from p1d_cluster_ensemble import identity_sim as sim
from p1d_cluster_ensemble.gaussian_null import span_coordinates
from p1d_cluster_ensemble.methods import LayerData


def _gamma(n, beta, times, model="sa"):
    t, g = integrate_gamma(n, beta, t_max=max(times) + 0.01, dt=1e-3, model=model)
    return np.interp(times, t, g)


def _offdiag(G):
    return G[~np.eye(G.shape[0], dtype=bool)]


@pytest.mark.parametrize("n", [2, 5, 20])
@pytest.mark.parametrize("beta", [0.0, 1.0, 5.0])
def test_full_mask_orthogonal_start_follows_eq_6_9(n, beta):
    times = (0.0, 0.5, 1.0, 2.0, 4.0)
    S = sim.integrate(np.eye(n), beta, "full", times, dt=1 / 256)
    want = _gamma(n, beta, times)
    for X, g in zip(S, want):
        ips = _offdiag(X @ X.T)
        assert np.ptp(ips) < 1e-9          # all pairs stay equal
        assert abs(ips.mean() - g) < 1e-6


@pytest.mark.parametrize("beta", [0.0, 1.0, 5.0])
def test_causal_pair_is_eq_6_9_at_half_speed(beta):
    times = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    S = sim.integrate(np.eye(2), beta, "causal", times, dt=1 / 256)
    want = _gamma(2, beta, [t / 2 for t in times])
    for X, g in zip(S, want):
        assert abs(X[0] @ X[1] - g) < 1e-6
        assert np.allclose(X[0], [1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("beta", [0.0, 1.0, 4.0])
def test_thm_4_1_first_token_fixed_and_everything_goes_to_it(beta):
    rng = np.random.default_rng(3)
    X0 = sim._unit(rng.standard_normal((6, 6)))
    S = sim.integrate(X0, beta, "causal", (0.0, 2.0, 400.0), dt=1 / 16)
    for X in S:
        assert np.allclose(X[0], X0[0], atol=1e-12)
    assert np.min(S[-1] @ X0[0]) > 0.999
    assert np.min(S[1] @ X0[0]) < np.min(S[-1] @ X0[0])   # not there yet at t = 2


def test_full_mask_is_symmetric_and_causal_is_not():
    rng = np.random.default_rng(0)
    X = sim._unit(rng.standard_normal((5, 5)))
    A = sim.attention(X, 2.0, "causal")
    assert np.allclose(np.triu(A, 1), 0.0) and np.allclose(A.sum(1), 1.0)
    assert np.allclose(A[0], [1, 0, 0, 0, 0])
    assert np.allclose(sim.velocity(X, 2.0, "causal")[0], 0.0)
    assert np.all(sim.attention(X, 2.0, "full") > 0)


def test_span_reduction_is_exact():
    rng = np.random.default_rng(1)
    X0 = sim._unit(rng.standard_normal((8, 30)) + 1.5)
    times = (0.0, 1.0, 4.0)
    for mask in sim.MASKS:
        A = sim.integrate(X0, 3.0, mask, times, dt=1 / 64)
        B = sim.integrate(span_coordinates(X0), 3.0, mask, times, dt=1 / 64)
        assert B.shape[2] == 8
        for a, b in zip(A, B):
            assert np.max(np.abs(a @ a.T - b @ b.T)) < 1e-10


def test_integrate_converged_halves_until_the_gram_settles():
    rng = np.random.default_rng(2)
    X0 = sim._unit(rng.standard_normal((10, 10)))
    S, info = sim.integrate_converged(X0, 16.0, "causal", (0.0, 1.0, 2.0))
    assert info["converged"] and info["gram_change"] <= sim.GRAM_TOL and info["halvings"] >= 1
    ref = sim.integrate(X0, 16.0, "causal", (0.0, 1.0, 2.0), dt=info["dt"] / 4)
    assert max(np.max(np.abs(a @ a.T - b @ b.T)) for a, b in zip(S, ref)) < 10 * sim.GRAM_TOL
    with pytest.raises(ValueError, match="multiples"):
        sim.integrate(X0, 1.0, "causal", (0.3,), dt=0.25)


def test_initial_dt_is_a_power_of_two_dividing_the_grid():
    for b in sim.BETAS:
        dt = sim.initial_dt(b)
        assert dt <= 0.05 and (b == 0 or dt <= 0.25 / b)
        assert all(abs(t / dt - round(t / dt)) < 1e-12 for t in sim.TIMES)


def test_float_floor_is_above_admissions_distance_resolution():
    # admission's route reads float32 rows (`LayerData.from_normed`); at the
    # floor a pair's 1 - cos must still be resolved, not a tie at 0.
    rng = np.random.default_rng(0)
    for m in (30, 273):
        x = sim._unit(rng.standard_normal((1, m)))[0]
        v = rng.standard_normal(m)
        v -= (v @ x) * x
        v /= np.linalg.norm(v)
        th = np.arccos(1 - sim.FLOAT_FLOOR)
        X = np.stack([x, np.cos(th) * x + np.sin(th) * v])
        d = LayerData.from_normed(X).cos_dist[0, 1]
        assert abs(d - sim.FLOAT_FLOOR) / sim.FLOAT_FLOOR < 1e-2


def test_theory_clusters_and_recovery():
    X = np.array([[1, 0, 0], [1, 1e-3, 0], [1, 0, 1e-3],   # tight triple
                  [0, 1, 0], [0, 1, 1e-3],                  # tight pair
                  [0, 0, 1]], dtype=float)
    lab = sim.theory_clusters(X, 1e-3)
    assert lab.tolist() == [0, 0, 0, 1, 1, -1]
    assert (sim.theory_clusters(X, 1e-9) == -1).all()
    r = sim.recovery(lab, [[0, 1, 2], [5]], min_size=2)
    assert r["n_theory"] == 2 and r["best_jaccard"] == [1.0, 0.0]
    assert r["recall"] == 0.5 and r["precision"] == 0.5
    r = sim.recovery(lab, [[0, 1, 2], [3, 4]], min_size=2)
    assert r["recall"] == 1.0 and r["ari"] == pytest.approx(1.0)
    assert sim.recovery(lab, [], min_size=4)["recall"] is None


def _fake_run(tmp_path, name, n=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    run = tmp_path / name
    run.mkdir(parents=True)
    acts = sim._unit(rng.standard_normal((n, d)) + 0.5)[None].repeat(3, 0)
    tokens = [f"t{i}" for i in range(n - 2)] + ["t0", "t1"]   # two later repeats
    np.savez(run / "activations.npz", activations=acts.astype(np.float32))
    (run / "geometry.json").write_text(json.dumps({"tokens": tokens}))
    return run


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "TIMES", (0.0, 0.5, 1.0))
    run = _fake_run(tmp_path, "pythia-410m-step0_wiki_paragraph")
    out = tmp_path / "out"
    assert sim.main(["simulate", "--runs", str(run), "--out", str(out), "--betas", "0", "16",
                     "--workers", "1"]) == 0
    s = json.loads((out / "simulate.json").read_text())
    assert len(s["parts"]) == 4 and not s["unconverged"]
    part = json.loads(open(s["parts"][0]).read())
    assert part["n_kept"] == 38 and len(part["snapshots"]) == 3
    assert part["snapshots"][0]["cos_to_first"][0] == pytest.approx(1.0)
    for extra in ([], ["--calibrate"]):
        assert sim.main(["admit", "--out", str(out), "--n-draws", "5", "--workers", "1",
                         *extra]) == 0
    a = json.loads((out / "admit_real.json").read_text())
    # the start is admitted once per input, every later snapshot once per trajectory
    assert len(a["parts"]) == 2 * (1 + 4 * 2)
    assert sim.main(["report", "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert {r["mask"] for r in rep["rows"]} == {"causal", "full"}
    assert all("recall" in r or "skipped" in r for r in rep["rows"])
    # resumable: a second simulate reuses every part
    assert sim.main(["simulate", "--runs", str(run), "--out", str(out), "--betas", "0", "16",
                     "--workers", "1"]) == 0


def test_admit_job_refuses_below_the_float_floor(tmp_path):
    rng = np.random.default_rng(0)
    X = sim._unit(rng.standard_normal((24, 5)))
    X[1] = sim._unit((X[0] + 1e-6 * rng.standard_normal(5))[None])[0]   # 1 - cos ~ 1e-12
    traj = tmp_path / "t.npz"
    np.savez(traj, snaps=X[None], times=np.array([0.0]), keep=np.arange(24))
    rec = sim._admit_job((str(traj), 0, "raw", 3, 0, False, "p"))
    assert "float floor" in rec["skipped"]


def test_admit_job_refuses_when_a_null_draw_collapses(tmp_path):
    # above the floor itself, but so tight that its own Gaussian draws are not
    rng = np.random.default_rng(1)
    c = sim._unit(rng.standard_normal((1, 6)))[0]
    X = sim._unit(c + 1e-4 * rng.standard_normal((30, 6)))
    assert sim.min_distance(X) > sim.FLOAT_FLOOR
    traj = tmp_path / "t.npz"
    np.savez(traj, snaps=X[None], times=np.array([0.0]), keep=np.arange(30))
    rec = sim._admit_job((str(traj), 0, "raw", 50, 0, False, "p"))
    assert "null draw" in rec["skipped"]
