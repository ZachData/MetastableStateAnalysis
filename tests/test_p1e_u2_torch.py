"""
`p1e_energy_field/u2_torch.py` against `u2_block`'s numpy reference: torch on the CPU in float64
must give the same record; CUDA (when present) within ``AGREE_TOL``.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.deps
torch = pytest.importorskip("torch")

from p1e_energy_field import u2_block as u2  # noqa: E402
from p1e_energy_field.u2_torch import Ops  # noqa: E402


def _run(tmp_path, n=120, d=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(25, n, d)) * 5 + rng.normal(size=d) * 3
    norms = np.linalg.norm(X, axis=2)
    np.savez(tmp_path / "activations.npz", activations=(X / norms[..., None]).astype(np.float32),
             norms=norms.astype(np.float32))
    ln1 = {"w": 1 + 0.1 * rng.normal(size=(24, d)), "b": 0.05 * rng.normal(size=(24, d)), "eps": 1e-5}
    return ln1, {"t12": np.arange(1, n), "t123": np.arange(3, n, 2)}


def _same(a, b, tol):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        for k, v in x.items():
            if isinstance(v, float) and np.isfinite(v):
                assert abs(v - y[k]) <= tol, (k, v, y[k], x["block"], x["source"])
            else:
                assert v == y[k] or (v != v and y[k] != y[k])


@pytest.mark.parametrize("mode", ["frozen", "shared"])
def test_torch_cpu_float64_matches_numpy(tmp_path, mode):
    ln1, tg = _run(tmp_path)
    ref = u2.read_run(tmp_path, ln1, tg, "k", blocks=(0, 5), mode=mode)
    got = u2.read_run(tmp_path, ln1, tg, "k", blocks=(0, 5), mode=mode,
                      ops=Ops("cpu", torch.float64))
    _same(got, ref, 1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("mode", ["frozen", "shared"])
def test_cuda_float64_matches_numpy_within_tolerance(tmp_path, mode):
    ln1, tg = _run(tmp_path)
    ref = u2.read_run(tmp_path, ln1, tg, "k", blocks=(0, 5), mode=mode)
    got = u2.read_run(tmp_path, ln1, tg, "k", blocks=(0, 5), mode=mode, ops=u2.get_ops("cuda"))
    _same(got, ref, u2.AGREE_TOL)
