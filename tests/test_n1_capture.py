"""trackN_neurons/n1_capture.py's statistics against brute force (CPU, synthetic)."""
import pytest

pytestmark = pytest.mark.deps  # torch

torch = pytest.importorskip("torch")

from trackN_neurons import n1_capture as n1  # noqa: E402


def _brute_band(v, norm, bias, k):
    h = n1.gelu(norm * v + bias)
    idx = torch.topk(h, k, largest=False).indices
    return set(idx.tolist()), (norm * v[idx] + bias - n1.Z_STAR).abs().mean() / norm


def test_band_matches_brute_force_for_scaled_rows():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(4000, 16, generator=g, dtype=torch.float64)
    D = torch.randn(5, 16, generator=g, dtype=torch.float64)
    sv, _, s = n1.null_sorted(X, D, chunk=2)
    _, si = torch.sort((X @ D.T).T, dim=1)
    norm = torch.tensor([0.3, 1.0, 2.5], dtype=torch.float64)
    bias = torch.tensor([0.2, -0.5, 1.0], dtype=torch.float64)
    vs = ((n1.Z_STAR - bias) / norm)[:, None].expand(-1, 5)
    cols, d = n1.band(sv, vs, norm[:, None].expand(-1, 5), 20)
    for i in range(3):
        for j in range(5):
            v = (X @ D[j])
            want, dmean = _brute_band(v, norm[i], bias[i], 20)
            assert set(si[j, cols[i, j]].tolist()) == want
            assert torch.isclose(d[i, j].mean(), dmean)


def test_pm_stats_affine_invariant_and_b_stat_matches_column_stats():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(3000, 8, generator=g, dtype=torch.float64)
    w = torch.randn(8, generator=g, dtype=torch.float64)
    a = n1.column_stats((X @ w)[:, None])["stat"]
    b = n1.column_stats((3.7 * (X @ w) - 0.4)[:, None])["stat"]
    assert torch.allclose(a[:2], b[:2])
    # B through null_b's v-space equals B on the row's own z
    sv, _, s = n1.null_sorted(X, (w / w.norm())[None], chunk=1)
    norm, bias = 3.7 * w.norm(), torch.tensor(-0.4, dtype=torch.float64)
    vs = ((n1.Z_STAR - bias) / norm).reshape(1, 1)
    _, d = n1.band(sv, vs, norm.reshape(1, 1), n1.KMAX)
    for kk, k in enumerate(n1.KS):
        assert torch.isclose(-d[0, 0, :k].mean() / s[0], b[2, kk, 0])


def test_column_stats_rows_are_the_readings():
    z = torch.linspace(-4, 4, 801, dtype=torch.float64)[torch.randperm(801)]
    cs = n1.column_stats(z[:, None])
    top, bot, band = (z[cs["rows"][r, :, 0]] for r in range(3))
    assert torch.equal(top, torch.sort(z, descending=True).values[:n1.KMAX])
    assert torch.equal(bot, torch.sort(z).values[:n1.KMAX])
    assert set(band.tolist()) == set(z[torch.topk(n1.gelu(z), n1.KMAX, largest=False).indices].tolist())
