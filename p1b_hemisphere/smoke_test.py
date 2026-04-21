"""
Smoke test for phase1h/bipartition_detect.py and phase1h/hemisphere_tracking.py.

Constructs three synthetic regimes we know the answer for:
  1. Cone-collapsed prompt:  all tokens near one point.  Regime should
                             be "collapsed" at every layer.
  2. Antipodal bipartition:  two tight clusters at +v and -v.
                             Regime should be "strong_bipartition" with
                             centroid angle ≈ π.
  3. Rotating axis:          cluster A/B stable in membership, but the
                             Fiedler vector's sign happens to flip at
                             one layer.  Block 1 must correct via the
                             sign-flip Jaccard and NOT emit a swap.

Run:
    python -m phase1h.smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running this file directly from /home/claude/phase1h/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase1h.bipartition_detect import analyze_bipartition, bipartition_to_json
from phase1h.hemisphere_tracking import (
    analyze_hemisphere_tracking,
    hemisphere_tracking_to_json,
)


def _l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def make_collapsed(n_layers=6, n_tokens=40, d=16, rng=None):
    """Single outlier in a tight cluster.  The Fiedler partition should
    isolate the outlier, giving minority_fraction = 1/n < 0.05 → regime
    'collapsed'."""
    rng = rng or np.random.default_rng(0)
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    # Outlier is antipodal to the cluster so the clipped Gram gives it
    # zero affinity with the rest — the block-diagonal structure the
    # Fiedler vector picks up to isolate it.
    X = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        main = +v + 0.02 * rng.standard_normal((n_tokens - 1, d))
        out  = -v + 0.02 * rng.standard_normal((1, d))
        X[L] = np.vstack([main, out])
    return _l2(X)


def make_cone_cloud(n_layers=6, n_tokens=40, d=16, rng=None):
    """All tokens near one direction, no outlier.  Fiedler partition is
    noise-driven → roughly 50/50 membership but near-zero centroid angle
    → regime 'weak_bipartition'."""
    rng = rng or np.random.default_rng(42)
    direction = rng.standard_normal(d); direction /= np.linalg.norm(direction)
    X = direction[None, None, :] + 0.02 * rng.standard_normal((n_layers, n_tokens, d))
    return _l2(X)


def make_antipodal(n_layers=6, n_tokens=40, d=16, rng=None):
    rng = rng or np.random.default_rng(1)
    v = rng.standard_normal(d); v /= np.linalg.norm(v)
    half = n_tokens // 2
    acts = np.zeros((n_layers, n_tokens, d))
    for L in range(n_layers):
        a = +v + 0.05 * rng.standard_normal((half, d))
        b = -v + 0.05 * rng.standard_normal((n_tokens - half, d))
        acts[L] = np.vstack([a, b])
    return _l2(acts)


def make_sign_flip(n_layers=4, n_tokens=40, d=16, rng=None):
    """Same antipodal structure at every layer; Block 0 should produce
    regime='strong_bipartition' each layer and Block 1 should align out
    the arbitrary per-layer sign of the Fiedler vector.  Passing means
    zero swap events and match_overlap very close to 1 throughout."""
    return make_antipodal(n_layers, n_tokens, d, rng)


def test_collapsed():
    acts = make_collapsed()
    r = analyze_bipartition(acts)
    regimes = list(r["regime"])
    assert all(x == "collapsed" for x in regimes), regimes
    # Minority fraction is exactly 1/n_tokens on every valid layer (the
    # Fiedler vector isolates the single outlier).
    mf = r["minority_fraction"]
    finite_mf = mf[np.isfinite(mf)]
    assert finite_mf.size and (finite_mf < 0.05).all(), finite_mf
    print(f"[OK] collapsed (outlier): regime='collapsed', "
          f"minority max={finite_mf.max():.4f}")


def test_cone_cloud():
    acts = make_cone_cloud()
    r = analyze_bipartition(acts)
    regimes = list(r["regime"])
    assert all(x == "weak_bipartition" for x in regimes), regimes
    ca = r["centroid_angle"]
    finite_ca = ca[np.isfinite(ca)]
    assert finite_ca.size and finite_ca.max() < 0.5, finite_ca
    print(f"[OK] cone cloud: regime='weak_bipartition', "
          f"centroid_angle max={finite_ca.max():.3f} rad")


def test_antipodal():
    acts = make_antipodal()
    r = analyze_bipartition(acts)
    regimes = list(r["regime"])
    assert all(x == "strong_bipartition" for x in regimes), regimes

    ca = r["centroid_angle"]
    assert (ca > 2.8).all(), ca   # ~π
    within = r["within_half_ip"]
    assert (within > 0.5).all(), within
    eg = r["bipartition_eigengap"]
    assert np.all(np.isfinite(eg)), eg
    print(f"[OK] antipodal: centroid_angle mean={ca.mean():.3f} rad, "
          f"within_ip mean={within.mean():.3f}, "
          f"eigengap mean={eg.mean():.3f}")


def test_tracking_sign_flip():
    """The antipodal construction already triggers sign flips naturally —
    scipy's eigh returns eigenvectors with an arbitrary per-layer sign.
    The aligner must equalize them so that downstream work can compare
    hemisphere membership across layers."""
    acts = make_sign_flip(n_layers=8)
    block0 = analyze_bipartition(acts)

    # Sanity: raw assignments do disagree across layers in at least one
    # place (otherwise this test proves nothing).
    raw = block0["assignments"]
    disagreements = [
        int((raw[L] != raw[L + 1]).sum()) for L in range(raw.shape[0] - 1)
    ]
    assert max(disagreements) > 0, \
        "trivial: scipy happened to give identical signs, test is uninformative"

    tracking = analyze_hemisphere_tracking(block0)

    # After alignment, all aligned assignments should be identical
    # (the underlying geometry is constant).
    aa = tracking["aligned_assignments"]
    for L in range(aa.shape[0] - 1):
        assert np.array_equal(aa[L], aa[L + 1]), (L, aa[L] - aa[L + 1])

    # match_overlap should be ~1.0 on every transition.
    mo = tracking["match_overlap"]
    assert (mo[np.isfinite(mo)] > 0.99).all(), mo

    # No swap events.
    swap_events = [e for e in tracking["events"] if e["type"] == "swap"]
    assert not swap_events, swap_events

    # At least one flip had to happen to correct the natural sign variation.
    n_flips = int(tracking["flips_applied"].sum())
    assert n_flips >= 1, tracking["flips_applied"]
    print(f"[OK] sign flip: {n_flips} layer(s) flipped by aligner, "
          f"match_overlap min={mo[np.isfinite(mo)].min():.3f}, "
          f"no spurious swaps")


def test_json_shapes():
    acts = make_antipodal(n_layers=8, n_tokens=30, d=12)
    b0 = analyze_bipartition(acts)
    b1 = analyze_hemisphere_tracking(b0)
    j0 = bipartition_to_json(b0)
    j1 = hemisphere_tracking_to_json(b1)

    assert len(j0["per_layer"]) == 8
    assert len(j1["per_transition"]) == 7
    assert "regime_counts" in j0["summary"]
    assert "event_counts"  in j1["summary"]
    # All values in j0 summary should be JSON-native (no numpy types).
    import json
    json.dumps(j0)   # will raise if anything is non-serializable
    json.dumps(j1)
    print("[OK] json: per-layer and per-transition shapes, JSON-serializable")


def test_regime_boundaries():
    # Manually build a minority-fraction-tipping case.
    from phase1h.bipartition_detect import classify_regime
    # Collapsed.
    assert classify_regime(0.04, np.pi,     0.9, 0.9) == "collapsed"
    # Weak: minority in [0.05, 0.10).
    assert classify_regime(0.07, np.pi,     0.9, 0.9) == "weak_bipartition"
    # Weak: angle below π/2.
    assert classify_regime(0.30, np.pi/3,   0.9, 0.9) == "weak_bipartition"
    # Diffuse: minority and angle ok, but within_ip low.
    assert classify_regime(0.30, np.pi,     0.1, 0.1) == "diffuse"
    assert classify_regime(0.30, np.pi,     0.5, 0.2) == "diffuse"
    # Strong.
    assert classify_regime(0.30, np.pi,     0.5, 0.5) == "strong_bipartition"
    # nan → collapsed.
    assert classify_regime(float("nan"), np.pi, 0.5, 0.5) == "collapsed"
    print("[OK] regime boundaries")


if __name__ == "__main__":
    test_collapsed()
    test_cone_cloud()
    test_antipodal()
    test_tracking_sign_flip()
    test_json_shapes()
    test_regime_boundaries()
    print("\nAll smoke tests passed.")
