"""
tests/test_p2_producer_changes.py

Two producer changes, tested at the level where they can go wrong.

1. Head-core spectra (weights.head_core_spectrum, the `ov_head_core` field,
   head_ov_analysis's low-rank path). The load-bearing claim is an algebraic
   identity — the nonzero spectrum of A@B equals the spectrum of B@A — so
   the tests assert the identity directly, in both weight-storage
   conventions the extractors use, rather than asserting a plausible-looking
   number. The dilution test asserts the thing that motivated the change:
   the composed-matrix fraction is pinned near 0.5 with a spread several
   times smaller than the core's, because most of what it counts is
   floating-point noise in a null space.

2. full_analysis's `profiles` block. Tested for index alignment against the
   violation loop's own `t = v_layer - 1` convention, since a silent
   off-by-one between the per-layer and per-transition axes would be
   invisible in a figure and wrong everywhere.
"""

import numpy as np
import pytest
from scipy.linalg import eigvals

from p2_eigenspectra.weights import head_core_spectrum, head_core_spectra
from p2_eigenspectra.head_ov_analysis import (
    analyze_per_head_ov, _analyze_head_core, _analyze_single_head,
)


# ─────────────────────────────────────────────────────────────────────────────
# The identity the change rests on
# ─────────────────────────────────────────────────────────────────────────────

def _linear_head(rng, d, dh):
    """ALBERT / BERT / GPT-NeoX convention: OV_h = W_V_h.T @ W_O_h.T."""
    W_V_h = rng.normal(0, d ** -0.5, (dh, d))    # (d_head, d_model)
    W_O_h = rng.normal(0, d ** -0.5, (d, dh))    # (d_model, d_head)
    return W_V_h.T @ W_O_h.T, W_O_h.T @ W_V_h.T


def _conv1d_head(rng, d, dh):
    """GPT-2 Conv1D convention: OV_h = W_V_h @ W_O_h, factors reversed."""
    W_V_h = rng.normal(0, d ** -0.5, (d, dh))    # (d_model, d_head)
    W_O_h = rng.normal(0, d ** -0.5, (dh, d))    # (d_head, d_model)
    return W_V_h @ W_O_h, W_O_h @ W_V_h


@pytest.mark.parametrize("build", [_linear_head, _conv1d_head])
def test_core_spectrum_equals_nonzero_spectrum_of_composed_ov(build):
    rng = np.random.default_rng(0)
    d, dh = 192, 24
    OV, core = build(rng, d, dh)

    assert core.shape == (dh, dh)
    assert np.linalg.matrix_rank(OV) == dh

    e_full = eigvals(OV)
    top = e_full[np.argsort(-np.abs(e_full))][:dh]
    err = np.max(np.abs(np.sort_complex(top) - np.sort_complex(eigvals(core))))
    assert err < 1e-10, f"core spectrum differs from OV_h's nonzero spectrum: {err}"


@pytest.mark.parametrize("build", [_linear_head, _conv1d_head])
def test_core_fraction_has_more_between_head_spread_than_diluted(build):
    """
    The motivating defect. Both estimate the same quantity; the composed
    matrix's version is mostly counting sign-randomised numerical zeros, so
    its spread across heads collapses towards zero.
    """
    rng = np.random.default_rng(3)
    d, dh, n_heads = 256, 16, 12
    diluted, core_based = [], []
    for _ in range(n_heads):
        OV, core = build(rng, d, dh)
        diluted.append(float((np.real(eigvals(OV)) < 0).mean()))
        core_based.append(head_core_spectrum(core)["frac_repulsive"])

    assert abs(np.mean(diluted) - 0.5) < 0.02          # pinned at chance
    assert np.std(core_based) > 3 * np.std(diluted)    # signal recovered


def test_head_core_spectrum_fractions_sum_to_one():
    rng = np.random.default_rng(1)
    core = rng.normal(size=(32, 32))
    s = head_core_spectrum(core)
    assert s["n_eigenvalues"] == 32 and s["n_negligible"] == 0
    assert abs(s["frac_repulsive"] + s["frac_attractive"] - 1.0) < 1e-12
    assert s["spectral_radius"] > 0


def test_head_core_spectrum_reports_rank_deficiency_rather_than_hiding_it():
    """A genuinely rank-deficient core must show up as n_negligible, not be
    counted into one of the sign fractions."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(16, 4))
    core = A @ A.T * 0 + np.diag(np.concatenate([np.ones(4), np.zeros(12)]))
    s = head_core_spectrum(core)
    assert s["n_negligible"] == 12
    assert s["n_eigenvalues"] == 4
    assert s["frac_attractive"] == 1.0


def test_zero_core_is_nan_not_a_confident_half():
    s = head_core_spectrum(np.zeros((8, 8)))
    assert np.isnan(s["frac_repulsive"]) and s["n_negligible"] == 8


# ─────────────────────────────────────────────────────────────────────────────
# Wiring
# ─────────────────────────────────────────────────────────────────────────────

def _fake_ov_data(rng, n_layers=3, n_heads=4, d=64, dh=8, with_cores=True):
    per_layer_heads, per_layer_cores = [], []
    for _ in range(n_layers):
        heads, cores = [], []
        for _ in range(n_heads):
            OV, core = _linear_head(rng, d, dh)
            heads.append(OV)
            cores.append(core)
        per_layer_heads.append(heads)
        per_layer_cores.append(np.stack(cores))
    data = {
        "is_per_layer": True, "ov_per_head": per_layer_heads,
        "d_model": d, "d_head": dh, "n_heads": n_heads,
    }
    if with_cores:
        data["ov_head_core"] = per_layer_cores
    return data


def test_analyze_per_head_ov_uses_cores_when_present():
    rng = np.random.default_rng(5)
    out = analyze_per_head_ov(_fake_ov_data(rng))
    assert out["low_rank"] is True
    assert out["n_layers"] == 3 and out["n_heads"] == 4
    h = out["per_layer_per_head"][0][0]
    assert h["n_eigenvalues"] == 8         # d_head, not d_model
    assert h["low_rank"] is True
    assert h["spectral_norm"] is None      # not preserved by the swap
    assert h["spectral_radius"] > 0


def test_analyze_per_head_ov_falls_back_without_cores():
    """Old ov_data dicts must still work, and must say which path ran."""
    rng = np.random.default_rng(5)
    out = analyze_per_head_ov(_fake_ov_data(rng, with_cores=False))
    assert out["low_rank"] is False
    h = out["per_layer_per_head"][0][0]
    assert h["n_eigenvalues"] == 64        # d_model — the diluted count
    assert h["low_rank"] is False


def test_both_paths_share_the_keys_cross_reference_reads():
    rng = np.random.default_rng(6)
    OV, core = _linear_head(rng, 64, 8)
    a, b = _analyze_single_head(OV), _analyze_head_core(core)
    for k in ("frac_repulsive", "frac_attractive", "eig_real_mean",
              "n_positive", "n_negative"):
        assert k in a and k in b, k


def test_head_core_spectra_is_one_record_per_head():
    rng = np.random.default_rng(7)
    cores = np.stack([_linear_head(rng, 48, 6)[1] for _ in range(5)])
    out = head_core_spectra(cores)
    assert len(out) == 5 and all("frac_repulsive" in h for h in out)


# ─────────────────────────────────────────────────────────────────────────────
# full_analysis profiles
# ─────────────────────────────────────────────────────────────────────────────

def _fake_traj(n_layers=6, violation_layers=(2, 4)):
    n_trans = n_layers - 1
    return {
        "events": {"energy_violations": {1.0: list(violation_layers),
                                         5.0: []}},
        "steps": {
            "step_mean": np.arange(n_trans, dtype=float),
            "step_std": np.ones(n_trans),
            "global_mean": 1.0, "global_std": 0.5,
            "overshoot_threshold": 2.0,
        },
        "disp": {
            "sym_repulse_disp_frac": np.linspace(0, 1, n_trans),
            "sym_attract_disp_frac": np.linspace(1, 0, n_trans),
            "schur_repulse_disp_frac": np.zeros(n_trans),
            "schur_attract_disp_frac": np.zeros(n_trans),
            "total_disp_energy": np.ones(n_trans),
        },
        "self_int": {
            "self_int": np.zeros((n_layers, 32)),      # must NOT be forwarded
            "self_int_mean": np.zeros(n_layers),
            "self_int_std": np.ones(n_layers),
            "frac_negative": np.linspace(0, 1, n_layers),
        },
        "subspace": {
            "schur_attract_frac": np.zeros(n_layers),
            "schur_repulse_frac": np.ones(n_layers),
            "sym_attract_frac": np.zeros(n_layers),
            "sym_repulse_frac": np.ones(n_layers),
        },
        "rescaled": {"n_violations": {1.0: 1, 5.0: 0},
                     "comparison_with_meanv": {"beta_1.0": {"approx_error_pct": 12.5}}},
    }


def test_profiles_axes_have_the_lengths_their_names_promise():
    from p2_eigenspectra.analysis_p2 import full_analysis

    n_layers = 6
    res = full_analysis(_fake_traj(n_layers), {"is_per_layer": True})
    prof = res["profiles"]
    assert prof["n_layers"] == n_layers
    assert prof["n_transitions"] == n_layers - 1
    for v in prof["per_layer"].values():
        assert len(v) == n_layers
    for v in prof["per_transition"].values():
        assert len(v) == n_layers - 1


def test_profiles_index_alignment_matches_the_violation_loop():
    """
    full_analysis classifies violation layer L using transition index
    L-1. A consumer reading profiles must be able to reproduce that
    classification exactly; if it can't, the two axes have drifted.
    """
    from p2_eigenspectra.analysis_p2 import full_analysis

    traj = _fake_traj(6, violation_layers=(2, 4))
    res = full_analysis(traj, {"is_per_layer": True})
    prof = res["profiles"]
    rep = prof["per_transition"]["sym_repulse_disp_frac"]
    neg = prof["per_layer"]["frac_negative"]

    for rec in res["violations_beta1.0"]["per_violation"]:
        L = rec["layer"]
        assert rec["repulsive"] == bool(rep[L - 1] > 0.5)
        assert rec["self_int_neg"] == bool(neg[L] > 0.5)
        assert rec["overshoot"] == bool(
            prof["per_transition"]["step_mean"][L - 1]
            > prof["overshoot_threshold"])


def test_profiles_do_not_forward_the_per_token_matrices():
    """self_int is (n_layers, n_tokens) and step_norms is (n_trans,
    n_tokens); forwarding either would put a token-sized array in every
    saved sub-experiment JSON."""
    from p2_eigenspectra.analysis_p2 import full_analysis

    prof = full_analysis(_fake_traj(), {"is_per_layer": True})["profiles"]
    assert "self_int" not in prof["per_layer"]
    assert "step_norms" not in prof["per_transition"]


def test_rescaled_comparison_is_forwarded():
    from p2_eigenspectra.analysis_p2 import full_analysis

    res = full_analysis(_fake_traj(), {"is_per_layer": True})
    assert res["rescaled_comparison"]["beta_1.0"]["approx_error_pct"] == 12.5


def test_profiles_degrade_rather_than_raise_on_a_partial_traj():
    """The shared-weight path returns a subset of these blocks."""
    from p2_eigenspectra.analysis_p2 import full_analysis

    traj = _fake_traj()
    del traj["subspace"]
    prof = full_analysis(traj, {"is_per_layer": False})["profiles"]
    assert "frac_negative" in prof["per_layer"]
    assert not any(k.startswith("subspace_") for k in prof["per_layer"])


# ─────────────────────────────────────────────────────────────────────────────
# Schur sort guard
# ─────────────────────────────────────────────────────────────────────────────

def test_rank_deficient_ov_does_not_crash_eigendecompose():
    """
    LAPACK's Schur reordering fails when eigenvalues cluster on the sorting
    boundary, which a rank-deficient OV guarantees: its null eigenvalues sit
    at the origin. Unguarded this raised LinAlgError, and analyze_weights is
    called outside run_2's per-model try, so one bad checkpoint took the rest
    of the sweep with it.
    """
    from numpy.linalg import LinAlgError
    from p2_eigenspectra.weights import eigendecompose, build_subspace_projectors

    rng = np.random.default_rng(11)
    d, n_retry = 32, 0
    for _ in range(25):
        r = int(rng.integers(2, 8))
        M = rng.normal(size=(d, r)) @ rng.normal(size=(r, d))
        try:
            dec = eigendecompose(M)
        except LinAlgError as exc:
            pytest.fail(f"rank-{r} OV still crashes: {exc}")
        n_retry += int(dec["schur_sort_tol"] > 0)

        # The retry must still produce a genuinely SORTED Schur form, or the
        # projectors built from Z are silently wrong.
        P = build_subspace_projectors(dec)
        Pa, Pr = P["schur_attract"], P["schur_repulse"]
        assert np.allclose(Pa @ Pa, Pa, atol=1e-8), "not idempotent"
        assert np.allclose(Pa, Pa.T, atol=1e-8), "not symmetric"
        assert np.allclose(Pa + Pr, np.eye(d), atol=1e-8), "not complementary"

    assert n_retry > 0, "guard never exercised — test no longer covers the path"


def test_full_rank_ov_does_not_use_the_tolerance_band():
    """The retry must stay off the normal path: for standard MHA,
    n_heads * d_head == d_model and OV_total is generically full rank."""
    from p2_eigenspectra.weights import eigendecompose

    rng = np.random.default_rng(4)
    dec = eigendecompose(rng.normal(size=(48, 48)) / 48 ** 0.5)
    assert dec["schur_sort_tol"] == 0.0
