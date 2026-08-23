"""
tests/test_phase1b_v2.py — the Phase 1b revision.

Companion to tests/test_phase1b.py, which is unchanged and still passes in
full (65/65). This file covers only what the revision added or corrected, so
the two can be read as "the behaviour that was already specified" and "the
behaviour that changed, and why".

Several tests below are written as regressions against a specific defect and
say so in their docstring. Those are the ones worth keeping if this file is
ever pruned: each encodes a failure that shipped, was reported as a finding,
and is not visible from the code alone.

Everything here is pure numpy/scipy — no torch, no model. That is deliberate:
every bug this file guards against lived in code that could not be imported
without a GPU stack and therefore had no test.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import fiedler_and_eigengap
from p1b_hemisphere.axis_identity import (
    analyze_axis_identity,
    axis_alignment,
    axis_in_activation_space,
    axis_settling_step,
    cross_checkpoint_axis_rotation,
    mean_direction,
)
from p1b_hemisphere.bipartition_detect import (
    CONNECTIVITY_FLOOR,
    analyze_bipartition,
    classify_regime,
    classify_regime_relative,
    extract_bipartition_spectrum,
)
from p1b_hemisphere.cone_collapse import (
    analyze_cone_collapse,
    classify_cone_regime,
    cone_collapse_to_json,
    cone_margin_lp,
    normalized_margin_of,
    uniform_sphere_null,
)
from p1b_hemisphere.hemisphere_membership import (
    analyze_hemisphere_membership,
    border_vs_noise,
    compute_token_trajectories,
    membership_to_json,
)
from p1b_hemisphere.hemisphere_tracking import (
    _match_overlap,
    analyze_hemisphere_tracking,
)
from p1b_hemisphere.p1b_report import (
    LONG_PROMPT_TOKENS,
    aggregate_by_checkpoint,
    checkpoint_base,
    checkpoint_step,
    cross_run_markdown,
    global_verdict,
)

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pyproject.toml [tool.pytest.ini_options].markers.
pytestmark = pytest.mark.pure


# ---------------------------------------------------------------------------
# Fixtures (plain functions — no pytest fixture injection, so the project's
# manual runner can execute these too)
# ---------------------------------------------------------------------------

def _rng(seed=0):
    return np.random.default_rng(seed)


def make_cap(n=40, d=12, seed=0, spread=0.25):
    """Points in a narrow cap around e0. Cone-collapsed by construction."""
    r = _rng(seed)
    X = r.standard_normal((n, d)) * spread
    X[:, 0] += 3.0
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def make_two_clusters(n=40, d=16, angle=np.pi, seed=3, noise=0.2):
    """Two equal clusters whose centres are `angle` apart."""
    r = _rng(seed)
    a = np.zeros(d); a[0] = 1.0
    b = np.cos(angle) * a + np.sin(angle) * np.eye(d)[1]
    X = np.vstack([a + noise * r.standard_normal((n // 2, d)),
                   b + noise * r.standard_normal((n // 2, d))])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def make_antipodal(n=20, d=8, seed=1):
    r = _rng(seed)
    Y = r.standard_normal((n // 2, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return np.vstack([Y, -Y])


def stack(X, n_layers=4):
    return np.stack([X] * n_layers)


# ---------------------------------------------------------------------------
# core.metrics: connectivity_floor and clip_negative
# ---------------------------------------------------------------------------

class TestFiedlerParameters:

    def test_floor_default_is_zero_and_preserves_phase1(self):
        """
        The default must reproduce Phase 1's graph exactly. Phase 1b's
        1e-4/n floor is opt-in; if it ever became the default, every Phase 1
        spectral.json would silently stop matching a re-run.
        """
        G = make_cap() @ make_cap().T
        a = fiedler_and_eigengap(G, max_k=2, return_fiedler_vec=True)
        b = fiedler_and_eigengap(G, max_k=2, return_fiedler_vec=True,
                                 connectivity_floor=0.0)
        assert a["connectivity_floor"] == 0.0
        assert a["eigenvalues"] == b["eigenvalues"]

    def test_floor_reconnects_a_disconnected_antipodal_graph(self):
        """
        Regression / rationale: clipping negatives on antipodal geometry
        removes every cross-group edge, so lambda_2 collapses to numerical
        zero and the Fiedler eigenspace is degenerate. This is why the floor
        exists at all, and the magnitude of the difference is why the two
        graphs were never interchangeable.
        """
        # Two TIGHT antipodal clusters, which is what actually disconnects.
        # Note that X-and-minus-X on random directions does NOT disconnect
        # (measured lambda_2 = 0.28 there): within-group inner products are
        # near zero and survive clipping. The floor matters exactly for the
        # geometry this phase is about — two compact, mutually negative
        # groups — which is why the fixture is specific.
        G = make_two_clusters(n=40, d=16, angle=np.pi)
        G = G @ G.T
        no_floor = fiedler_and_eigengap(G, max_k=2)["eigenvalues"][1]
        floored  = fiedler_and_eigengap(G, max_k=2,
                                        connectivity_floor=1e-4)["eigenvalues"][1]
        assert no_floor < 1e-12
        assert floored > 1e6 * max(no_floor, 1e-30)

    def test_clip_negative_false_is_carried_not_dropped(self):
        """
        bipartition_detect genuinely supported a signed Laplacian. Folding it
        into the shared implementation had to carry that, not silently remove
        it — a capability deleted during de-duplication is the same class of
        failure as the duplication.
        """
        X = make_cap(n=20, d=8)
        G = X @ X.T
        signed  = fiedler_and_eigengap(G, max_k=2, clip_negative=False)
        clipped = fiedler_and_eigengap(G, max_k=2, clip_negative=True)
        assert signed["clip_negative"] is False
        assert clipped["clip_negative"] is True

    def test_signed_laplacian_degrades_instead_of_raising(self):
        """
        Regression: a signed Gram can give vanishing or negative degrees, so
        the normalized Laplacian contains inf/nan and scipy's eigh raises
        rather than returning them. bipartition_detect used to absorb that in
        a bare `except Exception: continue`; after delegation the shared
        function has to handle it, or an advertised option crashes.
        """
        X = make_antipodal(n=16, d=8)
        out = fiedler_and_eigengap(X @ X.T, max_k=2, clip_negative=False)
        assert out["eigenvalues"] == []
        assert out["fiedler_value"] != out["fiedler_value"]   # nan


# ---------------------------------------------------------------------------
# Block 0
# ---------------------------------------------------------------------------

class TestBlock0Delegation:

    def test_spectrum_matches_core_metrics_exactly(self):
        """Block 0 must not maintain a second Laplacian."""
        X = make_two_clusters()
        spec = extract_bipartition_spectrum(stack(X, 1),
                                            connectivity_floor=CONNECTIVITY_FLOOR)
        direct = fiedler_and_eigengap(X @ X.T, max_k=2, return_fiedler_vec=True,
                                      connectivity_floor=CONNECTIVITY_FLOOR)
        assert np.allclose(spec["eigvals"][0], direct["eigenvalues"][:3])
        assert np.allclose(np.abs(spec["fiedler_vecs"][0]),
                           np.abs(np.array(direct["fiedler_vec"])))

    def test_frame_travels_with_the_result(self):
        r = analyze_bipartition(stack(make_cap()))
        assert r["frame"].kind == "l2_sphere"
        assert r["connectivity_floor"] == CONNECTIVITY_FLOOR


class TestRelativeRegimeClassifier:

    def test_antipodal_agrees_with_the_legacy_classifier(self):
        r = analyze_bipartition(stack(make_two_clusters(angle=np.pi), 1))
        assert str(r["regime"][0]) == "strong_bipartition"
        assert str(r["regime_relative"][0]) == "separated"

    def test_two_clusters_inside_one_hemisphere(self):
        """
        The case the antipodal classifier cannot express, and the reason
        "0% strong bipartition" is not the same claim as "no bipartition".

        Two genuinely separated clusters 60 degrees apart: both halves
        populated, cross-half similarity well below within-half. The
        antipodal rule rejects it purely because the centroid angle is under
        pi/2 — a condition cone-collapse makes unreachable.
        """
        r = analyze_bipartition(stack(make_two_clusters(angle=np.pi / 3), 1))
        assert str(r["regime"][0]) == "weak_bipartition"
        assert str(r["regime_relative"][0]) == "separated"
        assert r["centroid_angle"][0] < np.pi / 2
        assert r["separation_ratio"][0] < 0.9
        assert r["minority_fraction"][0] > 0.3

    def test_tight_cloud_is_not_separated(self):
        r = analyze_bipartition(stack(make_cap(spread=0.25), 1))
        assert str(r["regime_relative"][0]) in ("graded", "uniform", "collapsed")

    def test_separation_ratio_decreases_as_the_cloud_spreads(self):
        """
        The relative classifier has to be monotone in the thing it claims to
        measure, or its thresholds are arbitrary. A tight cap has no angular
        structure; a wide one does.
        """
        ratios = [analyze_bipartition(stack(make_cap(spread=s), 1))["separation_ratio"][0]
                  for s in (0.25, 0.6, 1.0, 2.0)]
        assert all(a > b for a, b in zip(ratios, ratios[1:])), ratios

    def test_nan_inputs_collapse(self):
        assert classify_regime_relative(float("nan"), 0.5) == "collapsed"
        assert classify_regime_relative(0.4, float("nan")) == "collapsed"

    def test_thresholds_are_ordered(self):
        assert classify_regime_relative(0.4, 0.5) == "separated"
        assert classify_regime_relative(0.4, 0.95) == "graded"
        assert classify_regime_relative(0.4, 1.0) == "uniform"
        assert classify_regime_relative(0.01, 0.1) == "collapsed"

    def test_legacy_classifier_is_untouched(self):
        assert classify_regime(0.3, np.pi * 0.9, 0.5, 0.5) == "strong_bipartition"
        assert classify_regime(0.01, np.pi * 0.9, 0.5, 0.5) == "collapsed"


# ---------------------------------------------------------------------------
# Block 3
# ---------------------------------------------------------------------------

class TestConeMargin:

    def test_cap_is_cone_collapsed(self):
        r = cone_margin_lp(make_cap(), pca_n_components=None)
        assert r["solved"] and r["cone_margin"] > 0
        assert classify_cone_regime(r["cone_margin"]) == "cone_collapse"

    def test_antipodal_is_split_not_borderline(self):
        r = cone_margin_lp(make_antipodal(), pca_n_components=None)
        assert r["degenerate_w"] is True
        assert classify_cone_regime(r["cone_margin"]) == "split"

    def test_pca_witness_lifts_exactly_to_full_dimension(self):
        """
        The asymmetry the old docstring got wrong. A reduced-space witness
        satisfies every full-space constraint identically, because
        X @ (Vt[:k].T @ w_r) == (X @ Vt[:k].T) @ w_r. So a cone_collapse
        verdict under PCA is sound; only the split direction can be a
        projection artifact, which is what escalate_on_split exists for.
        """
        X = make_cap(n=40, d=20)
        red = cone_margin_lp(X, pca_n_components=5)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        w_full = Vt[:5].T @ red["w_opt"]
        assert float((X @ w_full).min()) == pytest.approx(red["cone_margin"], abs=1e-9)

    def test_binding_tokens_are_original_indices(self):
        """
        The whole point of recording them is asking whether position 0 holds
        up the cone. Indices into a post-drop row order could not answer that.
        """
        X = make_cap(n=12, d=6)
        r = cone_margin_lp(X, pca_n_components=None, drop_indices=[0])
        assert 0 not in r["binding_tokens"]
        assert set(r["binding_tokens"]).issubset(set(range(1, 12)))
        assert r["n_used"] == 11

    def test_escalation_fires_only_on_non_collapse(self):
        cap = analyze_cone_collapse(stack(make_cap(n=40, d=20), 2),
                                    pca_n_components=2)
        assert not cap["escalated"].any()

        r = _rng(9)
        Y = r.standard_normal((12, 30)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        hard = analyze_cone_collapse(stack(Y, 1), pca_n_components=2)
        assert bool(hard["escalated"][0])
        assert hard["d_eff"][0] > 2

    def test_escalation_can_be_disabled(self):
        r = _rng(9)
        Y = r.standard_normal((12, 30)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        out = analyze_cone_collapse(stack(Y, 1), pca_n_components=2,
                                    escalate_on_split=False)
        assert not out["escalated"].any()


class TestConeNulls:

    def test_uniform_null_at_matched_n_and_d(self):
        """
        n points in d dimensions with n > d positively span, so the uniform
        control is reliably NOT cone-collapsed. That contrast is what makes
        an observed cone-collapse a statement about the model rather than
        about dimension counting.
        """
        vals = uniform_sphere_null(n=40, d=8, n_draws=5,
                                   pca_n_components=None, rng=_rng(1))
        assert np.all(np.isfinite(vals))
        assert np.all(vals <= 0)

    def test_observed_cap_beats_the_uniform_null(self):
        res = analyze_cone_collapse(stack(make_cap(n=30, d=8), 1),
                                    pca_n_components=None, n_null=5, rng=_rng(2))
        nl = res["nulls"][0]
        assert nl["uniform_cone_fraction"] == 0.0
        assert nl["observed"] > 0

    def test_degenerate_null_reports_nan_z_not_a_fabricated_number(self):
        """
        A constant null has zero variance. sigma_from_null returns nan there
        by design; the cone-fraction statistic is what carries the answer, and
        a table that only had z would have shown a blank where the strongest
        result was.
        """
        res = analyze_cone_collapse(stack(make_cap(n=30, d=8), 1),
                                    pca_n_components=None, n_null=5, rng=_rng(2))
        j = cone_collapse_to_json(res)["per_layer"][0]
        assert j["z_vs_uniform"] is None
        assert j["uniform_cone_fraction"] == 0.0

    def test_nulls_are_off_by_default(self):
        res = analyze_cone_collapse(stack(make_cap(), 2))
        assert res["nulls"] == {}

    def test_normalized_margin_is_scale_free(self):
        X = make_cap()
        assert normalized_margin_of(X, None) == pytest.approx(
            normalized_margin_of(X * 3.0, None), abs=1e-6)


# ---------------------------------------------------------------------------
# Block 1
# ---------------------------------------------------------------------------

class TestMatcherDelegation:

    def test_backends_agree_on_random_label_pairs(self):
        """
        Regression: Hungarian is free to return either pairing on an exact
        tie, and did on 4 of 500 random pairs. align_hemisphere_labels chains
        anchors forward, so each of those would have inverted the hemisphere
        labelling for the remainder of the run on a coin toss.
        """
        r = _rng(7)
        for _ in range(400):
            n = int(r.integers(4, 50))
            a = r.integers(0, 2, n)
            b = r.integers(0, 2, n)
            assert _match_overlap(a, b, "hungarian") == _match_overlap(a, b, "local")

    def test_flip_and_identity(self):
        a = np.array([0, 0, 0, 1, 1, 1])
        assert _match_overlap(a, 1 - a) == (1.0, True)
        assert _match_overlap(a, a) == (1.0, False)

    def test_tie_breaks_to_identity(self):
        a = np.array([0, 1, 0, 1])
        score, flip = _match_overlap(a, np.array([0, 1, 1, 0]))
        assert flip is False

    def test_unknown_matcher_rejected(self):
        with pytest.raises(ValueError):
            _match_overlap(np.zeros(4, int), np.ones(4, int), matcher="nope")


class TestRegimeVocabularyWiring:

    def test_antipodal_vocabulary_forecloses_persistence_under_cone_collapse(self):
        """
        Regression, and the reason the original run reported zero events:
        persistence and birth/collapse were hardcoded to
        "strong_bipartition", which cone-collapse makes unreachable. The
        statistic looked measured and was foreclosed.
        """
        b0 = analyze_bipartition(stack(make_two_clusters(angle=np.pi / 3), 5))
        anti = analyze_hemisphere_tracking(b0, regime_key="regime")
        rel  = analyze_hemisphere_tracking(b0, regime_key="regime_relative")
        assert anti["persistence_length"].max() == 0
        assert rel["persistence_length"].max() > 0
        assert rel["stable_label"] == "separated"

    def test_unknown_regime_key_raises(self):
        b0 = analyze_bipartition(stack(make_cap(), 3))
        with pytest.raises(KeyError):
            analyze_hemisphere_tracking(b0, regime_key="not_a_regime")


# ---------------------------------------------------------------------------
# Block 2
# ---------------------------------------------------------------------------

class TestFirstStableLayer:

    def test_final_reference_matches_the_documented_definition(self):
        """
        Regression: the docstring said "matches its final-layer assignment"
        and the code compared against the most-held label. A token that
        switches once and then stays was recorded as never stable.
        """
        aligned = np.array([[0], [0], [0], [1], [1]], dtype=np.int8)
        f = np.ones((5, 1))
        valid = np.ones(5, dtype=bool)

        fin = compute_token_trajectories(aligned, f, valid, stable_reference="final")
        dom = compute_token_trajectories(aligned, f, valid, stable_reference="dominant")

        assert fin["first_stable_layer"][0] == 3
        assert dom["first_stable_layer"][0] == -1
        assert fin["final_hemisphere"][0] == 1
        assert fin["dominant_hemisphere"][0] == 0

    def test_stable_token_agrees_under_both_references(self):
        aligned = np.zeros((5, 1), dtype=np.int8)
        f = np.ones((5, 1))
        valid = np.ones(5, dtype=bool)
        for ref in ("final", "dominant"):
            t = compute_token_trajectories(aligned, f, valid, stable_reference=ref)
            assert t["first_stable_layer"][0] == 0

    def test_bad_reference_raises(self):
        with pytest.raises(ValueError):
            compute_token_trajectories(np.zeros((3, 2), np.int8), np.ones((3, 2)),
                                       np.ones(3, bool), stable_reference="modal")


class TestBorderVsNoise:

    def _fixture(self, mode, n=60, n_layers=4, seed=2):
        r = _rng(seed)
        fv = np.zeros((n_layers, n))
        labels = {}
        for L in range(n_layers):
            v = r.standard_normal(n)
            v = np.sign(v) * (0.5 + np.abs(v))
            idx = np.argsort(np.abs(v))
            v[idx[:15]] *= 0.02
            fv[L] = v
            lab = np.zeros(n, int)
            if mode == "aligned":
                lab[idx[:15]] = -1
            elif mode == "inverted":
                lab[np.argsort(np.abs(v))[-15:]] = -1
            else:
                lab = np.where(r.random(n) < 0.25, -1, 0)
            labels[L] = lab
        return fv, labels, np.ones(n_layers, bool)

    def test_noise_at_the_boundary_gives_auc_one(self):
        fv, labels, valid = self._fixture("aligned")
        assert border_vs_noise(fv, labels, valid)["overall"]["mean_auc"] == pytest.approx(1.0, abs=1e-9)

    def test_noise_deepest_gives_auc_zero(self):
        fv, labels, valid = self._fixture("inverted")
        assert border_vs_noise(fv, labels, valid)["overall"]["mean_auc"] == pytest.approx(0.0, abs=1e-9)

    def test_random_noise_is_near_half(self):
        fv, labels, valid = self._fixture("random")
        assert abs(border_vs_noise(fv, labels, valid)["overall"]["mean_auc"] - 0.5) < 0.15

    def test_layers_without_both_populations_are_skipped(self):
        fv, labels, valid = self._fixture("aligned", n_layers=2)
        labels[0] = np.zeros(60, int)          # no noise at all
        out = border_vs_noise(fv, labels, valid)
        assert 0 not in out["per_layer"]
        assert out["overall"]["n_analyzed_layers"] == 1

    def test_absent_without_hdbscan_labels(self):
        b0 = analyze_bipartition(stack(make_cap(), 3))
        b1 = analyze_hemisphere_tracking(b0)
        j = membership_to_json(analyze_hemisphere_membership(b0, b1))
        assert "border_vs_noise" not in j


# ---------------------------------------------------------------------------
# Block A — axis identity
# ---------------------------------------------------------------------------

class TestAxisIdentity:

    def test_axis_is_a_unit_vector_with_a_fixed_sign(self):
        X = make_two_clusters(angle=np.pi / 2)
        b0 = analyze_bipartition(stack(X, 1))
        f = b0["fiedler_vecs"][0]
        a1 = axis_in_activation_space(X, f)
        a2 = axis_in_activation_space(X, -f)
        assert np.linalg.norm(a1) == pytest.approx(1.0, abs=1e-9)
        # A global sign flip of the eigenvector must not flip the axis, or no
        # two axes are comparable across layers or checkpoints.
        assert np.allclose(a1, a2, atol=1e-9)

    def test_axis_is_mean_orthogonal_by_construction(self):
        """
        Regression against this module's own first version, which gave
        `redundancy` a "mean_direction" branch. That branch is unreachable:
        the Fiedler vector is the second Laplacian eigenvector and is
        orthogonal to the first (D^(1/2)·1), so X^T f cancels whatever every
        token shares and the axis comes out near-orthogonal to the mean.
        Measured |cos(axis, mean)| across these fixtures: 0.000 to 0.085.

        Shipping that branch would have repeated the exact defect this
        revision flags in classify_regime, where strong_bipartition requires
        an angle cone-collapse makes unattainable and the resulting 0% was
        read as evidence. cos_axis_mean is now a degeneracy diagnostic.
        """
        for X in (make_cap(spread=0.25), make_cap(spread=1.0),
                  make_two_clusters(angle=np.pi / 3),
                  make_two_clusters(angle=np.pi)):
            b0 = analyze_bipartition(stack(X, 1))
            out = axis_alignment(X, b0["fiedler_vecs"][0])
            assert out["cos_axis_mean"] < 0.2, out["cos_axis_mean"]

    def test_redundancy_never_returns_the_removed_branch(self):
        for X in (make_cap(), make_two_clusters(angle=np.pi / 3), make_antipodal()):
            b0 = analyze_bipartition(stack(X, 1))
            assert axis_alignment(X, b0["fiedler_vecs"][0])["redundancy"] != "mean_direction"

    def test_pc1_case_is_detected(self):
        """A split along the cloud's dominant variance direction."""
        r = _rng(5)
        d = 12
        base = np.zeros(d); base[0] = 4.0
        off = np.zeros(d); off[1] = 1.0
        X = np.vstack([base + off + 0.15 * r.standard_normal((30, d)),
                       base - off + 0.15 * r.standard_normal((30, d))])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        b0 = analyze_bipartition(stack(X, 1))
        out = axis_alignment(X, b0["fiedler_vecs"][0])
        assert out["cos_axis_pc1"] > 0.9
        assert out["redundancy"] == "pc1"

    def test_pc_subspace_fraction_is_bounded_and_at_least_the_pc1_cosine(self):
        """
        The subspace fraction must dominate any single component's squared
        cosine, or it is not measuring containment in the block.
        """
        X = make_two_clusters(angle=np.pi / 3)
        b0 = analyze_bipartition(stack(X, 1))
        out = axis_alignment(X, b0["fiedler_vecs"][0], n_components=3)
        assert 0.0 <= out["pc_subspace_fraction"] <= 1.0 + 1e-9
        assert out["pc_subspace_fraction"] >= out["cos_axis_pc1"] ** 2 - 1e-9

    def test_isotropic_baseline_is_reported(self):
        """
        Without 1/sqrt(d) beside it, a cosine of 0.3 is not obviously
        different from chance at d = 1024.
        """
        X = make_cap(d=16)
        b0 = analyze_bipartition(stack(X, 1))
        out = axis_alignment(X, b0["fiedler_vecs"][0])
        assert out["isotropic_cos"] == pytest.approx(0.25, abs=1e-9)

    def test_degenerate_input(self):
        out = axis_alignment(np.zeros((2, 4)), np.zeros(2))
        assert out["redundancy"] == "degenerate"

    def test_pipeline_shapes(self):
        X = make_two_clusters()
        b0 = analyze_bipartition(stack(X, 4))
        out = analyze_axis_identity(b0["frame_activations"], b0["fiedler_vecs"],
                                    b0["valid"])
        assert out["axes"].shape == (4, X.shape[1])
        assert out["modal_redundancy"] in (
            "mean_direction", "pc1", "distinct", "degenerate")


class TestCrossCheckpointAxis:

    def test_rotation_to_final_and_settling_step(self):
        """
        The statistic PREDICTIONS.md claim (b) needs: an axis that reaches its
        trained direction at some step and stays. Adjacent-pair rotation
        cannot answer "when did it settle" on its own.
        """
        d = 8
        target = np.zeros(d); target[0] = 1.0
        early  = np.zeros(d); early[1] = 1.0
        axes = {0: early, 8: early, 512: target, 2000: target, 143000: target}
        rot = cross_checkpoint_axis_rotation(axes, reference="final")
        assert rot["steps"] == [0, 8, 512, 2000, 143000]
        assert rot["rotation"][0] == pytest.approx(np.pi / 2, abs=1e-6)
        assert rot["rotation"][-1] == pytest.approx(0.0, abs=1e-6)
        assert axis_settling_step(rot) == 512

    def test_never_settles_returns_none(self):
        d = 8
        axes = {0: np.eye(d)[0], 100: np.eye(d)[1], 200: np.eye(d)[0],
                300: np.eye(d)[1]}
        assert axis_settling_step(
            cross_checkpoint_axis_rotation(axes, reference="final")) is None

    def test_log_step_axis_not_index(self):
        """
        Pythia's checkpoints are log-spaced to 512 and linear after, so a
        derivative over checkpoint index peaks where the release schedule
        changes spacing rather than where training does.
        """
        axes = {0: np.eye(4)[0], 512: np.eye(4)[0], 143000: np.eye(4)[0]}
        rot = cross_checkpoint_axis_rotation(axes)
        assert rot["log_step"][0] == pytest.approx(0.0)
        assert rot["log_step"][1] == pytest.approx(np.log10(513))

    def test_single_checkpoint_is_empty_not_an_error(self):
        out = cross_checkpoint_axis_rotation({0: np.eye(4)[0]})
        assert out["rotation"].size == 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fake_run(model, n_tokens=150, cone=1.0, sep=0.8, strong=0.0,
              overlap=0.9, uniform_cone=0.0, redundancy="distinct"):
    return {
        "model": model,
        "prompt": "wiki_paragraph",
        "n_tokens": n_tokens,
        "per_layer": [{"match_overlap": overlap} for _ in range(4)],
        "summary": {
            "strong_bipartition_layer_fraction": strong,
            "separated_layer_fraction": sep,
            "cone_collapse_layer_fraction": cone,
            "mean_uniform_cone_fraction": uniform_cone,
            "axis_modal_redundancy": redundancy,
            "hdbscan_nesting_overall": {"fully_nested_fraction": 0.9},
        },
    }


class TestGlobalVerdict:

    def test_cone_field_is_named_for_what_true_means(self):
        """
        Regression: the field was computed as `mean(cone_fraction) < 0.5` and
        published as `cone_collapse_regime_at_long_prompts`, so it read True
        exactly when cone-collapse was rare. The derived paper_alignment
        string was right, so the JSON carried a correct verdict next to a
        boolean asserting its opposite.
        """
        v = global_verdict([_fake_run("gpt2", cone=1.0)])
        assert v["split_regime_at_long_prompts"] is False
        assert v["cone_collapse_regime_at_long_prompts"] is True
        assert v["paper_alignment"] == "cone_collapse"

    def test_split_case(self):
        v = global_verdict([_fake_run("gpt2", cone=0.1)])
        assert v["split_regime_at_long_prompts"] is True
        assert v["cone_collapse_regime_at_long_prompts"] is False
        assert v["paper_alignment"] == "split"

    def test_no_long_prompts_yields_none_not_a_guess(self):
        v = global_verdict([_fake_run("gpt2", n_tokens=10)])
        assert v["split_regime_at_long_prompts"] is None
        assert v["cone_collapse_regime_at_long_prompts"] is None
        assert v["paper_alignment"] == "mixed"
        assert v["n_long_prompt_runs"] == 0

    def test_threshold_is_recorded_with_the_verdict(self):
        """
        The token threshold is tokenizer-dependent, so a verdict that does not
        carry it cannot be compared across model families.
        """
        v = global_verdict([_fake_run("gpt2")], long_prompt_tokens=500)
        assert v["long_prompt_token_threshold"] == 500
        assert v["n_long_prompt_runs"] == 0
        assert LONG_PROMPT_TOKENS == 100

    def test_both_classifiers_are_reported(self):
        v = global_verdict([_fake_run("gpt2", strong=0.0, sep=0.8)])
        assert v["antipodal_bipartition_present_universally"] is False
        assert v["separated_under_relative_classifier"] is True

    def test_dimension_null_verdict_is_none_without_nulls(self):
        r = _fake_run("gpt2")
        del r["summary"]["mean_uniform_cone_fraction"]
        assert global_verdict([r])["cone_collapse_above_dimension_null"] is None

    def test_dimension_null_verdict_when_null_reproduces_it(self):
        v = global_verdict([_fake_run("gpt2", uniform_cone=1.0)])
        assert v["cone_collapse_above_dimension_null"] is False


class TestCheckpointAggregation:

    def test_step_and_base_parsing(self):
        assert checkpoint_step("pythia-410m-step2000") == 2000
        assert checkpoint_base("pythia-410m-step2000") == "pythia-410m"
        assert checkpoint_step("gpt2-large") is None

    def test_random_baseline_is_not_on_the_step_axis(self):
        """
        pythia-1.4b-random deliberately carries no checkpoint_step: it is not
        a point on the training trajectory and must not be drawn on the step
        axis (core/pythia_registry.py).
        """
        assert checkpoint_step("pythia-1.4b-random") is None
        out = aggregate_by_checkpoint([_fake_run("pythia-1.4b-random")])
        assert out == {}

    def test_family_grouping_and_log_axis(self):
        runs = [_fake_run(f"pythia-410m-step{s}") for s in (0, 512, 143000)]
        out = aggregate_by_checkpoint(runs)
        fam = out["pythia-410m"]
        assert fam["steps"] == [0, 512, 143000]
        assert fam["log_step"][0] == pytest.approx(0.0)
        assert fam["per_step"][512]["n_runs"] == 1

    def test_non_checkpoint_models_are_excluded(self):
        assert aggregate_by_checkpoint([_fake_run("gpt2-large")]) == {}


class TestGeneratedNarrative:

    def _md(self, runs):
        by_model = {}
        for r in runs:
            by_model.setdefault(r["model"], []).append(r)
        from p1b_hemisphere.p1b_report import aggregate
        cross = {
            "by_model": {m: aggregate(rs) for m, rs in by_model.items()},
            "by_prompt": {},
            "by_checkpoint": aggregate_by_checkpoint(runs),
            "global_verdict": global_verdict(runs),
        }
        return cross_run_markdown(cross, by_model, {})

    def test_does_not_claim_a_split_when_none_was_found(self):
        """
        Regression: _write_cross_run_md ended with three hardcoded paragraphs
        asserting that long prompts enter a split regime at mid-depth. The run
        found the opposite at every layer of every model.
        """
        md = self._md([_fake_run("gpt2", cone=1.0)])
        assert "every layer admits an enclosing open half-space" in md
        assert "enter a split regime" not in md

    def test_reports_a_split_when_one_was_found(self):
        md = self._md([_fake_run("gpt2", cone=0.1)])
        assert "admit no enclosing half-space" in md

    def test_states_the_null_caveat_when_no_null_was_run(self):
        r = _fake_run("gpt2")
        del r["summary"]["mean_uniform_cone_fraction"]
        md = self._md([r])
        assert "--n-null" in md

    def test_bipartition_paragraph_follows_the_relative_classifier(self):
        md = self._md([_fake_run("gpt2", strong=0.0, sep=0.9)])
        assert "cannot be met under cone-collapse" in md

    def test_axis_paragraph_follows_the_verdict(self):
        assert "IS the top principal component" in self._md(
            [_fake_run("gpt2", redundancy="pc1")])
        assert "worth a probe of its own" in self._md(
            [_fake_run("gpt2", redundancy="distinct")])
        assert "connectivity floor" in self._md(
            [_fake_run("gpt2", redundancy="degenerate")])
