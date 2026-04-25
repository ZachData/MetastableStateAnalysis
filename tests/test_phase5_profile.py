"""
tests/test_phase5_profile.py
Analytically-known-case tests for Phase 5 profile modules.
Groups A (cluster_profile), B (v_alignment), D (feature_signature), G (sibling_contrast).

No model required. All cases have closed-form expected values.

Geometry:
  n_layers=6, n_tokens=20, d=16, n_features=10
"""
import sys
import unittest
import numpy as np

sys.path.insert(0, "/home/claude")

from p5_single_mstate_analysis.cluster_profile import compute_profile
from p5_single_mstate_analysis.v_alignment import (
    centroid_subspace_trajectory,
    displacement_subspace_trajectory,
    _split_vector,
)
from p5_single_mstate_analysis.feature_signature import rank_identity_features
from p5_single_mstate_analysis.sibling_contrast import run_sibling_contrast

# ---------------------------------------------------------------------------
# Shared geometry constants
# ---------------------------------------------------------------------------
N_LAYERS   = 6
N_TOKENS   = 20
D          = 16
N_FEATURES = 10
HALF       = N_TOKENS // 2   # cluster size = 10


def _unit_vec(idx: int) -> np.ndarray:
    v = np.zeros(D)
    v[idx] = 1.0
    return v


def _tokens():
    return [f"t{i}" for i in range(N_TOKENS)]


def _empty_metrics():
    return {"layers": []}


def _trajectory(tid: int, cluster_id: int, n_layers: int = N_LAYERS) -> dict:
    return {
        "id":          tid,
        "chain":       [(l, cluster_id) for l in range(n_layers)],
        "start_layer": 0,
        "end_layer":   n_layers - 1,
        "lifespan":    n_layers,
    }


# ---------------------------------------------------------------------------
# Group A — compute_profile: all tokens identical unit vectors in one cluster
#
# Analytically:
#   intra_ip  = 1.0   (all pairs have dot product 1)
#   radius    = 0.0   (all tokens equal the centroid)
#   jaccard   = 1.0   (same tokens at every layer)
# ---------------------------------------------------------------------------

class TestProfileIdenticalUnitVectors(unittest.TestCase):
    """Single cluster, all tokens = e_0 across all layers."""

    def setUp(self):
        e0 = _unit_vec(0)
        self.acts    = np.tile(e0, (N_LAYERS, N_TOKENS, 1))
        self.labels  = [np.zeros(N_TOKENS, dtype=int)] * N_LAYERS
        self.traj    = _trajectory(0, cluster_id=0)
        self.profile = compute_profile(
            self.acts, self.labels,
            self.traj, None,
            _tokens(), _empty_metrics(),
        )

    def test_ip_mean_all_layers(self):
        for row in self.profile["per_layer"]:
            self.assertAlmostEqual(row["ip_mean"], 1.0, places=6,
                msg=f"Layer {row['layer']}: ip_mean={row['ip_mean']}")

    def test_radius_all_layers(self):
        for row in self.profile["per_layer"]:
            self.assertLess(row["radius"], 1e-6,
                msg=f"Layer {row['layer']}: radius={row['radius']}")

    def test_jaccard_stability_all_one(self):
        for j in self.profile["membership_jaccard"]:
            self.assertAlmostEqual(j, 1.0, places=6, msg=f"Jaccard={j}")

    def test_summary_mean_ip(self):
        self.assertAlmostEqual(self.profile["summary"]["mean_ip_mean"], 1.0, places=6)

    def test_summary_mean_radius(self):
        self.assertLess(self.profile["summary"]["mean_radius"], 1e-6)

    def test_jaccard_count(self):
        self.assertEqual(len(self.profile["membership_jaccard"]), N_LAYERS - 1)

    def test_cumulative_arc_starts_zero(self):
        self.assertAlmostEqual(self.profile["cumulative_arc"][0], 0.0)

    def test_per_layer_count(self):
        self.assertEqual(len(self.profile["per_layer"]), N_LAYERS)


# ---------------------------------------------------------------------------
# Group A — antipodal clusters, centroid rotates each layer
#
# Setup: cluster 0 tokens all point toward e_l at layer l.
# Angular step between e_l and e_{l+1} = pi/2 radians.
# ---------------------------------------------------------------------------

class TestProfileRotatingCentroid(unittest.TestCase):
    """Cluster-0 centroid rotates 90 deg per layer => all angular steps = pi/2."""

    def setUp(self):
        acts   = np.zeros((N_LAYERS, N_TOKENS, D))
        labels = []
        for l in range(N_LAYERS):
            acts[l, :HALF] = _unit_vec(l)
            acts[l, HALF:] = _unit_vec((l + 1) % D)
            lbl = np.ones(N_TOKENS, dtype=int)
            lbl[:HALF] = 0
            labels.append(lbl)
        self.acts    = acts
        self.labels  = labels
        self.traj    = _trajectory(0, cluster_id=0)
        self.profile = compute_profile(
            self.acts, self.labels, self.traj, None, _tokens(), _empty_metrics(),
        )

    def test_angular_steps_nonzero(self):
        steps = self.profile["centroid_angular_steps"]
        self.assertEqual(len(steps), N_LAYERS - 1)
        for s in steps:
            self.assertGreater(s, 0.1, f"Expected non-zero step, got {s}")

    def test_angular_steps_are_pi_over_2(self):
        for s in self.profile["centroid_angular_steps"]:
            self.assertAlmostEqual(s, np.pi / 2, places=3,
                msg=f"Step {s} != pi/2")

    def test_total_arc_value(self):
        expected = (N_LAYERS - 1) * np.pi / 2
        arc = self.profile["summary"]["total_arc_length"]
        self.assertAlmostEqual(arc, expected, places=2)

    def test_cumulative_arc_monotone(self):
        arc = self.profile["cumulative_arc"]
        for i in range(1, len(arc)):
            self.assertGreaterEqual(arc[i], arc[i - 1])


# ---------------------------------------------------------------------------
# Group A — symmetric clusters: profiles agree on compactness metrics
# ---------------------------------------------------------------------------

class TestProfileSymmetry(unittest.TestCase):

    def setUp(self):
        e0 = _unit_vec(0)
        e1 = _unit_vec(1)
        acts   = np.zeros((N_LAYERS, N_TOKENS, D))
        labels = []
        for l in range(N_LAYERS):
            acts[l, :HALF] = e0
            acts[l, HALF:] = e1
            lbl = np.ones(N_TOKENS, dtype=int)
            lbl[:HALF] = 0
            labels.append(lbl)
        traj_p = _trajectory(0, cluster_id=0)
        traj_s = _trajectory(1, cluster_id=1)
        self.prof_p = compute_profile(acts, labels, traj_p, traj_s, _tokens(), _empty_metrics())
        self.prof_s = compute_profile(acts, labels, traj_s, traj_p, _tokens(), _empty_metrics())

    def test_mean_ip_match(self):
        self.assertAlmostEqual(
            self.prof_p["summary"]["mean_ip_mean"],
            self.prof_s["summary"]["mean_ip_mean"], places=6)

    def test_mean_radius_match(self):
        self.assertAlmostEqual(
            self.prof_p["summary"]["mean_radius"],
            self.prof_s["summary"]["mean_radius"], places=6)

    def test_jaccard_match(self):
        for jp, js in zip(self.prof_p["membership_jaccard"],
                          self.prof_s["membership_jaccard"]):
            self.assertAlmostEqual(jp, js, places=6)


# ---------------------------------------------------------------------------
# Group B — V-alignment subspace decomposition
#
# U_att = e_0,  U_rep = e_1.
# v = e_0  =>  attr_frac=1, rep_frac=0, orth_frac=0
# v = e_1  =>  attr_frac=0, rep_frac=1
# v = e_2  =>  attr_frac=0, rep_frac=0, orth_frac=1
# ---------------------------------------------------------------------------

class TestVAlignmentSubspaceDecomposition(unittest.TestCase):

    def setUp(self):
        self.U_att = _unit_vec(0).reshape(D, 1)
        self.U_rep = _unit_vec(1).reshape(D, 1)

    def test_fully_attractive_vector(self):
        d = _split_vector(_unit_vec(0), self.U_att, self.U_rep)
        self.assertAlmostEqual(d["attr_frac"], 1.0, places=6)
        self.assertLess(d["rep_frac"],  1e-6)
        self.assertLess(d["orth_frac"], 1e-6)

    def test_fully_repulsive_vector(self):
        d = _split_vector(_unit_vec(1), self.U_att, self.U_rep)
        self.assertLess(d["attr_frac"], 1e-6)
        self.assertAlmostEqual(d["rep_frac"], 1.0, places=6)

    def test_orthogonal_vector(self):
        d = _split_vector(_unit_vec(2), self.U_att, self.U_rep)
        self.assertLess(d["attr_frac"], 1e-6)
        self.assertLess(d["rep_frac"],  1e-6)
        self.assertAlmostEqual(d["orth_frac"], 1.0, places=6)

    def test_fracs_sum_to_one_for_all_basis_vectors(self):
        for idx in range(min(D, 8)):
            d = _split_vector(_unit_vec(idx), self.U_att, self.U_rep)
            total = d["attr_frac"] + d["rep_frac"] + d["orth_frac"]
            self.assertAlmostEqual(total, 1.0, places=5,
                msg=f"Fracs sum = {total} for e_{idx}")

    def test_centroid_trajectory_all_attractive(self):
        centroids = np.tile(_unit_vec(0), (N_LAYERS, 1))
        out = centroid_subspace_trajectory(centroids, self.U_att, self.U_rep)
        self.assertEqual(len(out), N_LAYERS)
        for step in out:
            self.assertAlmostEqual(step["attr_frac"], 1.0, places=6)
            self.assertLess(step["rep_frac"], 1e-6)

    def test_displacement_trajectory_length(self):
        centroids = np.tile(_unit_vec(0), (N_LAYERS, 1))
        out = displacement_subspace_trajectory(centroids, self.U_att, self.U_rep)
        self.assertEqual(len(out), N_LAYERS - 1)

    def test_zero_displacement_fracs_are_zero(self):
        """Identical centroids => displacement = 0 => total_sq = 0, all fracs = 0."""
        centroids = np.tile(_unit_vec(0), (N_LAYERS, 1))
        out = displacement_subspace_trajectory(centroids, self.U_att, self.U_rep)
        for step in out:
            self.assertEqual(step["total_sq"], 0.0)
            self.assertEqual(step["attr_frac"], 0.0)


# ---------------------------------------------------------------------------
# Group D — feature_signature: one perfectly discriminating feature
#
# feature 0 = 2.0 for cluster tokens, 0.0 outside => max MI, ranked first.
# ---------------------------------------------------------------------------

class TestFeaturePerfectDiscriminator(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.acts = rng.uniform(0, 0.5, (N_TOKENS, N_FEATURES)).astype(np.float32)
        self.mask = np.zeros(N_TOKENS, dtype=bool)
        self.mask[:HALF] = True
        self.acts[self.mask,  0] = 2.0
        self.acts[~self.mask, 0] = 0.0
        self.ranked = rank_identity_features(self.acts, self.mask, top_k=N_FEATURES)

    def test_perfect_feature_is_first(self):
        self.assertEqual(self.ranked[0]["feature"], 0,
            f"Expected feature 0 first, got {self.ranked[0]['feature']}")

    def test_mi_monotone_non_increasing(self):
        mis = [r["mi_bits"] for r in self.ranked]
        for i in range(len(mis) - 1):
            self.assertGreaterEqual(mis[i], mis[i + 1])

    def test_active_rate_in_cluster_is_one(self):
        f0 = next(r for r in self.ranked if r["feature"] == 0)
        self.assertAlmostEqual(f0["active_rate_in_cluster"], 1.0, places=6)

    def test_active_rate_out_is_zero(self):
        f0 = next(r for r in self.ranked if r["feature"] == 0)
        self.assertLess(f0["active_rate_out"], 1e-6)

    def test_returns_top_k_entries(self):
        self.assertEqual(len(self.ranked), N_FEATURES)

    def test_mi_nonnegative(self):
        for r in self.ranked:
            self.assertGreaterEqual(r["mi_bits"], 0.0)


class TestFeatureNoDiscriminator(unittest.TestCase):
    """All features zero => MI = 0 for all."""

    def test_all_zero_features(self):
        acts = np.zeros((N_TOKENS, N_FEATURES), dtype=np.float32)
        mask = np.zeros(N_TOKENS, dtype=bool)
        mask[:HALF] = True
        ranked = rank_identity_features(acts, mask, top_k=N_FEATURES)
        for r in ranked:
            self.assertEqual(r["mi_bits"], 0.0)


# ---------------------------------------------------------------------------
# Group G — sibling_contrast
# Two perfectly compact clusters: sibling ip_mean should = 1.0
# ---------------------------------------------------------------------------

class TestSiblingContrastStructure(unittest.TestCase):

    def setUp(self):
        e0 = _unit_vec(0)
        e1 = _unit_vec(1)
        acts   = np.zeros((N_LAYERS, N_TOKENS, D))
        labels = []
        for l in range(N_LAYERS):
            acts[l, :HALF] = e0
            acts[l, HALF:] = e1
            lbl = np.ones(N_TOKENS, dtype=int)
            lbl[:HALF] = 0
            labels.append(lbl)
        primary = _trajectory(0, cluster_id=0)
        sibling = _trajectory(1, cluster_id=1)
        self.result = run_sibling_contrast(
            acts, None, labels, primary, sibling, _tokens(), _empty_metrics(),
        )

    def test_top_level_keys(self):
        for key in ("sibling_profile", "contrast_summary"):
            self.assertIn(key, self.result)

    def test_sibling_profile_no_error(self):
        self.assertNotIn("error", self.result["sibling_profile"])

    def test_contrast_summary_keys(self):
        cs = self.result["contrast_summary"]
        for key in ("sibling_mean_ip", "random_mean_ip",
                    "sibling_mean_silhouette_all", "random_mean_silhouette_all"):
            self.assertIn(key, cs, f"Missing key '{key}' in contrast_summary")

    def test_sibling_perfectly_compact(self):
        sib_ip = self.result["contrast_summary"]["sibling_mean_ip"]
        self.assertIsNotNone(sib_ip)
        self.assertAlmostEqual(sib_ip, 1.0, places=5)

    def test_random_control_profile_generated(self):
        self.assertIn("random_control_profile", self.result)

    def test_sibling_trajectory_id(self):
        self.assertEqual(self.result["sibling_profile"]["trajectory_id"], 1)


class TestSiblingContrastNullSibling(unittest.TestCase):
    """sibling_trajectory=None: contrast runs, sibling slots are None."""

    def test_no_sibling(self):
        e0 = _unit_vec(0)
        acts   = np.tile(e0, (N_LAYERS, N_TOKENS, 1))
        labels = [np.zeros(N_TOKENS, dtype=int)] * N_LAYERS
        result = run_sibling_contrast(
            acts, None, labels,
            _trajectory(0, cluster_id=0), None,
            _tokens(), _empty_metrics(),
        )
        self.assertIn("contrast_summary", result)
        self.assertIsNone(result["contrast_summary"].get("sibling_mean_ip"))


# new code

# ---------------------------------------------------------------------------
# Group A — varying membership: Jaccard < 1.0
#
# Layer 0: cluster = tokens 0..9   (e_0)
# Layer 1: cluster = tokens 5..14  (e_0) — 5 tokens enter, 5 leave
# Intersection = tokens 5..9  (5), Union = tokens 0..14 (15)
# Expected Jaccard at layer boundary = 5/15 ≈ 0.333
# ---------------------------------------------------------------------------

class TestProfileVaryingMembership(unittest.TestCase):
    """Cluster membership shifts across layers → Jaccard < 1."""

    def setUp(self):
        e0 = _unit_vec(0)
        acts   = np.tile(e0, (N_LAYERS, N_TOKENS, 1))
        labels = []
        for l in range(N_LAYERS):
            lbl = np.full(N_TOKENS, -1, dtype=int)  # -1 = not in cluster
            start = l                                 # membership shifts each layer
            lbl[start : start + HALF] = 0
            labels.append(lbl)
        traj = _trajectory(0, cluster_id=0)
        self.prof = compute_profile(acts, labels, traj, None, _tokens(), _empty_metrics())

    def test_jaccard_below_one(self):
        """At least one consecutive-layer Jaccard must be < 1.0."""
        jaccards = self.prof["membership_jaccard"]
        self.assertTrue(
            any(j < 0.999 for j in jaccards),
            f"All Jaccards are 1.0; membership shift was not detected: {jaccards}",
        )

    def test_jaccard_layer0_to_layer1(self):
        # Layer 0 members: {0..9}, Layer 1 members: {1..10}
        # Intersection = {1..9} = 9, Union = {0..10} = 11 → J = 9/11
        expected = 9 / 11
        self.assertAlmostEqual(self.prof["membership_jaccard"][0], expected, places=4)

    def test_per_layer_count_correct(self):
        for layer_row in self.prof["per_layer"]:
            self.assertEqual(layer_row["n_tokens"], HALF)


# ---------------------------------------------------------------------------
# Group A — silhouette with two perfectly separated clusters
#
# e_0 cluster vs e_1 cluster: inter-cluster IP = 0, intra = 1.
# For any member of cluster 0:
#   a (mean intra distance) = 0,  b (mean inter distance) = sqrt(2)  → sil = 1.0
# Similarly for all-clusters silhouette if only 2 clusters exist.
# ---------------------------------------------------------------------------

class TestProfileSilhouette(unittest.TestCase):
    """Silhouette = 1.0 for two perfectly orthogonal clusters."""

    def setUp(self):
        e0, e1 = _unit_vec(0), _unit_vec(1)
        acts   = np.zeros((N_LAYERS, N_TOKENS, D))
        labels = []
        for l in range(N_LAYERS):
            acts[l, :HALF] = e0
            acts[l, HALF:] = e1
            lbl = np.ones(N_TOKENS, dtype=int)
            lbl[:HALF] = 0
            labels.append(lbl)
        traj_p = _trajectory(0, cluster_id=0)
        traj_s = _trajectory(1, cluster_id=1)
        self.prof = compute_profile(acts, labels, traj_p, traj_s, _tokens(), _empty_metrics())

    def test_silhouette_vs_sibling_is_one(self):
        for row in self.prof["per_layer"]:
            sil = row.get("silhouette_vs_sibling")
            if sil is not None:
                self.assertAlmostEqual(sil, 1.0, places=4,
                    msg=f"Layer {row['layer']}: silhouette_vs_sibling={sil}")

    def test_silhouette_vs_all_is_one(self):
        for row in self.prof["per_layer"]:
            sil = row.get("silhouette_vs_all")
            if sil is not None:
                self.assertAlmostEqual(sil, 1.0, places=4,
                    msg=f"Layer {row['layer']}: silhouette_vs_all={sil}")

    def test_summary_mean_silhouette_sibling(self):
        s = self.prof["summary"].get("mean_silhouette_sibling")
        if s is not None:
            self.assertAlmostEqual(s, 1.0, places=4)


# ---------------------------------------------------------------------------
# Group B — non-zero displacement with known fractions
#
# Layer 0 centroid = e_0 (fully attractive)
# Layer 1 centroid = e_1 (fully repulsive)
# Displacement = e_1 - e_0 = [-1, 1, 0, ...]
# U_att = e_0, U_rep = e_1
# proj onto attr: (-1)^2 / |displacement|^2 = 1/2 → attr_frac = 0.5
# proj onto rep :  (1)^2 / |displacement|^2 = 1/2 → rep_frac  = 0.5
# orth_frac = 0.0
# ---------------------------------------------------------------------------

class TestDisplacementNonZero(unittest.TestCase):
    """Known displacement between attr and rep axes → fracs = 0.5 each."""

    def setUp(self):
        self.U_att = _unit_vec(0).reshape(D, 1)
        self.U_rep = _unit_vec(1).reshape(D, 1)
        centroids  = np.array([_unit_vec(0), _unit_vec(1)])  # 2 layers
        self.out   = displacement_subspace_trajectory(centroids, self.U_att, self.U_rep)

    def test_one_step_produced(self):
        self.assertEqual(len(self.out), 1)

    def test_attr_frac(self):
        self.assertAlmostEqual(self.out[0]["attr_frac"], 0.5, places=6)

    def test_rep_frac(self):
        self.assertAlmostEqual(self.out[0]["rep_frac"], 0.5, places=6)

    def test_orth_frac(self):
        self.assertAlmostEqual(self.out[0]["orth_frac"], 0.0, places=6)

    def test_total_sq_correct(self):
        # |e_1 - e_0|^2 = 2
        self.assertAlmostEqual(self.out[0]["total_sq"], 2.0, places=6)


# ---------------------------------------------------------------------------
# Group B — compute_v_alignment full pipeline smoke test
#
# Uses a trivial projector dict so no Phase 2 files are needed.
# Verifies the output schema; does not check numerics beyond non-null.
# ---------------------------------------------------------------------------

from p5_single_mstate_analysis.v_alignment import compute_v_alignment

class TestComputeVAlignmentSmoke(unittest.TestCase):
    """compute_v_alignment returns expected top-level keys for valid input."""

    def setUp(self):
        e0  = _unit_vec(0)
        # Centroid trajectory: all layers pointing at e_0
        centroids = np.tile(e0, (N_LAYERS, 1))
        v_proj    = {
            "U_attractive": _unit_vec(0).reshape(D, 1),
            "U_repulsive":  _unit_vec(1).reshape(D, 1),
            "path": "synthetic",
        }
        traj = _trajectory(0, cluster_id=0)
        self.result = compute_v_alignment(centroids, v_proj, traj)

    def test_top_level_keys(self):
        for key in ("centroid_trajectory", "displacement_trajectory", "summary"):
            self.assertIn(key, self.result, f"Missing key: {key}")

    def test_centroid_trajectory_length(self):
        self.assertEqual(len(self.result["centroid_trajectory"]), N_LAYERS)

    def test_displacement_trajectory_length(self):
        self.assertEqual(len(self.result["displacement_trajectory"]), N_LAYERS - 1)

    def test_summary_mean_attr_frac_is_one(self):
        # All centroids = e_0 = fully attractive → mean attr_frac = 1.0
        frac = self.result["summary"].get("mean_attr_frac")
        self.assertIsNotNone(frac)
        self.assertAlmostEqual(frac, 1.0, places=5)


# ---------------------------------------------------------------------------
# Group D — perfectly discriminating feature ranks first with MI = 1 bit
#
# feature 0: 2.0 for cluster tokens (0..HALF-1), 0.0 outside  → H(Y|X=0)=0, H(Y)=1
# feature 1: identical activation for both groups (1.0 everywhere) → MI=0
# Expected: feature 0 rank=0, MI≈1.0; feature 1 rank=1, MI=0.0
# ---------------------------------------------------------------------------

class TestRankIdentityFeaturesPositive(unittest.TestCase):
    """One perfectly discriminating feature should rank first with MI ~ 1 bit."""

    def setUp(self):
        acts = np.ones((N_TOKENS, N_FEATURES), dtype=np.float32)
        # Feature 0 discriminates perfectly
        acts[:HALF,  0] = 2.0
        acts[HALF:,  0] = 0.0
        mask = np.zeros(N_TOKENS, dtype=bool)
        mask[:HALF]     = True
        self.ranked = rank_identity_features(acts, mask, top_k=N_FEATURES)

    def test_top_feature_is_feature_zero(self):
        self.assertEqual(self.ranked[0]["feature_idx"], 0)

    def test_top_feature_mi_near_one_bit(self):
        self.assertGreater(self.ranked[0]["mi_bits"], 0.9)

    def test_uniform_features_have_zero_mi(self):
        # Features 1..N_FEATURES-1 are all 1.0 everywhere → MI = 0
        mi_vals = [r["mi_bits"] for r in self.ranked if r["feature_idx"] != 0]
        for mi in mi_vals:
            self.assertAlmostEqual(mi, 0.0, places=4)


class TestRankIdentityFeaturesOrdering(unittest.TestCase):
    """Features with graded separation should be returned in descending MI order."""

    def setUp(self):
        rng  = np.random.default_rng(42)
        acts = np.zeros((N_TOKENS, N_FEATURES), dtype=np.float32)
        mask = np.zeros(N_TOKENS, dtype=bool)
        mask[:HALF] = True
        # Feature i has separation magnitude (i+1); more separation → higher MI
        for i in range(N_FEATURES):
            acts[:HALF, i] = float(i + 1)
            acts[HALF:, i] = 0.0
        self.ranked = rank_identity_features(acts, mask, top_k=N_FEATURES)

    def test_mi_is_non_increasing(self):
        mis = [r["mi_bits"] for r in self.ranked]
        for i in range(len(mis) - 1):
            self.assertGreaterEqual(mis[i], mis[i + 1],
                msg=f"MI not sorted at position {i}: {mis[i]:.4f} < {mis[i+1]:.4f}")

    def test_top_feature_has_largest_separation(self):
        # Feature N_FEATURES-1 has the largest separation → should rank first
        self.assertEqual(self.ranked[0]["feature_idx"], N_FEATURES - 1)


# ---------------------------------------------------------------------------
# Group G — random control ip_mean < real cluster ip_mean
#
# Real cluster: all tokens = e_0 → ip_mean = 1.0
# Random control: mix of e_0 and e_1 → ip_mean < 1.0
# ---------------------------------------------------------------------------

class TestSiblingRandomControlWeaker(unittest.TestCase):
    """Random control must have strictly lower ip_mean than the real cluster."""

    def setUp(self):
        e0, e1 = _unit_vec(0), _unit_vec(1)
        acts   = np.zeros((N_LAYERS, N_TOKENS, D))
        labels = []
        for l in range(N_LAYERS):
            acts[l, :HALF] = e0
            acts[l, HALF:] = e1
            lbl = np.ones(N_TOKENS, dtype=int)
            lbl[:HALF] = 0
            labels.append(lbl)
        self.result = run_sibling_contrast(
            acts, None, labels,
            _trajectory(0, cluster_id=0),
            _trajectory(1, cluster_id=1),
            _tokens(), _empty_metrics(),
        )

    def test_random_ip_less_than_real(self):
        cs          = self.result["contrast_summary"]
        real_ip     = cs["sibling_mean_ip"]     # sibling is also compact → 1.0
        random_ip   = cs.get("random_mean_ip")
        self.assertIsNotNone(random_ip)
        # Real (or sibling) cluster is compact; random subset mixes e_0 and e_1
        # so random ip_mean must be strictly below 1.0
        self.assertLess(random_ip, real_ip,
            msg=f"random_mean_ip={random_ip:.4f} >= sibling_mean_ip={real_ip:.4f}")

    def test_random_ip_below_one(self):
        random_ip = self.result["contrast_summary"].get("random_mean_ip")
        self.assertIsNotNone(random_ip)
        self.assertLess(random_ip, 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
