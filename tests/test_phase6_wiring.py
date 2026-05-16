"""
tests/test_phase6_wiring.py

Tests for build_context ctx key correctness and registry prerequisite gating.

Three bugs under test:

  Bug W1 — qk_logit_matrices never computed
    build_context sets ctx["qk_logit_matrices"] = None unconditionally.
    head_classify requires it → always silently skipped.
    Fix: compute X @ M_h @ X.T per head when qk_matrices present.

  Bug W2 — layer_names vs event layer names mismatch
    ctx["layer_names"] = ["iter_0", "iter_1", ...] (ALBERT convention)
    events have "layer_name"/"layer_from" = "2", "3", ... (string integers)
    _classify_layer_types never finds a match → everything is "plateau".
    Fix: normalise layer name comparison (strip "iter_" prefix).

  Bug W3 — empty qk_matrices passes prerequisites_met
    ctx["qk_matrices"] = [] (not None) when WQ/WK absent from npz.
    SubexperimentSpec.prerequisites_met checks "is None" → empty list passes.
    run_qk_decompose receives empty input.
    Fix: coerce [] → None in build_context.

Run:
    pytest tests/test_phase6_wiring.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from p6_subspace.p6_io import SubexperimentSpec


# ---------------------------------------------------------------------------
# Helpers: minimal ctx factories
# ---------------------------------------------------------------------------

def _make_projectors(d: int = 32, n_layers: int = 1) -> dict:
    """Minimal projectors dict (ALBERT shared-weight layout)."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    P = Q[:, :4] @ Q[:, :4].T
    entry = {
        "P_S": P.astype(np.float32),
        "P_A": (np.eye(d) - P).astype(np.float32),
        "U_S": Q[:, :4].astype(np.float32),
        "U_A": Q[:, 4:8].astype(np.float32),
        "U_pos": Q[:, :2].astype(np.float32),
        "U_neg": Q[:, 2:4].astype(np.float32),
        "dim_S": 4, "dim_A": 4,
        "frac_S": 4/d, "frac_A": 4/d,
    }
    return {
        "is_per_layer":  False,
        "layer_names":   ["shared"] * n_layers,
        "d_model":       d,
        "per_layer":     [entry] * n_layers,
    }


def _make_base_ctx(
    n_layers: int = 4,
    n_tokens: int = 12,
    d: int = 32,
    n_heads: int = 4,
    with_qk: bool = True,
    with_attentions: bool = True,
) -> dict:
    """
    Minimal ctx that has all activation + weight keys populated correctly.
    Does NOT include model/tokenizer (Track C stays gated off).
    """
    rng = np.random.default_rng(1)

    # Activations
    acts = rng.standard_normal((n_layers, n_tokens, d)).astype(np.float32)
    acts /= np.linalg.norm(acts, axis=-1, keepdims=True) + 1e-8
    acts_per_layer = [acts[L] for L in range(n_layers)]

    # Labels
    labels_per_layer = [
        rng.integers(0, 2, size=n_tokens).astype(np.int32)
        for _ in range(n_layers)
    ]

    # Projectors
    projectors = _make_projectors(d=d, n_layers=1)

    # WO per head (d x d/n_heads)
    dh = d // n_heads
    wo_matrices = [rng.standard_normal((d, dh)).astype(np.float32) for _ in range(n_heads)]

    # QK per head
    qk_matrices = None
    if with_qk:
        qk_matrices = [
            (rng.standard_normal((d, dh)).astype(np.float32),
             rng.standard_normal((d, dh)).astype(np.float32))
            for _ in range(n_heads)
        ]

    # Attention matrices per head
    attn_matrices = None
    if with_attentions:
        raw = np.abs(rng.standard_normal((n_heads, n_tokens, n_tokens))).astype(np.float32)
        raw /= raw.sum(axis=-1, keepdims=True)
        attn_matrices = [raw[h] for h in range(n_heads)]

    # Token ids
    token_ids = rng.integers(0, 1000, size=n_tokens).astype(np.int64)

    # Layer names and type labels
    layer_names = [str(L) for L in range(n_layers)]
    layer_type_labels = ["plateau"] * n_layers

    ctx = {
        "model_name":            "albert-xlarge-v2",
        "stem":                  "albert_xlarge_v2",
        "projectors":            projectors,
        "layer_name":            "shared",
        "layer_idx":             0,
        "load_model":            False,
        "activations_per_layer": acts_per_layer,
        "labels_per_layer":      labels_per_layer,
        "layer_type_labels":     layer_type_labels,
        "layer_names":           layer_names,
        "token_ids":             token_ids,
        "tokens":                [f"t{i}" for i in range(n_tokens)],
        "token_activations":     acts[0],
        "wo_matrices":           wo_matrices,
        "qk_matrices":           qk_matrices,
        "attn_matrices":         attn_matrices,
        "merge_events":          [],
        "rot_energy_fracs":      None,
    }
    return ctx


# ============================================================================
# Bug W1 — qk_logit_matrices must be computed, not left as None
# ============================================================================

class TestQkLogitMatricesComputed(unittest.TestCase):
    """
    build_context sets ctx["qk_logit_matrices"] = None unconditionally.
    It must instead compute X @ (WQ_h @ WK_h^T) @ X^T per head.
    """

    def _apply_qk_logit_computation(self, ctx: dict) -> dict:
        """
        The computation that build_context should perform but currently doesn't.
        Extracted here so the test can call it directly and verify the contract.
        This function should be moved into build_context / a helper.
        """
        from run_6 import _compute_qk_logit_matrices
        return _compute_qk_logit_matrices(ctx)

    def test_qk_logit_matrices_not_none_when_qk_present(self):
        """When qk_matrices and token_activations exist, result must not be None."""
        ctx = _make_base_ctx(with_qk=True)
        ctx = self._apply_qk_logit_computation(ctx)
        self.assertIsNotNone(ctx.get("qk_logit_matrices"))

    def test_qk_logit_matrices_is_list_of_correct_length(self):
        n_heads = 4
        ctx = _make_base_ctx(n_heads=n_heads, with_qk=True)
        ctx = self._apply_qk_logit_computation(ctx)
        qkl = ctx["qk_logit_matrices"]
        self.assertIsInstance(qkl, list)
        self.assertEqual(len(qkl), n_heads)

    def test_each_logit_matrix_is_n_by_n(self):
        n_tokens, n_heads = 12, 4
        ctx = _make_base_ctx(n_tokens=n_tokens, n_heads=n_heads, with_qk=True)
        ctx = self._apply_qk_logit_computation(ctx)
        for h, mat in enumerate(ctx["qk_logit_matrices"]):
            self.assertEqual(mat.shape, (n_tokens, n_tokens), msg=f"head {h} wrong shape")

    def test_logit_matrix_equals_X_M_XT(self):
        """QK logit matrix must equal X @ (WQ @ WK^T) @ X^T."""
        ctx = _make_base_ctx(n_heads=2, d=16, n_tokens=8, with_qk=True)
        ctx = self._apply_qk_logit_computation(ctx)
        X = ctx["token_activations"]   # (n, d)
        for h, (WQ, WK) in enumerate(ctx["qk_matrices"]):
            M = WQ @ WK.T              # (d, d)
            expected = X @ M @ X.T    # (n, n)
            np.testing.assert_allclose(
                ctx["qk_logit_matrices"][h], expected, atol=1e-4,
                err_msg=f"head {h}: logit matrix mismatch",
            )

    def test_qk_logit_none_when_qk_matrices_absent(self):
        """When qk_matrices is None, qk_logit_matrices must remain None."""
        ctx = _make_base_ctx(with_qk=False)
        ctx["qk_matrices"] = None
        from run_6 import _compute_qk_logit_matrices
        ctx = _compute_qk_logit_matrices(ctx)
        self.assertIsNone(ctx.get("qk_logit_matrices"))

    def test_head_classify_prereqs_met_after_computation(self):
        """After computing qk_logit_matrices, head_classify prereqs must pass."""
        ctx = _make_base_ctx(with_qk=True, with_attentions=True)
        from run_6 import _compute_qk_logit_matrices
        ctx = _compute_qk_logit_matrices(ctx)
        spec = SubexperimentSpec(
            name="head_classify",
            run=lambda c: None,
            requires=["attn_matrices", "qk_logit_matrices", "token_activations"],
        )
        ok, reason = spec.prerequisites_met(ctx)
        self.assertTrue(ok, msg=f"head_classify still skipped: {reason}")


# ============================================================================
# Bug W2 — layer type classification: "iter_N" vs "N" name mismatch
# ============================================================================

class TestLayerTypeClassification(unittest.TestCase):
    """
    _classify_layer_types checks `if lname in merge_layers` where merge_layers
    contains string-integer layer names like "2", "3" (from events.json).
    ctx["layer_names"] is set to ["iter_0", "iter_1", ...] in build_context.
    "iter_2" is not in {"2"} → merge never detected → all labels are "plateau".

    Fix: normalise layer names before comparison (strip "iter_" prefix).
    """

    def _classify(self, layer_names, events, trajectories=None):
        from run_6 import _classify_layer_types
        return _classify_layer_types(layer_names, events, trajectories or [])

    def test_merge_detected_with_iter_prefix_names(self):
        """Merge event at layer 2 must produce label 'merge' for iter_2."""
        events = [{"type": "merge", "layer_name": "2", "layer_from": "2"}]
        layer_names = ["iter_0", "iter_1", "iter_2", "iter_3"]
        labels = self._classify(layer_names, events)
        self.assertEqual(labels[2], "merge", msg=f"Expected 'merge', got labels={labels}")

    def test_non_merge_layers_stay_plateau(self):
        events = [{"type": "merge", "layer_name": "2", "layer_from": "2"}]
        layer_names = ["iter_0", "iter_1", "iter_2", "iter_3"]
        labels = self._classify(layer_names, events)
        self.assertEqual(labels[0], "plateau")
        self.assertEqual(labels[1], "plateau")
        self.assertEqual(labels[3], "plateau")

    def test_merge_detected_with_plain_int_names(self):
        """Verify the plain-integer case still works (no regression)."""
        events = [{"type": "merge", "layer_name": "1", "layer_from": "1"}]
        layer_names = ["0", "1", "2"]
        labels = self._classify(layer_names, events)
        self.assertEqual(labels[1], "merge")

    def test_empty_events_all_plateau(self):
        labels = self._classify(["iter_0", "iter_1", "iter_2"], [])
        self.assertTrue(all(l == "plateau" for l in labels))

    def test_multiple_merge_layers(self):
        events = [
            {"type": "merge", "layer_name": "1", "layer_from": "1"},
            {"type": "merge", "layer_name": "3", "layer_from": "3"},
        ]
        layer_names = [f"iter_{i}" for i in range(5)]
        labels = self._classify(layer_names, events)
        self.assertEqual(labels[1], "merge")
        self.assertEqual(labels[3], "merge")
        self.assertEqual(labels[0], "plateau")
        self.assertEqual(labels[2], "plateau")
        self.assertEqual(labels[4], "plateau")

    def test_layer_type_labels_length_matches_layer_names(self):
        layer_names = [f"iter_{i}" for i in range(6)]
        events = [{"type": "merge", "layer_name": "2"}]
        labels = self._classify(layer_names, events)
        self.assertEqual(len(labels), len(layer_names))


# ============================================================================
# Bug W3 — empty qk_matrices passes prerequisites_met
# ============================================================================

class TestEmptyQkMatricesGating(unittest.TestCase):
    """
    prerequisites_met checks ctx.get(k) is None.
    An empty list [] is not None, so qk_decompose is not skipped even
    though there's nothing to decompose.

    Fix: normalise qk_matrices=[] → None in build_context so the gate fires.
    """

    def _spec_for(self, name, requires):
        return SubexperimentSpec(name=name, run=lambda c: None, requires=requires)

    def test_empty_list_currently_passes_prereq_gate(self):
        """
        Demonstrates the bug: [] is not None, so the spec proceeds.
        This test documents the current (broken) behaviour and will need
        to be updated once the fix is applied (or the spec itself is made
        to check truthiness).
        """
        ctx = {"qk_matrices": [], "token_ids": [1, 2], "token_activations": np.ones((2, 4))}
        spec = self._spec_for("qk_decompose", ["qk_matrices", "token_ids", "token_activations"])
        ok, _ = spec.prerequisites_met(ctx)
        # BUG: this should be False but currently passes
        # Once fixed, this assertion should become: self.assertFalse(ok)
        # For now we assert the current wrong behaviour to mark it:
        self.assertTrue(ok, msg="Bug W3 confirmed: empty list passes prereq gate")

    def test_none_qk_matrices_correctly_skips_qk_decompose(self):
        """After the fix, None must skip qk_decompose."""
        ctx = {"qk_matrices": None, "token_ids": [1, 2], "token_activations": np.ones((2, 4))}
        spec = self._spec_for("qk_decompose", ["qk_matrices", "token_ids", "token_activations"])
        ok, _ = spec.prerequisites_met(ctx)
        self.assertFalse(ok)

    def test_build_context_coerces_empty_qk_to_none(self):
        """
        build_context must normalise qk_matrices=[] to None so the prereq
        gate fires correctly.
        """
        from run_6 import _normalise_empty_lists
        ctx = {"qk_matrices": [], "wo_matrices": [np.ones((4, 4))]}
        ctx = _normalise_empty_lists(ctx)
        self.assertIsNone(ctx["qk_matrices"],
                          msg="Empty qk_matrices must be coerced to None")
        self.assertIsNotNone(ctx["wo_matrices"],
                             msg="Non-empty wo_matrices must not be coerced")

    def test_empty_wo_matrices_coerced_to_none(self):
        from run_6 import _normalise_empty_lists
        ctx = {"wo_matrices": [], "qk_matrices": None}
        ctx = _normalise_empty_lists(ctx)
        self.assertIsNone(ctx["wo_matrices"])

    def test_write_subspace_skipped_when_wo_empty(self):
        """write_subspace must not run when wo_matrices is coerced to None."""
        from run_6 import _normalise_empty_lists
        ctx = _make_base_ctx()
        ctx["wo_matrices"] = []
        ctx = _normalise_empty_lists(ctx)
        spec = self._spec_for("write_subspace", ["wo_matrices", "projectors"])
        ok, _ = spec.prerequisites_met(ctx)
        self.assertFalse(ok)


# ============================================================================
# Full ctx coverage: all non-model-dependent registry entries
# ============================================================================

class TestRegistryPrereqCoverage(unittest.TestCase):
    """
    With all three bugs fixed, a ctx built from synthetic data should have
    prerequisites_met return True for every Track A and B/D sub-experiment.
    Track C dissociation stays gated off (load_model=False).
    """

    def setUp(self):
        from run_6 import _compute_qk_logit_matrices, _normalise_empty_lists
        ctx = _make_base_ctx(with_qk=True, with_attentions=True)
        ctx = _compute_qk_logit_matrices(ctx)
        ctx = _normalise_empty_lists(ctx)
        self.ctx = ctx

    def _spec(self, name, requires, applicable=None):
        return SubexperimentSpec(name=name, run=lambda c: None,
                                 requires=requires, applicable=applicable)

    def test_head_classify_prereqs_met(self):
        spec = self._spec("head_classify",
                          ["attn_matrices", "qk_logit_matrices", "token_activations"])
        ok, r = spec.prerequisites_met(self.ctx)
        self.assertTrue(ok, msg=r)

    def test_qk_decompose_prereqs_met(self):
        spec = self._spec("qk_decompose",
                          ["qk_matrices", "token_ids", "token_activations"])
        ok, r = spec.prerequisites_met(self.ctx)
        self.assertTrue(ok, msg=r)

    def test_induction_ov_prereqs_met(self):
        spec = self._spec("induction_ov",
                          ["attn_matrices", "wo_matrices", "token_ids",
                           "token_activations", "projectors"])
        ok, r = spec.prerequisites_met(self.ctx)
        self.assertTrue(ok, msg=r)

    def test_eigenspace_degeneracy_prereqs_met(self):
        spec = self._spec("eigenspace_degeneracy",
                          ["activations_per_layer", "labels_per_layer",
                           "layer_type_labels", "projectors"])
        ok, r = spec.prerequisites_met(self.ctx)
        self.assertTrue(ok, msg=r)

    def test_centroid_velocity_prereqs_met(self):
        spec = self._spec("centroid_velocity",
                          ["activations_per_layer", "labels_per_layer",
                           "layer_type_labels", "projectors"])
        ok, r = spec.prerequisites_met(self.ctx)
        self.assertTrue(ok, msg=r)

    def test_local_contraction_prereqs_met(self):
        spec = self._spec("local_contraction",
                          ["activations_per_layer", "labels_per_layer", "layer_type_labels"])
        ok, r = spec.prerequisites_met(self.ctx)
        self.assertTrue(ok, msg=r)

    def test_probe_subspace_prereqs_met(self):
        spec = self._spec("probe_subspace",
                          ["activations_per_layer", "labels_per_layer",
                           "layer_type_labels", "projectors"])
        ok, r = spec.prerequisites_met(self.ctx)
        self.assertTrue(ok, msg=r)

    def test_write_subspace_prereqs_met(self):
        spec = self._spec("write_subspace", ["wo_matrices", "projectors"])
        ok, r = spec.prerequisites_met(self.ctx)
        self.assertTrue(ok, msg=r)

    def test_dissociation_gated_off_without_load_model(self):
        """dissociation must not run when load_model=False."""
        spec = self._spec(
            "dissociation",
            ["model", "tokenizer", "text", "token_ids", "projectors", "hook_targets"],
            applicable=lambda c: c.get("load_model", False),
        )
        ok, reason = spec.prerequisites_met(self.ctx)
        self.assertFalse(ok, msg="dissociation must be gated off without --load-model")
        self.assertIn("not applicable", reason)


if __name__ == "__main__":
    unittest.main()
