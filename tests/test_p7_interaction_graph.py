"""
tests/test_p7_interaction_graph.py — the Phase 7 producer.

Oracle-tier where it can be: the force decomposition has an exact
known-correct answer on constructed operators, and the magnitude identity
||A_ij * (x_j @ OV)|| = A_ij * ||x_j @ OV|| is checkable against the
brute-force tensor on small inputs. That identity is what lets the producer
skip building an (n^2, d) array, so it gets a direct test rather than being
trusted.

Also covered: the frame guard (the error that is invisible in the shapes),
causal masking, and per-target selection.
"""
from __future__ import annotations

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core.interactions import InteractionTable
from p7_motifs.interaction_graph import (
    build_head_edges,
    edge_force_magnitudes,
    select_edges,
)


def _softmax_causal(logits):
    n = logits.shape[0]
    out = np.zeros_like(logits)
    for i in range(n):
        row = logits[i, : i + 1]
        e = np.exp(row - row.max())
        out[i, : i + 1] = e / e.sum()
    return out


def _setup(n=6, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    A = _softmax_causal(rng.standard_normal((n, n)))
    OV = rng.standard_normal((d, d))
    return X, A, OV


class TestMagnitudeIdentity:
    """The optimization the whole producer rests on."""

    def test_matches_the_brute_force_tensor(self):
        X, A, OV = _setup()
        n, d = X.shape
        mags, moved = edge_force_magnitudes(X, A, OV, causal=True)

        brute = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                brute[i, j] = np.linalg.norm(A[i, j] * (X[j] @ OV))
        assert np.allclose(mags, brute)

    def test_moved_is_the_reusable_push_direction(self):
        X, A, OV = _setup()
        _, moved = edge_force_magnitudes(X, A, OV)
        assert np.allclose(moved, X @ OV)

    def test_causal_mask_removes_the_upper_triangle(self):
        X, A, OV = _setup()
        mags, _ = edge_force_magnitudes(X, A, OV, causal=True)
        assert np.allclose(np.triu(mags, k=1), 0.0)

    def test_non_causal_keeps_both_triangles(self):
        X, A, OV = _setup()
        A_full = np.abs(np.random.default_rng(1).standard_normal((6, 6)))
        mags, _ = edge_force_magnitudes(X, A_full, OV, causal=False)
        assert np.any(np.triu(mags, k=1) > 0)

    def test_zero_ov_moves_nothing(self):
        X, A, _ = _setup()
        mags, _ = edge_force_magnitudes(X, A, np.zeros((4, 4)))
        assert np.allclose(mags, 0.0)

    def test_attention_shape_mismatch_raises(self):
        X, _, OV = _setup()
        with pytest.raises(ValueError, match="expected"):
            edge_force_magnitudes(X, np.eye(3), OV)

    def test_wrong_ov_shape_names_the_likely_cause(self):
        """W_O alone instead of the composed W_V @ W_O is the specific
        mistake p6's induction_ov.py documented; the error says so."""
        X, A, _ = _setup()
        with pytest.raises(ValueError, match="composed"):
            edge_force_magnitudes(X, A, np.zeros((4, 7)))


class TestSelection:

    def test_top_k_is_applied_per_target_not_globally(self):
        """A global top-k lets a few high-norm particles eat the budget and
        leaves others with no incoming edges — which reads as 'not moved'
        rather than 'not measured'."""
        mags = np.tril(np.ones((5, 5)))
        mags[4, :] *= 100.0                      # one dominant target row
        targets, sources, _ = select_edges(mags, top_k_per_target=2)
        counts = np.bincount(targets, minlength=5)
        # every target with candidates keeps up to 2, none is starved
        assert counts[1] == 2 and counts[4] == 2
        assert counts[0] == 1                    # only one causal candidate

    def test_keeps_the_largest_edges(self):
        mags = np.zeros((2, 2))
        mags[1, 0], mags[1, 1] = 0.1, 0.9
        targets, sources, _ = select_edges(mags, top_k_per_target=1)
        assert sources[targets == 1].tolist() == [1]

    def test_none_retains_everything_above_threshold(self):
        mags = np.tril(np.ones((4, 4)))
        targets, _, ret = select_edges(mags, top_k_per_target=None)
        assert len(targets) == 10                # full lower triangle
        assert ret["mode"] == "threshold_only"

    def test_min_magnitude_drops_negligible_edges(self):
        mags = np.tril(np.full((3, 3), 0.01))
        mags[2, 0] = 5.0
        targets, _, _ = select_edges(mags, top_k_per_target=None, min_magnitude=1.0)
        assert len(targets) == 1

    def test_retention_is_recorded_as_placed(self):
        _, _, ret = select_edges(np.tril(np.ones((3, 3))), top_k_per_target=2)
        assert ret["status"] == "placed"
        assert ret["k"] == 2

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            select_edges(np.ones((2, 2)), top_k_per_target=0)


class TestFrameGuard:
    """The error that is invisible in the shapes."""

    def test_normalized_activations_declared_raw_are_refused(self):
        X, A, OV = _setup()
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        with pytest.raises(ValueError, match="frame mismatch"):
            build_head_edges("m", "p", 0, 0, X_norm, A, OV, declared_frame="raw")

    def test_raw_activations_declared_normalized_are_refused(self):
        X, A, OV = _setup()
        with pytest.raises(ValueError, match="frame mismatch"):
            build_head_edges("m", "p", 0, 0, X, A, OV, declared_frame="l2_sphere")

    def test_matching_declaration_proceeds_and_is_stamped(self):
        X, A, OV = _setup()
        t = build_head_edges("m", "p", 0, 0, X, A, OV, declared_frame="raw")
        assert t.retention["frame"] == "raw"


class TestBuildHeadEdges:

    def test_force_vectors_match_the_definition_for_retained_edges(self):
        """Oracle: f_ij = A_ij * (x_j @ OV), exactly."""
        X, A, OV = _setup()
        t = build_head_edges("m", "p", 2, 3, X, A, OV, top_k_per_target=None)
        for row in range(len(t)):
            i = int(t.columns["target"][row])
            j = int(t.columns["source"][row])
            expected = np.linalg.norm(A[i, j] * (X[j] @ OV))
            assert t.columns["force_magnitude"][row] == pytest.approx(expected)

    def test_weight_column_is_the_attention_entry(self):
        X, A, OV = _setup()
        t = build_head_edges("m", "p", 0, 0, X, A, OV, top_k_per_target=None)
        for row in range(len(t)):
            i, j = int(t.columns["target"][row]), int(t.columns["source"][row])
            assert t.columns["weight"][row] == pytest.approx(A[i, j])

    def test_only_causal_edges_are_emitted(self):
        X, A, OV = _setup()
        t = build_head_edges("m", "p", 0, 0, X, A, OV, top_k_per_target=None)
        assert np.all(t.columns["source"] <= t.columns["target"])

    def test_channels_are_populated_when_projectors_supplied(self):
        X, A, OV = _setup()
        rng = np.random.default_rng(9)
        Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        t = build_head_edges("m", "p", 0, 0, X, A, OV,
                             U_pos=Q[:, :2] @ Q[:, :2].T,
                             U_neg=Q[:, 2:] @ Q[:, 2:].T,
                             top_k_per_target=None)
        both = t.columns["attractive_frac"] + t.columns["repulsive_frac"]
        assert np.allclose(both, 1.0)

    def test_pair_types_are_passed_through_not_re_derived(self):
        X, A, OV = _setup()
        t = build_head_edges("m", "p", 0, 0, X, A, OV,
                             induction_pairs=[(3, 1)], top_k_per_target=None)
        sel = (t.columns["target"] == 3) & (t.columns["source"] == 1)
        assert t.columns["pair_type"][sel][0] == "induction"

    def test_a_head_that_moves_nothing_returns_an_empty_table_not_an_error(self):
        """A dead head is an observation. It must still concatenate with
        its siblings rather than breaking the merge."""
        X, A, _ = _setup()
        t = build_head_edges("m", "p", 0, 0, X, A, np.zeros((4, 4)),
                             top_k_per_target=None)
        assert len(t) == 0
        assert t.retention["mode"] == "threshold_only"

    def test_retention_travels_with_the_table(self):
        X, A, OV = _setup()
        t = build_head_edges("m", "p", 0, 0, X, A, OV, top_k_per_target=4)
        assert t.retention["k"] == 4
        assert t.retention["status"] == "placed"

    def test_heads_built_with_the_same_cutoff_concatenate(self):
        X, A, OV = _setup()
        heads = [build_head_edges("m", "p", 0, h, X, A, OV, top_k_per_target=3)
                 for h in range(3)]
        merged = InteractionTable.concat(heads)
        assert merged.retention["k"] == 3
        assert len(merged) == sum(len(h) for h in heads)

    def test_heads_built_with_different_cutoffs_refuse_to_merge(self):
        """The retention contract, end to end from the producer."""
        X, A, OV = _setup()
        a = build_head_edges("m", "p", 0, 0, X, A, OV, top_k_per_target=2)
        b = build_head_edges("m", "p", 0, 1, X, A, OV, top_k_per_target=5)
        with pytest.raises(ValueError, match="different retention"):
            InteractionTable.concat([a, b])

    def test_top_k_reduces_the_edge_count(self):
        X, A, OV = _setup(n=8)
        full = build_head_edges("m", "p", 0, 0, X, A, OV, top_k_per_target=None)
        thin = build_head_edges("m", "p", 0, 0, X, A, OV, top_k_per_target=2)
        assert len(thin) < len(full)


class TestEndToEndPlantedRelay:
    """
    The strongest oracle available without a model: plant an induction
    circuit in constructed attention and OV weights, run it through the
    real producer, and require the real motif finder to recover it.

    The unit tests above check the producer against its definition and
    tests/test_p7_motif_alphabet.py checks the finder against hand-built
    edges. This is the seam between them, which is where a producer that
    is individually correct can still emit something the finder cannot
    read (transposed indices, wrong offset sign, pair types keyed the
    other way round).
    """

    D = 4
    N = 12
    TAG = 5          # the particle the tag is written into
    PREV = 4         # TAG - 1
    MATCH_TGT = 9    # the particle that later matches on the tag

    def _attractive_ov(self):
        """An OV circuit whose output lands entirely in U_pos."""
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((self.D, self.D)))
        P_pos = Q[:, :2] @ Q[:, :2].T
        P_neg = Q[:, 2:] @ Q[:, 2:].T
        return P_pos, P_neg

    def _attention(self, target, source):
        """Causal attention that puts essentially all of `target`'s mass on
        `source`, and is diagonal elsewhere."""
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            A[i, i] = 1.0
        A[target, target] = 0.02
        A[target, source] = 0.98
        return A

    def _build(self, layer, head, target, source, OV, pairs=None, U_pos=None, U_neg=None):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((self.N, self.D))
        return build_head_edges(
            "m", "p", layer, head, X, self._attention(target, source), OV,
            U_pos=U_pos, U_neg=U_neg,
            induction_pairs=pairs, checkpoint_step=1000,
            top_k_per_target=2, declared_frame="raw",
        )

    def _planted_table(self, stage2_ov=None, stage2_pairs=None):
        P_pos, P_neg = self._attractive_ov()
        stage1 = self._build(2, 0, self.TAG, self.PREV, P_pos,
                             U_pos=P_pos, U_neg=P_neg)
        stage2 = self._build(7, 3, self.MATCH_TGT, self.TAG,
                             stage2_ov if stage2_ov is not None else P_pos,
                             pairs=(stage2_pairs if stage2_pairs is not None
                                    else [(self.MATCH_TGT, self.TAG)]),
                             U_pos=P_pos, U_neg=P_neg)
        return InteractionTable.concat([stage1, stage2])

    def test_the_planted_relay_survives_the_whole_pipeline(self):
        from p7_motifs.motif_alphabet import relay_strength
        assert relay_strength(self._planted_table()) == {(2, 0, 7, 3): 1}

    def test_both_stages_are_visible_as_their_own_motifs(self):
        from p7_motifs.motif_alphabet import motif_mask
        t = self._planted_table()
        assert motif_mask("prev_token", t)["count"] >= 1
        assert motif_mask("match", t)["count"] >= 1

    def test_a_repulsive_stage_two_breaks_the_relay(self):
        """The force decomposition doing work the attention pattern alone
        could not: identical routing, opposite-signed OV, no circuit."""
        from p7_motifs.motif_alphabet import relay_strength
        _, P_neg = self._attractive_ov()
        assert relay_strength(self._planted_table(stage2_ov=P_neg)) == {}

    def test_without_the_induction_pair_there_is_no_relay(self):
        from p7_motifs.motif_alphabet import relay_strength
        assert relay_strength(self._planted_table(stage2_pairs=[])) == {}

    def test_the_tag_particle_is_flagged_at_the_stage_one_layer(self):
        """Producer -> finder -> event level, the full chain P-I4 needs."""
        from p7_motifs.events import relay_target_flags
        from p7_motifs.motif_alphabet import find_relays
        from core.particles import ParticleTable

        relays = find_relays(self._planted_table())
        particles = ParticleTable.concat([
            ParticleTable.from_layer(model="m", prompt_key="p", layer=layer,
                                     cluster_labels=[-1] * self.N,
                                     checkpoint_step=1000)
            for layer in (2, 7)
        ])
        flags = relay_target_flags(particles, relays)
        sel = ((particles.columns["layer"] == 2)
               & (particles.columns["token_position"] == self.TAG))
        assert flags[sel][0]
        assert flags.sum() == 1
