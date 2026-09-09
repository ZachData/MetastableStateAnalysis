"""
tests/test_p7_motif_alphabet.py — oracle tier for p7_motifs/motif_alphabet.py.

Oracle-tier in the sense tests/test_core_oracle.py uses: these are
correctness tests against constructed inputs whose answer is known in
advance, not regression tests against recorded output. If a planted relay
motif is not recovered, or a matched random graph reports one, the motif
machinery is wrong and no amount of "maybe the model doesn't show it here"
applies.

Two properties matter most and get the most tests:

  1. The planted relay is found, with its head-pair attribution intact.
  2. A random graph reads at the null rate. A motif finder that reports
     structure in noise is worse than useless — it would confirm P-I1 on
     any checkpoint, including step 0.

Plus the negative controls that keep `relay` from silently degenerating
into the behavioural induction score, which is the tautology risk
design-7.md names as the phase's central methodological danger.
"""
from __future__ import annotations

import numpy as np
import pytest

# Tier: pure -- this module's whole test set passes with torch,
# transformers, scikit-learn and matplotlib all unimportable. Measured,
# not assumed; see pytest.ini [pytest] markers.
pytestmark = pytest.mark.pure

from core.interactions import InteractionTable
from p7_motifs.motif_alphabet import (
    ALPHABET_VERSION,
    DEFAULTS,
    MOTIF_NAMES,
    find_relays,
    motif_mask,
    relay_strength,
)


D = 4
# An orthonormal split of R^4: U_pos = span(e0,e1), U_neg = span(e2,e3).
U_POS = np.eye(D)[:, :2]
U_NEG = np.eye(D)[:, 2:]

ATTRACTIVE_F = np.array([1.0, 0.0, 0.0, 0.0])   # entirely in U_pos
REPULSIVE_F  = np.array([0.0, 0.0, 1.0, 0.0])   # entirely in U_neg


def _edges(specs, checkpoint_step=1000, model="m", prompt="p"):
    """
    Build an InteractionTable from a list of edge specs:
        (layer, head, target, source, kind, pair_type)
    where kind is "a" (attractive) or "r" (repulsive).

    Built per (layer, head) group through from_head and concatenated, which
    is how a real producer would do it, so these tests exercise the real
    construction path rather than poking columns in directly.
    """
    groups = {}
    for layer, head, target, source, kind, pt in specs:
        groups.setdefault((layer, head), []).append((target, source, kind, pt))

    tables = []
    for (layer, head), rows in sorted(groups.items()):
        force = np.array([ATTRACTIVE_F if k == "a" else REPULSIVE_F
                          for _, _, k, _ in rows])
        tables.append(InteractionTable.from_head(
            model=model, prompt_key=prompt, layer=layer, head=head,
            targets=[t for t, _, _, _ in rows],
            sources=[s for _, s, _, _ in rows],
            weight=np.ones(len(rows)),
            force=force,
            U_pos=U_POS, U_neg=U_NEG,
            pair_type=[pt for _, _, _, pt in rows],
            checkpoint_step=checkpoint_step,
        ))
    return InteractionTable.concat(tables)


# ---------------------------------------------------------------------------
# The planted relay
# ---------------------------------------------------------------------------

class TestPlantedRelayRecovered:
    """
    The oracle. Plant one induction circuit and nothing else:

      stage 1, layer 2 head 0 : particle 5 attends to particle 4 (offset -1)
                                -> the tag is written into particle 5
      stage 2, layer 7 head 3 : particle 9 attends to particle 5, and (9,5)
                                is an induction pair

    That is exactly one relay, at (2, 0, 7, 3), through tag particle 5.
    """

    def _planted(self):
        return _edges([
            (2, 0, 5, 4, "a", "neither"),      # stage 1: prev_token
            (7, 3, 9, 5, "a", "induction"),    # stage 2: match, sourced at the tag
        ])

    def test_exactly_one_relay_found(self):
        assert len(find_relays(self._planted())) == 1

    def test_head_pair_attribution_is_exact(self):
        assert relay_strength(self._planted()) == {(2, 0, 7, 3): 1}

    def test_tag_particle_is_the_composition_point(self):
        r = find_relays(self._planted())[0]
        assert r.tag_position == 5      # target of stage 1, source of stage 2
        assert r.match_target == 9

    def test_both_stages_are_detected_individually(self):
        t = self._planted()
        assert motif_mask("prev_token", t)["count"] == 1
        assert motif_mask("match", t)["count"] == 1


class TestRelayNegativeControls:
    """Each removes exactly one necessary condition. All must give zero —
    these are what keep `relay` from collapsing into 'a match edge exists',
    which is the behavioural induction score and would make P-I3
    tautological."""

    def test_no_relay_without_the_prev_token_stage(self):
        t = _edges([(7, 3, 9, 5, "a", "induction")])
        assert find_relays(t) == []

    def test_no_relay_when_stages_do_not_share_the_tag_particle(self):
        """Stage 1 writes into particle 5; stage 2 matches on particle 6.
        Two real edges, no composition."""
        t = _edges([
            (2, 0, 5, 4, "a", "neither"),
            (7, 3, 9, 6, "a", "induction"),
        ])
        assert find_relays(t) == []

    def test_no_relay_when_stage_2_precedes_stage_1(self):
        """Information cannot be matched before it is written. L2 < L1 is a
        coincidence of two edges, not a circuit."""
        t = _edges([
            (7, 0, 5, 4, "a", "neither"),      # prev_token, late
            (2, 3, 9, 5, "a", "induction"),    # match, early
        ])
        assert find_relays(t) == []

    def test_no_relay_within_a_single_layer(self):
        t = _edges([
            (4, 0, 5, 4, "a", "neither"),
            (4, 3, 9, 5, "a", "induction"),
        ])
        assert find_relays(t) == []

    def test_no_relay_when_stage_2_is_repulsive(self):
        """A match edge that pushes the particle away is not stage 2 of an
        induction circuit. This is the force decomposition doing work the
        attention pattern alone could not."""
        t = _edges([
            (2, 0, 5, 4, "a", "neither"),
            (7, 3, 9, 5, "r", "induction"),
        ])
        assert find_relays(t) == []

    def test_no_relay_when_the_pair_is_not_an_induction_pair(self):
        t = _edges([
            (2, 0, 5, 4, "a", "neither"),
            (7, 3, 9, 5, "a", "same_content"),
        ])
        assert find_relays(t) == []


class TestRelayMultiplicity:

    def test_two_independent_circuits_counted_separately(self):
        t = _edges([
            (2, 0, 5, 4, "a", "neither"),
            (7, 3, 9, 5, "a", "induction"),
            (2, 1, 12, 11, "a", "neither"),
            (8, 2, 20, 12, "a", "induction"),
        ])
        assert relay_strength(t) == {(2, 0, 7, 3): 1, (2, 1, 8, 2): 1}

    def test_one_tag_feeding_two_match_heads_gives_two_relays(self):
        """A single written tag consumed by two downstream heads is two
        relays, not one — the head pair is part of the identity."""
        t = _edges([
            (2, 0, 5, 4, "a", "neither"),
            (7, 3, 9, 5, "a", "induction"),
            (7, 6, 9, 5, "a", "induction"),
        ])
        assert relay_strength(t) == {(2, 0, 7, 3): 1, (2, 0, 7, 6): 1}


# ---------------------------------------------------------------------------
# The join scope: a position only names a particle inside its own context
# ---------------------------------------------------------------------------

class TestRelayJoinIsPerContext:
    """
    p7_motifs/run_7.py writes all 8 battery prompts into one
    InteractionTable, and `tag_position` is a per-prompt token index. A
    join on position alone therefore composes a tag written in one prompt
    with a match found in another. It does not error; the positions
    collide and the pair reads as a relay. On pythia-410m at step 54000 it
    was a 9.0x inflation (23,050,007 against 2,560,483).

    The halves below are each a complete relay in isolation and must
    produce exactly one relay each — never a third from the cross terms.
    """

    def _stage_1_only(self, prompt):
        return _edges([(2, 0, 5, 4, "a", "neither")], prompt=prompt)

    def _stage_2_only(self, prompt):
        return _edges([(7, 3, 9, 5, "a", "induction")], prompt=prompt)

    def test_a_tag_in_one_prompt_does_not_feed_a_match_in_another(self):
        """The defect, isolated: stage 1 exists only in prompt A, stage 2
        only in prompt B. Neither prompt holds a relay, so there is none."""
        t = InteractionTable.concat([self._stage_1_only("prompt_a"),
                                     self._stage_2_only("prompt_b")])
        assert find_relays(t) == []

    def test_two_prompts_each_holding_one_relay_give_two(self):
        """The join must still fire WITHIN each prompt — a scope fix that
        also dropped the real relays would pass the test above."""
        t = InteractionTable.concat([
            _edges([(2, 0, 5, 4, "a", "neither"),
                    (7, 3, 9, 5, "a", "induction")], prompt="prompt_a"),
            _edges([(2, 0, 5, 4, "a", "neither"),
                    (7, 3, 9, 5, "a", "induction")], prompt="prompt_b"),
        ])
        assert len(find_relays(t)) == 2
        assert relay_strength(t) == {(2, 0, 7, 3): 2}
        assert {r.prompt_key for r in find_relays(t)} == {"prompt_a", "prompt_b"}

    def test_concatenating_prompts_equals_summing_them_separately(self):
        """The property the sweep's numbers depend on: one table holding
        the battery must count what the per-prompt tables counted. This is
        the assertion the 9.0x inflation violated."""
        halves = [
            _edges([(2, 0, 5, 4, "a", "neither"),
                    (7, 3, 9, 5, "a", "induction"),
                    (3, 1, 8, 7, "a", "neither")], prompt="prompt_a"),
            _edges([(2, 0, 6, 5, "a", "neither"),
                    (7, 3, 9, 5, "a", "induction"),
                    (4, 2, 12, 11, "a", "neither")], prompt="prompt_b"),
        ]
        summed = {}
        for h in halves:
            for k, v in relay_strength(h).items():
                summed[k] = summed.get(k, 0) + v
        assert relay_strength(InteractionTable.concat(halves)) == summed

    def test_the_same_prompt_at_two_checkpoints_does_not_join(self):
        """Positions repeat across checkpoints too, and
        InteractionTable.concat does not forbid mixing them."""
        t = InteractionTable.concat([
            _edges([(2, 0, 5, 4, "a", "neither")], checkpoint_step=0),
            _edges([(7, 3, 9, 5, "a", "induction")], checkpoint_step=1000),
        ])
        assert find_relays(t) == []

    def test_a_relay_carries_the_prompt_its_positions_belong_to(self):
        """`tag_position` without `prompt_key` cannot be resolved back to a
        particle, so the finder must report both."""
        r = find_relays(_edges([(2, 0, 5, 4, "a", "neither"),
                               (7, 3, 9, 5, "a", "induction")],
                              prompt="homer_iliad"))[0]
        assert r.prompt_key == "homer_iliad"


# ---------------------------------------------------------------------------
# The null: random graphs must read as nothing
# ---------------------------------------------------------------------------

class TestRandomGraphReadsNull:

    def _random_graph(self, seed, n_tokens=24, n_layers=6, n_heads=4):
        """Causal edges with random forces and no planted structure. Pair
        types are all 'neither', so no edge can be a `match` and therefore
        no relay can exist however the forces fall."""
        rng = np.random.default_rng(seed)
        tables = []
        for layer in range(n_layers):
            for head in range(n_heads):
                tgt, src = [], []
                for i in range(1, n_tokens):
                    for j in rng.choice(i, size=min(3, i), replace=False):
                        tgt.append(i)
                        src.append(int(j))
                tables.append(InteractionTable.from_head(
                    model="m", prompt_key="p", layer=layer, head=head,
                    targets=tgt, sources=src,
                    weight=rng.random(len(tgt)),
                    force=rng.standard_normal((len(tgt), D)),
                    U_pos=U_POS, U_neg=U_NEG,
                    pair_type=["neither"] * len(tgt),
                    checkpoint_step=0,
                ))
        return InteractionTable.concat(tables)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_no_relays_in_an_unstructured_graph(self, seed):
        assert find_relays(self._random_graph(seed)) == []

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_no_match_edges_without_induction_pairs(self, seed):
        assert motif_mask("match", self._random_graph(seed))["count"] == 0

    def test_attractive_and_repulsive_split_near_half_on_random_forces(self):
        """A random force in R^4 with a 2/2 subspace split should land
        attractive about half the time. Far from 0.5 means the channel
        decomposition is biased, which would tilt every motif count."""
        t = self._random_graph(9)
        attr = motif_mask("repulsor", t)["count"] / len(t)
        assert 0.35 < attr < 0.65


# ---------------------------------------------------------------------------
# Single-edge and structural motifs
# ---------------------------------------------------------------------------

class TestSingleEdgeMotifs:

    def test_prev_token_requires_offset_one_and_attraction(self):
        t = _edges([
            (1, 0, 5, 4, "a", "neither"),   # offset 1, attractive  -> yes
            (1, 0, 6, 4, "a", "neither"),   # offset 2              -> no
            (1, 0, 8, 7, "r", "neither"),   # offset 1, repulsive   -> no
        ])
        assert motif_mask("prev_token", t)["count"] == 1

    def test_sink_does_not_require_attraction(self):
        """Sink behaviour is defined by where attention goes. Whether sink
        edges are attractive is a question, not part of the definition."""
        t = _edges([
            (1, 0, 5, 0, "r", "neither"),
            (1, 0, 6, 0, "a", "neither"),
            (1, 0, 7, 3, "a", "neither"),
        ])
        assert motif_mask("sink", t)["count"] == 2

    def test_mutual_requires_both_directions_attractive(self):
        t = _edges([
            (1, 0, 5, 6, "a", "neither"),
            (1, 0, 6, 5, "a", "neither"),   # reciprocal -> both count
            (1, 0, 7, 8, "a", "neither"),
            (1, 0, 8, 7, "r", "neither"),   # reverse is repulsive -> neither counts
        ])
        assert motif_mask("mutual", t)["count"] == 2

    def test_mutual_does_not_cross_heads(self):
        """An edge in head 0 and its reverse in head 1 are not a bound
        pair: they are two heads doing unrelated things."""
        t = _edges([
            (1, 0, 5, 6, "a", "neither"),
            (1, 1, 6, 5, "a", "neither"),
        ])
        assert motif_mask("mutual", t)["count"] == 0

    def test_hub_detects_a_planted_attractor(self):
        """One particle attracting many, against a background of pairs.

        The background deliberately does not source anything at particle 3,
        so the expected count is exactly the 20 planted edges. (An earlier
        version of this test used a background that did, making 21 the
        right answer and the assertion ambiguous about which behaviour it
        was pinning.)"""
        specs = [(1, 0, i, 3, "a", "neither") for i in range(10, 30)]
        specs += [(1, 0, i, i - 1, "a", "neither") for i in range(5, 10)]
        res = motif_mask("hub", _edges(specs))
        assert res["count"] == 20    # exactly the edges into particle 3

    def test_hub_needs_something_to_stand_out_from(self):
        """A head where every attractive edge shares one source has no
        comparison population. Undecidable, and must not default to 'hub'
        — otherwise every single-source head reads as an attractor."""
        specs = [(1, 0, i, 3, "a", "neither") for i in range(10, 20)]
        assert motif_mask("hub", _edges(specs))["count"] == 0

    def test_hub_survives_a_larger_background(self):
        """The leave-one-out rule must not depend on the background being
        small: the same attractor against 30 background particles is still
        a hub. A rule using the pooled standard deviation gets harder to
        satisfy as the outlier grows; this one does not."""
        specs = [(1, 0, i, 3, "a", "neither") for i in range(40, 70)]
        specs += [(1, 0, i, i - 1, "a", "neither") for i in range(5, 35)]
        assert motif_mask("hub", _edges(specs))["count"] == 30

    def test_uniform_in_degree_has_no_hub(self):
        """A flat attention pattern has no attractor, and must not
        manufacture one out of zero variance."""
        specs = [(1, 0, i, j, "a", "neither")
                 for i in (10, 11, 12) for j in (1, 2, 3)]
        assert motif_mask("hub", _edges(specs))["count"] == 0

    # -- the structural motifs join by position too, and positions are
    # -- per-prompt token indices. Same defect as TestRelayJoinIsPerContext.

    def test_mutual_does_not_cross_prompts(self):
        """5<-4 in one prompt and 4<-5 in another are not each other's
        reverse: they are edges between four different particles. Joined on
        position alone, both read as a bound pair neither prompt has."""
        t = InteractionTable.concat([
            _edges([(1, 0, 5, 6, "a", "neither")], prompt="prompt_a"),
            _edges([(1, 0, 6, 5, "a", "neither")], prompt="prompt_b"),
        ])
        assert motif_mask("mutual", t)["count"] == 0

    def test_mutual_still_fires_within_one_prompt_of_a_battery(self):
        """The scope fix must not cost the real bound pairs."""
        t = InteractionTable.concat([
            _edges([(1, 0, 5, 6, "a", "neither"),
                    (1, 0, 6, 5, "a", "neither")], prompt="prompt_a"),
            _edges([(1, 0, 9, 8, "a", "neither")], prompt="prompt_b"),
        ])
        assert motif_mask("mutual", t)["count"] == 2

    def _flat_head(self, sources, in_degree, first_target):
        """One (layer, head) whose every source has the SAME in-degree, so
        it holds no attractor on its own. `first_target` keeps the target
        positions distinct within the group."""
        specs, tgt = [], first_target
        for src in sources:
            for _ in range(in_degree):
                specs.append((1, 0, tgt, src, "a", "neither"))
                tgt += 1
        return specs

    def test_hub_in_degree_is_not_pooled_across_prompts(self):
        """Two prompts, each internally FLAT — no attractor in either. They
        share source 3 and differ elsewhere, so pooling their in-degrees
        gives 3 a count of 9 against a 4/4/5/5 background and manufactures
        an attractor that neither prompt contains.

        This is what a battery table does to `hub` by default: prompts
        differ in length and in which positions carry content, so pooled
        in-degrees are a sum over texts the position has nothing to do
        with. Manufacture is the direction planted here because it is the
        one a test can pin; on the real step-54000 table the net effect ran
        the other way (see hub_mask's docstring), and both follow from the
        same pooled baseline."""
        a = self._flat_head(sources=(1, 2, 3), in_degree=4, first_target=10)
        b = self._flat_head(sources=(3, 4, 5), in_degree=5, first_target=10)
        assert motif_mask("hub", _edges(a, prompt="prompt_a"))["count"] == 0
        assert motif_mask("hub", _edges(b, prompt="prompt_b"))["count"] == 0

        t = InteractionTable.concat([_edges(a, prompt="prompt_a"),
                                     _edges(b, prompt="prompt_b")])
        assert motif_mask("hub", t)["count"] == 0

    def test_hub_still_fires_within_one_prompt_of_a_battery(self):
        planted = [(1, 0, i, 3, "a", "neither") for i in range(10, 30)]
        planted += [(1, 0, i, i - 1, "a", "neither") for i in range(5, 10)]
        background = [(1, 0, i, i - 1, "a", "neither") for i in range(5, 10)]
        t = InteractionTable.concat([_edges(planted, prompt="prompt_a"),
                                     _edges(background, prompt="prompt_b")])
        assert motif_mask("hub", t)["count"] == 20

    def test_hub_does_not_pool_across_checkpoints(self):
        """The same collision one axis over: positions repeat at every
        checkpoint, and InteractionTable.concat does not forbid mixing
        them."""
        a = self._flat_head(sources=(1, 2, 3), in_degree=4, first_target=10)
        b = self._flat_head(sources=(3, 4, 5), in_degree=5, first_target=10)
        t = InteractionTable.concat([_edges(a, checkpoint_step=0),
                                     _edges(b, checkpoint_step=1000)])
        assert motif_mask("hub", t)["count"] == 0


class TestReportingContract:

    def test_unknown_channel_is_reported_not_silently_zero(self):
        """With no U_pos supplied every attractive_frac is NaN, so the
        count is 0 — but the reader must be able to tell that from an
        honest zero."""
        t = InteractionTable.from_head(
            model="m", prompt_key="p", layer=1, head=0,
            targets=[5, 6], sources=[4, 5],
            weight=[1.0, 1.0], force=np.tile(ATTRACTIVE_F, (2, 1)),
            checkpoint_step=0,
        )
        res = motif_mask("prev_token", t)
        assert res["count"] == 0
        assert res["unknown_channel"] == 2

    def test_honest_zero_reports_no_unknown_channel(self):
        t = _edges([(1, 0, 6, 4, "a", "neither")])   # offset 2, not prev_token
        res = motif_mask("prev_token", t)
        assert res["count"] == 0
        assert res["unknown_channel"] == 0

    def test_thresholds_are_reported_as_placed(self):
        """Standing rule 6: an underived threshold says so, in the output."""
        res = motif_mask("prev_token", _edges([(1, 0, 5, 4, "a", "neither")]))
        assert set(res["threshold_status"].values()) == {"placed"}
        assert res["thresholds"].keys() == DEFAULTS.keys()

    def test_alphabet_version_travels_with_every_count(self):
        res = motif_mask("sink", _edges([(1, 0, 5, 0, "a", "neither")]))
        assert res["alphabet_version"] == ALPHABET_VERSION

    def test_relay_has_no_row_mask(self):
        with pytest.raises(ValueError, match="two-edge composition"):
            motif_mask("relay", _edges([(1, 0, 5, 4, "a", "neither")]))

    def test_unknown_motif_lists_the_alphabet(self):
        with pytest.raises(ValueError, match="Unknown motif"):
            motif_mask("nonexistent", _edges([(1, 0, 5, 4, "a", "neither")]))

    def test_every_named_motif_is_reachable(self):
        """MOTIF_NAMES and the dispatch table must not drift apart."""
        t = _edges([(1, 0, 5, 4, "a", "induction")])
        for name in MOTIF_NAMES:
            if name == "relay":
                assert find_relays(t) == []
            else:
                assert motif_mask(name, t)["motif"] == name
