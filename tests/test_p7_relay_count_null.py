"""
tests/test_p7_relay_count_null.py — oracle tier for
p7_motifs/relay_count_null.py, P-I1's relay-count null (PROJECT.md §3.4).

Two properties matter most, matching the discipline `test_p7_motif_alphabet.py`
already uses for `find_relays` itself:

  1. A planted relay collapses under the shuffle — the null erases the
     structure it is supposed to erase.
  2. A table with no real structure reads near the same chance rate whether
     or not it is shuffled — the null does not manufacture signal from
     nothing, nor suppress it below what randomness alone gives.

Plus the mechanics a payload shuffle has to get right: exact edge-count
preservation per (prompt, layer, head), no duplicate positions within one
head, every non-positional column untouched row-for-row, `pair_type`
recomputed to agree with `core.interactions.classify_pair_types`'
precedence, and a context boundary that does not leak across prompts.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.pure

from core.battery_structure import induction_candidates, same_content_candidates
from core.interactions import InteractionTable, classify_pair_types
from p7_motifs.motif_alphabet import find_relays
from p7_motifs.relay_count_null import (
    PromptNullContext,
    RelayNullRefused,
    build_prompt_context,
    causal_pool,
    null_envelope,
    shuffle_replicate,
)

D = 4
U_POS = np.eye(D)[:, :2]
U_NEG = np.eye(D)[:, 2:]
ATTRACTIVE_F = np.array([1.0, 0.0, 0.0, 0.0])
REPULSIVE_F = np.array([0.0, 0.0, 1.0, 0.0])


def _edges(specs, checkpoint_step=1000, model="m", prompt="p"):
    """(layer, head, target, source, kind in {'a','r'}, pair_type) -> table."""
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
            weight=np.ones(len(rows)), force=force,
            U_pos=U_POS, U_neg=U_NEG,
            pair_type=[pt for _, _, _, pt in rows],
            checkpoint_step=checkpoint_step,
        ))
    return InteractionTable.concat(tables)


class TestCausalPool:

    def test_every_pair_is_causal_and_distinct(self):
        src, tgt = causal_pool(6)
        assert (src < tgt).all()
        pairs = set(zip(src.tolist(), tgt.tolist()))
        assert len(pairs) == len(src)
        assert len(pairs) == 6 * 5 // 2

    def test_refuses_fewer_than_two_tokens(self):
        with pytest.raises(RelayNullRefused, match="no causal"):
            causal_pool(1)


class TestBuildPromptContext:

    def test_n_induction_is_a_property_of_the_ids_alone(self):
        """
        The claim §3.4's first constraint rests on: the induction candidate
        set -- and therefore its SIZE, `n_induction` -- does not depend on
        anything the null draws. Same ids, same context, called twice.
        """
        ids = [3, 1, 4, 1, 5, 9, 2, 6, 1, 4]
        c1 = build_prompt_context("p", ids)
        c2 = build_prompt_context("p", ids)
        assert c1.induction_keys.tolist() == c2.induction_keys.tolist()
        real = induction_candidates(ids)
        assert len(c1.induction_keys) == len(real)

    def test_pool_size_matches_the_token_count(self):
        ctx = build_prompt_context("p", list(range(12)))
        assert ctx.pool_size == 12 * 11 // 2


class TestPairTypesAgreeWithClassifyPairTypes:

    def test_agrees_on_random_positions(self):
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 8, size=40).tolist()
        ctx = build_prompt_context("p", ids)
        ind = induction_candidates(ids)
        strict = induction_candidates(ids, strict=True)
        same = same_content_candidates(ids, ind)

        src, tgt = causal_pool(len(ids))
        draw = rng.choice(len(src), size=200, replace=True)
        targets, sources = tgt[draw], src[draw]

        from p7_motifs.relay_count_null import _pair_types
        fast = _pair_types(targets, sources, ctx)
        slow = classify_pair_types(targets, sources, ind, strict, same)
        assert (fast == slow).all()


class TestShuffleReplicate:

    def _table_and_ctx(self, n_tokens=16, n_edges_per_head=6, seed=0,
                       return_ids=False):
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, 5, size=n_tokens).tolist()
        ctx = build_prompt_context("p", ids)
        specs = []
        for layer, head in ((0, 0), (0, 1), (1, 0)):
            for _ in range(n_edges_per_head):
                tgt = int(rng.integers(1, n_tokens))
                src = int(rng.integers(0, tgt))
                kind = "a" if rng.random() < 0.5 else "r"
                specs.append((layer, head, tgt, src, kind, "neither"))
        t = _edges(specs, prompt="p")
        if return_ids:
            return t, {"p": ctx}, ids
        return t, {"p": ctx}

    def test_refuses_a_prompt_with_no_context(self):
        t, contexts = self._table_and_ctx()
        with pytest.raises(RelayNullRefused, match="no PromptNullContext"):
            shuffle_replicate(t, {}, np.random.default_rng(1))

    def test_refuses_when_retention_exceeds_the_pool(self):
        # 4 tokens -> pool of 6 causal pairs; ask for 7 edges in one head.
        t = _edges([(0, 0, 1, 0, "a", "neither")] * 7, prompt="p")
        ctx = build_prompt_context("p", [0, 1, 2, 3])
        with pytest.raises(RelayNullRefused, match="causal pool"):
            shuffle_replicate(t, {"p": ctx}, np.random.default_rng(2))

    def test_edge_count_is_exact_per_head(self):
        t, contexts = self._table_and_ctx()
        shuf = shuffle_replicate(t, contexts, np.random.default_rng(3))
        assert len(shuf) == len(t)
        for (l, h) in ((0, 0), (0, 1), (1, 0)):
            m_before = (t.columns["layer"] == l) & (t.columns["head"] == h)
            m_after = (shuf.columns["layer"] == l) & (shuf.columns["head"] == h)
            assert m_before.sum() == m_after.sum()

    def test_positions_are_distinct_within_one_head(self):
        t, contexts = self._table_and_ctx(n_edges_per_head=8)
        shuf = shuffle_replicate(t, contexts, np.random.default_rng(4))
        for (l, h) in ((0, 0), (0, 1), (1, 0)):
            m = (shuf.columns["layer"] == l) & (shuf.columns["head"] == h)
            pairs = list(zip(shuf.columns["target"][m].tolist(),
                             shuf.columns["source"][m].tolist()))
            assert len(set(pairs)) == len(pairs)

    def test_every_shuffled_position_is_causal(self):
        t, contexts = self._table_and_ctx()
        shuf = shuffle_replicate(t, contexts, np.random.default_rng(5))
        assert (shuf.columns["source"] < shuf.columns["target"]).all()
        assert (shuf.columns["offset"] == shuf.columns["target"]
               - shuf.columns["source"]).all()

    def test_the_force_payload_travels_with_its_row_unchanged(self):
        """
        Only target/source/offset/pair_type may move. Every other column,
        row for row, is bit-identical to the input -- this IS the
        "degree-preserving at the head level, full force distribution
        preserved" claim, not an aggregate check of it.
        """
        t, contexts = self._table_and_ctx()
        shuf = shuffle_replicate(t, contexts, np.random.default_rng(6))
        for col in ("model", "checkpoint_step", "prompt_key", "layer", "head",
                   "weight", "force_magnitude", "attractive_frac",
                   "repulsive_frac"):
            assert np.array_equal(t.columns[col], shuf.columns[col]), col

    def test_pair_type_is_recomputed_from_the_new_position(self):
        t, contexts, ids = self._table_and_ctx(return_ids=True)
        shuf = shuffle_replicate(t, contexts, np.random.default_rng(7))
        ind = induction_candidates(ids)
        expected = classify_pair_types(
            shuf.columns["target"], shuf.columns["source"], ind,
            induction_candidates(ids, strict=True),
            same_content_candidates(ids, ind))
        assert (shuf.columns["pair_type"] == expected).all()

    def test_two_prompts_do_not_leak_positions_into_each_other(self):
        """
        Each prompt's edges are redrawn only from ITS OWN pool. A leak would
        show up as a position wider than the smaller prompt's token count.
        """
        rng = np.random.default_rng(8)
        ids_small = rng.integers(0, 4, size=5).tolist()
        ids_big = rng.integers(0, 4, size=40).tolist()
        specs = [(0, 0, 3, 1, "a", "neither")] * 3       # prompt "small"
        t_small = _edges(specs, prompt="small")
        specs2 = [(0, 0, 30, 5, "a", "neither")] * 3     # prompt "big"
        t_big = _edges(specs2, prompt="big")
        t = InteractionTable.concat([t_small, t_big])
        contexts = {"small": build_prompt_context("small", ids_small),
                   "big": build_prompt_context("big", ids_big)}
        for seed in range(5):
            shuf = shuffle_replicate(t, contexts, np.random.default_rng(seed))
            m_small = shuf.columns["prompt_key"] == "small"
            assert (shuf.columns["target"][m_small] < len(ids_small)).all()


class TestNullEnvelope:

    def _planted_and_ids(self):
        """
        One genuine relay: ids give a real induction pair (query, key); the
        stage-2 edge sits exactly there, and a stage-1 edge feeds it (target
        = key, source = key - 1, offset 1, attractive) in an earlier layer.
        """
        ids = [3, 1, 4, 1, 5, 9, 2, 6, 1, 4, 1, 5, 9, 2, 6]
        ind = induction_candidates(ids)
        assert ind, "fixture must produce at least one induction pair"
        query, key = ind[0]
        assert key >= 1, "need room for a stage-1 edge at key-1 -> key"
        specs = [
            (1, 0, key, key - 1, "a", "neither"),     # stage 1: prev_token
            (5, 3, query, key, "a", "induction"),     # stage 2: match
        ]
        t = _edges(specs, prompt="p", checkpoint_step=54000)
        ctx = build_prompt_context("p", ids)
        return t, {"p": ctx}, (1, 0, 5, 3)

    def test_the_planted_relay_is_found_in_the_real_table(self):
        t, _, key = self._planted_and_ids()
        relays = find_relays(t)
        assert len(relays) == 1
        r = relays[0]
        assert (r.layer_1, r.head_1, r.layer_2, r.head_2) == key

    def test_the_planted_relay_collapses_under_the_null(self):
        """
        The point of the whole construction: a real structure the raw count
        reports as 1 relay reads at approximately the CHANCE rate once the
        positions are shuffled -- averaged over enough replicates that one
        lucky draw cannot carry the assertion.
        """
        t, contexts, matcher_head = self._planted_and_ids()
        env = null_envelope(t, contexts, relay_owner="matcher",
                            n_replicates=60, seed=0)
        matcher_key = (matcher_head[2], matcher_head[3])
        null_mean = env.get(matcher_key, {"mean": 0.0})["mean"]
        assert null_mean < 0.5, (
            f"null mean {null_mean} is not near-chance for a single planted "
            f"relay; the shuffle is not erasing the structure it should")

    def test_deterministic_given_the_seed(self):
        t, contexts, _ = self._planted_and_ids()
        e1 = null_envelope(t, contexts, relay_owner="matcher",
                           n_replicates=10, seed=42)
        e2 = null_envelope(t, contexts, relay_owner="matcher",
                           n_replicates=10, seed=42)
        assert e1 == e2

    def test_refuses_fewer_than_two_replicates(self):
        t, contexts, _ = self._planted_and_ids()
        with pytest.raises(RelayNullRefused, match="n_replicates"):
            null_envelope(t, contexts, relay_owner="matcher",
                          n_replicates=1, seed=0)

    def test_a_structureless_table_reads_near_chance_shuffled_or_not(self):
        """
        Calibration, mirroring `test_p7_motif_alphabet.py`'s
        `TestRandomGraphReadsNull`: a table with no planted structure and
        pair_type all "neither" cannot form a relay whether or not it is
        shuffled, because `match_mask` requires an induction/strict pair_type
        that this fixture never assigns. The null must not manufacture one.
        """
        rng = np.random.default_rng(9)
        n_tokens = 24
        ids = rng.integers(0, 6, size=n_tokens).tolist()
        specs = []
        for layer in range(3):
            for head in range(3):
                for _ in range(10):
                    tgt = int(rng.integers(1, n_tokens))
                    src = int(rng.integers(0, tgt))
                    specs.append((layer, head, tgt, src,
                                 "a" if rng.random() < 0.5 else "r", "neither"))
        t = _edges(specs, prompt="p")
        ctx = build_prompt_context("p", ids)
        assert find_relays(t) == []
        env = null_envelope(t, {"p": ctx}, relay_owner="matcher",
                            n_replicates=20, seed=1)
        assert all(v["mean"] < 1.0 for v in env.values())
