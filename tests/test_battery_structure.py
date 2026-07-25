"""
tests/test_battery_structure.py — oracle tests for battery_structure.py.

The load-bearing tests are the degeneracy ones. A battery that tokenizes
into no usable induction structure produces a null result indistinguishable
from a real negative, so the module's job is to refuse before the run, not
to describe afterwards.
"""

import numpy as np
import pytest

from core.battery_structure import (
    analyze_prompt,
    assert_battery_structure,
    battery_summary_lines,
    induction_candidates,
    same_content_candidates,
    tokenize_prompt,
    verify_battery_structure,
)


class FakeTokenizer:
    """
    Word-level tokenizer with a stable vocabulary. Deterministic across
    processes, unlike the hash() ids in run_6.py that this module replaces.
    """

    def __init__(self, bos_token_id=None, prepend_bos=False, name="fake"):
        self.bos_token_id = bos_token_id
        self.name_or_path = name
        self._prepend = prepend_bos
        self._vocab = {}

    def _id(self, w):
        return self._vocab.setdefault(w, len(self._vocab) + 100)

    def __call__(self, text):
        ids = [self._id(w) for w in text.split()]
        if self._prepend and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}

    def __len__(self):
        return len(self._vocab) + 100


class MergingTokenizer(FakeTokenizer):
    """
    Collapses every token to a single id. Stands in for the real failure:
    a prompt whose designed repeat structure does not survive retokenization.
    """

    def _id(self, w):
        return 42


class TestTokenization:

    def test_real_ids_not_string_hashes(self):
        """
        Item 12: ids must be reproducible. Two separate tokenizer instances
        over the same text give the same ids; Python's salted hash() would
        not across processes.
        """
        a = tokenize_prompt(FakeTokenizer(), "a b a b")["ids"]
        b = tokenize_prompt(FakeTokenizer(), "a b a b")["ids"]
        assert a == b
        assert a[0] == a[2] and a[1] == a[3]

    def test_no_bos_by_default(self):
        assert tokenize_prompt(FakeTokenizer(), "a b")["has_bos"] is False

    def test_bos_detected(self):
        t = FakeTokenizer(bos_token_id=0, prepend_bos=True)
        assert tokenize_prompt(t, "a b")["has_bos"] is True

    def test_distinct_count(self):
        out = tokenize_prompt(FakeTokenizer(), "a b a b c")
        assert out["n_tokens"] == 5 and out["n_distinct"] == 3


_USABLE_TEXT = (
    "alpha beta gamma delta alpha beta epsilon zeta alpha beta gamma eta "
    "theta alpha beta gamma delta iota alpha beta gamma kappa"
)


class TestPairFinding:

    def test_induction_condition_matches_the_repo(self):
        """ids[key-1] == ids[query-1], the condition qk_decompose states."""
        ids = [1, 2, 3, 1, 2, 3]
        pairs = induction_candidates(ids, min_offset=2)
        # key=1 has ids[0]=1; query=4 has ids[3]=1 -> (4, 1) qualifies
        assert (4, 1) in pairs

    def test_strict_condition_is_a_different_set(self):
        """
        The docstring cites the Anthropic formulation but the code tests a
        different one. Divergence must be visible, not assumed away.
        """
        ids = [1, 2, 3, 1, 2, 3]
        loose = set(induction_candidates(ids, 2, strict=False))
        strict = set(induction_candidates(ids, 2, strict=True))
        assert loose != strict

    def test_query_always_after_key(self):
        ids = [1, 2, 1, 2, 1, 2]
        for q, k in induction_candidates(ids, 2):
            assert q > k

    def test_min_offset_respected(self):
        ids = [1, 1, 1, 1]
        assert all(q - k >= 3 for q, k in induction_candidates(ids, min_offset=3))

    def test_same_content_excludes_induction(self):
        ids = [1, 2, 3, 1, 2, 3]
        ind = induction_candidates(ids, 2)
        same = same_content_candidates(ids, ind, 2)
        assert not (set(ind) & set(same))

    def test_same_content_uses_id_equality_only(self):
        """
        Frame-independent by construction. The original implementation's
        cosine fallback makes the null set depend on the activation frame —
        i.e. on the bug being fixed.
        """
        ids = [1, 2, 3, 4, 1]
        same = same_content_candidates(ids, [], 2)
        assert (4, 0) in same


class TestDegeneracy:

    def test_uniform_prompt_is_degenerate(self):
        """
        config.PROMPTS["repeated_tokens"] is ". " x 264. Every token is
        identical, so every causal pair is trivially an induction pair and
        the same-content null is EMPTY. P6-I2b cannot be evaluated on it.
        """
        out = analyze_prompt(FakeTokenizer(), "repeated_tokens", ". " * 40)
        assert "uniform" in out["flags"]
        assert out["verdict"] == "degenerate"

    def test_uniform_prompt_null_collapses_onto_the_sink(self):
        """
        Subtler than an empty null, and worse. On a uniform prompt the only
        pairs NOT classified as induction are those with key = 0, because
        induction requires key >= 1. So the same-content null is entirely the
        attention-sink column, and P6-I2b would be comparing induction
        structure against sink behaviour rather than against content-matched
        pairs (policy P1).
        """
        out = analyze_prompt(FakeTokenizer(), "repeated_tokens", ". " * 40)
        assert out["n_same_content"] > 0
        assert out["null_sink_fraction"] == pytest.approx(1.0)
        assert "null_is_sink" in out["flags"]

    def test_retokenization_can_destroy_structure(self):
        """The failure the text hash cannot see."""
        text = "alpha beta gamma alpha beta gamma"
        good = analyze_prompt(FakeTokenizer(), "p", text)
        bad = analyze_prompt(MergingTokenizer(), "p", text)
        assert good["n_distinct_tokens"] > 1
        assert bad["n_distinct_tokens"] == 1
        assert bad["verdict"] == "degenerate"

    def test_single_offset_flagged(self):
        """N3 has no power when every induction pair shares one offset."""
        out = analyze_prompt(FakeTokenizer(), "p", "a b c d a b c d")
        if out["n_distinct_induction_offsets"] == 1:
            assert "single_offset" in out["flags"]

    def test_too_few_pairs_is_insufficient(self):
        out = analyze_prompt(FakeTokenizer(), "p", "a b c d e f g")
        assert out["verdict"] == "insufficient"

    def test_usable_prompt(self):
        out = analyze_prompt(FakeTokenizer(), "p", _USABLE_TEXT)
        assert out["verdict"] == "usable"
        assert out["n_induction"] >= 3
        assert out["n_same_content_nonsink"] >= 3

    def test_sink_only_null_needs_enough_non_sink_pairs(self):
        """
        The criterion is the count of null pairs surviving removal of the
        sink column, not the fraction — with three pairs a fraction is noise.
        """
        out = analyze_prompt(FakeTokenizer(), "p", _USABLE_TEXT)
        assert "null_is_sink" not in out["flags"]

    def test_induction_definitions_diverge_on_periodic_text(self):
        """
        The repo matches on ids[key-1] == ids[query-1]; the standard induction
        definition matches on ids[key-1] == ids[query]. They differ by one
        position, so on periodic text they select DISJOINT pair sets at
        offsets differing by 1 — and with rotary, a_frac depends on offset.
        Which definition is in force is therefore not a naming question.
        """
        text = " ".join(["alpha beta gamma delta epsilon zeta"] * 4)
        out = analyze_prompt(FakeTokenizer(), "p", text)
        assert out["n_induction"] > 0 and out["n_induction_strict"] > 0
        assert out["condition_agreement"] == 0.0


class TestStructureHash:

    def test_same_text_same_tokenizer_matches(self):
        a = analyze_prompt(FakeTokenizer(), "p", "a b a b")
        b = analyze_prompt(FakeTokenizer(), "p", "a b a b")
        assert a["structure_hash"] == b["structure_hash"]

    def test_tokenizer_swap_changes_it(self):
        """
        The text hash is blind to this. The structure hash is the field that
        catches a tokenizer change that leaves the battery text untouched.
        """
        text = "alpha beta alpha beta"
        a = analyze_prompt(FakeTokenizer(), "p", text)
        b = analyze_prompt(MergingTokenizer(), "p", text)
        assert a["structure_hash"] != b["structure_hash"]


class TestBatteryVerification:

    def _battery(self):
        return {
            "usable": _USABLE_TEXT,
            "repeated_tokens": ". " * 40,
            "short": "one two three",
        }

    def test_report_separates_usable_from_degenerate(self):
        rep = verify_battery_structure(FakeTokenizer(), self._battery())
        assert rep["usable_names"] == ["usable"]
        by_name = {p["name"]: p for p in rep["per_prompt"]}
        assert by_name["repeated_tokens"]["verdict"] == "degenerate"
        assert by_name["short"]["verdict"] == "insufficient"

    def test_ok_when_one_prompt_usable(self):
        assert verify_battery_structure(FakeTokenizer(), self._battery())["ok"]

    def test_not_ok_when_none_usable(self):
        rep = verify_battery_structure(MergingTokenizer(), self._battery())
        assert rep["ok"] is False
        assert rep["n_usable"] == 0

    def test_assert_raises_with_a_per_prompt_breakdown(self):
        with pytest.raises(ValueError) as e:
            assert_battery_structure(MergingTokenizer(), self._battery(),
                                     context="P6-I2b")
        msg = str(e.value)
        assert "P6-I2b" in msg
        assert "tokenizer artifact" in msg
        assert "repeated_tokens=" in msg

    def test_assert_returns_report_on_success(self):
        rep = assert_battery_structure(FakeTokenizer(), self._battery())
        assert rep["n_usable"] >= 1

    def test_require_usable_threshold(self):
        with pytest.raises(ValueError):
            assert_battery_structure(FakeTokenizer(), self._battery(),
                                     require_usable=2)

    def test_battery_hash_is_order_independent_of_dict_order(self):
        b = self._battery()
        rev = dict(reversed(list(b.items())))
        assert (verify_battery_structure(FakeTokenizer(), b)["battery_structure_hash"]
                == verify_battery_structure(FakeTokenizer(), rev)["battery_structure_hash"])

    def test_summary_flags_missing_bos(self):
        rep = verify_battery_structure(FakeTokenizer(), self._battery())
        text = "\n".join(battery_summary_lines(rep))
        assert "NO (position 0 is a content token)" in text
        assert "uniform" in text
