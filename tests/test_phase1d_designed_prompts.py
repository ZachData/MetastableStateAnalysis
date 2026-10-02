"""
tests/test_phase1d_designed_prompts.py — the designed-content prompts are
frozen and labelled as their rule says (`p1d_cluster_ensemble/designed_prompts.py`).
"""

from __future__ import annotations

from collections import Counter

import pytest

pytestmark = pytest.mark.deps

from p1d_cluster_ensemble import designed_prompts as dp

#: Frozen 2026-10-02 before any forward pass. A change here is a new prompt set.
FROZEN_HASH = "ae6a4312126c"


def test_texts_are_frozen():
    assert dp.designed_hash() == FROZEN_HASH


def test_list_holds_each_category_word_once_in_a_non_periodic_order():
    text, spans = dp.prompts()["designed_category_list"]
    words = [text[s:e] for s, e, _ in spans]
    assert Counter(lab for *_, lab in spans) == {"animal": 24, "colour": 24, "body": 24}
    assert len(words) == len(set(words)) == 72
    seq = [lab for *_, lab in spans]
    # category is not a function of the index mod 2, 3 or 4
    for m in (2, 3, 4):
        assert all(len({seq[i] for i in range(r, 72, m)}) == 3 for r in range(m))


def test_prose_and_code_alternate_and_cover_the_text():
    text, spans = dp.prompts()["designed_prose_code"]
    assert [lab for *_, lab in spans] == ["prose", "code"] * 4 + ["prose"]
    gaps = [text[e:s] for (_, e, _), (s, _, _) in zip(spans, spans[1:])]
    assert set(gaps) == {"\n\n"} and spans[0][0] == 0 and spans[-1][1] == len(text)


def test_every_entity_word_occurs_and_no_word_belongs_to_two():
    text, spans = dp.prompts()["designed_entities"]
    found = Counter(text[s:e] for s, e, _ in spans)
    words = [w for ws in dp.ENTITY_WORDS.values() for w in ws]
    assert len(words) == len(set(words)) and all(found[w] >= 1 for w in words)


def test_designed_keys_stay_out_of_the_battery():
    from core.holdout import V1_PROMPT_KEYS
    assert not set(dp.KEYS) & set(V1_PROMPT_KEYS)


class _Tok:
    """Whitespace-prefixed word tokenizer with offsets, enough for `token_labels`."""

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        import re
        offs = [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]
        return {"input_ids": list(range(len(offs))), "offset_mapping": offs}


def test_token_labels_reads_spans_and_refuses_a_straddle():
    ids, labels = dp.token_labels("a dog ran", [(2, 5, "animal")], _Tok())
    assert labels == [None, "animal", None]
    with pytest.raises(ValueError, match="straddles"):
        dp.token_labels("ab", [(0, 1, "x"), (1, 2, "y")], _Tok())
