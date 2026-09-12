"""`p7d_redundancy/fv_score.py`'s prompt construction.

The forward-pass half needs real weights and is exercised by running the
script. What is tested here is the part that can be silently wrong while
producing a plausible-looking number: **the corruption**. If the label
derangement fails, the "corrupted" prompt still carries the task, the baseline
`P(answer)` is the clean one, and every CIE collapses toward zero — a null
result manufactured by a bug, indistinguishable from a null result about the
model. PROJECT.md §3.18 rests on that difference.
"""
import numpy as np
import pytest

from p7d_redundancy.fv_score import (
    TASKS, answer_token, build_prompt, group_by_length, make_prompts,
)

#: `pure` — a fake tokeniser and no model, so no torch forward pass is needed.
pytestmark = pytest.mark.pure


class FakeTok:
    """Whitespace tokeniser: one id per word, ':' and '\\n' their own ids.

    Enough to exercise the prompt arithmetic without loading a model. `multi`
    names words that deliberately tokenise to two ids, so the single-token
    filter has something to reject.
    """

    def __init__(self, multi=()):
        self.multi = set(multi)
        self.vocab = {}

    def _id(self, s):
        return self.vocab.setdefault(s, len(self.vocab) + 1)

    def __call__(self, text, return_tensors=None):
        ids = []
        for line in text.split("\n"):
            if line != text.split("\n")[0] or text.startswith("\n"):
                pass
            for chunk in line.split(":"):
                for w in chunk.split():
                    ids.extend([self._id(w), self._id(w + "#2")]
                               if w in self.multi else [self._id(w)])
                ids.append(self._id(":"))
            ids.pop()
            ids.append(self._id("\n"))
        ids.pop()
        return {"input_ids": ids}


def test_answer_token_rejects_multi_token_answers():
    tok = FakeTok(multi={"cold"})
    assert answer_token(tok, "small") is not None
    assert answer_token(tok, "cold") is None, (
        "a two-token answer must be dropped: the readout is the probability of "
        "ONE token and a multi-token answer has a different chance level")


def test_make_prompts_deranges_every_label():
    """No demonstration may keep its own label, or the task survives corruption."""
    tok = FakeTok()
    pairs = [(f"x{i}", f"y{i}") for i in range(12)]
    prompts, n_usable = make_prompts(tok, pairs, n_shot=5, n_prompts=20,
                                     rng=np.random.default_rng(0))
    assert n_usable == 12
    assert len(prompts) == 20
    for p in prompts:
        assert p["clean"] != p["corrupt"], "corruption did not change the prompt"


def test_corruption_preserves_query_and_answer():
    """Only the demonstrations' labels move: the query and the correct answer
    are the same in both arms, so CIE is a difference on one readout."""
    tok = FakeTok()
    pairs = [(f"x{i}", f"y{i}") for i in range(12)]
    prompts, _ = make_prompts(tok, pairs, n_shot=5, n_prompts=10,
                              rng=np.random.default_rng(1))
    for p in prompts:
        # The query and trailing ':' are the tail of both arms.
        assert p["clean"][-2:] == p["corrupt"][-2:]
        assert len(p["clean"]) == len(p["corrupt"])
        assert p["answer_id"] == answer_token(tok, p["answer"])


def test_derangement_is_a_real_derangement():
    """Directly: the corrupted labels are a permutation of the clean ones with
    no fixed point. Read back off the token ids rather than trusting the loop."""
    tok = FakeTok()
    pairs = [(f"x{i}", f"y{i}") for i in range(12)]
    rng = np.random.default_rng(2)
    prompts, _ = make_prompts(tok, pairs, n_shot=6, n_prompts=30, rng=rng)
    colon, nl = tok._id(":"), tok._id("\n")
    for p in prompts:
        def labels(ids):
            # each demonstration is [x, ':', y, '\n']
            return [ids[i + 2] for i in range(0, len(ids) - 2, 4)]
        c, k = labels(p["clean"]), labels(p["corrupt"])
        assert sorted(c) == sorted(k), "labels must be a permutation, not new words"
        assert all(a != b for a, b in zip(c, k)), "a label kept its own slot"
        assert p["clean"][1] == colon and p["clean"][3] == nl


def test_group_by_length_partitions_without_padding():
    items = [{"k": [1, 2, 3]}, {"k": [1, 2]}, {"k": [4, 5, 6]}, {"k": [7]}]
    g = group_by_length(items, "k")
    assert g == {3: [0, 2], 2: [1], 1: [3]}
    assert sum(len(v) for v in g.values()) == len(items), "every item placed once"
    for ln, idxs in g.items():
        assert all(len(items[i]["k"]) == ln for i in idxs)


def test_build_prompt_ends_on_the_query():
    tok = FakeTok()
    ids = build_prompt(tok, [("a", "b"), ("c", "d")], "e")
    assert ids[-2:] == [tok._id("e"), tok._id(":")], (
        "the model must be answering the query, not reading a label")


@pytest.mark.parametrize("name", sorted(TASKS))
def test_task_pairs_are_unique_and_nontrivial(name):
    pairs = TASKS[name]
    xs = [x for x, _ in pairs]
    assert len(set(xs)) == len(xs), f"{name} has a duplicated input"
    assert all(x != y for x, y in pairs), f"{name} has an identity pair"
    assert len(pairs) >= 12, (
        f"{name} needs enough pairs that n_shot+1 can be drawn without replacement")
