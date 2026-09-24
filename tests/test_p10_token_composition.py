"""
`tools/run/p10_token_composition.py` — Stage 1 step 2, the token-composition table.

The table is only as good as its features and its refusals, so these check the
byte-level decoding and classes, the three-way copies column (the post-hoc
split that separates a first copy with later twins from a unique token), and
that a run with non-native, mismatched or all-noise labels is refused.
"""
import json

import numpy as np
import pytest

from tools.run.p10_token_composition import (
    CompositionError,
    cluster_count_summary,
    contrast_verdict,
    decode,
    measure_run,
    rank_level,
    token_class,
    token_features,
)

# Tier: numpy only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure

VOCAB = {"The": 464, "Ġcat": 3797, "Ġsat": 3332, ".": 15, "Ġthe": 262,
         "Ċ": 187, "ing": 272, "Ã": 127, "12": 805, "    ": 50260}
ADDED = {50260}


def test_decode_byte_level():
    assert decode("Ġcat") == " cat"
    assert decode("Ċ") == "\n"
    assert decode("Ã") is None          # half of a two-byte character


def test_classes():
    assert token_class(" cat", None, 3) == "word_start"
    assert token_class("ing", " walk", 4) == "continuation"
    assert token_class("ing", "\n", 4) == "word_start"
    assert token_class("The", None, 0) == "word_start"
    assert token_class(".", "cat", 2) == "punct"
    assert token_class("\n", ".", 3) == "whitespace"
    assert token_class(" 12", None, 5) == "numeric"
    assert token_class(None, None, 5) == "byte_fragment"


def test_rank_levels_and_added_tokens():
    assert rank_level(15) == "<1k"
    assert rank_level(3797) == "1k-5k"
    assert rank_level(25000) == ">=20k"
    assert rank_level(50260, ADDED) == "added"


def test_copies_three_way():
    toks = np.array(["The", "Ġcat", "Ġsat", ".", "Ġthe", "Ġcat", ".", "Ġcat"], dtype=object)
    f = token_features(toks, VOCAB, ADDED)
    assert [x["copies"] for x in f] == ["unique", "first", "unique", "first",
                                         "unique", "repeat", "repeat", "repeat"]
    assert f[1]["count"] == "3-5" and f[0]["count"] == "1"
    assert f[0]["pos"] == "0"


def test_added_token_is_whitespace():
    f = token_features(np.array(["The", "    "], dtype=object), VOCAB, ADDED)
    assert f[1]["cls"] == "whitespace" and f[1]["rank"] == "added"


def test_unknown_token_refused():
    with pytest.raises(CompositionError, match="not in the vocab"):
        token_features(np.array(["Ġzzzz"], dtype=object), VOCAB, ADDED)


def _run(tmp_path, labels, tokens=("The", "Ġcat", "Ġsat", ".", "Ġcat")):
    d = tmp_path / "pythia-410m-step0_wiki_paragraph"
    d.mkdir()
    (d / "tokens.txt").write_text("".join(f"{i:3d}  {t}\n" for i, t in enumerate(tokens)))
    (d / "hdbscan_labels.json").write_text(json.dumps(labels))
    return d


def test_measure_run_counts(tmp_path):
    d = _run(tmp_path, {"0": [-1, 0, -1, -1, 0], "1": [1, 0, 1, -1, 0]})
    r = measure_run(d, VOCAB, ADDED)
    assert r["cells"][0][("all", "all", "first")] == [1, 1]
    assert r["cells"][0][("all", "all", "repeat")] == [1, 1]
    assert r["cells"][0][("all", "all", "unique")] == [3, 0]
    assert r["cells"][1][("cls", "word_start", "unique")] == [2, 2]
    assert r["noise_rate"][0] == pytest.approx(0.6)


@pytest.mark.parametrize("labels,match", [
    ({"0": [-1] * 5, "1": [-1] * 5}, "noise at every layer"),
    ({"0": [0, 0, -1]}, "3 labels, 5 tokens"),
])
def test_refusals(tmp_path, labels, match):
    with pytest.raises(CompositionError, match=match):
        measure_run(_run(tmp_path, labels), VOCAB, ADDED)


def test_backfilled_labels_refused(tmp_path):
    d = _run(tmp_path, {"0": [0, 0, -1, -1, 0]})
    (d / "hdbscan_backfill.json").write_text(json.dumps({"labels": {"0": [0, 0, -1, -1, 0]}}))
    with pytest.raises(CompositionError, match="not native"):
        measure_run(d, VOCAB, ADDED)


def test_contrast_verdict_rule():
    def layer(hi, lo, ws, wd):
        return {"hi_freq": (hi, 5), "lo_freq": (lo, 5), "ws_punct": (ws, 5), "word_start": (wd, 5)}
    assert contrast_verdict([{0: layer(0.5, 0.2, 0.5, 0.2)}])["trash_collection"] == "consistent"
    assert contrast_verdict([{0: layer(0.2, 0.5, 0.2, 0.5)}])["trash_collection"] == "against"
    assert contrast_verdict([{0: layer(0.5, 0.2, 0.2, 0.5)}])["trash_collection"] == "unclear"
    assert contrast_verdict([{0: layer(None, 0.2, 0.5, 0.2)}])["trash_collection"] == "unavailable"


def test_cluster_count_columns(tmp_path):
    # tokens The cat sat . cat: one repeated type (cat)
    d = _run(tmp_path, {"0": [-1, 0, -1, -1, 0], "1": [1, 0, 1, -1, 0]})
    cc = measure_run(d, VOCAB, ADDED)["cluster_count"]
    assert cc["n_repeated_types"] == 1
    assert cc["by_layer"][0] == {"n_clusters": 1, "single_type": 1, "holds_repeat": 1}
    assert cc["by_layer"][1] == {"n_clusters": 2, "single_type": 1, "holds_repeat": 1}
    assert cc["max_alive"] == 2 and cc["max_alive_layers"] == [1]


def test_cluster_count_summary_drops_repeated_tokens():
    def run(n_rep, n_cl):
        return {"cluster_count": {"n_repeated_types": n_rep, "max_alive": n_cl,
                                  "max_alive_layers": [0], "by_layer": {0: {"n_clusters": n_cl, "single_type": n_cl,
                                                   "holds_repeat": 0}}}}
    s = cluster_count_summary({"a": run(4, 4), "repeated_tokens": run(10, 1)})
    assert s["all"][0]["ratio"] == pytest.approx((1.0 + 0.1) / 2)
    assert s["without_repeated_tokens"][0]["ratio"] == pytest.approx(1.0)
    assert s["without_repeated_tokens"]["n_runs"] == 1
    assert s["all"]["max_alive_at_layer0"] == 2 and s["without_repeated_tokens"]["max_alive"] == 4
