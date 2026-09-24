"""
`tools/run/p10_ext_sem_threshold.py` — Stage 1 step 1, the threshold sweep.

The runner rebuilds a cosine that was never stored and trusts it only because
it reproduces the stored tags, so the first tests check that the reproduction
gate fires. The rest check the two properties the design rests on: quantile
cuts ignore any increasing map of the Gram (the scale control), and the
verdict follows the rule written in the module docstring.
"""
import json

import numpy as np
import pytest

from tools.run.p10_ext_sem_threshold import (
    FROZEN_STEP,
    ReproductionError,
    aggregate,
    layer0_gram,
    measure_layer,
    measure_run,
    offdiag_sorted,
    read_tokens,
    verdicts,
)

# Tier: numpy only -- runs in `scripts/check.sh pure`.
pytestmark = pytest.mark.pure


def _unit(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _write_run(tmp_path, name, act0, pairs, labels, n_ext=None):
    """A run dir with activations.npz (layer 0 only matters) and one layer."""
    d = tmp_path / name
    d.mkdir()
    np.savez_compressed(d / "activations.npz", activations=act0[None])
    gram = act0 @ act0.T
    mp = [{"i": i, "j": j, "tok_i": f"t{i}", "tok_j": f"t{i}" if i == 0 else f"t{j}",
           "cross_method_tag": "same_cluster" if labels[i] == labels[j] != -1 else "noise"}
          for i, j in pairs]
    if n_ext is None:
        n_ext = int(sum(gram[i, j] > 0.5 for i, j in pairs))
    toks = [f"t{k}" for k in range(len(act0))]
    for i, j in pairs:
        if i == 0:
            toks[j] = toks[i]
    (d / "tokens.txt").write_text("".join(f"{k:>3}  {t}\n" for k, t in enumerate(toks)))
    (d / "clustering.json").write_text(json.dumps({"layers": [{
        "layer": 0, "pair_agreement": {
            "mutual_pairs": mp, "n_ext_semantic": n_ext, "ext_sem_threshold": 0.5}}]}))
    return d


def _frames(d):
    g = layer0_gram(d)
    return {"self": (g, offdiag_sorted(g))}


def test_stored_tags_are_reproduced(tmp_path):
    rng = np.random.default_rng(0)
    a = _unit(rng, 12, 3)  # low d: plenty of cosines on both sides of 0.5
    d = _write_run(tmp_path, "r", a, [(0, 1), (2, 3), (4, 5)], [0, 0, 1, 1, -1, 2] + [0] * 6)
    out = measure_run(d, _frames(d))
    assert out["self"][0]["n_pairs"] == 3
    assert out["self"][0]["identical_token_fraction"] == pytest.approx(1 / 3)


def test_a_mismatch_with_the_stored_count_is_refused(tmp_path):
    rng = np.random.default_rng(0)
    a = _unit(rng, 12, 3)
    pairs = [(0, 1), (2, 3), (4, 5)]
    true_n = int(sum((a @ a.T)[i, j] > 0.5 for i, j in pairs))
    d = _write_run(tmp_path, "r", a, pairs, [0] * 12, n_ext=true_n + 1)
    with pytest.raises(ReproductionError):
        measure_run(d, _frames(d))


def test_an_empty_pair_record_is_refused(tmp_path):
    a = _unit(np.random.default_rng(0), 6, 3)
    d = _write_run(tmp_path, "r", a, [], [0] * 6, n_ext=0)
    with pytest.raises(ReproductionError):
        measure_run(d, _frames(d))


def test_quantile_cuts_ignore_an_increasing_map_of_the_gram():
    rng = np.random.default_rng(1)
    a = _unit(rng, 40, 8)
    g = a @ a.T
    i, j = np.array([0, 2, 4, 6, 8]), np.array([1, 3, 5, 7, 9])
    same = np.array([True, False, True, True, False])
    base = measure_layer(g[i, j], same, offdiag_sorted(g))
    g2 = 0.2 + 0.5 * g ** 3 + 0.3 * g  # strictly increasing on [-1, 1]
    moved = measure_layer(g2[i, j], same, offdiag_sorted(g2))
    for c in base["ext_semantic_fraction"]:
        if c.startswith("q_"):
            assert base["ext_semantic_fraction"][c] == moved["ext_semantic_fraction"][c]
    assert base["mean_pair_percentile"] == moved["mean_pair_percentile"]
    # ...while the absolute cuts do move: that is the artifact being controlled.
    assert any(base["ext_semantic_fraction"][c] != moved["ext_semantic_fraction"][c]
               for c in base["ext_semantic_fraction"] if c.startswith("abs_"))


def _by_step(a0, a1):
    def rec(v):
        return {"ext_semantic_fraction": v, "mean_pair_percentile": 0.5}
    return {"self": {0: rec(a0), FROZEN_STEP: rec(a1)}}


def test_verdict_survives_dead_mixed():
    q = {"q_0.5": 0.8, "q_0.9": 0.4}
    assert verdicts(_by_step({**q, "abs_0.5": 0.8}, {"q_0.5": 0.7, "q_0.9": 0.3, "abs_0.5": 0.7}))["self"]["verdict"] == "survives"
    assert verdicts(_by_step({**q, "abs_0.5": 0.8}, {"q_0.5": 0.7, "q_0.9": 0.4, "abs_0.5": 0.7}))["self"]["verdict"] == "dead"
    assert verdicts(_by_step({**q, "abs_0.5": 0.6}, {"q_0.5": 0.7, "q_0.9": 0.3, "abs_0.5": 0.7}))["self"]["verdict"] == "mixed"


def test_a_saturated_absolute_cut_is_not_judged():
    v = verdicts(_by_step({"q_0.5": 0.8, "abs_0.1": 1.0}, {"q_0.5": 0.7, "abs_0.1": 1.0}))["self"]
    assert v["verdict"] == "survives" and "abs_0.1" not in v["judged_cuts"]


def test_aggregate_means_over_prompts_and_layers(tmp_path):
    rng = np.random.default_rng(2)
    results = {}
    for step in (0, FROZEN_STEP):
        for k in ("p", "q"):
            a = _unit(rng, 12, 3)
            d = _write_run(tmp_path, f"{step}{k}", a, [(0, 1), (2, 3)], [0] * 12)
            results[(step, k)] = measure_run(d, _frames(d))
    s = aggregate(results)["by_step"]["self"][0]
    want = np.mean([results[(0, k)]["self"][0]["ext_semantic_fraction"]["abs_0.5"] for k in "pq"])
    assert s["n_runs"] == 2 and s["ext_semantic_fraction"]["abs_0.5"] == pytest.approx(want)


def test_tokens_txt_round_trips_its_writer_format(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "tokens.txt").write_text("  0  \u0120part\n  1  ,\n  2  \u0120part\n")
    assert list(read_tokens(d)) == ["\u0120part", ",", "\u0120part"]


def test_repeats_at_cosine_one_do_not_leak_into_the_non_repeat_block(tmp_path):
    """The real data's case: repeated tokens share one vector, a pile at cos 1.
    Most mutual pairs are repeats; the non-repeat block must see only the rest,
    and rank them against non-repeat pairs, not the pile."""
    rng = np.random.default_rng(3)
    base = _unit(rng, 6, 16)
    a = base[[0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5]]      # heavy repetition
    toks = ["a", "a", "a", "a", "b", "b", "b", "c", "d", "e", "f", "f"]
    d = tmp_path / "r"
    d.mkdir()
    np.savez_compressed(d / "activations.npz", activations=a[None])
    (d / "tokens.txt").write_text("".join(f"{k:>3}  {t}\n" for k, t in enumerate(toks)))
    pairs = [(0, 1), (4, 5), (10, 11), (7, 8)]           # three repeats, one not
    g = a @ a.T
    mp = [{"i": i, "j": j, "tok_i": toks[i], "tok_j": toks[j], "cross_method_tag": "noise"}
          for i, j in pairs]
    (d / "clustering.json").write_text(json.dumps({"layers": [{"layer": 0, "pair_agreement": {
        "mutual_pairs": mp, "ext_sem_threshold": 0.5,
        "n_ext_semantic": int(sum(g[i, j] > 0.5 for i, j in pairs))}}]}))
    out = measure_run(d, _frames(d))
    lr = out["self"][0]
    assert lr["identical_token_fraction"] == 0.75
    assert lr["non_repeat"]["n"] == 1
    # the all-pairs 0.9 quantile sits in the pile; the non-repeat one does not
    assert np.quantile(offdiag_sorted(g), 0.9) > 0.999
    nr_rank = lr["non_repeat"]["mean_percentile"]
    assert 0.0 <= nr_rank < 1.0 and lr["non_repeat"]["frac_above_q"]["q_0.99"] in (0.0, 1.0)


def test_a_tokens_file_that_disagrees_with_the_stored_pairs_is_refused(tmp_path):
    a = _unit(np.random.default_rng(0), 12, 3)
    d = _write_run(tmp_path, "r", a, [(0, 1), (2, 3)], [0] * 12)
    (d / "tokens.txt").write_text("".join(f"{k:>3}  x\n" for k in range(12)))
    with pytest.raises(ReproductionError):
        measure_run(d, _frames(d))
