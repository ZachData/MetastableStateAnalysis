"""core/holdout.py: the Phase 10 confirmation set is refused in code."""

import ast
import json
from pathlib import Path

import pytest

from core.holdout import (
    HELD_OUT_PROMPT_KEYS,
    V1_PROMPT_KEYS,
    HoldoutError,
    held_out_reason,
    refuse_held_out,
)

pytestmark = pytest.mark.pure

REPO = Path(__file__).resolve().parents[1]


def _battery_keys():
    """PROMPTS's keys in source order, read off core/config.py with ast:
    importing it needs torch, and tests/conftest.py stubs it."""
    tree = ast.parse((REPO / "core" / "config.py").read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "PROMPTS" for t in node.targets)):
            return [k.value for k in node.value.keys]
    raise AssertionError("PROMPTS not found in core/config.py")


def test_key_lists_match_the_battery():
    keys = _battery_keys()
    assert set(keys) == HELD_OUT_PROMPT_KEYS | V1_PROMPT_KEYS
    assert len(HELD_OUT_PROMPT_KEYS) == 12 and len(V1_PROMPT_KEYS) == 9
    # v1's nine come first in PROMPTS; the twelve are the v2 extension.
    assert set(keys[:9]) == V1_PROMPT_KEYS


def test_no_v1_name_is_mistaken_for_a_held_out_one():
    for k in V1_PROMPT_KEYS:
        assert held_out_reason(f"pythia-410m-step0_{k}") is None
        assert held_out_reason(f"cross_model_{k}.png") is None


def _run(ts: Path, name: str, key=None) -> Path:
    d = ts / name
    d.mkdir(parents=True)
    if key is not None:
        (d / "manifest.json").write_text(json.dumps({"prompt_key": key}))
    return d


def test_what_is_held_out(tmp_path):
    ts = tmp_path / "2026-09-24_05-35-20"
    held = _run(ts, "pythia-410m-step512_moby_loomings")
    v1 = _run(ts, "pythia-410m-step512_homer_iliad")
    renamed = _run(ts, "odd_name", key="latex_article")
    v1_manifest = _run(ts, "pythia-410m-step1_latex_monograph", key="latex_monograph")
    pooled = ts / "pair_agreement.json"
    pooled.write_text("{}")
    v1_plot = ts / "pythia-410m-step512_homer_iliad_pca.png"
    v1_plot.write_text("")
    held_plot = ts / "gpt2-large_paper_svflow_pca.png"
    held_plot.write_text("")

    assert held_out_reason(held)
    assert held_out_reason(renamed)                  # by manifest, not name
    assert held_out_reason(pooled)                   # beside a held-out run
    assert held_out_reason(held_plot)
    assert held_out_reason(ts)                       # contains a held-out run
    assert held_out_reason(tmp_path / "claims" / "audits" / "claim_c_real_run.json")
    assert held_out_reason(v1) is None
    assert held_out_reason(v1_manifest) is None
    assert held_out_reason(v1_plot) is None


def test_a_pooled_file_with_no_held_out_neighbour_is_readable(tmp_path):
    ts = tmp_path / "2026-08-31_11-08-06"
    _run(ts, "pythia-410m-step0_wiki_paragraph")
    (ts / "pair_agreement.json").write_text("{}")
    assert held_out_reason(ts / "pair_agreement.json") is None


def test_refuse_drop_allow(tmp_path):
    held = _run(tmp_path, "pythia-410m-step0_wiki_byzantium")
    v1 = _run(tmp_path, "pythia-410m-step0_wiki_paragraph")

    with pytest.raises(HoldoutError, match="wiki_byzantium"):
        refuse_held_out([held, v1])

    kept, rec = refuse_held_out([held, v1], drop=True)
    assert kept == [v1] and rec["n_dropped"] == 1 and not rec["allowed"]

    kept, rec = refuse_held_out([held, v1], allow=True)
    assert kept == [held, v1] and rec["allowed"] and rec["n_held_out"] == 1

    kept, rec = refuse_held_out([v1])
    assert kept == [v1] and rec["n_held_out"] == 0

    with pytest.raises(ValueError):
        refuse_held_out([v1], allow=True, drop=True)


# Runners that read data/phase12 without the guard, and why that is safe.
EXEMPT = {
    "backfill_hdbscan.py": "producer: writes labels, checks they are populated",
    "stage0_chunk.py": "producer: runs Stage 0 and indexes it",
    "dissipation.py": "fixed v1 prompt list (SCORED_PROMPTS + repeated_tokens)",
    "dissipation_sublayer.py": "fixed v1 prompt list (SCORED_PROMPTS + repeated_tokens)",
    "relay_null.py": "prompts come from Phase 7's v1 motif table",
    "ov_per_head.py": "reads OV weights (p2_eigenspectra_*), no prompt data",
    "induction_rank_sweep.py": "reads OV weights (p2_eigenspectra_*), no prompt data",
    "induction_composition_whitening.py": "reads OV weights; its PROMPTS pass "
        "is the live-battery issue in STATE.md Blocked 2",
}


def test_every_phase12_reader_is_guarded_or_exempt():
    """A new runner over data/phase12 (Stage 1 onwards) must screen its inputs
    or say here why it need not."""
    readers = sorted(p for p in (REPO / "tools" / "run").glob("*.py")
                     if "phase12" in p.read_text())
    assert readers
    guarded = {p.name for p in readers
               if ("refuse_held_out(" in p.read_text()
                   and "add_holdout_args(ap)" in p.read_text())
               or "HELD_OUT_PROMPT_KEYS" in p.read_text()}
    names = {p.name for p in readers}
    assert not names - guarded - set(EXEMPT), \
        f"runners over data/phase12 with no holdout guard: {sorted(names - guarded - set(EXEMPT))}"
    assert not set(EXEMPT) & guarded, "guarded now: drop from EXEMPT"
    assert set(EXEMPT) <= names, f"stale EXEMPT entries: {sorted(set(EXEMPT) - names)}"
    assert all(n in guarded for n in names if n.startswith("p10_") or n == "transport.py")


def test_pooled_inputs_are_not_dropped_silently(tmp_path):
    ts = tmp_path / "2026-09-24_05-35-20"
    _run(ts, "pythia-410m-step0_moby_loomings")
    v1 = _run(ts, "pythia-410m-step0_homer_iliad")
    pooled = ts / "pair_agreement.json"
    pooled.write_text("{}")
    with pytest.raises(HoldoutError, match="per prompt"):
        refuse_held_out([v1, pooled], drop=True)
    logs = tmp_path / "stage0_logs"
    logs.mkdir()
    for n in ("chunk_x_001_step16.out", "chunk_x.log", "stage0_index.json"):
        (logs / n).write_text("")
    assert held_out_reason(logs / "chunk_x_001_step16.out")
    assert held_out_reason(logs / "chunk_x.log")
    assert held_out_reason(logs / "stage0_index.json") is None


def test_a_reader_refuses_by_default(tmp_path, monkeypatch):
    import sys
    from tools.run import p10_anchor
    _run(tmp_path / "2026-09-24_05-35-20", "pythia-410m-step0_moby_loomings")
    monkeypatch.setattr(sys, "argv", ["p10_anchor", "--root", str(tmp_path),
                                      "--out", str(tmp_path / "out.json")])
    with pytest.raises(HoldoutError):
        p10_anchor.main()
