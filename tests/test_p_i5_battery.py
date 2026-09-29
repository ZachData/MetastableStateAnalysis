"""
tests/test_p_i5_battery.py — P-I5 reads its pinned 8 prompts, not the live
battery (STATE.md Blocked 2; LESSONS.md lesson 3).

Every P-I5 loop used to iterate core.config.PROMPTS, so battery v2 made the
registered gate a 20-prompt test under an 8-prompt calibration; the nightly
smoke's `assert 20 == 8` was the only thing that noticed. The real battery is
read off core/config.py with ast, because tests/conftest.py replaces
core.config with a 2-prompt stub (LESSONS.md lesson 4).
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.pure

REPO = Path(__file__).resolve().parent.parent

from core.holdout import V1_PROMPT_KEYS  # noqa: E402
from p7_motifs.p_i5_gate import (  # noqa: E402
    DEGENERATE_PROMPT, P_I5_BATTERY_HASH, p_i5_battery,
)

#: v1's order, the order claims/calibration/p_i5_real_ablation.json ran in.
CALIBRATED_ORDER = [
    "short_heterogeneous", "wiki_paragraph", "sullivan_ballou", "paper_excerpt",
    "homer_iliad", "hdbscan_code", "camus_letranger", "latex_monograph",
]


def _real_prompts() -> dict:
    tree = ast.parse((REPO / "core" / "config.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "PROMPTS" for t in node.targets)):
            return ast.literal_eval(node.value)
    raise AssertionError("PROMPTS not found in core/config.py")


def test_the_real_battery_gives_the_calibrated_8_in_order():
    prompts = _real_prompts()
    assert len(prompts) > 9                     # the live battery has grown
    battery = p_i5_battery(prompts)
    assert [k for k, _ in battery] == CALIBRATED_ORDER
    assert all(text == prompts[k] for k, text in battery)
    assert set(CALIBRATED_ORDER) == V1_PROMPT_KEYS - {DEGENERATE_PROMPT}


def test_new_prompts_anywhere_do_not_change_it():
    prompts = _real_prompts()
    grown = {"new_first": "x y z"}
    for k, v in prompts.items():
        grown[k] = v
        grown[k + "_sibling"] = v + " more"
    assert p_i5_battery(grown) == p_i5_battery(prompts)


def test_a_changed_v1_text_refuses():
    prompts = dict(_real_prompts())
    prompts["wiki_paragraph"] += " "
    with pytest.raises(ValueError, match=P_I5_BATTERY_HASH):
        p_i5_battery(prompts)


def test_a_missing_v1_prompt_refuses():
    prompts = dict(_real_prompts())
    del prompts["homer_iliad"]
    with pytest.raises(ValueError, match="homer_iliad"):
        p_i5_battery(prompts)


def test_the_degenerate_prompt_is_not_required():
    prompts = dict(_real_prompts())
    del prompts[DEGENERATE_PROMPT]
    assert [k for k, _ in p_i5_battery(prompts)] == CALIBRATED_ORDER


@pytest.mark.parametrize("module", ["p7_motifs.p_i5_ablation",
                                    "p7_motifs.p_i5_structured_control"])
def test_run_all_on_loaded_model_runs_the_8(module, monkeypatch):
    # The smoke test's `n_prompts == 8`, without the model: run_prompt is
    # replaced, and core.config (the conftest stub) gets the real battery.
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(sys.modules["core.config"], "PROMPTS", _real_prompts())
    seen = []

    def fake_run_prompt(model, tokenizer, text, rng, *args, **kwargs):
        seen.append(text)
        return {"delta_geometric": float(rng.standard_normal()),
                "delta_logit": float(rng.standard_normal())}

    monkeypatch.setattr(mod, "run_prompt", fake_run_prompt)
    result = mod.run_all_on_loaded_model(None, None, target_head=(3, 6), seed=1)
    real = _real_prompts()
    assert result["n_prompts"] == 8
    assert list(result["per_prompt"]) == CALIBRATED_ORDER
    assert seen == [real[k] for k in CALIBRATED_ORDER]
    if module.endswith("ablation"):
        assert result["battery_hash"] == P_I5_BATTERY_HASH


def test_no_p_i5_module_iterates_the_live_battery():
    # The pin holds only if every P-I5 loop goes through p_i5_battery.
    for path in sorted((REPO / "p7_motifs").glob("p_i5_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.For)
                    and isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Attribute)
                    and node.iter.func.attr == "items"
                    and getattr(node.iter.func.value, "id", None) == "PROMPTS"):
                pytest.fail(f"{path.name}:{node.lineno} iterates PROMPTS directly")
