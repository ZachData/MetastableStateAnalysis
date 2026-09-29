"""
tests/test_phase1d_long_prompts.py — the committed long prompts obey their
rule (`p1d_cluster_ensemble/long_prompts.py`).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.deps

import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

from p1d_cluster_ensemble import long_prompts as lp  # noqa: E402

# conftest stubs `core.config` with short fixture prompts; the rule is about
# the real v1 texts, so read the real module by path.
_spec = importlib.util.spec_from_file_location(
    "_real_core_config", Path(__file__).resolve().parents[1] / "core" / "config.py")
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)
PROMPTS = _real.PROMPTS


def test_each_long_prompt_starts_with_its_v1_text():
    for key, text in lp.load().items():
        v1 = key[: -len("_long")]
        assert v1 in lp.SOURCE_KEYS and v1 not in lp.REFUSED
        assert text.startswith(PROMPTS[v1]) and len(text) > len(PROMPTS[v1])


def test_long_prompts_stay_out_of_the_battery():
    assert not set(lp.load()) & set(PROMPTS)


def test_join_refuses_an_ambiguous_ending():
    with pytest.raises(ValueError):
        lp._after("abc end of it. xyz end of it. more", "start. end of it.", "k")
