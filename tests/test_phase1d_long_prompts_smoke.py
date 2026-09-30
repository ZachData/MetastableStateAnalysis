"""
tests/test_phase1d_long_prompts_smoke.py — with the real tokenizer, each
long prompt's first tokens are v1's and it fits Pythia's context (rules 2, 4).
Run with ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_prefix_tokens_and_cap():
    transformers = pytest.importorskip("transformers")
    from core.config import PROMPTS
    from p1d_cluster_ensemble import long_prompts as lp
    try:
        t = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    except Exception:
        pytest.skip("pythia tokenizer not cached")
    for key, text in lp.load().items():
        ids, ids_v1 = t(text)["input_ids"], t(PROMPTS[key[: -len("_long")]])["input_ids"]
        assert ids[: len(ids_v1)] == ids_v1
        assert len(ids) <= lp.MAX_TOKENS
