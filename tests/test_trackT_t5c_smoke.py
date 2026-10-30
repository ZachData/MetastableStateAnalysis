"""T5 U2 (`trackT_timaeus/t5c.py`): a `gpt-2-small` checkpoint loads as Pythia-160m's GPT-NeoX with
every weight placed (skipped where the checkpoint is not cached).
    SMOKE_REAL_DEPS=1 python -m pytest tests/test_trackT_t5c_smoke.py -m smoke
"""
import pytest

pytestmark = pytest.mark.smoke
from trackT_timaeus import t5, t5c


def test_gpt2_checkpoint_loads():
    torch = pytest.importorskip("torch")
    try:
        t5.locate(t5c.GPT2, "main", t5c.DEFAULT_ROOTS)
        m, info = t5c.load_gpt2(100, t5c.DEFAULT_ROOTS, "cpu")
    except Exception as e:  # not cached here (CI)
        if "in no cache" in str(e) or "LocalEntryNotFound" in type(e).__name__ or "offline" in str(e).lower():
            pytest.skip(str(e))
        raise
    assert info["stored_dtypes"] and next(m.parameters()).dtype == torch.float32
    assert sum(p.numel() for p in m.parameters()) > 1e8
    o = m(input_ids=torch.arange(1, 12)[None], output_hidden_states=True)
    assert len(o.hidden_states) == 13 and torch.isfinite(o.hidden_states[-1]).all()
