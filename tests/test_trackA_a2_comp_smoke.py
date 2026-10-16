"""
`trackA_force/a2_comp.py`'s hooked split on a tiny random GPT-NeoX (no download): heads + b_O add up
to attention's output, the counted parts to the block's update, the path integral to the chord.

Needs the real torch/transformers (the deps tier's are stubs): ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
import pytest

from trackA_force import a2_comp as ac

pytestmark = pytest.mark.smoke


def test_hooked_split_on_a_tiny_model():
    torch = pytest.importorskip("torch")
    from test_p1e_u2_attn import _tiny
    from p1e_energy_field.u2_attn import ln_from
    m = _tiny()
    m.config.num_attention_heads = 4
    old = ac.COMPS
    ac.COMPS = tuple(f"h{h}" for h in range(4)) + ("b_O", "mlp")
    try:
        hk = ac.comp_hooked(m)
        ids = torch.randint(0, 97, (1, 24))
        ln = ln_from(m)
        hs, rows = hk.run_read(ids, ln, [3.5] * 3, nodes=64)   # block 0's update is 25× its input here
        hk.close()
    finally:
        ac.COMPS = old
    assert sorted(rows) == [0, 1, 2]
    assert ac.check_recon(hs, rows) < 1e-5
    for L in rows:
        assert rows[L]["split_rel"] < 1e-5 and rows[L]["g0"] < 1e-6
        assert len(rows[L]["proj"]) == 6 + 8
