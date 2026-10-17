"""
`trackA_force/a2_ablate.py`'s mean-ablation hooks on a tiny random GPT-NeoX (no download): no set
changes nothing; a head ablated to its own per-token mean leaves its block's update at that mean;
the MLP's output is replaced at positions 1…n−1, position 0 untouched.

Needs the real torch/transformers: ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
import numpy as np
import pytest

from trackA_force import a2_ablate as ab
from trackA_force import a2_comp as ac

pytestmark = pytest.mark.smoke


def test_mean_ablation_hooks_on_a_tiny_model():
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
        hs, rows = hk.run_read(ids, ln_from(m), [3.5] * 3, nodes=64, part_sum=True)
        hk.close()
    finally:
        ac.COMPS = old
    k = 6
    means = np.concatenate([rows[L]["part_sum"] / 23.0 for L in range(3)])     # (3·6, d)
    A = ab.Ablate(m, means)
    with torch.no_grad():
        base = m(input_ids=ids, output_hidden_states=True).hidden_states
        A.set([])
        same = m(input_ids=ids, output_hidden_states=True).hidden_states
        assert max(float((a - b).abs().max()) for a, b in zip(base, same)) == 0.0
        A.set([0 * k + 1, 0 * k + 5])                                         # block 0: head 1, the MLP
        hs2 = m(input_ids=ids, output_hidden_states=True).hidden_states
    A.close()
    d_base = (base[1] - base[0])[0].double().numpy()
    d_abl = (hs2[1] - hs2[0])[0].double().numpy()
    assert np.abs(d_abl[0] - d_base[0]).max() < 1e-5                         # position 0 untouched
    want = d_base[1:].copy()
    # block 0's input is the embedding, so only head 1 and the MLP change there, each swapped for its
    # own mean over rows 1…: the update's mean over those rows is unchanged
    assert np.abs(d_abl[1:].mean(0) - want.mean(0)).max() < 1e-4 * np.abs(want).max()
    assert np.abs(d_abl[1:] - want).max() > 1e-3                             # and it did change something
