"""
tests/test_phase1d_neox_block_smoke.py — `p1d_cluster_ensemble/neox_block.py`
against transformers' own GPTNeoXLayer on a tiny random config. Null A
(`attention_null.py`) is only the model's map if this holds; the real
checkpoint is checked by `attention_null.py --verify` on every run.

Needs the real torch/transformers: ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------

def test_block_matches_transformers_layer():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from p1d_cluster_ensemble.neox_block import (attention_probs, block_forward,
                                                 weights_from_hf_layer)
    cfg = transformers.GPTNeoXConfig(hidden_size=32, num_attention_heads=4,
                                     intermediate_size=128, num_hidden_layers=1,
                                     rotary_pct=0.25, vocab_size=50,
                                     max_position_embeddings=64,
                                     use_parallel_residual=True)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.GPTNeoXModel(cfg).eval()
    x = torch.randn(1, 20, 32) * 5
    with torch.no_grad():
        out = model(inputs_embeds=x, output_attentions=True, output_hidden_states=True)
    W = weights_from_hf_layer(model.layers[0], cfg)
    P = attention_probs(x[0].numpy(), W)
    np.testing.assert_allclose(P, out.attentions[0][0].numpy(), atol=1e-5)
    # hidden_states[1] is after the final LN when there is one layer; compare pre-LN
    H = block_forward(x[0].numpy(), W)
    ref = model.final_layer_norm(torch.as_tensor(H)[None])[0].detach().numpy()
    np.testing.assert_allclose(ref, out.hidden_states[1][0].numpy(), atol=1e-4)
    # positions shift rotary: a sub-sequence at its own positions matches the full run's rows
    keep = np.array([0, 3, 4, 9, 15])
    Psub = attention_probs(x[0].numpy()[keep], W, positions=keep)
    assert Psub.shape == (4, 5, 5)
    np.testing.assert_allclose(Psub.sum(-1), 1.0, atol=1e-6)
