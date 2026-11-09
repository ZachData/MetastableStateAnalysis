"""C4 H5 (`trackC_writeup/c4_train.py`): the loop against a hand-written AdamW on a tiny GPT-NeoX
in float64 (T5's known answer, `tests/test_trackT_t5h_smoke.py`, at rel 1e-10), three updates at
rates 0, peak, peak/2 (an update at rate 0 still enters Adam's moments, as NeoX's does).
    SMOKE_REAL_DEPS=1 python -m pytest tests/test_trackC_c4_smoke.py -m smoke
"""
import copy

import pytest

pytestmark = pytest.mark.smoke
from trackC_writeup import c4_train as ct


def _tiny():
    torch = pytest.importorskip("torch")
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=64, rotary_pct=0.25, max_position_embeddings=64,
                        use_parallel_residual=True, tie_word_embeddings=False,
                        attention_dropout=0.0, hidden_dropout=0.0)
    cfg._attn_implementation = "sdpa"                  # training's (the eager mask overflows in float64)
    torch.manual_seed(0)
    m = GPTNeoXForCausalLM(cfg).double()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "layernorm" in n and n.endswith("weight"):
                p.uniform_(0.5, 1.5)                    # gains away from 1 so the decay exclusion matters
            if n.endswith(".bias"):
                p.uniform_(-0.1, 0.1)
    return torch, m


def _ref(model, rows, lrs):
    import torch
    import torch.nn.functional as F
    ps = list(model.named_parameters())
    m = [torch.zeros_like(p) for _, p in ps]
    v = [torch.zeros_like(p) for _, p in ps]
    b1, b2 = ct.BETAS
    for t, lr in enumerate(lrs, start=1):
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=rows[:, :-1]).logits
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), rows[:, 1:].reshape(-1)).backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for _, p in ps))
        coef = min(1.0, ct.CLIP / (float(gn) + 1e-6))
        with torch.no_grad():
            for i, (n, p) in enumerate(ps):
                g = p.grad * coef
                if ct.decays(n, p):
                    p.mul_(1 - lr * ct.WD)
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                p.sub_(lr * (m[i] / (1 - b1 ** t)) / ((v[i] / (1 - b2 ** t)).sqrt() + ct.EPS))


@pytest.mark.parametrize("clip", [1e9, 1.0])
def test_the_loop_equals_a_hand_written_adamw_in_float64(clip, monkeypatch):
    torch, model = _tiny()
    monkeypatch.setattr(ct, "CLIP", clip)
    ref = copy.deepcopy(model)
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    rows = torch.randint(0, 64, (4, 17), generator=torch.Generator().manual_seed(1))
    lrs = (0.0, 0.05, 0.025)
    opt = ct.make_opt(model)
    for lr in lrs:
        ct.step_once(model, opt, rows, lr, n_micro=2, amp=False)
    _ref(ref, rows, lrs)
    worst = 0.0
    for (n, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        d_loop, d_ref = p.detach() - p0[n], q.detach() - p0[n]
        worst = max(worst, float((d_loop - d_ref).abs().max() / d_ref.abs().max()))
    assert worst <= 1e-10, worst
