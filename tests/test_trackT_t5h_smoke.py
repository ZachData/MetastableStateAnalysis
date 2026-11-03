"""T5 U4 (`trackT_timaeus/t5h.py`): the loop against a hand-written float32 AdamW on a tiny Llama
(design-T5.md "U4", gate row "the loop": 2 updates on fixed tokens, rel 1e-5).
    SMOKE_REAL_DEPS=1 python -m pytest tests/test_trackT_t5h_smoke.py -m smoke
"""
import copy

import pytest

pytestmark = pytest.mark.smoke
from trackT_timaeus import t5h


def _tiny():
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, intermediate_size=64, tie_word_embeddings=True,
                      max_position_embeddings=64, attn_implementation="eager")
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg).float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)                    # gains away from 1 so decay exclusion matters
    return torch, m


def _ref_adamw(model, tokens, steps, sched, n_micro=1):
    """Hand-written AdamW: the mean loss over ``n_micro`` equal micro-batches' means, global-norm
    clip, decoupled decay on 2-D weights, bias-corrected moments."""
    import torch
    import torch.nn.functional as F
    ps = [p for p in model.parameters()]
    m = [torch.zeros_like(p) for p in ps]
    v = [torch.zeros_like(p) for p in ps]
    b1, b2 = t5h.BETAS
    mb = tokens.shape[0] // n_micro
    for t in steps:
        model.zero_grad(set_to_none=True)
        for k in range(n_micro):
            x = tokens[k * mb:(k + 1) * mb]
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            (loss / n_micro).backward()
        gn = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in ps))
        coef = min(1.0, t5h.CLIP / (float(gn) + 1e-6))
        lr = sched(t)
        with torch.no_grad():
            for i, p in enumerate(ps):
                g = p.grad * coef
                if p.ndim == 2:
                    p.mul_(1 - lr * t5h.WD)
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh, vh = m[i] / (1 - b1 ** t), v[i] / (1 - b2 ** t)
                p.sub_(lr * mh / (vh.sqrt() + t5h.EPS))
    return float(gn)


def _run(clip, dtype, n_micro_ref, monkeypatch):
    """The loop (two micro-batches) and the reference (``n_micro_ref``) for two updates (warmup then
    cosine: lr peak, then peak/2); returns per-tensor max |Δloop − Δref| / max |Δref|, the grad norms."""
    import torch
    monkeypatch.setattr(t5h, "CLIP", clip)
    _, model = _tiny()
    model = model.to(dtype)
    ref = copy.deepcopy(model)
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    tokens = torch.randint(0, 64, (4, 16), generator=torch.Generator().manual_seed(1))
    sched = lambda s: t5h.lr_at(s, peak=0.05, warmup=1, total=3)
    assert (sched(1), sched(2)) == (pytest.approx(0.05), pytest.approx(0.025))
    opt = t5h.make_opt(model)
    gns = [t5h.step_once(model, opt, tokens, s, n_micro=2, amp=False, sched=sched)["grad_norm"] for s in (1, 2)]
    gn_ref = _ref_adamw(ref, tokens, (1, 2), sched, n_micro=n_micro_ref)
    assert gns[1] == pytest.approx(gn_ref, rel=1e-5)
    assert (max(gns) > t5h.CLIP) == (clip == 1.0)        # the tiny model's norms are 1.1-2.1: the clip binds
    groups = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    for n, p in model.named_parameters():
        assert groups[id(p)] == (t5h.WD if p.ndim == 2 else 0.0), n
    rel = {}
    for (n, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        d_loop, d_ref = p.detach() - p0[n], q.detach() - p0[n]
        rel[n] = float((d_loop - d_ref).abs().max() / d_ref.abs().max())
    return rel


@pytest.mark.parametrize("clip", [1e9, 1.0])
def test_the_loop_equals_a_hand_written_adamw(clip, monkeypatch):
    """The gate row "the loop", run in float64: the loop's two micro-batches against the reference's
    full batch, the clip off (1e9) and at the recipe's 1.0. Exact to 1e-10 (measured 1e-14)."""
    import torch
    rel = _run(clip, torch.float64, 1, monkeypatch)
    assert max(rel.values()) <= 1e-10, max(rel.items(), key=lambda kv: kv[1])


@pytest.mark.parametrize("clip", [1e9, 1.0])
def test_the_loop_in_float32(clip, monkeypatch):
    """The same in float32, both sides on the loop's micro-batches. The design's bar was rel 1e-5;
    measured 1.2-2.5e-5, while each float32 side is ~2.4e-4 from the float64 result (op order
    amplified by Adam's normalisation), so 1e-5 is below float32's own rounding here and the gate
    is the float64 test above (`status-T.md` "T5 U4a"). This is a sanity bound only."""
    import torch
    rel = _run(clip, torch.float32, 2, monkeypatch)
    assert max(rel.values()) <= 1e-3, max(rel.items(), key=lambda kv: kv[1])
