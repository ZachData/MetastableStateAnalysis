"""
`p1e_energy_field/u2_heads.py` — the per-head arm on a tiny random GPT-NeoX (no download): each
head's keys and sink parts add up to attention's output, the kernel fields are the explicit sums
over keys, and every cell the rule names is read (`design-1e.md` "U2's per-head arm: the rule").

Needs the real torch/transformers (the deps tier's are stubs): ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
import numpy as np
import pytest

from p1e_energy_field import u2_block as ub
from p1e_energy_field import u2_heads as uh
from test_p1e_u2_attn import _tiny

pytestmark = pytest.mark.smoke


def _run(m, n=24):
    torch = pytest.importorskip("torch")
    from p1e_energy_field import u2_attn as ua
    from p1e_energy_field.u2_torch import Ops
    ops = Ops("cpu", torch.float64)
    ids = torch.randint(0, 97, (1, n))
    ln = ua.ln_from(m)
    t = np.arange(1, n)
    ctx = {"ln_w": ops.to(ln["w"]), "ln_b": ops.to(ln["b"]), "eps": ln["eps"],
           "t": torch.as_tensor(t), "t_np": t, "perms_for": ub.make_perms("tiny"), "tname": "t12"}
    hk = uh.HeadHooked(m, ops)
    hs, comps, per = hk.run_heads(ids, range(3), ctx)
    hk.close()
    return ids, ln, t, hs, comps, per


def test_heads_add_up_and_kernels_are_the_explicit_sums():
    torch = pytest.importorskip("torch")
    m = _tiny()
    torch.manual_seed(1)
    ids, ln, t, hs, comps, per = _run(m)
    chk = uh.check_heads(hs, per)
    assert chk["headsum_rel"] < 1e-5 and chk["upper"] == 0
    torch.set_grad_enabled(False)
    full = m(input_ids=ids, output_hidden_states=True, output_attentions=True)
    for L in range(3):
        layer = m.gpt_neox.layers[L]
        att = layer.attention
        H, hd = m.config.num_attention_heads, att.head_size
        A = full.attentions[L][0].double()                                   # (H, n, n)
        U = ub.unit_rows(hs[L], ln["w"][L], ln["b"][L], ln["eps"])
        x = layer.input_layernorm(full.hidden_states[L][0])
        qkv = att.query_key_value(x).view(-1, H, 3 * hd)
        bV = att.query_key_value.bias.view(H, 3 * hd)[:, 2 * hd:]
        V = (qkv[:, :, 2 * hd:] - bV).permute(1, 0, 2).double()              # (H, n, hd)
        Wo = att.dense.weight.view(-1, H, hd).double()
        keys_h = torch.einsum("hij,hjk,dhk->hid", A[:, :, 1:], V[:, 1:], Wo).numpy()
        sink_h = torch.einsum("hi,hk,dhk->hid", A[:, :, 0], V[:, 0], Wo).numpy()
        np.testing.assert_allclose(per[L]["mean_keys_h"], keys_h[:, t].mean(axis=1), atol=1e-5)
        np.testing.assert_allclose(per[L]["mean_sink_h"], sink_h[:, t].mean(axis=1), atol=1e-5)
        # the heads' keys parts sum to the attention arm's keys
        np.testing.assert_allclose(keys_h.sum(0), comps[L]["keys"], atol=2e-4)
        # head 0's kernel field over keys 1…i, by hand, against the alignment the pass recorded
        mns = A[0, :, 1:].numpy() @ U[1:]
        g_ns = ub.tangent(U, mns)[t]
        g_phi = ub.forces(U, 3.5, only=("causal",))["causal"][t]
        ok = np.linalg.norm(g_ns, axis=1) >= ub.TINY     # token 1's only key 1…i is itself
        assert not ok[0]
        cos = np.sum(g_ns * g_phi, 1)[ok] / np.linalg.norm(g_ns, axis=1)[ok] / np.linalg.norm(g_phi, axis=1)[ok]
        assert per[L]["heads"][0]["align_phi"] == pytest.approx(cos.mean(), abs=1e-6)
        assert per[L]["heads"][0]["a0"] == pytest.approx(A[0, t, 0].mean().item(), abs=1e-6)
        # every cell the rule names, for every head
        srcs = {c["source"] for c in per[L]["cells"] if "head" not in c}
        assert srcs == {f"{f}:{k}:{r}" for f, k, r in uh.ATTN_CELLS}
        hc = {(c["head"], c["source"]) for c in per[L]["cells"] if "head" in c}
        assert hc == {(h, s) for h in range(H) for s in ("kernns_h:keys_h:r1out", "kern_h:head_h:frozen")}
    # saved means: the parts sum to the block
    torch.set_grad_enabled(True)
    means = uh.saved_means(comps, per, t)
    assert means["sum_dev"] < 1e-6 and means["parts"].shape == (3, len(uh.SAVED), hs.shape[2])


def test_hooks_capture_nothing_without_a_context():
    torch = pytest.importorskip("torch")
    from p1e_energy_field.u2_torch import Ops
    m = _tiny()
    hk = uh.HeadHooked(m, Ops("cpu", torch.float64))
    hk.run(torch.randint(0, 97, (1, 10)), range(3))     # the attention arm's pass alone
    assert not any(k[0] in ("w", "v", "x") for k in hk.cap)
    hk.close()
