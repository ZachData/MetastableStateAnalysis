"""core/nudge.py's Stack on a tiny random GPT-NeoX in float64 (`trackB_circuit/design-B.md` "B0"):
the loop reproduces the HF forward; the nudge adds what it says (G1); a nudge at j moves nothing
before j (G4a); A/ε meets the block's JVP (G2); a planted loss gradient ranks first (G3).
Needs the real torch/transformers: ``SMOKE_REAL_DEPS=1 pytest -m smoke``.
"""
import numpy as np
import pytest

from core import nudge as nd
from tests.test_p10_ov1_cut_smoke import tiny_model, torch64  # noqa: F401  (the fixture)

pytestmark = pytest.mark.smoke
IDS = [3, 17, 5, 40, 22, 9, 63, 1, 28, 11, 50, 7]


def stack(torch):
    return nd.Stack(tiny_model(torch))


def test_loop_is_the_hf_forward(torch64):
    st = stack(torch64)
    with torch64.no_grad():
        hf = st.model(torch64.tensor([IDS])).logits
        mine = st.logits(st.run(st.embed(IDS), 0, keep={st.L})["resid"][st.L])
    assert torch64.equal(hf, mine)


def test_nudge_adds_what_it_says_and_nothing_before_j(torch64):
    st = stack(torch64)
    x = st.embed(IDS)
    R = nd.random_directions(x.shape[-1], 4, 0)
    j, eps = 6, 0.05
    with torch64.no_grad():
        out = st.run(x.expand(9, -1, -1), 0, keep=set(range(st.L + 1)), loss_ids=IDS,
                     delta=st.deltas(R, eps, [j], len(IDS)))
    A, S = nd.anti_sym(*nd.split_rows(out["resid"][0])[1:], nd.split_rows(out["resid"][0])[0])
    assert np.allclose(A[:, j].numpy(), eps * R, atol=1e-14) and float(S.abs().max()) < 1e-14
    assert float(A[:, :j].abs().max()) == 0 and float(A[:, j + 1:].abs().max()) == 0
    for t in [out["resid"][l] for l in range(1, st.L + 1)] + [out["loss"]]:
        b, p, m = nd.split_rows(t)
        A, S = nd.anti_sym(p, m, b)
        assert float(A[:, :j].abs().max()) == 0 and float(S[:, :j].abs().max()) == 0
        assert float(A[:, j:].abs().max()) > 0


def test_antisymmetric_part_is_the_jvp(torch64):
    st = stack(torch64)
    with torch64.no_grad():
        x = st.embed(IDS)
        d = torch64.as_tensor(nd.random_directions(x.shape[-1], 1, 1)[0])
        t = torch64.zeros_like(x)
        t[0, 4] = d
        _, Jd = torch64.func.jvp(lambda z: st.block(0, z), (x,), (t,))
        r = nd.rms_norm(x[0].numpy())
        errs = []
        for c in nd.C_GRID:
            eps = c * r
            A, _ = nd.anti_sym(st.block(0, x + eps * t), st.block(0, x - eps * t), st.block(0, x))
            errs.append(float((A[0, 4:] / eps - Jd[0, 4:]).norm() / Jd[0, 4:].norm()))
    assert errs[0] < 1e-3                                  # O(ε²) at the smallest c
    assert errs == sorted(errs)                            # departs with ε, as the expansion says


def test_planted_gradient_ranks_first(torch64):
    st = stack(torch64)
    x = st.embed(IDS).detach().clone().requires_grad_(True)
    i = 8
    (g,) = torch64.autograd.grad(st.run(x, 0, loss_ids=IDS, loss_at=[i])["loss"][0, 0], x)
    g = g[0, i].numpy()
    dirs = np.vstack([nd.unit(g), nd.random_directions(g.size, 16, 2)])
    r = nd.rms_norm(x[0].detach().numpy())
    with torch64.no_grad():
        eps = 0.01 * r
        loss = st.run(x.detach().expand(len(dirs) * 2 + 1, -1, -1), 0, loss_ids=IDS, loss_at=[i],
                      delta=st.deltas(dirs, eps, [i], len(IDS)))["loss"][:, 0].numpy()
    b, p, m = nd.split_rows(loss)
    A, _ = nd.anti_sym(p, m, b)
    assert A[0] / eps == pytest.approx(np.linalg.norm(g), rel=0.02)
    assert nd.exact_rank(abs(A[0]), np.abs(A[1:])) == 1


def test_refuses_a_non_eager_or_non_neox_model(torch64):
    m = tiny_model(torch64)
    m.gpt_neox.config._attn_implementation = "sdpa"
    with pytest.raises(nd.NudgeError):
        nd.Stack(m)
    with pytest.raises(nd.NudgeError):
        nd.Stack(torch64.nn.Linear(2, 2))
