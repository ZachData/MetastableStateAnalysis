"""
tests/test_p2_decompose_dispatch.py

`extract_decomposed_standard` used to branch on substrings of the model
name, so GPT-NeoX matched no branch, registered no hooks, and returned a
well-formed dict with empty delta lists. Every assertion here is aimed at
that failure mode rather than at the happy path, because the happy path
never looked broken.

The models are hand-built torch modules that reproduce each family's
residual arithmetic exactly. Real Pythia weights would test the same code
paths but take a download and a minute per checkpoint; what matters is the
arithmetic — `out = x + attn(ln(x)) + mlp(ln(x))` for parallel, `x1 = x +
attn(ln1(x)); x2 = x1 + mlp(ln2(x1))` for sequential — and that is exactly
reproducible in twenty lines.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


# ─────────────────────────────────────────────────────────────────────────────
# Minimal models with real residual arithmetic
# ─────────────────────────────────────────────────────────────────────────────

class _Attn(nn.Module):
    """Returns a tuple, as GPT-2 and GPT-NeoX attention modules do — the
    capture must unwrap it rather than storing the tuple."""

    def __init__(self, d, n_heads=2):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.n_heads = n_heads

    def forward(self, x, **kw):
        b, s, _ = x.shape
        weights = torch.softmax(torch.randn(b, self.n_heads, s, s), dim=-1)
        return self.proj(x), weights


class _MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class _NeoXBlock(nn.Module):
    """use_parallel_residual=True: both branches read the block input."""

    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention, self.mlp = _Attn(d), _MLP(d)

    def forward(self, x):
        a, w = self.attention(self.ln1(x))
        f = self.mlp(self.ln2(x))
        return x + a + f, w


class _GPT2Block(nn.Module):
    """Sequential: the FFN branch reads the post-attention stream."""

    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = _Attn(d), _MLP(d)

    def forward(self, x):
        a, w = self.attn(self.ln1(x))
        x1 = x + a
        return x1 + self.mlp(self.ln2(x1)), w


class _Out:
    def __init__(self, hidden_states, attentions):
        self.hidden_states = hidden_states
        self.attentions = attentions


class _Model(nn.Module):
    def __init__(self, block_cls, d=16, n_layers=3, parallel=True, attr="layers"):
        super().__init__()
        self.embed = nn.Embedding(50, d)
        blocks = nn.ModuleList([block_cls(d) for _ in range(n_layers)])
        setattr(self, attr, blocks)
        self._blocks = blocks
        self.config = type("cfg", (), {"use_parallel_residual": parallel})()

    def forward(self, input_ids=None, output_hidden_states=False,
                output_attentions=False, **kw):
        x = self.embed(input_ids)
        hs, attns = [x], []
        for b in self._blocks:
            x, w = b(x)
            hs.append(x)
            attns.append(w)
        return _Out(tuple(hs), tuple(attns))


class _Tok:
    def __call__(self, text, **kw):
        n = min(len(text.split()), 12)
        return _Batch({"input_ids": torch.arange(1, n + 1).unsqueeze(0)})

    def convert_ids_to_tokens(self, ids):
        return [f"t{int(i)}" for i in ids]


class _Batch(dict):
    def to(self, _device):
        return self


def _neox_model(**kw):
    m = _Model(_NeoXBlock, attr="layers", **kw)
    m.gpt_neox = type("base", (), {"layers": m.layers})()
    return m


def _gpt2_model(**kw):
    m = _Model(_GPT2Block, attr="h", parallel=False, **kw)
    m.transformer = type("base", (), {"h": m.h})()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# The branch that did not exist
# ─────────────────────────────────────────────────────────────────────────────

def test_gptneox_produces_nonempty_deltas():
    """
    The whole defect in one assertion. Before family dispatch this returned
    a dict with empty attn_deltas and ffn_deltas, and every consumer
    downstream degraded quietly rather than raising.
    """
    from p2_eigenspectra.decompose import extract_decomposed_standard

    out = extract_decomposed_standard(
        _neox_model(), _Tok(), "a b c d e f", "pythia-410m-step1000")

    assert len(out["attn_deltas"]) == 3
    assert len(out["ffn_deltas"]) == 3
    assert out["semantics"] == "pre-ln-parallel"
    assert out["parallel_residual"] is True


def test_parallel_residual_identity_holds_exactly():
    """
    x_{L+1} - x_L == attn_delta + ffn_delta. On a parallel-residual model
    in float32 this is exact to rounding, which is what makes it a usable
    end-to-end check that the hooks are on the right modules: no correct
    capture can fail it, and most incorrect ones do.
    """
    from p2_eigenspectra.decompose import extract_decomposed_standard

    out = extract_decomposed_standard(
        _neox_model(), _Tok(), "a b c d e f", "pythia-410m-step1000")
    ident = out["residual_identity"]
    assert ident["checked"] == 3
    assert ident["rel_err"] < 1e-4, ident


def test_sequential_residual_identity_also_holds():
    """
    True for GPT-2 as well, despite the FFN branch reading the
    post-attention stream: both deltas are still added to the same
    residual. The identity does not distinguish the two arrangements —
    `semantics` does.
    """
    from p2_eigenspectra.decompose import extract_decomposed_standard

    out = extract_decomposed_standard(
        _gpt2_model(), _Tok(), "a b c d e f", "gpt2-large")
    assert out["semantics"] == "pre-ln-sequential"
    assert out["parallel_residual"] is False
    assert out["residual_identity"]["rel_err"] < 1e-4


def test_parallel_flag_is_read_from_config_not_assumed():
    """GPT-NeoX supports both; Pythia happens to use parallel at every
    scale. A model that says otherwise must be believed."""
    from p2_eigenspectra.decompose import extract_decomposed_standard

    out = extract_decomposed_standard(
        _neox_model(parallel=False), _Tok(), "a b c", "pythia-410m-step0")
    assert out["semantics"] == "pre-ln-sequential"


def test_attention_matrices_survive_the_tuple_unwrap():
    """The dynamic head test needs outputs.attentions; the attention module
    returns a tuple and the capture must take element 0 of the OUTPUT while
    leaving outputs.attentions alone."""
    from p2_eigenspectra.decompose import extract_decomposed_standard

    out = extract_decomposed_standard(
        _neox_model(), _Tok(), "a b c d", "pythia-410m-step1000")
    assert len(out["attentions"]) == 3
    assert out["attentions"][0].ndim == 3      # (n_heads, seq, seq)
    assert out["attn_deltas"][0].ndim == 2     # (n_tokens, d)


def test_unsupported_architecture_raises_rather_than_returning_empty():
    """
    The core of the fix. "No branch for this family" and "the branch exists
    and the hooks misfired" must not produce the same artifact — that
    equivalence is what let a Pythia sweep look like it had run the
    decomposition.
    """
    from core.sublayer_streams import UnsupportedArchitecture
    from p2_eigenspectra.decompose import extract_decomposed_standard

    class _Nothing(nn.Module):
        def forward(self, **kw):
            raise AssertionError("should not reach a forward pass")

    with pytest.raises(UnsupportedArchitecture):
        extract_decomposed_standard(_Nothing(), _Tok(), "a b", "mamba-130m")


def test_albert_is_routed_to_its_own_extractor():
    from core.sublayer_streams import UnsupportedArchitecture
    from p2_eigenspectra.decompose import extract_decomposed_standard

    with pytest.raises(UnsupportedArchitecture, match="extract_decomposed_albert"):
        extract_decomposed_standard(_neox_model(), _Tok(), "a b", "albert-base-v2")


def test_hooks_are_removed_even_when_the_forward_raises():
    """A leaked hook corrupts every later prompt in the same process, and a
    27-checkpoint sweep reuses one process per model."""
    from p2_eigenspectra.decompose import extract_decomposed_standard

    model = _neox_model()
    original = model._blocks[1].forward

    def _boom(_x):
        raise RuntimeError("simulated OOM mid-forward")

    model._blocks[1].forward = _boom
    try:
        before = sum(len(b.mlp._forward_hooks) for b in model.layers)
        with pytest.raises(RuntimeError, match="simulated OOM"):
            extract_decomposed_standard(model, _Tok(), "a b", "pythia-410m-step0")
        after = sum(len(b.mlp._forward_hooks) for b in model.layers)
        assert before == 0, "hooks leaked from an earlier call"
        assert after == 0, "hooks survived the exception"
    finally:
        model._blocks[1].forward = original


# ─────────────────────────────────────────────────────────────────────────────
# Downstream guard
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_deltas_are_inapplicable_not_a_mixed_verdict():
    """
    Previously a decomposed dict with empty deltas reached
    analyze_violations_decomposed, produced zero attributions, and was
    reported as channel "mixed" with n=0 — a verdict field that looks
    measured and is not.
    """
    from p2_eigenspectra.subexp_wrappers import _decomposed_violations_subexp

    ctx = {
        "trajectory_result": {"events": {"energy_violations": {1.0: [3]}}},
        "decomposed": {"trajectory": [], "attn_deltas": [], "ffn_deltas": []},
    }
    res = _decomposed_violations_subexp(ctx)
    assert res.applicable is False
    assert "channel" not in res.verdict_contribution
