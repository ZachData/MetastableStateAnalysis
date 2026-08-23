"""Tests for core/sublayer_streams.py, against the three defects in the
version that lived inside run_1._run_sublayer_analysis."""
import pytest
import torch
import torch.nn as nn

from core.sublayer_streams import extract_sublayer_streams, UnsupportedArchitecture

# Tier: deps -- needs the heavy tier importable (torch / transformers /
# scikit-learn / matplotlib). No model download, no run artifacts.
# Measured, not assumed; see pyproject.toml markers.
pytestmark = pytest.mark.deps


class _Tok:
    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        return {"input_ids": torch.zeros(1, 5, dtype=torch.long)}

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in range(len(ids))]


class _Delta(nn.Module):
    """Sub-block returning a constant delta, so streams are checkable by hand."""
    def __init__(self, value, d=8):
        super().__init__()
        self.value = value
        self.lin = nn.Linear(d, d)      # gives the module a parameter
    def forward(self, x, *a, **k):
        return torch.full_like(x if torch.is_tensor(x) else x[0], self.value)


class _ParallelBlock(nn.Module):
    """x + attn(x) + mlp(x) — GPT-NeoX with use_parallel_residual=True."""
    def __init__(self, d=8):
        super().__init__()
        self.attention = _Delta(1.0, d)
        self.mlp       = _Delta(2.0, d)
    def forward(self, x, *a, **k):
        return x + self.attention(x) + self.mlp(x)


class _AlbertBlock(nn.Module):
    """attention -> full_layer_layer_norm. Post-LN: each submodule output
    is already the residual stream, matching real AlbertLayer's shape
    (which has no .mlp — _ParallelBlock's attention/mlp pair is a
    GPT-NeoX-shaped block, not an ALBERT-shaped one)."""
    def __init__(self, d=8):
        super().__init__()
        self.attention = _Delta(1.0, d)
        self.full_layer_layer_norm = _Delta(2.0, d)
    def forward(self, x, *a, **k):
        return self.full_layer_layer_norm(self.attention(x))


class _SequentialBlock(nn.Module):
    """x1 = x + attn(x); x2 = x1 + mlp(x1) — GPT-2 ordering."""
    def __init__(self, d=8):
        super().__init__()
        self.attn = _Delta(1.0, d)
        self.mlp  = _Delta(2.0, d)
    def forward(self, x, *a, **k):
        x = x + self.attn(x)
        return x + self.mlp(x)


def _stack(block_cls, n, attr, d=8):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            setattr(self, attr, nn.ModuleList([block_cls(d) for _ in range(n)]))
            self.config = type("C", (), {"use_parallel_residual":
                                         block_cls is _ParallelBlock})()
        def forward(self, input_ids=None, **k):
            x = torch.zeros(1, input_ids.shape[1], d)
            for b in getattr(self, attr):
                x = b(x)
            return x
    return Model()


def test_gpt2_captures_streams_not_deltas():
    """The headline regression: post_attn must be x + delta, not delta.

    With a zero input and a constant attention delta of 1.0, the delta and
    the stream are numerically distinguishable only because the block
    accumulates across layers — layer 1's input is 3.0, so its post_attn
    stream is 4.0 while its delta is still 1.0."""
    model = _stack(_SequentialBlock, 2, "h")
    s = extract_sublayer_streams(model, _Tok(), "x", "gpt2")
    assert s.semantics == "pre-ln-sequential"
    assert torch.allclose(s.post_attn[0], torch.ones_like(s.post_attn[0]) * 1.0)
    assert torch.allclose(s.post_ffn[0],  torch.ones_like(s.post_ffn[0])  * 3.0)
    # Layer 1 input is 3.0 → stream 4.0. A delta capture would give 1.0.
    assert torch.allclose(s.post_attn[1], torch.ones_like(s.post_attn[1]) * 4.0)


def test_gptneox_parallel_branches_are_symmetric():
    """Both branches read the same block input, so neither stream is
    downstream of the other — the property design-1 wants."""
    model = _stack(_ParallelBlock, 2, "layers")
    s = extract_sublayer_streams(model, _Tok(), "x", "pythia-410m-step0")
    assert s.semantics == "pre-ln-parallel" and s.parallel_residual is True
    assert torch.allclose(s.post_attn[0], torch.ones_like(s.post_attn[0]) * 1.0)
    assert torch.allclose(s.post_ffn[0],  torch.ones_like(s.post_ffn[0])  * 2.0)
    # Block output is 3.0, so layer 1 gives 4.0 and 5.0.
    assert torch.allclose(s.post_attn[1], torch.ones_like(s.post_attn[1]) * 4.0)
    assert torch.allclose(s.post_ffn[1],  torch.ones_like(s.post_ffn[1])  * 5.0)


def test_weight_shared_layer_yields_one_capture_per_call():
    """ALBERT's shared layer is called N times; capture must append, not
    index by module position."""
    d = 8
    shared = _AlbertBlock(d)

    class Albert(nn.Module):
        def __init__(self):
            super().__init__()
            group = nn.Module()
            group.albert_layers = nn.ModuleList([shared])
            self.encoder = nn.Module()
            self.encoder.albert_layer_groups = nn.ModuleList([group])
        def forward(self, input_ids=None, **k):
            x = torch.zeros(1, input_ids.shape[1], d)
            for _ in range(4):
                x = shared(x)
            return x

    s = extract_sublayer_streams(Albert(), _Tok(), "x", "albert-base-v2")
    assert s.n_layers == 4, "one capture per call, not one per module"
    assert s.semantics == "post-ln"


def test_unsupported_architecture_raises_rather_than_returning_empty():
    class Bare(nn.Module):
        def forward(self, input_ids=None, **k):
            return torch.zeros(1, 5, 8)
    with pytest.raises(UnsupportedArchitecture):
        extract_sublayer_streams(Bare(), _Tok(), "x", "some-unregistered-model")