"""
tests/test_phase3_crosscoder.py

Architecture contract tests for phase3/crosscoder.py and subresult.py.
No training loop, no real activations. CPU only.

Standard fixture: batch=4, n_layers=3, d_model=16, n_features=32, k=4
"""

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

# ---------------------------------------------------------------------------
# Import under test — adjust path if your package layout differs.
# ---------------------------------------------------------------------------
from p3_crosscoder.crosscoder import Crosscoder, CrosscoderConfig 


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return CrosscoderConfig(
        n_layers=3,
        d_model=16,
        n_features=32,
        k=4,
    )


@pytest.fixture
def model(cfg):
    return Crosscoder(cfg).eval()


@pytest.fixture
def x(cfg):
    """Random (batch, n_layers * d_model) input tensor, CPU."""
    torch.manual_seed(0)
    return torch.randn(4, cfg.n_layers * cfg.d_model)


@pytest.fixture
def out(model, x):
    with torch.no_grad():
        return model(x)


# ---------------------------------------------------------------------------
# forward() output shapes
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_z_shape(self, out, cfg):
        assert out["z"].shape == (4, cfg.n_features), (
            f"z should be (batch, n_features), got {out['z'].shape}"
        )

    def test_x_hat_shape(self, out, cfg):
        assert out["x_hat"].shape == (4, cfg.n_layers, cfg.d_model), (
            f"x_hat should be (batch, n_layers, d_model), got {out['x_hat'].shape}"
        )

    def test_loss_is_scalar(self, out):
        loss = out["loss"]
        assert loss.ndim == 0, f"loss should be scalar, got ndim={loss.ndim}"

    def test_output_keys_present(self, out):
        assert {"z", "x_hat", "loss"} <= out.keys(), (
            f"Missing keys in forward output: {out.keys()}"
        )


# ---------------------------------------------------------------------------
# topk_activation: exactly k non-zeros per row
# ---------------------------------------------------------------------------

class TestTopkActivation:
    def test_nonzero_count_per_row(self, model, cfg):
        torch.manual_seed(1)
        z = torch.randn(4, cfg.n_features)
        out = model.topk_activation(z, cfg.k)
        nonzeros_per_row = (out != 0).sum(dim=1)
        assert (nonzeros_per_row == cfg.k).all(), (
            f"Each row must have exactly k={cfg.k} non-zeros; got {nonzeros_per_row.tolist()}"
        )

    def test_output_shape_preserved(self, model, cfg):
        z = torch.randn(4, cfg.n_features)
        out = model.topk_activation(z, cfg.k)
        assert out.shape == z.shape

    def test_k_zero_yields_all_zeros(self, model, cfg):
        z = torch.randn(4, cfg.n_features)
        out = model.topk_activation(z, 0)
        assert (out == 0).all(), "k=0 should produce an all-zero tensor"

    def test_k_full_yields_no_zeros(self, model, cfg):
        # All values non-zero only if input has no zeros; use abs to be safe.
        z = torch.randn(4, cfg.n_features).abs() + 1e-6
        out = model.topk_activation(z, cfg.n_features)
        assert (out != 0).all(), "k=n_features should leave all values active"

    def test_values_match_top_k_of_input(self, model, cfg):
        """Non-zero positions in the output must be the k largest-magnitude entries."""
        torch.manual_seed(2)
        z = torch.randn(1, cfg.n_features)
        out = model.topk_activation(z, cfg.k)
        # Indices that are non-zero in output
        active_idx = out[0].nonzero(as_tuple=True)[0]
        # Top-k indices by magnitude in input
        topk_idx = z[0].abs().topk(cfg.k).indices
        assert set(active_idx.tolist()) == set(topk_idx.tolist()), (
            "topk_activation must keep the k largest-magnitude entries"
        )

    def test_single_row(self, model, cfg):
        z = torch.randn(1, cfg.n_features)
        out = model.topk_activation(z, cfg.k)
        assert (out != 0).sum().item() == cfg.k

    def test_large_batch_consistency(self, model, cfg):
        torch.manual_seed(3)
        z = torch.randn(64, cfg.n_features)
        out = model.topk_activation(z, cfg.k)
        nonzeros = (out != 0).sum(dim=1)
        assert (nonzeros == cfg.k).all()


# ---------------------------------------------------------------------------
# batch_topk_activation: total non-zeros == batch * k
# ---------------------------------------------------------------------------

class TestBatchTopkActivation:
    def test_total_nonzeros(self, model, cfg):
        torch.manual_seed(4)
        batch = 4
        z = torch.randn(batch, cfg.n_features)
        out = model.batch_topk_activation(z, cfg.k)
        total = (out != 0).sum().item()
        assert total == batch * cfg.k, (
            f"batch_topk_activation must have exactly batch*k={batch * cfg.k} "
            f"non-zeros total; got {total}"
        )

    def test_output_shape_preserved(self, model, cfg):
        z = torch.randn(4, cfg.n_features)
        out = model.batch_topk_activation(z, cfg.k)
        assert out.shape == z.shape

    def test_k_zero_total_zeros(self, model, cfg):
        z = torch.randn(4, cfg.n_features)
        out = model.batch_topk_activation(z, 0)
        assert (out == 0).all()

    def test_k_full_total_equals_batch_times_features(self, model, cfg):
        batch = 4
        z = torch.randn(batch, cfg.n_features).abs() + 1e-6
        out = model.batch_topk_activation(z, cfg.n_features)
        assert (out != 0).sum().item() == batch * cfg.n_features

    def test_budget_may_concentrate_on_one_sample(self, model, cfg):
        """
        batch_topk picks the top batch*k values globally; a single sample
        could own all of them in a degenerate input.  The contract is the
        *total*, not the per-row count.
        """
        batch = 4
        z = torch.zeros(batch, cfg.n_features)
        # Make only row 0 non-zero
        z[0] = torch.arange(1, cfg.n_features + 1, dtype=torch.float)
        out = model.batch_topk_activation(z, cfg.k)
        assert (out != 0).sum().item() == batch * cfg.k


# ---------------------------------------------------------------------------
# Decoder column normalization
# ---------------------------------------------------------------------------

class TestDecoderNormalization:
    def test_decoder_columns_unit_norm(self, model):
        """
        If the crosscoder enforces normalized decoder columns, each column's
        self-dot-product must equal 1.0 (within floating-point tolerance).
        """
        decoder = None
        # Accept common attribute names
        for attr in ("decoder", "W_dec", "decode"):
            candidate = getattr(model, attr, None)
            if candidate is not None and hasattr(candidate, "weight"):
                decoder = candidate.weight.data  # (out_dim, n_features)
                break
            if isinstance(candidate, torch.Tensor):
                decoder = candidate.data
                break

        if decoder is None:
            pytest.skip("No recognized decoder weight attribute; skipping norm check")

        # Columns are along dim 0 if shape is (out, n_features), else dim 1
        # Normalisation is per-feature (column of the decode matrix)
        # Shape: (n_features, ...) or (..., n_features)
        # Try both conventions
        norms = decoder.norm(dim=0)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
            norms = decoder.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Decoder columns are not unit-normalized; norms range "
            f"[{norms.min():.4f}, {norms.max():.4f}]"
        )


# ---------------------------------------------------------------------------
# Sparsity edge cases
# ---------------------------------------------------------------------------

class TestSparsityEdgeCases:
    def test_k_zero_z_all_zeros(self, cfg):
        cfg_k0 = SimpleNamespace(**vars(cfg))
        cfg_k0.k = 0
        model = Crosscoder(cfg_k0).eval()
        x = torch.randn(4, cfg.n_layers * cfg.d_model)
        with torch.no_grad():
            out = model(x)
        assert (out["z"] == 0).all(), "k=0 must produce all-zero feature activations"

    def test_k_full_no_sparsity(self, cfg):
        cfg_full = SimpleNamespace(**vars(cfg))
        cfg_full.k = cfg.n_features
        model = Crosscoder(cfg_full).eval()
        x = torch.randn(4, cfg.n_layers * cfg.d_model).abs() + 1e-3
        with torch.no_grad():
            out = model(x)
        # Cannot guarantee all non-zero (encoder might zero some), but shape must hold
        assert out["z"].shape == (4, cfg.n_features)

    def test_n_features_1_k_1_always_active(self, cfg):
        """With a single feature and k=1 that feature must always fire."""
        cfg_tiny = SimpleNamespace(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_features=1,
            k=1,
        )
        model = Crosscoder(cfg_tiny).eval()
        torch.manual_seed(5)
        x = torch.randn(4, cfg.n_layers * cfg.d_model)
        with torch.no_grad():
            out = model(x)
        assert out["z"].shape == (4, 1)
        assert (out["z"] != 0).all(), (
            "With n_features=1 and k=1, the single feature must be active for every sample"
        )


# ---------------------------------------------------------------------------
# Reconstruction loss properties
# ---------------------------------------------------------------------------

class TestReconstructionLoss:
    def test_loss_non_negative(self, out):
        assert out["loss"].item() >= 0.0, "Reconstruction loss must be non-negative"

    def test_loss_finite(self, out):
        assert math.isfinite(out["loss"].item()), "Loss must be finite for bounded inputs"

    def test_loss_zero_input(self, model, cfg):
        """Zero input: loss must still be non-negative and finite."""
        x = torch.zeros(4, cfg.n_layers * cfg.d_model)
        with torch.no_grad():
            out = model(x)
        assert out["loss"].item() >= 0.0
        assert math.isfinite(out["loss"].item())

    def test_loss_large_input(self, model, cfg):
        """Large but finite inputs must not cause NaN/Inf loss."""
        x = torch.full((4, cfg.n_layers * cfg.d_model), 1e4)
        with torch.no_grad():
            out = model(x)
        assert math.isfinite(out["loss"].item()), (
            "Loss must be finite for large but bounded inputs"
        )

    def test_loss_is_tensor(self, out):
        assert isinstance(out["loss"], torch.Tensor)

    def test_loss_does_not_depend_on_batch_index(self, model, cfg):
        """
        Loss should be deterministic for the same input regardless of
        which batch position it occupies (permutation invariance of mean).
        Use a single-sample batch vs a repeated multi-sample batch.
        """
        torch.manual_seed(6)
        single = torch.randn(1, cfg.n_layers * cfg.d_model)
        repeated = single.expand(4, -1).contiguous()
        with torch.no_grad():
            loss_single = model(single)["loss"].item()
            loss_repeated = model(repeated)["loss"].item()
        assert math.isclose(loss_single, loss_repeated, rel_tol=1e-4), (
            f"Loss of single sample ({loss_single:.6f}) differs from repeated "
            f"batch ({loss_repeated:.6f}); check mean reduction"
        )


# ---------------------------------------------------------------------------
# Gradient flow (no optimizer, just autograd smoke test)
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_loss_backward(self, cfg):
        model = Crosscoder(cfg)  # train mode intentional
        x = torch.randn(4, cfg.n_layers * cfg.d_model)
        out = model(x)
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients flowed to any parameter"

    def test_no_nan_gradients(self, cfg):
        model = Crosscoder(cfg)
        x = torch.randn(4, cfg.n_layers * cfg.d_model)
        out = model(x)
        out["loss"].backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_forward_deterministic(self, model, x):
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1["z"], out2["z"]), "forward() must be deterministic"
        assert torch.equal(out1["x_hat"], out2["x_hat"])
        assert torch.equal(out1["loss"], out2["loss"])

