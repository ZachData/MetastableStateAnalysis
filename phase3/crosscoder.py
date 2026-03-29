"""
crosscoder.py — Sparse crosscoder architecture.

A crosscoder is a sparse autoencoder whose input is the concatenation of
residual stream activations across L_sampled layers.  Each learned feature
has a separate decoder direction per layer, revealing where and when it is
active across depth.

Encoder: (L_sampled * d_model) → n_features   (one shared linear map)
Decoder: n_features → (L_sampled, d_model)     (L_sampled separate linear maps)

Activation functions are pluggable: TopK or BatchTopK.

This module is a pure nn.Module.  No I/O, no data loading, no knowledge
of what model produced the activations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class ActivationType(Enum):
    TOPK = "topk"
    BATCH_TOPK = "batch_topk"


def topk_activation(z: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-k activations per sample, zero the rest."""
    # z: (batch, n_features)
    values, indices = torch.topk(z, k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, indices, values)
    return out


def batch_topk_activation(z: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep the top (batch_size * k) activations across the entire batch,
    zero everything else.  Distributes sparsity budget across the batch
    so that some samples can use more features if they need them.
    """
    batch_size = z.shape[0]
    total_k = batch_size * k
    flat = z.view(-1)
    total_k = min(total_k, flat.numel())
    values, indices = torch.topk(flat, total_k)
    out = torch.zeros_like(flat)
    out.scatter_(0, indices, values)
    return out.view_as(z)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CrosscoderConfig:
    """All hyperparameters that define the crosscoder architecture."""

    d_model: int                      # hidden dimension of the source model
    n_layers: int                     # number of sampled layers in the stack
    n_features: int                   # dictionary size
    activation: ActivationType = ActivationType.BATCH_TOPK
    k: int = 64                       # sparsity parameter for TopK / BatchTopK

    # Derived
    @property
    def d_input(self) -> int:
        return self.n_layers * self.d_model

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_features": self.n_features,
            "activation": self.activation.value,
            "k": self.k,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CrosscoderConfig":
        d = dict(d)
        d["activation"] = ActivationType(d["activation"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class Crosscoder(nn.Module):
    """
    Sparse crosscoder for cross-layer feature extraction.

    Forward pass:
      1. Concatenate L sampled layers into a flat vector  (batch, L*d)
      2. Subtract pre-encoder bias (learned mean)
      3. Encode → (batch, n_features) via W_enc
      4. Apply activation function (TopK or BatchTopK on ReLU'd values)
      5. Decode → (batch, L, d) via L separate decoder weight matrices
      6. Add pre-decoder bias per layer

    The loss is the sum of per-layer MSE between input and reconstruction.
    """

    def __init__(self, cfg: CrosscoderConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.d_input
        F_   = cfg.n_features
        L    = cfg.n_layers
        d    = cfg.d_model

        # Pre-encoder bias: learned estimate of the activation mean.
        # Subtracting this before encoding lets the encoder focus on
        # deviations from the mean, improving feature quality.
        self.b_pre = nn.Parameter(torch.zeros(d_in))

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_in, F_))
        self.b_enc = nn.Parameter(torch.zeros(F_))

        # Per-layer decoders: each is (n_features, d_model).
        # Stored as a single (L, n_features, d_model) tensor for
        # vectorized matmul, but conceptually L separate matrices.
        self.W_dec = nn.Parameter(torch.empty(L, F_, d))

        # Per-layer decoder bias
        self.b_dec = nn.Parameter(torch.zeros(L, d))

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming uniform for encoder, decoder columns initialized to
        unit norm (Anthropic convention).
        """
        nn.init.kaiming_uniform_(self.W_enc)

        # Initialize decoder columns to unit norm
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            # Normalize each feature's decoder direction per layer
            norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.W_dec.div_(norms)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Encode stacked activations into pre-activation feature values.

        Parameters
        ----------
        x_flat : (batch, n_layers * d_model)

        Returns
        -------
        z_pre : (batch, n_features) — pre-activation
        """
        return (x_flat - self.b_pre) @ self.W_enc + self.b_enc

    def activate(self, z_pre: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU then sparsity constraint.

        Parameters
        ----------
        z_pre : (batch, n_features)

        Returns
        -------
        z : (batch, n_features) — sparse activations
        """
        z = F.relu(z_pre)
        if self.cfg.activation == ActivationType.TOPK:
            z = topk_activation(z, self.cfg.k)
        elif self.cfg.activation == ActivationType.BATCH_TOPK:
            z = batch_topk_activation(z, self.cfg.k)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to per-layer activations.

        Parameters
        ----------
        z : (batch, n_features)

        Returns
        -------
        x_hat : (batch, n_layers, d_model)
        """
        # z: (B, F) @ W_dec: (L, F, d) → einsum → (B, L, d)
        x_hat = torch.einsum("bf,lfd->bld", z, self.W_dec) + self.b_dec
        return x_hat

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        """
        Full forward pass.

        Parameters
        ----------
        x : (batch, n_layers, d_model) — stacked per-layer activations

        Returns
        -------
        dict with:
          x_hat     : (batch, n_layers, d_model) — reconstruction
          z         : (batch, n_features) — sparse feature activations
          z_pre     : (batch, n_features) — pre-activation values
          loss      : scalar — sum of per-layer MSE
          per_layer_mse : (n_layers,) — MSE per layer (detached)
        """
        B, L, d = x.shape
        assert L == self.cfg.n_layers and d == self.cfg.d_model

        x_flat = x.reshape(B, L * d)
        z_pre  = self.encode(x_flat)
        z      = self.activate(z_pre)
        x_hat  = self.decode(z)

        # Per-layer MSE: mean over batch and d_model, sum over layers.
        # Summing over layers (not averaging) ensures each layer's
        # reconstruction is weighted equally regardless of n_layers.
        per_layer_mse = ((x - x_hat) ** 2).mean(dim=(0, 2))  # (L,)
        loss = per_layer_mse.sum()

        return {
            "x_hat": x_hat,
            "z": z,
            "z_pre": z_pre,
            "loss": loss,
            "per_layer_mse": per_layer_mse.detach(),
        }

    # ------------------------------------------------------------------
    # Decoder norm constraint (called after each optimizer step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder(self):
        """
        Project decoder columns back to unit norm.

        Standard SAE practice: prevents the model from trading off
        feature magnitude against sparsity.  Applied per-layer per-feature.
        """
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    # ------------------------------------------------------------------
    # Feature diagnostics (no gradients, used during training monitoring)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def feature_activity(self, z: torch.Tensor) -> dict:
        """
        Compute feature-level statistics from a batch of sparse activations.

        Parameters
        ----------
        z : (batch, n_features) — sparse activations from forward()

        Returns
        -------
        dict with:
          alive_mask   : (n_features,) bool — True if feature fired at all
          fire_rate    : (n_features,) float — fraction of batch where f > 0
          mean_act     : (n_features,) float — mean activation when active
          n_dead       : int — count of features that never fired
        """
        active = z > 0                                    # (B, F)
        fire_rate = active.float().mean(dim=0)            # (F,)
        alive = fire_rate > 0                             # (F,)

        # Mean activation conditioned on being active
        act_sum   = z.sum(dim=0)                          # (F,)
        act_count = active.sum(dim=0).clamp(min=1)        # (F,)
        mean_act  = act_sum / act_count                   # (F,)

        return {
            "alive_mask": alive,
            "fire_rate":  fire_rate,
            "mean_act":   mean_act,
            "n_dead":     int((~alive).sum()),
        }

    # ------------------------------------------------------------------
    # Feature decoder geometry (for analysis, not training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decoder_norms(self) -> torch.Tensor:
        """
        Per-feature, per-layer decoder direction norms.

        Returns
        -------
        (n_features, n_layers) — ||W_dec[l, f, :]|| for each (f, l)
        """
        # W_dec is (L, F, d), we want (F, L)
        return self.W_dec.norm(dim=-1).permute(1, 0)

    @torch.no_grad()
    def decoder_directions(self) -> torch.Tensor:
        """
        Unit-normalized decoder directions per feature per layer.

        Returns
        -------
        (n_layers, n_features, d_model) — same shape as W_dec but unit-normed
        """
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.W_dec / norms
