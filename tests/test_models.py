"""
Stub core/models.py for testing.
Provides only layernorm_to_sphere; no model loading, no device code.
"""
import torch
import torch.nn.functional as F


def layernorm_to_sphere(activation: torch.Tensor) -> torch.Tensor:
    """L2-normalize each token vector onto the unit sphere."""
    return F.normalize(activation.float(), p=2, dim=-1)
