"""
Stub core/models.py for testing.
Provides only layernorm_to_sphere; no model loading, no device code.
"""
import torch
import torch.nn.functional as F

import pytest

# Tier: deps -- needs the heavy tier importable (torch / transformers /
# scikit-learn / matplotlib). No model download, no run artifacts.
# Measured, not assumed; see pyproject.toml markers.
pytestmark = pytest.mark.deps

def layernorm_to_sphere(activation: torch.Tensor) -> torch.Tensor:
    """L2-normalize each token vector onto the unit sphere."""
    return F.normalize(activation.float(), p=2, dim=-1)
