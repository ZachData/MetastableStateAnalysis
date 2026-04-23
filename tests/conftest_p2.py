"""
conftest.py — pytest session-scoped import stubs.

eigendecompose / build_subspace_projectors / rescale_matrix and the
trajectory helpers are pure numpy/scipy.  They don't call torch or
transformers at test-time, but those packages are imported at the
module level in core/config.py, which is itself imported at the top
of weights.py and trajectory.py.

Stub every heavy dep in sys.modules *before* the first project import
so pytest can collect and run tests without a GPU or model weights.
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1. Stub torch so core/config.py doesn't need a real installation
# ---------------------------------------------------------------------------
_torch = MagicMock()
_torch.cuda.is_available.return_value = False
sys.modules.setdefault("torch", _torch)

# ---------------------------------------------------------------------------
# 2. Stub transformers (model classes imported at top of config.py)
# ---------------------------------------------------------------------------
for _mod in [
    "transformers",
    "transformers.models",
    "transformers.models.albert",
]:
    sys.modules.setdefault(_mod, MagicMock())

# ---------------------------------------------------------------------------
# 3. Inject a minimal core.config so the real module never runs
#    (avoids the torch.cuda call and model-class imports)
# ---------------------------------------------------------------------------
_core = types.ModuleType("core")
_core_config = types.ModuleType("core.config")
_core_config.BETA_VALUES = [0.1, 1.0, 2.0, 5.0]
_core_config.MODEL_CONFIGS = {}
_core_config.BASE_RESULTS_DIR = None
_core_config.DEVICE = "cpu"

sys.modules["core"] = _core
sys.modules["core.config"] = _core_config
