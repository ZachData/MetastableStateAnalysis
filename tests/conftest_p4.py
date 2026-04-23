"""
conftest.py — bootstrap stubs required by p4_mstate_features modules.

activation_trajectories.py and chorus.py both import torch (unavailable in the
test environment) and phase3.{crosscoder,data} (require a live model checkpoint).
This file installs minimal stubs into sys.modules before any test module is
collected, so the pure-numpy/scipy computation logic can be imported and tested
without those dependencies.

The stub must satisfy three constraints:
  1. torch.Tensor must be a real class — scipy.stats uses issubclass() on it.
  2. torch.no_grad() must behave as a context manager (used in
     extract_activation_trajectories, which we never call in tests).
  3. chorus.py uses a relative import  `from .activation_trajectories import …`
     which requires both modules to share the p4_mstate_features package
     identity in sys.modules.
"""
import sys
import types
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 1.  torch stub
# ---------------------------------------------------------------------------

def _make_torch_stub() -> types.ModuleType:
    stub = types.ModuleType("torch")

    class Tensor:
        """Minimal real class so scipy's issubclass checks don't raise."""

    _cm = MagicMock()
    _cm.__enter__ = MagicMock(return_value=None)
    _cm.__exit__ = MagicMock(return_value=False)

    stub.Tensor = Tensor
    stub.no_grad = MagicMock(return_value=_cm)
    stub.nn = MagicMock()
    return stub


def _install_stubs() -> None:
    if "torch" in sys.modules:
        return  # already done (e.g. pytest re-uses the process)

    torch_stub = _make_torch_stub()
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_stub.nn

    for name in ("phase3", "phase3.crosscoder", "phase3.data"):
        sys.modules[name] = MagicMock()


# ---------------------------------------------------------------------------
# 2.  p4_mstate_features package + submodule loader
# ---------------------------------------------------------------------------

_P4_SRC = Path(__file__).parent.parent  # project root (where *.py files live)


def _ensure_p4_package() -> None:
    """Register a fake p4_mstate_features package pointing at the project root."""
    if "p4_mstate_features" in sys.modules:
        return

    pkg = types.ModuleType("p4_mstate_features")
    pkg.__path__ = [str(_P4_SRC)]
    pkg.__package__ = "p4_mstate_features"
    sys.modules["p4_mstate_features"] = pkg


def _load_p4_submodule(filename: str) -> types.ModuleType:
    """Load a file from the project root as p4_mstate_features.<stem>."""
    stem = Path(filename).stem
    full_name = f"p4_mstate_features.{stem}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    filepath = _P4_SRC / filename
    spec = importlib.util.spec_from_file_location(full_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "p4_mstate_features"
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 3.  Run at collection time
# ---------------------------------------------------------------------------

_install_stubs()
_ensure_p4_package()

# Pre-load both p4 modules so their symbols are available to test files that
# import them at module level.  Order matters: chorus imports from
# activation_trajectories via a relative import.
_load_p4_submodule("activation_trajectories.py")
_load_p4_submodule("chorus.py")
