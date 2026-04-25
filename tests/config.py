"""
Stub core/config.py for testing.
Provides only the constants consumed by p1_mstate_tracking modules.
No model registry, no transformers imports.
"""
import numpy as np

SINKHORN_MAX_ITER = 100
SINKHORN_TOL     = 1e-6
SPECTRAL_MAX_K   = 15

# Must match the production value so test fixtures are geometrically consistent.
DISTANCE_THRESHOLDS = np.linspace(0.05, 0.6, 12)
K_RANGE             = range(2, 10)

N_LAYERS=6
N_TOKENS=40
D=16