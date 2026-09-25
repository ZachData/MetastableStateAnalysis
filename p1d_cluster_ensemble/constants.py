"""
p1d_cluster_ensemble/constants.py — the two Phase 1 sweep constants, read
out of core/config.py without importing it.

core/config.py imports torch and transformers at module level. This phase
is pure re-analysis: numpy, scipy and sklearn, no model, no forward pass,
and every test in tests/ that touches it should run in an environment with
none of those installed. Importing core.config to reach DISTANCE_THRESHOLDS
would drag both in.

The alternative — copying the two literals here — is the failure this
project has already paid for once: p1_visualization/checkpoint_scalars.py
carried a hand-synced copy of ENERGY_VIOLATION_REL_TOL with a comment
asking future editors to remember, and the fix was to parse the constant
out of the source with `ast`. Same mechanism here, same reason: if either
constant is renamed, moved, or given a form this reader does not
understand, this module raises at import instead of quietly falling back
to a stale value.

The two constants matter because Phase 1d's agglomerative and k-based
grids must contain the settings Phase 1 actually ran, or the "is the
shipped setting the tuned one?" comparison (P-C2) is being made against a
grid point that does not exist.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "core" / "config.py"
_CLUSTER_METHODS = (_ROOT / "p1_mstate_tracking" / "visualization" / "cluster_methods.py")


def _literal_call(node: ast.AST) -> List[float]:
    """
    Evaluate the small set of constructor calls core/config.py actually
    uses for these two constants — np.linspace / np.arange / range with
    literal arguments — and nothing else.

    Deliberately not a general evaluator. A constant that grows a
    data-dependent definition should not be silently re-derived here; it
    should raise, and whoever changed it should decide what Phase 1d's
    grid ought to contain.
    """
    if isinstance(node, (ast.List, ast.Tuple, ast.Constant)):
        return [float(v) for v in _as_iterable(ast.literal_eval(node))]

    if not isinstance(node, ast.Call):
        raise ImportError(f"p1d.constants: unsupported constant form {ast.dump(node)[:80]}")

    name = (node.func.attr if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None))
    args = [ast.literal_eval(a) for a in node.args]

    if name == "linspace":
        import numpy as np
        return [float(v) for v in np.linspace(*args)]
    if name == "arange":
        import numpy as np
        return [float(v) for v in np.arange(*args)]
    if name == "range":
        return [float(v) for v in range(*args)]
    raise ImportError(
        f"p1d.constants: {name!r} is not a constant form this reader "
        "understands. Update the reader rather than restoring a literal."
    )


def _as_iterable(value):
    return value if isinstance(value, (list, tuple)) else [value]


def _constant(name: str, src: Path = None) -> List[float]:
    src = src or _CONFIG
    if not src.exists():
        raise ImportError(
            f"p1d.constants: cannot locate {src} to read {name}. Phase 1d "
            "must use the same constants the code it compares against uses; "
            "refusing to guess them."
        )
    tree = ast.parse(src.read_text())
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                return _literal_call(node.value)
    raise ImportError(
        f"p1d.constants: {name} not found in {src}. It was renamed or moved; "
        "update this reader rather than restoring a literal."
    )


#: The agglomerative distance-threshold sweep Phase 1 runs (12 values,
#: 0.05 .. 0.6). Phase 1 persists labels only at the middle threshold;
#: Phase 1d re-fits at every one of them.
DISTANCE_THRESHOLDS: List[float] = _constant("DISTANCE_THRESHOLDS")

#: The k range Phase 1's KMeans silhouette search covers (2 .. 9).
K_VALUES: List[int] = [int(k) for k in _constant("K_RANGE")]

#: The KMeans trust gate p1_visualization/cluster_methods.py applies before
#: counting KMeans's k as a real estimate, and which
#: reporting_p1._method_agreement applies in the text report. Phase 1d needs
#: both to reconstruct the "methods agree here" layer set P-C1 is registered
#: about — read out of that module rather than copied, for the same reason
#: DISTANCE_THRESHOLDS is: two hand-synced copies of an agreement criterion
#: produce two different agreement layer sets and nothing says which is stale.
KMEANS_SIL_MIN: float = _constant("KMEANS_SIL_MIN", _CLUSTER_METHODS)[0]
KMEANS_RANK_MIN: float = _constant("KMEANS_RANK_MIN", _CLUSTER_METHODS)[0]

#: HDBSCAN's shipped Phase 1 setting, the partition every cluster-conditioned
#: result in this project was computed on. P-C2 is adjudicated against
#: exactly this, not against a re-run with different defaults.
SHIPPED_HDBSCAN_PARAMS = {
    "min_cluster_size": 2,
    "min_samples": None,
    "cluster_selection_method": "eom",
    "cluster_selection_epsilon": 0.0,
}

#: A cluster smaller than this is a refusal spelled differently, not a
#: cluster. Same number and same argument as
#: p1_visualization/cluster_methods.noise_audit's min_cluster_size, kept
#: identical on purpose so the figure and this phase never disagree about
#: what counts as "placed in structure". PLACED, not calibrated.
SUBSTANTIAL_CLUSTER_SIZE = 4
