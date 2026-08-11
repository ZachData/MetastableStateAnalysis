"""
p2d_operator_activation — Phase 2d: Operator-Activation Pairing.

A Phase 2 extension, not part of Phase 1c: everything here needs P2's M_h
and W_OV^{(h)}. Sequencing per the update plan: 2d runs AFTER 1c-B lands,
since the T_eff result determines whether the energy-monotonicity break is
even the right thing to attribute.

    D1  gradient_flow_condition.py  which heads are inside sec. 3.4      [W]
    D2  operator_pairing.py         PR_M, the operator/activation pair   [R+W]
    D3  table1_predictions.py       Table 1 as a falsifiable claim       [R+W]
    D4  operator_pairing.py         the model's own energy E_beta^(h)    [R+W]

Tests P-M1 (D1) and P-T1 (D3), both registered in PREDICTIONS.md.

    p2d_io.py   the join between P2 operators and P1 activations, with the
                revision / frame / width guards. This is the only genuinely
                dangerous step in the phase and is isolated here.
    run_2d.py   driver.
"""

from .gradient_flow_condition import (
    qk_matrix, symmetry_split, ov_qk_alignment, head_regime, layer_regimes,
    adjudicate_p_m1, SYMMETRY_TOL, ALIGN_TOL,
)
from .operator_pairing import (
    token_covariance, operator_conditioned_rank, spectral_pairing,
    generalized_energy, energy_attribution, monotonicity_compare,
)
from .p2d_io import (
    load_operators, join, revision_from_run, JoinRefused, resolve_ln_params,
)
from .table1_predictions import (
    classify_ov_row, projection_modality, modality_stability,
    rescaled_modality, adjudicate_p_t1,
)

__all__ = [
    "qk_matrix", "symmetry_split", "ov_qk_alignment", "head_regime",
    "layer_regimes", "adjudicate_p_m1", "SYMMETRY_TOL", "ALIGN_TOL",
    "token_covariance", "operator_conditioned_rank", "spectral_pairing",
    "generalized_energy", "energy_attribution", "monotonicity_compare",
    "classify_ov_row", "projection_modality", "modality_stability",
    "rescaled_modality",
    "adjudicate_p_t1",
    "load_operators", "join", "revision_from_run", "JoinRefused",
    "resolve_ln_params",
]
