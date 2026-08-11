"""
p1c_frames — Phase 1c: Frames, Moments, and the Closed-Form Trajectory.

A re-analysis phase. Its unit of work is the whole checkpoint series at
once, not one forward pass, which is why it is a directory rather than an
extension of analysis_p1's per-layer loop. It has its own falsification
structure (P-γ1, P-γ2, P-H1, P-S1 in PREDICTIONS.md), which is what has
always earned a phase directory here.

Depends on : Phase 1 artifacts (geometry.json, energies.json,
             activations.npz, sublayer_streams), core/ln_frame.py,
             core/beta_eff.py, core/functional_distance.py.
Depends NOT on Phase 2. Runnable now.

Sub-experiments:
    A  integration_time.py  T_eff = sum_l h_l                        [R]
    B  gamma_null.py        the gamma_beta residual curve            [R+W]
    C  moments.py           cumulant ladder / moment identity        [R]
    D  frame_table.py       four-frame comparison                    [W]
    E  hemisphere_feasibility.py  the cone condition (Lemma 6.4)     [R]
    F  design_test.py       spherical designs (sec. 9.1)             [R]

All six are implemented. Driver: run_1c.py; artifact IO: p1c_io.py.

F is wired as of this revision. The concern that blocked it — that the
clusterer choice moves the random baseline through the centroid count m —
was measured and does not hold: the matched-(m, d) baseline keeps
Q_k/Q_k^random flat at 1 across a 32x range in m. See centroids.py.
"""

from .gamma_ode import (
    integrate_gamma, integrate_gamma_converged, gamma_at,
    collapse_time, collapse_time_table, time_to_threshold,
)
from .integration_time import (
    sa_field, field_magnitude, step_sizes, cumulative_time, verdict,
)
from .gamma_null import (
    gamma_null, residual_curve, collapse_fraction, adjudicate_p_gamma1,
    time_residual_curve,
)
from .moments import (
    verify_moment_identity, ladder_from_layer, rank_panel,
    adjudicate_sink_hypothesis,
)
from .hemisphere_feasibility import (
    wendel_probability, hull_min_norm, hemisphere_test, hemisphere_profile,
)
from .frame_table import (
    gamma_dynamic_range, sphere_license, bias_energy_floor, torgerson_gram,
    frame_moments, frame_table, frame_disagreement,
)
from .beta_reduction import (
    reduce_beta, reduction_report, beta_envelope, envelope_verdict,
    residual_bracket, REDUCTIONS,
)
from .centroids import (
    centroids_from_labels, load_centroids, random_band, run_design_test,
    adjudicate_p_s1_banded,
)
from .p1c_io import load_run, raw_states, layer_series, save_p1c
from .design_test import (
    gegenbauer_normalized, gegenbauer_moments, design_order,
    random_baseline_Q, inner_product_modes, design_report, adjudicate_p_s1,
)

__all__ = [
    "integrate_gamma", "integrate_gamma_converged", "gamma_at",
    "collapse_time", "collapse_time_table", "time_to_threshold",
    "sa_field", "field_magnitude", "step_sizes", "cumulative_time", "verdict",
    "gamma_null", "residual_curve", "collapse_fraction", "adjudicate_p_gamma1",
    "time_residual_curve",
    "verify_moment_identity", "ladder_from_layer", "rank_panel",
    "adjudicate_sink_hypothesis",
    "wendel_probability", "hull_min_norm", "hemisphere_test",
    "hemisphere_profile",
    "gegenbauer_normalized", "gegenbauer_moments", "design_order",
    "random_baseline_Q", "inner_product_modes", "design_report",
    "adjudicate_p_s1",
    "gamma_dynamic_range", "sphere_license", "bias_energy_floor",
    "torgerson_gram", "frame_moments", "frame_table", "frame_disagreement",
    "load_run", "raw_states", "layer_series", "save_p1c",
    "centroids_from_labels", "load_centroids", "random_band",
    "run_design_test", "adjudicate_p_s1_banded",
    "reduce_beta", "reduction_report", "beta_envelope", "envelope_verdict",
    "residual_bracket", "REDUCTIONS",
]
