"""
phase5_case — Single-cluster case study.

Consumes artifacts from Phases 1, 2, 2i, 3, 4 and reconstructs the end-to-end
mechanism for one HDBSCAN trajectory: structural profile, V-eigenspace
alignment, attention and FFN contributions, feature signatures, tuned-lens
semantics, causal interventions, sibling contrast.

Modules
-------
select_cluster     : rank and select primary + sibling trajectories
cluster_profile    : Group A — structural profile across lifespan
v_alignment        : Group B — paper-theoretical alignment
head_contributions : Group C.1 — per-head attention + cohesion scalars
ffn_contributions  : Group C.2 — FFN projection onto cluster directions
feature_signature  : Group D — identity features, chorus, LDA
tuned_lens_cluster : Group E — tuned lens decoding
causal_tests       : Group F — ablation, steering, patching
sibling_contrast   : Group G — sibling + random control
report             : assemble cluster_report.txt
run                : CLI entry point
"""
