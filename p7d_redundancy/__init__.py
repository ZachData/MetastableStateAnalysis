"""
p7d_redundancy — the redundancy set: which heads hold the induction regime,
when each of them formed, and what makes them a set rather than a list.

The causal instrument throughout is OV-ablation `dNLL`, never a weights-only
proxy: PROJECT.md §3.12-R ruled out `||OV||_F` (r² = 0.001, relation inverted),
§3.12-G6 ruled out every spectral field, and §3.12-S showed weight-space and
function-space overlap come apart. Membership here is defined causally for that
reason.

See design-7d.md for the five questions and why the serial reading was dropped,
status-7d.md for what is answered, and PROJECT.md §3.12-S/T/U for the results
and §3.14.2 for the programme. Nothing in this phase is registered: every
measurement is on pythia-410m, which is spent under `check_registry` rule 3.

NOT to be confused with `p7_motifs/`, which is the motif/relay programme behind
P-I1. In particular `p7_motifs/formation_curve.py` is a BEHAVIOURAL relay curve;
`member_formation_curves.py` here is the CAUSAL ablation curve. Different
instruments, different questions, similar names.
"""
