"""Dissipation v2: the violation-restricted subspace split (status-2.md item 5).

Phase 2's beta1.0_frac_repulsive decays 1.00 -> 0.56 over steps 8000..54000
while the repulsive share of |dissipation| over ALL boundaries stays flat. The
proposed discriminator is the repulsive share of the POSITIVE first-order term
at dE>0 boundaries -- restrict to energy violations, and to the part of the
first-order term that pushes E up, then ask how much of that is in the
repulsive OV subspace.

TWO cuts, and they disagree:

  BOUNDARY-LEVEL (this file's `_boundary` series): keep (step,prompt,layer)
    boundaries with actual_delta_E>0 and first_order>0, then
    sum(d_repulsive)/sum(first_order). Uses only the scalars already in
    dissipation_series.json / the pre-v2 sublayer series.

  PER-PARTICLE (`_perparticle` series): within each dE>0 boundary keep the
    particles whose first-order contribution is positive, sum their repulsive
    part / their total. Needs the v2_attn_pos_* fields added to
    dissipation_sublayer.py (2026-09-08 re-run).

RESULT: the per-particle cut -- the clean one -- does NOT reproduce the
frac_repulsive decay. The attention channel's violation-restricted repulsive
share sits at 0.6-0.9 across the whole trained regime and RISES over
8000->54000 (0.62 -> 0.89), opposite to Phase 2. So the decay is not an
artefact of the all-boundary aggregation; it is absent from the
dissipation-identity view of the attention channel entirely. Item 5 points
back at Phase 2's own frac_repulsive construction (a violation classification
on the OV eigenspectrum), not the energy-flow decomposition.
Early steps (<=8) are below the forward-Euler regime; their v2 values are not
meaningful.
"""
import json
from pathlib import Path
import numpy as np

AN = Path(__file__).resolve().parent
A = json.load(open(AN / "dissipation_series.json"))
B = json.load(open(AN / "dissipation_sublayer_series.json"))
STEPS = A["steps"]
PROMPTS = A["scored_prompts"]
NL = A["n_blocks"]
FORMING = range(8, 24)
P2_FRAC_REPULSIVE = {512: 1.00, 2000: 1.00, 8000: 0.97, 32000: 0.64,
                     54000: 0.56, 143000: 0.73}


def boundary_series(records, key_fmt, rep_key, att_key, layers):
    """repulsive share of the positive first-order term, boundary-level."""
    out = {}
    for s in STEPS:
        num = den = 0.0
        num_all = den_all = 0.0
        n_v = 0
        for p in PROMPTS:
            for l in layers:
                r = records.get(key_fmt(s, p, l))
                if r is None or r["actual_delta_E"] is None:
                    continue
                fo, dE = r["first_order"], r["actual_delta_E"]
                rep, att = r[rep_key], r[att_key]
                num_all += rep
                den_all += abs(rep) + abs(att)
                if dE > 0 and fo > 0:
                    n_v += 1
                    num += rep
                    den += fo
        out[s] = {"repulsive_share": (num / den) if den > 0 else None,
                  "all_boundary_signed_share": (num_all / den_all) if den_all > 0 else None,
                  "n_violation_boundaries": n_v}
    return out


def perparticle_series(pooled, layers):
    """repulsive share of the positive-part attention first-order term at
    dE>0 boundaries, per step, pooled over prompts x forming layers."""
    out = {}
    for s in STEPS:
        fo = rep = 0.0
        nb = 0
        for l in layers:
            p = pooled.get(f"{s}|{l}")
            if not p:
                continue
            fo += p.get("v2_fo", 0.0)
            rep += p.get("v2_rep", 0.0)
            nb += p.get("v2_n", 0)
        out[s] = {"repulsive_share": (rep / fo) if fo > 0 else None,
                  "n_boundaries": nb}
    return out


tierA_bnd = boundary_series(A["per_step_layer"], lambda s, p, l: f"1.0|{s}|{p}|{l}",
                            "d_repulsive", "d_attractive", FORMING)
tierB_bnd = boundary_series(B["per_step_layer"], lambda s, p, l: f"{s}|{p}|{l}",
                            "d_attn_repulsive", "d_attn_attractive", FORMING)
tierB_pp = perparticle_series(B["pooled_by_step_layer"], FORMING)

# per-head per-particle share (relay-carrying forming heads)
per_head = {}
for k, rec in B["per_head"].items():
    fo = np.array([x if x is not None else np.nan for x in rec["v2_pos_first_order"]], float)
    rp = np.array([x if x is not None else np.nan for x in rec["v2_pos_repulsive"]], float)
    per_head[k] = list(np.where(fo > 0, rp / fo, np.nan))

result = {
    "_what_this_is": __doc__,
    "steps": STEPS,
    "phase2_beta1.0_frac_repulsive_from_status2": P2_FRAC_REPULSIVE,
    "tierA_total_dX_boundary_forming_L8_23": tierA_bnd,
    "tierB_attn_channel_boundary_forming_L8_23": tierB_bnd,
    "tierB_attn_channel_perparticle_forming_L8_23": tierB_pp,
    "per_head_perparticle_repulsive_share": per_head,
}
dest = AN / "dissipation_v2_series.json"
json.dump(result, open(dest, "w"), indent=1)

hdr = f"{'step':>7} | {'P2':>5} | {'A bnd':>7} {'B bnd':>7} | {'B per-particle (clean)':>22} | nb"
print(hdr); print("-" * len(hdr))
for s in STEPS:
    def g(d, k="repulsive_share"):
        v = d[s][k]
        return f"{v:+.3f}" if v is not None else "   -   "
    p2 = P2_FRAC_REPULSIVE.get(s)
    print(f"{s:>7} | {(f'{p2:.2f}' if p2 else '  -  '):>5} | "
          f"{g(tierA_bnd):>7} {g(tierB_bnd):>7} | {g(tierB_pp):>22} | "
          f"{tierB_pp[s]['n_boundaries']}")
print(f"\nwrote {dest}")
