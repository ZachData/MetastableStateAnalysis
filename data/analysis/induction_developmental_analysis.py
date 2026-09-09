"""Read-out of the 2026-09-08 induction generalisation batch (PROJECT.md
sec 3.11): does the L7H8 Stage-2 picture hold across development and across
the other behavioural induction heads?

Inputs (all exploratory, nothing registered):
  induction_rank_sweep_s<step>_L7H8.json  -- L7H8 OV rank sweep at 19 steps
  induction_rank_sweep.json               -- the step-4000 one, committed
  induction_subspace_characterize[_s<step>_L<layer>].json -- Stage 2 on the
                                             top behavioural heads

Prints two tables and writes induction_developmental_series.json.
"""
import json
import glob
import re
from pathlib import Path
import numpy as np

AN = Path(__file__).resolve().parent


def _stepof(f):
    m = re.search(r'_s(\d+)_L7H8', f)
    return int(m.group(1)) if m else 4000


def rank_sweep_trajectory():
    files = sorted(glob.glob(str(AN / 'induction_rank_sweep_s*_L7H8.json'))
                   + [str(AN / 'induction_rank_sweep.json')], key=_stepof)
    rows = []
    for f in files:
        d = json.load(open(f))
        s = _stepof(f)
        base = d['baseline']['second_copy_nll']
        C = {b: {r['rank']: r for r in d['curves'][b]} for b in d['curves']}
        r0 = C['svd'][0]['second_copy_nll']
        dn = r0 - base
        def fr(b, r):
            return ((r0 - C[b][r]['second_copy_nll']) / dn) if abs(dn) > 1e-5 else None
        rows.append({
            "step": s,
            "delta_ov_nll": dn,
            "kl_full_ablation": C['svd'][0]['kl_from_baseline'],
            "svd_frac": {r: fr('svd', r) for r in (1, 2, 4, 8, 16)},
            "schur_frac": {r: fr('schur', r) for r in (1, 2, 8, 16, 24)},
            "random_frac": {r: fr('random', r) for r in (16, 32, 48)},
        })
    return rows


TARGETS = [  # file, layer, head, step  (behavioural leaders + peak step)
    ("induction_subspace_characterize.json", 7, 8, 4000),
    ("induction_subspace_characterize.json", 7, 0, 4000),
    ("induction_subspace_characterize_s8000_L7.json", 7, 3, 8000),
    ("induction_subspace_characterize_s8000_L7.json", 7, 12, 8000),
    ("induction_subspace_characterize_s16000_L6.json", 6, 0, 16000),
    ("induction_subspace_characterize_s32000_L2.json", 2, 10, 32000),
    ("induction_subspace_characterize_s2000_L9.json", 9, 9, 2000),
    ("induction_subspace_characterize_s8000_L9.json", 9, 8, 8000),
    ("induction_subspace_characterize_s16000_L1.json", 1, 15, 16000),
]


def qk_trajectory():
    """L7H8 static-QK rank sweep across the axis: induction-attn recovered as
    a fraction of (baseline - full-static-QK-ablation), and r* per basis
    (smallest rank reaching >= 0.5 of the effect)."""
    def _s(f):
        m = re.search(r'_s(\d+)_L7H8', f)
        return int(m.group(1)) if m else 4000
    files = sorted(glob.glob(str(AN / 'induction_qk_sweep_s*_L7H8.json'))
                   + [str(AN / 'induction_qk_sweep.json')], key=_s)
    rows = []
    for f in files:
        d = json.load(open(f))
        s = _s(f)
        base = d['baseline']['induction_attn']
        C = {b: {r['rank']: r['induction_attn'] for r in d['curves'][b]} for b in d['curves']}
        r0 = C['svd'][0]
        dn = base - r0
        def fr(b, r):
            return ((C[b][r] - r0) / dn) if abs(dn) > 1e-4 else None
        def rstar(b):
            for r in sorted(C[b]):
                v = fr(b, r)
                if v is not None and v >= 0.5:
                    return r
            return None
        rows.append({
            "step": s, "baseline_induction_attn": base, "full_ablation_attn": r0,
            "svd_frac": {r: fr('svd', r) for r in (8, 16, 24, 32)},
            "schur_frac": {r: fr('schur', r) for r in (8, 12, 16, 24)},
            "random_frac": {r: fr('random', r) for r in (16, 24, 32)},
            "r_star": {"svd": rstar('svd'), "schur": rstar('schur'), "random": rstar('random')},
        })
    return rows


def head_characterisations():
    out = []
    for fn, L, H, S in TARGETS:
        d = json.load(open(AN / fn))
        lh = d.get('layer_heads') or d.get('layer7_heads')
        t = lh[str(H)]
        out.append({
            "layer": L, "head": H, "step": S,
            "delta_ov_nll": t["full_ablation_delta_nll"],
            "svd_r1_frac": t["svd_r1_frac_of_effect"],
            "schur_r1_frac": t["schur_r1_frac_of_effect"],
            "attractive_energy_fraction_core": t["attractive_energy_fraction_core"],
            "lambda_top_sign": t["lambda_top_sign"],
            "phi": t["phi_antisym_fro_fraction"],
            "henrici": t["henrici"],
            "svd_s1_share_of_fro": t["svd_s1_share_of_fro"],
            "approx_copy_diag_z_mean": t["approx_copy_diag_z_mean"],
            "approx_copy_frac_diag_is_rowmax": t["approx_copy_frac_diag_is_rowmax"],
        })
    return out


def _f(x, w=6, p=2):
    return (f"{x:+{w}.{p}f}" if x is not None else " " * (w - 3) + "  -")


if __name__ == "__main__":
    traj = rank_sweep_trajectory()
    print("=== A. L7H8 OV rank sweep across the axis "
          "(frac of full-ablation ΔNLL recovered) ===\n")
    print(f"{'step':>7} {'ΔOVnll':>8} {'KL':>6} | {'svd@1':>6} {'svd@2':>6} | "
          f"{'sch@1':>6} {'sch@8':>6} {'sch@16':>6} | {'rnd@16':>6}")
    for r in traj:
        print(f"{r['step']:>7} {r['delta_ov_nll']:+8.4f} {r['kl_full_ablation']:6.3f} | "
              f"{_f(r['svd_frac'][1])} {_f(r['svd_frac'][2])} | "
              f"{_f(r['schur_frac'][1])} {_f(r['schur_frac'][8])} {_f(r['schur_frac'][16])} | "
              f"{_f(r['random_frac'][16])}")

    hc = head_characterisations()
    print("\n=== B. Stage-2 across the top behavioural induction heads ===\n")
    print(f"{'head':>7} {'step':>6} | {'ΔOVnll':>7} | {'svd_r1':>6} {'sch_r1':>6} | "
          f"{'att_efr':>7} {'top_sgn':>9} {'φ':>5} {'henr':>5} | {'copy_z':>7} {'diagmax':>8}")
    for h in hc:
        print(f"L{h['layer']}H{h['head']:<2} {h['step']:>6} | {h['delta_ov_nll']:+7.3f} | "
              f"{h['svd_r1_frac']:+6.2f} {h['schur_r1_frac']:+6.2f} | "
              f"{h['attractive_energy_fraction_core']:7.3f} {h['lambda_top_sign']:>9} "
              f"{h['phi']:5.2f} {h['henrici']:5.2f} | "
              f"{h['approx_copy_diag_z_mean']:+7.3f} {h['approx_copy_frac_diag_is_rowmax']:8.5f}")

    qk = qk_trajectory()
    print("\n=== C. L7H8 STATIC-QK rank sweep across the axis "
          "(induction-attn recovered; r* = rank reaching 0.5) ===\n")
    print(f"{'step':>7} {'base':>6} | {'svd@16':>7} {'svd@32':>7} | {'sch@12':>7} {'sch@16':>7} | "
          f"{'rnd@16':>7} | {'r*svd':>6} {'r*sch':>6} {'r*rnd':>6}")
    for r in qk:
        print(f"{r['step']:>7} {r['baseline_induction_attn']:6.3f} | "
              f"{_f(r['svd_frac'][16], 7)} {_f(r['svd_frac'][32], 7)} | "
              f"{_f(r['schur_frac'][12], 7)} {_f(r['schur_frac'][16], 7)} | "
              f"{_f(r['random_frac'][16], 7)} | "
              f"{str(r['r_star']['svd']):>6} {str(r['r_star']['schur']):>6} {str(r['r_star']['random']):>6}")

    dest = AN / "induction_developmental_series.json"
    json.dump({"_what_this_is": __doc__,
               "rank_sweep_trajectory_L7H8_OV": traj,
               "rank_sweep_trajectory_L7H8_QK_static": qk,
               "head_characterisations": hc}, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")
