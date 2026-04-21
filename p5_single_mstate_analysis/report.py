"""
report.py — Assemble cluster_report.txt from Group A–G JSON fragments.

Pure text consumer. Reads whichever group_{A,B,C1,C2,D,E,F,G}_*.json files
exist in the run directory, produces a single diffable ASCII report.

Conventions
-----------
- No plots, no embedded images, no nested JSON
- ASCII `===` for section banners, `---` for subsections
- Fixed-precision numbers (4 decimals by default, 2 for sizes)
- Every verdict line names its metric and threshold
- Missing groups are noted explicitly rather than silently skipped
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

BAR_THICK = "=" * 64
BAR_THIN  = "-" * 64


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": f"failed to load {path.name}: {e}"}


def _fmt(v, p=4):
    """Fixed-precision number formatter. Handles None / nan / bool cleanly."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        return f"{v:.{p}f}"
    return str(v)


def _bullet(label: str, value, p=4) -> str:
    return f"  {label:<40s} {_fmt(value, p)}"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(selection: dict, model: str) -> list:
    primary = selection["primary"]
    runner_up = selection.get("runner_up")
    sibling = selection.get("sibling")
    merge = primary.get("merge_event")

    lines = [
        BAR_THICK,
        "PHASE 5 CASE STUDY — CLUSTER REPORT",
        BAR_THICK,
        f"model:           {model}",
        f"prompt:          {primary['prompt_key']}",
        f"trajectory_id:   {primary['id']}",
        f"start_layer:     {primary['start_layer']}",
        f"end_layer:       {primary['end_layer']}",
        f"lifespan:        {primary['lifespan']}",
        f"merge_event:     " + (
            f"L{merge['layer_from']} -> L{merge['layer_to']} "
            f"(prev_ids={merge['prev_ids']}, curr_id={merge['curr_id']})"
            if merge else "none"
        ),
        f"sibling_id:      {primary.get('sibling_id')}",
        f"total_score:     {_fmt(primary.get('total_score'), 3)}",
    ]
    sub = primary.get("sub_scores", {})
    if sub:
        lines.append("sub_scores:")
        for k in ("lifespan", "merge", "semantic", "preferred_prompt",
                  "size", "sibling"):
            lines.append(f"  {k:<20s} {_fmt(sub.get(k), 3)}")

    if runner_up:
        lines += [
            "",
            f"runner_up:       id={runner_up['id']} prompt={runner_up['prompt_key']} "
            f"score={_fmt(runner_up.get('total_score'), 3)}",
        ]
    if sibling and "note" in sibling:
        lines += [f"sibling_note:    {sibling['note']}"]

    return lines


def _render_group_A(profile: Optional[dict]) -> list:
    if profile is None:
        return [BAR_THIN, "A. STRUCTURAL PROFILE", BAR_THIN, "  (group A: no data)"]
    if "_error" in profile:
        return [BAR_THIN, "A. STRUCTURAL PROFILE", BAR_THIN,
                f"  ERROR: {profile['_error']}"]

    lines = [BAR_THIN, "A. STRUCTURAL PROFILE", BAR_THIN]

    # A.1 Token membership
    lines += ["", "A.1 Token membership by layer"]
    for p in profile.get("per_layer", []):
        toks = p.get("tokens", [])
        lines.append(
            f"  L{p['layer']:02d} (cid={p['cluster_id']}, n={p['n']}): "
            + " ".join(toks[:12]) + (" …" if len(toks) > 12 else "")
        )

    # A.2 Compactness per layer
    lines += ["", "A.2 Compactness per layer (ip_mean, radius, sub_k)"]
    for p in profile.get("per_layer", []):
        lines.append(
            f"  L{p['layer']:02d}: ip_mean={_fmt(p.get('ip_mean'))}, "
            f"radius={_fmt(p.get('radius'))}, "
            f"sub_k={p.get('nesting_sub_k')}"
        )

    # A.3 Silhouettes per layer
    lines += ["", "A.3 Silhouettes per layer (vs sibling | vs complement)"]
    for p in profile.get("per_layer", []):
        lines.append(
            f"  L{p['layer']:02d}: sib={_fmt(p.get('silhouette_sib'))}  "
            f"all={_fmt(p.get('silhouette_all'))}"
        )

    # A.4 Centroid geometry
    lines += ["", "A.4 Centroid geometry"]
    steps = profile.get("centroid_angular_steps", [])
    arc   = profile.get("cumulative_arc", [])
    if steps:
        lines.append(
            "  angular steps (rad): " + ", ".join(f"{s:.3f}" for s in steps)
        )
        lines.append(f"  cumulative arc (rad): {_fmt(arc[-1] if arc else None)}")

    # A.5 Membership stability
    lines += ["", "A.5 Membership stability (Jaccard L→L+1)"]
    jac = profile.get("membership_jaccard", [])
    if jac:
        lines.append("  " + ", ".join(f"{j:.3f}" if j is not None else "n/a"
                                       for j in jac))

    # A.6 CKA
    cka = profile.get("cka_restricted", [])
    if cka:
        lines += ["", "A.6 CKA restricted to cluster (consecutive layers)",
                  "  " + ", ".join(f"{c:.3f}" if c is not None else "n/a"
                                    for c in cka)]

    # A.7 Mass-near-1 contribution
    lines += ["", "A.7 Mass-near-1 contribution (fraction of layer's "
              "|<x,x>|>0.95 pairs that are cluster-internal)"]
    for p in profile.get("per_layer", []):
        m = p.get("mass_near_1", {})
        lines.append(
            f"  L{p['layer']:02d}: {m.get('cluster_pairs', 0)}/"
            f"{m.get('total_pairs_near_1', 0)} = {_fmt(m.get('fraction'))}"
        )

    # Summary
    s = profile.get("summary", {})
    lines += ["", "A.SUMMARY"]
    for k in ("mean_size", "mean_ip_mean", "mean_radius",
              "mean_silhouette_sib", "mean_silhouette_all",
              "mean_angular_step", "total_arc_length",
              "mean_jaccard_stability", "mean_cka_restricted",
              "mean_mass_near_1_frac"):
        if k in s:
            lines.append(_bullet(k, s[k]))
    return lines


def _render_group_B(v: Optional[dict]) -> list:
    if v is None:
        return [BAR_THIN, "B. PAPER-THEORETICAL ALIGNMENT", BAR_THIN,
                "  (group B: no data)"]
    if "_error" in v:
        return [BAR_THIN, "B. PAPER-THEORETICAL ALIGNMENT", BAR_THIN,
                f"  ERROR: {v['_error']}"]

    lines = [BAR_THIN, "B. PAPER-THEORETICAL ALIGNMENT", BAR_THIN]

    # B.1 beta estimates
    lines += ["", "B.1 Effective β per layer (mean over heads)"]
    for b in v.get("beta_per_layer", []):
        lines.append(
            f"  L{b['layer']:02d}: mean={_fmt(b.get('mean'), 3)}  "
            f"median={_fmt(b.get('median'), 3)}"
        )

    # B.2 mass-near-1 trajectory vs theorem 6.3
    lines += ["", "B.2 Cluster-internal mass-near-1 (with Thm 6.3 bound)"]
    for m in v.get("mass_trajectory", []):
        lines.append(
            f"  L{m['layer']:02d} (n={m['n']}): "
            f"mass={_fmt(m.get('mass_near_1'))}  "
            f"thm6.3_pred={_fmt(m.get('theorem_6_3_pred'))}"
        )

    # B.3 E_β trajectory
    lines += ["", "B.3 Cluster energy E_β per layer"]
    E = v.get("energy_trajectory", [])
    lines.append("  " + ", ".join(_fmt(e) for e in E))

    # B.4 centroid decomposition
    lines += ["", "B.4 Centroid decomposition (attr | rep | orth fractions)"]
    for d in v.get("centroid_decomp", []):
        lines.append(
            f"  step {d['step']:02d}: "
            f"attr={_fmt(d.get('attr_frac'))}  "
            f"rep={_fmt(d.get('rep_frac'))}  "
            f"orth={_fmt(d.get('orth_frac'))}"
        )

    # B.5 displacement decomposition
    lines += ["", "B.5 Displacement Δx̄ decomposition"]
    for d in v.get("displacement_decomp", []):
        lines.append(
            f"  step {d['step']:02d}: "
            f"attr={_fmt(d.get('attr_frac'))}  "
            f"rep={_fmt(d.get('rep_frac'))}  "
            f"orth={_fmt(d.get('orth_frac'))}"
        )

    # B.6 S/A local test
    sa = v.get("sa_local_test", {})
    lines += ["", "B.6 Local rotational (S/A) test"]
    if sa.get("available"):
        lines.append(_bullet("mean_asym_frac", sa.get("mean_asym_frac")))
        lines.append(_bullet("verdict (thresh 0.1)", sa.get("verdict")))
    else:
        lines.append("  (phase2i artifacts unavailable)")

    # B.7 Schur blocks
    lines += ["", "B.7 Top Schur blocks by centroid overlap"]
    for b in v.get("schur_blocks", []):
        lines.append(
            f"  idx={b['start_idx']:03d}: overlap_c={_fmt(b.get('overlap_c'))}  "
            f"eig=({_fmt(b.get('eig_real'))}{'+' if b.get('eig_imag',0)>=0 else ''}"
            f"{_fmt(b.get('eig_imag'))}i)"
        )

    # B.8 merge geometry
    mg = v.get("merge_geometry")
    lines += ["", "B.8 Merge-event geometry"]
    if mg:
        lines.append(_bullet("pre_merge_cosine", mg.get("pre_merge_cosine")))
        lines.append(_bullet("pre_merge_angle_rad", mg.get("pre_merge_angle_rad")))
        lines.append(_bullet("fusion_dir_magnitude", mg.get("fusion_dir_magnitude")))
        lines.append(_bullet("fusion_attr_alignment", mg.get("fusion_attr_alignment")))
        lines.append(_bullet("fusion_rep_alignment", mg.get("fusion_rep_alignment")))
        lines.append(_bullet("verdict", mg.get("verdict")))
    else:
        lines.append("  (no merge event)")

    # Summary
    s = v.get("summary", {})
    lines += ["", "B.SUMMARY"]
    for k in ("mean_centroid_attr_frac", "mean_centroid_rep_frac",
              "mean_beta_estimate", "mean_cluster_energy",
              "rotational_neutral_local"):
        if k in s:
            lines.append(_bullet(k, s[k]))
    return lines


def _render_group_C1(c1: Optional[dict]) -> list:
    if c1 is None:
        return [BAR_THIN, "C.1 PER-HEAD ATTENTION CONTRIBUTIONS", BAR_THIN,
                "  (group C.1: no data)"]
    if "_error" in c1:
        return [BAR_THIN, "C.1 PER-HEAD ATTENTION CONTRIBUTIONS", BAR_THIN,
                f"  ERROR: {c1['_error']}"]

    lines = [BAR_THIN, "C.1 PER-HEAD ATTENTION CONTRIBUTIONS", BAR_THIN]

    # Top heads by cumulative cohesion
    lines += ["", "C.1.0 Top attractor heads (cumulative cohesion over lifespan)"]
    for h in c1.get("top_attractor_heads", []):
        lines.append(f"  head {h['head']:02d}: cohesion={_fmt(h.get('cohesion'))}")

    # Per-layer head summary (classifications only — full per-head would
    # bloat the report)
    lines += ["", "C.1.1 Per-layer head classifications (counts)"]
    for pl in c1.get("per_layer", []):
        counts = {}
        for h in pl.get("per_head", []):
            counts[h["classification"]] = counts.get(h["classification"], 0) + 1
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"  L{pl['layer']:02d}: {counts_str}")

    # Per-layer top heads with cohesion details
    lines += ["", "C.1.2 Per-layer detail for top-3 attractor heads"]
    top_ids = [h["head"] for h in c1.get("top_attractor_heads", [])[:3]]
    for pl in c1.get("per_layer", []):
        for h in pl.get("per_head", []):
            if h["head"] not in top_ids:
                continue
            lines.append(
                f"  L{pl['layer']:02d} h{h['head']:02d}: "
                f"class={h['classification']}  "
                f"inward={_fmt(h.get('inward_mass'))}  "
                f"coh={_fmt(h.get('cohesion'))}  "
                f"fiedler={_fmt(h.get('fiedler'))}"
            )
            # Top QK pairs for this head at this layer
            for pair in h.get("top_pairs", [])[:2]:
                lines.append(
                    f"      pair ({pair['tok_i']!r}, {pair['tok_j']!r}) "
                    f"A={_fmt(pair['attn'])}"
                )
            # OV alignment
            ov = h.get("ov_alignment", {})
            top_ev = ov.get("top_eigenvectors", [])
            if top_ev:
                evstr = "; ".join(
                    f"λ={_fmt(e['eigval'],3)} ({e['sign']}) "
                    f"overlap={_fmt(e['overlap'])}"
                    for e in top_ev[:2]
                )
                lines.append(f"      OV: {evstr}")

    return lines


def _render_group_C2(c2: Optional[dict]) -> list:
    if c2 is None:
        return [BAR_THIN, "C.2 FFN CONTRIBUTIONS", BAR_THIN,
                "  (group C.2: no data — phase2 decompose artifacts missing?)"]
    if "_error" in c2:
        return [BAR_THIN, "C.2 FFN CONTRIBUTIONS", BAR_THIN,
                f"  ERROR: {c2['_error']}"]

    lines = [BAR_THIN, "C.2 FFN CONTRIBUTIONS", BAR_THIN]

    lines += ["", "C.2.1 Projections per layer (FFN and ATTN onto centroid, LDA)"]
    for p in c2.get("per_layer", []):
        head = f"  L{p['layer']:02d} (cid={p['cluster_id']}): "
        parts = []
        if "ffn_on_centroid" in p:
            parts.append(
                f"ffn·c̄  mean={_fmt(p['ffn_on_centroid']['mean_proj'])} "
                f"frac={_fmt(p['ffn_on_centroid']['energy_frac'])}"
            )
        if "attn_on_centroid" in p:
            parts.append(
                f"attn·c̄ mean={_fmt(p['attn_on_centroid']['mean_proj'])} "
                f"frac={_fmt(p['attn_on_centroid']['energy_frac'])}"
            )
        if "ffn_on_lda" in p:
            parts.append(
                f"ffn·LDA mean={_fmt(p['ffn_on_lda']['mean_proj'])} "
                f"frac={_fmt(p['ffn_on_lda']['energy_frac'])}"
            )
        if "attn_on_lda" in p:
            parts.append(
                f"attn·LDA mean={_fmt(p['attn_on_lda']['mean_proj'])}"
            )
        if parts:
            lines.append(head + "  ".join(parts))
        else:
            lines.append(head + "(no ffn deltas available for this layer)")

    s = c2.get("summary", {})
    lines += ["", "C.2.SUMMARY"]
    for k in ("mean_ffn_cohesion", "mean_attn_cohesion",
              "mean_ffn_lda_frac", "ffn_cohesion_verdict"):
        if k in s:
            lines.append(_bullet(k, s[k]))
    return lines


def _render_group_D(d: Optional[dict]) -> list:
    if d is None:
        return [BAR_THIN, "D. FEATURE SIGNATURES", BAR_THIN,
                "  (group D: no data)"]
    if "_error" in d:
        return [BAR_THIN, "D. FEATURE SIGNATURES", BAR_THIN,
                f"  ERROR: {d['_error']}"]

    lines = [BAR_THIN, "D. FEATURE SIGNATURES", BAR_THIN]

    lines += ["", "D.1 Top identity features per layer (by MI)"]
    for p in d.get("per_layer", []):
        top = p.get("top_features", [])
        top_str = ", ".join(
            f"#{t['feature']}({_fmt(t['mi_bits'],3)}b)" for t in top[:6]
        )
        lines.append(f"  L{p['layer']:02d}: {top_str}")

    lines += ["", "D.2 Chorus: connected components (Jaccard ≥ 0.3)"]
    for p in d.get("per_layer", []):
        c = p.get("chorus", {})
        comps = c.get("components", [])
        if comps:
            comp_strs = ", ".join(f"{{{','.join(str(x) for x in comp[:6])}"
                                   + (",…" if len(comp) > 6 else "")
                                   + "}" for comp in comps[:3])
            lines.append(
                f"  L{p['layer']:02d}: n_components={c.get('n_components',0)} "
                f"max_jac={_fmt(c.get('max_jaccard'))} : {comp_strs}"
            )
        else:
            lines.append(f"  L{p['layer']:02d}: no multi-feature components")

    # Activation trajectories for a few top features
    lines += ["", "D.3 Activation trajectories (top features, rates per layer)"]
    traj = d.get("activation_trajectories", {})
    for f, pts in list(traj.items())[:6]:
        rates = ", ".join(f"L{p['layer']:02d}:{p['rate']:.2f}" for p in pts)
        lines.append(f"  feature #{f}: {rates}")

    # Merge dynamics
    md = d.get("merge_dynamics")
    lines += ["", "D.4 Merge-event feature dynamics"]
    if md:
        lines += [
            _bullet("n_died", md.get("n_died")),
            _bullet("n_born", md.get("n_born")),
            _bullet("n_survived", md.get("n_survived")),
        ]
        if md.get("died"):
            lines.append(f"  died_examples:    {md['died'][:10]}")
        if md.get("born"):
            lines.append(f"  born_examples:    {md['born'][:10]}")
        if md.get("survived"):
            lines.append(f"  survived_examples:{md['survived'][:10]}")
    else:
        lines.append("  (no merge event or feature acts missing at merge layers)")

    # LDA stability + V alignment
    ls = d.get("lda_stability", {})
    lv = d.get("lda_v_alignment", {})
    lines += ["", "D.5 LDA stability and V-alignment at merge"]
    if ls:
        lines += [
            _bullet("mean_stability (cos L→L+1)", ls.get("mean_stability")),
            _bullet("min_stability",  ls.get("min_stability")),
        ]
    if lv.get("available"):
        lines += [
            _bullet("lda_layer_used", lv.get("lda_layer_used")),
            _bullet("attr_alignment",  lv.get("attr_alignment")),
            _bullet("rep_alignment",   lv.get("rep_alignment")),
            _bullet("verdict",          lv.get("verdict")),
        ]

    # Decoder geometry
    geo = d.get("decoder_geometry", [])
    if geo:
        lines += ["", "D.6 Decoder-direction geometry (top features)"]
        for g in geo[:6]:
            parts = [f"#{g['feature']:04d}"]
            for k in ("cos_centroid", "cos_lda", "attr_alignment", "rep_alignment"):
                if k in g:
                    parts.append(f"{k}={_fmt(g[k])}")
            lines.append("  " + "  ".join(parts))

    return lines


def _render_group_E(e: Optional[dict]) -> list:
    if e is None:
        return [BAR_THIN, "E. TUNED-LENS DECODING", BAR_THIN,
                "  (group E: no data)"]
    if "_error" in e or "error" in e:
        return [BAR_THIN, "E. TUNED-LENS DECODING", BAR_THIN,
                f"  ERROR: {e.get('_error') or e.get('error')}"]

    lines = [BAR_THIN, "E. TUNED-LENS DECODING", BAR_THIN]
    lines.append(f"  used_tuned_lens: {e.get('used_tuned_lens')}")
    lines += ["", "E.1 Per-layer centroid top tokens (with entropy)"]
    for p in e.get("per_layer", []):
        top = p.get("top_centroid", [])
        top_str = ", ".join(
            f"{t['token']!s}({_fmt(t['prob'],4)})" for t in top[:8]
        )
        lines.append(
            f"  L{p['layer']:02d}: H={_fmt(p.get('entropy'),3)}  {top_str}"
        )

    lines += ["", "E.2 Top-1/top-5 stability across layers"]
    for s in e.get("stability", []):
        lines.append(
            f"  L{s['layer_from']:02d}→L{s['layer_to']:02d}: "
            f"top1_match={str(s['top1_match']).lower()}  "
            f"top5_overlap={s['top5_overlap']}"
        )

    return lines


def _render_group_F(f: Optional[dict]) -> list:
    if f is None:
        return [BAR_THIN, "F. CAUSAL TESTS", BAR_THIN,
                "  (group F: no data)"]
    if "_error" in f:
        return [BAR_THIN, "F. CAUSAL TESTS", BAR_THIN,
                f"  ERROR: {f['_error']}"]

    lines = [BAR_THIN, "F. CAUSAL TESTS", BAR_THIN]
    lines.append(f"  target_layer: {f.get('target_layer')}")

    baseline = f.get("baseline_state", [])
    if baseline:
        lines += ["", "F.0 Baseline cluster state"]
        for b in baseline:
            lines.append(
                f"  L{b['layer']:02d}: size={b['size']}  "
                f"mass_near_1={_fmt(b.get('mass_near_1'))}"
            )

    interventions = f.get("interventions", {})
    for name, iv in interventions.items():
        lines += ["", f"F.{name.upper()}"]
        for k in ("head", "target_layer", "alpha", "patched_token_idx"):
            if k in iv:
                lines.append(_bullet(k, iv[k]))
        rc = iv.get("recluster", {}).get("per_layer", [])
        if rc:
            lines.append("  recluster outcome per layer:")
            for r in rc:
                lines.append(
                    f"    L{r['layer']:02d}: frac_together={_fmt(r.get('frac_together'))}  "
                    f"fraction_noise={_fmt(r.get('fraction_noise'))}  "
                    f"dominant_label={r.get('dominant_new_label')}"
                )
    return lines


def _render_group_G(g: Optional[dict]) -> list:
    if g is None:
        return [BAR_THIN, "G. SIBLING + RANDOM CONTROL", BAR_THIN,
                "  (group G: no data)"]
    if "_error" in g:
        return [BAR_THIN, "G. SIBLING + RANDOM CONTROL", BAR_THIN,
                f"  ERROR: {g['_error']}"]

    lines = [BAR_THIN, "G. SIBLING + RANDOM CONTROL", BAR_THIN]

    # Contrast summary first
    cs = g.get("contrast_summary", {})
    lines += ["", "G.0 Contrast summary"]
    for k in ("sibling_mean_ip", "random_mean_ip",
              "sibling_mean_silhouette_all", "random_mean_silhouette_all"):
        if k in cs:
            lines.append(_bullet(k, cs[k]))

    # Sibling profile summary
    sp = g.get("sibling_profile", {}).get("summary", {})
    if sp:
        lines += ["", "G.1 Sibling profile summary"]
        for k in ("mean_size", "mean_ip_mean", "mean_radius",
                  "mean_silhouette_all", "mean_jaccard_stability",
                  "mean_cka_restricted"):
            if k in sp:
                lines.append(_bullet(k, sp[k]))

    # Random control summary
    rp = g.get("random_control_profile", {}).get("summary", {})
    if rp:
        lines += ["", "G.2 Random-control profile summary"]
        for k in ("mean_size", "mean_ip_mean", "mean_radius",
                  "mean_silhouette_all", "mean_jaccard_stability",
                  "mean_cka_restricted"):
            if k in rp:
                lines.append(_bullet(k, rp[k]))

    # Sibling head top attractors
    sh = g.get("sibling_heads", {})
    if sh and "top_attractor_heads" in sh:
        lines += ["", "G.3 Sibling top attractor heads"]
        for h in sh["top_attractor_heads"][:5]:
            lines.append(f"  head {h['head']:02d}: cohesion={_fmt(h.get('cohesion'))}")

    return lines


def _render_ranked_appendix(selection: dict, top_n: int = 10) -> list:
    lines = [BAR_THIN, "APPENDIX: FULL RANKED TRAJECTORY LIST", BAR_THIN]
    lines.append(
        f"  {'rank':<4} {'prompt':<20} {'id':<5} "
        f"{'life':<5} {'score':<7} {'sub_scores'}"
    )
    for rank, c in enumerate(selection.get("ranked", [])[:top_n]):
        sub = c.get("sub_scores", {})
        subs = "/".join(f"{sub.get(k, 0):.2f}" for k in
                         ("lifespan", "merge", "semantic",
                          "preferred_prompt", "size", "sibling"))
        lines.append(
            f"  {rank:<4} {c['prompt_key'][:20]:<20} {c['id']:<5} "
            f"{c['lifespan']:<5} {c.get('total_score',0):<7.3f} {subs}"
        )
    return lines


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

def build_report(
    run_dir: Path,
    model: str,
    tag: str = "primary",
) -> str:
    """
    Read all group fragments from run_dir and return the full report as a
    single string.
    """
    run_dir = Path(run_dir)

    # Selection metadata
    selection = _load_json(run_dir / "cluster_metadata.json") or {}
    if "primary" not in selection:
        return BAR_THICK + "\nERROR: cluster_metadata.json missing or invalid\n" + BAR_THICK

    profile = _load_json(run_dir / f"group_A_profile_{tag}.json")
    v_align = _load_json(run_dir / f"group_B_v_alignment_{tag}.json")
    c1      = _load_json(run_dir / f"group_C1_heads_{tag}.json")
    c2      = _load_json(run_dir / f"group_C2_ffn_{tag}.json")
    d_feat  = _load_json(run_dir / f"group_D_features_{tag}.json")
    e_lens  = _load_json(run_dir / f"group_E_tuned_lens_{tag}.json")
    f_cause = _load_json(run_dir / "group_F_causal.json")
    g_sib   = _load_json(run_dir / "group_G_sibling_contrast.json")

    all_lines = []
    all_lines += _render_header(selection, model)
    all_lines += [""]
    all_lines += _render_group_A(profile)
    all_lines += [""]
    all_lines += _render_group_B(v_align)
    all_lines += [""]
    all_lines += _render_group_C1(c1)
    all_lines += [""]
    all_lines += _render_group_C2(c2)
    all_lines += [""]
    all_lines += _render_group_D(d_feat)
    all_lines += [""]
    all_lines += _render_group_E(e_lens)
    all_lines += [""]
    all_lines += _render_group_F(f_cause)
    all_lines += [""]
    all_lines += _render_group_G(g_sib)
    all_lines += [""]
    all_lines += _render_ranked_appendix(selection)
    all_lines += [BAR_THICK, "END OF REPORT", BAR_THICK]

    return "\n".join(all_lines) + "\n"


def write_report(run_dir: Path, model: str, tag: str = "primary") -> Path:
    out_path = Path(run_dir) / "cluster_report.txt"
    text = build_report(run_dir, model, tag=tag)
    out_path.write_text(text)
    return out_path
