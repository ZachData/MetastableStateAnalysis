"""
report.py — Assemble cluster_report.txt from Group A–G JSON fragments.

Design principle: the .txt file is an LLM-friendly summary.
Per-layer tables, per-head breakdowns, and raw activation trajectories live in
the group JSON files. This report contains only:
  - identity / selection header
  - one top-level VERDICTS block (all groups, one line each)
  - per-group summary scalars and key verdicts
  - ranked trajectory appendix

If you need per-layer data, load the corresponding group_X_*.json directly.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

BAR_THICK = "=" * 64
BAR_THIN  = "-" * 64


# ---------------------------------------------------------------------------
# Helpers
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


def _kv(label: str, value, p=4) -> str:
    return f"  {label:<42s} {_fmt(value, p)}"


def _section(title: str) -> list:
    return [BAR_THIN, title, BAR_THIN]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _render_header(selection: dict, model: str) -> list:
    primary  = selection["primary"]
    runner   = selection.get("runner_up")
    sibling  = selection.get("sibling")
    merge    = primary.get("merge_event")

    lines = [
        BAR_THICK,
        "PHASE 5 CASE STUDY — CLUSTER REPORT",
        BAR_THICK,
        f"model:           {model}",
        f"prompt:          {primary['prompt_key']}",
        f"trajectory_id:   {primary['id']}",
        f"layers:          L{primary['start_layer']} → L{primary['end_layer']}"
        f"  (lifespan={primary['lifespan']})",
        "merge_event:     " + (
            f"L{merge['layer_from']} → L{merge['layer_to']} "
            f"(prev_ids={merge['prev_ids']}, curr_id={merge['curr_id']})"
            if merge else "none"
        ),
        f"sibling_id:      {primary.get('sibling_id')}",
        f"total_score:     {_fmt(primary.get('total_score'), 3)}",
    ]

    sub = primary.get("sub_scores", {})
    if sub:
        parts = "  ".join(
            f"{k}={_fmt(sub.get(k), 2)}"
            for k in ("lifespan", "merge", "semantic",
                      "preferred_prompt", "size", "sibling")
            if k in sub
        )
        lines.append(f"sub_scores:      {parts}")

    if runner:
        lines.append(
            f"runner_up:       id={runner['id']} "
            f"prompt={runner['prompt_key']} "
            f"score={_fmt(runner.get('total_score'), 3)}"
        )
    if sibling and "note" in sibling:
        lines.append(f"sibling_note:    {sibling['note']}")

    return lines


# ---------------------------------------------------------------------------
# Top-level verdicts block
# ---------------------------------------------------------------------------

def _render_verdicts(
    profile, v_align, c1, c2, d_feat, e_lens, f_cause, g_sib
) -> list:
    """One-line verdict per group. Leads the report so an LLM gets the
    conclusions before the supporting numbers."""

    def _vline(label, value):
        return f"  {label:<10s} {value}"

    def _miss(group):
        return f"(group {group} not run)"

    lines = [BAR_THICK, "VERDICTS", BAR_THICK]

    # A
    if profile is None:
        lines.append(_vline("A", _miss("A")))
    elif "_error" in profile:
        lines.append(_vline("A", f"ERROR — {profile['_error']}"))
    else:
        s = profile.get("summary", {})
        lines.append(_vline(
            "A",
            f"mean_ip={_fmt(s.get('mean_ip_mean'), 3)}  "
            f"mean_jaccard={_fmt(s.get('mean_jaccard_stability'), 3)}  "
            f"mean_silhouette_sib={_fmt(s.get('mean_silhouette_sib'), 3)}",
        ))

    # B
    if v_align is None:
        lines.append(_vline("B", _miss("B")))
    elif "_error" in v_align:
        lines.append(_vline("B", f"ERROR — {v_align['_error']}"))
    else:
        s = v_align.get("summary", {})
        mg = v_align.get("merge_geometry") or {}
        lines.append(_vline(
            "B",
            f"mean_centroid_attr_frac={_fmt(s.get('mean_centroid_attr_frac'), 3)}  "
            f"mean_centroid_rep_frac={_fmt(s.get('mean_centroid_rep_frac'), 3)}  "
            f"rot_neutral_local={_fmt(s.get('rotational_neutral_local'))}  "
            f"merge_verdict={_fmt(mg.get('verdict'))}",
        ))

    # C1
    if c1 is None:
        lines.append(_vline("C1", _miss("C1")))
    elif "_error" in c1:
        lines.append(_vline("C1", f"ERROR — {c1['_error']}"))
    else:
        top = c1.get("top_attractor_heads", [])
        top_str = ", ".join(
            f"h{h['head']:02d}({_fmt(h.get('cohesion'), 3)})"
            for h in top[:4]
        )
        lines.append(_vline("C1", f"top_attractor_heads: {top_str or 'none'}"))

    # C2
    if c2 is None:
        lines.append(_vline("C2", _miss("C2")))
    elif "_error" in c2:
        lines.append(_vline("C2", f"ERROR — {c2['_error']}"))
    else:
        s = c2.get("summary", {})
        lines.append(_vline(
            "C2",
            f"ffn_cohesion_verdict={_fmt(s.get('ffn_cohesion_verdict'))}  "
            f"mean_ffn_coh={_fmt(s.get('mean_ffn_cohesion'), 3)}  "
            f"mean_attn_coh={_fmt(s.get('mean_attn_cohesion'), 3)}",
        ))

    # D
    if d_feat is None:
        lines.append(_vline("D", _miss("D")))
    elif "_error" in d_feat:
        lines.append(_vline("D", f"ERROR — {d_feat['_error']}"))
    else:
        lv = d_feat.get("lda_v_alignment", {})
        md = d_feat.get("merge_dynamics") or {}
        lines.append(_vline(
            "D",
            f"lda_v_verdict={_fmt(lv.get('verdict'))}  "
            f"merge_died={_fmt(md.get('n_died'))}  "
            f"merge_born={_fmt(md.get('n_born'))}",
        ))

    # E
    if e_lens is None:
        lines.append(_vline("E", _miss("E")))
    elif "_error" in e_lens or "error" in e_lens:
        lines.append(_vline("E", f"ERROR — {e_lens.get('_error') or e_lens.get('error')}"))
    else:
        # Derive a rough stability summary from the stability list
        stab = e_lens.get("stability", [])
        top1_matches = [s["top1_match"] for s in stab if "top1_match" in s]
        top1_rate = (
            f"{sum(top1_matches)}/{len(top1_matches)}"
            if top1_matches else "n/a"
        )
        lines.append(_vline(
            "E",
            f"top1_stable={top1_rate}  "
            f"tuned_lens={_fmt(e_lens.get('used_tuned_lens'))}",
        ))

    # F
    if f_cause is None:
        lines.append(_vline("F", _miss("F")))
    elif "_error" in f_cause:
        lines.append(_vline("F", f"ERROR — {f_cause['_error']}"))
    else:
        ivs = f_cause.get("interventions", {})
        iv_names = list(ivs.keys())
        lines.append(_vline(
            "F",
            f"target_layer={_fmt(f_cause.get('target_layer'))}  "
            f"interventions_run={iv_names or 'none'}",
        ))

    # G
    if g_sib is None:
        lines.append(_vline("G", _miss("G")))
    elif "_error" in g_sib:
        lines.append(_vline("G", f"ERROR — {g_sib['_error']}"))
    else:
        cs = g_sib.get("contrast_summary", {})
        lines.append(_vline(
            "G",
            f"sibling_mean_ip={_fmt(cs.get('sibling_mean_ip'), 3)}  "
            f"random_mean_ip={_fmt(cs.get('random_mean_ip'), 3)}  "
            f"sibling_sil={_fmt(cs.get('sibling_mean_silhouette_all'), 3)}",
        ))

    return lines


# ---------------------------------------------------------------------------
# Per-group summary sections (no per-layer loops)
# ---------------------------------------------------------------------------

def _render_group_A(profile: Optional[dict]) -> list:
    lines = _section("A. STRUCTURAL PROFILE")
    if profile is None:
        return lines + ["  (group A: no data)"]
    if "_error" in profile:
        return lines + [f"  ERROR: {profile['_error']}"]

    s = profile.get("summary", {})
    for k in ("mean_size", "mean_ip_mean", "mean_radius",
              "mean_silhouette_sib", "mean_silhouette_all",
              "mean_angular_step", "total_arc_length",
              "mean_jaccard_stability", "mean_cka_restricted",
              "mean_mass_near_1_frac"):
        if k in s:
            lines.append(_kv(k, s[k]))

    n_layers = len(profile.get("per_layer", []))
    lines.append(_kv("n_layers_in_trajectory", n_layers))
    return lines


def _render_group_B(v: Optional[dict]) -> list:
    lines = _section("B. PAPER-THEORETICAL ALIGNMENT")
    if v is None:
        return lines + ["  (group B: no data)"]
    if "_error" in v:
        return lines + [f"  ERROR: {v['_error']}"]

    s = v.get("summary", {})
    for k in ("mean_centroid_attr_frac", "mean_centroid_rep_frac",
              "mean_beta_estimate", "mean_cluster_energy",
              "rotational_neutral_local"):
        if k in s:
            lines.append(_kv(k, s[k]))

    # S/A local test verdict
    sa = v.get("sa_local_test", {})
    if sa.get("available"):
        lines.append(_kv("sa_mean_asym_frac", sa.get("mean_asym_frac")))
        lines.append(_kv("sa_verdict", sa.get("verdict")))
    else:
        lines.append("  sa_local_test: (phase2i artifacts unavailable)")

    # Merge geometry
    mg = v.get("merge_geometry")
    if mg:
        lines += ["", "  merge geometry:"]
        for k in ("pre_merge_cosine", "pre_merge_angle_rad",
                  "fusion_attr_alignment", "fusion_rep_alignment", "verdict"):
            if k in mg:
                lines.append(_kv(f"  {k}", mg[k]))
    else:
        lines.append("  merge geometry: none")

    # Top Schur blocks (condensed — just top 3)
    blocks = v.get("schur_blocks", [])
    if blocks:
        lines += ["", "  top schur blocks (by centroid overlap):"]
        for b in blocks[:3]:
            lines.append(
                f"    idx={b['start_idx']:03d}: "
                f"overlap_c={_fmt(b.get('overlap_c'))}  "
                f"eig=({_fmt(b.get('eig_real'), 3)}"
                f"{'+' if (b.get('eig_imag', 0) or 0) >= 0 else ''}"
                f"{_fmt(b.get('eig_imag'), 3)}i)"
            )

    return lines


def _render_group_C1(c1: Optional[dict]) -> list:
    lines = _section("C.1 PER-HEAD ATTENTION CONTRIBUTIONS")
    if c1 is None:
        return lines + ["  (group C.1: no data)"]
    if "_error" in c1:
        return lines + [f"  ERROR: {c1['_error']}"]

    top = c1.get("top_attractor_heads", [])
    if top:
        lines += ["", "  top attractor heads (cumulative cohesion):"]
        for h in top[:8]:
            lines.append(
                f"    head {h['head']:02d}: "
                f"cohesion={_fmt(h.get('cohesion'))}  "
                f"ov_top_eigval={_fmt(h.get('ov_top_eigval'), 3)}"
            )
    else:
        lines.append("  no attractor heads identified")

    # Cumulative cohesion trace (single line)
    cc = c1.get("cumulative_cohesion", [])
    if cc:
        lines.append(
            "  cumulative_cohesion trace: "
            + "  ".join(_fmt(x, 3) for x in cc)
        )

    return lines


def _render_group_C2(c2: Optional[dict]) -> list:
    lines = _section("C.2 FFN CONTRIBUTIONS")
    if c2 is None:
        return lines + ["  (group C.2: no data — phase2 decompose artifacts missing?)"]
    if "_error" in c2:
        return lines + [f"  ERROR: {c2['_error']}"]

    s = c2.get("summary", {})
    for k in ("mean_ffn_cohesion", "mean_attn_cohesion",
              "mean_ffn_lda_frac", "ffn_cohesion_verdict"):
        if k in s:
            lines.append(_kv(k, s[k]))
    return lines


def _render_group_D(d: Optional[dict]) -> list:
    lines = _section("D. FEATURE SIGNATURES")
    if d is None:
        return lines + ["  (group D: no data)"]
    if "_error" in d:
        return lines + [f"  ERROR: {d['_error']}"]

    # LDA stability + V alignment
    ls = d.get("lda_stability", {})
    lv = d.get("lda_v_alignment", {})
    if ls:
        lines.append(_kv("lda_mean_stability", ls.get("mean_stability")))
        lines.append(_kv("lda_min_stability",  ls.get("min_stability")))
    if lv.get("available"):
        lines.append(_kv("lda_layer_used",    lv.get("lda_layer_used")))
        lines.append(_kv("lda_attr_alignment", lv.get("attr_alignment")))
        lines.append(_kv("lda_rep_alignment",  lv.get("rep_alignment")))
        lines.append(_kv("lda_v_verdict",      lv.get("verdict")))

    # Merge dynamics summary
    md = d.get("merge_dynamics")
    if md:
        lines += ["", "  merge-event feature dynamics:"]
        lines.append(_kv("  n_died",     md.get("n_died")))
        lines.append(_kv("  n_born",     md.get("n_born")))
        lines.append(_kv("  n_survived", md.get("n_survived")))
        if md.get("died"):
            lines.append(f"  died (examples):     {md['died'][:8]}")
        if md.get("born"):
            lines.append(f"  born (examples):     {md['born'][:8]}")
    else:
        lines.append("  merge dynamics: (no merge event or feature acts missing)")

    # Decoder geometry — top 5 features (condensed)
    geo = d.get("decoder_geometry", [])
    if geo:
        lines += ["", "  top decoder-direction alignments (feature, cos_centroid, cos_lda):"]
        for g in geo[:5]:
            lines.append(
                f"    #{g['feature']:04d}: "
                f"cos_c={_fmt(g.get('cos_centroid'))}  "
                f"cos_lda={_fmt(g.get('cos_lda'))}  "
                f"attr={_fmt(g.get('attr_alignment'))}  "
                f"rep={_fmt(g.get('rep_alignment'))}"
            )

    # Chorus summary (peak layer)
    chorus_peak = None
    for p in d.get("per_layer", []):
        c = p.get("chorus", {})
        n = c.get("n_components", 0)
        if n and (chorus_peak is None or n > chorus_peak["n"]):
            chorus_peak = {"layer": p["layer"], "n": n,
                           "max_jac": c.get("max_jaccard")}
    if chorus_peak:
        lines.append(
            f"  chorus peak: L{chorus_peak['layer']:02d}  "
            f"n_components={chorus_peak['n']}  "
            f"max_jaccard={_fmt(chorus_peak.get('max_jac'), 3)}"
        )

    return lines


def _render_group_E(e: Optional[dict]) -> list:
    lines = _section("E. TUNED-LENS DECODING")
    if e is None:
        return lines + ["  (group E: no data)"]
    if "_error" in e or "error" in e:
        return lines + [f"  ERROR: {e.get('_error') or e.get('error')}"]

    lines.append(_kv("used_tuned_lens", e.get("used_tuned_lens")))

    # Top-1 stability rate
    stab = e.get("stability", [])
    if stab:
        top1 = [s["top1_match"] for s in stab if "top1_match" in s]
        top5 = [s["top5_overlap"] for s in stab if "top5_overlap" in s]
        lines.append(_kv("top1_stable_transitions",
                         f"{sum(top1)}/{len(top1)}" if top1 else "n/a"))
        lines.append(_kv("mean_top5_overlap",
                         _fmt(float(np.mean(top5)), 3) if top5 else "n/a"))

    # First and last layer centroid decoding
    per = e.get("per_layer", [])
    if per:
        def _layer_summary(p):
            top = p.get("top_centroid", [])
            toks = ", ".join(
                f"{t['token']!s}({_fmt(t['prob'], 3)})" for t in top[:5]
            )
            return f"L{p['layer']:02d} H={_fmt(p.get('entropy'), 3)}: {toks}"
        lines += ["", "  first layer: " + _layer_summary(per[0])]
        if len(per) > 1:
            lines.append("  last layer:  " + _layer_summary(per[-1]))

    return lines


def _render_group_F(f: Optional[dict]) -> list:
    lines = _section("F. CAUSAL TESTS")
    if f is None:
        return lines + ["  (group F: no data)"]
    if "_error" in f:
        return lines + [f"  ERROR: {f['_error']}"]

    lines.append(_kv("target_layer", f.get("target_layer")))

    ivs = f.get("interventions", {})
    for name, iv in ivs.items():
        lines += ["", f"  intervention: {name}"]
        for k in ("head", "target_layer", "alpha", "patched_token_idx"):
            if k in iv:
                lines.append(_kv(f"    {k}", iv[k]))
        # Summarise recluster: mean frac_together
        rc = iv.get("recluster", {}).get("per_layer", [])
        if rc:
            ft = [r["frac_together"] for r in rc
                  if r.get("frac_together") is not None]
            lines.append(
                _kv("    mean_frac_together",
                    _fmt(float(np.mean(ft)), 3) if ft else "n/a")
            )

    return lines


def _render_group_G(g: Optional[dict]) -> list:
    lines = _section("G. SIBLING + RANDOM CONTROL")
    if g is None:
        return lines + ["  (group G: no data)"]
    if "_error" in g:
        return lines + [f"  ERROR: {g['_error']}"]

    cs = g.get("contrast_summary", {})
    for k in ("sibling_mean_ip", "random_mean_ip",
              "sibling_mean_silhouette_all", "random_mean_silhouette_all"):
        if k in cs:
            lines.append(_kv(k, cs[k]))

    sp = g.get("sibling_profile", {}).get("summary", {})
    if sp:
        lines += ["", "  sibling profile:"]
        for k in ("mean_ip_mean", "mean_silhouette_all",
                  "mean_jaccard_stability", "mean_cka_restricted"):
            if k in sp:
                lines.append(_kv(f"  {k}", sp[k]))

    rp = g.get("random_control_profile", {}).get("summary", {})
    if rp:
        lines += ["", "  random-control profile:"]
        for k in ("mean_ip_mean", "mean_silhouette_all",
                  "mean_jaccard_stability", "mean_cka_restricted"):
            if k in rp:
                lines.append(_kv(f"  {k}", rp[k]))

    sh = g.get("sibling_heads", {})
    if sh and "top_attractor_heads" in sh:
        lines += ["", "  sibling top attractor heads:"]
        for h in sh["top_attractor_heads"][:4]:
            lines.append(
                f"    head {h['head']:02d}: cohesion={_fmt(h.get('cohesion'))}"
            )

    return lines


def _render_ranked_appendix(selection: dict, top_n: int = 10) -> list:
    lines = _section("APPENDIX: RANKED TRAJECTORY LIST")
    lines.append(
        f"  {'rank':<4} {'prompt':<20} {'id':<5} "
        f"{'life':<5} {'score':<7} sub_scores (life/merge/sem/prompt/size/sib)"
    )
    for rank, c in enumerate(selection.get("ranked", [])[:top_n]):
        sub = c.get("sub_scores", {})
        subs = "/".join(
            f"{sub.get(k, 0):.2f}"
            for k in ("lifespan", "merge", "semantic",
                      "preferred_prompt", "size", "sibling")
        )
        lines.append(
            f"  {rank:<4} {c['prompt_key'][:20]:<20} {c['id']:<5} "
            f"{c['lifespan']:<5} {c.get('total_score', 0):<7.3f} {subs}"
        )
    return lines


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_report(run_dir: Path, model: str, tag: str = "primary") -> str:
    run_dir = Path(run_dir)

    selection = _load_json(run_dir / "cluster_metadata.json") or {}
    if "primary" not in selection:
        return (BAR_THICK + "\nERROR: cluster_metadata.json missing or invalid\n"
                + BAR_THICK)

    profile  = _load_json(run_dir / f"group_A_profile_{tag}.json")
    v_align  = _load_json(run_dir / f"group_B_v_alignment_{tag}.json")
    c1       = _load_json(run_dir / f"group_C1_heads_{tag}.json")
    c2       = _load_json(run_dir / f"group_C2_ffn_{tag}.json")
    d_feat   = _load_json(run_dir / f"group_D_features_{tag}.json")
    e_lens   = _load_json(run_dir / f"group_E_tuned_lens_{tag}.json")
    f_cause  = _load_json(run_dir / "group_F_causal.json")
    g_sib    = _load_json(run_dir / "group_G_sibling_contrast.json")

    all_lines: list[str] = []
    all_lines += _render_header(selection, model)
    all_lines += [""]
    all_lines += _render_verdicts(
        profile, v_align, c1, c2, d_feat, e_lens, f_cause, g_sib
    )
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
