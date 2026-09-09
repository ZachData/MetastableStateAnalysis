"""Tier B of the dissipation-identity run (docs/dissipation_checkpoint_axis_scoping.md).

Adds what Tier A (`tools/run/dissipation.py`) could not: the exact
attention-vs-FFN split of the first-order energy change, and a per-head
roll-up on the 116 P-I1 forming heads. Needs one forward pass per
(step, prompt) with sublayer capture -- 19 x 7 = 133 passes, CPU.

For Pythia's parallel residual the block is  x_out = x + attn(ln1(x)) + mlp(ln2(x)),
so with x_in the block input and the two module outputs a, f (pre-residual):

    dX_attn = a           dX_ffn = f           dX_total = a + f   (exact)

and per head h, splitting the attention output projection `dense` (Linear d->d)
by its input columns:

    dX_attn[h] = ctx[:, h*hd:(h+1)*hd] @ dense.weight[:, h*hd:(h+1)*hd].T
    sum_h dX_attn[h] + dense.bias = dX_attn           (asserted per layer)

Everything then goes through core.dissipation unchanged. All 24 layers are
kept -- layer 0 and layer 23 are carried, not dropped (author's call
2026-09-06: fine for them to have unexplained behaviour in analysis, not
fine to exclude them from the measurement).

PATHS DERIVED: METS_REPO / METS_DATA override. METS_DISS_BETA (default 1.0).
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

_want_prefix = str(REPO / ".venv")
if not sys.prefix.startswith(_want_prefix):
    raise SystemExit(f"wrong interpreter: sys.prefix={sys.prefix!r}, need {_want_prefix!r}")

import numpy as np
import scipy
import torch

from core.changepoint_colocation import REGISTERED_P_I1_SWEEP
from core.config import PROMPTS
from core.dissipation import (
    dissipation,
    dissipation_by_channel,
    dissipation_by_subspace,
    gradient_flow_alignment,
)
from core.lm_loading import load_causal_lm
from core.model_family import model_family
from core.sublayer_streams import (
    blocks_of, attn_module, ffn_module, uses_parallel_residual, _tensor_of,
)

STEPS = list(REGISTERED_P_I1_SWEEP)
BETA = float(os.environ.get("METS_DISS_BETA", "1.0"))
SCORED_PROMPTS = [
    "camus_letranger", "hdbscan_code", "homer_iliad", "latex_monograph",
    "paper_excerpt", "sullivan_ballou", "wiki_paragraph",
]
CARRIED_BESIDE = "repeated_tokens"
N_BLOCKS = 24
MAX_LEN = 512


def _projector_npz(step: int) -> Path:
    hits = list(DATA.glob(f"phase12/p2_eigenspectra_*/ov_projectors_pythia-410m-step{step}.npz"))
    if len(hits) != 1:
        raise SystemExit(f"step {step}: {len(hits)} ov_projectors npz, need 1")
    return hits[0]


def _phase1_ntok(step: int, prompt: str) -> int:
    hits = [
        p for p in DATA.glob(f"phase12/*/pythia-410m-step{step}_{prompt}/activations.npz")
        if not p.parent.parent.name.startswith("p2_eigenspectra_")
    ]
    if len(hits) != 1:
        raise SystemExit(f"step {step} {prompt!r}: {len(hits)} Phase 1 activations.npz, need 1")
    with np.load(hits[0], allow_pickle=False) as z:
        return int(z["activations"].shape[1])


def _capture(model, tokenizer, family, blocks, text: str):
    """One forward pass. Returns per-layer x_in, attn_delta, ffn_delta, and the
    per-head attention deltas (n_blocks x n_heads x n x d)."""
    x_in, a_out, f_out, ctx_in = [], [], [], []
    handles = []
    for blk in blocks:
        am, fm = attn_module(blk, family), ffn_module(blk, family)
        handles.append(blk.register_forward_pre_hook(lambda m, i, s=x_in: s.append(_tensor_of(i))))
        handles.append(am.register_forward_hook(lambda m, i, o, s=a_out: s.append(_tensor_of(o))))
        handles.append(fm.register_forward_hook(lambda m, i, o, s=f_out: s.append(_tensor_of(o))))
        handles.append(am.dense.register_forward_pre_hook(
            lambda m, i, s=ctx_in: s.append(_tensor_of(i))))
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            model(**{k: v.to(next(model.parameters()).device) for k, v in enc.items()})
    finally:
        for h in handles:
            h.remove()
    for name, seq in (("x_in", x_in), ("a_out", a_out), ("f_out", f_out), ("ctx_in", ctx_in)):
        if len(seq) != N_BLOCKS:
            raise RuntimeError(f"{name}: captured {len(seq)} != {N_BLOCKS} blocks")

    n_heads = model.config.num_attention_heads
    hd = model.config.hidden_size // n_heads
    per_head = []                                   # [n_blocks][n_heads] -> (n,d)
    for l, blk in enumerate(blocks):
        W = attn_module(blk, family).dense.weight.detach().to(torch.float64).cpu().numpy()  # (d,d)
        b = attn_module(blk, family).dense.bias
        b = None if b is None else b.detach().to(torch.float64).cpu().numpy()
        ctx = ctx_in[l].to(torch.float64).numpy()   # (n, d)
        heads_l = []
        acc = np.zeros_like(ctx)
        for h in range(n_heads):
            sl = slice(h * hd, (h + 1) * hd)
            dh = ctx[:, sl] @ W[:, sl].T             # (n, d)
            heads_l.append(dh)
            acc += dh
        if b is not None:
            acc += b
        rel = np.abs(acc - a_out[l].to(torch.float64).numpy()).max() / max(
            1e-12, np.abs(a_out[l].to(torch.float64).numpy()).max())
        if rel > 1e-5:
            raise RuntimeError(f"layer {l}: per-head sum vs attn delta rel err {rel:.2e}")
        per_head.append(heads_l)
    return (
        [t.to(torch.float64).numpy() for t in x_in],
        [t.to(torch.float64).numpy() for t in a_out],
        [t.to(torch.float64).numpy() for t in f_out],
        per_head,
    )


def _gfa_reduce(g):
    return {k: g[k] for k in ("mean", "median", "q10", "q90", "frac_descending",
                              "n_defined", "status")}


def main() -> None:
    import subprocess
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    import transformers

    rn = json.loads((DATA / "analysis" / "relay_null_series.json").read_text())
    forming = [tuple(int(x) for x in h.split(",")) for h in rn["forming_heads"]]
    forming_by_layer = {}
    for l, h in forming:
        forming_by_layer.setdefault(l, []).append(h)
    print(f"{len(forming)} forming heads, layers {sorted(forming_by_layer)}", flush=True)

    out = {
        "_what_this_is":
            "Tier B of the dissipation-identity run. Exact attention/FFN split "
            "of the first-order energy change (core.dissipation.dissipation_by_"
            "channel) per (step, prompt, layer), plus a per-head roll-up of the "
            "attention channel on the P-I1 forming heads. One forward pass per "
            "(step, prompt). frame=l2_sphere. v2 fields "
            "(v2_attn_pos_*, pooled v2_attn_repulsive_share, per-head "
            "v2_pos_*): status-2.md item 5's violation-restricted split -- the "
            "repulsive share of the POSITIVE part of the attention channel's "
            "first-order term, at boundaries where actual dE > 0.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git_sha": git_sha,
        "lib_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "torch": torch.__version__,
                         "transformers": transformers.__version__},
        "frame": "l2_sphere", "beta": BETA, "steps": STEPS,
        "scored_prompts": SCORED_PROMPTS, "carried_beside": CARRIED_BESIDE,
        "n_blocks": N_BLOCKS,
        "per_step_layer": {},        # "<step>|<prompt>|<layer>" -> channel reductions
        "pooled_by_step_layer": {},  # "<step>|<layer>" -> 7-prompt pooled
        "per_head": {f"{l},{h}": {"d_attn_repulsive": [None] * len(STEPS),
                                  "d_attn_total": [None] * len(STEPS),
                                  "gfa_cos": [None] * len(STEPS),
                                  "v2_pos_first_order": [None] * len(STEPS),
                                  "v2_pos_repulsive": [None] * len(STEPS)}
                     for l, h in forming},
        "max_channel_sum_check": 0.0,
        "input_provenance": {},
    }
    prompts_all = SCORED_PROMPTS + [CARRIED_BESIDE]
    t_start = time.time()

    for si, step in enumerate(STEPS):
        model, tokenizer = load_causal_lm(f"pythia-410m-step{step}")
        family = model_family(f"pythia-410m-step{step}")
        blocks = blocks_of(model, family)
        if not uses_parallel_residual(model, blocks):
            raise SystemExit(f"step {step}: not parallel residual; channel split not exact")

        pnpz = _projector_npz(step)
        out["input_provenance"][f"projectors_step{step}"] = pnpz.parent.name
        pz = np.load(pnpz, allow_pickle=False)
        P = {l: (pz[f"schur_attract_layer_{l}"].astype(np.float64),
                 pz[f"schur_repulse_layer_{l}"].astype(np.float64)) for l in range(N_BLOCKS)}
        pz.close()

        step_t0 = time.time()
        pooled = {l: {"d_attn": 0.0, "d_ffn": 0.0, "d_attn_rep": 0.0, "d_attn_att": 0.0,
                      "d_ffn_rep": 0.0, "d_ffn_att": 0.0, "first_order": 0.0,
                      "actual_delta_E": 0.0, "n": 0,
                      "v2_fo": 0.0, "v2_rep": 0.0, "v2_att": 0.0, "v2_n": 0}
                  for l in range(N_BLOCKS)}

        for prompt in prompts_all:
            want_n = _phase1_ntok(step, prompt)
            x_in, a_out, f_out, per_head = _capture(
                model, tokenizer, family, blocks, PROMPTS[prompt])
            if x_in[0].shape[0] != want_n:
                raise SystemExit(
                    f"step {step} {prompt}: captured {x_in[0].shape[0]} tokens, "
                    f"activations.npz has {want_n} -- tokenisation mismatch")

            for l in range(N_BLOCKS):
                X, da, df = x_in[l], a_out[l], f_out[l]
                Pa, Pr = P[l]
                ch = dissipation_by_channel(X, da, df, BETA)
                out["max_channel_sum_check"] = max(out["max_channel_sum_check"],
                                                   float(ch["sum_check"]))
                sub_a = dissipation_by_subspace(X, da, BETA, Pa, Pr)
                sub_f = dissipation_by_subspace(X, df, BETA, Pa, Pr)
                tot = dissipation(X, da + df, BETA)
                gfa_a = gradient_flow_alignment(X, da, BETA)
                gfa_f = gradient_flow_alignment(X, df, BETA)

                # v2 (status-2.md item 5): the violation-restricted split.
                # Restrict to the particles whose attention-channel first-order
                # contribution is POSITIVE -- the part of the term that pushes
                # E_beta up -- and split THAT by OV subspace. Gate on
                # actual_delta_E > 0 at analysis time (stored below).
                _ppr = np.asarray(sub_a["per_particle_repulsive"])
                _ppa = np.asarray(sub_a["per_particle_attractive"])
                _m = (_ppr + _ppa) > 0
                v2_fo = float((_ppr + _ppa)[_m].sum())
                v2_rep = float(_ppr[_m].sum())
                v2_att = float(_ppa[_m].sum())

                key = f"{step}|{prompt}|{l}"
                out["per_step_layer"][key] = {
                    "d_attn": ch["attn"], "d_ffn": ch["ffn"],
                    "attn_share": ch["attn_share"], "channel_sum_check": float(ch["sum_check"]),
                    "d_attn_repulsive": sub_a["repulsive"], "d_attn_attractive": sub_a["attractive"],
                    "d_ffn_repulsive": sub_f["repulsive"], "d_ffn_attractive": sub_f["attractive"],
                    "first_order": tot["first_order"], "actual_delta_E": tot["actual_delta_E"],
                    "relative_residual": tot["relative_residual"],
                    "v2_attn_pos_first_order": v2_fo,
                    "v2_attn_pos_repulsive": v2_rep,
                    "v2_attn_pos_attractive": v2_att,
                    "gfa_attn": _gfa_reduce(gfa_a), "gfa_ffn": _gfa_reduce(gfa_f),
                }

                if prompt in SCORED_PROMPTS:
                    p = pooled[l]
                    p["d_attn"] += ch["attn"]; p["d_ffn"] += ch["ffn"]
                    p["d_attn_rep"] += sub_a["repulsive"]; p["d_attn_att"] += sub_a["attractive"]
                    p["d_ffn_rep"] += sub_f["repulsive"]; p["d_ffn_att"] += sub_f["attractive"]
                    p["first_order"] += tot["first_order"]
                    p["actual_delta_E"] += (tot["actual_delta_E"] or 0.0)
                    p["n"] += 1
                    if (tot["actual_delta_E"] or 0.0) > 0:   # dE>0 gate
                        p["v2_fo"] += v2_fo
                        p["v2_rep"] += v2_rep
                        p["v2_att"] += v2_att
                        p["v2_n"] += 1

                # per-head roll-up (attention channel), scored prompts summed
                if prompt in SCORED_PROMPTS and l in forming_by_layer:
                    for h in forming_by_layer[l]:
                        dh = per_head[l][h]
                        s = dissipation_by_subspace(X, dh, BETA, Pa, Pr)
                        gh = gradient_flow_alignment(X, dh, BETA)
                        rec = out["per_head"][f"{l},{h}"]
                        rec["d_attn_repulsive"][si] = (rec["d_attn_repulsive"][si] or 0.0) + s["repulsive"]
                        rec["d_attn_total"][si] = (rec["d_attn_total"][si] or 0.0) + s["total"]
                        # v2 per head: positive-part split, restricted to the
                        # boundaries where the layer's actual dE > 0.
                        if (tot["actual_delta_E"] or 0.0) > 0:
                            _hr = np.asarray(s["per_particle_repulsive"])
                            _ha = np.asarray(s["per_particle_attractive"])
                            _hm = (_hr + _ha) > 0
                            rec["v2_pos_first_order"][si] = (rec["v2_pos_first_order"][si] or 0.0) + float((_hr + _ha)[_hm].sum())
                            rec["v2_pos_repulsive"][si] = (rec["v2_pos_repulsive"][si] or 0.0) + float(_hr[_hm].sum())
                        # gfa cos: accumulate a prompt-mean at the end; store sum + count via list trick
                        prev = rec["gfa_cos"][si]
                        rec["gfa_cos"][si] = (0.0 if prev is None else prev) + (
                            gh["mean"] if gh["status"] == "ok" else 0.0)

        # finalise per-head gfa as prompt-mean
        for l, h in forming:
            rec = out["per_head"][f"{l},{h}"]
            if rec["gfa_cos"][si] is not None:
                rec["gfa_cos"][si] = rec["gfa_cos"][si] / len(SCORED_PROMPTS)

        for l in range(N_BLOCKS):
            p = pooled[l]
            mag = abs(p["d_attn_rep"]) + abs(p["d_attn_att"])
            p["attn_repulsive_share"] = (abs(p["d_attn_rep"]) / mag) if mag > 0 else None
            magt = abs(p["d_attn"]) + abs(p["d_ffn"])
            p["attn_share_of_channels"] = (abs(p["d_attn"]) / magt) if magt > 0 else None
            # v2: repulsive share of the positive first-order term, dE>0 only
            p["v2_attn_repulsive_share"] = (p["v2_rep"] / p["v2_fo"]) if p["v2_fo"] > 0 else None
            out["pooled_by_step_layer"][f"{step}|{l}"] = p

        del model, tokenizer
        dt = time.time() - step_t0
        tot_a = sum(pooled[l]["d_attn"] for l in range(N_BLOCKS))
        tot_f = sum(pooled[l]["d_ffn"] for l in range(N_BLOCKS))
        # attn repulsive share pooled over the forming-head layers only
        fl = sorted(forming_by_layer)
        ar = sum(pooled[l]["d_attn_rep"] for l in fl)
        aa = sum(pooled[l]["d_attn_att"] for l in fl)
        share = abs(ar) / (abs(ar) + abs(aa)) if (abs(ar) + abs(aa)) > 0 else float("nan")
        print(f"step{step:<7d} {dt:6.1f}s  Sigma d_attn={tot_a:+.3e}  Sigma d_ffn={tot_f:+.3e}  "
              f"attn_repulsive_share[L{fl[0]}-{fl[-1]}]={share:.3f}", flush=True)

    out["_elapsed_seconds"] = round(time.time() - t_start, 1)
    dest = Path(os.environ.get("METS_SCRATCH", str(DATA / "analysis"))) / "dissipation_sublayer_series.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nmax channel sum_check over the run: {out['max_channel_sum_check']:.2e}")
    print(f"WROTE {dest}  ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
