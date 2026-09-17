"""What IS the direction MLP 6 rotates into when `L5H2` is ablated?

WHY THIS EXISTS
---------------
`PROJECT.md` §3.23 established that `L7H8`'s matching, with `L5H2` gone,
depends on a specific direction in MLP 6's output: writing MLP 6's post-ablation
mean `mu_cond` into its slot restores the matcher (attention 0.643 / 0.567 at
steps 16000 / 143000) while its CLEAN-state mean `mu_clean` restores nothing
(0.038 / 0.016, on a par with zero and with a norm-matched random constant).
The rotation between them is only `cos = 0.837`, so the entire effect lives in
the component of `mu_cond` orthogonal to what MLP 6 was already writing. §3.23
closed by naming the decode as the obvious next step. This is it.

THREE READS, each with `mu_clean` and random directions as controls, because
the question is always "more than what?".

1. **The LayerNorm guard, first because it could invalidate everything.**
   GPT-NeoX LayerNorm subtracts the mean across the hidden dimension, so any
   component of `mu` along the uniform vector `1/sqrt(d)` is deleted before
   `L7H8` reads anything. If `mu_cond` were mostly uniform it could not matter,
   and the measured effect would have to be something else. Reported as
   `cos(mu, uniform)` and the norm share surviving mean-subtraction.

2. **Into `L7H8`'s read-space.** The sharp one. `W_Q` and `W_K` for that head
   are 64x1024, so their rowspaces are 64 of 1024 dimensions and a random
   direction lands `64/1024 = 0.0625` of its squared norm there. If the
   rotation is aimed at the matcher, `mu_cond` should sit further into that
   rowspace than `mu_clean` does. **The read is taken through the layer's own
   LayerNorm gain**: `W_K` acts on `diag(gamma) (I - uu^T) x`, so the vector is
   pushed through mean-subtraction and the elementwise gain before projecting.
   The per-position `1/std` is a positive scalar and changes no fraction or
   cosine, so it is dropped -- that is the one approximation here and it is
   named rather than buried.

3. **The logit lens**, for interpretability rather than mechanism: `W_U` applied
   to the direction (through the final LayerNorm gain, same approximation) and
   the top promoted and suppressed tokens. A caveat that matters: `mu` is a
   direction, not a residual state, so this reads what the direction *pushes
   toward*, not what the model predicts. And `mu` is position-independent, so
   whatever it does for the matcher it cannot be carrying per-token match
   information (§3.23 already said this from the other side).

Weights plus two forward passes; no p-value; nothing registered; pythia-410m
spent under `check_registry` rule 3.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))
_want = str(REPO / ".venv")
# Script-time only: modules are importable by tests on runners that are not
# this machine\'s .venv; a real run still refuses the wrong interpreter.
if __name__ == "__main__" and not sys.prefix.startswith(_want):
    raise SystemExit(f"wrong interpreter: {sys.prefix!r}, need {_want!r}")

import numpy as np
import torch

from core.lm_loading import load_causal_lm
from tools.run.induction_rank_sweep import PROBE_ARMS, EVAL_SEED, ablated
from p7d_redundancy.member_formation_curves import batch
from p7d_redundancy.mlp_backup_check import mlp_output_means

OUT = DATA / "analysis" / "mlp6_decode_direction.json"


def parse_head(s):
    return (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))


def rowspace_fraction(W, v):
    """Share of `v`'s squared norm lying in the rowspace of `W` (r x d)."""
    # Right singular vectors spanning the rowspace.
    _, S, Vt = torch.linalg.svd(W, full_matrices=False)
    keep = Vt[S > S.max() * 1e-10]
    return float((keep @ v).pow(2).sum() / max(float(v.pow(2).sum()), 1e-30))


def through_ln(v, gamma):
    """Mean-subtract across the hidden dim, then apply the LayerNorm gain.

    The per-position 1/std is a positive scalar: it rescales but changes no
    cosine and no norm FRACTION, so it is omitted.
    """
    return gamma * (v - v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--steps", default="16000,143000")
    ap.add_argument("--source", default="L5H2")
    ap.add_argument("--target", default="L7H8")
    ap.add_argument("--mlp", type=int, default=6)
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--controls", type=int, default=200,
                     help="random directions for the rowspace null")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--probe", default="wide", choices=tuple(PROBE_ARMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    source, target = parse_head(args.source), parse_head(args.target)
    out = Path(args.out) if args.out else OUT
    git_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    steps = [int(x) for x in args.steps.split(",") if x]

    res = {"_what_this_is": __doc__, "git_sha": git_sha, "model": args.model,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
           "source": args.source, "target": args.target, "mlp": args.mlp,
           "probe": args.probe, "n_seqs": args.seqs, "steps": steps,
           "per_step": {}}

    for s in steps:
        model, tok = load_causal_lm(f"{args.model}-step{s}")
        model.eval()
        d_model = model.config.hidden_size
        n_heads = model.config.num_attention_heads
        d_head = d_model // n_heads
        tgt_layer, tgt_head = target

        ids = batch(np.random.default_rng(EVAL_SEED), args.seqs,
                    *PROBE_ARMS[args.probe])
        means_clean = mlp_output_means(model, ids, args.chunk)
        with ablated(model, [source], mode="ov"):
            means_cond = mlp_output_means(model, ids, args.chunk)
        mu_clean = means_clean[args.mlp].double()
        mu_cond = means_cond[args.mlp].double()
        # The component the rotation added -- §3.23 showed this is where the
        # whole effect lives, so it is decoded in its own right.
        delta = mu_cond - mu_clean * (torch.dot(mu_cond, mu_clean)
                                      / torch.dot(mu_clean, mu_clean))

        gamma = model.gpt_neox.layers[tgt_layer].input_layernorm.weight.detach().double()
        qkv = model.gpt_neox.layers[tgt_layer].attention.query_key_value.weight
        q3 = qkv.detach().double().reshape(n_heads, 3 * d_head, d_model)
        W_Q = q3[tgt_head, 0:d_head, :]
        W_K = q3[tgt_head, d_head:2 * d_head, :]

        uniform = torch.ones(d_model, dtype=torch.float64) / np.sqrt(d_model)
        rng = torch.Generator().manual_seed(EVAL_SEED)

        vecs = {"mu_cond": mu_cond, "mu_clean": mu_clean,
                "rotation_component": delta}
        rows = {}
        for name, v in vecs.items():
            vln = through_ln(v, gamma)
            rows[name] = {
                "norm": float(v.norm()),
                "cos_uniform": float(torch.dot(v, uniform) / max(float(v.norm()), 1e-30)),
                "share_surviving_mean_subtraction": float(
                    (v - v.mean()).pow(2).sum() / max(float(v.pow(2).sum()), 1e-30)),
                "rowspace_frac_K": rowspace_fraction(W_K, vln),
                "rowspace_frac_Q": rowspace_fraction(W_Q, vln)}

        # Null for the rowspace fractions: random directions, same treatment.
        nullK, nullQ = [], []
        for _ in range(args.controls):
            r = torch.randn(d_model, generator=rng, dtype=torch.float64)
            rln = through_ln(r, gamma)
            nullK.append(rowspace_fraction(W_K, rln))
            nullQ.append(rowspace_fraction(W_Q, rln))
        nullK, nullQ = np.array(nullK), np.array(nullQ)

        # Logit lens.
        gamma_f = model.gpt_neox.final_layer_norm.weight.detach().double()
        W_U = model.embed_out.weight.detach().double()
        lens = {}
        for name in ("mu_cond", "rotation_component"):
            v = through_ln(vecs[name], gamma_f)
            logits = W_U @ v
            top = torch.topk(logits, args.top_k)
            bot = torch.topk(-logits, args.top_k)
            lens[name] = {
                "top": [[tok.decode([int(i)]), float(x)]
                        for i, x in zip(top.indices, top.values)],
                "bottom": [[tok.decode([int(i)]), -float(x)]
                           for i, x in zip(bot.indices, bot.values)]}

        # SPECIFICITY: is the rotation aimed at THIS head's key space, or at
        # attention machinery generally? Every head that reads a residual the
        # MLP can reach (layers > the MLP) is scored the same way, and the
        # target's rank among them is the answer. Without this the 7-8x result
        # above is consistent with a direction aimed at no head in particular.
        spec = []
        for L in range(args.mlp + 1, model.config.num_hidden_layers):
            g = model.gpt_neox.layers[L].input_layernorm.weight.detach().double()
            q3L = (model.gpt_neox.layers[L].attention.query_key_value.weight
                   .detach().double().reshape(n_heads, 3 * d_head, d_model))
            dln = through_ln(delta, g)
            for h in range(n_heads):
                spec.append({
                    "head": f"L{L}H{h}",
                    "frac_K": rowspace_fraction(q3L[h, d_head:2 * d_head, :], dln),
                    "frac_Q": rowspace_fraction(q3L[h, 0:d_head, :], dln)})
        spec.sort(key=lambda r: -r["frac_K"])
        tgt_tag = f"L{tgt_layer}H{tgt_head}"
        tgt_rank = next(i for i, r in enumerate(spec) if r["head"] == tgt_tag)
        fk = np.array([r["frac_K"] for r in spec])

        res["per_step"][str(s)] = {
            "specificity": {
                "n_heads_scored": len(spec),
                "target": tgt_tag, "target_rank_by_frac_K": int(tgt_rank),
                "target_frac_K": float(fk[tgt_rank]),
                "population_median": float(np.median(fk)),
                "population_p99": float(np.percentile(fk, 99)),
                "top10": spec[:10],
                # Full list kept: the top-10 view cannot separate "aimed at the
                # redundancy set" from "aimed at whatever is in the next layer",
                # and the layer profile is what decides that.
                "all_heads": spec},
            "vectors": rows,
            "rowspace_null": {
                "n": int(args.controls),
                "K_mean": float(nullK.mean()), "K_sd": float(nullK.std()),
                "K_max": float(nullK.max()),
                "Q_mean": float(nullQ.mean()), "Q_sd": float(nullQ.std()),
                "Q_max": float(nullQ.max()),
                "expected_rank_over_d": d_head / d_model},
            "logit_lens": lens,
            "cos_mu_clean_cond": float(
                torch.dot(mu_clean, mu_cond)
                / max(float(mu_clean.norm() * mu_cond.norm()), 1e-30))}

        print(f"== step {s}")
        print(f"  cos(mu_clean, mu_cond) = "
              f"{res['per_step'][str(s)]['cos_mu_clean_cond']:.4f}")
        print(f"  {'vector':>20} {'norm':>8} {'cos_unif':>9} "
              f"{'survives_LN':>12} {'frac in K':>10} {'frac in Q':>10}")
        for name, r in rows.items():
            print(f"  {name:>20} {r['norm']:>8.3f} {r['cos_uniform']:>+9.4f} "
                  f"{r['share_surviving_mean_subtraction']:>12.4f} "
                  f"{r['rowspace_frac_K']:>10.4f} {r['rowspace_frac_Q']:>10.4f}")
        print(f"  random null (n={args.controls}): K {nullK.mean():.4f} "
              f"+- {nullK.std():.4f} (max {nullK.max():.4f})   "
              f"Q {nullQ.mean():.4f} +- {nullQ.std():.4f} "
              f"(max {nullQ.max():.4f})   expected {d_head / d_model:.4f}")
        sp = res["per_step"][str(s)]["specificity"]
        print(f"  specificity of the rotation across {sp['n_heads_scored']} "
              f"downstream heads: {tgt_tag} frac_K {sp['target_frac_K']:.4f}, "
              f"**rank {sp['target_rank_by_frac_K']}**, "
              f"median {sp['population_median']:.4f}, "
              f"p99 {sp['population_p99']:.4f}")
        print("     top: " + ", ".join(
            f"{r['head']} {r['frac_K']:.3f}" for r in sp["top10"][:6]))
        for name, L in lens.items():
            print(f"  logit lens, {name}:")
            print("     promotes: " + ", ".join(f"{t!r}" for t, _ in L["top"][:8]))
            print("     suppresses: " + ", ".join(f"{t!r}" for t, _ in L["bottom"][:8]))
        print()

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        del model

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
