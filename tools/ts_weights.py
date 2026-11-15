"""Weights-only readings for `MATH_TRAINING_SYMMETRY.md` §8 (tier 1, description; no forward pass).

WHY THIS EXISTS
---------------
§8 derives what AdamW must do to directions the loss cannot see, and how far a weight can move
(tau_1 = sum of the rate, tau_2 = sum of its square). Released checkpoints test both with no
model run. This is the producer of every number §8 quotes from weights.

WHAT IT READS (per model, per step)
-----------------------------------
- **clock**: embedding rows that never left their init direction (cosine > 0.99999; the 27
  unused ids 50277+ and about 215 tokens the corpus never shows). Their gradient is exactly
  zero, so AdamW leaves row_t = exp(-lambda tau_1(t)) row_0. Reported: implied lambda under
  NeoX's schedule (linear ramp, then max(min_lr, peak/2 (1 + cos(pi (i - W)/T)))).
- **writers along 1**: every block reads the stream through a LayerNorm, which removes the
  mean over coordinates, so 1^T W (attention out, MLP out) and W_E 1 get zero gradient summed
  over the coordinate index. Reported: the coefficient of 1^T W_t on 1^T W_0, the residual's
  norm over the init's, and the same for a fixed random unit direction orthogonal to 1.
- **unembedding mean**: adding one vector to every row of W_U shifts all logits alike, so the
  gradient summed over the vocabulary is zero. Reported: the vocabulary mean's norm, the median
  row norm before and after removing it.
- **key bias**: the non-rotary part of each head's key bias adds a key-independent term to the
  scores (E0). Reported per layer: rms per coordinate, over tau_1, share above tau_1 / 2.
- **bias exchange**: attention-out and MLP-out biases enter as a sum (parallel residual).
- **displacement**: for a few matrices, rms(W_t - W_0) over tau_1 and over sqrt(tau_2), the
  share of |W_t - W_0|_F^2 in its top singular direction, and its operator norm over W_0's.

WHAT IT DOES NOT SHOW
---------------------
One run per model. Checkpoints are fp16, so differences below about 1e-5 are rounding (steps
before 16 are not read). The "clock" assumes Adam leaves an exactly-zero gradient alone (true
for apex FusedAdam: 0 / (0 + eps)). Nothing here says why the key bias moves.

RUN: python tools/ts_weights.py pythia-70m | pythia-410m   (writes data/ts/<model>.json in the main tree)
"""
import json, math, sys
import numpy as np
from safetensors import safe_open

HUB = "/run/media/system/WDS_500/Mets/data/hf/hub"
OUT = "/run/media/system/WDS_500/Mets/data/ts"
CFG = {"pythia-70m": (1e-3, [16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 16000, 64000, 143000]),
       "pythia-410m": (3e-4, [16, 64, 256, 512, 1000, 4000, 32000, 143000])}
W, T, LAM = 1430, 143000, 0.1


class Ckpt:
    def __init__(self, model, step):
        root = f"{HUB}/models--EleutherAI--{model}"
        self.rev = open(f"{root}/refs/step{step}").read().strip()
        self.f = safe_open(f"{root}/snapshots/{self.rev}/model.safetensors", "np")

    def __getitem__(self, k):
        return self.f.get_tensor(k).astype(np.float64)


def lr(i, peak):
    return peak * i / W if i <= W else max(0.1 * peak, peak / 2 * (math.cos(math.pi * (i - W) / T) + 1))


def coef(a_t, a_0):
    c = float(a_t @ a_0 / (a_0 @ a_0))
    return c, float(np.linalg.norm(a_t - c * a_0) / np.linalg.norm(a_0))


def main(model):
    peak, steps = CFG[model]
    p0 = Ckpt(model, 0)
    E0, U0 = p0["gpt_neox.embed_in.weight"], p0["embed_out.weight"]
    d = E0.shape[1]; L = {512: 6, 1024: 24}[d]; nh = {512: 8, 1024: 16}[d]; dh = d // nh; rot = dh // 4
    one = np.ones(d) / math.sqrt(d)
    ctrl = np.random.default_rng(0).standard_normal(d); ctrl -= ctrl.mean(); ctrl /= np.linalg.norm(ctrl)
    lay = lambda l, n: f"gpt_neox.layers.{l}.{n}"
    rec = {"model": model, "peak": peak, "warmup": W, "lambda_assumed": LAM, "rev0": p0.rev, "steps": {}}
    for t in steps:
        p = Ckpt(model, t)
        l_ = np.array([lr(i, peak) for i in range(t)]); t1, t2 = float(l_.sum()), float((l_ ** 2).sum())
        r = {"rev": p.rev, "tau1": t1, "sqrt_tau2": math.sqrt(t2), "decay": math.exp(-LAM * t1)}
        E, U = p["gpt_neox.embed_in.weight"], p["embed_out.weight"]
        num = (E * E0).sum(1); ratio = num / (E0 * E0).sum(1)
        fr = num / (np.linalg.norm(E, axis=1) * np.linalg.norm(E0, axis=1)) > 0.99999
        r["clock"] = {"rows": int(fr.sum()), "ratio_med": float(np.median(ratio[fr])),
                      "ratio_range": [float(ratio[fr].min()), float(ratio[fr].max())],
                      "lambda": float(-math.log(np.median(ratio[fr])) / t1)}
        wr = {}
        for n in ("attention.dense.weight", "mlp.dense_4h_to_h.weight"):
            a = [coef(one @ p[lay(l, n)], one @ p0[lay(l, n)]) + coef(ctrl @ p[lay(l, n)], ctrl @ p0[lay(l, n)]) for l in range(L)]
            wr[n] = dict(zip(("coef", "resid", "ctrl_coef", "ctrl_resid"), np.median(np.array(a), 0).tolist()))
        wr["embed_in"] = dict(zip(("coef", "resid", "ctrl_coef", "ctrl_resid"), coef(E @ one, E0 @ one) + coef(E @ ctrl, E0 @ ctrl)))
        r["writers_along_1"] = wr
        mu = U.mean(0)
        r["unembed_mean"] = {"norm": float(np.linalg.norm(mu)), "norm_init": float(np.linalg.norm(U0.mean(0))),
                             "row_med": float(np.median(np.linalg.norm(U, axis=1))),
                             "row_med_centred": float(np.median(np.linalg.norm(U - mu, axis=1)))}
        kb = {}
        for l in range(L):
            b = p[lay(l, "attention.query_key_value.bias")].reshape(nh, 3, dh); x = b[:, 1, rot:]
            kb[l] = {"rms": float(np.sqrt((x ** 2).mean())), "rms_over_tau1": float(np.sqrt((x ** 2).mean()) / t1),
                     "norm": float(np.linalg.norm(x)), "max": float(np.abs(x).max()),
                     "share_above_half_tau1": float((np.abs(x) > 0.5 * t1).mean()),
                     "rms_rotary": float(np.sqrt((b[:, 1, :rot] ** 2).mean())), "rms_query_nonrot": float(np.sqrt((b[:, 0, rot:] ** 2).mean()))}
        r["key_bias"] = kb
        ba = [(p[lay(l, "attention.dense.bias")], p[lay(l, "mlp.dense_4h_to_h.bias")]) for l in range(L)]
        r["bias_exchange"] = {"bit_equal": all(np.array_equal(a, b) for a, b in ba),
                              "max_rel_diff": float(max(np.abs(a - b).max() / np.abs(a).max() for a, b in ba))}
        dis = {}
        for n in [lay(0, "attention.query_key_value.weight"), lay(0, "attention.dense.weight"),
                  lay(L // 2, "mlp.dense_h_to_4h.weight"), lay(L // 2, "mlp.dense_4h_to_h.weight"), lay(L - 1, "mlp.dense_4h_to_h.weight")]:
            W0 = p0[n]; D = p[n] - W0; s = np.linalg.svd(D, compute_uv=False); rms = math.sqrt((D ** 2).mean())
            dis[n[9:]] = {"rms_over_tau1": rms / t1, "rms_over_sqrt_tau2": rms / math.sqrt(t2), "rank1_share": float(s[0] ** 2 / (s ** 2).sum()),
                          "op_over_coherent_max": float(s[0] / (t1 * math.sqrt(D.size))), "op_over_init_op": float(s[0] / np.linalg.norm(W0, 2)),
                          "rms_over_init_rms": rms / float(W0.std())}
        r["displacement"] = dis
        rec["steps"][t] = r
        k = r["clock"]; kbl = kb[L - 1]; dm = dis[lay(L - 1, "mlp.dense_4h_to_h.weight")[9:]]
        print(f"{model} {t:>6}: tau1 {t1:.4g} | clock lambda {k['lambda']:.4f} ({k['rows']} rows) | 1^T attn-out coef {wr['attention.dense.weight']['coef']:.4f} "
              f"(decay {r['decay']:.4f}) resid {wr['attention.dense.weight']['resid']:.3f} vs ctrl {wr['attention.dense.weight']['ctrl_resid']:.3f} | "
              f"W_U mean {r['unembed_mean']['norm']:.3f} / row {r['unembed_mean']['row_med_centred']:.3f} | bK last layer rms/tau1 {kbl['rms_over_tau1']:.3f} "
              f"share>tau1/2 {kbl['share_above_half_tau1']:.2f} | last MLP-out: rms/tau1 {dm['rms_over_tau1']:.3f} rms/sqrt(tau2) {dm['rms_over_sqrt_tau2']:.2f} "
              f"rank1 {dm['rank1_share']:.2f} op/op0 {dm['op_over_init_op']:.2f} | bias exch bit-equal {r['bias_exchange']['bit_equal']} ({r['bias_exchange']['max_rel_diff']:.1e})", flush=True)
    import os; os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/{model}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
