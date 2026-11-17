"""The first Adam updates read from Pythia's weights (`MATH_TRAINING_SYMMETRY.md` §9; tier 1, description).

WHY THIS EXISTS
---------------
§9 derives what the first update from the init must look like (D10) and that two checkpoints
give Adam's mean normalised update between them exactly (D9). Pythia's checkpoints 0, 1, 2, 4, 8,
16 are on the Hub, and so are the batches it trained on, so both can be tested with no forward
pass. This is the producer of every number §9 quotes from weights.

WHAT IT READS (per model)
-------------------------
Inputs: the checkpoints in `data/hf/hub`, and the token counts u (targets) and u_in (inputs) of
iterations 0 and 1 of `pile-standard-pythia-preshuffled` shard 0 (`c4_train.py`'s rows; every
Pythia model saw the same order). Checkpoints are fp16. A parameter that starts at 0 (every bias)
is stored exactly from the first update on (fp16 subnormals step by 2^-24); a weight is read at
step 2 only where |W_0| < 2^-11 (its fp16 step is then under half the update), and everywhere
from step 16.

- **adam**: checkpoint 1 equals checkpoint 0 (the ramp's first rate is 0), so checkpoint 2 is the
  first update that moves. |bias| at 2, 4, 8, 16 in units of the cumulative rate: Adam's mean
  normalised update over each interval (1 would be a sign step).
- **writer_bias**: per layer, the sign of the attention-out bias at step 2 (bit-equal to the
  MLP-out bias) against the prediction from W_U(0) and u alone, sign(centre(W_U(0)^T (u - 1/V)));
  and the final LayerNorm's bias against the uncentred form.
- **reader_bias**: per layer, sign(MLP-in bias) against sign(W_out(0)^T sign(MLP-out bias)); D10c
  gives 0.794 for a Gaussian back-signal.
- **matrices** (step 2 on the readable entries, step 16 on all): the share of entries whose sign is
  (row factor) x (column factor), the row factor being the sign of that matrix's own bias (for W_U,
  sign(u_v - 1/V)) and the column factor fitted on a random half of the rows and scored on the
  other half. For block 0's readers the column factor is also predicted from the init:
  sign(sum_v u_in,v LN(E_0[v])).
- **controls** (`/challenge-pr` on #355): the last block's writer bias against the same prediction
  from a far pair of iterations' counts and from permuted token ids; and, on the tokens whose side
  of 1/V differs between the two pairs, which pair W_U's step-2 rows side with (the only reading
  here that bears on the data order).
- **key_bias**: the non-rotary key bias (E0's direction) per layer: share nonzero and rms over the
  cumulative rate at 2, 4, 8, 16.
- **intervals** (D9): per pair of consecutive checkpoints, the rms of the mean normalised update
  u_bar, its cosine with the first interval's and with the previous interval's, for the last
  block's writer bias, the final LayerNorm's bias, the last and first layers' non-rotary key bias
  and (from step 16) three matrices, decay removed exactly.

WHAT IT DOES NOT SHOW
---------------------
One run per model; the readings were chosen while looking at 70m's numbers (410m is then a
replication of the same rules on another model, same data order). u is the count over the two
batches that enter the first moving update; 1/V stands in for the model's mean prediction at the
init (no forward pass). The column factor is a fit (split-half, so not biased upward), except
where it is named as predicted. Why Adam's first normalised update is 0.85 rather than 1 is not
shown. Matrices before step 16 are a 1.4-2.6 % sample chosen by |W_0|, which is independent of
the gradient's sign. CPU: sign statistics on a few matrices, under two minutes.

RUN: python tools/ts_first_step.py pythia-70m | pythia-410m   (writes data/ts/first_step_<model>.json in the main tree)
"""
import hashlib, json, math, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ts_weights import HUB, OUT, Ckpt, lr, LAM  # noqa: E402

PILE = Path(os.environ.get("C4_ROOT", "/run/media/system/HDD_1TB/mets_data/c4")) / "pile" / "document-00000-of-00020.bin"
ROW, BATCH, V, UNIT = 2049, 1024, 50304, 2.0 ** -24
PEAK = {"pythia-70m": 1e-3, "pythia-410m": 3e-4}
STEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 120000, 143000]
OTHER = 3000                                             # a far pair of iterations, for the controls
lay = lambda l, n: f"gpt_neox.layers.{l}.{n}"


def unigram(i=0):
    """Target and input token frequencies of iterations i and i + 1, and the rows' hash."""
    rows = np.memmap(PILE, dtype="<u2", mode="r", shape=(PILE.stat().st_size // (2 * ROW), ROW))
    b = np.asarray(rows[i * BATCH:(i + 2) * BATCH])
    return (np.bincount(b[:, 1:].ravel(), minlength=V) / b[:, 1:].size, np.bincount(b[:, :-1].ravel(), minlength=V) / b[:, :-1].size,
            hashlib.sha256(b.tobytes()).hexdigest()[:12])


def split_agree(D, mask, row, rng, col=None):
    """Share of readable entries with sign(D) = row_i col_j. col fitted on half the rows, scored on the other half."""
    S = np.sign(D) * mask
    if col is not None:
        return float(((S * row[:, None] * col) > 0).sum() / np.count_nonzero(S)), None
    half = rng.permutation(len(row)) < len(row) // 2; hit = n = 0; cols = []
    for fit, score in ((half, ~half), (~half, half)):
        c = np.where((S[fit] * row[fit, None]).sum(0) >= 0, 1.0, -1.0); cols.append(c)
        hit += ((S[score] * row[score, None] * c) > 0).sum(); n += np.count_nonzero(S[score])
    full = np.where((S * row[:, None]).sum(0) >= 0, 1.0, -1.0)
    return float(hit / n), full


def main(model):
    peak = PEAK[model]; u, ui, bhash = unigram(); rng = np.random.default_rng(0)
    have = [s for s in STEPS if os.path.exists(f"{HUB}/models--EleutherAI--{model}/refs/step{s}")]
    P = {s: Ckpt(model, s) for s in have + [1]}; p0 = P[0]
    U0, E0 = p0["embed_out.weight"], p0["gpt_neox.embed_in.weight"]; d = U0.shape[1]
    L = {512: 6, 1024: 24}[d]; nh = {512: 8, 1024: 16}[d]; dh = d // nh; rot = dh // 4
    tau1 = lambda a, b: sum(lr(i, peak) for i in range(a, b))
    rec = {"model": model, "peak": peak, "batch_sha256": bhash, "revs": {s: P[s].rev for s in P},
           "code_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12],
           "unigram": {"sum_u2": float((u ** 2).sum()), "share_above_1_over_V": float((u > 1 / V).mean()), "share_never": float((u == 0).mean())}}
    keys = [k for k in P[1].f.keys()]
    rec["step1_equals_step0"] = bool(all(np.array_equal(P[1].f.get_tensor(k), p0.f.get_tensor(k)) for k in keys))

    # ---- adam: the writer bias of the last block, in units of the cumulative rate
    kb = lay(L - 1, "mlp.dense_4h_to_h.bias"); adam = {}; prev = (0, 0.0)
    for s in (2, 4, 8, 16):
        a = np.abs(P[s][kb]); med = float(np.median(a)); t1 = tau1(0, s)
        adam[s] = {"median_units": med / UNIT, "tau1_units": t1 / UNIT, "median_over_tau1": med / t1,
                   "interval_mean_u": (med - prev[1]) / (t1 - tau1(0, prev[0]))}
        prev = (s, med)
    vals, cnt = np.unique(np.round(np.abs(P[2][kb]) / UNIT).astype(int), return_counts=True)
    adam["step2_units_hist"] = dict(zip(vals.tolist(), cnt.tolist())); rec["adam"] = adam

    # ---- biases at step 2
    t = U0.T @ (u - 1 / V); pred_c = np.sign(t - t.mean()); p2 = P[2]
    fb = np.sign(p2["gpt_neox.final_layer_norm.bias"])
    wb = {"final_ln_bias_vs_uncentred": float(np.mean(fb == np.sign(t))), "layers": {}}
    rb = {}
    for l in range(L):
        a, m = p2[lay(l, "attention.dense.bias")], p2[lay(l, "mlp.dense_4h_to_h.bias")]
        wb["layers"][l] = {"vs_prediction": float(np.mean(np.sign(a) == pred_c)), "attn_equals_mlp": bool(np.array_equal(a, m)),
                           "vs_last_layer": float(np.mean(np.sign(a) == np.sign(p2[lay(L - 1, "attention.dense.bias")])))}
        rb[l] = float(np.mean(np.sign(p2[lay(l, "mlp.dense_h_to_4h.bias")]) == np.sign(p0[lay(l, "mlp.dense_4h_to_h.weight")].T @ np.sign(m))))
    rec["writer_bias"], rec["reader_bias_vs_Wout_sign"] = wb, rb

    # ---- controls (`/challenge-pr` on #355, finding 2): is it these two batches, or the corpus?
    wlast = np.sign(p2[lay(L - 1, "attention.dense.bias")]); crng = np.random.default_rng(1)
    centred = lambda f: np.sign(U0.T @ (f - 1 / V) - (U0.T @ (f - 1 / V)).mean())
    u_far = unigram(OTHER)[0]
    ctl = {"other_batches": OTHER, "writer_bias_last_other_batches": float(np.mean(wlast == centred(u_far))),
           "writer_bias_last_permuted_ids": float(np.mean(wlast == centred(u[crng.permutation(V)])))}
    # tokens on opposite sides of 1/V in the two pairs: whose side do W_U's step-2 rows take?
    D2 = p2["embed_out.weight"] - U0; S2 = np.sign(D2) * (np.abs(U0) < 2.0 ** -11)
    side0, side1 = np.where(u > 1 / V, 1.0, -1.0), np.where(u_far > 1 / V, 1.0, -1.0); differ = side0 != side1
    col = np.where((S2[~differ] * side0[~differ, None]).sum(0) >= 0, 1.0, -1.0)                 # fitted on the tokens that agree
    n_tok = 2 * BATCH * (ROW - 1); gap = np.abs(u - u_far) * n_tok
    for nm, sel in (("all", differ), ("count_gap_60", differ & (gap >= 60))):
        ctl[f"W_U_differing_tokens_{nm}"] = {"tokens": int(sel.sum()),
                                             "sides_with_first_batches": float(((S2[sel] * side0[sel, None] * col) > 0).sum() / np.count_nonzero(S2[sel]))}
    rec["controls"] = ctl

    # ---- matrices: rank-one sign structure
    lnE = (E0 - E0.mean(1, keepdims=True)) / E0.std(1, keepdims=True); r0 = np.sign(ui @ lnE)
    mats = {}
    for s in (2, 16):
        p = P[s]; thr = 2.0 ** -11 if s == 2 else math.inf; out = {}
        for l in (0, L // 2, L - 1):
            bo, bi = np.sign(p[lay(l, "mlp.dense_4h_to_h.bias")]), np.sign(p[lay(l, "mlp.dense_h_to_4h.bias")])
            bq = np.sign(p[lay(l, "attention.query_key_value.bias")]).reshape(nh, 3, dh)
            qkv0 = p0[lay(l, "attention.query_key_value.weight")]; dq = (p[lay(l, "attention.query_key_value.weight")] - qkv0).reshape(nh, 3, dh, d)
            mq = (np.abs(qkv0) < thr).reshape(nh, 3, dh, d)
            todo = {"mlp_out": (lay(l, "mlp.dense_4h_to_h.weight"), bo), "attn_out": (lay(l, "attention.dense.weight"), bo), "mlp_in": (lay(l, "mlp.dense_h_to_4h.weight"), bi)}
            for nm, (k, row) in todo.items():
                W0 = p0[k]; D = p[k] - W0; mask = np.abs(W0) < thr
                a, col = split_agree(D, mask, row, rng); e = {"agree": a, "read": float(mask.mean())}
                if nm == "mlp_out":
                    e["col_share_positive"] = float((col > 0).mean())
                if nm == "mlp_in" and l == 0:
                    e["agree_col_predicted"] = split_agree(D, mask, row, rng, r0)[0]; e["col_vs_predicted"] = float((col == r0).mean())
                out[f"L{l}.{nm}"] = e
            for ix, nm in enumerate("QKV"):
                Dx, mx, row = dq[:, ix].reshape(-1, d), mq[:, ix].reshape(-1, d), bq[:, ix].reshape(-1)
                row = np.where(row == 0, 1.0, row); a, col = split_agree(Dx, mx, row, rng); e = {"agree": a}
                if l == 0:
                    e["agree_col_predicted"] = split_agree(Dx, mx, row, rng, r0)[0]; e["col_vs_predicted"] = float((col == r0).mean())
                out[f"L{l}.{nm}"] = e
        D = p["embed_out.weight"] - U0; mask = np.abs(U0) < thr; row = np.where(u > 1 / V, 1.0, -1.0)
        a, col = split_agree(D, mask, row, rng); e = {"agree": a, "read": float(mask.mean()), "col_vs_final_ln_bias": float((col == fb).mean())}
        S = np.sign(D) * mask
        for nm, sel in (("above_1_over_V", u > 1 / V), ("seen_below", (u > 0) & (u <= 1 / V)), ("never", u == 0)):
            e[f"rows_{nm}"] = float(((S[sel] * row[sel, None] * col) > 0).sum() / np.count_nonzero(S[sel]))
        out["W_U"] = e
        De = p["gpt_neox.embed_in.weight"] - E0; me = np.abs(E0) < thr; never = ui == 0
        out["W_E"] = {"moved_share_seen_rows": float(np.count_nonzero((np.sign(De) * me)[~never]) / me[~never].sum()),
                      "moved_share_never_rows": float(np.count_nonzero((np.sign(De) * me)[never]) / max(1, me[never].sum()))}
        mats[s] = out
    rec["matrices"] = mats

    # ---- key bias, non-rotary: the first steps
    kbias = {}
    for s in (2, 4, 8, 16):
        t1 = tau1(0, s); kbias[s] = {}
        for l in (0, L // 2, L - 1):
            x = P[s][lay(l, "attention.query_key_value.bias")].reshape(nh, 3, dh)[:, 1, rot:]
            kbias[s][l] = {"nonzero": float((x != 0).mean()), "rms_over_tau1": float(np.sqrt((x ** 2).mean()) / t1)}
    rec["key_bias"] = kbias

    # ---- intervals (D9): the mean normalised update between consecutive checkpoints
    nonrot = lambda x: x.reshape(nh, 3, dh)[:, 1, rot:]
    names = {"writer_bias_last": (lay(L - 1, "mlp.dense_4h_to_h.bias"), 0.0, None, 0), "final_ln_bias": ("gpt_neox.final_layer_norm.bias", 0.0, None, 0),
             "key_bias_last": (lay(L - 1, "attention.query_key_value.bias"), 0.0, nonrot, 0), "key_bias_first": (lay(0, "attention.query_key_value.bias"), 0.0, nonrot, 0),
             "W_U": ("embed_out.weight", LAM, None, 16), "mlp_out_last": (lay(L - 1, "mlp.dense_4h_to_h.weight"), LAM, None, 16),
             "qkv_first": (lay(0, "attention.query_key_value.weight"), LAM, None, 16)}
    first, last, iv = {}, {}, {}
    for a, b in zip(have[:-1], have[1:]):
        et = np.array([lr(i, peak) for i in range(a, b)]); r = {"dtau1": float(et.sum()), "diffusive_scale": float(math.sqrt((et ** 2).sum()) / et.sum())}
        for nm, (k, lam, f, frm) in names.items():
            if a < frm:
                continue
            x0, x1 = P[a][k], P[b][k]
            if f:
                x0, x1 = f(x0), f(x1)
            dec = np.cumprod((1 - lam * et)[::-1])[::-1]; w = et * np.append(dec[1:], 1.0)
            ub = -(x1 - dec[0] * x0) / w.sum(); n = np.linalg.norm(ub)
            cos = lambda z: float((ub * z).sum() / (n * np.linalg.norm(z))) if n > 0 and np.linalg.norm(z) > 0 else None
            r[nm] = {"rms": float(np.sqrt((ub ** 2).mean())), "cos_first": cos(first[nm]) if nm in first else None,
                     "cos_prev": cos(last[nm]) if nm in last else None, "rms_lambda_w": float(lam * np.sqrt((x1 ** 2).mean())),
                     # a window of many decay lengths forces rms(u_bar) toward lambda rms(W) (D9): only short windows test it
                     "window_in_decay_lengths": float(lam * et.sum())}
            first.setdefault(nm, ub); last[nm] = ub
        iv[f"{a}-{b}"] = r
    rec["intervals"] = iv

    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(f"{OUT}/first_step_{model}.json", "w"), indent=1)
    report(rec, L)


def report(rec, L):
    m = rec["model"]; a = rec["adam"]; w = rec["writer_bias"]; M = rec["matrices"]
    print(f"{m}: checkpoint 1 equals 0: {rec['step1_equals_step0']}; batches {rec['batch_sha256']}; sum u^2 {rec['unigram']['sum_u2']:.4f}, "
          f"{100 * rec['unigram']['share_above_1_over_V']:.1f} % of tokens above 1/V, {100 * rec['unigram']['share_never']:.1f} % unseen")
    print(f"  adam: step-2 |bias| in units of 2^-24 {a['step2_units_hist']} (rate {a[2]['tau1_units']:.2f} units); mean u per interval "
          + ", ".join(f"to {s}: {a[s]['interval_mean_u']:.3f}" for s in (2, 4, 8, 16)))
    print("  writer bias at step 2 against sign centre(W_U0^T (u - 1/V)), by layer: " + " ".join(f"{w['layers'][l]['vs_prediction']:.3f}" for l in range(L))
          + f"; against the last layer's: " + " ".join(f"{w['layers'][l]['vs_last_layer']:.3f}" for l in range(L))
          + f"; attn = mlp bias in all layers: {all(w['layers'][l]['attn_equals_mlp'] for l in range(L))}; final-LN bias (uncentred form) {w['final_ln_bias_vs_uncentred']:.3f}")
    c = rec["controls"]; print(f"  controls, last block's writer bias: token counts of iterations {c['other_batches']}-{c['other_batches'] + 1} {c['writer_bias_last_other_batches']:.3f}; "
                               f"permuted token ids {c['writer_bias_last_permuted_ids']:.3f}; W_U rows of tokens on opposite sides of 1/V in the two pairs side with the first: "
                               + "; ".join(f"{k[len('W_U_differing_tokens_'):]} {v['sides_with_first_batches']:.3f} ({v['tokens']} tokens)" for k, v in c.items() if k.startswith("W_U_")))
    rb = rec["reader_bias_vs_Wout_sign"]; print(f"  reader bias against sign(W_out0^T sign b_out): min {min(rb.values()):.3f} median {float(np.median(list(rb.values()))):.3f} max {max(rb.values()):.3f} (D10c 0.794)")
    for s in (2, 16):
        o = M[s]; print(f"  matrices at step {s}: " + "; ".join(f"{k} {v['agree']:.3f}" + (f" (col predicted {v['agree_col_predicted']:.3f}, col match {v['col_vs_predicted']:.3f})" if "agree_col_predicted" in v else "")
                                                             for k, v in o.items() if k not in ("W_U", "W_E")))
        e = o["W_U"]; print(f"    W_U {e['agree']:.3f} (rows above 1/V {e['rows_above_1_over_V']:.3f}, seen below {e['rows_seen_below']:.3f}, never {e['rows_never']:.3f}); "
                            f"W_E moved: seen rows {o['W_E']['moved_share_seen_rows']:.3f}, never-seen rows {o['W_E']['moved_share_never_rows']:.4f}; MLP-out columns positive: "
                            + " ".join(f"{k.split('.')[0]} {v['col_share_positive']:.2f}" for k, v in o.items() if "col_share_positive" in v))
    kbs = rec["key_bias"]; print("  key bias (non-rotary), nonzero share / rms over tau_1, layers first-mid-last: "
                                + "; ".join(f"step {s}: " + " ".join(f"{v['nonzero']:.2f}/{v['rms_over_tau1']:.4f}" for v in kbs[s].values()) for s in kbs))
    print("  intervals, rms of the mean normalised update [cos with the first interval's]:")
    for k, r in rec["intervals"].items():
        f = lambda nm: (f"{r[nm]['rms']:.3f}" + (f" [{r[nm]['cos_first']:+.2f}]" if r[nm]["cos_first"] is not None else "")) if nm in r else "-"
        print(f"    {k:>13}: writer bias {f('writer_bias_last')} | final-LN bias {f('final_ln_bias')} | key bias last {f('key_bias_last')} first {f('key_bias_first')} | "
              f"W_U {f('W_U')} | MLP-out {f('mlp_out_last')} (lambda rms W {r['mlp_out_last']['rms_lambda_w']:.4f}; window {r['mlp_out_last']['window_in_decay_lengths']:.2g} decay lengths) | QKV0 {f('qkv_first')}" if "mlp_out_last" in r else
              f"    {k:>13}: writer bias {f('writer_bias_last')} | final-LN bias {f('final_ln_bias')} | key bias last {f('key_bias_last')} first {f('key_bias_first')}")


if __name__ == "__main__":
    main(sys.argv[1])
