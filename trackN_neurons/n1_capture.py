"""
trackN_neurons/n1_capture.py — N1 U1: the populated rule's gate, then the capture
(`trackN_neurons/design-N1.md`, frozen 2026-11-08: "What a pole is under GeLU", "U1").

Each neuron i at a token: z = (W_in LN(x) + b_in)_i, h = GeLU(z) (exact, z Phi(z)). Three readings:
  P+  the top k tokens by z;  P-  the bottom k by z;  B  the k smallest h (the band at z* = -0.75179).
Per neuron and step z is standardised over the library's tokens, z~ = (z - median) / (1.4826 MAD):
  stat P+ = mean z~ of P+'s k,  stat P- = -mean z~ of P-'s k,  stat B = -mean |z - z*| / s of B's k.
A reading is populated if its stat exceeds the 95th percentile of the same stat over 512 Haar rows
at the same layer and step, scaled to the neuron's ||W_in,i|| with its own b_in,i. P+'s and P-'s
stats are invariant to a row's positive scale and its bias, so their null is one threshold per
layer, step and k; B's depends on (norm, bias) through v* = (z* - b) / ||w||, so it is per neuron.
k = 10, 20, 50 are all reported, none picked.

  gate     layer 0 of pythia-410m step 143000 (the MLP reads LN(embedding): a token's z depends on
           its id alone, checked). Planted per token t: P+ row = its LN'd input's direction, P- row
           = the negation, B row = the direction with the bias that puts t at z*; each must be
           populated with t's positions as its whole top k, at every k. 512 held-out Haar rows
           (median norm and bias) must read populated at <= 10 % for every reading and k.
           A failed arm stops the counts (exit 1).
  capture  steps 0 and 143000, all 24 x 4096 neurons: stats, nulls, populated flags, each
           reading's top 50 positions, z's quantiles; counts per reading, layer and k; 23.417
           printed. Refuses unless the gate's record passed.

Input: core.library.load("val", 1024) (library_hash recorded), the model's tokenizer at the
library's pinned revision, no special tokens, each item one forward pass, float32, eager, TF32 off,
CUDA. Description only: nothing here is scored.

  METS_DATA=<main>/data HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
    python -m trackN_neurons.n1_capture gate|capture [--out DIR]
Outputs: DIR (default $METS_DATA/trackN/n1) gate.json, step{S}.npz, capture_step{S}.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODEL = "EleutherAI/pythia-410m"
TOKENIZER_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"   # prompt_library/build_l1.py
SPLIT, K_ITEMS = "val", 1024
Z_STAR = -0.751791524693564                                       # tools/math_checks/n1_gelu_pole.py G1
KS = (10, 20, 50)
KMAX = max(KS)
N_NULL = 512
PCT = 95.0
HELD_OUT_BAR = 0.10
GATE_STEP, GATE_LAYER, GATE_TOKENS = 143000, 0, 8
GATE_COUNT_RANGE = (200, 5000)
CAPTURE_STEPS = (0, 143000)
LAYERS_PER_PASS = 4
BLOCK = 128
READINGS = ("P+", "P-", "B")
QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
KNOWN = (23, 417)                                                 # Stolfo et al. 2406.16254


class Refused(SystemExit):
    pass


# ------------------------------------------------------------------ statistics (device-agnostic)
# Layout: one row per neuron or direction, its T values along dim 1, sorted ascending.

def gelu(z):
    import torch
    return 0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))


def med_scale(sv):
    """Median and 1.4826 MAD per row of row-sorted values sv (n, T)."""
    import torch
    T = sv.shape[1]
    med = 0.5 * (sv[:, (T - 1) // 2] + sv[:, T // 2])
    dev = (sv - med[:, None]).abs()
    lo = torch.kthvalue(dev, (T - 1) // 2 + 1, dim=1).values
    hi = torch.kthvalue(dev, T // 2 + 1, dim=1).values
    return med, 1.4826 * 0.5 * (lo + hi)


def band(sv, vstar, scale, k):
    """Columns (into sv's rows) of the k smallest h(scale (v - v*) + z*) per row, i.e. the k
    smallest GeLU outputs of a row with norm `scale` and bias z* - scale v*. h is unimodal in v
    with its minimum at v*, so the k smallest form a contiguous run around v* in sorted order and
    a window of k on each side of v*'s insertion point holds it exactly.
    sv (n, T) row-sorted; vstar, scale (m, n). Returns cols (m, n, k) and |v - v*| there."""
    import torch
    n, T = sv.shape
    ins = torch.searchsorted(sv, vstar.T.contiguous())                       # (n, m)
    lo = (ins - k).clamp(0, max(T - 2 * k, 0))
    cols = (lo[..., None] + torch.arange(2 * k, device=sv.device)).clamp(max=T - 1)
    m = vstar.shape[0]
    vals = torch.gather(sv, 1, cols.reshape(n, -1)).reshape(n, m, 2 * k)
    off = vals - vstar.T[..., None]
    h = gelu(scale.T[..., None] * off + Z_STAR)
    _, pick = torch.topk(h, k, dim=-1, largest=False)
    cols, d = torch.gather(cols, -1, pick), torch.gather(off.abs(), -1, pick)
    return cols.permute(1, 0, 2), d.permute(1, 0, 2)                         # (m, n, k)


def pm_stats(sv, med, s):
    """P+ and P- stats (2, len(KS), n) from row-sorted values."""
    import torch
    T = sv.shape[1]
    return torch.stack([torch.stack([(sv[:, T - k:].mean(1) - med) / s for k in KS]),
                        torch.stack([(med - sv[:, :k].mean(1)) / s for k in KS])])


def column_stats(Z, want_rows=True):
    """Per column of Z (T, n) (one neuron or row each): the three readings' stats for each k, the
    top-KMAX token positions of each reading, med, s and z's quantiles. B ranks by h(z) exactly."""
    import torch
    sv, si = torch.sort(Z.T.contiguous(), dim=1)                             # (n, T)
    n, T = sv.shape
    med, s = med_scale(sv)
    one = torch.ones((1, n), device=Z.device, dtype=Z.dtype)
    bcol, bd = band(sv, Z_STAR * one, one, KMAX)                             # (1, n, k)
    st = torch.cat([pm_stats(sv, med, s),
                    torch.stack([-bd[0, :, :k].mean(1) / s for k in KS])[None]])
    q = torch.stack([sv[:, min(T - 1, int(round(p * (T - 1))))] for p in QUANTILES])
    out = {"stat": st, "med": med, "s": s, "q": q}
    if want_rows:
        out["rows"] = torch.stack([si[:, -KMAX:].flip(1), si[:, :KMAX],
                                   torch.gather(si, 1, bcol[0])]).permute(0, 2, 1)  # (3, k, n)
    return out


def row_stats(X, W, b, want_rows=True):
    """column_stats of z = X W^T + b in blocks of BLOCK rows, concatenated along the row axis."""
    import torch
    parts = [column_stats(X @ W[c:c + BLOCK].T + b[c:c + BLOCK], want_rows)
             for c in range(0, W.shape[0], BLOCK)]
    ax = {"stat": 2, "rows": 2, "med": 0, "s": 0, "q": 1}
    return {k: torch.cat([p[k] for p in parts], dim=ax[k]) for k in parts[0]}


def null_sorted(X, D, chunk=128):
    """Haar projections X D^T row-sorted (512, T), their med and scale, built in chunks."""
    import torch
    sv, med, s = [], [], []
    for c in range(0, D.shape[0], chunk):
        v, _ = torch.sort((X @ D[c:c + chunk].T).T.contiguous(), dim=1)
        m_, s_ = med_scale(v)
        sv.append(v); med.append(m_); s.append(s_)
    return torch.cat(sv), torch.cat(med), torch.cat(s)


def null_pm(sv, med, s):
    """95th-percentile thresholds (2, len(KS)) for P+ and P- (affine-invariant, one per k)."""
    import torch
    return torch.quantile(pm_stats(sv, med, s), PCT / 100.0, dim=-1)


def null_b(sv, s_d, norm, bias, chunk=256):
    """Per neuron (norm, bias), B's 95th percentile over the Haar rows scaled to it: (len(KS), n).
    In v = x.d units the row's band sits at v* = (z* - bias) / norm with scale `norm`."""
    import torch
    out = []
    nd = sv.shape[0]
    for c0 in range(0, norm.shape[0], chunk):
        nc, bc = norm[c0:c0 + chunk], bias[c0:c0 + chunk]
        vs = ((Z_STAR - bc) / nc)[:, None].expand(-1, nd)
        _, d = band(sv, vs, nc[:, None].expand(-1, nd), KMAX)               # (c, 512, k)
        st = torch.stack([-d[..., :k].mean(-1) / s_d for k in KS])          # (K, c, 512)
        out.append(torch.quantile(st, PCT / 100.0, dim=-1))
    return torch.cat(out, dim=1)


def haar(d, n, seed, device):
    import torch
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n, d, generator=g, dtype=torch.float64)
    return (D / D.norm(dim=1, keepdim=True)).to(device, torch.float32)        # (n, d) unit rows


# ------------------------------------------------------------------ model passes

def torch_setup():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise Refused("refusing: no CUDA device (N1's passes are the GPU's, design-N1 U1 step 3)")
    return torch, torch.device("cuda")


def load_model(step, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    m = from_pretrained_eager(GPTNeoXForCausalLM, MODEL, revision=f"step{step}",
                              torch_dtype=torch.float32).to(device).eval()
    if next(m.parameters()).dtype != torch.float32:
        raise Refused(f"step {step}: weights are not float32")
    if not m.config.use_parallel_residual or m.config.hidden_act != "gelu":
        raise Refused("expected Pythia's parallel residual and exact gelu")
    return m


def library_tokens():
    from transformers import AutoTokenizer
    from core import library
    items = library.load(SPLIT, K_ITEMS, context="trackN_neurons/n1_capture")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=TOKENIZER_REVISION)
    ids = []
    for it in items:
        t = tok(it.text, add_special_tokens=False)["input_ids"][:512]
        if len(t) != it.n_tokens:
            raise Refused(f"{it.key}: {len(t)} tokens, the manifest {it.n_tokens}")
        ids.append(t)
    return tok, ids, library.library_hash(SPLIT, K_ITEMS)


class _Stop(Exception):
    pass


def capture_inputs(model, ids, layers, device):
    """LN'd MLP inputs (T, d) float32 on CPU for each layer in `layers`, one item per pass; the
    pass stops after the deepest layer needed."""
    import torch
    buf, hooks, last = {L: [] for L in layers}, [], max(layers)
    for L in layers:
        def hook(m, inp, L=L):
            buf[L].append(inp[0][0].detach().to("cpu"))
            if L == last:
                raise _Stop
        hooks.append(model.gpt_neox.layers[L].mlp.dense_h_to_4h.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for t in ids:
                try:
                    model.gpt_neox(torch.tensor([t], device=device))
                except _Stop:
                    pass
    finally:
        for hk in hooks:
            hk.remove()
    return {L: torch.cat(buf[L]) for L in layers}


def code_sha():
    """HEAD, refusing when this file differs from it (`/challenge-pr` on #328, finding 4)."""
    import hashlib
    me = Path(__file__).resolve()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", str(me)],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise Refused(f"refusing: {me.name} is uncommitted; commit it so the record names its code")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    return f"{head} sha256:{hashlib.sha256(me.read_bytes()).hexdigest()[:16]}"


def meta(torch, lib_hash):
    return {"design": "trackN_neurons/design-N1.md (frozen 2026-11-08)", "code": code_sha(),
            "model": MODEL, "library": [SPLIT, K_ITEMS], "library_hash": lib_hash,
            "tokenizer_revision": TOKENIZER_REVISION, "dtype": "float32", "tf32": False,
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "ks": list(KS), "n_null": N_NULL, "pct": PCT, "z_star": Z_STAR}


# ------------------------------------------------------------------ gate

def gate(a):
    torch, device = torch_setup()
    tok, ids, lib_hash = library_tokens()
    t0 = time.time()
    model = load_model(GATE_STEP, device)
    X = capture_inputs(model, ids, [GATE_LAYER], device)[GATE_LAYER].to(device)
    mlp = model.gpt_neox.layers[GATE_LAYER].mlp.dense_h_to_4h
    W, b = mlp.weight.detach().float().clone(), mlp.bias.detach().float().clone()
    ln = model.gpt_neox.layers[GATE_LAYER].post_attention_layernorm
    emb = model.gpt_neox.embed_in
    flat = torch.tensor([i for t in ids for i in t], device=device)
    T = X.shape[0]
    rec = {**meta(torch, lib_hash), "step": GATE_STEP, "layer": GATE_LAYER, "T": T}

    # checks: the hook reads LN(embedding) at layer 0 (parallel residual); it is the token's alone
    with torch.no_grad():
        ref = ln(emb(torch.tensor(ids[0], device=device)))
        rec["hook_check_max_abs"] = float((ref - X[:len(ids[0])]).abs().max())
    if rec["hook_check_max_abs"] > 1e-4:
        raise Refused(f"the hook does not read LN(embedding) at layer 0: {rec}")
    del model, ln, emb, mlp
    torch.cuda.empty_cache()
    uniq, inv, cnt = torch.unique(flat, return_inverse=True, return_counts=True)
    first = torch.full((len(uniq),), X.shape[0], device=device).scatter_reduce_(
        0, inv, torch.arange(X.shape[0], device=device), "amin")
    mean_u = X[first]                     # each token's input at its first position
    rec["token_determinism_max_abs"] = float((X - mean_u[inv]).abs().max())
    if rec["token_determinism_max_abs"] > 1e-4:
        raise Refused(f"layer 0's MLP input is not a function of the token: {rec}")

    # planted tokens: seeded, among tokens with 200..5000 positions
    lo, hi = GATE_COUNT_RANGE
    cand = uniq[(cnt >= lo) & (cnt <= hi)].cpu().numpy()
    rng = np.random.default_rng(0)
    toks = sorted(int(t) for t in rng.choice(cand, GATE_TOKENS, replace=False))
    norms = W.norm(dim=1)
    a_med, b_med = float(norms.median()), float(b.median())
    rec.update({"candidates": int(len(cand)), "tokens": toks,
                "token_text": [tok.decode([t]) for t in toks], "a_med": a_med, "b_med": b_med})

    rows, bias, name = [], [], []
    for t in toks:
        u = mean_u[(uniq == t).nonzero()[0, 0]]
        uh = u / u.norm()
        rows += [a_med * uh, -a_med * uh, a_med * uh]
        bias += [b_med, b_med, Z_STAR - a_med * float(u.norm())]
        name += [(t, "P+"), (t, "P-"), (t, "B")]
    Wp, bp = torch.stack(rows), torch.tensor(bias, device=device)
    cs = column_stats(X @ Wp.T + bp)

    D0 = haar(X.shape[1], N_NULL, 0, device)
    sv0, m0, s0 = null_sorted(X, D0)
    thr_pm = null_pm(sv0, m0, s0)
    thr_b = null_b(sv0, s0, torch.full((len(rows),), a_med, device=device), bp)   # (K, rows)

    planted = []
    ok_planted = True
    for j, (t, r) in enumerate(name):
        ri = READINGS.index(r)
        thr = thr_pm[ri] if ri < 2 else thr_b[:, j]
        pos_tok = flat[cs["rows"][ri, :, j]]
        per_k = []
        for kk, k in enumerate(KS):
            prec = float((pos_tok[:k] == t).float().mean())
            pop = bool(cs["stat"][ri, kk, j] > thr[kk])
            per_k.append({"k": k, "stat": float(cs["stat"][ri, kk, j]), "thr": float(thr[kk]),
                          "populated": pop, "precision": prec})
            ok_planted &= pop and prec == 1.0
        planted.append({"token": t, "reading": r, "per_k": per_k})
    rec["planted"] = planted

    D1 = haar(X.shape[1], N_NULL, 1, device)
    ch = row_stats(X, a_med * D1, torch.full((N_NULL,), b_med, device=device), want_rows=False)
    thr_bh = null_b(sv0, s0, torch.full((N_NULL,), a_med, device=device),
                    torch.full((N_NULL,), b_med, device=device))
    held = {}
    ok_held = True
    for ri, r in enumerate(READINGS):
        thr = thr_pm[ri][:, None] if ri < 2 else thr_bh
        rate = (ch["stat"][ri] > thr).float().mean(-1)
        held[r] = {str(k): float(rate[kk]) for kk, k in enumerate(KS)}
        ok_held &= bool((rate <= HELD_OUT_BAR).all())
    rec["held_out_rate"] = held
    rec["thr_pm"] = {r: thr_pm[i].tolist() for i, r in enumerate(READINGS[:2])}
    arms = {r: all(all(x["populated"] and x["precision"] == 1.0 for x in p["per_k"])
                   for p in planted if p["reading"] == r) for r in READINGS}
    rec["arms"] = {**{f"planted {r}": v for r, v in arms.items()}, "held-out": ok_held}
    rec["passed"] = bool(ok_planted and ok_held)
    rec["wall_s"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("arms", "held_out_rate", "passed", "wall_s",
                                          "token_text", "hook_check_max_abs",
                                          "token_determinism_max_abs")}, indent=1))
    return 0 if rec["passed"] else 1


def sizing(a):
    """Post hoc, for Blocked 145 (`/challenge-pr` on #328, finding 3), not a gate and no count:
    the gate's planted P+ rows against (i) the Haar null with 4 seeds, to size the threshold's
    sampling spread, and (ii) a covariance null, rows ~ N(0, Cov(X)) normalised, the repair
    proposed for N2. Layer 0, step 143000, as the gate."""
    torch, device = torch_setup()
    tok, ids, lib_hash = library_tokens()
    g = json.loads((Path(a.out) / "gate.json").read_text())
    model = load_model(GATE_STEP, device)
    X = capture_inputs(model, ids, [GATE_LAYER], device)[GATE_LAYER].to(device)
    del model
    torch.cuda.empty_cache()
    flat = torch.tensor([i for t in ids for i in t], device=device)
    uniq, inv = torch.unique(flat, return_inverse=True)
    first = torch.full((len(uniq),), X.shape[0], device=device).scatter_reduce_(
        0, inv, torch.arange(X.shape[0], device=device), "amin")
    U = torch.stack([X[first[(uniq == t).nonzero()[0, 0]]] for t in g["tokens"]])
    U = U / U.norm(dim=1, keepdim=True)
    planted = row_stats(X, U, torch.zeros(len(U), device=device), want_rows=False)["stat"][0]
    rec = {**meta(torch, lib_hash), "kind": "post hoc sizing for Blocked 145; not a gate",
           "tokens": g["tokens"], "planted_P+": planted.T.tolist()}
    rec["haar_thr_P+_by_seed"] = {}
    for seed in (0, 2, 3, 4):
        sv, m_, s_ = null_sorted(X, haar(X.shape[1], N_NULL, seed, device))
        rec["haar_thr_P+_by_seed"][seed] = null_pm(sv, m_, s_)[0].tolist()
        del sv
    mu = X.double().mean(0)
    C = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=device)
    for c in range(0, X.shape[0], 65536):
        xc = X[c:c + 65536].double() - mu
        C += xc.T @ xc
    C /= X.shape[0] - 1
    ev, V = torch.linalg.eigh(C)
    gen = torch.Generator().manual_seed(7)
    Gs = torch.randn(N_NULL, X.shape[1], generator=gen, dtype=torch.float64).to(device)
    R = (Gs * ev.clamp(min=0).sqrt()) @ V.T
    R = (R / R.norm(dim=1, keepdim=True)).float()
    sv, m_, s_ = null_sorted(X, R)
    rec["cov_thr_P+"] = null_pm(sv, m_, s_)[0].tolist()
    rec["top_eig_share"] = float(ev[-1] / ev.sum())
    out = Path(a.out) / "sizing.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("planted_P+", "haar_thr_P+_by_seed", "cov_thr_P+",
                                          "top_eig_share")}, indent=0))
    return 0


# ------------------------------------------------------------------ capture

def capture(a):
    out = Path(a.out)
    g = json.loads((out / "gate.json").read_text()) if (out / "gate.json").exists() else None
    if not g or not g.get("passed"):
        raise Refused("refusing: the gate's record is missing or failed (design-N1 U1 step 2)")
    torch, device = torch_setup()
    tok, ids, lib_hash = library_tokens()
    if g["library_hash"] != lib_hash:
        raise Refused("the gate ran on another library prefix")
    flat = np.array([i for t in ids for i in t], dtype=np.int64)
    starts = np.cumsum([0] + [len(t) for t in ids])
    for step in a.steps:
        f = out / f"step{step}.npz"
        if f.exists():
            print(f"step {step}: on disk, skipped")
            continue
        t0 = time.time()
        model = load_model(step, device)
        nL = model.config.num_hidden_layers
        res = {k: [] for k in ("stat", "thr_pm", "thr_b", "rows", "med", "s", "q", "norm", "bias")}
        for g0 in range(0, nL, LAYERS_PER_PASS):
            Ls = list(range(g0, min(nL, g0 + LAYERS_PER_PASS)))
            Xs = capture_inputs(model, ids, Ls, device)
            for L in Ls:
                X = Xs.pop(L).to(device)
                lin = model.gpt_neox.layers[L].mlp.dense_h_to_4h
                W, b = lin.weight.detach().float(), lin.bias.detach().float()
                D = haar(X.shape[1], N_NULL, 1000 + L, device)
                svd, m_d, s_d = null_sorted(X, D)
                thr_pm = null_pm(svd, m_d, s_d)
                norm = W.norm(dim=1)
                thr_b = null_b(svd, s_d, norm, b)
                del svd
                cs = {k: v.cpu().numpy() for k, v in row_stats(X, W, b).items()}
                res["stat"].append(cs["stat"]); res["rows"].append(cs["rows"].astype(np.int32))
                res["med"].append(cs["med"]); res["s"].append(cs["s"]); res["q"].append(cs["q"])
                res["thr_pm"].append(thr_pm.cpu().numpy()); res["thr_b"].append(thr_b.cpu().numpy())
                res["norm"].append(norm.cpu().numpy()); res["bias"].append(b.cpu().numpy())
                del X, cs
                torch.cuda.empty_cache()
                print(f"step {step} layer {L} done {time.time() - t0:.0f}s", flush=True)
        arr = {k: np.stack(v) for k, v in res.items()}
        stat = arr["stat"]                                                     # (L, 3, K, n)
        thr = np.concatenate([np.broadcast_to(arr["thr_pm"][..., None], stat[:, :2].shape),
                              arr["thr_b"][:, None]], axis=1)
        pop = stat > thr
        if not np.isfinite(stat).all() or (arr["s"] <= 0).any():
            raise Refused(f"step {step}: non-finite stats or a zero scale; nothing written")
        np.savez(f, **arr, populated=pop, tokens=flat, starts=starts)
        wall = time.time() - t0
        summ = {**meta(torch, lib_hash), "step": step, "T": int(len(flat)), "wall_s": round(wall, 1),
                "counts": {r: {str(k): pop[:, ri, kk].sum(-1).tolist() for kk, k in enumerate(KS)}
                           for ri, r in enumerate(READINGS)},
                "totals": {r: {str(k): int(pop[:, ri, kk].sum()) for kk, k in enumerate(KS)}
                           for ri, r in enumerate(READINGS)},
                "n_neurons": int(pop.shape[0] * pop.shape[-1])}
        L, i = KNOWN
        look = {}
        for ri, r in enumerate(READINGS):
            ctx = []
            for p in arr["rows"][L, ri, :10, i]:
                d = int(np.searchsorted(starts, p, side="right") - 1)
                ctx.append(tok.decode(flat[max(starts[d], p - 8):p].tolist()) + "[["
                           + tok.decode([int(flat[p])]) + "]]")
            look[r] = {"stat": stat[L, ri, :, i].tolist(), "populated": pop[L, ri, :, i].tolist(),
                       "top10": ctx}
        summ["known_23_417"] = look
        (out / f"capture_step{step}.json").write_text(json.dumps(summ, indent=1) + "\n")
        print(json.dumps(summ["totals"]), f"wall {wall:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("stage", choices=["gate", "capture", "sizing"])
    p.add_argument("--out", default=str(Path(os.environ.get("METS_DATA", REPO / "data"))
                                        / "trackN" / "n1"))
    p.add_argument("--steps", type=int, nargs="+", default=list(CAPTURE_STEPS))
    a = p.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return {"gate": gate, "capture": capture, "sizing": sizing}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
