"""
trackT_timaeus/t5f.py — T5 U3's split: which paths write the shared direction, at Dodecahedron
`-32m`'s 9 split steps and Pythia-410m's 8 (`trackT_timaeus/design-T5.md` "U3" and "U3 revised
toward the late rise"; the details below are fixed in "U3's split, the details", before any output).

Per block ℓ and prompt p, u_ℓp is the mean over the prompt's tokens (position 0 included) of the
block's raw output (the residual stream; at the last block, before the final norm), normalised.
The block's write w_i = x_out,i − x_in,i is split into six vectors, and each token's component on
u_ℓp is taken:

| path | vector |
|---|---|
| attn_sink | Σ_h a^h_i0 · W_O^h W_V^h x̃_0 (x̃ the attention's input, after its norm; bias-free value) |
| attn_bias | b_O + Σ_h W_O^h b_V^h (rows of a sum to 1); zero on Dodecahedron unless planted |
| attn_rest | Σ_h Σ_{j≥1} a^h_ij · W_O^h W_V^h x̃_j |
| mlp_bias | the MLP's output bias; zero on Dodecahedron |
| mlp_const | the bias-free MLP write's mean over every token of the 7 prompts (per block) |
| mlp_rest | the bias-free MLP write less mlp_const |

Per path: **share** = Σ over blocks, prompts, tokens of its component / the same sum of the write's;
**size** per block = its mean component per token / the root mean square token norm at the block's
output (dimensionless), and its mean over blocks. Beside: the same sums without position 0.

Gates (`design-T5.md` "Calibration gate", U3, and "all"): the parts sum to the block's write (max
per-token relative error ≤ 1e-4); a planted o_proj bias at `-32m` 9536 (one vector, 0.5 × the median
token norm over the middle states, a fixed random direction, in every block) has its bias path ≥ 0.9
of the planted component (the planted model's summed write component less the unplanted one's); ρ_ℓ
from the same forward equals T2's `diag` at 32m 1, 9536 and 410m 0, 4000 to 1e-6.

Tier 1, description (nothing here is registered; no H2 label: H2 is read only by U4).
    METS_DATA=<main>/data python -m trackT_timaeus.t5f run --out <dir>
    METS_DATA=<main>/data python -m trackT_timaeus.t5f read --out <dir> --first <t5u3 first-read dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from trackA_force.a1a import DATA, REPO, V1, prompts_hash

PROMPTS_HASH = "976b62d677bc"
PATHS = ("attn_sink", "attn_bias", "attn_rest", "mlp_bias", "mlp_const", "mlp_rest")
#: model → (repo, split steps, tokenizer); design-T5.md "U3" (step 1 kept, finding 2 on #301)
MODELS = {
    "32m": ("timaeus/dodecahedron-32m", (1, 917, 1253, 1713, 2340, 3198, 4369, 6979, 9536), "llama"),
    "410m": ("EleutherAI/pythia-410m", (0, 16, 128, 512, 1000, 4000, 32000, 143000), "neox"),
}
#: the rise steps whose first-read row is reported beside (design-T5.md "then the split")
RISE = (917, 1253, 1713, 2340, 3198, 4369, 6979, 9536)
# the gates (design-T5.md "Calibration gate"; fixed before output)
SUM_TOL = 1e-4
PLANT_STEP, PLANT_SCALE, PLANT_MIN, PLANT_SEED = 9536, 0.5, 0.9, 20261102
KNOWN_DIR = "trackT/t2u1/2026-10-29-diag"
KNOWN = {"32m": (1, 9536), "410m": (0, 4000)}
KNOWN_TOL = 1e-6
DEFAULT_ROOTS = (str(DATA / "hf"), "/run/media/system/HDD_1TB/mets_data/hf")


class T5fRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t5f.py", "trackT_timaeus/t5.py", "trackT_timaeus/t2.py", "trackC_writeup/c2_aniso.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the model's maps

def arch_of(model) -> str:
    t = model.config.model_type
    if t not in ("llama", "gpt_neox"):
        raise T5fRefused(f"model_type {t}: the split knows llama and gpt_neox only")
    return t


def blocks(model):
    """(layer, attention, mlp) per block."""
    if arch_of(model) == "llama":
        return [(L, L.self_attn, L.mlp) for L in model.model.layers]
    return [(L, L.attention, L.mlp) for L in model.gpt_neox.layers]


def head_maps(model, attn, mlp):
    """W_V (H, hs, d), b_V (H, hs) or None, W_O (d, H, hs), b_O (d) or None, the MLP's output bias
    (d) or None; float64. GQA's KV heads are repeated as `repeat_kv` does (head h reads KV h // n_rep);
    NeoX's fused qkv is laid out per head as (q, k, v) (`_attn_projections_and_rope`)."""
    import torch
    cfg = model.config
    d, H = cfg.hidden_size, cfg.num_attention_heads
    hs = d // H
    f = lambda t: None if t is None else t.detach().to(torch.float64)
    if arch_of(model) == "llama":
        kv = cfg.num_key_value_heads
        Wv = f(attn.v_proj.weight).view(kv, hs, d).repeat_interleave(H // kv, 0)
        bv = attn.v_proj.bias
        bv = None if bv is None else f(bv).view(kv, hs).repeat_interleave(H // kv, 0)
        Wo, bo, bm = f(attn.o_proj.weight).view(d, H, hs), f(attn.o_proj.bias), f(mlp.down_proj.bias)
    else:
        Wv = f(attn.query_key_value.weight).view(H, 3, hs, d)[:, 2]
        bq = attn.query_key_value.bias
        bv = None if bq is None else f(bq).view(H, 3, hs)[:, 2]
        Wo, bo, bm = f(attn.dense.weight).view(d, H, hs), f(attn.dense.bias), f(mlp.dense_4h_to_h.bias)
    return Wv, bv, Wo, bo, bm


# ---------------------------------------------------------------- one prompt

def forward_capture(model, enc, device):
    """One forward with hooks: per block the input, output, attention input and output, MLP output;
    and HF's hidden states and attention weights."""
    import torch
    cap: Dict[tuple, object] = {}
    hooks = []

    def first(args, kwargs):
        return args[0] if args else kwargs["hidden_states"]
    for l, (L, A, M) in enumerate(blocks(model)):
        hooks.append(L.register_forward_pre_hook(lambda m, a, k, l=l: cap.__setitem__(("in", l), first(a, k)), with_kwargs=True))
        hooks.append(L.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("out", l), o[0])))
        hooks.append(A.register_forward_pre_hook(lambda m, a, k, l=l: cap.__setitem__(("ax", l), first(a, k)), with_kwargs=True))
        hooks.append(A.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("ao", l), o[0])))
        hooks.append(M.register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(("mo", l), o)))
    try:
        with torch.no_grad():
            o = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True,
                      output_attentions=True)
    finally:
        for h in hooks:
            h.remove()
    return cap, o


def split_block(x_in, x_out, ax, att, ao, mo, maps) -> dict:
    """One block of one prompt (all float64 torch, batch dropped): per path its summed component
    on u (and without position 0), the bias-free MLP write's token sum, and the checks."""
    import torch
    Wv, bv, Wo, bo, bm = maps
    n, d = x_out.shape
    V = torch.einsum("hrd,nd->hnr", Wv, ax)                  # bias-free values
    Z = torch.einsum("hnr,dhr->hnd", V, Wo)                  # each head's value through W_O
    sink = torch.einsum("hi,hd->id", att[:, :, 0], Z[:, 0])
    rest = torch.einsum("hij,hjd->id", att[:, :, 1:], Z[:, 1:])
    abias = torch.zeros(d, dtype=x_out.dtype, device=x_out.device)
    if bo is not None:
        abias = abias + bo
    if bv is not None:
        abias = abias + torch.einsum("dhr,hr->d", Wo, bv)
    mbias = bm if bm is not None else torch.zeros_like(abias)
    mfree = mo - mbias
    write = x_out - x_in
    attn = sink + rest + abias
    rel = lambda a, b: float((torch.linalg.norm(a - b, dim=-1) / torch.linalg.norm(b, dim=-1)).max())
    mu = x_out.mean(0)
    u = mu / torch.linalg.norm(mu)
    comps = {"attn_sink": sink @ u, "attn_bias": (abias @ u).expand(n), "attn_rest": rest @ u,
             "mlp_bias": (mbias @ u).expand(n), "mlp_free": mfree @ u, "write": write @ u}
    return {
        "n": n, "u": u, "mfree_sum": mfree.sum(0),
        "sum": {k: float(v.sum()) for k, v in comps.items()},
        "sum_no0": {k: float(v[1:].sum()) for k, v in comps.items()},
        "sq": float((x_out * x_out).sum()),
        "err_sum": rel(attn + mbias + mfree, write),          # the gate: the parts sum to the write
        "err_attn": rel(attn, ao),                            # beside: the recomputed attention
        "err_modules": rel(ao + mo, write),                   # beside: the hooks cover the write
        "bias_norm": {"attn": float(torch.linalg.norm(abias)), "mlp": float(torch.linalg.norm(mbias))},
    }


def split_prompt(model, enc, device) -> dict:
    """``enc``: the tokenizer's tensors, as T2's `c.states` makes them (so ρ_ℓ is T2's)."""
    import torch
    from trackT_timaeus import t5
    cap, o = forward_capture(model, enc, device)
    f = lambda t: t[0].detach().to(torch.float64)
    per = []
    for l, (L, A, M) in enumerate(blocks(model)):
        att = o.attentions[l][0].detach().to(torch.float64)
        per.append(split_block(f(cap[("in", l)]), f(cap[("out", l)]), f(cap[("ax", l)]), att,
                               f(cap[("ao", l)]), f(cap[("mo", l)]), head_maps(model, A, M)))
    hs = [h[0].to(torch.float32).cpu().numpy() for h in o.hidden_states]
    rho = [t5.state_stats(h)["rho"] for h in hs]
    mid = np.concatenate([np.linalg.norm(np.asarray(h, np.float64), axis=1) for h in hs[1:-1]])
    return {"blocks": per, "rho": rho, "mid_norms": mid}


# ---------------------------------------------------------------- over prompts (pure numpy)

def aggregate(prompts: List[dict]) -> dict:
    """The record of one model: per block and path the summed component (mlp_free split into its
    token-constant part and the rest, the mean taken over every token of every prompt), shares,
    sizes, the gate's errors. ``prompts``: `split_prompt`'s blocks, with u and mfree_sum as numpy."""
    nb = len(prompts[0])
    out = {k: [] for k in PATHS}
    out_no0 = {k: [] for k in PATHS}
    write, write_no0, size, size_write, rms = [], [], {k: [] for k in PATHS}, [], []
    for l in range(nb):
        rows = [p[l] for p in prompts]
        N = sum(r["n"] for r in rows)
        m = np.sum([r["mfree_sum"] for r in rows], 0) / N
        const = [r["n"] * float(m @ r["u"]) for r in rows]
        const0 = [(r["n"] - 1) * float(m @ r["u"]) for r in rows]
        c = {k: sum(r["sum"][k] for r in rows) for k in ("attn_sink", "attn_bias", "attn_rest", "mlp_bias")}
        c0 = {k: sum(r["sum_no0"][k] for r in rows) for k in c}
        c["mlp_const"], c0["mlp_const"] = sum(const), sum(const0)
        c["mlp_rest"] = sum(r["sum"]["mlp_free"] for r in rows) - c["mlp_const"]
        c0["mlp_rest"] = sum(r["sum_no0"]["mlp_free"] for r in rows) - c0["mlp_const"]
        R = float(np.sqrt(sum(r["sq"] for r in rows) / N))
        for k in PATHS:
            out[k].append(c[k])
            out_no0[k].append(c0[k])
            size[k].append(c[k] / N / R)
        write.append(sum(r["sum"]["write"] for r in rows))
        write_no0.append(sum(r["sum_no0"]["write"] for r in rows))
        size_write.append(write[-1] / N / R)
        rms.append(R)
    T, T0 = sum(write), sum(write_no0)
    rec = {
        "paths": list(PATHS),
        "per_block": {"component": out, "component_no0": out_no0, "write": write, "write_no0": write_no0,
                      "size": size, "size_write": size_write, "rms_norm": rms,
                      "share": {k: [out[k][l] / write[l] for l in range(nb)] for k in PATHS}},
        "write_total": T, "write_total_no0": T0,
        "share": {k: sum(out[k]) / T for k in PATHS},
        "share_no0": {k: sum(out_no0[k]) / T0 for k in PATHS},
        "size": {k: float(np.mean(size[k])) for k in PATHS},
        "size_write": float(np.mean(size_write)),
        "closure": abs(sum(sum(out[k]) for k in PATHS) - T) / abs(T),
        "err_sum": max(r["err_sum"] for p in prompts for r in p),
        "err_attn": max(r["err_attn"] for p in prompts for r in p),
        "err_modules": max(r["err_modules"] for p in prompts for r in p),
        "bias_norm_max": {k: max(r["bias_norm"][k] for p in prompts for r in p) for k in ("attn", "mlp")},
        "n_tokens": [p[0]["n"] for p in prompts], "n_blocks": nb,
    }
    if not all(np.isfinite(v) for v in [T, *rec["share"].values(), *rec["size"].values()]) or nb == 0:
        raise T5fRefused("record not populated")
    return rec


def plant_share(planted: dict, base: dict) -> dict:
    """The planted bias's gate: its bias path over the planted component (the planted model's
    summed write component less the unplanted one's)."""
    dT = planted["write_total"] - base["write_total"]
    b = sum(planted["per_block"]["component"]["attn_bias"])
    share = b / dT
    return {"bias_path": b, "planted_component": dT, "share": share, "bar": PLANT_MIN, "pass": share >= PLANT_MIN,
            "bias_share_of_write": planted["share"]["attn_bias"]}


# ---------------------------------------------------------------- commands

def _np(blocks_: List[dict]) -> List[dict]:
    return [{**b, "u": b["u"].cpu().numpy(), "mfree_sum": b["mfree_sum"].cpu().numpy()} for b in blocks_]


def read_model(model, tok, device) -> dict:
    from core.config import PROMPTS
    from trackC_writeup import c2_aniso as c
    prompts, rhos, mids = [], [], []
    for p in V1:
        enc = tok(PROMPTS[p], return_tensors="pt", truncation=True, max_length=c.MAX_TOKENS)
        r = split_prompt(model, enc, device)
        prompts.append(_np(r["blocks"]))
        rhos.append(r["rho"])
        mids.append(r["mid_norms"])
    rec = aggregate(prompts)
    rec["rho"] = np.mean(rhos, 0).tolist()
    rec["median_mid_norm"] = float(np.median(np.concatenate(mids)))
    return rec


def plant(model, b: np.ndarray) -> None:
    """Add ``b`` as the o_proj bias of every block (Dodecahedron's o_proj has none)."""
    import torch
    for _, A, _ in blocks(model):
        if A.o_proj.bias is not None:
            raise T5fRefused("o_proj already has a bias")
        A.o_proj.bias = torch.nn.Parameter(torch.as_tensor(b, dtype=A.o_proj.weight.dtype,
                                                           device=A.o_proj.weight.device), requires_grad=False)


def plant_vector(d: int, norm: float) -> np.ndarray:
    e = np.random.default_rng(PLANT_SEED).standard_normal(d)
    return norm * e / np.linalg.norm(e)


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(path)


def run(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if prompts_hash() != PROMPTS_HASH:
        raise T5fRefused(f"prompts_hash {prompts_hash()} != {PROMPTS_HASH}")
    from trackC_writeup import c2_aniso as c
    from trackT_timaeus import t2, t5
    torch, device = c._torch_setup()
    import transformers
    out = Path(a.out)
    meta = {"design": "trackT_timaeus/design-T5.md U3's split", "code": code_sha(), "prompts_hash": prompts_hash(),
            "prompts": list(V1), "max_tokens": t2.MAX_TOKENS, "dtype": "float32 forward, float64 split",
            "tf32": False, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "models": {k: list(v[1]) for k, v in MODELS.items()},
            "plant": {"step": PLANT_STEP, "scale": PLANT_SCALE, "seed": PLANT_SEED}}
    _write(out / "meta.json", meta)
    toks = {"neox": t2.p_tokenizer(), "llama": t2.tokenizer("32m")}
    t0 = time.time()
    for mk in a.models:
        repo, steps, tk = MODELS[mk]
        for s in (a.steps or steps):
            p = out / mk / f"step{s}.json"
            if p.exists() and not a.force:
                continue
            m = t5.load_real(repo, f"step{s}", a.hf_roots, device)
            rec = read_model(m, toks[tk], device)
            rec.update({"model": repo, "step": s, "commit": t2.commit_of(m), "code": meta["code"],
                        "wall_s": round(time.time() - t0, 1)})
            _write(p, rec)
            print(json.dumps({"wrote": str(p.relative_to(out)), "err_sum": f"{rec['err_sum']:.2e}",
                              "share": {k: round(v, 3) for k, v in rec["share"].items()}, "t": rec["wall_s"]}), flush=True)
            if mk == "32m" and s == PLANT_STEP:
                b = plant_vector(m.config.hidden_size, PLANT_SCALE * rec["median_mid_norm"])
                plant(m, b)
                pr = read_model(m, toks[tk], device)
                pr.update({"model": repo, "step": s, "planted_norm": float(np.linalg.norm(b)), "code": meta["code"],
                           "wall_s": round(time.time() - t0, 1)})
                _write(out / mk / f"step{s}_planted.json", pr)
                print(json.dumps({"wrote": f"{mk}/step{s}_planted.json", "gate": plant_share(pr, rec)}), flush=True)
            del m
            torch.cuda.empty_cache()
    print(json.dumps({"done": True, "wall_s": round(time.time() - t0, 1)}))
    return 0


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def gate(out: Path, known_dir: Path) -> dict:
    g = {}
    errs = {f"{mk}@{s}": _load(out / mk / f"step{s}.json")["err_sum"] for mk, (_, st, _) in MODELS.items() for s in st}
    g["parts_sum"] = {"max_rel_err": max(errs.values()), "bar": SUM_TOL, "pass": max(errs.values()) <= SUM_TOL}
    base = _load(out / "32m" / f"step{PLANT_STEP}.json")
    g["planted_bias"] = plant_share(_load(out / "32m" / f"step{PLANT_STEP}_planted.json"), base)
    diffs = {}
    for mk, steps in KNOWN.items():
        ref = _load(known_dir / f"diag_{mk}.json")["populations"]
        for s in steps:
            got = _load(out / mk / f"step{s}.json")["rho"]
            diffs[f"{mk}@{s}"] = float(np.max(np.abs(np.asarray(got) - np.asarray(ref[f"step{s}"]["mean_cos"]))))
    g["known_answer"] = {"max_diff": diffs, "bar": KNOWN_TOL, "pass": max(diffs.values()) <= KNOWN_TOL}
    g["pass"] = all(v["pass"] for v in g.values())
    return g


def first_read_beside(first: dict) -> Dict[int, dict]:
    """Each rise step's row of U3's first read (`t5e`'s reading.json)."""
    rows = {}
    for s in RISE:
        r = first["per_step"][str(s)]
        cen = r["centred"]
        rows[s] = {"centred_any_P": cen.get("any_trained_part") if "refused" not in cen else "refused",
                   "fixed_tau_A": r["fixed_tau"]["0.5"]["0.003"]["A_share"],
                   "uncentred_A": r["uncentred_A_share"], **{k: r["beside"][k] for k in r["beside"]}}
    return rows


def read(a) -> int:
    out = Path(a.out)
    g = gate(out, DATA / KNOWN_DIR)
    res: Dict[str, object] = {"design": "trackT_timaeus/design-T5.md U3's split", "gate": g}
    if not g["pass"]:
        res["refused"] = "a calibration gate failed; back to the user before any reading"
    else:
        first = _load(Path(a.first) / "reading.json")
        beside = first_read_beside(first)
        tab = {}
        for mk, (_, steps, _) in MODELS.items():
            for s in steps:
                r = _load(out / mk / f"step{s}.json")
                tab[f"{mk}@{s}"] = {"share": r["share"], "share_no0": r["share_no0"], "size": r["size"],
                                    "size_write": r["size_write"], "code": r["code"],
                                    "first_read": beside.get(s) if mk == "32m" else None}
        res.update({"per_step": tab, "where_each_count_stops": first["where_each_count_stops"]})
    _write(out / "reading.json", res)
    print(json.dumps(res["gate"], indent=1))
    if "per_step" in res:
        print(f"{'step':>14} " + " ".join(f"{k:>10}" for k in PATHS) + "   size_write")
        for key, r in res["per_step"].items():
            print(f"{key:>14} " + " ".join(f"{r['share'][k]:10.3f}" for k in PATHS) + f"   {r['size_write']:.4f}")
    return 0 if "refused" not in res else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="t5f")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    r.add_argument("--steps", nargs="+", type=int, default=None)
    r.add_argument("--hf-roots", nargs="+", default=list(DEFAULT_ROOTS))
    r.add_argument("--force", action="store_true")
    q = sub.add_parser("read")
    q.add_argument("--out", required=True)
    q.add_argument("--first", default=str(DATA / "trackT/t5u3/2026-11-02"))
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else read(a)


if __name__ == "__main__":
    raise SystemExit(main())
