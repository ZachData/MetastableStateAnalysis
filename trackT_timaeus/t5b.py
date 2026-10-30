"""
trackT_timaeus/t5b.py — T5 U1b: which part of the Llama makes the init matter (`design-T5.md` "U1b",
added by Blocked 124 (a); rule and gate fixed before any U1b output).

Base: U1's cell Llama-32m without biases, under P and F, rebuilt through `t5.build` with U1's seeds
(so its records must equal U1's). Swaps, one at a time to GPT-NeoX's, each under both inits, 20
seeds: `ln` LayerNorm, `par` parallel residual, `gelu` GeLU MLP at 4d, `rot` rotary on 0.25 of each
head, `mha` 6 KV heads, `tok` GPT-NeoX's tokenizer; `untied` the planted null; `all` every swap,
which is GPT-NeoX-32m without biases in distribution. Each model is read as U1 reads it
(`t5.read_model`: the 7 v1 prompts, ≤ 512 tokens, float32, eager, GPU).

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t5b run --out <dir>
    METS_DATA=<main>/data python -m trackT_timaeus.t5b read --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, prompts_hash
from trackC_writeup import c2_aniso as c
from trackT_timaeus import t2, t5

SHAPE = "32m"
PARTS = ("ln", "par", "gelu", "rot", "mha", "tok")
NULL = "untied"
ALL = "all"
SWAPS = PARTS + (NULL, ALL)
NEOX_ROT_PCT, NEOX_ROT_BASE, NEOX_VOCAB, LN_EPS = 0.25, 10000.0, 50304, 1e-5
U1_DIR = "trackT/t5u1/2026-10-30"
# the gate (design-T5.md "U1b")
KNOWN_TOL = 1e-6
SD_TOL = 0.02
NULL_MAX = 0.05
NEOX_MAX = 0.05


class T5bRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t5b.py", "trackT_timaeus/t5.py", "trackT_timaeus/t2.py", "trackC_writeup/c2_aniso.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the cells and the rule (pure)

def cell_key(init: str, swap: Optional[str]) -> str:
    """The base is U1's key (so U1's seeds); a swap appends ``+<swap>``."""
    base = t5.cell_key(SHAPE, "llama", init, False)
    return base if swap is None else f"{base}+{swap}"


def cells() -> List[Tuple[str, Optional[str]]]:
    return [(i, s) for s in (None,) + SWAPS for i in t5.INITS]


def parts_of(swap: Optional[str]) -> Tuple[str, ...]:
    return () if swap is None else PARTS + (NULL,) if swap == ALL else (swap,)


def partial_inv_freq(head_dim: int, pct: float = NEOX_ROT_PCT, base: float = NEOX_ROT_BASE) -> np.ndarray:
    """Llama's inv_freq (one per pair (i, i + head_dim/2)) giving NeoX's partial rotary up to a
    permutation of head dimensions: the first rot/2 pairs at NeoX's frequencies base^(−2j/rot), the
    rest at 0 (cos 1, sin 0: not rotated)."""
    rot = int(head_dim * pct)
    f = np.zeros(head_dim // 2)
    f[: rot // 2] = 1.0 / base ** (np.arange(0, rot, 2) / rot)
    return f


def neox_to_llama_dims(head_dim: int, pct: float = NEOX_ROT_PCT) -> np.ndarray:
    """``src[j]``: the NeoX head dimension that Llama dimension j stands in for under `partial_inv_freq`."""
    rot, half = int(head_dim * pct), head_dim // 2
    src = np.full(head_dim, -1)
    src[: rot // 2] = np.arange(rot // 2)                       # pair partners i ↔ i + rot/2 in NeoX
    src[half: half + rot // 2] = rot // 2 + np.arange(rot // 2)
    src[src < 0] = np.arange(rot, head_dim)                      # NeoX's pass-through dims
    return src


def label_of(de: float, e0: float, bar: float) -> str:
    if de >= e0 / 2:
        return "carries"
    if de <= -bar:
        return "amplifies"
    if abs(de) < bar:
        return "does not"
    return "partly"


def u1b_reading(med: Dict[Tuple[str, Optional[str]], float]) -> dict:
    """The rule (design-T5.md "U1b") from the cell medians of ρ_mid, ``med[(init, swap)]``."""
    B = {i: med[(i, None)] for i in t5.INITS}
    e0 = B["F"] - B["P"]
    M = {s: {i: med[(i, s)] - B[i] for i in t5.INITS} for s in SWAPS}
    tau = max(t5.TAU_MIN, *(abs(v) for v in M[NULL].values()))
    bar = max(t5.E_MIN, 2 * tau)
    E = {s: med[("F", s)] - med[("P", s)] for s in SWAPS}
    dE = {s: e0 - E[s] for s in SWAPS}
    lab = {s: label_of(dE[s], e0, bar) for s in SWAPS}
    moves = {s: (M[s]["P"] * M[s]["F"] > 0 and min(abs(M[s]["P"]), abs(M[s]["F"])) >= bar) for s in SWAPS}
    carry = [s for s in PARTS if lab[s] == "carries"]
    return {"B": B, "E0": e0, "tau": tau, "bar": bar, "M": M, "E": E, "dE": dE, "label": lab,
            "moves_at_both_inits_beside": moves, "carries": carry,
            "reading": ("carried by " + ", ".join(carry)) if carry else "no single part",
            "additivity_beside": {"sum_dE_parts": float(sum(dE[s] for s in PARTS)), "dE_all": dE[ALL]}}


# ---------------------------------------------------------------- the swaps

def _gelu_mlp(d: int):
    """GPT-NeoX's MLP without biases (torch imported at run time)."""
    import torch.nn as nn

    class NeoXMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_h_to_4h = nn.Linear(d, 4 * d, bias=False)
            self.dense_4h_to_h = nn.Linear(4 * d, d, bias=False)
            self.act = nn.GELU()                 # exact, as NeoX's "gelu"

        def forward(self, x):
            return self.dense_4h_to_h(self.act(self.dense_h_to_4h(x)))
    return NeoXMLP()


def _par_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                 output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kw):
    """GPT-NeoX's parallel residual: x + attn(ln₁ x) + mlp(ln₂ x)."""
    attn = self.self_attn(hidden_states=self.input_layernorm(hidden_states), attention_mask=attention_mask,
                          position_ids=position_ids, past_key_value=None, output_attentions=False,
                          use_cache=False, cache_position=cache_position, position_embeddings=position_embeddings)[0]
    return (hidden_states + attn + self.mlp(self.post_attention_layernorm(hidden_states)),)


def set_partial_rotary(model) -> None:
    import torch
    hd = model.config.hidden_size // model.config.num_attention_heads
    mods = [model.model.rotary_emb] + [l.self_attn.rotary_emb for l in model.model.layers
                                       if hasattr(l.self_attn, "rotary_emb")]
    for r in mods:
        f = torch.tensor(partial_inv_freq(hd), dtype=r.inv_freq.dtype, device=r.inv_freq.device)
        r.register_buffer("inv_freq", f, persistent=False)
        r.original_inv_freq = f


def cfg_edit(parts: Sequence[str]):
    def f(cfg):
        if "mha" in parts:
            cfg.num_key_value_heads = cfg.num_attention_heads
        if "untied" in parts:
            cfg.tie_word_embeddings = False
        if "tok" in parts:
            cfg.vocab_size = NEOX_VOCAB
    return f


def model_edit(parts: Sequence[str]):
    def f(m):
        import torch.nn as nn
        d = m.config.hidden_size
        for layer in m.model.layers:
            if "ln" in parts:
                layer.input_layernorm = nn.LayerNorm(d, eps=LN_EPS)
                layer.post_attention_layernorm = nn.LayerNorm(d, eps=LN_EPS)
            if "gelu" in parts:
                layer.mlp = _gelu_mlp(d)
            if "par" in parts:
                layer.forward = types.MethodType(_par_forward, layer)
        if "ln" in parts:
            m.model.norm = nn.LayerNorm(d, eps=LN_EPS)
        if "rot" in parts:
            set_partial_rotary(m)
    return f


def check_swaps(m, parts: Sequence[str]) -> None:
    """Refuse a model whose structure is not what its swaps say."""
    import torch.nn as nn
    cfg, layer = m.config, m.model.layers[0]
    want = {"ln": isinstance(m.model.norm, nn.LayerNorm) and isinstance(layer.input_layernorm, nn.LayerNorm),
            "gelu": hasattr(layer.mlp, "dense_h_to_4h"),
            "par": getattr(layer.forward, "__func__", None) is _par_forward,
            "rot": int((m.model.rotary_emb.inv_freq > 0).sum()) == int(cfg.hidden_size // cfg.num_attention_heads
                                                                      * NEOX_ROT_PCT) // 2,
            "mha": cfg.num_key_value_heads == cfg.num_attention_heads,
            "tok": cfg.vocab_size == NEOX_VOCAB,
            "untied": m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr()}
    bad = [k for k, v in want.items() if v != (k in parts)]
    if bad:
        raise T5bRefused(f"swaps {list(parts)}: structure disagrees on {bad}")


# ---------------------------------------------------------------- commands

def run(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if prompts_hash() != t5.PROMPTS_HASH:
        raise T5bRefused(f"prompts_hash {prompts_hash()} != {t5.PROMPTS_HASH}")
    torch, device = c._torch_setup()
    import transformers
    from transformers import AutoConfig
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"design": "trackT_timaeus/design-T5.md U1b", "code": code_sha(), "prompts_hash": prompts_hash(),
            "max_tokens": t2.MAX_TOKENS, "dtype": "float32", "tf32": False, "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "transformers": transformers.__version__, "seed": t5.SEED,
            "n_seeds": t5.N_SEEDS, "shape": t5.SHAPES[SHAPE], "swaps": SWAPS}
    t5._write(out / "meta.json", meta)
    toks = {"neox": t2.p_tokenizer(), "llama": t2.tokenizer("32m")}
    repo = t5.KNOWN["32m"][0]
    dodeca_cfg = AutoConfig.from_pretrained(repo, revision="step1", cache_dir=t5.locate(repo, "step1", a.hf_roots))
    t0 = time.time()
    for init, swap in cells():
        key, parts = cell_key(init, swap), parts_of(swap)
        for s in range(t5.N_SEEDS):
            p = out / "cells" / key / f"s{s:02d}.json"
            if p.exists() and not a.force:
                continue
            m = t5.build(SHAPE, "llama", init, False, t5.seed_of(key, s), dodeca_cfg, device,
                         cfg_edit=cfg_edit(parts), model_edit=model_edit(parts))
            check_swaps(m, parts)
            rec = t5.read_model(m, toks["neox" if "tok" in parts else "llama"], device)
            rec.update({"cell": key, "init": init, "swap": swap, "seed_index": s, "seed": t5.seed_of(key, s),
                        "sds": t5.class_sds(m) if s == 0 else None, "code": meta["code"],
                        "wall_s": round(time.time() - t0, 1)})
            t5._write(p, rec)
            print(json.dumps({"wrote": str(p.relative_to(out)), "rho_mid": round(rec["rho_mid"], 4),
                              "t": rec["wall_s"]}), flush=True)
            del m
            torch.cuda.empty_cache()
    print(json.dumps({"done": True, "wall_s": round(time.time() - t0, 1)}))
    return 0


def _cell(d: Path) -> List[dict]:
    rs = [t5._load(p) for p in sorted(d.glob("s*.json"))]
    if len(rs) != t5.N_SEEDS:
        raise T5bRefused(f"{d}: {len(rs)} of {t5.N_SEEDS} seeds")
    return rs


def gate(recs: Dict[Tuple[str, Optional[str]], List[dict]], u1: Path) -> dict:
    g = {}
    diffs = []
    for i in t5.INITS:
        ref = _cell(u1 / "factorial" / cell_key(i, None))
        diffs += [float(np.max(np.abs(np.asarray(r["rho"]) - np.asarray(q["rho"])))) for r, q in zip(recs[(i, None)], ref)]
    g["known_answer"] = {"max_diff": max(diffs), "n": len(diffs), "pass": max(diffs) <= KNOWN_TOL}
    d, L = t5.SHAPES[SHAPE][:2]
    sd = {}
    for (i, s), rs in recs.items():
        got = rs[0]["sds"]
        rel = {k: got[k] / t5.init_sd(k, i, d, L) - 1 for k in ("emb", "qkv", "attn_out", "mlp_in", "mlp_out")}
        sd[cell_key(i, s)] = max(abs(v) for v in rel.values())
    g["init_sds"] = {"max_rel_by_cell": sd, "pass": max(sd.values()) <= SD_TOL}
    med = {k: float(np.median([r["rho_mid"] for r in v])) for k, v in recs.items()}
    nul = {i: med[(i, NULL)] - med[(i, None)] for i in t5.INITS}
    g["planted_null"] = {"M_untied": nul, "pass": max(abs(v) for v in nul.values()) <= NULL_MAX}
    neox = {i: float(np.median([r["rho_mid"] for r in _cell(u1 / "factorial" / t5.cell_key(SHAPE, "neox", i, False))]))
            for i in t5.INITS}
    gap = {i: med[(i, ALL)] - neox[i] for i in t5.INITS}
    g["all_is_neox"] = {"all": {i: med[(i, ALL)] for i in t5.INITS}, "u1_neox_nob": neox, "diff": gap,
                        "pass": max(abs(v) for v in gap.values()) <= NEOX_MAX}
    r = u1b_reading(med)
    g["positive_control"] = {"dE_all": r["dE"][ALL], "E0": r["E0"], "label_all": r["label"][ALL],
                             "pass": r["label"][ALL] == "carries"}
    g["pass"] = all(v["pass"] for v in g.values() if isinstance(v, dict))
    return g


def read(a) -> int:
    out = Path(a.out)
    recs = {(i, s): _cell(out / "cells" / cell_key(i, s)) for i, s in cells()}
    codes = {r["code"] for v in recs.values() for r in v}
    if len(codes) != 1:
        raise T5bRefused(f"records from more than one code version: {codes}")
    g = gate(recs, Path(a.u1))
    res = {"code": codes.pop(), "reader": code_sha(), "gate": g}
    if g["pass"]:
        med = {k: float(np.median([r["rho_mid"] for r in v])) for k, v in recs.items()}
        res["reading"] = u1b_reading(med)
        res["cells"] = {cell_key(*k): {"median_rho_mid": med[k],
                                       "range": [min(r["rho_mid"] for r in v), max(r["rho_mid"] for r in v)],
                                       "median_rho_15": float(np.median([r["rho_15"] for r in v])),
                                       "median_rho_mid_no0": float(np.median([r["rho_mid_no0"] for r in v]))}
                        for k, v in recs.items()}
    else:
        res["refused"] = "the gate failed: no reading (design-T5.md 'U1b'; back to the user)"
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    show = {"gate": {k: (v["pass"] if isinstance(v, dict) else v) for k, v in g.items()}}
    if g["pass"]:
        rd = res["reading"]
        show.update({k: rd[k] for k in ("E0", "tau", "bar", "dE", "label", "moves_at_both_inits_beside", "reading",
                                        "additivity_beside")})
    print(json.dumps(show, indent=1))
    return 0 if g["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("run")
    sp.add_argument("--out", required=True)
    sp.add_argument("--hf-roots", nargs="+", default=list(t5.DEFAULT_ROOTS))
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("read")
    sp.add_argument("--out", required=True)
    sp.add_argument("--u1", default=str(DATA / U1_DIR))
    a = ap.parse_args(argv)
    return {"run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
