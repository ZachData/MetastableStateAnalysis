"""
trackT_timaeus/t5g.py — T5 U4s: the screen at init (`design-T5.md` "U4s, the screen at init", D6 (b),
the user's; rule and implementation fixed before any output).

Extends U1b (`t5b.py`, imported, its code hash left alone): the same base (Llama-32m without biases),
swaps, seeds (20), inits (P, F), statistic (ρ_mid, per-cell median), E0, τ and bar = max(0.1, 2τ).
Cells (79): every subset of the six swaps (64; ∅ the base, the six-set U1b's `all`); `untied` (τ);
`gelu` split into `gate` (ungated), `act` (GeLU), `width` (1536), alone, in pairs and all three;
the reverse ablation, U1b's `all` with one Llama part put back (`rev-<part>`) and with all six
(`rev-all6`). Each model is read as U1 reads it (`t5.read_model`).

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t5g run --out <dir>
    METS_DATA=<main>/data python -m trackT_timaeus.t5g read --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, prompts_hash
from trackC_writeup import c2_aniso as c
from trackT_timaeus import t2, t5, t5b

PARTS, NULL, ALL = t5b.PARTS, t5b.NULL, t5b.ALL
SPLIT = ("gate", "act", "width")
SPLIT_ALL = "gate+act+width"
REV_ALL = "rev-all6"
GELU_WIDTH = 4 * t5.SHAPES[t5b.SHAPE][0]          # 1536
U1B_DIR = "trackT/t5u1b/2026-10-30"
# the gate (design-T5.md "U4s")
KNOWN_TOL = 1e-6
SD_TOL = 0.02
BUILT_TOL = 0.05                                  # the split's three together vs `gelu`; reverse all-six vs base
ROLES = ("emb", "qkv", "attn_out", "mlp_in", "mlp_out")


class T5gRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t5g.py", "trackT_timaeus/t5b.py", "trackT_timaeus/t5.py", "trackT_timaeus/t2.py",
              "trackC_writeup/c2_aniso.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the cells (pure)

def subsets() -> List[Tuple[str, ...]]:
    """All 64 subsets of the six swaps, by size, then in PARTS order."""
    return [S for n in range(len(PARTS) + 1) for S in itertools.combinations(PARTS, n)]


def subset_name(S: Sequence[str]) -> Optional[str]:
    """∅ is the base (None); the six-set is U1b's `all`; else the parts joined by '+' in PARTS order."""
    S = tuple(p for p in PARTS if p in S)
    return None if not S else ALL if len(S) == len(PARTS) else "+".join(S)


def split_names() -> List[str]:
    return ["+".join(S) for n in (1, 2, 3) for S in itertools.combinations(SPLIT, n)]


def rev_names() -> List[str]:
    return [f"rev-{p}" for p in PARTS] + [REV_ALL]


def names() -> List[Optional[str]]:
    return [subset_name(S) for S in subsets()] + [NULL] + split_names() + rev_names()


def parts_of(name: Optional[str]) -> Tuple[str, ...]:
    """The swaps a cell's model carries. `all` and every reverse cell carry `untied`, as U1b's `all`."""
    if name is None:
        return ()
    if name == ALL:
        return t5b.parts_of(ALL)
    if name == REV_ALL:
        return (NULL,)
    if name.startswith("rev-"):
        k = name[4:]
        if k not in PARTS:
            raise T5gRefused(f"{name}: no such part")
        return tuple(p for p in PARTS if p != k) + (NULL,)
    ps = tuple(name.split("+"))
    if any(p not in PARTS + (NULL,) + SPLIT for p in ps) or ("gelu" in ps and set(ps) & set(SPLIT)):
        raise T5gRefused(f"{name}: not a cell")
    return ps


def cells() -> List[Tuple[str, Optional[str]]]:
    return [(i, n) for n in names() for i in t5.INITS]


def cell_key(init: str, name: Optional[str]) -> str:
    return t5b.cell_key(init, name)


# ---------------------------------------------------------------- the rule (pure)

def rev_label(g: float, e0: float, bar: float) -> str:
    if g >= e0 / 2:
        return "suffices"
    if g <= -bar:
        return "opposes"
    if abs(g) < bar:
        return "does not"
    return "partly"


def minimal_carrying(lab: Dict[Optional[str], str]) -> List[str]:
    """Subsets labelled "carries" with no carrying proper subset (non-empty subsets only)."""
    carry = [S for S in subsets()[1:] if lab[subset_name(S)] == "carries"]
    return [subset_name(S) for S in carry if not any(set(T) < set(S) for T in carry)]


def _scale_view(v: Dict[Tuple[str, Optional[str]], float]) -> dict:
    """E0, ΔE per cell (as a share of E0), pair interactions and the mean ΔE by subset size, on one scale."""
    e0 = v[("F", None)] - v[("P", None)]
    dE = {n: e0 - (v[("F", n)] - v[("P", n)]) for n in names()}
    pairs = {f"{j}+{k}": dE[subset_name((j, k))] - dE[j] - dE[k] for j, k in itertools.combinations(PARTS, 2)}
    size = {n: float(np.mean([dE[subset_name(S)] for S in subsets() if len(S) == n])) for n in range(1, 7)}
    return {"E0": e0, "dE": {str(n): dE[n] for n in names()}, "share": {str(n): dE[n] / e0 for n in names()},
            "pair_interaction": pairs, "mean_dE_by_size": size}


def u4s_reading(med: Dict[Tuple[str, Optional[str]], float],
                beside_scales: Optional[Dict[str, Dict[Tuple[str, Optional[str]], float]]] = None) -> dict:
    """The rule (design-T5.md "U4s") from the cell medians of ρ_mid, ``med[(init, name)]``."""
    B = {i: med[(i, None)] for i in t5.INITS}
    e0 = B["F"] - B["P"]
    M_null = {i: med[(i, NULL)] - B[i] for i in t5.INITS}
    tau = max(t5.TAU_MIN, *(abs(v) for v in M_null.values()))
    bar = max(t5.E_MIN, 2 * tau)
    E = {n: med[("F", n)] - med[("P", n)] for n in names()}
    dE = {n: e0 - E[n] for n in names()}
    screen = [subset_name(S) for S in subsets()[1:]]
    lab = {n: t5b.label_of(dE[n], e0, bar) for n in screen}
    mins = minimal_carrying(lab)
    reading = ("only all six together" if mins == [ALL] else "none carries" if not mins
               else "minimal carrying sets: " + "; ".join(mins))
    split = {n: t5b.label_of(dE[n], e0, bar) for n in split_names() if n != SPLIT_ALL}
    G = {n: E[n] - E[ALL] for n in rev_names()}
    rev = {n: rev_label(G[n], e0, bar) for n in rev_names() if n != REV_ALL}
    suff = [n[4:] for n, l in rev.items() if l == "suffices"]
    out = {"B": B, "E0": e0, "M_untied": M_null, "tau": tau, "bar": bar,
           "E": {str(n): E[n] for n in names()}, "dE": {str(n): dE[n] for n in names()},
           "screen_label": lab, "minimal_carrying_sets": mins, "reading": reading,
           "split_label": split, "split_all_three_dE_beside": dE[SPLIT_ALL],
           "reverse_G": G, "reverse_label": rev,
           "reverse_reading": ("suffices alone: " + ", ".join(suff)) if suff else "no single part suffices",
           "beside": {"rho": _scale_view(med)}}
    for k, v in (beside_scales or {}).items():
        out["beside"][k] = _scale_view(v)
    out["beside"]["screen_vs_reverse_draws"] = {
        p: {"screen_E": E[subset_name(tuple(q for q in PARTS if q != p))], "reverse_E": E[f"rev-{p}"]}
        for p in PARTS}
    return out


def logit(x: float) -> float:
    return math.log(x / (1 - x))


# ---------------------------------------------------------------- the split's MLP and the build

def _split_mlp(d: int, width: int, gated: bool, act: str):
    import torch.nn as nn

    class SplitMLP(nn.Module):
        def __init__(self):
            super().__init__()
            if gated:
                self.gate_proj = nn.Linear(d, width, bias=False)
            self.up_proj = nn.Linear(d, width, bias=False)
            self.down_proj = nn.Linear(width, d, bias=False)
            self.act_fn = nn.GELU() if act == "gelu" else nn.SiLU()
            self.t5g = {"gated": gated, "act": act, "width": width}

        def forward(self, x):
            h = self.act_fn(self.gate_proj(x)) * self.up_proj(x) if gated else self.act_fn(self.up_proj(x))
            return self.down_proj(h)
    return SplitMLP()


def cfg_edit(parts: Sequence[str]):
    return t5b.cfg_edit(parts)


def model_edit(parts: Sequence[str]):
    base = t5b.model_edit(parts)
    sp = [p for p in parts if p in SPLIT]

    def f(m):
        base(m)
        if sp:
            d = m.config.hidden_size
            for layer in m.model.layers:
                layer.mlp = _split_mlp(d, GELU_WIDTH if "width" in sp else m.config.intermediate_size,
                                       "gate" not in sp, "gelu" if "act" in sp else "silu")
    return f


def check_cell(m, parts: Sequence[str]) -> None:
    """U1b's structure check, and the split's MLP as its parts say."""
    t5b.check_swaps(m, parts)
    sp = set(parts) & set(SPLIT)
    mlp = m.model.layers[0].mlp
    got = getattr(mlp, "t5g", None)
    if not sp:
        if got is not None:
            raise T5gRefused(f"{list(parts)}: a split MLP where none was asked")
        return
    want = {"gated": "gate" not in sp, "act": "gelu" if "act" in sp else "silu",
            "width": GELU_WIDTH if "width" in sp else m.config.intermediate_size}
    if got != want:
        raise T5gRefused(f"{list(parts)}: split MLP {got} != {want}")


# ---------------------------------------------------------------- commands

def run(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if prompts_hash() != t5.PROMPTS_HASH:
        raise T5gRefused(f"prompts_hash {prompts_hash()} != {t5.PROMPTS_HASH}")
    torch, device = c._torch_setup()
    import transformers
    from transformers import AutoConfig
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"design": "trackT_timaeus/design-T5.md U4s", "code": code_sha(), "prompts_hash": prompts_hash(),
            "max_tokens": t2.MAX_TOKENS, "dtype": "float32", "tf32": False, "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "transformers": transformers.__version__, "seed": t5.SEED,
            "n_seeds": t5.N_SEEDS, "shape": t5.SHAPES[t5b.SHAPE], "cells": [str(n) for n in names()]}
    t5._write(out / "meta.json", meta)
    toks = {"neox": t2.p_tokenizer(), "llama": t2.tokenizer("32m")}
    repo = t5.KNOWN["32m"][0]
    dodeca_cfg = AutoConfig.from_pretrained(repo, revision="step1", cache_dir=t5.locate(repo, "step1", a.hf_roots))
    t0 = time.time()
    for init, name in cells():
        key, parts = cell_key(init, name), parts_of(name)
        for s in range(t5.N_SEEDS):
            p = out / "cells" / key / f"s{s:02d}.json"
            if p.exists() and not a.force:
                continue
            m = t5.build(t5b.SHAPE, "llama", init, False, t5.seed_of(key, s), dodeca_cfg, device,
                         cfg_edit=cfg_edit(parts), model_edit=model_edit(parts))
            check_cell(m, parts)
            rec = t5.read_model(m, toks["neox" if "tok" in parts else "llama"], device)
            rec.update({"cell": key, "init": init, "name": name, "parts": list(parts), "seed_index": s,
                        "seed": t5.seed_of(key, s), "sds": t5.class_sds(m) if s == 0 else None,
                        "code": meta["code"], "wall_s": round(time.time() - t0, 1)})
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
        raise T5gRefused(f"{d}: {len(rs)} of {t5.N_SEEDS} seeds")
    return rs


def _med(rs: List[dict], f=lambda x: x) -> float:
    return float(np.median([f(r["rho_mid"]) for r in rs]))


def gate(recs: Dict[Tuple[str, Optional[str]], List[dict]], u1b: Path) -> dict:
    g = {}
    diffs, seeds_ok, n = [], True, 0
    for i in t5.INITS:
        for name in (None,) + PARTS + (NULL, ALL):
            ref = _cell(u1b / "cells" / cell_key(i, name))
            for r, q in zip(recs[(i, name)], ref):
                diffs.append(float(np.max(np.abs(np.asarray(r["rho"]) - np.asarray(q["rho"])))))
                seeds_ok &= r["seed"] == q["seed"]
                n += 1
    g["u1b_reproduced"] = {"max_diff": max(diffs), "n_models": n, "same_seeds": bool(seeds_ok),
                           "pass": max(diffs) <= KNOWN_TOL and seeds_ok}
    med = {k: _med(v) for k, v in recs.items()}
    sp = {i: med[(i, SPLIT_ALL)] - med[(i, "gelu")] for i in t5.INITS}
    g["split_built_together"] = {"diff": sp, "pass": max(abs(v) for v in sp.values()) <= BUILT_TOL}
    rv = {i: med[(i, REV_ALL)] - med[(i, None)] for i in t5.INITS}
    g["reverse_all_six"] = {"diff": rv, "pass": max(abs(v) for v in rv.values()) <= BUILT_TOL}
    d, L = t5.SHAPES[t5b.SHAPE][:2]
    sd = {}
    for (i, name), rs in recs.items():
        got = rs[0]["sds"]
        sd[cell_key(i, name)] = max(abs(got[k] / t5.init_sd(k, i, d, L) - 1) for k in ROLES)
    g["init_sds"] = {"max_rel": max(sd.values()), "worst_cell": max(sd, key=sd.get), "pass": max(sd.values()) <= SD_TOL}
    g["pass"] = all(v["pass"] for v in g.values() if isinstance(v, dict))
    return g


def read(a) -> int:
    out = Path(a.out)
    recs = {(i, n): _cell(out / "cells" / cell_key(i, n)) for i, n in cells()}
    codes = {r["code"] for v in recs.values() for r in v}
    if len(codes) != 1:
        raise T5gRefused(f"records from more than one code version: {codes}")
    g = gate(recs, Path(a.u1b))
    res = {"code": codes.pop(), "reader": code_sha(), "n_cells": len(recs),
           "n_models": sum(len(v) for v in recs.values()), "gate": g}
    if g["pass"]:
        med = {k: _med(v) for k, v in recs.items()}
        scales = {"logit": {k: _med(v, logit) for k, v in recs.items()},
                  "atanh": {k: _med(v, math.atanh) for k, v in recs.items()}}
        res["reading"] = u4s_reading(med, scales)
        res["cells"] = {cell_key(*k): {"median_rho_mid": med[k],
                                       "range": [min(r["rho_mid"] for r in v), max(r["rho_mid"] for r in v)],
                                       "median_rho_15": float(np.median([r["rho_15"] for r in v])),
                                       "median_rho_mid_no0": float(np.median([r["rho_mid_no0"] for r in v]))}
                        for k, v in recs.items()}
    else:
        res["refused"] = "the gate failed: no reading (design-T5.md 'U4s'; back to the user)"
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    show = {"gate": {k: (v if isinstance(v, bool) else {kk: vv for kk, vv in v.items() if kk != "pass"} | {"pass": v["pass"]})
                     for k, v in g.items()}}
    if g["pass"]:
        rd = res["reading"]
        show.update({k: rd[k] for k in ("E0", "tau", "bar", "reading", "minimal_carrying_sets", "split_label",
                                        "reverse_label", "reverse_reading")})
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
    sp.add_argument("--u1b", default=str(DATA / U1B_DIR))
    a = ap.parse_args(argv)
    return {"run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
