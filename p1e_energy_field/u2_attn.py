"""
p1e_energy_field/u2_attn.py — U2's attention arm: which part of each block's update ascends the
field `φ_β` (`design-1e.md` "U2's attention arm: the rule", fixed before any output).

Pythia's residual is parallel, so block ℓ's update is exactly attention + MLP. One hooked forward
pass per (step, passage) on the GPU splits it into ``sink`` (attention to key 0), ``keys``
(attention to keys 1…i), ``mlpx`` (the MLP less its output bias) and ``bias`` (``b_O + W_O b_V +
b_mlp``, the same vector for every token); ``attn`` and ``mlp`` whole are read too, and
``block`` = attn + mlp checks the pass against `u2_block`'s stored records.

Each component ``c_i`` is added alone and read in the block arm's frame
(``u'_i = unit(LN1_ℓ(x_i + c_i))``), by `u2_block.cell`: frozen (causal at β 1.6 / 3.5 / 5.6 and the
β = 0 field ``mean0``), ``r1out`` (``c_i`` less its component along its own shared direction),
``resid`` (``mean_t c`` alone), ``residout`` (``c_i − mean_t c``). Beside: the block's shared
update split by component (``shares``), each component's sharedness, the attention to key 0.

Tier 1: exploratory, unregistered. GPU pass (float32, eager, TF32 off), cells in CUDA float64.
    python -m p1e_energy_field.u2_attn run --runs <p1e_long8 dir> --r0 <R0 labels> --out <dir>
    python -m p1e_energy_field.u2_attn report --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import u2_block as ub

#: Components read; ``block`` is the check row, ``bias`` a single vector for every token.
COMPONENTS = ("attn", "sink", "keys", "mlp", "mlpx", "bias", "block")
#: The parts that sum to the block's update.
PARTS = ("sink", "keys", "mlpx", "bias")
#: (field, β, reading) per component; ``bias`` and ``block`` read fewer (the rule's table).
READINGS = (("causal", 1.6, "frozen"), ("causal", 3.5, "frozen"), ("causal", 5.6, "frozen"),
            ("mean0", 0.0, "frozen"), ("causal", 3.5, "r1out"), ("mean0", 0.0, "r1out"),
            ("causal", 3.5, "resid"), ("causal", 3.5, "residout"))
ONLY = {"bias": (("causal", 3.5, "frozen"), ("mean0", 0.0, "frozen")),
        "block": (("causal", 3.5, "frozen"),)}
#: First checks (placed by the rule, not calibrated).
MATCH_TOL = {"long": 1e-5, "v1": 1e-4}
SPLIT_TOL = 1e-4
BLOCK_TOL = 1e-6


# ---------------------------------------------------------------- the hooked pass

def split_layer(attn: "torch.Tensor", mlp: "torch.Tensor", a0: "torch.Tensor", v0: "torch.Tensor",
                W_O: "torch.Tensor", b_O: "torch.Tensor", b_V: "torch.Tensor",
                b_mlp: "torch.Tensor") -> Dict:
    """
    One layer's update split by component. ``attn`` / ``mlp`` (n, d): the modules' outputs;
    ``a0`` (H, n): every head's attention to key 0; ``v0`` (H, hd): the value at position 0 with
    its bias; ``W_O`` (d, H·hd), ``b_O`` (d,); ``b_V`` (H, hd); ``b_mlp`` (d,).
    ``sink`` = Σ_h a0_h,i W_O^h (v0_h − b_V^h); the value bias reaches every token whole
    (attention rows sum to 1), so ``W_O b_V`` joins ``b_O`` and ``b_mlp`` in ``bias``.
    """
    H, n = a0.shape
    z = (a0[:, :, None] * (v0 - b_V)[:, None, :]).permute(1, 0, 2).reshape(n, -1)
    sink = z @ W_O.T
    c_attn = b_O + W_O @ b_V.reshape(-1)
    keys = attn - sink - c_attn
    return {"attn": attn, "sink": sink, "keys": keys, "mlp": mlp, "mlpx": mlp - b_mlp,
            "bias": (c_attn + b_mlp)[None, :].expand_as(attn), "block": attn + mlp}


class Hooked:
    """Hooks on every layer of a GPT-NeoX: attention and MLP outputs, attention to key 0, v at 0."""

    def __init__(self, model):
        self.model, self.cap, self.handles, self.orig = model, {}, [], {}
        self.layers = getattr(model, "gpt_neox", model).layers    # the LM or the bare model
        for l, layer in enumerate(self.layers):
            att = layer.attention
            if not getattr(model.config, "use_parallel_residual", False):
                raise ValueError("refusing: the split needs a parallel residual (attn + mlp)")
            self.orig[l] = att._attn

            def wrapped(q, k, v, attention_mask=None, head_mask=None, _l=l, _f=att._attn):
                out, w = _f(q, k, v, attention_mask, head_mask)
                self.cap[("a0", _l)] = w[0, :, :, 0].detach()
                self.cap[("v0", _l)] = v[0, :, 0, :].detach()
                return out, w
            att._attn = wrapped
            self.handles.append(att.register_forward_hook(
                lambda m, i, o, _l=l: self.cap.__setitem__(("attn", _l), o[0][0].detach())))
            self.handles.append(layer.mlp.register_forward_hook(
                lambda m, i, o, _l=l: self.cap.__setitem__(("mlp", _l), o[0].detach())))

    def close(self):
        for h in self.handles:
            h.remove()
        for l, layer in enumerate(self.layers):
            layer.attention._attn = self.orig[l]

    def run(self, ids: "torch.Tensor", layers: Sequence[int]) -> tuple:
        """``(hidden states (25, n, d) float32 on the CPU, {layer: {component: (n, d) float32}})``."""
        import torch
        self.cap.clear()
        with torch.no_grad():
            out = self.model(input_ids=ids, output_hidden_states=True, output_attentions=False,
                             use_cache=False)
            hs = torch.stack([h[0] for h in out.hidden_states]).float().cpu()
            comps = {}
            for l in layers:
                layer = self.layers[l]
                att = layer.attention
                H, hd = att.num_attention_heads, att.head_size
                b_V = att.query_key_value.bias.view(H, 3 * hd)[:, 2 * hd:]
                parts = split_layer(self.cap[("attn", l)], self.cap[("mlp", l)], self.cap[("a0", l)],
                                    self.cap[("v0", l)], att.dense.weight, att.dense.bias, b_V,
                                    layer.mlp.dense_4h_to_h.bias)
                comps[l] = {k: v.float().cpu().numpy() for k, v in parts.items()}
                comps[l]["a0_mean_heads"] = self.cap[("a0", l)].mean(dim=0).float().cpu().numpy()
        del out
        self.cap.clear()
        return hs.numpy(), comps


# ---------------------------------------------------------------- reading one pass

def shares(C: Dict[str, np.ndarray], t: np.ndarray) -> Dict:
    """The block's shared update ``c̄`` split by part, and each component's sharedness."""
    cbar = C["block"][t].mean(axis=0).astype(np.float64)
    nb = float(np.linalg.norm(cbar))
    chat = cbar / nb
    out = {"cbar_norm": nb,
           "share": {k: float(C[k][t].mean(axis=0).astype(np.float64) @ chat / nb) for k in PARTS},
           "sharedness": {k: float(np.linalg.norm(C[k][t].mean(axis=0)) /
                                   np.linalg.norm(C[k][t], axis=1).mean())
                          for k in COMPONENTS if k != "bias"},
           "a0_mean": float(C["a0_mean_heads"][t].mean())}
    return out


def moves(X: np.ndarray, c: np.ndarray, t: np.ndarray, frame) -> Dict[str, np.ndarray]:
    """The component's move under each reading, in β's frame (`u2_block.shared_cells`' maps)."""
    U = frame(X)
    cbar = c[t].mean(axis=0)
    out = {"frozen": ub.tangent(U, frame(X + c) - U),
           "resid": ub.tangent(U, frame(X + cbar) - U),
           "residout": ub.tangent(U, frame(X + c - cbar) - U)}
    nb = np.linalg.norm(cbar)
    if nb > 0:
        chat = cbar / nb
        out["r1out"] = ub.tangent(U, frame(X + c - np.outer(c @ chat, chat)) - U)
    return out


def read_pass(hs: np.ndarray, comps: Dict, ln: Dict, tgt: np.ndarray, key: str, ops,
              blocks: Sequence[int] = ub.BLOCKS) -> tuple:
    """Every cell of one pass (records as `u2_block`'s, ``source`` = field:component:reading)."""
    perms_for, cells, sh = ub.make_perms(key), [], {}
    for L in blocks:
        w, b, eps = ln["w"][L], ln["b"][L], ln["eps"]
        frame = lambda Y: ub.unit_rows(Y, w, b, eps)          # noqa: E731
        X = hs[L].astype(np.float64)
        U = frame(X)
        Ut = ops.to(U)
        S = Ut @ Ut.T
        f = {("causal", be): g for be in ub.BETAS
             for g in [ops.forces(Ut, be, S, only=("causal",))["causal"]]}
        f[("mean0", 0.0)] = ops.forces(Ut, 0.0, only=("mean0",))["mean0"]
        C = comps[L]
        sh[L] = shares(C, tgt)
        for comp in COMPONENTS:
            mv = moves(X, C[comp].astype(np.float64), tgt, frame)
            for field, be, rd in ONLY.get(comp, READINGS):
                if rd not in mv:
                    continue
                cells.append({"block": L, "beta": be, "source": f"{field}:{comp}:{rd}",
                              "targets": "t12" if key.endswith("_long") else "r0",
                              **ops.cell(ops.to(mv[rd]), f[(field, be)], Ut, tgt, perms_for)})
    return cells, sh


def check_pass(hs: np.ndarray, comps: Dict, run_dir: Path, kind: str) -> Dict:
    """The rule's first checks on one pass: hidden states against the stored run; the split adds up."""
    z = np.load(run_dir / "activations.npz")
    stored = z["activations"] * z["norms"][..., None]
    if stored.shape != hs.shape:
        raise SystemExit(f"refusing: {run_dir}: stored {stored.shape}, pass {hs.shape}")
    unit = lambda A: A / np.linalg.norm(A, axis=-1, keepdims=True)    # noqa: E731
    match = float(np.abs(unit(hs) - unit(stored)).max())
    if match > MATCH_TOL[kind]:
        raise SystemExit(f"refusing: {run_dir}: pass against stored unit rows {match:.1e} > {MATCH_TOL[kind]}")
    recon, split = 0.0, 0.0
    for L, C in comps.items():
        d = np.linalg.norm(hs[L + 1] - hs[L], axis=1).max()
        recon = max(recon, float(np.abs(hs[L] + C["block"] - hs[L + 1]).max() / d))
        split = max(split, float(np.abs(sum(C[k] for k in PARTS) - C["block"]).max() / d))
    if recon > SPLIT_TOL or split > SPLIT_TOL:
        raise SystemExit(f"refusing: {run_dir}: x + attn + mlp off by {recon:.1e}, parts off by {split:.1e}")
    return {"match_unit": match, "recon_rel": recon, "split_rel": split}


# ---------------------------------------------------------------- the batch

def ln_from(model) -> Dict:
    lay = getattr(model, "gpt_neox", model).layers
    return {"w": np.stack([l.input_layernorm.weight.detach().double().cpu().numpy() for l in lay]),
            "b": np.stack([l.input_layernorm.bias.detach().double().cpu().numpy() for l in lay]),
            "eps": float(lay[0].input_layernorm.eps)}


def token_ids(tok, run_dir: Path) -> List[int]:
    tokens = json.loads((run_dir / "geometry.json").read_text())["tokens"]
    ids = tok.convert_tokens_to_ids(tokens)
    if tok.convert_ids_to_tokens(ids) != tokens:
        raise SystemExit(f"refusing: {run_dir}: stored tokens do not round-trip through the tokenizer")
    return ids


def first_block_check(out: Path, rec: Dict, block_out: Path) -> float:
    """The ``block`` row against the block arm's stored frozen record (causal 3.5, ``t12``)."""
    ref = json.loads((block_out / "records" / "long" / f"step{rec['step']}_{rec['passage']}.json").read_text())
    want = {c["block"]: c["X"] for c in ref["cells"]
            if c["beta"] == 3.5 and c["source"] == "causal" and c["targets"] == "t12"}
    got = {c["block"]: c["X"] for c in rec["cells"] if c["source"] == "causal:block:frozen"}
    if set(want) != set(got):
        raise SystemExit("refusing: block rows do not cover the block arm's blocks")
    dev = max(abs(got[b] - want[b]) for b in want)
    if dev > BLOCK_TOL:
        raise SystemExit(f"refusing: block row's X differs from the block arm's by {dev:.1e} > {BLOCK_TOL}")
    return dev


def check_populated(rec: Dict) -> None:
    n = rec["targets"]
    bad = [c for c in rec["cells"] if c["n"] < 0.9 * n or not np.isfinite(c["X"])]
    seen = {(c["block"], c["source"].split(":")[1]) for c in rec["cells"]}
    want = {(b, k) for b in ub.BLOCKS for k in COMPONENTS}
    if bad or seen != want:
        raise SystemExit(f"refusing: first cell not populated ({len(bad)} bad cells, "
                         f"{len(want - seen)} (block, component) missing)")


def run(a) -> int:
    import torch
    from transformers import AutoTokenizer
    from core.models import load_model
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (the rule's pass is the GPU's)")
    code = ub.code_sha()
    jobs, _, meta = ub.plan(a.runs, a.r0, a.out / "_plan")
    jobs = [j[:5] + [j[5].get("t12", j[5].get("r0"))] for j in jobs]
    first = [j for j in jobs if j[0] == "long" and (j[1], j[2]) == ub.FIRST]
    rest = [j for j in jobs if j not in first]
    if a.steps:
        rest = [j for j in rest if j[1] in a.steps]
    meta.update(code=code, rule="design-1e.md \"U2's attention arm: the rule\"")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    ops = ub.get_ops("cuda")
    tok = AutoTokenizer.from_pretrained(ub.REPO, revision="step143000")
    order = first + ([] if a.first_only else sorted(rest, key=lambda j: int(j[1])))
    loaded, model, hk = None, None, None
    for i, (kind, step, key, rd, rev, tg) in enumerate(order):
        path = a.out / "records" / kind / f"step{step}_{key}.json"
        if path.exists():
            had = json.loads(path.read_text()).get("code")
            if had != code:
                raise SystemExit(f"refusing to resume: {path} was written by {had}, this is {code}")
            print(f"have {path.name}", flush=True)
            continue
        if loaded != step:
            if hk is not None:              # the hooks hold the model: drop both before loading
                hk.close()
            hk = model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            model, _ = load_model(f"pythia-410m-step{step}")
            if next(model.parameters()).dtype != torch.float32:
                raise SystemExit("refusing: model is not float32")
            man_rev = json.loads((Path(rd) / "manifest.json").read_text())["hf_revision"]
            if man_rev != rev:
                raise SystemExit(f"refusing: {rd} manifest revision {man_rev}, plan {rev}")
            hk, ln, loaded = Hooked(model), ln_from(model), step
        t0 = time.monotonic()
        ids = torch.tensor([token_ids(tok, Path(rd))], device=next(model.parameters()).device)
        hs, comps = hk.run(ids, ub.BLOCKS)
        chk = check_pass(hs, comps, Path(rd), kind)
        t = np.asarray(tg, dtype=int)
        cells, sh = read_pass(hs, comps, ln, t, key, ops)
        rec = {"kind": kind, "step": step, "passage": key, "run": rd, "code": code, "device": ops.name,
               "pass": "cuda:float32", "targets": int(t.size), "checks": chk,
               "shares": {str(k): v for k, v in sh.items()}, "cells": cells}
        if i == 0 and (kind, step, key) == ("long",) + ub.FIRST:
            check_populated(rec)
            rec["checks"]["block_vs_block_arm"] = first_block_check(a.out, rec, a.block_out)
            print(f"first cell populated; checks {rec['checks']}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec) + "\n")
        tmp.rename(path)
        print(f"done {path.name} {time.monotonic() - t0:.0f}s match {chk['match_unit']:.1e}", flush=True)
    if hk is not None:
        hk.close()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", type=Path, required=True)
    r.add_argument("--r0", type=Path, default=None, help="R0's labels dir (v1 beside)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--block-out", type=Path, required=True, help="the block arm's output (first check)")
    r.add_argument("--steps", nargs="*", default=None)
    r.add_argument("--first-only", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        from .u2_attn_report import report
        return report(a.out)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
