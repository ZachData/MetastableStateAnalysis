"""
p1e_energy_field/u2_heads.py — U2's per-head arm: attention's move against each head's own kernel
(`design-1e.md` "U2's per-head arm: the rule", fixed before any output).

The attention arm read attention's token-specific move against `φ_β` (one head, ``Q = K = V = I``).
Here the field is the model's own: head h at token i pulls towards ``Σ_{j≥1} A_h,ij u_j`` (its
causal softmax(QK^⊤/√d_h) row, values the unit LN1 rows, ``V = I`` in β's frame). ``kernns`` sums
that over heads (attention to keys 1…i with every ``V = I``), ``kern`` keeps key 0 in. One hooked
forward pass per (step, passage) on the GPU; each layer's full attention map is read inside that
layer's hook (CUDA float64) and dropped before the next.

Cells per block: ``kernns:keys:r1out`` (primary), ``kernns:keys:frozen``, ``kern:attn:r1out``,
``kern:attn:frozen``, the check row ``causal:keys:r1out`` (`φ_β`, β 3.5, the attention arm's);
per head ``kernns_h:keys_h:r1out`` and ``kern_h:head_h:frozen``. Saved per (run, block): every
part's mean update over the targets (Blocked 29), read on the CPU by ``ascent`` (Shapley shares
of the block's shared ascent over sink, keys, mlpx, bias).

Tier 1: exploratory, unregistered. GPU pass (float32, eager, TF32 off), cells CUDA float64.
    python -m p1e_energy_field.u2_heads run --runs <p1e_long8 dir> --r0 <R0 labels> --out <dir> \
        --block-out <block arm dir> --attn-out <attention arm dir>
    python -m p1e_energy_field.u2_heads ascent --out <dir> --block-out <block arm dir>
    python -m p1e_energy_field.u2_heads report --out <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import u2_attn as ua
from . import u2_block as ub

#: Whole-attention cells: (field, component, reading); the first is primary, the last the check row.
ATTN_CELLS = (("kernns", "keys", "r1out"), ("kernns", "keys", "frozen"), ("kern", "attn", "r1out"),
              ("kern", "attn", "frozen"), ("causal", "keys", "r1out"))
#: The parts saved per block (means over the targets); the first four sum to ``block``.
SAVED = ("sink", "keys", "mlpx", "bias", "attn", "mlp", "block")
#: First checks (placed by the rule, not calibrated).
ROW_TOL = 1e-5
HEADSUM_TOL = 1e-4
CHECK_TOL = 1e-6
MEANS_TOL = 1e-6           # placed, not calibrated (as the three above)


# ---------------------------------------------------------------- the frame on the device

def unit_rows_t(X, w, b, eps: float):
    """`u2_block.unit_rows` in torch (same map: LN with gain and bias, then unit rows)."""
    import torch
    mu = X.mean(dim=1, keepdim=True)
    Y = (X - mu) / torch.sqrt(((X - mu) ** 2).mean(dim=1, keepdim=True) + eps) * w + b
    return Y / torch.clamp(Y.norm(dim=1, keepdim=True), min=1e-12)


def moves_t(U, X, c, t, frame) -> Dict:
    """``frozen`` and ``r1out`` moves of component ``c`` (`u2_attn.moves`' maps, on the device)."""
    tan = lambda V: V - (V * U).sum(dim=1, keepdim=True) * U        # noqa: E731
    out = {"frozen": tan(frame(X + c) - U)}
    cbar = c[t].mean(dim=0)
    nb = cbar.norm()
    if nb > 0:
        chat = cbar / nb
        out["r1out"] = tan(frame(X + c - (c @ chat)[:, None] * chat[None, :]) - U)
    return out


def head_fields(A, U):
    """``(Σ_j A_ij u_j, Σ_{j≥1} A_ij u_j)`` for one head's map ``A`` (n, n)."""
    m = A @ U
    return m, m - A[:, :1] * U[:1]


#: A head's keys cell may leave out a token only if the head puts at least 1 − this on key 0 and
#: the token itself there (attending to yourself adds nothing to the ``V = I`` field).
SINK_ONLY = 1e-4


def left_out_keys(d, g, A, t) -> Dict:
    """
    The targets a head's keys cell leaves out (`u2_block.cell`'s ``TINY`` rule) and the largest
    weight that head puts on keys other than key 0 and the token itself at any of them.
    """
    if d is None:
        return {"n_left": int(t.numel()), "left_max_keys_w": float("nan")}
    out = (d[t].norm(dim=1) < ub.TINY) | (g[t].norm(dim=1) < ub.TINY)
    w = (A[t, 1:].sum(dim=1) - A[t, t])[out]
    return {"n_left": int(out.sum()), "left_max_keys_w": float(w.max()) if w.numel() else 0.0}


def head_parts(A, Vc, Wo_h):
    """Head h's keys part and sink part, each (n, d): ``W_O^h Σ_j A_ij (v_j − b_V)`` split at key 0."""
    z = A @ Vc
    sink = A[:, :1] * (Vc[:1] @ Wo_h.T)
    return z @ Wo_h.T - sink, sink


# ---------------------------------------------------------------- the hooked pass

class HeadHooked(ua.Hooked):
    """`u2_attn.Hooked` plus each layer's full attention map and its input, read in that layer's hook."""

    def __init__(self, model, ops):
        super().__init__(model)
        self.ops, self.ctx = ops, None
        for l, layer in enumerate(self.layers):
            att = layer.attention

            def wrapped(q, k, v, attention_mask=None, head_mask=None, _l=l, _f=att._attn):
                out, w = _f(q, k, v, attention_mask, head_mask)
                if self.ctx is not None and _l in self.ctx["blocks"]:
                    self.cap[("w", _l)], self.cap[("v", _l)] = w[0].detach(), v[0].detach()
                return out, w
            att._attn = wrapped
            self.handles.append(layer.input_layernorm.register_forward_pre_hook(
                lambda m, i, _l=l: self.cap.__setitem__(("x", _l), i[0][0].detach())
                if self.ctx is not None and _l in self.ctx["blocks"] else None))
            self.handles.append(att.register_forward_hook(
                lambda m, i, o, _l=l: self._read(_l, o[0][0].detach())))

    def _read(self, l: int, attn_out) -> None:
        if self.ctx is None or l not in self.ctx["blocks"]:
            return
        import torch
        ctx, ops = self.ctx, self.ops
        dt = ops.dtype
        att = self.layers[l].attention
        H, hd = att.num_attention_heads, att.head_size
        x, w, v = self.cap.pop(("x", l)), self.cap.pop(("w", l)), self.cap.pop(("v", l))
        lw, lb, eps = ctx["ln_w"][l], ctx["ln_b"][l], ctx["eps"]
        frame = lambda Y: unit_rows_t(Y, lw, lb, eps)            # noqa: E731
        X = x.to(dt)
        U = frame(X)
        t, tn = ctx["t"], ctx["t_np"]
        rowdev = float((w.sum(dim=-1, dtype=torch.float64) - 1).abs().max())
        upper = float(w.triu(1).abs().max())
        bV = att.query_key_value.bias.view(H, 3 * hd)[:, 2 * hd:]
        Wo = att.dense.weight.to(dt).view(-1, H, hd)
        Vc = (v - bV[:, None, :]).to(dt)
        g_phi = ops.forces(U, 3.5, U @ U.T, only=("causal",))["causal"]
        gphi_hat = g_phi[t] / g_phi[t].norm(dim=1, keepdim=True)
        tan = lambda V: V - (V * U).sum(dim=1, keepdim=True) * U   # noqa: E731
        kern, kernns = torch.zeros_like(U), torch.zeros_like(U)
        keys_sum, sink_sum = torch.zeros_like(U), torch.zeros_like(U)
        cells, stats, mk, ms = [], [], [], []
        perms_for = ctx["perms_for"]
        for h in range(H):
            A = w[h].to(dt)
            m, mns = head_fields(A, U)
            kern += m
            kernns += mns
            keys_h, sink_h = head_parts(A, Vc[h], Wo[:, h, :])
            keys_sum += keys_h
            sink_sum += sink_h
            g_ns, g_k = tan(mns), tan(m)
            mv_k, mv_h = moves_t(U, X, keys_h, t, frame), moves_t(U, X, keys_h + sink_h, t, frame)
            for src, g, mv, rd in (("kernns_h:keys_h:r1out", g_ns, mv_k, "r1out"),
                                   ("kern_h:head_h:frozen", g_k, mv_h, "frozen")):
                if rd in mv:
                    cells.append({"block": l, "head": h, "beta": 0.0, "source": src,
                                  "targets": ctx["tname"], **ops.cell(mv[rd], g, U, tn, perms_for)})
            gt = g_ns[t]
            ok = gt.norm(dim=1) >= ub.TINY
            align = float(((gt[ok] / gt[ok].norm(dim=1, keepdim=True)) * gphi_hat[ok]).sum(1).mean())
            kh = keys_h[t]
            stats.append({"block": l, "head": h, "align_phi": align, "a0": float(A[t, 0].mean()),
                          "sharedness": float(kh.mean(0).norm() / kh.norm(dim=1).mean()),
                          **left_out_keys(mv_k.get("r1out"), g_ns, A, t)})
            mk.append(keys_h[t].mean(0))
            ms.append(sink_h[t].mean(0))
            del A, m, mns, keys_h, sink_h, g_ns, g_k, mv_k, mv_h
        # the whole attention, against the summed kernels and against φ_β (check row)
        c_attn = att.dense.bias + att.dense.weight @ bV.reshape(-1)
        # keys as the attention arm computes it (its captured key-0 column), so the check row matches
        parts = ua.split_layer(attn_out, attn_out, self.cap[("a0", l)], self.cap[("v0", l)],
                               att.dense.weight, att.dense.bias, bV, torch.zeros_like(att.dense.bias))
        comp = {"keys": parts["keys"].float().to(dt), "attn": attn_out.float().to(dt)}
        fields = {"kernns": tan(kernns), "kern": tan(kern), "causal": g_phi}
        for field, cname, rd in ATTN_CELLS:
            mv = moves_t(U, X, comp[cname], t, frame)
            if rd in mv:
                cells.append({"block": l, "beta": 3.5 if field == "causal" else 0.0,
                              "source": f"{field}:{cname}:{rd}", "targets": ctx["tname"],
                              **ops.cell(mv[rd], fields[field], U, tn, perms_for)})
        gt = fields["kernns"][t]
        ok = gt.norm(dim=1) >= ub.TINY
        align_ns = float(((gt[ok] / gt[ok].norm(dim=1, keepdim=True)) * gphi_hat[ok]).sum(1).mean())
        headsum = float((keys_sum + sink_sum + c_attn.to(dt) - attn_out.to(dt)).abs().max())
        self.out[l] = {"cells": cells, "heads": stats, "align_kernns_phi": align_ns,
                       "rowdev": rowdev, "upper": upper, "headsum_abs": headsum,
                       "mean_keys_h": torch.stack(mk).cpu().numpy(),
                       "mean_sink_h": torch.stack(ms).cpu().numpy()}
        del w, v, X, U, kern, kernns, keys_sum, sink_sum, fields, g_phi

    def run_heads(self, ids, layers: Sequence[int], ctx: Dict) -> tuple:
        """The attention arm's pass, with every layer's per-head cells read on the way."""
        self.ctx, self.out = {**ctx, "blocks": set(layers)}, {}
        try:
            hs, comps = self.run(ids, layers)
        finally:
            self.ctx = None
        out, self.out = self.out, {}
        return hs, comps, out


# ---------------------------------------------------------------- checks

def check_heads(hs: np.ndarray, per: Dict) -> Dict:
    """Rows sum to 1, nothing above the diagonal, the heads add up to attention's output."""
    rowdev = max(v["rowdev"] for v in per.values())
    upper = max(v["upper"] for v in per.values())
    headsum = max(float(v["headsum_abs"] / np.linalg.norm(hs[L + 1] - hs[L], axis=1).max())
                  for L, v in per.items())
    if rowdev > ROW_TOL or upper > 0:
        raise SystemExit(f"refusing: attention rows off 1 by {rowdev:.1e} or weight {upper:.1e} above the diagonal")
    if headsum > HEADSUM_TOL:
        raise SystemExit(f"refusing: heads sum to attention's output only to {headsum:.1e} > {HEADSUM_TOL}")
    return {"rowdev": rowdev, "upper": upper, "headsum_rel": headsum}


def saved_means(comps: Dict, per: Dict, t: np.ndarray) -> Dict[str, np.ndarray]:
    """Every part's mean update over the targets, per block (float64), and its check."""
    blocks = sorted(per)
    parts = np.stack([[comps[L][k][t].astype(np.float64).mean(axis=0) for k in SAVED] for L in blocks])
    dev = max(float(np.linalg.norm(parts[i, :4].sum(axis=0) - parts[i, 6]) / np.linalg.norm(parts[i, 6]))
              for i in range(len(blocks)))
    if dev > MEANS_TOL:
        raise SystemExit(f"refusing: saved means' parts sum to the block's only to {dev:.1e} > {MEANS_TOL}")
    return {"blocks": np.asarray(blocks), "names": np.asarray(SAVED), "parts": parts,
            "keys_h": np.stack([per[L]["mean_keys_h"] for L in blocks]),
            "sink_h": np.stack([per[L]["mean_sink_h"] for L in blocks]),
            "targets": t, "sum_dev": dev}


def first_attn_check(rec: Dict, attn_out: Path) -> float:
    """The check row against the attention arm's stored ``causal:keys:r1out`` record."""
    ref = json.loads((attn_out / "records" / rec["kind"] / f"step{rec['step']}_{rec['passage']}.json").read_text())
    want = {c["block"]: c["X"] for c in ref["cells"] if c["source"] == "causal:keys:r1out" and c["beta"] == 3.5}
    got = {c["block"]: c["X"] for c in rec["cells"] if c["source"] == "causal:keys:r1out"}
    if set(want) != set(got):
        raise SystemExit("refusing: check rows do not cover the attention arm's blocks")
    dev = max(abs(got[b] - want[b]) for b in want)
    if dev > CHECK_TOL:
        raise SystemExit(f"refusing: check row's X differs from the attention arm's by {dev:.1e} > {CHECK_TOL}")
    return dev


def explained(c: Dict, sink_only: Dict) -> bool:
    """
    A head's keys cell short of targets only where that head puts all but ``SINK_ONLY`` of its
    weight on key 0 and the token itself (no keys part or keys field above ``TINY``).
    """
    s = sink_only[(c["block"], c["head"])]
    return (c["source"] == "kernns_h:keys_h:r1out" and c["left_out"] == s["n_left"]
            and s["left_max_keys_w"] <= SINK_ONLY)


def check_populated(rec: Dict) -> None:
    n = rec["targets"]
    sink_only = {(s["block"], s["head"]): s for s in rec["heads"]}
    bad = [c for c in rec["cells"] + rec["head_cells"] if not np.isfinite(c["X"]) or
           (c["n"] < 0.9 * n and not explained(c, sink_only))]
    seen = {(c["block"], c["source"]) for c in rec["cells"]}
    want = {(b, f"{f}:{k}:{r}") for b in ub.BLOCKS for f, k, r in ATTN_CELLS}
    heads = {(c["block"], c["head"], c["source"]) for c in rec["head_cells"]}
    H = 1 + max(c["head"] for c in rec["head_cells"])
    want_h = {(b, h, s) for b in ub.BLOCKS for h in range(H)
              for s in ("kernns_h:keys_h:r1out", "kern_h:head_h:frozen")}
    if bad or seen != want or heads != want_h:
        raise SystemExit(f"refusing: first cell not populated ({len(bad)} bad cells, "
                         f"{len(want - seen)} attention and {len(want_h - heads)} head cells missing): " +
                         "; ".join(f"L{c['block']} h{c.get('head', '-')} {c['source']} n {c['n']} of {n}"
                                   for c in bad[:10]))


# ---------------------------------------------------------------- the batch

def run(a) -> int:
    import gc
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
    meta.update(code=code, rule="design-1e.md \"U2's per-head arm: the rule\"")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps(meta, indent=1) + "\n")
    ops = ub.get_ops("cuda")
    tok = AutoTokenizer.from_pretrained(ub.REPO, revision="step143000")
    order = first + ([] if a.first_only else sorted(rest, key=lambda j: int(j[1])))
    loaded, model, hk = None, None, None
    for i, (kind, step, key, rd, rev, tg) in enumerate(order):
        path = a.out / "records" / kind / f"step{step}_{key}.json"
        mpath = a.out / "means" / kind / f"step{step}_{key}.npz"
        if path.exists():
            had = json.loads(path.read_text()).get("code")
            if had != code:
                raise SystemExit(f"refusing to resume: {path} was written by {had}, this is {code}")
            print(f"have {path.name}", flush=True)
            continue
        if loaded != step:
            if hk is not None:
                hk.close()
            hk = model = None
            gc.collect()
            torch.cuda.empty_cache()
            model, _ = load_model(f"pythia-410m-step{step}")
            if next(model.parameters()).dtype != torch.float32:
                raise SystemExit("refusing: model is not float32")
            man_rev = json.loads((Path(rd) / "manifest.json").read_text())["hf_revision"]
            if man_rev != rev:
                raise SystemExit(f"refusing: {rd} manifest revision {man_rev}, plan {rev}")
            hk, loaded = HeadHooked(model, ops), step
            ln = ua.ln_from(model)
            ln_t = {"ln_w": ops.to(ln["w"]), "ln_b": ops.to(ln["b"]), "eps": ln["eps"]}
        t0 = time.monotonic()
        dev = next(model.parameters()).device
        ids = torch.tensor([ua.token_ids(tok, Path(rd))], device=dev)
        t = np.asarray(tg, dtype=int)
        ctx = {**ln_t, "t": torch.as_tensor(t, device=dev), "t_np": t, "perms_for": ub.make_perms(key),
               "tname": "t12" if key.endswith("_long") else "r0"}
        hs, comps, per = hk.run_heads(ids, ub.BLOCKS, ctx)
        chk = ua.check_pass(hs, comps, Path(rd), kind)
        chk.update(check_heads(hs, per))
        means = saved_means(comps, per, t)
        chk["means_sum_dev"] = means.pop("sum_dev")
        cells = [c for L in sorted(per) for c in per[L]["cells"]]
        rec = {"kind": kind, "step": step, "passage": key, "run": rd, "code": code, "device": ops.name,
               "pass": "cuda:float32", "revision": rev, "targets": int(t.size), "checks": chk,
               "align_kernns_phi": {str(L): v["align_kernns_phi"] for L, v in per.items()},
               "heads": [s for L in sorted(per) for s in per[L]["heads"]],
               "cells": [c for c in cells if "head" not in c],
               "head_cells": [c for c in cells if "head" in c]}
        if i == 0 and (kind, step, key) == ("long",) + ub.FIRST:
            check_populated(rec)
            rec["checks"]["keys_vs_attn_arm"] = first_attn_check(rec, a.attn_out)
            print(f"first cell populated; checks {rec['checks']}", flush=True)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(mpath.with_suffix(".tmp.npz"), code=code, **means)
        mpath.with_suffix(".tmp.npz").rename(mpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec) + "\n")
        tmp.rename(path)
        print(f"done {path.name} {time.monotonic() - t0:.0f}s match {chk['match_unit']:.1e} "
              f"heads {chk['headsum_rel']:.1e}", flush=True)
    if hk is not None:
        hk.close()
    return 0


# ---------------------------------------------------------------- Blocked 29's ascent share (CPU)

PLAYERS = ("sink", "keys", "mlpx", "bias")


def shapley(F: Dict[frozenset, float], players: Sequence[str]) -> Dict[str, float]:
    """Exact Shapley values of a set function ``F`` (``F[frozenset()]`` = 0) over ``players``."""
    n, out = len(players), {}
    for k in players:
        rest = [p for p in players if p != k]
        v = 0.0
        for r in range(n):
            wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(rest, r):
                S = frozenset(S)
                v += wgt * (F[S | {k}] - F[S])
        out[k] = v
    return out


def ascent_block(X: np.ndarray, cbar: Dict[str, np.ndarray], t: np.ndarray, frame) -> Dict:
    """``F(S)`` over every subset of the parts and each part's Shapley value (one block)."""
    U = frame(X)
    g = ub.forces(U, 3.5, only=("causal",))["causal"][t]
    gh = g / np.linalg.norm(g, axis=1, keepdims=True)
    F = {}
    for r in range(len(PLAYERS) + 1):
        for S in itertools.combinations(PLAYERS, r):
            if not S:
                F[frozenset()] = 0.0
                continue
            c = sum(cbar[k] for k in S)
            d = ub.tangent(U[t], frame(X[t] + c) - U[t])
            F[frozenset(S)] = float(np.sum(d * gh, axis=1).mean())
    phi = shapley(F, PLAYERS)
    return {"F_all": F[frozenset(PLAYERS)], "shapley": phi,
            "F_single": {k: F[frozenset([k])] for k in PLAYERS}}


def _ascent_job(args) -> str:
    mpath, rec_path, ln1_path, dest = (Path(x) for x in args)
    if dest.exists():
        return f"have {dest.name}"
    rec = json.loads(rec_path.read_text())
    z = np.load(mpath)
    if str(z["code"]) != rec["code"]:
        raise SystemExit(f"refusing: {mpath} and {rec_path} name different producers")
    ln = ub.load_ln1(ln1_path)
    names = [str(s) for s in z["names"]]
    t = z["targets"].astype(int)
    st = np.load(Path(rec["run"]) / "activations.npz")
    acts, norms = st["activations"], st["norms"]
    out = {}
    for i, L in enumerate(z["blocks"].tolist()):
        X = (acts[L] * norms[L][:, None]).astype(np.float64)
        frame = lambda Y: ub.unit_rows(Y, ln["w"][L], ln["b"][L], ln["eps"])     # noqa: E731
        cbar = {k: z["parts"][i, names.index(k)] for k in PLAYERS}
        out[str(L)] = ascent_block(X, cbar, t, frame)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps({"kind": rec["kind"], "step": rec["step"], "passage": rec["passage"],
                               "code": rec["code"], "blocks": out}) + "\n")
    tmp.rename(dest)
    return f"done {dest.name}"


def ascent(a) -> int:
    args = []
    for mpath in sorted((a.out / "means").glob("*/*.npz")):
        if mpath.name.endswith(".tmp.npz"):
            continue
        kind = mpath.parent.name
        rec_path = a.out / "records" / kind / (mpath.stem + ".json")
        rev = json.loads(rec_path.read_text())["revision"]
        ln1 = a.block_out / "ln1" / f"{rev}.npz"
        if not ln1.exists():
            raise SystemExit(f"refusing: no LN1 export {ln1} for {rec_path.name}")
        args.append((str(mpath), str(rec_path), str(ln1), str(a.out / "ascent" / kind / (mpath.stem + ".json"))))
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(_ascent_job, args):
            print(msg, flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", type=Path, required=True)
    r.add_argument("--r0", type=Path, default=None, help="R0's labels dir (v1 beside)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--attn-out", type=Path, required=True, help="the attention arm's output (first check)")
    r.add_argument("--steps", nargs="*", default=None)
    r.add_argument("--first-only", action="store_true")
    s = sub.add_parser("ascent")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--block-out", type=Path, required=True, help="the block arm's output (its ln1/)")
    s.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("report")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        from .u2_heads_report import report
        return report(a.out)
    return {"run": run, "ascent": ascent}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
