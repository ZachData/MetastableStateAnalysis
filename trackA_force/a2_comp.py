"""
trackA_force/a2_comp.py — A2 unit 3: the per-component instrument (`design-A2.md` "Unit 3";
frozen 2026-10-16, Blocked 56 (a)).

Per block l (Pythia's parallel residual) the update ``v = Σ_c v_c`` over 18 components: each
head's own ``W_O^h`` slice of attention's output (value bias included, ``h0``…``h15``), the
attention output bias ``b_O``, the MLP with its bias ``mlp``. Beside, never counted: each head's
sink part ``A_i0 · v_0 W_O^h`` (``h*_sink0``) and the part through its most-received non-zero key
``A_ij* · v_j* W_O^h`` (``h*_topkey``, j* by attention mass received over all queries; its token is
recorded so a fixed key such as 1e's newline can be read).

**The path integral (Aumann–Shapley).** In the block's frame ``f_l(x) = unit(LN1_l(x))``,
``Δ_c = ∫₀¹ J_l(x + s v) ds · v_c`` by Gauss–Legendre on ``NODES`` nodes, so ``Σ_c Δ_c`` is the frame
chord ``f_l(x + v) − f_l(x)`` up to quadrature (G0 measures it per block). Positions 1…n−1, the field
F the causal self-attention field on ``U = f_l(x)[1:]`` at the layer's β. Per component, linear in
``Δ_c`` and so additive over components:

- ``proj``: the mean over tokens of ``⟨Δ_c^tan, F⟩ / |F|²`` (unlifted; tokens with no field out);
- ``proj_df``: ``proj`` with one shared drift removed (`integration_time.drift_free_slope`'s
  formula on the unlifted move);
- ``off``: ``Σ_i ⟨o_ci, o_i⟩ / Σ_i |o_i|²`` with ``o_c = Δ_c^tan − proj_c F`` and ``o = Σ_c o_c``
  over the 18 counted components (pooled over tokens; sums to 1);
- ``size``: the mean ``|v_c|`` over positions 1…n−1 (raw), unit 5's size deciles; ``disp`` the mean
  ``|Δ_c^tan|``.

The block's totals carry the lifted reads A1 used (log, its verdict; gnomonic, the design's text)
and their ratio to the unlifted mean, beside.

Tier 1: exploratory, unregistered. Torch float64; the pass float32, eager, TF32 off.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from .a1a import REPO

N_HEADS = 16
COMPS = tuple(f"h{h}" for h in range(N_HEADS)) + ("b_O", "mlp")
BESIDE = tuple(f"h{h}_sink0" for h in range(N_HEADS)) + tuple(f"h{h}_topkey" for h in range(N_HEADS))
NODES = 16                          # placed (design "Unit 3")
G0_TOL = 1e-6                       # placed: Σ_c Δ_c against the chord, relative (design G0)
SPLIT_TOL = 1e-4                    # the pass: heads + b_O against attention's output (u2_attn's)
MASS = 0.5                          # placed: the 50 % mass cut (design "Unit 4")
FEW = 43                            # placed: 10 % of 432 (design "Unit 4")


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackA_force/a2_comp.py", "trackA_force/a1a.py", "p1e_energy_field/u2_attn.py",
              "trackA_force/a1a_gate.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the frame and its Jacobian

def gl_nodes(k: int = NODES):
    s, w = np.polynomial.legendre.leggauss(k)
    return (s + 1.0) / 2.0, w / 2.0


def frame_t(x, w, b, eps: float):
    """``unit(LN1(x))`` row-wise over the last axis (torch)."""
    xc = x - x.mean(-1, keepdim=True)
    y = xc / ((xc * xc).mean(-1, keepdim=True) + eps).sqrt() * w + b
    return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-300)


def jvp_frame(x, dX, w, b, eps: float):
    """``J_f(x) · dX`` for every leading slice of ``dX`` (P, n, d), x (n, d); exact, no finite step."""
    xc = x - x.mean(-1, keepdim=True)
    sig = ((xc * xc).mean(-1, keepdim=True) + eps).sqrt()
    z = xc / sig
    y = z * w + b
    ny = y.norm(dim=-1, keepdim=True)
    f = y / ny
    C = dX - dX.mean(-1, keepdim=True)
    dz = (C - z * (z * C).mean(-1, keepdim=True)) / sig
    dy = dz * w
    return (dy - f * (f * dy).sum(-1, keepdim=True)) / ny


def field_t(U, beta: float):
    """The causal self-attention field at β on unit rows U (n, d), tangent at U."""
    import torch
    n = U.shape[0]
    S = (beta * (U @ U.T)).masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool, device=U.device), 1),
                                        -math.inf)
    tgt = torch.softmax(S, dim=-1) @ U
    return tgt - (tgt * U).sum(-1, keepdim=True) * U


def read_block(x, parts, n_counted: int, w, b, eps: float, beta: float, nodes: int = NODES) -> Dict:
    """
    One block. ``x`` (n, d) the block's input (raw residual), ``parts`` (P, n, d) the raw parts, the
    first ``n_counted`` summing to the block's update, the rest beside; ``w``, ``b`` LN1's affine.
    Torch float64 on the parts' device. Per part: ``proj``, ``proj_df``, ``off``, ``size``, ``disp``;
    the block: ``g0`` (quadrature, relative), the totals and the lifted reads.
    """
    import torch
    from .a1a_gate import _t_lifted
    dt = torch.float64
    x, parts = x.to(dt), parts.to(dt)
    w, b = torch.as_tensor(w, dtype=dt, device=x.device), torch.as_tensor(b, dtype=dt, device=x.device)
    v = parts[:n_counted].sum(0)
    s, ws = gl_nodes(nodes)
    D = torch.zeros_like(parts)
    for sk, wk in zip(s, ws):
        D += float(wk) * jvp_frame(x + float(sk) * v, parts, w, b, eps)
    f0, f1 = frame_t(x, w, b, eps), frame_t(x + v, w, b, eps)
    chord = (f1 - f0)[1:]
    D = D[:, 1:]
    g0 = float((D[:n_counted].sum(0) - chord).norm() / chord.norm().clamp_min(1e-300))
    U = f0[1:]
    F = field_t(U, beta)
    Dt = D - (D * U).sum(-1, keepdim=True) * U
    xx = (F * F).sum(-1)
    ok = xx >= 1e-18
    Fo = F[ok]
    proj_tok = (Dt[:, ok] * Fo).sum(-1) / xx[ok]                       # (P, m)
    proj = proj_tok.mean(1)
    O = Dt[:, ok] - proj_tok[..., None] * Fo
    o = O[:n_counted].sum(0)
    off = (O * o).sum((1, 2)) / (o * o).sum().clamp_min(1e-300)
    Fc = Fo - Fo.mean(0)
    Dc = Dt[:, ok] - Dt[:, ok].mean(1, keepdim=True)
    proj_df = (Dc * Fc).sum((1, 2)) / (Fc * Fc).sum().clamp_min(1e-300)
    tot = {"proj": float(((chord - (chord * U).sum(-1, keepdim=True) * U)[ok] * Fo).sum(-1).div(xx[ok]).mean())}
    for lift in ("log", "gnomonic"):
        L = _t_lifted(U, f1[1:], lift)[ok]
        val = (L * Fo).sum(-1) / xx[ok]
        tot[f"proj_{lift}"] = float(val[torch.isfinite(val)].mean())
        tot[f"ratio_{lift}"] = tot[f"proj_{lift}"] / tot["proj"] if tot["proj"] != 0 else float("nan")
    tot["field"] = float(xx[ok].sqrt().mean())
    tot["n_field"] = int(ok.sum())
    return {"proj": proj.cpu().numpy(), "proj_df": proj_df.cpu().numpy(), "off": off.cpu().numpy(),
            "size": parts[:, 1:].norm(dim=-1).mean(1).cpu().numpy(),
            "disp": Dt[:, ok].norm(dim=-1).mean(1).cpu().numpy(), "g0": g0, "total": tot}


# ---------------------------------------------------------------- the sets unit 4 reads

def mass_set(vals: np.ndarray, sign: int, mass: float = MASS) -> np.ndarray:
    """Indices of the fewest components whose values of ``sign`` sum to ≥ ``mass`` of all of that sign."""
    a = np.asarray(vals, dtype=np.float64) * sign
    idx = np.flatnonzero(a > 0)
    if idx.size == 0:
        return idx
    order = idx[np.argsort(-a[idx], kind="stable")]
    cum = np.cumsum(a[order])
    k = int(np.searchsorted(cum, mass * cum[-1] - 1e-15 * cum[-1])) + 1
    return order[:k]


def top_share(vals: np.ndarray, sign: int, k: int = FEW) -> float:
    """The share of all ``sign`` mass carried by its top ``k`` components."""
    a = np.asarray(vals, dtype=np.float64) * sign
    a = np.sort(a[a > 0])[::-1]
    return float(a[:k].sum() / a.sum()) if a.size else float("nan")


def gini(vals: np.ndarray) -> float:
    a = np.sort(np.abs(np.asarray(vals, dtype=np.float64)))
    n = a.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ a / (n * a.sum())) if a.sum() > 0 else float("nan")


# ---------------------------------------------------------------- the hooked pass

def _make_hooked():
    from p1e_energy_field.u2_attn import Hooked

    class CompHooked(Hooked):
        """
        `u2_attn.Hooked` plus, per layer, the block's input, the attention map and values; each layer
        is read by `read_block` in its MLP's forward hook (attention and MLP both done), so nothing
        but the per-layer reads is kept.
        """

        def __init__(self, model):
            super().__init__(model)
            self.ctx, self.rows = None, {}
            for l, layer in enumerate(self.layers):
                def capture(v, w, _l=l):
                    if self.ctx is not None:
                        self.cap[("w", _l)], self.cap[("v", _l)] = w[0].detach(), v[0].detach()
                self.on_attn[l].append(capture)
                self.handles.append(layer.input_layernorm.register_forward_pre_hook(
                    lambda m, i, _l=l: self.cap.__setitem__(("x", _l), i[0][0].detach())
                    if self.ctx is not None else None))
                self.handles.append(layer.mlp.register_forward_hook(
                    lambda m, i, o, _l=l: self._read(_l, o[0].detach())))

        def parts(self, l: int, mlp_out):
            """(18 + 32, n, d) float64: the counted components then the beside parts; the checks."""
            import torch
            att = self.layers[l].attention
            H, hd = self.n_heads, att.head_size
            x, w, v = self.cap.pop(("x", l)), self.cap.pop(("w", l)), self.cap.pop(("v", l))
            attn = self.cap[("attn", l)]
            dt = torch.float64
            Wo = att.dense.weight.to(dt).view(-1, H, hd)                      # (d, H, hd)
            W, V = w.to(dt), v.to(dt)                                          # (H, n, n), (H, n, hd)
            heads = torch.einsum("hij,hjk,dhk->hid", W, V, Wo)
            bO = att.dense.bias.to(dt)[None, :].expand(x.shape[0], -1)
            sink = torch.einsum("hi,hk,dhk->hid", W[:, :, 0], V[:, 0], Wo)
            recv = W[:, :, 1:].sum(1)                                          # mass received, keys 1…
            jstar = recv.argmax(1) + 1                                         # (H,)
            hh = torch.arange(H, device=W.device)
            top = torch.einsum("hi,hk,dhk->hid", W[hh, :, jstar], V[hh, jstar], Wo)
            P = torch.cat([heads, bO[None], mlp_out.to(dt)[None], sink, top], 0)
            scale = (attn.to(dt) + mlp_out.to(dt)).norm(dim=-1).max()
            chk = {"split_rel": float((heads.sum(0) + bO - attn.to(dt)).abs().max() / scale),
                   "jstar": jstar.tolist(),
                   "jstar_mass": (recv[hh, jstar - 1] / W.shape[1]).tolist()}
            return x.to(dt), P, chk

        def _read(self, l: int, mlp_out) -> None:
            if self.ctx is None:
                return
            x, P, chk = self.parts(l, mlp_out)
            c = self.ctx
            if chk["split_rel"] > SPLIT_TOL:
                raise SystemExit(f"refusing: block {l}: heads + b_O off attention's output by {chk['split_rel']:.1e}")
            r = read_block(x, P, len(COMPS), c["w"][l], c["b"][l], c["eps"], float(c["beta"][l]), c["nodes"])
            self.rows[l] = {**r, **chk, "sum_counted": P[:len(COMPS)].sum(0).float().cpu().numpy()}

        def run_read(self, ids, ln: Dict, beta: Sequence[float], nodes: int = NODES) -> tuple:
            """One pass; ``(hidden states (25, n, d) float32 CPU, {block: read})``."""
            import torch
            self.rows, self.cap = {}, {}
            self.ctx = {"w": ln["w"], "b": ln["b"], "eps": ln["eps"], "beta": list(beta), "nodes": nodes}
            try:
                with torch.no_grad():
                    out = self.model(input_ids=ids, output_hidden_states=True, use_cache=False)
                hs = torch.stack([h[0] for h in out.hidden_states]).float().cpu().numpy()
            finally:
                self.ctx = None
                self.cap.clear()
            return hs, self.rows

    return CompHooked


def comp_hooked(model):
    return _make_hooked()(model)


def check_recon(hs: np.ndarray, rows: Dict) -> float:
    """Σ counted parts against the block's update from the hidden states, relative to its largest row."""
    out = 0.0
    for L, r in rows.items():
        if L + 1 >= hs.shape[0] - 1:             # the last state is post-final-LN
            continue
        d = hs[L + 1].astype(np.float64) - hs[L]
        out = max(out, float(np.abs(r["sum_counted"] - d).max() / np.linalg.norm(d, axis=1).max()))
    return out


def flatten(rows: Dict[int, Dict], key: str, n_blocks: int = 24) -> np.ndarray:
    """(blocks × 18,) in block-major order: the 432 components unit 4 ranks."""
    return np.concatenate([np.asarray(rows[L][key])[:len(COMPS)] for L in range(n_blocks)])
