"""
p1e_energy_field/u2_torch.py — `u2_block`'s per-cell reading on a torch device (the GPU).

The numpy code in `u2_block` is the reference: this module mirrors ``forces``, ``null_cos`` and
``cell`` line for line, draws the same permutations (``u2_block.make_perms``, numpy, moved to
the device) and returns the same record. It is used only where the agreement check against
stored CPU records passes (`p1e_energy_field/status-1e.md` "GPU reading"). Activations are
read and put in β's frame in numpy (O(n d)); only the O(n² d) parts run here.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from .u2_block import SOURCES, TINY


class Ops:
    """The device-side operations `u2_block.read_run` needs: ``to``, ``forces``, ``cell``."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float64):
        self.device, self.dtype = torch.device(device), dtype
        self.name = f"{device}:{str(dtype).replace('torch.', '')}"

    def to(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), device=self.device, dtype=self.dtype)

    @staticmethod
    def tangent(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return V - (V * U).sum(dim=1, keepdim=True) * U

    def forces(self, U: torch.Tensor, beta: float, S=None,
               only: Sequence[str] = SOURCES) -> Dict[str, torch.Tensor]:
        n = U.shape[0]
        lower = torch.ones((n, n), dtype=torch.bool, device=U.device).tril()
        out = {}
        for src in ("causal", "nosink", "full"):
            if src not in only and not (src == "causal" and "local" in only):
                continue
            S = U @ U.T if S is None else S
            mask = lower.clone() if src != "full" else torch.ones_like(lower)
            if src == "nosink":
                mask[1:, 0] = False
            Z = (beta * S).masked_fill(~mask, float("-inf"))
            W = torch.exp(Z - Z.max(dim=1, keepdim=True).values)
            W = W / W.sum(dim=1, keepdim=True)
            out[src] = W @ U
        m0 = torch.cumsum(U, dim=0) / torch.arange(1, n + 1, device=U.device, dtype=U.dtype)[:, None]
        if "local" in only:
            out["local"] = out["causal"] - m0
        if "mean0" in only:
            out["mean0"] = m0
        return {k: self.tangent(U, m) for k, m in out.items() if k in only}

    @staticmethod
    def null_cos(Dg, Du, dn2, perms):
        ar = torch.arange(Dg.shape[0], device=Dg.device)
        a = Dg[ar, ar] / torch.sqrt(torch.clamp(dn2 - Du[ar, ar] ** 2, min=1e-300))
        num, du = Dg[perms, ar], Du[perms, ar]
        return a, (num / torch.sqrt(torch.clamp(dn2[perms] - du ** 2, min=1e-300))).mean(dim=1)

    def cell(self, d, g, U, tgt: np.ndarray, perms_for) -> Dict:
        from scipy.stats import spearmanr
        tg = torch.as_tensor(tgt, device=d.device)
        ok = ((d[tg].norm(dim=1) >= TINY) & (g[tg].norm(dim=1) >= TINY)).cpu().numpy()
        t = tgt[ok]
        ti = torch.as_tensor(t, device=d.device)
        D, Ut = d[ti], U[ti]
        Gh = g[ti] / g[ti].norm(dim=1, keepdim=True)
        P = torch.as_tensor(perms_for(t.size), device=d.device)
        Dg, Du, dn2 = D @ Gh.T, D @ Ut.T, (D * D).sum(dim=1)
        a, nul = self.null_cos(Dg, Du, dn2, P)
        dbar = D.mean(dim=0)
        at, nult = self.null_cos(Dg - (Gh @ dbar)[None, :], Du - (Ut @ dbar)[None, :],
                                 dn2 - 2 * (D @ dbar) + dbar @ dbar, P)
        a, nul, at, nult = (x.double().cpu().numpy() for x in (a, nul, at, nult))
        A, At = float(a.mean()), float(at.mean())
        rho = spearmanr(a, np.log(t)).statistic if t.size > 2 and t.min() > 0 else float("nan")
        return {"n": int(t.size), "left_out": int((~ok).sum()), "A": A,
                "null_mean": float(nul.mean()), "null_sd": float(nul.std()),
                "X": A - float(nul.mean()),
                "p_hi": float((1 + np.sum(nul >= A)) / (len(nul) + 1)),
                "p_lo": float((1 + np.sum(nul <= A)) / (len(nul) + 1)),
                "At": At, "Xt": At - float(nult.mean()),
                "pt_hi": float((1 + np.sum(nult >= At)) / (len(nult) + 1)),
                "pt_lo": float((1 + np.sum(nult <= At)) / (len(nult) + 1)),
                "rho_logpos": float(rho)}
