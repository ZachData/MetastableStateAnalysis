"""S2, what rotation does (`trackS_subspace/design-S2.md`; `docs/ROADMAP.md` §7a). Every attention
head's OV edited at once on 410m, read on A5's V (E₁'s violations), 7d's repeat gain G and the
centred displacement D.

**The frame** (OV1's, `tools/math_checks/s2_ov_arms.py`). Per head, B = Π W_O, Cᵀ = Π diag(γ) W_Vᵀ
(LN1's gain folded in, Π the centring), K = B C, the map the head applies to the attention-weighted
LN1-normalised input x̂. Z = [B, Cᵀ] = Q R with Q orthonormal (d × 2k), so every arm is
K′ = Q M Qᵀ with a 2k × 2k core M and the base model is M = K̃ = R_B R_Cᵀ. Q is orthonormal, so
Frobenius norms, symmetric parts and eigenvalues are those of the core.

**The hook.** Each head's V path is zeroed (W_V and b_V rows), its constant c = W_O(W_V β + b_V)
goes to the dense bias (rows of P sum to 1; OV1's C1), and the attention output gains, row-wise,
Σ_h P_h (X̂ Q_h) M_hᵀ Q_hᵀ. One part of the head's output lies outside the centred frame: its write
along 1, (1/d) 1 1ᵀ W_O W_V diag(γ) P x̂. No LayerNorm reads it, but V does (A5's E₁ normalises the
raw hidden state), so the hook keeps it as the model's own in every arm: the arms edit only what
the network reads. At M = K̃ the hook is the model (G0).

**Arm (b)** is (b-schur), Blocked 86 (1) (a), the user's, 2026-10-21: each 2 × 2 block of the real
Schur form of K's restriction to col(B), in an orthonormal basis, made symmetric. (b-eig),
λ → Re λ in A = C B's eigenbasis, is kept beside, sizes recorded, no label.

Tier 1: exploratory, unregistered.
    python -m trackS_subspace.s2 g0 --out <dir>
    python -m trackS_subspace.s2 preflight --out <dir>
    python -m trackS_subspace.s2 beside --out <dir>
    python -m trackS_subspace.s2 gates --out <dir>        # unit 2: base, controls, gain arms, G2, G3, G3′
    python -m trackS_subspace.s2 gate-read --out <dir>    # G1, G1′, G2, G3, G3′ (no GPU)
    python -m trackS_subspace.s2 arms --out <dir>         # a, b, b-eig; refuses before gate-read
    python -m trackS_subspace.s2 read --out <dir>         # R1, R2 (no GPU)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackB_circuit.b2a import D_MODEL, HEAD_DIM, N_HEADS, N_LAYERS, REPO, seed_for

SCORED_STEPS = (512, 1000, 2000, 4000, 143000)
BESIDE_STEPS = (0, 256)
UNIT2_STEPS = BESIDE_STEPS + SCORED_STEPS
R2_STEPS = (1000, 2000, 4000, 143000)
G2_STEPS = (512,) + R2_STEPS                               # 512 beside, as R2's
V_PLACED = 7                                               # R1, G3: at most one violation per prompt
G_HALF = 0.5                                               # R2: "survives" at G ≥ 0.5 G_base
G2_MEMBERS = ("L5H2", "L7H8", "L12H5", "L8H6", "L11H14", "L8H9", "L15H14", "L7H1", "L9H13", "L10H9")
G3P_BASE_TOL = 1e-4                                        # G3′'s frame at K_s against the base: D
G0_STEPS = (0, 256, 512, 1000, 2000, 4000, 143000)        # base V against the panel
PREFLIGHT_STEP = 8000
WIRING_STEPS = (0, 8000)
N_CTL = 40
G3_STEP, G3_T = 128, (0.25, 0.5, 1.0)
ARMS = ("a", "b")                                          # b = b-schur (Blocked 86 (1) (a))
WIRING_TOL = 1e-4                                          # placed (design G0): relative, float32
READBACK_TOL = 1e-6                                        # placed (design G0): of ‖K‖
COND_MAX = 1e8                                             # (b-eig) only: a head kept as is above it
REL_TOL = 1e-3                                             # A5's D2 (a)
N_REP, N_SEQS, VOCAB = 96, 16, (1000, 40000)               # 7d's probe, 16 sequences
EVAL_SEED = 20260907                                       # `tools/run/induction_rank_sweep.py`
N_READBACK_HEADS = 20
N_SHUFFLE = 2000
S1_STEPS = (0, 128, 256, 512, 1000, 2000, 4000, 143000)
R1B_PAIRS = (("ov_write", "ov_read"), ("ov_write", "qk_key"), ("ov_neg", "ov_pos"))


class S2Error(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    h.update((REPO / "trackS_subspace" / "s2.py").read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the frame and the arms (pure)

def head_frame(W_O: np.ndarray, W_V: np.ndarray, gamma: np.ndarray) -> Dict[str, np.ndarray]:
    """``Q`` (d × 2k, orthonormal), ``Bq``, ``Cq`` (2k × k) with Π W_O = Q Bq and
    Π diag(γ) W_Vᵀ = Q Cq; ``K`` = Bq Cqᵀ, the head's map in the core's coordinates."""
    W_O, W_V, gamma = (np.asarray(a, dtype=np.float64) for a in (W_O, W_V, gamma))
    k = W_O.shape[1]
    B = W_O - W_O.mean(axis=0, keepdims=True)
    Ct = gamma[:, None] * W_V.T
    Ct = Ct - Ct.mean(axis=0, keepdims=True)
    Q, R = np.linalg.qr(np.hstack([B, Ct]))
    Bq, Cq = R[:, :k], R[:, k:]
    return {"Q": Q, "Bq": Bq, "Cq": Cq, "K": Bq @ Cq.T}


def sym(M: np.ndarray) -> np.ndarray:
    return (M + M.T) / 2


def schur_blocks(S: np.ndarray, tol: float = 0.0) -> List[Tuple[int, int]]:
    """``(start, size)`` of the diagonal blocks of a real Schur form."""
    out, i, n = [], 0, S.shape[0]
    while i < n:
        if i + 1 < n and abs(S[i + 1, i]) > tol:
            out.append((i, 2))
            i += 2
        else:
            out.append((i, 1))
            i += 1
    return out


def arm_b_schur(f: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """(b-schur), math check C7: ``(K′, the arm's nonzero eigenvalues, info)``. B = U R, T = R C U
    is K's restriction to col(B); T = V S Vᵀ (real Schur); each 2 × 2 block of S replaced by its
    symmetric part; K′ = K + U V (S′ − S) Vᵀ Uᵀ, of size ‖S′ − S‖."""
    import scipy.linalg as sla
    U, R = np.linalg.qr(f["Bq"])
    T = R @ f["Cq"].T @ U
    S, V = sla.schur(T, output="real")
    S2 = S.copy()
    eig: List[float] = []
    n_pairs = 0
    for i, m in schur_blocks(S):
        if m == 2:
            blk = S[i:i + 2, i:i + 2]
            S2[i:i + 2, i:i + 2] = sym(blk)
            a, s = (blk[0, 0] + blk[1, 1]) / 2, (blk[0, 1] + blk[1, 0]) / 2
            d = (blk[0, 0] - blk[1, 1]) / 2
            r = np.sqrt(d * d + s * s)
            eig += [a + r, a - r]
            n_pairs += 1
        else:
            eig.append(S[i, i])
    Kp = f["K"] + U @ V @ (S2 - S) @ V.T @ U.T
    return Kp, np.array(eig), {"pairs": n_pairs}


def arm_b_eig(f: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """(b-eig), math check C2, beside: g(A) = Q diag(Re λ / λ) Q⁻¹ on A = C B, K′ = B g(A) C. A head
    whose eigenvector matrix has condition number above ``COND_MAX`` keeps its K (design fallback)."""
    A = f["Cq"].T @ f["Bq"]
    lam, Qe = np.linalg.eig(A)
    cond = float(np.linalg.cond(Qe))
    if not np.isfinite(cond) or cond > COND_MAX:
        return f["K"].copy(), lam, {"kept": True, "cond": cond, "pairs": 0}
    g = (Qe * (lam.real / np.where(lam == 0, 1, lam))[None, :]) @ np.linalg.inv(Qe)
    if np.abs(g.imag).max() > 1e-8 * max(np.abs(g.real).max(), 1e-300):
        raise S2Error(f"g(A) not real: {np.abs(g.imag).max():.1e}")
    return f["Bq"] @ g.real @ f["Cq"].T, lam.real, {"kept": False, "cond": cond,
                                                    "pairs": int((lam.imag > 0).sum())}


def arm_core(f: Dict[str, np.ndarray], arm: str) -> Tuple[np.ndarray, Dict]:
    """The arm's core K′ (2k × 2k) and its info (``eig``: the arm's nonzero eigenvalues, for b)."""
    if arm == "base":
        return f["K"].copy(), {}
    if arm == "a":
        return sym(f["K"]), {}
    if arm == "b":
        Kp, ev, info = arm_b_schur(f)
        return Kp, {**info, "eig": ev}
    if arm == "b_eig":
        Kp, ev, info = arm_b_eig(f)
        return Kp, {**info, "eig": ev}
    raise S2Error(f"unknown arm {arm}")


def arm_span(f: Dict[str, np.ndarray], arm: str) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal write and read bases (core coordinates) of the arm's edits: the controls' span
    (`lit-S2.md`: "the edit's own read and write spaces"). (a): all of Z; (b-schur): col(B) both
    sides (its edit is U V (S′ − S) Vᵀ Uᵀ); (b-eig): col(B) and row(C)."""
    n = f["K"].shape[0]
    if arm == "a":
        I = np.eye(n)
        return I, I
    Ub = np.linalg.qr(f["Bq"])[0]
    if arm == "b":
        return Ub, Ub
    if arm == "b_eig":
        return Ub, np.linalg.qr(f["Cq"])[0]
    raise S2Error(f"no span for {arm}")


def ip(X: np.ndarray, Y: np.ndarray) -> float:
    return float(np.sum(X * Y))


def gain_core(f: Dict[str, np.ndarray], Kp: np.ndarray) -> np.ndarray:
    """g_a, g_b: K := c K with c = ‖K′‖ / ‖K‖, the arm's change in output gain and nothing else."""
    return f["K"] * (np.linalg.norm(Kp) / np.linalg.norm(f["K"]))


def control_core(f: Dict[str, np.ndarray], Kp: np.ndarray, span: Tuple[np.ndarray, np.ndarray],
                 rng: np.random.Generator) -> np.ndarray:
    """A doubly matched control (math check C8): K + α K + N⊥, α = ⟨D, K⟩ / ‖K‖² of the arm's edit
    D = K′ − K, N⊥ a Gaussian direction in the arm's span made orthogonal to K (to K's projection
    on the span, so it stays inside), of norm ‖D − α K‖. So ‖control − K‖ = ‖D‖ and
    ‖control‖ = ‖K′‖ exactly."""
    K = f["K"]
    D = Kp - K
    alpha = ip(D, K) / ip(K, K)
    rest = np.linalg.norm(D - alpha * K)
    Ww, Wr = span
    X = Ww @ rng.standard_normal((Ww.shape[1], Wr.shape[1])) @ Wr.T
    Kp_ = Ww @ (Ww.T @ K @ Wr) @ Wr.T
    if ip(Kp_, Kp_) > 0:
        X = X - ip(X, Kp_) / ip(Kp_, Kp_) * Kp_
    nx = np.linalg.norm(X)
    X = X * (rest / nx) if nx > 0 else X * 0
    return K + alpha * K + X


def sizes(K: np.ndarray, Kp: np.ndarray) -> Tuple[float, float]:
    """(‖K′ − K‖ / ‖K‖, ‖K′‖ / ‖K‖)."""
    n = np.linalg.norm(K)
    return float(np.linalg.norm(Kp - K) / n), float(np.linalg.norm(Kp) / n)


def size_summary(x: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(x, dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


# ---------------------------------------------------------------- per step: every head

class Heads:
    """Every head's frame at one step, from a loaded GPT-NeoX (float64 on the CPU)."""

    def __init__(self, model):
        self.layers = getattr(model, "gpt_neox", model).layers
        att0 = self.layers[0].attention
        self.H, self.k = model.config.num_attention_heads, att0.head_size
        self.frames: List[List[Dict]] = []
        self.const: List[np.ndarray] = []           # Σ_h c_h per layer (d,)
        self.ones: List[np.ndarray] = []            # u_h per layer (H, d): the write along 1
        for lay in self.layers:
            g = lay.input_layernorm.weight.detach().double().cpu().numpy()
            b = lay.input_layernorm.bias.detach().double().cpu().numpy()
            qkv = lay.attention.query_key_value.weight.detach().double().cpu().numpy().reshape(self.H, 3, self.k, -1)
            bv = lay.attention.query_key_value.bias.detach().double().cpu().numpy().reshape(self.H, 3, self.k)[:, 2]
            ow = lay.attention.dense.weight.detach().double().cpu().numpy()
            row, c, u = [], 0.0, []
            for h in range(self.H):
                WO, WV = ow[:, h * self.k:(h + 1) * self.k], qkv[h, 2]
                row.append(head_frame(WO, WV, g))
                c = c + WO @ (WV @ b + bv[h])
                u.append(g * (WV.T @ WO.mean(axis=0)))
            self.frames.append(row)
            self.const.append(np.asarray(c))
            self.ones.append(np.stack(u))

    def each(self):
        for l, row in enumerate(self.frames):
            for h, f in enumerate(row):
                yield l, h, f

    def cores(self, arm: str, draw: Optional[int] = None, step: int = 0) -> Tuple[List[np.ndarray], Dict]:
        """Per layer (H, 2k, 2k) cores for ``arm`` (``base``, ``a``, ``b``, ``b_eig``, ``g_a``, ``g_b``,
        ``ctl_a``, ``ctl_b`` with ``draw``), and the per-head sizes and info."""
        out, d_sz, k_sz, info = [], [], [], []
        for l, row in enumerate(self.frames):
            cs = []
            for h, f in enumerate(row):
                if arm in ("g_a", "g_b", "ctl_a", "ctl_b"):
                    src = arm[-1]
                    Kp, inf = arm_core(f, src)
                    if arm.startswith("g_"):
                        M = gain_core(f, Kp)
                    else:
                        rng = np.random.default_rng(seed_for("s2-ctl", step, src, draw, l, h))
                        M = control_core(f, Kp, arm_span(f, src), rng)
                    inf = {}
                else:
                    M, inf = arm_core(f, arm)
                a, b = sizes(f["K"], M)
                d_sz.append(a)
                k_sz.append(b)
                info.append({kk: v for kk, v in inf.items() if kk != "eig"})
                cs.append(M)
            out.append(np.stack(cs))
        return out, {"edit": d_sz, "norm": k_sz, "info": info}


def size_record(rec: Dict) -> Dict:
    out = {"edit": size_summary(rec["edit"]), "norm": size_summary(rec["norm"])}
    kept = [i.get("kept") for i in rec["info"] if "kept" in i]
    if kept:
        out["kept_heads"] = int(sum(bool(x) for x in kept))
    pairs = [i.get("pairs") for i in rec["info"] if "pairs" in i]
    if pairs:
        out["pairs"] = int(sum(pairs))
    return out


# ---------------------------------------------------------------- the hook (torch)

class Hook:
    """Replaces every head's V path by Σ_h P_h (X̂ Q_h) M_hᵀ Q_hᵀ + its write along 1 + its constant
    (module docstring). ``set(cores)`` takes per-layer (H, 2k, 2k) cores; ``close()`` restores."""

    def __init__(self, model, heads: Heads):
        import torch
        self.torch, self.model, self.heads = torch, model, heads
        self.layers = heads.layers
        p = next(model.parameters())
        self.dev, self.dt = p.device, p.dtype
        H, k = heads.H, heads.k
        self.orig = []
        self.Q, self.U, self.M = [], [], [None] * len(self.layers)
        self._x: Dict[int, object] = {}
        self.handles = []
        with torch.no_grad():
            for l, lay in enumerate(self.layers):
                att = lay.attention
                qw = att.query_key_value.weight.view(H, 3, k, -1)
                qb = att.query_key_value.bias.view(H, 3, k)
                self.orig.append({"vw": qw[:, 2].clone(), "vb": qb[:, 2].clone(),
                                  "ob": att.dense.bias.clone()})
                qw[:, 2] = 0
                qb[:, 2] = 0
                att.dense.bias.add_(torch.as_tensor(heads.const[l], device=self.dev, dtype=self.dt))
                self.Q.append(torch.as_tensor(np.stack([f["Q"] for f in heads.frames[l]]),
                                              device=self.dev, dtype=self.dt))            # (H, d, 2k)
                self.U.append(torch.as_tensor(heads.ones[l].T, device=self.dev, dtype=self.dt))  # (d, H)
                self.handles.append(lay.register_forward_pre_hook(self._pre(l), with_kwargs=True))
                self.handles.append(att.register_forward_hook(self._post(l, lay.input_layernorm)))
        self.set(heads.cores("base")[0])

    def set(self, cores: Sequence[np.ndarray]) -> None:
        self.M = [self.torch.as_tensor(c, device=self.dev, dtype=self.dt) for c in cores]

    def set_frame(self, Qs: Optional[Sequence[np.ndarray]] = None) -> None:
        """Per-layer (H, d, m) bases the cores are written in (G3′'s wider frame, m = 4k); ``None``
        restores the heads' own (d × 2k). The cores set next must match."""
        if not hasattr(self, "Q0"):
            self.Q0 = self.Q
        self.Q = self.Q0 if Qs is None else [self.torch.as_tensor(q, device=self.dev, dtype=self.dt)
                                             for q in Qs]

    def _pre(self, l):
        def f(mod, args, kwargs):
            self._x[l] = args[0] if args else kwargs["hidden_states"]
        return f

    def _post(self, l, ln):
        torch = self.torch

        def f(mod, args, out):
            # The weights' place differs by version: (out, present, weights) in transformers 4.44,
            # (out, weights) in 4.57 (`LESSONS.md` 3, B0). Taken by shape, (b, H, n, n).
            P = next((o for o in out[1:] if torch.is_tensor(o) and o.dim() == 4), None)
            if P is None:
                raise S2Error("the hook needs the attention weights: call with output_attentions=True "
                              "on an eager model")
            x = self._x.pop(l)
            mu = x.mean(-1, keepdim=True)
            var = ((x - mu) ** 2).mean(-1, keepdim=True)
            xh = (x - mu) / torch.sqrt(var + ln.eps)                # (b, n, d)
            Y = torch.einsum("bnd,hdm->bhnm", xh, self.Q[l])
            Y = P.to(Y.dtype) @ Y                                  # (b, H, n, 2k)
            Y = torch.einsum("bhnm,hjm->bhnj", Y, self.M[l])       # · Mᵀ
            add = torch.einsum("bhnj,hdj->bnd", Y, self.Q[l])
            s = torch.einsum("bhnm,bmh->bn", P.to(xh.dtype), xh @ self.U[l])   # the write along 1
            return (out[0] + add + s[..., None],) + tuple(out[1:])
        return f

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        H, k = self.heads.H, self.heads.k
        with self.torch.no_grad():
            for lay, o in zip(self.layers, self.orig):
                att = lay.attention
                att.query_key_value.weight.view(H, 3, k, -1)[:, 2] = o["vw"]
                att.query_key_value.bias.view(H, 3, k)[:, 2] = o["vb"]
                att.dense.bias.copy_(o["ob"])


def readback(hook: Hook, l: int, h: int) -> np.ndarray:
    """K′ (d × d, float64) from the core and Q as the hook holds them (float32)."""
    Q = hook.Q[l][h].double().cpu().numpy()
    M = hook.M[l][h].double().cpu().numpy()
    return Q @ M @ Q.T


# ---------------------------------------------------------------- the passes

def probe_ids() -> np.ndarray:
    """7d's repeated-random-token probe, 16 sequences (the first 8 are 7d's)."""
    rng = np.random.default_rng(EVAL_SEED)
    seqs = []
    for _ in range(N_SEQS):
        s = rng.integers(VOCAB[0], VOCAB[1], size=N_REP)
        seqs.append(np.concatenate([s, s]))
    return np.stack(seqs)


def n_viol(e: Sequence[float]) -> int:
    from core.metrics import energy_violation_severity
    return int(energy_violation_severity(list(e), rel_tol=REL_TOL)["n_violations"])


def e1_series(hs) -> List[float]:
    """A5's e1: E₁ on the unit sphere for each hidden state (float32 Gram)."""
    import torch
    from core.metrics import interaction_energies_batched
    out = []
    for h in hs:
        U = torch.nn.functional.normalize(h.to(torch.float32), p=2, dim=-1)
        out.append(interaction_energies_batched((U @ U.T).cpu().numpy(), [1.0])[1.0])
    return out


def centred(h):
    return h - h.mean(-1, keepdim=True)


class Inputs:
    """The v1 prompts' token ids and the probe, on the device."""

    def __init__(self, tok, device):
        import torch
        from core.config import PROMPTS
        from trackA_force.a1a import V1
        self.v1 = {}
        for p in V1:
            enc = tok(PROMPTS[p], return_tensors="pt", truncation=True, max_length=512)
            self.v1[p] = enc["input_ids"].to(device)
        self.probe = torch.as_tensor(probe_ids(), device=device)


def run_config(model, inp: Inputs, base_hs: Optional[Dict] = None, keep_hs: bool = False) -> Dict:
    """One configuration's readouts: V (summed violations over the 7 v1 prompts), the v1 NLL, G and
    D (per layer, mean over tokens and prompts of the centred ‖h − h_base‖ / ‖h_base‖, median over
    layers 1–24) when ``base_hs`` is given."""
    import torch
    out = {"e1": {}, "viol": {}, "nll": {}}
    hs_keep, dl = {}, []
    with torch.no_grad():
        for p, ids in inp.v1.items():
            o = model(ids, output_hidden_states=True, output_attentions=True)
            hs = [h[0] for h in o.hidden_states]
            e = e1_series(hs)
            out["e1"][p], out["viol"][p] = e, n_viol(e)
            lp = torch.log_softmax(o.logits[0, :-1].float(), -1)
            out["nll"][p] = float(-lp.gather(-1, ids[0, 1:, None]).mean())
            if keep_hs:
                hs_keep[p] = [h.detach().clone() for h in hs]
            if base_hs is not None:
                dl.append([float((centred(hs[L]) - centred(base_hs[p][L])).norm(dim=-1).div(
                    centred(base_hs[p][L]).norm(dim=-1)).mean()) for L in range(1, len(hs))])
            del o, hs
        o = model(inp.probe, output_attentions=True)
        lp = torch.log_softmax(o.logits[:, :-1].float(), -1)
        tl = lp.gather(-1, inp.probe[:, 1:, None])[..., 0]
        first, second = float(-tl[:, :N_REP - 1].mean()), float(-tl[:, N_REP - 1:].mean())
        del o
    out["V"] = int(sum(out["viol"].values()))
    out["G"], out["nll_first"], out["nll_second"] = first - second, first, second
    if base_hs is not None:
        per_layer = np.mean(np.asarray(dl), axis=0)               # mean over prompts (each a token mean)
        out["D_layers"] = per_layer.tolist()
        out["D"] = float(np.median(per_layer))
    if keep_hs:
        out["_hs"] = hs_keep
    return out


def run_meta() -> Dict:
    """What a record ran on: code, versions, device, the steps' revisions (`/challenge-pr` on #247)."""
    import torch
    import transformers
    from core.lm_loading import resolve_lm_entry
    return {"design": "trackS_subspace/design-S2.md", "code": code_sha(), "dtype": "float32",
            "tf32": False, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "revisions": {s: snapshot(resolve_lm_entry(f"pythia-410m-step{s}"))
                          for s in sorted(set(G0_STEPS) | {PREFLIGHT_STEP, G3_STEP})}}


def snapshot(entry: Dict) -> str:
    """``branch@commit`` of the cached checkpoint (the HF cache's refs file), or the branch alone."""
    import os
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    ref = hub / f"models--{entry['repo_id'].replace('/', '--')}" / "refs" / entry["revision"]
    return f"{entry['revision']}@{ref.read_text().strip()}" if ref.exists() else entry["revision"]


def load(step: int):
    import torch
    from trackB_circuit.b0_gate import load as b0_load
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (S2's passes are the GPU's, design 'Compute')")
    model, tok = b0_load(step, "cuda", torch.float32)
    return model, tok


def _save(path: Path, rec: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    tmp.rename(path)


def check_populated(r: Dict, what: str) -> None:
    """Refuse rather than degrade: a configuration's readouts are finite and full-sized."""
    bad = [p for p, e in r["e1"].items() if len(e) != N_LAYERS + 1 or not np.all(np.isfinite(e))]
    if bad or not np.isfinite(r["G"]) or len(r["e1"]) != 7:
        raise S2Error(f"{what}: not populated (e1 {bad}, G {r['G']}, prompts {len(r['e1'])})")


def strip(r: Dict) -> Dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


# ---------------------------------------------------------------- G0 (unit 1)

def rel_err(a, b) -> float:
    """max over rows of ‖a − b‖ / ‖b‖."""
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).max())


def wiring(model, inp: Inputs, heads: Heads) -> Dict:
    """G0 (i): the hook at the base core reproduces the model's logits and hidden states; (ii) arm
    (b-eig) by the hook equals (b-eig) written in W_V (logits and centred hidden states: the
    written model changes the write along 1, which the hook keeps and no LayerNorm reads)."""
    import torch
    rec = {"base": {}, "b_eig_written": {}}
    feeds = dict(inp.v1, probe=inp.probe)

    def fwd(ids, centre=False):                     # held on the CPU: 8 feeds' logits fill the GPU
        o = model(ids, output_hidden_states=True, output_attentions=True)
        out = (o.logits.float().cpu(), [(centred(h) if centre else h).cpu() for h in o.hidden_states])
        del o
        return out

    with torch.no_grad():
        ref = {p: fwd(ids) for p, ids in feeds.items()}
        hook = Hook(model, heads)
        for p, ids in feeds.items():
            lg, hs = fwd(ids)
            rec["base"][p] = {"logits": rel_err(lg, ref[p][0]),
                              "hidden": max(rel_err(a, b) for a, b in zip(hs, ref[p][1]))}
        del ref
        cores, _ = heads.cores("b_eig")
        hook.set(cores)
        via_hook = {p: fwd(ids, centre=True) for p, ids in feeds.items()}
        hook.close()
        del hook
        written = write_b_eig(model, heads)
        for p, ids in feeds.items():
            lg, hs = fwd(ids, centre=True)
            rec["b_eig_written"][p] = {"logits": rel_err(via_hook[p][0], lg),
                                       "hidden_centred": max(rel_err(a, b) for a, b in zip(via_hook[p][1], hs))}
        del via_hook
        restore_v(model, heads, written)
    worst = {k: max(max(v.values()) for v in rec[k].values()) for k in rec}
    rec["worst"] = worst
    rec["pass"] = bool(max(worst.values()) <= WIRING_TOL)
    return rec


def write_b_eig(model, heads: Heads) -> List[Dict]:
    """(b-eig) written into the weights: W_V′ = g(A) W_V, b_V′ keeping W_V β + b_V (the constant)."""
    import torch
    H, k = heads.H, heads.k
    saved = []
    with torch.no_grad():
        for l, lay in enumerate(heads.layers):
            att = lay.attention
            qw = att.query_key_value.weight.view(H, 3, k, -1)
            qb = att.query_key_value.bias.view(H, 3, k)
            saved.append({"vw": qw[:, 2].clone(), "vb": qb[:, 2].clone()})
            beta = lay.input_layernorm.bias.detach().double().cpu().numpy()
            for h, f in enumerate(heads.frames[l]):
                A = f["Cq"].T @ f["Bq"]
                lam, Qe = np.linalg.eig(A)
                if np.linalg.cond(Qe) > COND_MAX:
                    continue
                g = ((Qe * (lam.real / lam)[None, :]) @ np.linalg.inv(Qe)).real
                WV = np.array(qw[h, 2].double().cpu().numpy())     # a copy: float64 shares memory
                bV = np.array(qb[h, 2].double().cpu().numpy())
                qw[h, 2] = torch.as_tensor(g @ WV, device=qw.device, dtype=qw.dtype)
                qb[h, 2] = torch.as_tensor(WV @ beta + bV - g @ WV @ beta, device=qb.device, dtype=qb.dtype)
    return saved


def restore_v(model, heads: Heads, saved: List[Dict]) -> None:
    import torch
    H, k = heads.H, heads.k
    with torch.no_grad():
        for lay, s in zip(heads.layers, saved):
            att = lay.attention
            att.query_key_value.weight.view(H, 3, k, -1)[:, 2] = s["vw"]
            att.query_key_value.bias.view(H, 3, k)[:, 2] = s["vb"]


def dense_K(heads: Heads, l: int, h: int) -> np.ndarray:
    """K = Π W_O W_V diag(γ) Π, dense, from the frame's factors (Q Bq Cqᵀ Qᵀ, float64)."""
    f = heads.frames[l][h]
    return f["Q"] @ f["K"] @ f["Q"].T


def match_eigs(x: np.ndarray, y: np.ndarray) -> float:
    """Largest distance from each of x to a distinct element of y (greedy)."""
    y = list(y)
    worst = 0.0
    for v in sorted(x, key=lambda z: -abs(z)):
        j = int(np.argmin([abs(v - w) for w in y]))
        worst = max(worst, abs(v - y[j]))
        y.pop(j)
    return worst


def readbacks(model, heads: Heads, step: int) -> Dict:
    """G0 (iii)–(iv) on ``N_READBACK_HEADS`` heads, within ``READBACK_TOL`` of ‖K‖: the read-back K′
    (as the hook holds it, float32) equals sym K (a) and the exact K′ (b, b-eig); the exact core has
    the arm's eigenvalues, real for b; every head's first control has the arm's two norms. Beside,
    not gated: the float32 core's own eigenvalues. (b-schur) leaves near-repeated eigenvalues (gaps
    to 1e-6 ‖K‖ at 8000), which float32 rounding moves by about √eps ≈ 3e-4 ‖K‖ while the map
    itself is exact to 3e-8 (unit 1, `status-S.md` "S2 unit 1")."""
    rng = np.random.default_rng(seed_for("s2-g0-heads", step))
    picks = sorted(rng.choice(N_LAYERS * N_HEADS, N_READBACK_HEADS, replace=False).tolist())
    hook = Hook(model, heads)
    out = {"heads": [[p // N_HEADS, p % N_HEADS] for p in picks], "a": [], "b": [], "b_eig": [],
           "b_real": [], "b_map": [], "b_eig_map": [], "beside_f32": {"b": [], "b_eig": [], "b_real": []}}
    try:
        for arm in ("a", "b", "b_eig"):
            cores, _ = heads.cores(arm)
            hook.set(cores)
            for p in picks:
                l, h = divmod(p, N_HEADS)
                f = heads.frames[l][h]
                Kn = np.linalg.norm(f["K"])
                Kr = readback(hook, l, h)
                if arm == "a":
                    out["a"].append(float(np.linalg.norm(Kr - sym(dense_K(heads, l, h))) / Kn))
                    continue
                M, info = arm_core(f, arm)
                if info.get("kept"):
                    out[arm].append(None)
                    continue
                out[f"{arm}_map"].append(float(np.linalg.norm(Kr - f["Q"] @ M @ f["Q"].T) / Kn))
                want = info["eig"][np.abs(info["eig"]) > 1e-9 * Kn]
                for core, key in ((M, None), (hook.M[l][h].double().cpu().numpy(), "beside_f32")):
                    ev = np.linalg.eigvals(core)
                    got = ev[np.argsort(-np.abs(ev))][:want.size]
                    dst = out if key is None else out[key]
                    dst[arm].append(float(match_eigs(got, want) / Kn))
                    if arm == "b":
                        dst["b_real"].append(float(np.abs(got.imag).max() / Kn))
    finally:
        hook.close()
    ctl = []
    for arm in ARMS:
        for l, h, f in heads.each():
            Kp, _ = arm_core(f, arm)
            rng = np.random.default_rng(seed_for("s2-ctl", step, arm, 0, l, h))
            C = control_core(f, Kp, arm_span(f, arm), rng)
            Kn = np.linalg.norm(f["K"])
            ctl.append(max(abs(np.linalg.norm(C - f["K"]) - np.linalg.norm(Kp - f["K"])),
                           abs(np.linalg.norm(C) - np.linalg.norm(Kp))) / Kn)
    worst = {k: max([x for x in out[k] if x is not None], default=0.0)
             for k in ("a", "b", "b_eig", "b_real", "b_map", "b_eig_map")}
    worst["controls"] = float(max(ctl))
    out["beside_f32_worst"] = {k: max(v, default=0.0) for k, v in out["beside_f32"].items()}
    out["worst"] = worst
    if any(not [x for x in out[k] if x is not None] for k in ("a", "b", "b_eig")):
        raise S2Error("read-back: an arm has no head checked (all kept or none picked)")
    out["pass"] = bool(max(worst.values()) <= READBACK_TOL)
    return out


def all_sizes(heads: Heads, step: int) -> Dict:
    """Every arm's per-head sizes at ``step`` (design G0 "Recorded"), and their summaries."""
    rec = {}
    for arm in ("a", "b", "b_eig", "g_a", "g_b"):
        _, r = heads.cores(arm, step=step)
        rec[arm] = {"summary": size_record(r), "edit": r["edit"], "norm": r["norm"]}
        if arm in ("b", "b_eig"):
            rec[arm]["info"] = r["info"]
    return rec


def panel_v() -> Dict[int, int]:
    """The known answer: the panel's `p1_energy_nviol_b1` (A5's `panel_v`; design G0 quotes 0, 19,
    56, 56, 61, 57, 70 at the 7 steps)."""
    from trackA_force.a5_seeds import STEPS, panel_v as a5_panel
    return {s: int(round(v)) for s, v in zip(STEPS, a5_panel())}


def g0(out: Path) -> None:
    """Unit 1's G0: wiring at 0 and 8000, read-backs at 8000, base V at the 7 steps against the
    panel, every arm's sizes at the 7 steps and 8000. No arm's V or G is computed."""
    import gc
    import torch
    out.mkdir(parents=True, exist_ok=True)
    meta = run_meta()
    _save(out / "plan.json", meta)
    t0 = time.time()
    res = {"code": meta["code"], "wiring": {}, "readback": None, "base_v": {}, "sizes": {}}
    for step in sorted(set(G0_STEPS) | {PREFLIGHT_STEP}):
        model, tok = load(step)
        heads = Heads(model)
        inp = Inputs(tok, "cuda")
        if step in WIRING_STEPS:
            res["wiring"][step] = wiring(model, inp, heads)
            print(f"step {step}: wiring {res['wiring'][step]['worst']}", flush=True)
        if step == PREFLIGHT_STEP:
            res["readback"] = readbacks(model, heads, step)
            print(f"step {step}: read-back {res['readback']['worst']}", flush=True)
        if step in G0_STEPS:
            panel = panel_v()
            hook = Hook(model, heads)
            r = run_config(model, inp)
            hook.close()
            del hook
            check_populated(r, f"step {step} base")
            res["base_v"][step] = {"V": r["V"], "panel": panel[step], "viol": r["viol"], "G": r["G"]}
            print(f"step {step}: base V {r['V']} (panel {panel[step]}), G {r['G']:.3f}", flush=True)
        res["sizes"][step] = all_sizes(heads, step)
        print(f"step {step}: sizes " + ", ".join(f"{a} {v['summary']['edit']['median']:.2f}"
                                                 for a, v in res["sizes"][step].items()) +
              f" ({time.time() - t0:.0f}s)", flush=True)
        _save(out / "g0.json", res)
        del model, heads, inp
        gc.collect()
        torch.cuda.empty_cache()
    res["pass"] = {"wiring": all(w["pass"] for w in res["wiring"].values()),
                   "readback": bool(res["readback"]["pass"]),
                   "base_v": all(v["V"] == v["panel"] for v in res["base_v"].values())}
    res["pass"]["all"] = all(res["pass"].values())
    _save(out / "g0.json", res)
    print(f"G0 {res['pass']} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- the preflight at 8000 (unit 1)

def interpolate(model, target: Dict[str, object], orig: Dict[str, object], t: float) -> None:
    """Every parameter W := W_s + t (W_128 − W_s) (G3); ``target`` and ``orig`` on the CPU, one
    tensor moved at a time (both on the GPU ran out of memory at 8000)."""
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_((orig[n] + t * (target[n] - orig[n])).to(p.device))


def preflight_inputs(rec: Dict, arm: str) -> Dict:
    """`tools/design_preflight.py`'s input for V under ``arm`` at 8000 (design "The gate"): the
    arm's 40 controls as the null, its gain arm and G1's 41st draw at dose 0, G3's t as doses,
    ``decision_p`` = 1/41. ``trained`` is the base's V, on the statistic's own scale: V is one
    number per model, so no per-unit population of 20 exists (the tool refuses it, as the design
    foresaw). The first run gave the per-prompt counts (7–10 against a summed null of 30–60), so
    its band check compared two scales; `preflight-tool` re-reads the records with this one."""
    c = rec["configs"]
    return {"statistic": f"S2: V (E₁'s violations, 7 v1 prompts summed) under arm {arm} at "
                         f"{PREFLIGHT_STEP}, against {N_CTL} doubly matched controls",
            "direction": "less",
            "nulls": {"controls": [c[f"ctl_{arm}_{i}"]["V"] for i in range(N_CTL)]},
            "moved_marks": [],
            "trained": [c["base"]["V"]],
            "trained_step": str(PREFLIGHT_STEP),
            "scored_steps": [str(s) for s in SCORED_STEPS + BESIDE_STEPS],
            "planted": {"0": [c[f"g_{arm}"]["V"], c[f"ctl_{arm}_{N_CTL}"]["V"]],
                        **{str(t): [c[f"g3_{t}"]["V"]] for t in G3_T}},
            "bounds": [0, 7 * N_LAYERS],
            "n_null_draws": N_CTL,
            "decision_p": 1 / (N_CTL + 1)}


def preflight(out: Path) -> None:
    """At 8000 (off the scored grid): base, a, b, b-eig, g_a, g_b, 40 + 1 controls per arm and G3's
    three doses, each read on V, G and D; then the preflight tool per arm."""
    import gc
    import torch
    from transformers import AutoModelForCausalLM
    from core.lm_loading import resolve_lm_entry
    from core.models import from_pretrained_eager
    out.mkdir(parents=True, exist_ok=True)
    path = out / "preflight_runs.json"
    rec = {"code": code_sha(), "step": PREFLIGHT_STEP, "meta": run_meta(), "configs": {}, "sizes": {}}
    if path.exists():                          # resume (`LESSONS.md` 11)
        prev = json.loads(path.read_text())
        if prev["code"] != rec["code"]:
            raise S2Error(f"{path} was written by code {prev['code']}; not resuming")
        rec = prev
    t0 = time.time()
    model, tok = load(PREFLIGHT_STEP)
    heads = Heads(model)
    inp = Inputs(tok, "cuda")
    hook = Hook(model, heads)
    base = run_config(model, inp, keep_hs=True)
    base_hs = base.pop("_hs")
    check_populated(base, "base")
    rec["configs"]["base"] = base
    print(f"first record populated: base V {base['V']}, G {base['G']:.3f}", flush=True)
    names = ["a", "b", "b_eig", "g_a", "g_b"] + [f"ctl_{a}_{i}" for a in ARMS for i in range(N_CTL + 1)]
    for name in names:
        if name in rec["configs"]:
            continue
        if name.startswith("ctl_"):
            _, a, i = name.split("_")
            cores, sz = heads.cores(f"ctl_{a}", draw=int(i), step=PREFLIGHT_STEP)
        else:
            cores, sz = heads.cores(name, step=PREFLIGHT_STEP)
        hook.set(cores)
        r = run_config(model, inp, base_hs=base_hs)
        check_populated(r, name)
        rec["configs"][name] = r
        rec["sizes"][name] = size_record(sz)
        if len(rec["configs"]) % 10 == 0 or not name.startswith("ctl_"):
            print(f"{name}: V {r['V']}, G {r['G']:.3f}, D {r['D']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            _save(path, rec)
    hook.close()
    _save(path, rec)
    if not all(f"g3_{t}" in rec["configs"] for t in G3_T):
        e = resolve_lm_entry(f"pythia-410m-step{G3_STEP}")
        m128 = from_pretrained_eager(AutoModelForCausalLM, e["repo_id"], revision=e["revision"],
                                     torch_dtype=torch.float32).eval()
        target = {n: p.detach().clone() for n, p in m128.named_parameters()}
        del m128
        orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        for t in G3_T:
            interpolate(model, target, orig, t)
            r = run_config(model, inp, base_hs=base_hs)
            check_populated(r, f"g3 {t}")
            rec["configs"][f"g3_{t}"] = r
            print(f"g3 t {t}: V {r['V']}, G {r['G']:.3f}, D {r['D']:.3f}", flush=True)
        interpolate(model, target, orig, 0.0)
        del target, orig
    _save(path, rec)
    preflight_tool(out)
    del model, heads, inp, base_hs
    gc.collect()
    torch.cuda.empty_cache()
    print(f"preflight done ({time.time() - t0:.0f}s)", flush=True)


def preflight_tool(out: Path) -> None:
    """`tools/design_preflight.py` per arm on the records in ``preflight_runs.json`` (no GPU)."""
    from tools import design_preflight as dp
    rec = json.loads((out / "preflight_runs.json").read_text())
    reports = {}
    for arm in ARMS:
        vals = preflight_inputs(rec, arm)
        (out / f"preflight_values_{arm}.json").write_text(json.dumps(vals, indent=1) + "\n")
        rep = dp.preflight(dp.Inputs.from_dict(vals))
        body = json.dumps(rep, indent=1, default=float) + "\n"
        (out / f"preflight_report_{arm}.json").write_text(body)
        reports[arm] = {"sha256": hashlib.sha256(body.encode()).hexdigest(),
                        "refusals": [r["code"] for r in rep["refusals"]]}
        print(f"preflight {arm}: refusals {reports[arm]['refusals']}", flush=True)
    tool = json.loads((out / f"preflight_report_{ARMS[0]}.json").read_text())["tool_sha256"]
    _save(out / "preflight_summary.json", {"code": rec["code"], "s2_code": code_sha(), "tool_sha256": tool,
                                           "reports": reports,
                                           "summary": summarise(rec)})


def summarise(rec: Dict) -> Dict:
    """Per arm at 8000: the arm, its gain arm, its G1 draw, G3's doses and the controls' range on
    V, G and D (description for the freeze; 8000 is not scored)."""
    c = rec["configs"]
    out = {"base": {k: c["base"][k] for k in ("V", "G")}}
    for arm in ARMS:
        ctl = [c[f"ctl_{arm}_{i}"] for i in range(N_CTL)]
        row = {}
        for k in ("V", "G", "D"):
            v = np.array([x[k] for x in ctl])
            row[k] = {"arm": c[arm][k], "gain": c[f"g_{arm}"][k], "g1": c[f"ctl_{arm}_{N_CTL}"][k],
                      "ctl_min": float(v.min()), "ctl_median": float(np.median(v)), "ctl_max": float(v.max())}
        out[arm] = row
    out["b_eig"] = {k: c["b_eig"][k] for k in ("V", "G", "D")}
    out["g3"] = {str(t): {k: c[f"g3_{t}"][k] for k in ("V", "G", "D")} for t in G3_T if f"g3_{t}" in c}
    return out


# ---------------------------------------------------------------- unit 2: labels and readings (pure)

def label(x: float, ctl: Sequence[float]) -> str:
    """Against the controls: *lower* (below every one), *higher* (above every one), else *as its
    controls*; ties go to "as its controls" (design "Readings")."""
    c = np.asarray(ctl, dtype=float)
    if x < c.min():
        return "lower"
    if x > c.max():
        return "higher"
    return "as its controls"


def cell(x: float, ctl: Sequence[float], gain: Optional[float] = None, d: Optional[float] = None,
         d_ctl: Optional[Sequence[float]] = None) -> Dict:
    """One cell before its reading: the raw label, whether the arm is beyond its gain arm in the
    label's direction, and where its displacement D lies against the controls'."""
    c = np.asarray(ctl, dtype=float)
    out = {"value": x, "raw": label(x, c), "as_its_gain": False, "displacement": None,
           "ctl": {"min": float(c.min()), "median": float(np.median(c)), "max": float(c.max())}}
    if gain is not None:
        out["gain"] = {"value": gain, "label": label(gain, c)}
        out["as_its_gain"] = bool((out["raw"] == "lower" and not x < gain)
                                  or (out["raw"] == "higher" and not x > gain))
    if d is not None:
        dc = np.asarray(d_ctl, dtype=float)
        out["D"] = {"value": d, "ctl_min": float(dc.min()), "ctl_max": float(dc.max())}
        out["displacement"] = "further" if d > dc.max() else "less" if d < dc.min() else "within"
    return out


def r1_read(c: Dict, guard: bool = False) -> Dict:
    """R1's reading of a V cell. A label is read only beyond its gain arm. "Moved further" is
    written beside, read (Blocked 88 (a), the user's: G2 and G3 showed the guard withholds planted
    effects); ``guard=True`` is the frozen one-sided guard (Blocked 87 (a)), which withheld *lower*.
    ``counts``: read *lower* with V ≤ 7."""
    lab, read = c["raw"], True
    if c["as_its_gain"]:
        lab, read = f"{lab}, as its gain", False
    elif c["displacement"] in ("further", "less"):
        lab = f"{lab}, moved {c['displacement']}"
        read = not (guard and c["raw"] == "lower" and c["displacement"] == "further")
    return {"label": lab, "read": read, "counts": bool(read and c["raw"] == "lower" and c["value"] <= V_PLACED)}


def r2_read(c: Dict, g_base: float, guard: bool = False) -> Dict:
    """R2's reading of a G cell: *survives* (G ≥ 0.5 G_base, not read *lower*), *lost* (read
    *lower*, G < 0.5 G_base), else *weakened*. "As its gain" is not read; "moved further" is written
    beside, read (Blocked 88 (a)); ``guard=True``, the frozen guard, withheld *lost* and *weakened*."""
    lower = c["raw"] == "lower" and not c["as_its_gain"]
    half = c["value"] >= G_HALF * g_base
    out = "survives" if half and not lower else "lost" if lower and not half else "weakened"
    lab, read = out, True
    if c["as_its_gain"]:
        lab, read = f"{out}, as its gain", False
    elif c["displacement"] in ("further", "less"):
        lab = f"{out}, moved {c['displacement']}"
        read = not (guard and out != "survives" and c["displacement"] == "further")
    return {"label": lab, "read": read, "outcome": out}


def g2_draws(n: int = N_CTL) -> List[List[Tuple[int, int]]]:
    """G2's null (the user's, 2026-10-21; the frozen doubly matched one is only the zeroing itself):
    per draw, each member stands in for a non-member of its own layer, drawn without replacement
    within the draw. The same draws at every step."""
    from trackB_circuit.b2a import parse_head
    members = [parse_head(m) for m in G2_MEMBERS]
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed_for("s2-g2", i))
        taken: set = set(members)
        row = []
        for l, _ in members:
            h = int(rng.choice([h for h in range(N_HEADS) if (l, h) not in taken]))
            taken.add((l, h))
            row.append((l, h))
        out.append(row)
    return out


def zeroed(cores: Sequence[np.ndarray], lh: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
    """``cores`` with the heads ``lh``'s K set to 0 (their constant and write along 1 kept, as in
    every arm)."""
    out = [c.copy() for c in cores]
    for l, h in lh:
        out[l][h] = 0
    return out


def g3p_frame(f_s: Dict[str, np.ndarray], f_t: Dict[str, np.ndarray]):
    """G3′'s per-head frame: Q4 (d × 4k) spans both steps' write and read spaces, so K_s (as the
    frame's ``K``) and the step-128 K_t are both cores in it; the controls' span is the two write
    spaces against the two read spaces. Returns ``(frame, K_t's core, span)``."""
    Bs, Cs = f_s["Q"] @ f_s["Bq"], f_s["Q"] @ f_s["Cq"]
    Bt, Ct = f_t["Q"] @ f_t["Bq"], f_t["Q"] @ f_t["Cq"]
    Q4 = np.linalg.qr(np.hstack([Bs, Cs, Bt, Ct]))[0]
    bs, cs, bt, ct = (Q4.T @ X for X in (Bs, Cs, Bt, Ct))
    span = (np.linalg.qr(np.hstack([bs, bt]))[0], np.linalg.qr(np.hstack([cs, ct]))[0])
    return {"Q": Q4, "Bq": bs, "Cq": cs, "K": bs @ cs.T}, bt @ ct.T, span


# ---------------------------------------------------------------- unit 2: the passes (GPU)

def load_cpu(step: int):
    """A checkpoint on the CPU in float32 (G3's target, G3′'s frames)."""
    import torch
    from transformers import AutoModelForCausalLM
    from core.lm_loading import resolve_lm_entry
    from core.models import from_pretrained_eager
    e = resolve_lm_entry(f"pythia-410m-step{step}")
    return from_pretrained_eager(AutoModelForCausalLM, e["repo_id"], revision=e["revision"],
                                 torch_dtype=torch.float32).eval()


def perturb(model, orig: Dict[str, object], dnorm: Dict[str, float], t: float, seed: int) -> None:
    """G3's control: every parameter W := W_s + N, N Gaussian of norm t ‖W_128 − W_s‖ (per tensor)."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            w = orig[n].to(p.device)
            if dnorm[n] > 0:
                z = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
                w = w + z * (t * dnorm[n] / z.norm())
            p.copy_(w)


def _resume(path: Path, rec: Dict) -> Dict:
    if path.exists():                          # resume (`LESSONS.md` 11)
        prev = json.loads(path.read_text())
        if prev["code"] != rec["code"]:
            raise S2Error(f"{path} was written by code {prev['code']}; not resuming")
        return prev
    return rec


class _Pass:
    """One step's hooked configurations against its base, saved as they come."""

    def __init__(self, path: Path, rec: Dict, model, inp: Inputs, hook: Hook, t0: float):
        self.path, self.rec, self.model, self.inp, self.hook, self.t0 = path, rec, model, inp, hook, t0
        base = run_config(model, inp, keep_hs=True)
        self.base_hs = base.pop("_hs")
        check_populated(base, f"step {rec['step']} base")
        old = rec["configs"].get("base")
        if old is not None and (old["V"] != base["V"] or abs(old["G"] - base["G"]) > 1e-6 * abs(base["G"])):
            raise S2Error(f"step {rec['step']}: the base moved since the record (V {old['V']} → {base['V']})")
        rec["configs"]["base"] = base
        print(f"step {rec['step']}: base populated, V {base['V']}, G {base['G']:.3f}", flush=True)

    def has(self, name: str) -> bool:
        return name in self.rec["configs"]

    def run(self, name: str, cores=None, sz: Optional[Dict] = None, hooked: bool = True) -> Dict:
        if hooked:
            self.hook.set(cores)
        r = run_config(self.model, self.inp, base_hs=self.base_hs)
        check_populated(r, f"step {self.rec['step']} {name}")
        self.rec["configs"][name] = r
        if sz is not None:
            self.rec["sizes"][name] = size_record(sz)
        n = len(self.rec["configs"])
        if n % 20 == 0 or not ("_ctl_" in name or name.startswith("ctl_")):
            print(f"step {self.rec['step']} {name}: V {r['V']}, G {r['G']:.3f}, D {r['D']:.3f} "
                  f"({n} configs, {time.time() - self.t0:.0f}s)", flush=True)
            _save(self.path, self.rec)
        return r


def gates(out: Path) -> None:
    """Unit 2's gate pass (design "The gate"), before any arm is run: at the 7 steps the base, the
    gain arms and 41 doubly matched controls per arm (draw 40 is G1's); G2 at 512 and R2's steps;
    G3′ and G3 at the scored steps. Arms a, b and b-eig are not computed here."""
    import gc
    import torch
    out.mkdir(parents=True, exist_ok=True)
    _save(out / "plan.json", {**run_meta(), "steps": UNIT2_STEPS, "g2_members": G2_MEMBERS,
                              "g2_draws": g2_draws()})
    t0 = time.time()
    m128 = load_cpu(G3_STEP)
    h128 = Heads(m128)
    target = {n: p.detach().clone() for n, p in m128.named_parameters()}
    del m128
    for step in UNIT2_STEPS:
        path = out / f"gates_{step}.json"
        rec = _resume(path, {"code": code_sha(), "step": step, "configs": {}, "sizes": {}})
        if rec.get("done"):
            continue
        model, tok = load(step)
        heads = Heads(model)
        inp = Inputs(tok, "cuda")
        hook = Hook(model, heads)
        ps = _Pass(path, rec, model, inp, hook, t0)
        names = [f"g_{a}" for a in ARMS] + [f"ctl_{a}_{i}" for a in ARMS for i in range(N_CTL + 1)]
        for name in names:
            if ps.has(name):
                continue
            if name.startswith("ctl_"):
                _, a, i = name.split("_")
                cores, sz = heads.cores(f"ctl_{a}", draw=int(i), step=step)
            else:
                cores, sz = heads.cores(name, step=step)
            ps.run(name, cores, sz)
        if step in G2_STEPS:
            from trackB_circuit.b2a import parse_head
            base_cores = heads.cores("base")[0]
            if not ps.has("g2"):
                ps.run("g2", zeroed(base_cores, [parse_head(m) for m in G2_MEMBERS]))
            for i, row in enumerate(g2_draws()):
                if not ps.has(f"g2_ctl_{i}"):
                    ps.run(f"g2_ctl_{i}", zeroed(base_cores, row))
        if step in SCORED_STEPS:
            fr = [[g3p_frame(f, h128.frames[l][h]) for h, f in enumerate(row)]
                  for l, row in enumerate(heads.frames)]
            hook.set_frame([np.stack([x[0]["Q"] for x in row]) for row in fr])
            if not ps.has("g3p_base"):
                r = ps.run("g3p_base", [np.stack([x[0]["K"] for x in row]) for row in fr])
                if r["D"] > G3P_BASE_TOL:
                    raise S2Error(f"step {step}: G3′'s frame at K_s is not the base (D {r['D']:.1e})")
            if not ps.has("g3p"):
                ps.run("g3p", [np.stack([x[1] for x in row]) for row in fr],
                       {"edit": [sizes(x[0]["K"], x[1])[0] for row in fr for x in row],
                        "norm": [sizes(x[0]["K"], x[1])[1] for row in fr for x in row], "info": []})
            for i in range(N_CTL):
                if not ps.has(f"g3p_ctl_{i}"):
                    ps.run(f"g3p_ctl_{i}", [np.stack([
                        control_core(x[0], x[1], x[2], np.random.default_rng(seed_for("s2-g3p", step, i, l, h)))
                        for h, x in enumerate(row)]) for l, row in enumerate(fr)])
            hook.set_frame(None)
            if not ps.has("g3p_gain"):
                ps.run("g3p_gain", [np.stack([gain_core(f, fr[l][h][1]) for h, f in enumerate(row)])
                                    for l, row in enumerate(heads.frames)])
            del fr
        hook.close()
        if step in SCORED_STEPS:                   # the whole model: no hook
            orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            dnorm = {n: float((target[n] - orig[n]).norm()) for n in orig}
            for t in G3_T:
                if not ps.has(f"g3_{t}"):
                    interpolate(model, target, orig, t)
                    ps.run(f"g3_{t}", hooked=False)
                for i in range(N_CTL):
                    if not ps.has(f"g3_{t}_ctl_{i}"):
                        perturb(model, orig, dnorm, t, seed_for("s2-g3", step, t, i))
                        ps.run(f"g3_{t}_ctl_{i}", hooked=False)
            interpolate(model, target, orig, 0.0)
            del orig
        rec["done"] = True
        _save(path, rec)
        print(f"step {step}: gates done, {len(rec['configs'])} configs ({time.time() - t0:.0f}s)", flush=True)
        del model, heads, inp, hook, ps
        gc.collect()
        torch.cuda.empty_cache()
    print(f"gates done ({time.time() - t0:.0f}s)", flush=True)


def _records(out: Path, kind: str) -> Tuple[Dict[int, Dict], Dict[str, str]]:
    recs, shas = {}, {}
    for s in UNIT2_STEPS:
        p = out / f"{kind}_{s}.json"
        if not p.exists():
            raise S2Error(f"{p} missing")
        body = p.read_bytes()
        recs[s] = json.loads(body)
        if not recs[s].get("done"):
            raise S2Error(f"{p} not done")
        shas[p.name] = hashlib.sha256(body).hexdigest()[:12]
    return recs, shas


def _ctl(c: Dict, prefix: str, key: str) -> List[float]:
    return [c[f"{prefix}{i}"][key] for i in range(N_CTL)]


def gate_read(out: Path) -> None:
    """G1, G1′, G2, G3, G3′ from the gate pass (design "The gate", unit 2; no GPU)."""
    g, shas = _records(out, "gates")
    res = {"code": code_sha(), "records": shas}
    g1 = {"R1": {}, "R2": {}}
    g1p = {"V": {}, "G": {}}
    for a in ARMS:
        for s in UNIT2_STEPS:
            c = g[s]["configs"]
            for key, cells in (("V", "R1"), ("G", "R2")):
                ctl = _ctl(c, f"ctl_{a}_", key)
                g1p[key][f"{a}@{s}"] = cell(c[f"g_{a}"][key], ctl)
                if s in (SCORED_STEPS if key == "V" else R2_STEPS):
                    g1[cells][f"{a}@{s}"] = label(c[f"ctl_{a}_{N_CTL}"][key], ctl)
    n1 = {k: sum(v != "as its controls" for v in g1[k].values()) for k in g1}
    res["G1"] = {**g1, "labelled": n1, "pass": n1["R1"] <= 2 and n1["R2"] <= 2}
    res["G1p"] = g1p
    g2 = {}
    for s in G2_STEPS:
        c = g[s]["configs"]
        cl = cell(c["g2"]["G"], _ctl(c, "g2_ctl_", "G"), d=c["g2"]["D"], d_ctl=_ctl(c, "g2_ctl_", "D"))
        g2[str(s)] = {**cl, "G_base": c["base"]["G"], **r2_read(cl, c["base"]["G"]),
                      "frozen_guard": r2_read(cl, c["base"]["G"], guard=True)}
    lost = sum(g2[str(s)]["read"] and g2[str(s)]["outcome"] == "lost" for s in R2_STEPS)
    lost0 = sum(g2[str(s)]["frozen_guard"]["read"] and g2[str(s)]["outcome"] == "lost" for s in R2_STEPS)
    res["G2"] = {"cells": g2, "lost": lost, "pass": lost >= 3, "frozen_guard": {"lost": lost0, "pass": lost0 >= 3}}
    g3 = {}
    for s in SCORED_STEPS:
        c = g[s]["configs"]
        for t in G3_T:
            pre = f"g3_{t}_ctl_"
            cl = cell(c[f"g3_{t}"]["V"], _ctl(c, pre, "V"), d=c[f"g3_{t}"]["D"], d_ctl=_ctl(c, pre, "D"))
            g3[f"{t}@{s}"] = {**cl, **r1_read(cl), "G": c[f"g3_{t}"]["G"], "frozen_guard": r1_read(cl, guard=True)}
    n3 = sum(g3[f"1.0@{s}"]["counts"] for s in SCORED_STEPS)
    n30 = sum(g3[f"1.0@{s}"]["frozen_guard"]["counts"] for s in SCORED_STEPS)
    mono = {str(s): g3[f"1.0@{s}"]["ctl"]["median"] <= V_PLACED for s in SCORED_STEPS}
    res["G3"] = {"cells": g3, "counts_at_1": n3, "pass": n3 >= 4, "controls_monotone_at_1": mono,
                 "frozen_guard": {"counts_at_1": n30, "pass": n30 >= 4}}
    g3p = {}
    for s in SCORED_STEPS:
        c = g[s]["configs"]
        cl = cell(c["g3p"]["V"], _ctl(c, "g3p_ctl_", "V"), gain=c["g3p_gain"]["V"], d=c["g3p"]["D"],
                  d_ctl=_ctl(c, "g3p_ctl_", "D"))
        g3p[str(s)] = {**cl, **r1_read(cl), "frozen_guard": r1_read(cl, guard=True), "G": c["g3p"]["G"],
                       "base_D": c["g3p_base"]["D"],
                       "sizes": g[s]["sizes"].get("g3p")}
    n3p = sum(v["counts"] for v in g3p.values())
    res["G3p"] = {"cells": g3p, "counts": n3p, "in_reach": n3p >= 4}
    _save(out / "gate_read.json", res)
    print(f"G1 labelled {n1} (pass {res['G1']['pass']}); G2 lost at {lost} of 4 (pass {res['G2']['pass']}); "
          f"G3 at t 1 counts {n3} of 5 (pass {res['G3']['pass']}); G3′ counts {n3p} of 5", flush=True)
    for k, v in g3.items():
        print(f"  G3 {k}: V {v['value']} ctl [{v['ctl']['min']:.0f}, {v['ctl']['median']:.0f}, "
              f"{v['ctl']['max']:.0f}] D {v['displacement']} → {v['label']}", flush=True)
    for k, v in g3p.items():
        print(f"  G3′ {k}: V {v['value']} ctl [{v['ctl']['min']:.0f}, {v['ctl']['max']:.0f}] gain "
              f"{v['gain']['value']} D {v['displacement']} → {v['label']}", flush=True)
    for k, v in g2.items():
        print(f"  G2 {k}: G {v['value']:.2f} (base {v['G_base']:.2f}) ctl [{v['ctl']['min']:.2f}, "
              f"{v['ctl']['max']:.2f}] D {v['displacement']} → {v['label']}", flush=True)


def arms(out: Path) -> None:
    """Unit 2's arms a, b (and b-eig beside) at the 7 steps, after the gates are read (design: G1–G3,
    then R1 and R2). The base is re-run and must equal the gate pass's."""
    import gc
    import torch
    if not (out / "gate_read.json").exists():
        raise SystemExit(f"refusing: {out / 'gate_read.json'} missing (the gates are read before any arm)")
    gates_rec, _ = _records(out, "gates")
    t0 = time.time()
    for step in UNIT2_STEPS:
        path = out / f"arms_{step}.json"
        rec = _resume(path, {"code": code_sha(), "step": step, "meta": run_meta(), "configs": {}, "sizes": {}})
        if rec.get("done"):
            continue
        rec["configs"].setdefault("base", gates_rec[step]["configs"]["base"])
        model, tok = load(step)
        heads = Heads(model)
        inp = Inputs(tok, "cuda")
        hook = Hook(model, heads)
        ps = _Pass(path, rec, model, inp, hook, t0)
        for name in ("a", "b", "b_eig"):
            if not ps.has(name):
                cores, sz = heads.cores(name, step=step)
                ps.run(name, cores, sz)
        hook.close()
        rec["done"] = True
        _save(path, rec)
        del model, heads, inp, hook, ps
        gc.collect()
        torch.cuda.empty_cache()
    print(f"arms done ({time.time() - t0:.0f}s)", flush=True)


def read(out: Path) -> None:
    """R1 and R2 (design "Readings", frozen by Blocked 87 (a)): each arm against its 40 controls, its
    gain arm and the one-sided guard; R2 read under arm b only (a as description); refusals from the
    gates applied. Steps 0 and 256, 512's R2 and (b-eig) beside."""
    gr = json.loads((out / "gate_read.json").read_text())
    g, gshas = _records(out, "gates")
    ar, ashas = _records(out, "arms")
    res = {"code": code_sha(), "records": {**gshas, **ashas},
           "gates": {k: gr[k]["pass"] for k in ("G1", "G2", "G3")}, "G3p_in_reach": gr["G3p"]["in_reach"],
           "R1": {}, "R2": {}, "beside": {}}
    g3_mono = not gr["G3"]["pass"] and any(gr["G3"]["controls_monotone_at_1"].values())
    for a in ARMS:
        cells = {}
        for s in UNIT2_STEPS:
            c, x = g[s]["configs"], ar[s]["configs"][a]
            cl = cell(x["V"], _ctl(c, f"ctl_{a}_", "V"), gain=c[f"g_{a}"]["V"], d=x["D"],
                      d_ctl=_ctl(c, f"ctl_{a}_", "D"))
            rr = r1_read(cl)
            if g3_mono and cl["ctl"]["median"] <= V_PLACED:
                rr = {"label": f"refused (G3: controls' median V ≤ {V_PLACED})", "read": False, "counts": False}
            cells[str(s)] = {**cl, **rr, "scored": s in SCORED_STEPS}
        n = sum(cells[str(s)]["counts"] for s in SCORED_STEPS)
        res["R1"][a] = {"cells": cells, "counts": n,
                        "answer": "the break goes with it" if n >= 4 else "not shown"}
    for a in ARMS:
        cells = {}
        for s in UNIT2_STEPS:
            c, x = g[s]["configs"], ar[s]["configs"][a]
            cl = cell(x["G"], _ctl(c, f"ctl_{a}_", "G"), gain=c[f"g_{a}"]["G"], d=x["D"],
                      d_ctl=_ctl(c, f"ctl_{a}_", "D"))
            rr = r2_read(cl, c["base"]["G"])
            if a == "a":
                rr = {**rr, "read": False, "label": f"{rr['label']} (description: Blocked 87 (a))"}
            elif not gr["G2"]["pass"]:
                rr = {**rr, "read": False, "label": f"{rr['label']} (refused: G2 failed)"}
            cells[str(s)] = {**cl, **rr, "G_base": c["base"]["G"], "scored": s in R2_STEPS}
        res["R2"][a] = {"cells": cells}
    for s in UNIT2_STEPS:
        x = ar[s]["configs"]["b_eig"]
        res["beside"][str(s)] = {"b_eig": {k: x[k] for k in ("V", "G", "D")},
                                 "sizes": {k: ar[s]["sizes"][k] for k in ("a", "b", "b_eig")}}
    _save(out / "readings.json", res)
    for a in ARMS:
        print(f"R1 {a}: {res['R1'][a]['answer']} ({res['R1'][a]['counts']} of 5)", flush=True)
        for s, v in res["R1"][a]["cells"].items():
            print(f"  {s}: V {v['value']} ctl [{v['ctl']['min']:.0f}, {v['ctl']['median']:.0f}, {v['ctl']['max']:.0f}]"
                  f" gain {v['gain']['value']} D {v['D']['value']:.3f} [{v['D']['ctl_min']:.3f}, "
                  f"{v['D']['ctl_max']:.3f}] → {v['label']}", flush=True)
    for a in ARMS:
        for s, v in res["R2"][a]["cells"].items():
            print(f"R2 {a} {s}: G {v['value']:.2f} (base {v['G_base']:.2f}) ctl [{v['ctl']['min']:.2f}, "
                  f"{v['ctl']['max']:.2f}] gain {v['gain']['value']:.2f} → {v['label']}", flush=True)


# ---------------------------------------------------------------- R1b, R2 (weights only, beside)

def overlap(U: np.ndarray, W: np.ndarray) -> float:
    """tr(P_U P_W) / min(dim U, dim W)."""
    return float(np.sum((U.T @ W) ** 2) / min(U.shape[1], W.shape[1]))


def shuffle_l1(bases: Sequence[np.ndarray], n_draw: int, seed: int, pool: int = N_HEADS,
               device: str = "cuda") -> np.ndarray:
    """λ₁ of the mean projector of ``n_draw`` random ``pool``-head sets drawn from ``bases`` (the
    step's 384, one family): S1's layer-shuffle null (`design-S1.md` R2). Each basis is padded to
    the family's widest with zero columns, which add nothing to P̄."""
    import torch
    r = max(b.shape[1] for b in bases)
    if r == 0:
        return np.zeros(n_draw)
    pad = np.zeros((len(bases), bases[0].shape[0], r))
    for i, b in enumerate(bases):
        pad[i, :, :b.shape[1]] = b
    Bt = torch.as_tensor(pad, device=device)
    rng = np.random.default_rng(seed)
    out = np.zeros(n_draw)
    batch = 100
    for s in range(0, n_draw, batch):
        m = min(batch, n_draw - s)
        idx = torch.as_tensor(np.stack([rng.choice(len(bases), pool, replace=False) for _ in range(m)]),
                              device=device)
        B = Bt[idx].permute(0, 2, 1, 3).reshape(m, Bt.shape[1], pool * r)
        G = B.transpose(1, 2) @ B
        out[s:s + m] = (torch.linalg.eigvalsh(G)[:, -1] / pool).cpu().numpy()
    return out


def haar_overlap(width: int, n_draw: int, seed: int, d: int = D_MODEL) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(n_draw)
    for i in range(n_draw):
        U = np.linalg.qr(rng.standard_normal((d, width)))[0]
        W = np.linalg.qr(rng.standard_normal((d, width)))[0]
        out[i] = overlap(U, W)
    return out


def beside(out: Path) -> None:
    """R1b and R2 at S1's 8 steps (S1's raw frame, r = 16, S = P̄'s top 10; description)."""
    from trackS_subspace import s1
    out.mkdir(parents=True, exist_ok=True)
    path = out / "beside.json"
    res = {"code": code_sha(), "s1_code": s1.code_sha(), "s_width": s1.S_WIDTH, "n_shuffle": N_SHUFFLE,
           "r1b": {}, "r2": {}}
    if path.exists():
        prev = json.loads(path.read_text())
        if prev["code"] != res["code"]:
            raise S2Error(f"{path} was written by code {prev['code']}; not resuming")
        res = prev
    ho = haar_overlap(s1.S_WIDTH, 2000, seed_for("s2-r1b-haar"))
    res["r1b_haar"] = {"mean": float(ho.mean()), "q999": float(np.quantile(ho, 0.999)),
                       "max": float(ho.max()), "exact_mean": s1.S_WIDTH / D_MODEL}
    t0 = time.time()
    for s in S1_STEPS:
        if str(s) in res["r2"]:
            continue
        hb = s1.step_bases(s)
        S = {f: s1.shared_space([x[f] for x in hb]) for f in s1.FAMILIES}
        res["r1b"][str(s)] = {
            "overlap": {f"{a}|{b}": overlap(S[a], S[b]) for a, b in R1B_PAIRS},
            "top_coords": {f: [np.argsort(-np.abs(S[f][:, i]))[:3].tolist() for i in range(s1.S_WIDTH)]
                           for f in s1.FAMILIES}}
        r2 = {}
        for f in s1.FAMILIES:
            bs = [x[f] for x in hb]
            null = shuffle_l1(bs, N_SHUFFLE, seed_for("s2-r2-shuffle", s, f))
            T = float(null.max())
            rows = []
            for l in range(N_LAYERS):
                ev = s1.mean_projector_eigs(bs[l * N_HEADS:(l + 1) * N_HEADS])
                l1 = float(ev[0]) if ev.size else 0.0
                rows.append({"l1": l1, "count": int((ev > T).sum()),
                             "p": float((1 + (null >= l1).sum()) / (N_SHUFFLE + 1))})
            r2[f] = {"shuffle_max": T, "shuffle_q99": float(np.quantile(null, 0.99)), "layers": rows,
                     "layers_counted": int(sum(r["count"] > 0 for r in rows))}
        res["r2"][str(s)] = r2
        _save(path, res)
        print(f"step {s}: R1b " + ", ".join(f"{k} {v:.3f}" for k, v in res["r1b"][str(s)]["overlap"].items())
              + "; R2 layers counted " + ", ".join(f"{f} {r2[f]['layers_counted']}" for f in s1.FAMILIES)
              + f" ({time.time() - t0:.0f}s)", flush=True)
    print(f"beside done ({time.time() - t0:.0f}s)", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("g0", "preflight", "preflight-tool", "beside", "gates", "gate-read",
                                    "arms", "read"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    {"g0": g0, "preflight": preflight, "preflight-tool": preflight_tool, "beside": beside, "gates": gates,
     "gate-read": gate_read, "arms": arms, "read": read}[a.cmd](a.out)


if __name__ == "__main__":
    main()
