"""core/nudge.py — the nudge harness (Track B, B0; `trackB_circuit/design-B.md` "B0").

Move a block's input (``resid_pre`` at layer ℓ) by ``±ε·d`` at a token set T and read what
follows. With ``f`` any readout, the paired passes give

    A(ε) = (f(x+εd) − f(x−εd)) / 2 = ε·J d + O(ε³)          (odd in d)
    S(ε) = (f(x+εd) + f(x−εd)) / 2 − f(x) = ε²·H[d,d]/2 + O(ε⁴)   (even in d)

(`tools/math_checks/b0_antisymmetric_nudge.py`). ``ε = c·r`` with ``r`` the site's RMS token
norm and ``c`` swept over :data:`C_GRID`; a direction's readout is ranked among :data:`K_RANDOM`
seeded random directions' at the same site, ε and readout.

**The mechanism.** :class:`Stack` runs a GPT-NeoX causal LM's blocks in its own loop from a
given block input, rather than through a forward hook on the HF forward: one loop gives the
nudge (added to block ℓ's input), truncated forwards (stop after a block, or keep only the
first m tokens: causal, so exact for every kept position) and block-only passes from a cached
input (G2's JVP, B1). The loop calls each ``GPTNeoXLayer`` exactly as
``GPTNeoXModel.forward`` does with no padding; eager attention applies its own causal mask, so
no attention mask is passed. The gate's G0 arm checks it against the HF forward.

**NUMERICS** (design-B.md): every pass of one comparison (base, ±, nulls) is one batch, or
batches of one shape, in one process, on one device, in one dtype. Nothing here moves a tensor
across devices mid-comparison; :func:`numerics_record` is what a run records.

Torch is imported at use, so the pure parts (seeds, directions, positions, the A/S algebra, the
rank and the turn-on) run in the dependency-free test tier.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

C_GRID = (0.01, 0.03, 0.1, 0.3)     # ε = c·r, all four always run
K_RANDOM = 32                        # random directions per (prompt, site): rank floor 1/(K+1)
FLOOR_MULT = 10.0                    # a readout under FLOOR_MULT × the floor is "below floor"


class NudgeError(RuntimeError):
    """A refusal: the harness would otherwise return something it cannot vouch for."""


# ---------------------------------------------------------------- seeds, directions, positions

def seed_for(*parts) -> int:
    """A 63-bit seed from the parts' text; the same parts give the same seed in any process."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def random_directions(dim: int, k: int, seed: int) -> np.ndarray:
    """``k`` Gaussian directions in ``R^dim``, each normalised; float64, shape (k, dim)."""
    g = np.random.default_rng(seed).standard_normal((k, dim))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def draw_positions(key: str, n: int, m: int = 32, lo: int = 8, tail: int = 5) -> np.ndarray:
    """``m`` distinct nudge positions ``j`` with ``lo ≤ j`` and ``j + tail ≤ n − 1``, drawn once
    from ``key`` (B1: ``j ≥ 8`` with ``j + 5`` inside the prompt); sorted."""
    hi = n - 1 - tail
    if hi - lo + 1 < m:
        raise NudgeError(f"{key}: {n} tokens leave {max(hi - lo + 1, 0)} positions, {m} asked")
    rng = np.random.default_rng(seed_for("positions", key))
    return np.sort(rng.choice(np.arange(lo, hi + 1), size=m, replace=False))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise NudgeError("a zero or non-finite direction cannot be normalised")
    return v / n


def rms_norm(x) -> float:
    """``r = sqrt(mean_i ‖x_i‖²)`` over the tokens of one prompt's site, (n, D)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt((x ** 2).sum(-1).mean()))


# ---------------------------------------------------------------- the paired-pass algebra

def signed_rows(dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The batch's row layout: row 0 base, then ``+d₁, −d₁, +d₂, −d₂, …``. Returns the per-row
    direction index (−1 for base) and sign (0 for base)."""
    k = len(dirs)
    idx = np.concatenate([[-1], np.repeat(np.arange(k), 2)])
    sign = np.concatenate([[0], np.tile([1, -1], k)])
    return idx, sign


def split_rows(out):
    """``(base, plus, minus)`` from a batch laid out by :func:`signed_rows` (first axis)."""
    return out[0], out[1::2], out[2::2]


def anti_sym(plus, minus, base):
    """``A = (f₊ − f₋)/2``, ``S = (f₊ + f₋)/2 − f₀`` (works on numpy arrays and tensors)."""
    return (plus - minus) / 2, (plus + minus) / 2 - base


def exact_rank(stat: float, null: Sequence[float]) -> int:
    """1 + the number of null values strictly above ``stat`` (1 = largest; ties favour the
    direction, so a rank-1 rejection needs no null to exceed it)."""
    return 1 + int(np.sum(np.asarray(null, dtype=np.float64) > stat))


def turn_on(cs: Sequence[float], absA: Sequence[float], floors: Sequence[float],
            mult: float = FLOOR_MULT) -> Optional[float]:
    """The smallest ``c`` whose ``|A|`` exceeds ``mult ×`` that ``c``'s floor, else None."""
    for c, a, f in sorted(zip(cs, absA, floors)):
        if a > mult * f:
            return float(c)
    return None


# ---------------------------------------------------------------- the stack (torch)

class Stack:
    """A GPT-NeoX causal LM's blocks, run in the harness's own loop.

    ``run(x, start)`` takes block ``start``'s input ``x`` (B, n, D) and returns, for the layers in
    ``keep``, the input each block received (``resid_pre``; ``keep`` may include ``L``, the final
    residual before the final LayerNorm), and with ``loss_ids`` the next-token loss at
    ``loss_at``. ``delta`` (B, n, D) is added to ``x`` first: that is the nudge.
    """

    def __init__(self, model):
        import torch
        self.torch = torch
        self.model = model
        body = getattr(model, "gpt_neox", None)
        if body is None or not hasattr(model, "embed_out"):
            raise NudgeError(f"{type(model).__name__}: not a GPT-NeoX causal LM")
        if getattr(body.config, "_attn_implementation", "eager") != "eager":
            raise NudgeError("attention must be eager: the loop relies on its own causal mask")
        self.body, self.layers = body, body.layers
        self.L = len(self.layers)
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype

    def embed(self, ids: Sequence[int]):
        """Block 0's input for one prompt, (1, n, D)."""
        torch = self.torch
        t = torch.as_tensor([list(ids)], device=self.device)
        return self.body.emb_dropout(self.body.embed_in(t))

    def block(self, l: int, x):
        """Block ``l`` on ``x`` (B, n, D): the call ``GPTNeoXModel.forward`` makes, unpadded.

        The causal mask is always passed, additive (0 on and below the diagonal, the dtype's min
        above), the form HF's own forward builds: transformers 4.44's eager attention also masks
        by itself, 4.57's masks only by what it is given, so leaving it out there would let every
        token see later ones with no error. 4.57's layers also take the rotary ``(cos, sin)``
        from the model (``position_embeddings``); 4.44's compute it inside."""
        torch = self.torch
        n = x.shape[1]
        pos = torch.arange(n, device=x.device)[None]
        mask = torch.full((n, n), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)
        kw = {"attention_mask": mask.triu(1)[None, None], "position_ids": pos, "use_cache": False}
        rot = getattr(self.body, "rotary_emb", None)
        if rot is not None:
            kw["position_embeddings"] = rot(x, pos)
        out = self.layers[l](x, **kw)
        return out[0] if isinstance(out, tuple) else out

    def run(self, x, start: int, stop: Optional[int] = None, keep: Iterable[int] = (),
            delta=None, loss_ids: Optional[Sequence[int]] = None,
            loss_at: Optional[Sequence[int]] = None) -> Dict:
        stop = self.L if stop is None else stop
        keep = set(keep)
        if not 0 <= start <= stop <= self.L or any(not start <= m <= stop for m in keep):
            raise NudgeError(f"start {start}, stop {stop}, keep {sorted(keep)} out of order")
        if loss_ids is not None and stop != self.L:
            raise NudgeError("the loss needs the full stack (stop = L)")
        if delta is not None:
            x = x + delta.to(x.dtype)
        out = {"resid": {}}
        for l in range(start, stop):
            if l in keep:
                out["resid"][l] = x
            x = self.block(l, x)
        if stop in keep:
            out["resid"][stop] = x
        if loss_ids is not None:
            out["loss"] = self.loss(x, loss_ids, loss_at)
        return out

    def logits(self, x_final, at: Optional[Sequence[int]] = None):
        h = x_final if at is None else x_final[:, list(at)]
        return self.model.embed_out(self.body.final_layer_norm(h))

    def loss(self, x_final, ids: Sequence[int], at: Optional[Sequence[int]] = None):
        """Next-token cross-entropy at positions ``at`` (default every position with a next
        token), (B, P); logits only at those positions, one row at a time (memory)."""
        torch = self.torch
        n = x_final.shape[1]
        at = list(range(n - 1)) if at is None else list(at)
        if any(not 0 <= i < len(ids) - 1 or i >= n for i in at):
            raise NudgeError("a loss position has no next token in the pass")
        tgt = torch.as_tensor([ids[i + 1] for i in at], device=x_final.device)
        rows = []
        for b in range(x_final.shape[0]):
            lp = torch.log_softmax(self.logits(x_final[b:b + 1], at)[0], dim=-1)
            rows.append(-lp.gather(-1, tgt[:, None])[:, 0])
        return torch.stack(rows)

    def deltas(self, dirs: np.ndarray, eps: float, tokens: Sequence[int], n: int):
        """The batch's ``delta`` (1 + 2k, n, D): row 0 zero, then ``±ε·d`` at every token of T."""
        torch = self.torch
        idx, sign = signed_rows(dirs)
        d = torch.as_tensor(np.asarray(dirs), device=self.device, dtype=self.dtype)
        delta = torch.zeros((len(idx), n, d.shape[1]), device=self.device, dtype=self.dtype)
        for r in range(1, len(idx)):
            delta[r, list(tokens)] = (sign[r] * eps) * d[idx[r]]
        return delta


def numerics_record(stack: Stack) -> Dict:
    """What a run records about where it ran (NUMERICS)."""
    import torch
    import transformers
    rec = {"device": str(stack.device), "dtype": str(stack.dtype).replace("torch.", ""),
           "torch": torch.__version__, "transformers": transformers.__version__,
           "deterministic": torch.are_deterministic_algorithms_enabled(),
           "tf32": bool(torch.backends.cuda.matmul.allow_tf32)}
    if stack.device.type == "cuda":
        rec["gpu"] = torch.cuda.get_device_name(stack.device)
    return rec


def set_numerics() -> None:
    """Deterministic algorithms on, TF32 off. Call before the first CUDA op; cuBLAS needs
    ``CUBLAS_WORKSPACE_CONFIG`` set before CUDA initialises, so it is set here if absent."""
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
