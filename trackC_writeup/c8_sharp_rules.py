"""
trackC_writeup/c8_sharp_rules.py — C8 U2's rule R_ii, pure (`design-C8.md` "U2"; constants pushed
here before any λ is computed). Its sha256 is the one U2's draft entry names.

**The quantity.** At a saved state s of a re-run, λ_P(s) = λ_max(P⁻¹H), with H the Hessian of the
mean next-token loss on a fixed batch: the first `CTX` predictions of each of `BATCH_ROWS` of C4's
held-out rows (`c4_train.HELD_OUT`; no arm trains on them), float32, eager attention, no autocast.
`CTX` is 1024, not the 2048 the arms train on, because the double backward at 2048 does not fit the
3080's free memory (placed on the released step143000, not a C8 arm: 1024 peaks at 3.8 GiB, 1536 at
5.5, 2048 runs out). The preconditioner is Adam's, as PyTorch's AdamW applies it,
which is Cohen et al.'s Eq. (2) (2207.14484 v2; `lit-C8.md` Parked 2, read before this freeze):

    P = (1 − β₁^k) · diag(√(v / (1 − β₂^k)) + ε),   k the optimiser's step count at s.

The design wrote P = diag(√v̂ + ε). The factor (1 − β₁^k) is how the bias-corrected first moment
enters the step (AdamW's step size lr / (1 − β₁^k)), so it is part of the operator whose stability
the threshold bounds. It is ≥ 0.81 at every state the rule reads except the first ~16 updates of a
fresh optimiser. **Coordinates with v = 0 exactly are excluded** (D = P^{-1/2} set to 0 there):
they are embedding rows of tokens no batch since the optimiser's start has held, so Adam has never
moved them, but a held-out token among them would put ≈ H_ii / ((1 − β₁^k) ε) ~ 1e8 · H_ii into
λ_P. Counted on C4's order before any λ: 118 of the held-out batch's 6629 input tokens are unseen
after one batch, 1 after four, 0 after sixteen; so this bites at F2's state 4 and B_P2's first
offsets. The count masked is recorded at every state. Computed on P^{-1/2} H P^{-1/2} (same spectrum) by Lanczos from a fixed-seed Gaussian vector:
the design's "power iteration", replaced before any λ because the planted quadratic with a
dominant negative eigenvalue (gap ratio 0.93) missed by 4 % under a shifted power iteration at 40
steps. Lanczos uses the same H·v products and returns the top algebraic Ritz value, a lower bound
on λ_max, so an unconverged estimate errs toward "below".

**Precision** (the gate, `c8_sharp.g_hvp`, on Pythia's released step128/256/512, not C8 arms):
fp32 against fp64 along a high-curvature direction, ≤ 1e-3 relative. At step143000 fp32 is off by
~1e-2 (sharp late attention), so the instrument is not claimed at the end of training.

**Threshold** 38/η = 3.8e4 at η = 1e-3 (`c8_thresholds.py` (1)); a crossing is λ_P ≥ θ · 3.8e4.
θ = 0.5: Cohen et al. §5 find minibatch Adam "never rises more than a bit beyond" the full-batch
threshold and sits lower at smaller batches, "almost" reaching it at their largest batch; C4's
batch (2.1M tokens) is large, but the update is also clipped at 1.0 and decays weights (0.1,
decoupled), neither in their threshold, and λ_P here is on 32 held-out rows of 1024 predictions (a minibatch Hessian, whose top eigenvalue
errs high against the full loss's), not the full loss.
θ is placed from the paper, not from any λ of C8's.

**Windows** are (i)'s (`c8_rules.window`): (t⁻ − 16, t] at each gated τ, from each re-run's own
rule-T onset (t⁻, t] on its own records; a branch's window starts no earlier than 256 + 8.

**Per re-run** (F2, BP2 carry a transient; BK2 must not):
- F2, BP2: "at" if the first saved state with λ_P ≥ θ·thr lies in the window; "outside" if it
  lies before or after it; "never" if no saved state to 128 updates crosses.
- BK2: "crosses" if any saved state to 128 updates crosses, else "below".

**Label at a τ arm:**

| F2 | BP2 | BK2 | R_ii |
|---|---|---|---|
| at | at | below | **crosses at the onset** |
| any "outside", or BK2 "crosses" | | | **crosses, not at the onset** |
| never | never | below | **never crosses** |
| anything else (at with never) | | | **mixed** |

**Refused at a τ** (cause named): F2 or BP2 has no transient by rule T to 128, or BK2 has one
(the premise; `design-C8.md` "U2"); or an unconverged state (`CONV_TOL` not met within `N_MAX`
iterations) whose being a crossing would change the label (the label is computed with each such
state below and crossing; if they differ, refused). Joint across τ: one label everywhere, else
"mixed". λ_P at fresh-optimiser start states (no moments: F2 at 0, BP2 at 256) is undefined and
not read.

**Beside, never read:** raw λ_max(H) on F¹, F½, F⅒, B_P, B_O, B_K at the originals' U1 windows
± 2 saved states and on A0's grid to 128 (the design's arms); λ_P / thr at every state; the label at θ = 1; the
clip's share (grad norm > 1 in the window, from the logs).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from trackC_writeup import c8_rules as r8

THETA = 0.5
THETA_BESIDE = 1.0
BETA1, BETA2, EPS = 0.9, 0.95, 1e-8          # c4_train.BETAS, EPS
ETA = 1e-3                                    # every re-run is flat at the peak
THRESH = (2 + 2 * BETA1) / ((1 - BETA1) * ETA)  # 38/η = 3.8e4
BATCH_ROWS = 32                               # C4's held-out rows HELD_OUT[0] .. + 32 (placed on timing, no λ seen)
CTX = 1024                                    # predictions a row (memory; see the docstring)
N_MIN, N_MAX = 10, 40                         # power iterations
CONV_TOL = 1e-3                               # |Δμ|/|μ| between successive iterations, twice running
SEED = 8                                      # the start vector's
HORIZON = 128
RERUNS = ("F2", "BP2", "BK2")
WITH_TRANSIENT = ("F2", "BP2")


class C8SharpRuleRefused(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def threshold(eta: float = ETA, b1: float = BETA1) -> float:
    return (2 + 2 * b1) / ((1 - b1) * eta)


def window(onset: Tuple[int, int], branch: Optional[int]) -> Tuple[int, int]:
    return r8.window(onset[0], onset[1], branch)


def first_crossing(lam: Dict[int, float], bar: float, assume_cross: Sequence[int] = ()) -> Optional[int]:
    for s in sorted(lam):
        if lam[s] >= bar or s in assume_cross:
            return s
    return None


def per_rerun(arm: str, lam: Dict[int, float], bar: float, win: Optional[Tuple[int, int]],
              assume_cross: Sequence[int] = ()) -> str:
    s = first_crossing(lam, bar, assume_cross)
    if arm == "BK2":
        return "crosses" if s is not None else "below"
    if s is None:
        return "never"
    lo, hi = win
    return "at" if lo < s <= hi else "outside"


def label(f2: str, bp2: str, bk2: str) -> str:
    if f2 == "at" and bp2 == "at" and bk2 == "below":
        return "crosses at the onset"
    if "outside" in (f2, bp2) or bk2 == "crosses":
        return "crosses, not at the onset"
    if f2 == "never" and bp2 == "never" and bk2 == "below":
        return "never crosses"
    return "mixed"


def label_at_tau(lams: Dict[str, Dict[int, float]], unconv: Dict[str, Sequence[int]],
                 wins: Dict[str, Tuple[int, int]], premise: Dict[str, bool], theta: float = THETA) -> dict:
    """``lams[arm]``: λ_P per state read (start states excluded); ``unconv[arm]``: states whose
    estimate did not converge; ``wins``: F2's and BP2's windows; ``premise[arm]``: rule T's presence."""
    if not (premise["F2"] and premise["BP2"] and not premise["BK2"]):
        return {"label": "refused", "why": f"premise: transient present {premise} (want F2, BP2 yes; BK2 no)"}
    bar = theta * THRESH
    lo = {a: per_rerun(a, lams[a], bar, wins.get(a)) for a in RERUNS}
    hi = {a: per_rerun(a, lams[a], bar, wins.get(a),
                       [s for s in unconv.get(a, ()) if lams[a][s] < bar]) for a in RERUNS}
    l0, l1 = label(lo["F2"], lo["BP2"], lo["BK2"]), label(hi["F2"], hi["BP2"], hi["BK2"])
    if l0 != l1:
        return {"label": "refused", "why": f"unconverged states decide it ({l0} vs {l1})", "per_rerun": lo}
    return {"label": l0, "per_rerun": lo, "bar": bar}


def joint(labels: Dict[str, str]) -> str:
    v = set(labels.values())
    return v.pop() if len(v) == 1 else "mixed"
