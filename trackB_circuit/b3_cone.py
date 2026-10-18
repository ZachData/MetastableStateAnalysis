"""B3, routing, unit 2 (`trackB_circuit/design-B3.md` "L", frozen 2026-10-18, Blocked 65 (a); L
fixed by Blocked 67 (a) with the same-type arm, the user's, 2026-10-18): the light cone on B0's
harness (`core/nudge.py`), the gate G2 (L1), G4–G6, then L0 and L1.

**Inputs** (design "L"): 32 seeded sequences of 64 uniform token ids in [1000, 30000) repeated
twice (n = 128, no BOS), and their controls (the same first half, a fresh seeded second half).
Per sequence, 16 seeded nudge positions j in [8, 56) and K = 32 seeded random unit directions,
the same at every step. The site is ``resid_pre`` of block 4 at token j, ε = c·r with r the
sequence's RMS token norm there; c = 0.03 primary, 0.01 and 0.1 beside. One batch of 1 + 2K rows
per (sequence, j, c), base then ±ε·d_k.

**Read.** ``σ_t = median over the K directions of ‖A_t‖ / ε`` at token t of the final residual
(before the final LayerNorm), ``A = (f₊ − f₋)/2``. Per (sequence, j) the log ratio
``log(σ_i / median σ_m)``, m over i ± 2, ± 3, ± 4; ℓ is its median over the 512 (sequence, j),
its 99 % interval a percentile bootstrap over the 32 sequences (10,000 draws, seeded).

**Partner (Blocked 67 (a)).** i = j + 63: the typed induction members `L8H6`, `L11H14` peak at
offset +1 (query 64 + j′ → key j′ + 1; checked against unit 1's types at 143000 before anything is
read), so the query that reads key j sits at j + 63. j + 64 (`L7H8`'s offset 0) is beside.

**L0** (positive control, outside the family): ℓ at 143000; its interval must lie above 0 or the
instrument fails. **L1** (in the family of four at 0.99): Δℓ = ℓ − ℓ_abl at 143000 with the three
typed induction members `L7H8`, `L8H6`, `L11H14` mean-ablated (each head's output before
``attention.dense`` replaced by its mean over the 32 sequences' 128 positions on the base pass),
ranked against 200 seeded sets of 3 layer-matched non-members (layers 7, 8, 11; the six members
excluded) ablated the same way. **Same-type arm (beside, Blocked 67):** the same Δℓ with the
3 non-members nearest the three in type score (unit 1's types, mean over steps ≥ 512 at each
member's own offset; nearest first, taken in member order without reuse), ranked against the same
null. Its gap to L1 separates "goes through these heads" from "goes through induction heads".

**Gate** (design "Units 1 and 2's gates", before any member value):

| arm | here |
|---|---|
| G2 (L1) | each of the 200 null sets as "members" against the other 199: rank ≥ 0.99 in ≤ 2 % |
| G4 | per seed, one layer-8 non-member head h and a distance d ∈ [16, 60] (seeded); on the controls at 143000, in each nudge batch h's output at query j + d is set to its value at key j (attention mass 1 on j); ℓ with j + d as partner has its 99 % interval above 0 in ≥ 80 % of 50 seeds |
| G5 | ℓ's 99 % interval contains 0 at step 0 (repeated sequences, partner j + 63) and at 143000 on the controls (j + 63 as "partner") |
| G6 | per G4 seed, Δℓ with h mean-ablated (control means; the ablation overwrites the plant) ranked against L1's null: rank ≥ 0.99 in ≥ 80 % of 50 seeds |

**Choices the design left open, made before any output.** (1) G6's null is L1's own (the 200
random 3-sets on the repeated sequences): a planted null per seed (layer 8's 14 other heads on
that seed's plant) would cost 50 × 14 more ablation runs, about 20 GPU hours. So G6 asks whether
removing one known route clears the spread that L1 is ranked against, and not whether the planted
head stands out among its own layer. (2) Rounding floor as B1's: per (sequence, j) the larger of
the base row's distance to the same sequence run alone (another batch shape) and float32's eps
times the largest base norm; a σ whose median ‖A‖ is ≤ 10× it is counted, and a σ under it is
replaced by it. (3) Causal truncation: an L1 or G4/G6 batch keeps the first j + 68 (G4/G6: j + 65)
tokens, exact for every token read. The blocks before the first hooked layer are run once per
(sequence, j) and shared across configurations: from block 7 for L1, from block 8 for G4/G6.
(4) ℓ_none, the unablated ℓ that Δℓ subtracts from, is recomputed in L1's own truncated, cached
pass. L0 reads the full-length sweep; the two are recorded side by side.

**Beside:** ℓ against the control sequence's token i (same i − j, no partner); ℓ at j + 64; ℓ per
step and its onset (first step whose interval is above 0, held at the next) against R's formation
steps; the confinement share ``σ_i / Σ_{t ≥ j+2} σ_t``; ℓ at ``resid_pre`` of every block from 5
(which block the response first reaches the partner after); ℓ at c = 0.01, 0.1; below-floor counts.

NUMERICS: GPU float32, deterministic algorithms, TF32 off (`core.nudge.set_numerics`), one batch
per comparison; differences and norms in float64; statistics on the CPU in float64.

Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b3_cone means --out <dir>
    python -m trackB_circuit.b3_cone sweep --out <dir> [--steps 143000,0,...]
    python -m trackB_circuit.b3_cone gate  --out <dir>      # G4, G6 runs
    python -m trackB_circuit.b3_cone null  --out <dir>      # L1's ℓ_none and 200 null sets
    python -m trackB_circuit.b3_cone l1    --out <dir>      # members and same-type (after the gate)
    python -m trackB_circuit.b3_cone read  --out <dir> --types <unit 1 dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))

import numpy as np

from core import nudge as nd

STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)
LAST = 143000
N_SEQ, HALF, VOCAB = 32, 64, (1000, 30000)
N = 2 * HALF
N_J, J_LO, J_HI = 16, 8, 56                 # j in [8, 56)
SITE = 4                                    # placed (design): resid_pre of block 4
N_LAYERS, N_HEADS, HEAD_DIM, D_MODEL = 24, 16, 64, 1024
READ_LAYERS = tuple(range(SITE + 1, N_LAYERS + 1))   # resid_pre 5..23 and the final residual (24)
K = nd.K_RANDOM                             # 32
CS = (0.01, 0.03, 0.1)
C_STAT = 0.03
PARTNER, PARTNER0 = 63, 64                  # Blocked 67 (a): j + 63 primary, j + 64 beside
MATCH = (-4, -3, -2, 2, 3, 4)
CUT_L1 = PARTNER + 4 + 1                    # an L1 batch keeps tokens < j + 68
MEMBERS = ("L5H2", "L7H8", "L12H5", "L8H6", "L11H14", "L10H7")
L1_SET = ("L7H8", "L8H6", "L11H14")         # Blocked 67 (a): the 3 typed induction members
L1_FROM = 7                                 # first layer any L1 configuration hooks
N_NULL = 200
N_BOOT = 10000
CI = (0.005, 0.995)
RANK_PASS = 0.99
G2_MAX = 0.02                               # placed (design G2)
G4_SEEDS, G4_LAYER, G4_D = 50, 8, (16, 60)
CUT_G4 = G4_D[1] + 4 + 1                    # tokens < j + 65
G4_PASS = 0.8
U32 = float(np.finfo(np.float32).eps)
FORMATION = {"L7H8": 4000, "L8H6": 1000, "L11H14": 1000}   # R's, `status-B.md` "B3 unit 1b"


class B3ConeError(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256((REPO / "trackB_circuit/b3_cone.py").read_bytes())
    h.update((REPO / "core/nudge.py").read_bytes())
    return h.hexdigest()[:12]


def head_index(name: str) -> int:
    l, h = name[1:].split("H")
    return int(l) * N_HEADS + int(h)


def head_name(i: int) -> str:
    return f"L{i // N_HEADS}H{i % N_HEADS}"


# ---------------------------------------------------------------- inputs (pure)

def tokens(s: int, control: bool = False) -> List[int]:
    first = np.random.default_rng(nd.seed_for("b3_cone tokens", s)).integers(*VOCAB, HALF)
    second = (np.random.default_rng(nd.seed_for("b3_cone control", s)).integers(*VOCAB, HALF)
              if control else first)
    return [int(t) for t in np.concatenate([first, second])]


def positions(s: int) -> np.ndarray:
    rng = np.random.default_rng(nd.seed_for("b3_cone positions", s))
    return np.sort(rng.choice(np.arange(J_LO, J_HI), size=N_J, replace=False))


def directions(s: int) -> np.ndarray:
    return nd.random_directions(D_MODEL, K, nd.seed_for("b3_cone", s, "resid_pre", SITE))


def null_sets(n: int = N_NULL) -> np.ndarray:
    """(n, 3) layer-matched non-member sets for L1_SET, seeded."""
    mem = {head_index(m) for m in MEMBERS}
    pools = [np.array([l * N_HEADS + h for h in range(N_HEADS) if l * N_HEADS + h not in mem])
             for l in (head_index(m) // N_HEADS for m in L1_SET)]
    g = np.random.default_rng(nd.seed_for("b3_cone null sets"))
    return np.stack([p[g.integers(0, len(p), n)] for p in pools], 1)


def g4_draws(n: int = G4_SEEDS) -> List[Tuple[int, int]]:
    """Per seed, (head index in layer 8, d): a non-member head and the planted distance."""
    mem = {head_index(m) for m in MEMBERS}
    pool = [G4_LAYER * N_HEADS + h for h in range(N_HEADS) if G4_LAYER * N_HEADS + h not in mem]
    out = []
    for k in range(n):
        g = np.random.default_rng(nd.seed_for("b3_cone g4", k))
        out.append((int(pool[g.integers(0, len(pool))]), int(g.integers(G4_D[0], G4_D[1] + 1))))
    return out


def same_type_set(types_dir: Path) -> Dict:
    """The 3 non-members nearest L1_SET's members in type score: unit 1's types, the mean over
    steps ≥ 512 at each member's own offset (the larger of ind0, ind1 at 143000), nearest first,
    in member order without reuse. Also returns the offsets the partner rests on."""
    t = {s: json.loads((Path(types_dir) / f"step{s}.json").read_text()) for s in STEPS}
    mem = [head_index(m) for m in MEMBERS]
    taken, out, off = [], [], {}
    for m in L1_SET:
        h = head_index(m)
        key = "ind0" if t[LAST]["ind0"][h] >= t[LAST]["ind1"][h] else "ind1"
        off[m] = {"key": key, "ind0": t[LAST]["ind0"][h], "ind1": t[LAST]["ind1"][h]}
        sc = np.mean([t[s][key] for s in STEPS if s >= 512], 0)
        d = np.abs(sc - sc[h])
        d[mem + taken] = np.inf
        pick = int(np.argmin(d))
        taken.append(pick)
        out.append({"member": m, "head": head_name(pick), "score": float(sc[pick]),
                    "member_score": float(sc[h])})
    return {"set": out, "offsets": off}


# ---------------------------------------------------------------- the statistic (pure)

def log_ratio(sig: np.ndarray, J: np.ndarray, off: int) -> np.ndarray:
    """``sig`` (S, J, T) σ by token, ``J`` (S, J) nudge positions → (S, J) log(σ_i / median σ_m),
    i = j + off, m over i + MATCH."""
    S, nJ, _ = sig.shape
    i = J + off
    s_idx = np.arange(S)[:, None]
    j_idx = np.arange(nJ)[None, :]
    num = sig[s_idx, j_idx, i]
    den = np.median(np.stack([sig[s_idx, j_idx, i + m] for m in MATCH], -1), -1)
    return np.log(num / den)


def ell(vals: np.ndarray) -> float:
    return float(np.median(vals))


def boot_ci(vals: np.ndarray, seed: int, n: int = N_BOOT) -> Tuple[float, float]:
    """Percentile interval of the pooled median, resampling the first axis (sequences)."""
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(vals), (n, len(vals)))
    meds = np.median(vals[idx].reshape(n, -1), axis=1)
    lo, hi = np.quantile(meds, CI)
    return float(lo), float(hi)


def onset(steps: Sequence[int], lo: Sequence[float]) -> Optional[int]:
    """First step whose interval's lower end is above 0 and stays so at the next step."""
    for k in range(len(steps) - 1):
        if lo[k] > 0 and lo[k + 1] > 0:
            return int(steps[k])
    return None


def confinement(sig: np.ndarray, J: np.ndarray, off: int = PARTNER) -> np.ndarray:
    """(S, J) σ at the partner over the summed σ at every token from j + 2."""
    S, nJ, T = sig.shape
    t = np.arange(T)[None, None, :]
    mask = t >= (J + 2)[:, :, None]
    tot = np.where(mask, sig, 0.0).sum(-1)
    return sig[np.arange(S)[:, None], np.arange(nJ)[None, :], J + off] / tot


def rank_of(obs: float, null: np.ndarray) -> float:
    return float((np.asarray(null) < obs).mean())


def loo_false_pass(null: np.ndarray, thr: float = RANK_PASS) -> float:
    """Each null value as "members" against the others: the share with rank ≥ thr."""
    null = np.asarray(null)
    r = [rank_of(v, np.delete(null, k)) for k, v in enumerate(null)]
    return float(np.mean(np.array(r) >= thr))


def floored(medA: np.ndarray, floor: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """σ = medA / ε with values under the floor replaced by it; and the ≤ 10× floor mask."""
    f = np.asarray(floor)[..., None]
    return np.maximum(medA, f) / eps, medA <= nd.FLOOR_MULT * f


# ---------------------------------------------------------------- hooks (torch)

class Hooks:
    """Mean-ablation of head outputs and the G4 plant, on ``attention.dense``'s input.

    ``ablate``: {head index: mean (HEAD_DIM,) tensor}. ``plant``: (head index, i, j): the head's
    output at query i set to its value vector at key j (NeoX's per-head [q | k | v] rows)."""

    def __init__(self, st: "nd.Stack"):
        self.st, self.ablate, self.plant, self.v = st, {}, None, {}
        self.handles = []
        for l in range(len(st.layers)):
            att = st.layers[l].attention
            self.handles.append(att.dense.register_forward_pre_hook(self._pre(l)))
            self.handles.append(att.query_key_value.register_forward_hook(self._qkv(l)))

    def _qkv(self, l):
        def f(mod, args, out):
            if self.plant is not None and self.plant[0] // N_HEADS == l:
                self.v[l] = out
        return f

    def _pre(self, l):
        def f(mod, args):
            heads = [h for h in self.ablate if h // N_HEADS == l]
            pl = self.plant if self.plant is not None and self.plant[0] // N_HEADS == l else None
            if not heads and pl is None:
                return None
            z = args[0]
            B, T, _ = z.shape
            z = z.clone().view(B, T, N_HEADS, HEAD_DIM)
            if pl is not None:
                h, i, j = pl
                v = self.v[l].view(B, T, N_HEADS, 3 * HEAD_DIM)[..., 2 * HEAD_DIM:]
                z[:, i, h % N_HEADS] = v[:, j, h % N_HEADS]
            for h in heads:                                  # an ablation overwrites a plant
                z[:, :, h % N_HEADS] = self.ablate[h]
            return (z.view(B, T, N_HEADS * HEAD_DIM),) + tuple(args[1:])
        return f

    def set(self, ablate: Optional[Dict] = None, plant: Optional[Tuple[int, int, int]] = None):
        self.ablate, self.plant, self.v = dict(ablate or {}), plant, {}

    def remove(self):
        for h in self.handles:
            h.remove()


def head_means(st, control: bool) -> np.ndarray:
    """(384, HEAD_DIM) each head's output before ``attention.dense``, mean over the 32 sequences'
    positions on the base pass (float64)."""
    import torch
    acc = torch.zeros((N_LAYERS, N_HEADS, HEAD_DIM), dtype=torch.float64, device=st.device)
    hs = []
    for l in range(N_LAYERS):
        def f(mod, args, l=l):
            z = args[0]
            acc[l] += z.view(*z.shape[:2], N_HEADS, HEAD_DIM).double().sum((0, 1))
        hs.append(st.layers[l].attention.dense.register_forward_pre_hook(f))
    with torch.no_grad():
        for s in range(N_SEQ):
            st.run(st.embed(tokens(s, control)), 0)
    for h in hs:
        h.remove()
    return (acc / (N_SEQ * N)).reshape(N_LAYERS * N_HEADS, HEAD_DIM).cpu().numpy()


# ---------------------------------------------------------------- the GPU pass

def site_of(st, ids):
    import torch
    with torch.no_grad():
        x = st.run(st.embed(ids), 0, stop=SITE, keep={SITE})["resid"][SITE]
    return x, nd.rms_norm(x[0].double().cpu().numpy())


def med_norms(X) -> np.ndarray:
    """(1 + 2K, T, D) → (T,) median over K of ‖A‖."""
    b, p, m = nd.split_rows(X.double())
    A, _ = nd.anti_sym(p, m, b)
    return np.median(A.norm(dim=-1).cpu().numpy(), axis=0)


def floor_of(base_row, alone) -> float:
    b, a = base_row.double(), alone.double()
    return max(float((b - a).norm(dim=-1).max()), U32 * float(a.norm(dim=-1).max()))


def batch_input(st, x, R, eps: float, j: int, n: int):
    return x[:, :n].expand(2 * K + 1, -1, -1) + st.deltas(R, eps, [j], n)


def sweep_step(step: int, out: Path) -> Dict:
    """Repeated sequences at every c (by-layer at C_STAT), controls at C_STAT (final only)."""
    import torch
    from trackB_circuit.b0_gate import load
    t0 = time.monotonic()
    model, _ = load(step, "cuda", torch.float32)
    st = nd.Stack(model)
    sig = {c: np.full((N_SEQ, N_J, N), np.nan) for c in CS}
    lay = np.full((N_SEQ, N_J, len(READ_LAYERS), N), np.nan)
    ctrl = np.full((N_SEQ, N_J, N), np.nan)
    below = {c: np.zeros((N_SEQ, N_J, N), bool) for c in CS}
    floor, floor_c, r_rep, r_ctl = (np.zeros((N_SEQ, N_J)), np.zeros((N_SEQ, N_J)),
                                    np.zeros(N_SEQ), np.zeros(N_SEQ))
    with torch.no_grad():
        for s in range(N_SEQ):
            J, R = positions(s), directions(s)
            for control in (False, True):
                x, r = site_of(st, tokens(s, control))
                alone = st.run(x, SITE, keep={N_LAYERS})["resid"][N_LAYERS][0]
                (r_ctl if control else r_rep)[s] = r
                for a, j in enumerate(J):
                    j = int(j)
                    for c in ((C_STAT,) if control else CS):
                        eps = c * r
                        keep = set(READ_LAYERS) if (c == C_STAT and not control) else {N_LAYERS}
                        o = st.run(batch_input(st, x, R, eps, j, N), SITE, keep=keep)["resid"]
                        fl = floor_of(o[N_LAYERS][0], alone)
                        mA = med_norms(o[N_LAYERS])
                        if control:
                            floor_c[s, a] = fl
                            ctrl[s, a] = floored(mA, fl, eps)[0]
                            continue
                        floor[s, a] = fl
                        sig[c][s, a], below[c][s, a] = floored(mA, fl, eps)
                        if c == C_STAT:
                            for k, l in enumerate(READ_LAYERS):
                                lay[s, a, k] = med_norms(o[l]) / eps
    rec = {"step": step, "code": code_sha(), "numerics": nd.numerics_record(st),
           "seconds": time.monotonic() - t0, "r_rep": r_rep.tolist(), "r_ctl": r_ctl.tolist()}
    np.savez_compressed(out / "sweep" / f"step{step}.npz",
                        **{f"sig_{c}": sig[c] for c in CS}, **{f"below_{c}": below[c] for c in CS},
                        layers=lay, ctrl=ctrl, floor=floor, floor_ctrl=floor_c,
                        J=np.stack([positions(s) for s in range(N_SEQ)]))
    (out / "sweep" / f"step{step}.json").write_text(json.dumps(rec) + "\n")
    del model
    torch.cuda.empty_cache()
    return rec


def config_pass(st, hooks: "Hooks", configs: Sequence[Dict], control: bool, cut: int,
                cache_at: int, out_file: Path, s: int, means: np.ndarray) -> None:
    """One sequence: for each j, blocks SITE..cache_at−1 once, then each configuration from
    ``cache_at`` (hooks set per configuration; none touches a layer below ``cache_at``)."""
    import torch
    J, R = positions(s), directions(s)
    x, r = site_of(st, tokens(s, control))
    eps = C_STAT * r
    for cf in configs:
        if any(h // N_HEADS < cache_at for h in cf.get("ablate", ())) or \
                (cf.get("plant") and cf["plant"][0] // N_HEADS < cache_at):
            raise B3ConeError(f"configuration {cf['name']} hooks a layer below {cache_at}")
    mean_t = torch.as_tensor(means, device=st.device, dtype=st.dtype)
    T = int(J.max()) + cut
    sig = np.full((len(configs), N_J, T), np.nan)
    below = np.zeros((len(configs), N_J, T), bool)
    floor = np.zeros(N_J)
    with torch.no_grad():
        for a, j in enumerate(J):
            j = int(j)
            n = j + cut
            hooks.set()
            xc = st.run(batch_input(st, x, R, eps, j, n), SITE, stop=cache_at, keep={cache_at})["resid"][cache_at]
            for k, cf in enumerate(configs):
                pl = cf.get("plant")
                hooks.set({h: mean_t[h] for h in cf.get("ablate", ())},
                          None if pl is None else (pl[0], j + pl[1], j))
                o = st.run(xc, cache_at, keep={N_LAYERS})["resid"][N_LAYERS]
                if k == 0:                                   # the floor: config 0, its hooks on
                    alone = st.run(x[:, :n], SITE, keep={N_LAYERS})["resid"][N_LAYERS][0]
                    floor[a] = floor_of(o[0], alone)
                sig[k, a, :n], below[k, a, :n] = floored(med_norms(o), floor[a], eps)
            hooks.set()
    np.savez_compressed(out_file, sig=sig, below=below, floor=floor, J=J,
                        names=np.array([c["name"] for c in configs]))


def run_configs(out: Path, sub: str, configs: List[Dict], control: bool, cut: int, cache_at: int) -> None:
    import torch
    from trackB_circuit.b0_gate import load
    means = np.load(out / "means.npz")["control" if control else "repeated"]
    model, _ = load(LAST, "cuda", torch.float32)
    st = nd.Stack(model)
    hooks = Hooks(st)
    d = out / sub
    d.mkdir(parents=True, exist_ok=True)
    meta = {"code": code_sha(), "numerics": nd.numerics_record(st), "control": control, "cut": cut,
            "cache_at": cache_at, "configs": configs}
    (d / "meta.json").write_text(json.dumps(meta) + "\n")
    for s in range(N_SEQ):
        f = d / f"seq{s:02d}.npz"
        if f.exists():
            continue
        t = time.monotonic()
        tmp = d / f"seq{s:02d}.tmp.npz"
        config_pass(st, hooks, configs, control, cut, cache_at, tmp, s, means)
        tmp.rename(f)
        print(f"{sub} seq {s}: {time.monotonic() - t:.0f}s", flush=True)
    hooks.remove()


def null_configs() -> List[Dict]:
    return [{"name": "none"}] + [{"name": f"null{k:03d}", "ablate": [int(h) for h in row]}
                                 for k, row in enumerate(null_sets())]


def g4_configs() -> List[Dict]:
    out = []
    for k, (h, d) in enumerate(g4_draws()):
        out.append({"name": f"g4_{k:02d}", "plant": [h, d], "d": d, "head": head_name(h)})
        out.append({"name": f"g6_{k:02d}", "plant": [h, d], "ablate": [h], "d": d, "head": head_name(h)})
    return out


def l1_configs(types_dir: Path) -> List[Dict]:
    st = same_type_set(types_dir)
    return [{"name": "none"}, {"name": "members", "ablate": [head_index(m) for m in L1_SET]},
            {"name": "same_type", "ablate": [head_index(x["head"]) for x in st["set"]]}]


# ---------------------------------------------------------------- read

def load_configs(d: Path) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    files = sorted(d.glob("seq[0-9][0-9].npz"))
    if len(files) != N_SEQ:
        raise B3ConeError(f"{d}: {len(files)} of {N_SEQ} sequences")
    zs = [np.load(f) for f in files]
    T = max(z["sig"].shape[-1] for z in zs)
    pad = lambda a, v: np.concatenate([a, np.full(a.shape[:-1] + (T - a.shape[-1],), v)], -1)
    sig = np.stack([pad(z["sig"], np.nan) for z in zs], 1)            # (configs, S, J, T)
    below = np.stack([pad(z["below"], False) for z in zs], 1)
    return sig, below, [str(x) for x in zs[0]["names"]], np.stack([z["J"] for z in zs])


def stat_row(vals: np.ndarray, tag: str) -> Dict:
    lo, hi = boot_ci(vals, nd.seed_for("b3_cone boot", tag))
    return {"ell": ell(vals), "lo": lo, "hi": hi}


def read_sweep(out: Path, step: int) -> Dict:
    z = np.load(out / "sweep" / f"step{step}.npz")
    J = z["J"]
    sig = z[f"sig_{C_STAT}"]
    row = {"step": step, "L": stat_row(log_ratio(sig, J, PARTNER), f"L {step}"),
           "partner0": stat_row(log_ratio(sig, J, PARTNER0), f"p0 {step}"),
           "control_matched": stat_row(log_ratio(z["ctrl"], J, PARTNER), f"ctrl {step}"),
           "vs_control_token": stat_row(np.log(sig[np.arange(N_SEQ)[:, None], np.arange(N_J), J + PARTNER]
                                               / z["ctrl"][np.arange(N_SEQ)[:, None], np.arange(N_J), J + PARTNER]),
                                        f"vsctrl {step}"),
           "confinement": float(np.median(confinement(sig, J))),
           "c": {str(c): stat_row(log_ratio(z[f"sig_{c}"], J, PARTNER), f"c{c} {step}") for c in CS}}
    s_idx, j_idx = np.arange(N_SEQ)[:, None], np.arange(N_J)[None, :]
    tok = np.concatenate([J[..., None] + PARTNER + np.array([0, *MATCH])], -1)
    row["below_floor_partner_matched"] = int(z[f"below_{C_STAT}"][s_idx[..., None], j_idx[..., None], tok].sum())
    row["by_layer"] = {}
    for k, l in enumerate(READ_LAYERS):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = log_ratio(z["layers"][:, :, k], J, PARTNER)
        row["by_layer"][l] = stat_row(v, f"layer {l} {step}") if np.all(np.isfinite(v)) else None
    first = [l for l in READ_LAYERS if row["by_layer"][l] and row["by_layer"][l]["lo"] > 0]
    row["first_layer_above"] = first[0] if first else None
    return row


def above(r: Dict) -> bool:
    return r["lo"] > 0


def contains0(r: Dict) -> bool:
    return r["lo"] <= 0 <= r["hi"]


def read_null(out: Path) -> Dict:
    sig, below, names, J = load_configs(out / "null")
    vals = {nm: log_ratio(sig[k], J, PARTNER) for k, nm in enumerate(names)}
    e = {nm: ell(v) for nm, v in vals.items()}
    null = np.array([e["none"] - e[nm] for nm in names if nm.startswith("null")])
    return {"ell_none": e["none"], "null": null, "J": J}


def gate(out: Path, nul: Dict) -> Dict:
    res = {}
    fp = loo_false_pass(nul["null"])
    res["G2_L1"] = {"false_pass": fp, "pass": fp <= G2_MAX}
    sig, below, names, J = load_configs(out / "gate")
    meta = json.loads((out / "gate" / "meta.json").read_text())
    by = {c["name"]: c for c in meta["configs"]}
    g4, g6 = [], []
    for k in range(G4_SEEDS):
        a, b = names.index(f"g4_{k:02d}"), names.index(f"g6_{k:02d}")
        d = by[f"g4_{k:02d}"]["d"]
        v4, v6 = log_ratio(sig[a], J, d), log_ratio(sig[b], J, d)
        r4 = stat_row(v4, f"g4 {k}")
        g4.append({"seed": k, "head": by[f"g4_{k:02d}"]["head"], "d": d, **r4, "pass": above(r4)})
        dl = ell(v4) - ell(v6)
        rk = rank_of(dl, nul["null"])
        g6.append({"seed": k, "delta": dl, "rank": rk, "pass": rk >= RANK_PASS})
    n4, n6 = sum(x["pass"] for x in g4), sum(x["pass"] for x in g6)
    res["G4"] = {"seeds": g4, "n_pass": n4, "pass": n4 >= G4_PASS * G4_SEEDS}
    res["G6"] = {"seeds": g6, "n_pass": n6, "pass": n6 >= G4_PASS * G4_SEEDS}
    s0, sl = read_sweep(out, 0), read_sweep(out, LAST)
    res["G5"] = {"step0": s0["L"], "control_143000": sl["control_matched"],
                 "pass": contains0(s0["L"]) and contains0(sl["control_matched"])}
    res["pass"] = all(res[a]["pass"] for a in ("G2_L1", "G4", "G5", "G6"))
    return res


def cmd_read(a) -> int:
    out = Path(a.out)
    nul = read_null(out)
    g = gate(out, nul)
    st = same_type_set(Path(a.types))
    for m, o in st["offsets"].items():
        want = "ind0" if m == "L7H8" else "ind1"
        if o["key"] != want:
            raise B3ConeError(f"{m}'s offset at 143000 is {o['key']}, Blocked 67 (a) rests on {want}")
    res = {"code": code_sha(), "gate": {k: v for k, v in g.items()}, "same_type": st,
           "null_sets": [[head_name(h) for h in row] for row in null_sets()],
           "null_summary": {"median": float(np.median(nul["null"])), "q99": float(np.quantile(nul["null"], 0.99)),
                            "min": float(nul["null"].min()), "max": float(nul["null"].max())}}
    print("gate:", {k: g[k]["pass"] for k in ("G2_L1", "G4", "G5", "G6")}, f"G4 {g['G4']['n_pass']}/50,"
          f" G6 {g['G6']['n_pass']}/50, G2 {g['G2_L1']['false_pass']:.3f}; G5 step0 "
          f"[{g['G5']['step0']['lo']:.3f}, {g['G5']['step0']['hi']:.3f}], ctrl "
          f"[{g['G5']['control_143000']['lo']:.3f}, {g['G5']['control_143000']['hi']:.3f}]")
    if not g["pass"] or a.gate_only:
        (out / "gate.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        return 0 if g["pass"] else 1
    sl = read_sweep(out, LAST)
    res["L0"] = {**sl["L"], "pass": above(sl["L"])}
    sig, below, names, J = load_configs(out / "l1")
    e = {nm: ell(log_ratio(sig[k], J, PARTNER)) for k, nm in enumerate(names)}
    if abs(e["none"] - nul["ell_none"]) > 1e-9:
        raise B3ConeError(f"ℓ_none differs between the null pass ({nul['ell_none']}) and l1 ({e['none']})")
    dm, ds = e["none"] - e["members"], e["none"] - e["same_type"]
    res["L1"] = {"ell_none": e["none"], "ell_members": e["members"], "delta": dm,
                 "rank": rank_of(dm, nul["null"]), "reads": None}
    res["L1"]["reads"] = ("the cone goes through the members" if res["L1"]["rank"] >= RANK_PASS
                          else "no difference shown")
    res["same_type_arm"] = {"ell": e["same_type"], "delta": ds, "rank": rank_of(ds, nul["null"])}
    steps = [s for s in STEPS if (out / "sweep" / f"step{s}.npz").exists()]
    rows = [read_sweep(out, s) for s in steps]
    res["steps"] = rows
    res["onset"] = onset(steps, [r["L"]["lo"] for r in rows]) if len(steps) == len(STEPS) else None
    res["formation"] = FORMATION
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    report(res)
    return 0


def report(res: Dict) -> None:
    L0, L1, S = res["L0"], res["L1"], res["same_type_arm"]
    print(f"L0: ℓ {L0['ell']:.3f} [{L0['lo']:.3f}, {L0['hi']:.3f}] pass={L0['pass']}")
    print(f"L1: ℓ_none {L1['ell_none']:.3f}, ℓ_abl {L1['ell_members']:.3f}, Δℓ {L1['delta']:.3f}, "
          f"rank {L1['rank']:.3f}: {L1['reads']}  (null median {res['null_summary']['median']:.3f}, "
          f"99th {res['null_summary']['q99']:.3f})")
    print(f"same-type ({', '.join(x['head'] for x in res['same_type']['set'])}): Δℓ {S['delta']:.3f}, rank {S['rank']:.3f}")
    print("| step | ℓ [99 %] | ℓ at j+64 | ℓ vs control token | control, matched | confinement | first layer | ℓ c=0.01, 0.1 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in res["steps"]:
        f = lambda x: f"{x['ell']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"
        print(f"| {r['step']} | {f(r['L'])} | {f(r['partner0'])} | {f(r['vs_control_token'])} | "
              f"{f(r['control_matched'])} | {r['confinement']:.3f} | {r['first_layer_above']} | "
              f"{r['c']['0.01']['ell']:.3f}, {r['c']['0.1']['ell']:.3f} |")
    print(f"onset {res['onset']}; formation {res['formation']}")


# ---------------------------------------------------------------- CLI

def _gpu():
    nd.set_numerics()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device (B3's L is GPU float32)")


def cmd_means(a) -> int:
    _gpu()
    import torch
    from trackB_circuit.b0_gate import load
    model, _ = load(LAST, "cuda", torch.float32)
    st = nd.Stack(model)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out) / "means.npz", repeated=head_means(st, False), control=head_means(st, True))
    return 0


def cmd_sweep(a) -> int:
    _gpu()
    out = Path(a.out)
    (out / "sweep").mkdir(parents=True, exist_ok=True)
    steps = [int(x) for x in a.steps.split(",")] if a.steps else sorted(STEPS, key=lambda s: s != LAST)
    for s in steps:
        if (out / "sweep" / f"step{s}.npz").exists():
            continue
        rec = sweep_step(s, out)
        z = np.load(out / "sweep" / f"step{s}.npz")
        if not np.all(np.isfinite(z[f"sig_{C_STAT}"][:, :, -1])) or np.all(z[f"sig_{C_STAT}"][:, :, -1] == 0):
            raise B3ConeError(f"step {s}: σ at the last token is non-finite or all zero")
        print(f"sweep step {s}: {rec['seconds']:.0f}s", flush=True)
    return 0


def cmd_gate(a) -> int:
    _gpu()
    run_configs(Path(a.out), "gate", g4_configs(), control=True, cut=CUT_G4, cache_at=G4_LAYER)
    return 0


def cmd_null(a) -> int:
    _gpu()
    run_configs(Path(a.out), "null", null_configs(), control=False, cut=CUT_L1, cache_at=L1_FROM)
    return 0


def cmd_l1(a) -> int:
    out = Path(a.out)
    g = json.loads((out / "gate.json").read_text())
    if not g["gate"]["pass"] or g["code"] != code_sha():
        raise SystemExit("refusing: the gate has not passed on this code (run read --gate-only first)")
    _gpu()
    run_configs(out, "l1", l1_configs(Path(a.types)), control=False, cut=CUT_L1, cache_at=L1_FROM)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("means", "sweep", "gate", "null", "l1", "read"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name == "sweep":
            p.add_argument("--steps")
        if name in ("l1", "read"):
            p.add_argument("--types", required=True)
        if name == "read":
            p.add_argument("--gate-only", action="store_true")
    a = ap.parse_args(argv)
    return {"means": cmd_means, "sweep": cmd_sweep, "gate": cmd_gate, "null": cmd_null,
            "l1": cmd_l1, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
