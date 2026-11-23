r"""
trackC_writeup/c4peg_g.py — C4′ (g): do the two end states differ in function?
(`design-C4peg.md` "(g)", frozen on Blocked 171 D3 (a), D4 (a); "(g) as U2 runs it").

- **G1**: the held-out loss along the straight line between two checkpoints, θ(α) = (1−α)θ_X + αθ_Y
  over every parameter, α ∈ {0, 0.1, …, 1} (+ ± 0.05 around the grid's maximum, beside); the
  barrier B = max_α [L(α) − ((1−α)L(0) + αL(1))] on the grid. Pairs at 4000: (A0, F) the question,
  (A0, A0′) and (F, F′) the order-only scale, (released, A0) the null, (A0, `pythia-70m-seed1`) the
  effect; (A0′, F′) beside. At 1000, beside: (A0, F), (A0′, F′), (A0, A0′), (F, F′).
- **G2**: each arm's induction heads (score ≥ 0.5 at 4000) mean-ablated, against the other
  schedule's set of the same pair and 20 random sets outside the 11-head union; Δ of the
  induction loss (64 random 64-token sequences repeated, the second copy's predictable tokens) and
  of the held-out loss.

    python -m trackC_writeup.c4peg_g g1 --out <dir>      # GPU
    python -m trackC_writeup.c4peg_g g2 --out <dir>      # GPU
    python -m trackC_writeup.c4peg_g read --out <dir>    # labels, leans

Tier 1, description. Nothing here is registered (Blocked 171 D4 (a)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO
from trackC_writeup import c4_train as ct

# ---------------------------------------------------------------- constants (design-C4peg.md "(g)")

ALPHAS = tuple(round(0.1 * k, 1) for k in range(11))
REFINE = 0.05                                               # design: ± 0.05 around the grid max, beside
GATE_EFFECT_MIN, GATE_RATIO = 0.1, 4.0                      # design: B(effect) > 0.1 nats and > 4 B(null)
S_FLOOR = 0.01                                              # design: the scale's fixed floor (nats)
PAIRS_4000 = {"question": ("A0", "F"), "scale_A0": ("A0", "A0p"), "scale_F": ("F", "Fp"),
              "null": ("rel", "A0"), "effect": ("A0", "seed1"), "question_pair2": ("A0p", "Fp")}
PAIRS_1000 = {"question": ("A0", "F"), "question_pair2": ("A0p", "Fp"), "scale_A0": ("A0", "A0p"),
              "scale_F": ("F", "Fp")}
ARMS = ("A0", "F", "A0p", "Fp")
OTHER = {"A0": "F", "F": "A0", "A0p": "Fp", "Fp": "A0p"}
HEAD_BAR, LOW_BAR = 0.5, 0.2                                # design: own set ≥ 0.5; random sets from < 0.2
N_RANDOM = 20
N_LAYERS, N_HEADS, D_HEAD = 6, 8, 64
IND_LEN, IND_N = 64, 64                                     # design: C4's IND_LEN, IND_N raised to 64
IND_SEED, SET_SEED = 20261125, 20261126                     # placed (design: "a new seed", "seeded")
HEADS_REC = {"A0": DATA / "trackC/c4/2026-11-14-close/heads_A0.json",
             "F": DATA / "trackC/c4/2026-11-14-close/heads_F.json",
             "A0p": DATA / "trackC/c4pc/2026-11-22-u3/heads_A0p.json",
             "Fp": DATA / "trackC/c4pc/2026-11-22-u3/heads_Fp.json"}
EXPECT_SETS = {"A0": ("L3H0", "L3H1", "L3H5", "L3H6"), "A0p": ("L3H0", "L3H1", "L3H5", "L3H6"),
               "F": ("L2H6", "L3H3", "L3H4", "L3H7"), "Fp": ("L2H2", "L2H5", "L2H6", "L3H7", "L4H1")}
K_F = DATA / "trackC/c4p/2026-11-17-u2/K_F.npz"            # C4′ (b), beside, not recomputed
CKPT_ROOT = {"A0": ct.ROOT, "F": ct.ROOT, "A0p": ct.HDD / "c4pc", "Fp": ct.HDD / "c4pc"}
REPOS = {"rel": ct.BASE, "seed1": "EleutherAI/pythia-70m-seed1"}
EXACT_TOL = 1e-5


class C4pegGRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4peg_g.py", "trackC_writeup/c4_train.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure

def barrier(alphas: Sequence[float], losses: Sequence[float]) -> Tuple[float, float]:
    """B = max over α of L(α) − ((1−α)L(0) + αL(1)), and the α where it sits (endpoints give 0)."""
    a = np.asarray(alphas, np.float64)
    L = np.asarray(losses, np.float64)
    if a[0] != 0.0 or a[-1] != 1.0:
        raise C4pegGRefused("the grid must run from 0 to 1")
    ex = L - ((1 - a) * L[0] + a * L[-1])
    k = int(np.argmax(ex))
    return float(ex[k]), float(a[k])


def refine_alphas(a_max: float) -> List[float]:
    return [round(x, 2) for x in (a_max - REFINE, a_max + REFINE) if 0.0 < x < 1.0]


def gate_g1(b_effect: float, b_null: float) -> bool:
    return bool(b_effect > GATE_EFFECT_MIN and b_effect > GATE_RATIO * b_null)


def scale_g1(b_a0: float, b_f: float, b_null: float) -> float:
    return float(max(b_a0, b_f, b_null, S_FLOOR))


def label_g1(b_q: float, S: float, readable: bool) -> str:
    if not readable:
        return "unreadable"
    if b_q <= S:
        return "within the order-only spread"
    if b_q > 2 * S:
        return "beyond it"
    return "between"


def lean_g1(b_f: float, b_a0: float, b_null: float) -> str:
    """g1: B(F, F′) − B(A0, A0′) > max(B(null), 0.01) held; < −that broken; else not told apart."""
    bar = max(b_null, S_FLOOR)
    d = b_f - b_a0
    return "held" if d > bar else ("broken" if d < -bar else "not told apart")


def lean_g2(label: str) -> Optional[bool]:
    if label == "unreadable":
        return None
    return label in ("beyond it", "between")


def head_name(l: int, h: int) -> str:
    return f"L{l}H{h}"


def parse_head(n: str) -> Tuple[int, int]:
    l, h = n[1:].split("H")
    return int(l), int(h)


def own_set(scores: np.ndarray, bar: float = HEAD_BAR) -> Tuple[str, ...]:
    return tuple(head_name(l, h) for l in range(scores.shape[0]) for h in range(scores.shape[1]) if scores[l, h] >= bar)


def random_sets(scores: np.ndarray, union: Sequence[str], k: int, n: int, seed: int) -> List[Tuple[str, ...]]:
    """n sets of k heads, each drawn without replacement from heads below LOW_BAR on this arm and
    outside the union; not layer-matched (design, finding 1 on #366)."""
    pool = [head_name(l, h) for l in range(scores.shape[0]) for h in range(scores.shape[1])
            if scores[l, h] < LOW_BAR and head_name(l, h) not in set(union)]
    if len(pool) < k:
        raise C4pegGRefused(f"pool {len(pool)} < set size {k}")
    g = np.random.default_rng(seed)
    return [tuple(sorted(g.choice(pool, size=k, replace=False).tolist(), key=parse_head)) for _ in range(n)]


def label_g2(d_own: float, d_other: float, d_random: Sequence[float]) -> dict:
    top = float(max(d_random))
    return {"own_carry": bool(d_own > top), "index_carries": bool(d_other > top), "random_max": top}


def induction_batch(n: int = IND_N, L: int = IND_LEN, seed: int = IND_SEED) -> np.ndarray:
    half = np.random.default_rng(seed).integers(1000, 50000, (n, L))
    return np.concatenate([half, half], 1)


# ---------------------------------------------------------------- GPU

def _torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device")
    return torch, torch.device("cuda")


def load(kind: str, step: int, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    if kind in ARMS:
        src, kw = str(ct.ckpt_dir(CKPT_ROOT[kind], kind, step)), {}
    else:
        src, kw = REPOS[kind], {"revision": f"step{step}"}
    m = from_pretrained_eager(GPTNeoXForCausalLM, src, torch_dtype=torch.float32, **kw).to(device).eval()
    if any(p.dtype != torch.float32 for p in m.parameters()):
        raise C4pegGRefused(f"{kind}@{step}: not float32")
    return m


def held_rows() -> np.ndarray:
    rows = ct.open_rows(ct.ROOT)
    x = np.asarray(rows[ct.HELD_OUT[0]:ct.HELD_OUT[1]]).astype(np.int64)
    if x.shape != (64, ct.ROW):
        raise C4pegGRefused(f"held-out rows {x.shape}")
    return x


def held_loss(model, x: np.ndarray, device) -> float:
    import torch
    import torch.nn.functional as F
    tot, n = 0.0, 0
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        for k in range(0, len(xt), 8):
            r = xt[k:k + 8]
            lg = model(input_ids=r[:, :-1]).logits.float()
            tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), r[:, 1:].reshape(-1), reduction="sum"))
            n += r[:, 1:].numel()
    return tot / n


def induction_loss(model, seq: np.ndarray, device) -> float:
    """Mean next-token loss on the second copy's predictable tokens (targets L+1 … 2L−1)."""
    import torch
    import torch.nn.functional as F
    L = seq.shape[1] // 2
    with torch.no_grad():
        x = torch.from_numpy(seq).to(device)
        lg = model(input_ids=x).logits.float()
        pred = lg[:, L:2 * L - 1]
        tgt = x[:, L + 1:2 * L]
        return float(F.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1)))


def interpolate_into(model, sd_x: dict, sd_y: dict, alpha: float) -> None:
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_((1 - alpha) * sd_x[n] + alpha * sd_y[n])


def g1(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = _torch()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x = held_rows()
    res = {"code": code_sha(), "rows": list(ct.HELD_OUT), "alphas": list(ALPHAS), "pairs": {}}
    first = True
    for step, pairs in ((4000, PAIRS_4000), (1000, PAIRS_1000)):
        for role, (kx, ky) in pairs.items():
            mx, my = load(kx, step, device), load(ky, step, device)
            sx = {n: p.detach().clone() for n, p in mx.named_parameters()}
            sy = {n: p.detach().clone() for n, p in my.named_parameters()}
            if set(sx) != set(sy):
                raise C4pegGRefused(f"{kx}, {ky}: parameter names differ")
            l_end = (held_loss(mx, x, device), held_loss(my, x, device))
            del my
            L = []
            for al in ALPHAS:
                interpolate_into(mx, sx, sy, al)
                L.append(held_loss(mx, x, device))
            if abs(L[0] - l_end[0]) > EXACT_TOL * l_end[0] or abs(L[-1] - l_end[1]) > EXACT_TOL * l_end[1]:
                raise C4pegGRefused(f"{role}@{step}: endpoints {L[0]}, {L[-1]} ≠ {l_end}")
            b, a_max = barrier(ALPHAS, L)
            ref = {}
            for al in refine_alphas(a_max):
                interpolate_into(mx, sx, sy, al)
                ref[str(al)] = held_loss(mx, x, device) - ((1 - al) * L[0] + al * L[-1])
            res["pairs"][f"{role}@{step}"] = {"x": kx, "y": ky, "step": step, "L": L, "B": b, "alpha_max": a_max,
                                             "refined_excess": ref}
            if first:
                print(f"first output populated: {role}@{step}: L {[round(v, 4) for v in L]} B {b:.4f}", flush=True)
                first = False
            del mx, sx, sy
            torch.cuda.empty_cache()
            ct._jdump(out / "g1.json", res)
    print(json.dumps({k: round(v["B"], 5) for k, v in res["pairs"].items()}), flush=True)
    return 0


def head_scores(arm: str) -> np.ndarray:
    return np.asarray(json.loads(HEADS_REC[arm].read_text())["per_step"]["4000"], np.float64)


def sets_for_all() -> dict:
    own = {arm: own_set(head_scores(arm)) for arm in ARMS}
    for arm in ARMS:
        if own[arm] != EXPECT_SETS[arm]:
            raise C4pegGRefused(f"{arm}: own set {own[arm]} ≠ the design's {EXPECT_SETS[arm]}")
    union = sorted(set().union(*own.values()), key=parse_head)
    if len(union) != 11:
        raise C4pegGRefused(f"union of {len(union)} heads, the design says 11")
    out = {"union": union}
    for i, arm in enumerate(ARMS):
        out[arm] = {"own": list(own[arm]), "other": list(own[OTHER[arm]]),
                    "random": [list(s) for s in random_sets(head_scores(arm), union, len(own[arm]), N_RANDOM,
                                                          SET_SEED + i)]}
    return out


def dense_means(model, x: np.ndarray, device) -> List[np.ndarray]:
    """Per layer, the mean over the held-out positions of the attention's merged heads (the dense
    projection's input), (512,)."""
    import torch
    acc = [torch.zeros(N_HEADS * D_HEAD, dtype=torch.float64, device=device) for _ in range(N_LAYERS)]
    cnt = [0]
    hooks = []
    for l, layer in enumerate(model.gpt_neox.layers):
        def pre(mod, inp, l=l):
            acc[l] += inp[0].double().reshape(-1, inp[0].shape[-1]).sum(0)
            if l == 0:
                cnt[0] += inp[0].shape[0] * inp[0].shape[1]
        hooks.append(layer.attention.dense.register_forward_pre_hook(pre))
    try:
        xt = torch.from_numpy(x).to(device)
        with torch.no_grad():
            for k in range(0, len(xt), 8):
                model(input_ids=xt[k:k + 8, :-1])
    finally:
        for h in hooks:
            h.remove()
    return [(v / cnt[0]).float() for v in acc]


def ablate(model, heads: Sequence[str], means):
    """Register pre-hooks that replace each head's slice of the dense input by its mean; returns hooks."""
    by_layer: Dict[int, List[int]] = {}
    for n in heads:
        l, h = parse_head(n)
        by_layer.setdefault(l, []).append(h)
    hooks = []
    for l, hs in by_layer.items():
        def pre(mod, inp, l=l, hs=tuple(hs)):
            z = inp[0].clone()
            for h in hs:
                z[..., h * D_HEAD:(h + 1) * D_HEAD] = means[l][h * D_HEAD:(h + 1) * D_HEAD]
            return (z,) + tuple(inp[1:])
        hooks.append(model.gpt_neox.layers[l].attention.dense.register_forward_pre_hook(pre))
    return hooks


def g2(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = _torch()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x = held_rows()
    seq = induction_batch()
    sets = sets_for_all()
    res = {"code": code_sha(), "sets": sets, "ind": {"len": IND_LEN, "n": IND_N, "seed": IND_SEED}, "arms": {}}
    for arm in ARMS:
        m = load(arm, 4000, device)
        means = dense_means(m, x, device)
        base = {"ind": induction_loss(m, seq, device), "held": held_loss(m, x, device)}
        # known answer: ablating nothing through the hook path changes nothing
        hk = ablate(m, [], means)
        chk = induction_loss(m, seq, device)
        for h in hk:
            h.remove()
        if abs(chk - base["ind"]) > EXACT_TOL * base["ind"]:
            raise C4pegGRefused(f"{arm}: empty ablation moves the loss")
        rows = {}
        named = [("own", sets[arm]["own"]), ("other", sets[arm]["other"])] + \
                [(f"random{i}", s) for i, s in enumerate(sets[arm]["random"])]
        for name, hs in named:
            hk = ablate(m, hs, means)
            try:
                rows[name] = {"heads": list(hs), "d_ind": induction_loss(m, seq, device) - base["ind"],
                              "d_held": held_loss(m, x, device) - base["held"]}
            finally:
                for h in hk:
                    h.remove()
        res["arms"][arm] = {"base": base, "sets": rows}
        print(f"{arm}: base {base}; own Δind {rows['own']['d_ind']:.4f}, other {rows['other']['d_ind']:.4f}, "
              f"random max {max(rows[f'random{i}']['d_ind'] for i in range(N_RANDOM)):.4f}", flush=True)
        del m
        torch.cuda.empty_cache()
        ct._jdump(out / "g2.json", res)
    return 0


# ---------------------------------------------------------------- the reading (pure over records)

def read_g(g1r: dict, g2r: dict, k_beside: Optional[dict] = None) -> dict:
    B = {k: v["B"] for k, v in g1r["pairs"].items()}
    b_null, b_eff = B["null@4000"], B["effect@4000"]
    readable = gate_g1(b_eff, b_null)
    S = scale_g1(B["scale_A0@4000"], B["scale_F@4000"], b_null)
    lab = label_g1(B["question@4000"], S, readable)
    lab2 = label_g1(B["question_pair2@4000"], S, readable)
    res = {"g1": {"B": B, "readable": readable, "S": S, "label": lab, "label_pair2_beside": lab2},
           "g2": {}, "leans": {}}
    res["leans"]["g1"] = lean_g1(B["scale_F@4000"], B["scale_A0@4000"], b_null)
    res["leans"]["g2"] = {"held": lean_g2(lab), "label": lab, "pair2_beside": lab2}
    g3 = {}
    for arm, d in g2r["arms"].items():
        rs = d["sets"]
        rnd = [rs[f"random{i}"]["d_ind"] for i in range(N_RANDOM)]
        lb = label_g2(rs["own"]["d_ind"], rs["other"]["d_ind"], rnd)
        lb.update(d_own=rs["own"]["d_ind"], d_other=rs["other"]["d_ind"],
                  d_held_own=rs["own"]["d_held"], d_held_other=rs["other"]["d_held"],
                  random_layers=[sorted({parse_head(h)[0] for h in rs[f"random{i}"]["heads"]}) for i in range(N_RANDOM)])
        res["g2"][arm] = lb
        g3[arm] = lb["own_carry"] and not lb["index_carries"]
    res["g2"]["ratio_F_A0"] = res["g2"]["F"]["d_own"] / res["g2"]["A0"]["d_own"]
    res["g2"]["ratio_Fp_A0p"] = res["g2"]["Fp"]["d_own"] / res["g2"]["A0p"]["d_own"]
    res["leans"]["g3"] = {"per_arm": g3, "held": all(g3.values())}
    if k_beside:
        res["g1"]["K_beside"] = k_beside
    return res


def read(a) -> int:
    out = Path(a.out)
    g1r = json.loads((out / "g1.json").read_text())
    g2r = json.loads((out / "g2.json").read_text())
    codes = {g1r["code"], g2r["code"]}
    if len(codes) != 1:
        raise C4pegGRefused(f"records from several code versions: {codes}")
    z = np.load(K_F, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    K = z["K"]
    kb = {"K_F4000_A0_4000": float(K[keys.index("F@4000"), keys.index("A0@4000")]),
          "K_A0_3500_A0_4000": float(K[keys.index("A0@3500"), keys.index("A0@4000")])}
    res = read_g(g1r, g2r, kb)
    res.update(code=code_sha(), record_code=codes.pop(),
               inputs={f: hashlib.sha256((out / f).read_bytes()).hexdigest()[:16] for f in ("g1.json", "g2.json")})
    ct._jdump(out / "reading_g.json", res)
    print(json.dumps({"g1": res["g1"], "leans": res["leans"]}, indent=1), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(required=True)
    for name, fn in (("g1", g1), ("g2", g2), ("read", read)):
        q = sub.add_parser(name)
        q.add_argument("--out", required=True)
        q.set_defaults(fn=fn)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
