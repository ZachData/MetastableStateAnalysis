r"""
trackC_writeup/c4peg_read.py — C4′ (e): what the tokens' common direction is and what writes it
(`design-C4peg.md`, frozen 2026-11-23 on Blocked 170 all (a); "(e) as U1 runs it").

On the four trained arms (A0, F from C4; A0′, F′ from C4′ (c)), every saved step, v1's 7 prompts:

- **E1**: m̄_l (the mean over prompts of each prompt's mean of hᵢ/‖hᵢ‖ at state l) read through
  the unembedding, z = W_U(γ_f ⊙ x) at states 0–5 and W_U x at state 6, against the log unigram
  frequency of the arms' training labels: Spearman r, gated per checkpoint and readout by the
  least-squares unigram direction u\* (effect) and 1000 Gaussian directions (null, q = 99.9th pct)
  and 100 permutations of log f (u\* refitted). nKL (β fitted) and R² beside.
- **E2**: every (prompt, transition) fall of ‖m‖² split exactly, Δ = R + A + M (+ B at T5), and
  labelled mode × writer; the shares of m̄_l by component.
- **E3**: cos between arms' m̄_l at the same step; labels against the replicate pairs.

    python -m trackC_writeup.c4peg_read counts --out <dir>             # CPU, one pass over the rows
    python -m trackC_writeup.c4peg_read arm --out <dir> --arm A0       # GPU, all 29 steps
    python -m trackC_writeup.c4peg_read read --out <dir>               # labels, windows, leans

Tier 1, description. Nothing here is registered (Blocked 170 D4 (a)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, V1
from trackC_writeup import c4_train as ct

# ---------------------------------------------------------------- constants (design-C4peg.md)

ARMS = ("A0", "F", "A0p", "Fp")
PAIR_OF = {"A0": "F", "F": "A0", "A0p": "Fp", "Fp": "A0p"}
CKPT_ROOT = {"A0": ct.ROOT, "F": ct.ROOT, "A0p": ct.HDD / "c4pc", "Fp": ct.HDD / "c4pc"}
REC_DIR = {"A0": DATA / "trackC/c4/2026-11-12-u2", "F": DATA / "trackC/c4/2026-11-12-u2",
           "A0p": DATA / "trackC/c4pc/2026-11-22-u3", "Fp": DATA / "trackC/c4pc/2026-11-22-u3"}
STEPS = tuple(ct.SAVE_STEPS)                                # 29 steps, 0 … 4000 with 3500
N_STATES, N_LAYERS, D = 7, 6, 512
VOCAB = 50304                                              # embed_out rows (padded)
N_NULL, NULL_Q = 1000, 99.9                                # design: the null, its percentile
N_PERM, PERM_Q = 100, 99.0                                 # design: the permutation gate
NULL_SEED, PERM_SEED = 20261123, 20261124                  # placed (design: seeded)
TOP_FRAC = 0.7                                             # 2609.02959's support, beside
BETA_BOUND = 200.0                                         # placed: the nKL scale search
EXACT_TOL = 1e-5                                           # design: known answers, relative
LN_EPS = 1e-5                                              # GPT-NeoX's layer_norm_eps
E3_MARGIN = 0.1                                            # design: "schedule-specific" margin
WINDOWS = {                                                # design "Cells", per arm
    "A0": {"collapse": (24, 64), "V": (384, 384), "centred": (640, 640), "end": (4000, 4000)},
    "F": {"collapse": (1, 16), "V": (24, 24), "centred": (48, 48), "end": (4000, 4000)},
    "A0p": {"collapse": (24, 64), "V": (320, 320), "centred": (640, 640), "end": (4000, 4000)},
    "Fp": {"collapse": (1, 16), "V": (32, 32), "centred": (48, 48), "end": (4000, 4000)},
}
V_TRANSITION = {"A0": 2, "A0p": 2, "F": 1, "Fp": 1}         # C4 item 8, P-SCH3
EVENTS = ((48, 4), (96, 24), (384, 24), (640, 48), (384, 256), (768, 320), (1000, 640))  # (A0, F)
MID_STATES = (2, 3, 4)


class C4pegRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4peg_read.py", "trackC_writeup/c4_train.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def window_steps(arm: str, name: str, steps: Sequence[int] = STEPS) -> List[int]:
    lo, hi = WINDOWS[arm][name]
    return [s for s in steps if lo <= s <= hi]


# ---------------------------------------------------------------- E1 (pure)

def rank_avg(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), 0-based."""
    from scipy.stats import rankdata
    return rankdata(x, method="average") - 1.0


def _std_ranks(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, np.float64) - np.mean(r)
    return r / np.linalg.norm(r)


def spearman(z: np.ndarray, ell: np.ndarray) -> float:
    """Spearman's r with average ranks on both sides."""
    return float(_std_ranks(rank_avg(z)) @ _std_ranks(rank_avg(ell)))


def ls_direction(M: np.ndarray, ell: np.ndarray) -> Tuple[np.ndarray, float]:
    """2609.02959's fit: OLS of centred ell on centred rows of M (float64). Returns (u, R²)."""
    Mt = np.asarray(M, np.float64)
    Mt = Mt - Mt.mean(0)
    lt = np.asarray(ell, np.float64) - np.mean(ell)
    u, *_ = np.linalg.lstsq(Mt, lt, rcond=None)
    r2 = 1.0 - float(np.sum((lt - Mt @ u) ** 2) / np.sum(lt ** 2))
    return u, r2


def label_e1(r: float, q: float, readable: bool) -> str:
    if not readable:
        return "unreadable"
    if r > q:
        return "aligned"
    if r < -q:
        return "anti-aligned"
    return "not aligned"


def gate_e1(r_star: float, q: float, perm_q: float) -> bool:
    return bool(r_star > q and r_star > perm_q)


def nkl(z: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """min over β of KL(p ‖ softmax(β z)) / KL(p ‖ uniform), on the support; (nKL, β)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    z = np.asarray(z, np.float64)
    p = np.asarray(p, np.float64)
    neg_h = float(np.sum(p * np.log(p)))
    ref = neg_h + np.log(len(p))

    def kl(b):
        return neg_h - float(p @ (b * z) - logsumexp(b * z))

    res = minimize_scalar(kl, bounds=(-BETA_BOUND, BETA_BOUND), method="bounded")
    return float(kl(res.x) / ref), float(res.x)


# ---------------------------------------------------------------- E2 (pure, one prompt)

def unit_mean(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, np.float64)
    return (h / np.linalg.norm(h, axis=1, keepdims=True)).mean(0)


def ln_linear(c: np.ndarray, gamma: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """The final LN's linear part on a component: γ ⊙ (c − mean_d c) / σᵢ."""
    c = np.asarray(c, np.float64)
    return gamma * (c - c.mean(1, keepdims=True)) / sigma[:, None]


def split_transition(h: np.ndarray, a: np.ndarray, f: np.ndarray, *, ln: Optional[dict] = None,
                     s_next: Optional[np.ndarray] = None) -> dict:
    """Exact split of Δ‖m‖² over one transition (design E2).

    Without ``ln``: h' = h + a + f, uᵢ = hᵢ/‖hᵢ‖, weights wᵢ = 1/‖h'ᵢ‖; m' = m_old + m_a + m_f.
    With ``ln`` (T5; keys gamma, beta, eps): the state after is s' = LN(h + a + f), given as
    ``s_next``; the components pass through the LN's linear part and β is a fourth writer.
    R = ‖m_old‖² − ‖m‖²; writer k: 2⟨m_old, x_k⟩ + ⟨x_k, Σ writers⟩. g per writer: the same split
    of the (pre-LN) norm growth ‖h'ᵢ‖² − ‖hᵢ‖², token-averaged.
    """
    h, a, f = (np.asarray(x, np.float64) for x in (h, a, f))
    raw = h + a + f
    m0 = unit_mean(h)
    if ln is None:
        w = 1.0 / np.linalg.norm(raw, axis=1)
        old, xa, xf = ((w[:, None] * x).mean(0) for x in (h, a, f))
        writers = {"A": xa, "M": xf}
        m1_direct = unit_mean(raw)
    else:
        mu = raw.mean(1, keepdims=True)
        sigma = np.sqrt(((raw - mu) ** 2).mean(1) + ln["eps"])
        g = np.asarray(ln["gamma"], np.float64)
        b = np.asarray(ln["beta"], np.float64)
        s = np.asarray(s_next, np.float64)
        w = 1.0 / np.linalg.norm(s, axis=1)
        old, xa, xf = ((w[:, None] * ln_linear(x, g, sigma)).mean(0) for x in (h, a, f))
        xb = (w[:, None] * np.broadcast_to(b, s.shape)).mean(0)
        writers = {"A": xa, "M": xf, "B": xb}
        m1_direct = unit_mean(s)
    S = sum(writers.values())
    m1 = old + S
    scale = max(np.linalg.norm(m1_direct), 1e-12)
    resid = float(np.linalg.norm(m1 - m1_direct) / scale)
    terms = {"R": float(old @ old - m0 @ m0)}
    for k, x in writers.items():
        terms[k] = float(2 * old @ x + x @ S)
    delta = float(m1 @ m1 - m0 @ m0)
    hh = h
    gs = raw - h                                            # = a + f
    ga = float(np.mean(2 * np.sum(hh * a, 1) + np.sum(a * gs, 1)))
    gf = float(np.mean(2 * np.sum(hh * f, 1) + np.sum(f * gs, 1)))
    return {"delta": delta, **terms, "ga": ga, "gf": gf, "resid": resid,
            "sum_resid": float(abs(sum(terms.values()) - delta) / max(abs(delta), m0 @ m0, 1e-12))}


def fall_label(sp: dict) -> Optional[str]:
    """None if no fall; else mode × writer by the most negative term (design E2)."""
    if sp["delta"] >= 0:
        return None
    keys = [k for k in ("R", "A", "M", "B") if k in sp]
    k = min(keys, key=lambda k: sp[k])
    if k == "R":
        return "dilution, attention" if sp["ga"] >= sp["gf"] else "dilution, MLP"
    return {"A": "removal, attention", "M": "removal, MLP", "B": "LN bias"}[k]


def writer_of(label: str) -> str:
    return "attention" if label.endswith("attention") else ("MLP" if label.endswith("MLP") else "LN bias")


def component_means(embed: np.ndarray, attn: Sequence[np.ndarray], mlp: Sequence[np.ndarray],
                    hs: Sequence[np.ndarray], ln: dict) -> List[Dict[str, np.ndarray]]:
    """Per state l, {component: mean wᵢ cᵢ} with wᵢ = 1/‖state_l,i‖ (state 6 through the LN)."""
    out = []
    comps = {"embed": np.asarray(embed, np.float64)}
    for l in range(N_STATES):
        if l == N_STATES - 1:
            comps[f"a{l-1}"] = np.asarray(attn[l - 1], np.float64)
            comps[f"f{l-1}"] = np.asarray(mlp[l - 1], np.float64)
            raw = sum(comps.values())
            mu = raw.mean(1, keepdims=True)
            sigma = np.sqrt(((raw - mu) ** 2).mean(1) + ln["eps"])
            s = np.asarray(hs[l], np.float64)
            w = 1.0 / np.linalg.norm(s, axis=1)
            d = {k: (w[:, None] * ln_linear(c, ln["gamma"], sigma)).mean(0) for k, c in comps.items()}
            d["ln_bias"] = (w[:, None] * np.broadcast_to(np.asarray(ln["beta"], np.float64), s.shape)).mean(0)
        else:
            if l > 0:
                comps[f"a{l-1}"] = np.asarray(attn[l - 1], np.float64)
                comps[f"f{l-1}"] = np.asarray(mlp[l - 1], np.float64)
            w = 1.0 / np.linalg.norm(np.asarray(hs[l], np.float64), axis=1)
            d = {k: (w[:, None] * c).mean(0) for k, c in comps.items()}
        out.append(d)
    return out


def shares(comp: Dict[str, np.ndarray]) -> Dict[str, float]:
    m = sum(comp.values())
    n2 = float(m @ m)
    return {k: float(v @ m / n2) for k, v in comp.items()}


# ---------------------------------------------------------------- E3 and windows (pure)

def cos(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def label_e3(c_q: float, c_rep: float) -> str:
    if c_q >= c_rep:
        return "shared"
    if c_q < c_rep - E3_MARGIN:
        return "schedule-specific"
    return "between"


def window_reads(labels: Sequence[str], target: str) -> Optional[bool]:
    """≥ ⅔ of the window's cells read ``target``; None if any cell is unreadable."""
    if any(l == "unreadable" for l in labels):
        return None
    return bool(3 * sum(l == target for l in labels) >= 2 * len(labels))


# ---------------------------------------------------------------- counts (CPU)

def counts(a) -> int:
    out = Path(a.out)
    rows = ct.open_rows()
    if rows.shape[0] < ct.N_ROWS:
        raise C4pegRefused(f"rows {rows.shape[0]} < N_ROWS {ct.N_ROWS}")
    c = np.zeros(VOCAB, np.int64)
    chunk = 16384
    for i in range(0, ct.N_ROWS, chunk):
        blk = np.asarray(rows[i:min(i + chunk, ct.N_ROWS), 1:])
        c += np.bincount(blk.ravel(), minlength=VOCAB)[:VOCAB]
        if i == 0:
            print(f"first chunk populated: {int(c.sum())} tokens, {int((c > 0).sum())} types", flush=True)
    expect = ct.N_ROWS * (ct.ROW - 1)
    if int(c.sum()) != expect:
        raise C4pegRefused(f"counted {int(c.sum())} tokens, expected {expect}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "unigram_counts.npy", c)
    sha = hashlib.sha256((out / "unigram_counts.npy").read_bytes()).hexdigest()
    ct._jdump(out / "unigram_counts.json", {"code": code_sha(), "rows": [0, ct.N_ROWS], "cols": [1, ct.ROW],
                                            "total": int(c.sum()), "types": int((c > 0).sum()), "sha256": sha})
    print(json.dumps({"total": int(c.sum()), "types": int((c > 0).sum())}), flush=True)
    return 0


def load_counts(out: Path) -> np.ndarray:
    meta = json.loads((out / "unigram_counts.json").read_text())
    if hashlib.sha256((out / "unigram_counts.npy").read_bytes()).hexdigest() != meta["sha256"]:
        raise C4pegRefused("unigram_counts.npy does not match its sha256")
    return np.load(out / "unigram_counts.npy")


# ---------------------------------------------------------------- the GPU pass

def forward(model, tok, prompt: str, device):
    """hidden states (7, n, d), attention and MLP outputs per layer (6, n, d), float32 numpy."""
    import torch
    from trackC_writeup import c2_aniso as c
    from core.config import PROMPTS
    cap: Dict[str, np.ndarray] = {}
    hooks = []
    for l, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.attention.register_forward_hook(
            lambda m, i, o, l=l: cap.__setitem__(f"a{l}", o[0][0].float().cpu().numpy())))
        hooks.append(layer.mlp.register_forward_hook(
            lambda m, i, o, l=l: cap.__setitem__(f"f{l}", o[0].float().cpu().numpy())))
    try:
        enc = tok(PROMPTS[prompt], return_tensors="pt", truncation=True, max_length=c.MAX_TOKENS)
        with torch.no_grad():
            o = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()
    hs = [h[0].float().cpu().numpy() for h in o.hidden_states]
    return hs, [cap[f"a{l}"] for l in range(N_LAYERS)], [cap[f"f{l}"] for l in range(N_LAYERS)]


def check_residual(hs, attn, mlp, ln) -> float:
    """max relative residual of h_{l+1} = h_l + a_l + f_l (l < 5) and h_6 = LN(h_5 + a_5 + f_5)."""
    worst = 0.0
    for l in range(N_LAYERS - 1):
        pred = hs[l].astype(np.float64) + attn[l] + mlp[l]
        worst = max(worst, float(np.linalg.norm(pred - hs[l + 1]) / np.linalg.norm(hs[l + 1])))
    raw = hs[5].astype(np.float64) + attn[5] + mlp[5]
    mu = raw.mean(1, keepdims=True)
    s = ln["gamma"] * (raw - mu) / np.sqrt(((raw - mu) ** 2).mean(1, keepdims=True) + ln["eps"]) + ln["beta"]
    worst = max(worst, float(np.linalg.norm(s - hs[6]) / np.linalg.norm(hs[6])))
    return worst


def e1_checkpoint(WU: np.ndarray, gamma: np.ndarray, mbar: np.ndarray, cnt: np.ndarray, torch, device) -> dict:
    """E1 at one checkpoint: both readouts, every state, the gate and the beside rows."""
    sup = np.flatnonzero(cnt > 0)
    ell = np.log(cnt[sup] / cnt.sum())
    p_sup = cnt[sup] / cnt[sup].sum()
    k_top = int(round(TOP_FRAC * len(sup)))
    top = np.argsort(-cnt[sup], kind="stable")[:k_top]
    rl = torch.tensor(_std_ranks(rank_avg(ell)), device=device, dtype=torch.float64)
    G = np.random.default_rng(NULL_SEED).standard_normal((N_NULL, D))
    rng = np.random.default_rng(PERM_SEED)
    perms = [rng.permutation(len(sup)) for _ in range(N_PERM)]
    out = {"support": int(len(sup)), "top": k_top, "readouts": {}}
    for name, M in (("gamma", WU * gamma[None, :]), ("plain", WU)):
        Ms = np.asarray(M[sup], np.float64)
        Mt = torch.tensor(Ms, device=device, dtype=torch.float64)
        Z = Mt @ torch.tensor(G.T, device=device, dtype=torch.float64)          # (|S|, 1000)
        Rz = torch.argsort(torch.argsort(Z, 0), 0).double()
        Rz = Rz - Rz.mean(0)
        Rz = Rz / Rz.norm(dim=0)
        rnull = (Rz.T @ rl).cpu().numpy()
        q = float(np.percentile(rnull, NULL_Q))
        u, r2 = ls_direction(Ms, ell)
        r_star = spearman(Ms @ u, ell)
        Mc = Ms - Ms.mean(0)
        Q, Rm = np.linalg.qr(Mc)
        rp = []
        for pi in perms:
            lp = ell[pi]
            up = np.linalg.solve(Rm, Q.T @ (lp - lp.mean()))
            rp.append(spearman(Ms @ up, lp))
        pq = float(np.percentile(rp, PERM_Q))
        readable = gate_e1(r_star, q, pq)
        nk_star, b_star = nkl(Ms @ u, p_sup)
        states = []
        for l in range(N_STATES):
            x = mbar[l]
            z = Ms @ x
            r = spearman(z, ell)
            nk, b = nkl(z, p_sup)
            states.append({"r": r, "label": label_e1(r, q, readable), "frac_ceiling": r / r_star,
                           "r_top": spearman(z[top], ell[top]), "nkl": nk, "beta": b,
                           "cos_ustar": cos(x, u)})
        out["readouts"][name] = {"q": q, "null_max": float(rnull.max()), "r_star": r_star, "r2_star": r2,
                                 "perm_q": pq, "readable": readable, "nkl_star": nk_star, "beta_star": b_star,
                                 "states": states}
        del Mt, Z, Rz
    return out


def arm(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from trackC_writeup import c2_aniso as c
    from trackC_writeup import c4_read as cr
    from trackA_force.a5_seeds import TOKENIZER
    from transformers import AutoTokenizer
    torch, device = c._torch_setup()
    out = Path(a.out)
    cnt = load_counts(out)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    arm_ = a.arm
    steps = [s for s in STEPS if a.steps is None or s in a.steps]
    first = True
    for s in steps:
        f_out = out / arm_ / f"step{s}.json"
        if f_out.exists():
            continue
        d = ct.ckpt_dir(CKPT_ROOT[arm_], arm_, s)
        if not d.exists():
            raise C4pegRefused(f"{d}: missing")
        model = cr.load_local(d, device)
        sd = model.state_dict()
        WU = sd["embed_out.weight"].double().cpu().numpy()
        ln = {"gamma": sd["gpt_neox.final_layer_norm.weight"].double().cpu().numpy(),
              "beta": sd["gpt_neox.final_layer_norm.bias"].double().cpu().numpy(), "eps": LN_EPS}
        if abs(model.config.layer_norm_eps - LN_EPS) > 0 or not model.config.use_parallel_residual:
            raise C4pegRefused("config: layer_norm_eps or parallel residual not as the design assumes")
        per_prompt, comp_means, worst = {}, [], {"residual": 0.0, "e1": 0.0, "split": 0.0, "sum": 0.0, "share": 0.0}
        for p in V1:
            hs, attn, mlp = forward(model, tok, p, device)
            worst["residual"] = max(worst["residual"], check_residual(hs, attn, mlp, ln))
            rec = json.loads(cr.rec_path(REC_DIR[arm_], arm_, s, p).read_text())
            e1 = [c.state_row(h)["e1"] for h in hs]
            worst["e1"] = max(worst["e1"], float(np.max(np.abs(np.subtract(e1, rec["e1"])) / np.abs(rec["e1"]))))
            sp = []
            for l in range(N_LAYERS):
                if l < N_LAYERS - 1:
                    r = split_transition(hs[l], attn[l], mlp[l])
                else:
                    r = split_transition(hs[l], attn[l], mlp[l], ln=ln, s_next=hs[6])
                r["label"] = fall_label(r)
                worst["split"] = max(worst["split"], r["resid"])
                worst["sum"] = max(worst["sum"], r["sum_resid"])
                sp.append(r)
            cm = component_means(hs[0], attn, mlp, hs, ln)
            for l in range(N_STATES):
                m_direct = unit_mean(hs[l])
                worst["share"] = max(worst["share"], float(np.linalg.norm(sum(cm[l].values()) - m_direct)
                                                           / np.linalg.norm(m_direct)))
            per_prompt[p] = {"split": sp, "m": [unit_mean(h).tolist() for h in hs]}
            comp_means.append(cm)
        bad = {k: v for k, v in worst.items() if v > EXACT_TOL}
        if bad:
            raise C4pegRefused(f"{arm_} step {s}: known answers fail {bad}")
        mbar = np.mean([np.array(per_prompt[p]["m"]) for p in V1], 0)            # (7, 512)
        cbar = [{k: np.mean([cm[l][k] for cm in comp_means], 0) for k in comp_means[0][l]} for l in range(N_STATES)]
        rec = {"arm": arm_, "step": s, "code": code_sha(), "ckpt": str(d), "worst": worst,
               "mbar": mbar.tolist(), "mbar_norm2": [float(x @ x) for x in mbar],
               "shares": [shares(cbar[l]) for l in range(N_STATES)],
               "e1": e1_checkpoint(WU, ln["gamma"], mbar, cnt, torch, device),
               "r_ln_bias": spearman(WU[cnt > 0] @ ln["beta"], np.log(cnt[cnt > 0] / cnt.sum())),
               "per_prompt": {p: {"split": per_prompt[p]["split"]} for p in V1}}
        ct._jdump(f_out, rec)
        if first:
            g = rec["e1"]["readouts"]["gamma"]
            print(f"first output populated: {arm_} step {s}: worst {worst}; r* {g['r_star']:.3f} q {g['q']:.3f} "
                  f"perm {g['perm_q']:.3f}; r(m̄_3) {g['states'][3]['r']:.3f}", flush=True)
            first = False
        del model
        torch.cuda.empty_cache()
    print(json.dumps({"arm": arm_, "steps": steps}), flush=True)
    return 0


# ---------------------------------------------------------------- the reading (pure over records)

def load_arm(out: Path, arm_: str) -> Dict[int, dict]:
    recs = {}
    for s in STEPS:
        f = out / arm_ / f"step{s}.json"
        if not f.exists():
            raise C4pegRefused(f"{f}: missing")
        recs[s] = json.loads(f.read_text())
    return recs


def e1_label(rec: dict, l: int) -> str:
    ro = "plain" if l == N_STATES - 1 else "gamma"
    return rec["e1"]["readouts"][ro]["states"][l]["label"]


def read_all(R: Dict[str, Dict[int, dict]]) -> dict:
    res = {"e1": {}, "e2": {}, "e3": {}, "leans": {}}
    # E1 windows
    for a_ in ARMS:
        res["e1"][a_] = {}
        for w in WINDOWS[a_]:
            ws = window_steps(a_, w)
            res["e1"][a_][w] = {
                "steps": ws,
                "per_state": [{"labels": [e1_label(R[a_][s], l) for s in ws],
                               "aligned": window_reads([e1_label(R[a_][s], l) for s in ws], "aligned")}
                              for l in range(N_STATES)]}
    # E2 counts
    for a_ in ARMS:
        tab = {}
        for w in WINDOWS[a_]:
            for s in window_steps(a_, w):
                for p, pp in R[a_][s]["per_prompt"].items():
                    for t, sp in enumerate(pp["split"]):
                        if sp["label"] is None:
                            continue
                        key = f"{w}|T{t}"
                        tab.setdefault(key, {})
                        tab[key][sp["label"]] = tab[key].get(sp["label"], 0) + 1
        t_v = V_TRANSITION[a_]
        v0 = WINDOWS[a_]["V"][0]
        labs = [pp["split"][t_v]["label"] for s in STEPS if s >= v0 for pp in R[a_][s]["per_prompt"].values()
                if pp["split"][t_v]["label"] is not None]
        n_att = sum(writer_of(l) == "attention" for l in labs)
        res["e2"][a_] = {"windows": tab, "v_cells": {"transition": t_v, "from": v0, "falls": len(labs),
                                                     "attention": n_att,
                                                     "by_label": {k: labs.count(k) for k in sorted(set(labs))}}}
    # E3
    def c_at(x, y, s, l):
        return cos(R[x][s]["mbar"][l], R[y][s]["mbar"][l])
    same = {}
    for s in STEPS:
        if s == 0:
            continue
        same[s] = []
        for l in range(N_STATES):
            cq, ca, cf = c_at("A0", "F", s, l), c_at("A0", "A0p", s, l), c_at("F", "Fp", s, l)
            cq2 = c_at("A0p", "Fp", s, l)
            same[s].append({"c_A0_F": cq, "c_A0_A0p": ca, "c_F_Fp": cf, "c_A0p_Fp": cq2,
                            "label": label_e3(cq, min(ca, cf)), "label_pair2": label_e3(cq2, min(ca, cf))})
    res["e3"]["same_step"] = {str(s): v for s, v in same.items()}
    res["e3"]["windows"] = {}
    for a_ in ("A0", "F"):
        for w in WINDOWS[a_]:
            ws = [s for s in window_steps(a_, w) if s != 0]
            res["e3"]["windows"][f"{a_}|{w}"] = [
                {"labels": [same[s][l]["label"] for s in ws],
                 "shared": window_reads([same[s][l]["label"] for s in ws], "shared"),
                 "specific": window_reads([same[s][l]["label"] for s in ws], "schedule-specific")}
                for l in range(N_STATES)]
    res["e3"]["events"] = [{"A0": sa, "F": sf, "cos": [cos(R["A0"][sa]["mbar"][l], R["F"][sf]["mbar"][l])
                                                         for l in range(N_STATES)]} for sa, sf in EVENTS]
    # leans: read on pair 1 (A0, F); pair 2 (A0′, F′) reported as the replication (D2 (a))
    for tag, arms in (("", ("A0", "F")), ("_pair2", ("A0p", "Fp"))):
        e1 = {a_: {w: res["e1"][a_][w]["per_state"][6]["aligned"] for w in WINDOWS[a_]} for a_ in arms}
        vals = [v for d in e1.values() for v in d.values()]
        res["leans"]["e1" + tag] = {"cells": e1, "held": None if all(v is None for v in vals)
                                    else all(v for v in vals if v is not None)}
        e2 = {a_: res["e2"][a_]["v_cells"] for a_ in arms}
        res["leans"]["e2" + tag] = {"cells": {a_: f"{v['attention']} of {v['falls']}" for a_, v in e2.items()},
                                    "held": all(2 * v["attention"] > v["falls"] for v in e2.values() if v["falls"])}
    end = [res["e3"]["windows"][f"{a_}|end"][l]["shared"] for a_ in ("A0", "F") for l in MID_STATES]
    col = [res["e3"]["windows"][f"{a_}|collapse"][l]["specific"] for a_ in ("A0", "F") for l in MID_STATES]
    res["leans"]["e3"] = {"end_shared": end, "collapse_specific": col,
                          "end_held": all(end), "collapse_held": all(col)}
    return res


def read(a) -> int:
    out = Path(a.out)
    R = {a_: load_arm(out, a_) for a_ in ARMS}
    codes = {r["code"] for d in R.values() for r in d.values()}
    if len(codes) != 1:
        raise C4pegRefused(f"records from several code versions: {codes}")
    res = read_all(R)
    res.update(code=code_sha(), record_code=codes.pop(),
               inputs={a_: hashlib.sha256(b"".join((out / a_ / f"step{s}.json").read_bytes() for s in STEPS))
                       .hexdigest()[:16] for a_ in ARMS},
               counts_sha256=json.loads((out / "unigram_counts.json").read_text())["sha256"])
    ct._jdump(out / "reading.json", res)
    print(json.dumps(res["leans"], indent=1), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("counts"); p.add_argument("--out", required=True); p.set_defaults(fn=counts)
    p = sub.add_parser("arm"); p.add_argument("--out", required=True); p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--steps", type=int, nargs="*"); p.set_defaults(fn=arm)
    p = sub.add_parser("read"); p.add_argument("--out", required=True); p.set_defaults(fn=read)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
