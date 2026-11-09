"""
trackS_subspace/s4.py — S4 U1: the released arms (`trackS_subspace/design-S4.md`, frozen 2026-11-08
on Blocked 148: D1 (a′), D2–D5 (a)).

Arms: R2–R6 (`timaeus/L2` … `L6`, attention-only TransformerLens models, Shortformer, constant LR,
no warmup), R1 (`timaeus/L1W256-dense`, one layer, read with the fields its files show), and H
(HAPAX's `vanilla_1b` / `hapax_1b`, Pythia-1b's shape; **energy counts only**: D1 (a′) keeps
`p8_scale_ladder/design-8.md`'s reserve, so no induction score of ours runs on H, and `induction`
refuses an H arm).

Per (arm, step, prompt) the record is C2's: E₁ per state on unit tokens (`c2_aniso.state_row`:
e1, e1c, e1r, cos), with the token count and the largest single-token share beside. The toys'
states are resid_pre of each layer and the last resid_post (L + 1); H's are HF's hidden states
(17, the last after the final norm), as C2. Prompts: the text of C2's 7 v1 prompts cut at 512
Pythia tokens, retokenized per model. Per toy step, induction beside: Olsson's prefix-matching
score on repeated random tokens (`p7_motifs.formation_curve.behavioural_induction_score`, max over
heads), the previous-token score, the ICL score (loss at token 500 minus token 50).

Dates: D4 (a) as a pure function of (grid, V, window) (`half_rise`, tested equal to `a3.d4a` on
Pythia's grid). A date at the arm's first grid step, or a zero plateau, is "no break to move".
The gate: C2 unit 3b's preflight (plants A and B, 3 fresh inits, the null-matched and B-matched
centred arms with 1e-4 and 3e-4 between) at two steps of the arm. `read` refuses an arm's dates
until its gate record exists.

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackS_subspace.s4 fetch --arm R2 [R3 … H R1]
    HF_HOME=… python -m trackS_subspace.s4 check --arm R2 --out <dir>       # (i) populated, one checkpoint
    HF_HOME=… python -m trackS_subspace.s4 run --arm R2 --out <dir>         # every grid step's records
    python -m trackS_subspace.s4 induction --arm R2 --out <dir>             # induction's date (toys only)
    HF_HOME=… python -m trackS_subspace.s4 gate --arm R2 --out <dir>        # (ii) the gate at two steps
    python -m trackS_subspace.s4 read --out <dir>                           # (iii) every date
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO, V1, prompts_hash
from trackA_force.a5_seeds import MAX_TOKENS, REL_TOL, TOKENIZER as P_TOKENIZER, n_viol
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_dose as d
from trackC_writeup import c2_seeds as cs


class S4Refused(RuntimeError):
    pass


# ---------------------------------------------------------------- the arms

TOYS = {f"R{k}": f"timaeus/L{k}" for k in range(2, 7)}
R1_REPO = "timaeus/L1W256-dense"
H_REPOS = {"vanilla": "KeremSahin2210/vanilla_1b", "hapax": "KeremSahin2210/hapax_1b"}
ARMS = (*TOYS, "R1", "H_vanilla", "H_hapax")

#: The L-series' log grid (`training.yaml` `extra_save_steps`, unique, > 26); every step 1–26 is
#: also saved, and every 300 (unread here but 2100).
LOG_GRID = (28, 29, 30, 32, 33, 35, 36, 38, 40, 42, 44, 46, 48, 50, 52, 55, 57, 60, 63, 66, 69, 72,
            75, 79, 82, 86, 90, 94, 99, 103, 108, 113, 118, 124, 130, 136, 142, 149, 155, 163, 170,
            178, 186, 195, 204, 213, 223, 233, 244, 255, 267, 280, 293, 306, 320, 335, 350, 367, 384,
            401, 420, 439, 459, 481, 503, 526, 550, 576, 602, 630, 659, 690, 721, 755, 789, 826, 864,
            904, 946, 989, 1035, 1083, 1133, 1185, 1239, 1297, 1356, 1419, 1485, 1553, 1625, 1700,
            1778, 1860, 1946, 2035, 2129, 2228, 2330, 2438, 2550, 2668, 2791, 2920, 3054, 3195, 3343,
            3497, 3658, 3827, 4003, 4188, 4381, 4583, 4794, 5015, 5247, 5489, 5742, 6007, 6284, 6573,
            6876, 7194, 7525, 7872, 8235, 8615, 9012, 9428, 9863, 10318, 10794, 11291, 11812, 12357,
            12926, 13523, 14146, 14799, 15481, 16195, 16942, 17723, 18540, 19395, 20290, 21225, 22204,
            23228, 24299, 25420, 26592, 27818, 29101, 30443, 31847, 33315, 34851, 36458, 38140, 39898,
            41738, 43663, 45676, 47783, 49986, 52291, 54703, 57225, 59864, 62624, 65512, 68533, 71693,
            75000)
#: The design's window (525, 1050, 2100), at the nearest checkpoint (a tie goes to the later).
TOY_TARGETS = (525, 1050, 2100)


def nearest(target: int, have: Sequence[int]) -> int:
    return min(have, key=lambda s: (abs(s - target), -s))


def _toy_grid() -> Tuple[int, ...]:
    have = sorted(set(range(1, 27)) | set(LOG_GRID) | {2100})
    win = {nearest(t, have) for t in TOY_TARGETS}
    g = set(range(1, 27)) | set(LOG_GRID[::2]) | win | {LOG_GRID[-1]}
    return tuple(sorted(g))


TOY_GRID = _toy_grid()
#: U1's frozen window (526, 1035, 2100): it lies before induction on every toy (status-S "S4 unit 1").
TOY_WINDOW_U1 = tuple(nearest(t, sorted(set(range(1, 27)) | set(LOG_GRID) | {2100})) for t in TOY_TARGETS)
#: Blocked 149 (ii) (a), the user's: a late fixed window, the log-grid steps nearest ¼, ½ and all of
#: the run, the same on every toy; placed after seeing induction's and loss's curves, no V read.
TOY_TARGETS_LATE = (18750, 37500, 75000)
TOY_WINDOW = tuple(nearest(t, LOG_GRID) for t in TOY_TARGETS_LATE)
#: Blocked 149 (i) (a): R3–R6 (held-out loss rising from ~2100) are description only: their series
#: are recorded, no gate, no date, no outcome row. R2 alone is dated.
DATED_TOYS = ("R2",)
#: R1 saves every 10 steps from 0 to 50,000: the toy grid at its nearest saved step, and step 0.
R1_WINDOW = tuple(nearest(t, range(0, 50001, 10)) for t in TOY_TARGETS)
R1_GRID = tuple(sorted({0} | set(R1_WINDOW) | {nearest(s, range(0, 50001, 10)) for s in TOY_GRID if s <= 50000}))
#: H: every 100 from 100 to 2000, and 4000 (the window's last); window 1000, 2000, 4000 (design).
H_GRID = tuple(range(100, 2001, 100)) + (4000,)
H_WINDOW = (1000, 2000, 4000)

#: Placed before output (status-S "S4 U1"): induction's probe.
N_RAND, T_RAND = 16, 128            # 16 sequences of 128 random tokens, each repeated once
RAND_SEED = 20261108
ICL_LATE, ICL_EARLY = 500, 50       # Olsson's ICL score: loss at token 500 minus token 50
SHARE_BAR = 0.25                    # a prompt whose largest single-token share exceeds ¼ is beside
N_REINIT = 3
GATE_SEED = 20261108


def grid(arm: str) -> Tuple[int, ...]:
    return H_GRID if arm.startswith("H_") else R1_GRID if arm == "R1" else TOY_GRID


def window(arm: str) -> Tuple[int, ...]:
    return H_WINDOW if arm.startswith("H_") else R1_WINDOW if arm == "R1" else TOY_WINDOW


def n_layers(arm: str) -> int:
    return 16 if arm.startswith("H_") else 1 if arm == "R1" else int(arm[1:])


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackS_subspace/s4.py", "trackC_writeup/c2_aniso.py", "trackC_writeup/c2_dose.py",
              "trackC_writeup/c2_seeds.py", "trackA_force/a5_seeds.py",
              "p7_motifs/formation_curve.py", "p1d_cluster_ensemble/arch_null.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the reading

def half_rise(v: Sequence[float], win: Sequence[int], g: Sequence[int]) -> dict:
    """D4 (a) on grid ``g``: the first grid step with V ≥ half the median of V over ``win``."""
    v = np.asarray(v, float)
    g = list(g)
    p = float(np.median([v[g.index(s)] for s in win]))
    if p <= 0:
        return {"reading": "no change", "plateau_median": p, "index": None, "step": None}
    i = int(np.argmax(v >= p / 2))
    return {"reading": "located", "plateau_median": p, "index": i, "step": g[i]}


def date_reading(v: Sequence[float], win: Sequence[int], g: Sequence[int]) -> dict:
    """The design's outcome rows: a zero plateau, or a half-rise at the first grid step (V at its
    plateau from the first checkpoint), is "no break to move"; else the date."""
    r = half_rise(v, win, g)
    v = np.asarray(v, float)
    nb = r["step"] is None or r["index"] == 0
    return {"step": None if nb else r["step"], "plateau": r["plateau_median"],
            "first": float(v[0]), "no_break": nb}


def top_share(ids: Sequence[int]) -> float:
    _, n = np.unique(np.asarray(ids), return_counts=True)
    return float(n.max() / len(ids))


def series(recs: Dict[Tuple[int, str], dict], key: str, tol: float, g: Sequence[int],
           prompts: Sequence[str]) -> np.ndarray:
    return np.array([sum(n_viol(recs[(s, p)][key], tol) for p in prompts) for s in g], float)


def gate_steps(ind_step: Optional[int], g: Sequence[int], win: Sequence[int]) -> Tuple[int, int]:
    """The gate's two steps (placed before output): the last grid step ≤ half induction's date and
    the first ≥ twice it, else the window's first and last."""
    if ind_step is None:
        return win[0], win[-1]
    lo = [s for s in g if s <= ind_step / 2]
    hi = [s for s in g if s >= 2 * ind_step]
    return (lo[-1] if lo else g[0]), (hi[0] if hi else g[-1])


# ---------------------------------------------------------------- prompts

def p_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(P_TOKENIZER[0], revision=P_TOKENIZER[1])


def prompt_texts() -> Dict[str, str]:
    """C2's 7 prompts as text: each cut at the end character of its 512th Pythia token (the whole
    text if shorter); checked to re-encode to C2's ids."""
    from core.config import PROMPTS
    from tokenizers import processors
    tok = p_tokenizer()
    bt = tok.backend_tokenizer
    bt.post_processor = processors.ByteLevel(trim_offsets=False)
    out = {}
    for p in V1:
        enc = bt.encode(PROMPTS[p], add_special_tokens=False)
        ids = enc.ids[:MAX_TOKENS]
        text = PROMPTS[p][:enc.offsets[len(ids) - 1][1]]
        ref = tok(PROMPTS[p], truncation=True, max_length=MAX_TOKENS)["input_ids"]
        if bt.encode(text, add_special_tokens=False).ids != list(ref) or ids != list(ref):
            raise S4Refused(f"{p}: the cut text does not re-encode to C2's ids")
        out[p] = text
    return out


def texts_hash(texts: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in V1:
        h.update(k.encode() + b"\0" + texts[k].encode() + b"\0")
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the toys' forward pass

def toy_tokenizer(arm: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    repo = TOYS.get(arm, TOYS["R2"])       # R1 has no tokenizer file: the series' (vocab 5000)
    path = Path(hf_hub_download(repo, "tokenizer.json"))
    j = json.loads(path.read_text())
    m = j["model"]                      # tokenizers 0.19 reads merges as "a b" strings, not pairs
    if m["merges"] and isinstance(m["merges"][0], list):
        if any(" " in x for pair in m["merges"] for x in pair):
            raise S4Refused("a merge part holds a space: the string form would be ambiguous")
        m["merges"] = [" ".join(pair) for pair in m["merges"]]
    if m.pop("ignore_merges", False):
        raise S4Refused("ignore_merges is set: tokenizers 0.19 cannot reproduce it")
    return Tokenizer.from_str(json.dumps(j)), hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def toy_file(arm: str, step: int) -> Tuple[str, str]:
    if arm == "R1":
        return R1_REPO, f"checkpoints_early/checkpoint_{step:07d}.pth"
    return TOYS[arm], f"checkpoints/checkpoint-{step}.pt"


def toy_load(arm: str, step: int, device) -> Tuple[Dict[str, "object"], dict]:
    """The checkpoint's weights (float32, on ``device``) and its stored optimizer step; loaded with
    `weights_only=True`. Refuses on a missing key or a stored step ≠ ``step``."""
    import torch
    from huggingface_hub import hf_hub_download
    repo, f = toy_file(arm, step)
    ck = torch.load(hf_hub_download(repo, f), weights_only=True, map_location="cpu")
    if arm == "R1":
        sd, opt = ck["model_state_dict"], ck["optimizer_state_dict"]
    else:
        sd = {k[len("model."):]: v for k, v in ck["model"].items()}
        opt = ck["optimizer"]
    L = n_layers(arm)
    need = ["embed.W_E", "pos_embed.W_pos", "ln_final.w", "ln_final.b", "unembed.W_U", "unembed.b_U"]
    need += [f"blocks.{l}.{k}" for l in range(L) for k in
             ("ln1.w", "ln1.b", "attn.W_Q", "attn.W_K", "attn.W_V", "attn.W_O",
              "attn.b_Q", "attn.b_K", "attn.b_V", "attn.b_O")]
    miss = [k for k in need if k not in sd]
    extra = [k for k in sd if k.startswith(f"blocks.{L}.")]
    if miss or extra:
        raise S4Refused(f"{arm} step {step}: missing {miss[:4]}, extra {extra[:4]}")
    st = opt["state"]
    stored = int(st[min(st)]["step"]) if st else 0
    if stored != step:
        raise S4Refused(f"{arm} step {step}: the optimizer's stored step is {stored}")
    w = {k: sd[k].to(torch.float32).to(device) for k in need}
    groups = [{k: v for k, v in g.items() if k != "params"} for g in opt["param_groups"]]
    return w, {"stored_step": stored, "param_groups": groups}


def toy_init(arm: str, seed: int, device) -> Dict[str, "object"]:
    """A fresh draw from TransformerLens's default init of the arm's config (init_mode 'gpt2':
    every W_* ~ N(0, (0.8/√d_model)²), biases 0, LayerNorm (1, 0)), from torch.Generator(seed)."""
    import torch
    L, D, H, E, V, C = n_layers(arm), 256, 8, 32, 5000, 1024
    sd = 0.8 / math.sqrt(D)
    gen = torch.Generator().manual_seed(seed)

    def nrm(*shape):
        return (torch.randn(*shape, generator=gen) * sd).to(device)

    w = {"embed.W_E": nrm(V, D), "pos_embed.W_pos": nrm(C, D),
         "ln_final.w": torch.ones(D, device=device), "ln_final.b": torch.zeros(D, device=device),
         "unembed.W_U": nrm(D, V), "unembed.b_U": torch.zeros(V, device=device)}
    for l in range(L):
        p = f"blocks.{l}."
        w.update({p + "ln1.w": torch.ones(D, device=device), p + "ln1.b": torch.zeros(D, device=device),
                  p + "attn.W_Q": nrm(H, D, E), p + "attn.W_K": nrm(H, D, E),
                  p + "attn.W_V": nrm(H, D, E), p + "attn.W_O": nrm(H, E, D),
                  p + "attn.b_Q": torch.zeros(H, E, device=device),
                  p + "attn.b_K": torch.zeros(H, E, device=device),
                  p + "attn.b_V": torch.zeros(H, E, device=device),
                  p + "attn.b_O": torch.zeros(D, device=device)})
    return w


def _ln(x, w, b, eps: float = 1e-5):
    x = x - x.mean(-1, keepdim=True)
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt() * w + b


def toy_forward(w: Dict[str, "object"], ids, shortformer: bool = True):
    """TransformerLens's attention-only forward (`components/transformer_block.py`,
    `abstract_attention.py` of 2.16.1, read): Shortformer adds W_pos to the LN'd query and key
    inputs only. ``ids`` (B, n). Returns (states: L + 1 tensors (B, n, d), patterns: L tensors
    (B, H, n, n), logits (B, n, V))."""
    import torch
    L = sum(1 for k in w if k.endswith(".attn.W_Q"))
    n = ids.shape[1]
    x = w["embed.W_E"][ids]
    pos = w["pos_embed.W_pos"][:n][None]
    if not shortformer:
        x = x + pos
    states, pats = [x], []
    causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
    for l in range(L):
        p = f"blocks.{l}."
        h = _ln(x, w[p + "ln1.w"], w[p + "ln1.b"])
        qk = h + pos if shortformer else h
        q = torch.einsum("bnd,hde->bhne", qk, w[p + "attn.W_Q"]) + w[p + "attn.b_Q"][None, :, None]
        k = torch.einsum("bnd,hde->bhne", qk, w[p + "attn.W_K"]) + w[p + "attn.b_K"][None, :, None]
        v = torch.einsum("bnd,hde->bhne", h, w[p + "attn.W_V"]) + w[p + "attn.b_V"][None, :, None]
        s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        a = torch.softmax(torch.where(causal, s, torch.tensor(-torch.inf, device=x.device)), -1)
        x = x + torch.einsum("bhne,hed->bnd", a @ v, w[p + "attn.W_O"]) + w[p + "attn.b_O"]
        states.append(x)
        pats.append(a)
    logits = _ln(x, w["ln_final.w"], w["ln_final.b"]) @ w["unembed.W_U"] + w["unembed.b_U"]
    return states, pats, logits


def toy_ids(tok, texts: Dict[str, str]) -> Dict[str, List[int]]:
    return {p: tok.encode(texts[p], add_special_tokens=False).ids for p in V1}


def toy_states(w, ids: List[int], device, shortformer: bool = True):
    import torch
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        st, pats, logits = toy_forward(w, x, shortformer)
    return [h[0].cpu().numpy() for h in st], logits[0], x[0]


def toy_loss(logits, x) -> np.ndarray:
    import torch
    return torch.nn.functional.cross_entropy(logits[:-1], x[1:], reduction="none").cpu().numpy()


def rand_tokens(seed: int = RAND_SEED) -> np.ndarray:
    """N_RAND sequences of T_RAND tokens uniform on 0 … 4998 (4999 is <|endoftext|>), each repeated once."""
    r = np.random.default_rng(seed).integers(0, 4999, size=(N_RAND, T_RAND))
    return np.concatenate([r, r], axis=1)


def _probe(w, R: np.ndarray, device, shortformer: bool) -> dict:
    """Per (layer, head) on sequences R (B, 2T), each a block of T tokens repeated: the
    prefix-matching score (query i ≥ T attends to i − T + 1) and the previous-token score (i
    attends to i − 1, every i ≥ 1); the max over heads."""
    import torch
    from p7_motifs.formation_curve import behavioural_induction_score
    T = R.shape[1] // 2
    pm_pairs = [(i, i - T + 1) for i in range(T, 2 * T)]
    pt_pairs = [(i, i - 1) for i in range(1, 2 * T)]
    with torch.no_grad():
        _, pats, _ = toy_forward(w, torch.tensor(R, device=device), shortformer)
    A = torch.stack(pats, 1).cpu().numpy()          # (B, L, H, n, n)
    pm = [behavioural_induction_score(A[b], pm_pairs) for b in range(len(R))]
    pt = [behavioural_induction_score(A[b], pt_pairs) for b in range(len(R))]
    keys = sorted(pm[0])
    pmh = {f"{l}.{h}": float(np.mean([x[(l, h)] for x in pm])) for l, h in keys}
    pth = {f"{l}.{h}": float(np.mean([x[(l, h)] for x in pt])) for l, h in keys}
    return {"prefix": pmh, "prev": pth, "prefix_max": max(pmh.values()), "prev_max": max(pth.values()),
            "prefix_argmax": max(pmh, key=pmh.get), "prev_argmax": max(pth, key=pth.get)}


def induction_scores(w, device, shortformer: bool = True,
                     text_ids: Optional[Dict[str, List[int]]] = None) -> dict:
    """The design's probe on `rand_tokens` (primary); beside (placed before any induction date was
    read, after R2's last step scored only 0.08 on it while repeated text gave 0.48): the same
    scores on each prompt's first T_RAND toy tokens repeated once."""
    out = _probe(w, rand_tokens(), device, shortformer)
    if text_ids is not None:
        R = np.array([text_ids[p][:T_RAND] * 2 for p in V1])
        out["text"] = _probe(w, R, device, shortformer)
    return out


# ---------------------------------------------------------------- H's forward pass

def h_load(arm: str, step: int, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    m = from_pretrained_eager(GPTNeoXForCausalLM, H_REPOS[arm[2:]], subfolder=f"step={step}",
                              torch_dtype=torch.float32).to(device).eval()
    if next(m.parameters()).dtype != torch.float32:
        raise S4Refused(f"{arm} step {step}: weights are not float32")
    return m


def h_config_check(cfg) -> dict:
    """H's config against `pythia-1b`'s shape (the lit scan's reading); refuses on a mismatch."""
    want = {"num_hidden_layers": 16, "hidden_size": 2048, "num_attention_heads": 8,
            "rotary_pct": 0.25, "use_parallel_residual": True, "vocab_size": 50304}
    got = {k: getattr(cfg, k, None) for k in want}
    if got != want:
        raise S4Refused(f"H's config is not Pythia-1b's shape: {got}")
    return got


def h_states(model, tok, text: str, device):
    import torch
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        o = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
    return [h[0].to(torch.float32).cpu().numpy() for h in o.hidden_states], enc["input_ids"][0].tolist()


# ---------------------------------------------------------------- records

def rec_path(out: Path, arm: str, step: int, prompt: str) -> Path:
    return out / arm / "records" / f"step{step}_{prompt}.json"


def record(hs: Sequence[np.ndarray], ids: List[int], prompt: str, n_states: int) -> dict:
    rec = d.record(hs, ids, prompt, n_states)
    rec["top_share"] = top_share(ids)
    return rec


def meta(torch, arm: str, texts: Dict[str, str], extra: Optional[dict] = None) -> dict:
    return {"design": "trackS_subspace/design-S4.md", "arm": arm, "code": code_sha(),
            "prompts_hash": prompts_hash(), "texts_hash": texts_hash(texts), "prompts": list(V1),
            "dtype": "float32", "device": torch.cuda.get_device_name(0), "tf32": False,
            "torch": torch.__version__, "grid": list(grid(arm)), "window": list(window(arm)),
            **(extra or {})}


class Reader:
    """One arm's model at a step (or a fresh init), giving C2's states on the 7 texts."""

    def __init__(self, arm: str, texts: Dict[str, str], device, shortformer: Optional[bool] = None):
        self.arm, self.texts, self.device = arm, texts, device
        if arm.startswith("H_"):
            self.tok = p_tokenizer()
            self.tok_sha = None
            self.ids = {p: self.tok(texts[p], add_special_tokens=False)["input_ids"] for p in V1}
        else:
            self.tok, self.tok_sha = toy_tokenizer(arm)
            self.ids = toy_ids(self.tok, texts)
        self.shortformer = True if shortformer is None else shortformer

    def states_at(self, step: Optional[int], init_seed: Optional[int] = None):
        """{prompt: (states, ids)} and, for a toy, the per-token losses and the model's weights."""
        import torch
        out, losses, w = {}, {}, None
        if self.arm.startswith("H_"):
            if init_seed is not None:
                from transformers import AutoConfig
                from p1d_cluster_ensemble.arch_null import reinit_model
                cfg = AutoConfig.from_pretrained(H_REPOS[self.arm[2:]], subfolder=f"step={H_GRID[0]}")
                m = reinit_model(cfg, init_seed).to(self.device)
            else:
                m = h_load(self.arm, step, self.device)
            for p in V1:
                hs, ids = h_states(m, self.tok, self.texts[p], self.device)
                if ids != self.ids[p]:
                    raise S4Refused(f"{p}: H's ids differ from the tokenizer's")
                out[p] = (hs, ids)
            del m
            torch.cuda.empty_cache()
            return out, losses, None
        w = toy_init(self.arm, init_seed, self.device) if init_seed is not None else \
            toy_load(self.arm, step, self.device)[0]
        for p in V1:
            hs, logits, x = toy_states(w, self.ids[p], self.device, self.shortformer)
            out[p] = (hs, self.ids[p])
            losses[p] = toy_loss(logits, x)
        return out, losses, w


# ---------------------------------------------------------------- commands

def _dev():
    torch, device = c._torch_setup()
    return torch, device


def fetch(a) -> int:
    from huggingface_hub import hf_hub_download
    for arm in a.arm:
        t0 = time.time()
        if arm.startswith("H_"):
            for s in grid(arm):
                for f in ("config.json", "model.safetensors"):
                    hf_hub_download(H_REPOS[arm[2:]], f"step={s}/{f}")
        else:
            if arm != "R1":
                for f in ("tokenizer.json", "model.yaml", "training.yaml", "model.safetensors"):
                    hf_hub_download(TOYS[arm], f)
            for s in grid(arm):
                hf_hub_download(*toy_file(arm, s))
        print(f"{arm}: {len(grid(arm))} checkpoints in {time.time() - t0:.0f} s", flush=True)
    return 0


def check(a) -> int:
    """(i): one checkpoint per arm opened and checked populated (E₁ finite, not constant across
    layers); for a toy, the last step's prefix-matching score non-zero, its final weights equal to
    `model.safetensors`, the loss on the 7 texts; R1's shortformer / standard choice by that loss."""
    torch, device = _dev()
    texts = prompt_texts()
    out = Path(a.out)
    for arm in a.arm:
        g = grid(arm)
        rep: Dict[str, object] = {"arm": arm}
        modes = (True, False) if arm == "R1" else (True,)
        for sf in modes:
            r = Reader(arm, texts, device, sf)
            st, losses, w = r.states_at(g[-1])
            key = "shortformer" if sf else "standard"
            e1 = {p: [c.state_row(h)["e1"] for h in st[p][0]] for p in V1}
            flat = [x for v in e1.values() for x in v]
            if not np.all(np.isfinite(flat)) or any(np.ptp(v) == 0 for v in e1.values()):
                raise S4Refused(f"{arm} {key}: E₁ not populated")
            rep[key] = {"e1_last_step": e1,
                        "loss_mean": float(np.mean([losses[p].mean() for p in V1])) if losses else None}
            if w is not None:
                ind = induction_scores(w, device, sf)
                rep[key]["prefix_max"], rep[key]["prev_max"] = ind["prefix_max"], ind["prev_max"]
                if ind["prefix_max"] <= 0:
                    raise S4Refused(f"{arm}: prefix-matching score is 0 at the last step")
            rep["n_tokens"] = {p: len(r.ids[p]) for p in V1}
            rep["top_share"] = {p: top_share(r.ids[p]) for p in V1}
            rep["tokenizer_sha"] = r.tok_sha
        if arm in TOYS:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            sf_w = load_file(hf_hub_download(TOYS[arm], "model.safetensors"))
            w, info = toy_load(arm, g[-1], "cpu")
            keymap = {k: [x for x in sf_w if x.endswith(k)] for k in w}
            diffs = [float((w[k] - sf_w[v[0]].float()).abs().max()) for k, v in keymap.items() if len(v) == 1]
            diff = max(diffs) if diffs else None
            rep["final_vs_safetensors"] = {"max_abs_diff": diff,
                                           "unmatched": [k for k, v in keymap.items() if len(v) != 1]}
            if diff is None or diff > 0 or rep["final_vs_safetensors"]["unmatched"]:
                (out / arm).mkdir(parents=True, exist_ok=True)
                (out / arm / "check.json").write_text(json.dumps(rep, indent=1) + "\n")
                raise S4Refused(f"{arm}: the last checkpoint is not model.safetensors: {rep['final_vs_safetensors']}")
            rep["param_groups"] = info["param_groups"]
        if arm == "R1":
            rep["param_groups"] = toy_load(arm, g[-1], "cpu")[1]["param_groups"]
            rep["choice"] = "shortformer" if rep["shortformer"]["loss_mean"] <= rep["standard"]["loss_mean"] \
                else "standard"
        (out / arm).mkdir(parents=True, exist_ok=True)
        (out / arm / "check.json").write_text(json.dumps(rep, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rep.items() if k not in ("shortformer", "standard")}, indent=1)[:1500])
        for k in ("shortformer", "standard"):
            if k in rep:
                print(arm, k, "loss", rep[k]["loss_mean"], "prefix_max", rep[k].get("prefix_max"),
                      "prev_max", rep[k].get("prev_max"))
    return 0


#: Held-out text for the toys' loss (placed after the loss on C2's texts rose on R3–R6): the first
#: 16 Pile-CC documents over 4000 characters in A3's validation corpus, first 4000 characters, at
#: most 1024 toy tokens.
HELDOUT_CORPUS = "trackA/a3/corpus/val.jsonl.zst"
HELDOUT_N, HELDOUT_CHARS = 16, 4000
HELDOUT_STEPS = (526, 2100, 9012, 26592, 75000)


def heldout_docs(data: Path) -> Tuple[List[str], str]:
    import io
    import subprocess
    path = data / HELDOUT_CORPUS
    proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    docs = []
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        row = json.loads(line)
        if (row.get("meta") or {}).get("pile_set_name") == "Pile-CC" and len(row["text"]) > HELDOUT_CHARS:
            docs.append(row["text"][:HELDOUT_CHARS])
        if len(docs) == HELDOUT_N:
            break
    proc.kill()
    if len(docs) < HELDOUT_N:
        raise S4Refused(f"only {len(docs)} held-out documents in {path}")
    h = hashlib.sha256("\0".join(docs).encode()).hexdigest()[:12]
    return docs, h


def heldout(a) -> int:
    """Each toy's mean loss on the held-out documents at HELDOUT_STEPS (description: does the run
    keep learning text it was not trained on?)."""
    import torch
    from trackA_force.a1a import DATA
    torch_, device = _dev()
    docs, h = heldout_docs(DATA)
    out = Path(a.out)
    rows = {}
    for arm in a.arm:
        if arm not in TOYS:
            raise S4Refused(f"{arm}: heldout reads the L-series toys only")
        tok, _ = toy_tokenizer(arm)
        ids = [tok.encode(t, add_special_tokens=False).ids[:1024] for t in docs]
        rows[arm] = {}
        for s in HELDOUT_STEPS:
            w, _ = toy_load(arm, s, device)
            ls = []
            for x in ids:
                xt = torch.tensor([x], device=device)
                with torch.no_grad():
                    _, _, lg = toy_forward(w, xt)
                ls.append(float(torch.nn.functional.cross_entropy(lg[0, :-1], xt[0, 1:])))
            rows[arm][str(s)] = float(np.mean(ls))
        print(arm, {k: round(v, 2) for k, v in rows[arm].items()}, flush=True)
    rec = {"code": code_sha(), "corpus": HELDOUT_CORPUS, "docs_sha": h, "n_docs": HELDOUT_N,
           "chars": HELDOUT_CHARS, "max_tokens": 1024, "uniform": math.log(5000), "loss": rows}
    (out / "heldout.json").write_text(json.dumps(rec, indent=1) + "\n")
    return 0


def _shortformer(out: Path, arm: str) -> bool:
    if arm != "R1":
        return True
    ck = json.loads((out / arm / "check.json").read_text())
    return ck["choice"] == "shortformer"


def run(a) -> int:
    """Every grid step's records for one arm (and, for a toy, its induction probe and losses)."""
    torch, device = _dev()
    texts = prompt_texts()
    out = Path(a.out)
    arm = a.arm[0]
    r = Reader(arm, texts, device, _shortformer(out, arm))
    n_states = n_layers(arm) + 1
    (out / arm).mkdir(parents=True, exist_ok=True)
    (out / arm / "meta.json").write_text(json.dumps(meta(torch, arm, texts, {
        "tokenizer_sha": r.tok_sha, "shortformer": r.shortformer,
        "n_tokens": {p: len(r.ids[p]) for p in V1},
        "top_share": {p: top_share(r.ids[p]) for p in V1}}), indent=1) + "\n")
    t0 = time.time()
    for s in grid(arm):
        if all(rec_path(out, arm, s, p).exists() for p in V1) and \
                (arm.startswith("H_") or (out / arm / "induction" / f"step{s}.json").exists()):
            continue
        st, losses, w = r.states_at(s)
        for p in V1:
            rp = rec_path(out, arm, s, p)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(record(st[p][0], st[p][1], p, n_states)) + "\n")
        if w is not None:
            ind = induction_scores(w, device, r.shortformer, r.ids)
            # losses[j] is the loss on token j + 1 (0-based): token 500's is losses[499]
            ind["icl"] = {p: float(losses[p][ICL_LATE - 1] - losses[p][ICL_EARLY - 1]) for p in V1
                          if len(losses[p]) >= ICL_LATE}
            ind["icl_mean"] = float(np.mean(list(ind["icl"].values())))
            ind["loss_mean"] = float(np.mean([losses[p].mean() for p in V1]))
            ip = out / arm / "induction" / f"step{s}.json"
            ip.parent.mkdir(parents=True, exist_ok=True)
            ip.write_text(json.dumps(ind) + "\n")
        print(f"{arm} step {s} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


def load_recs(out: Path, arm: str) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in grid(arm):
        for p in V1:
            f = rec_path(out, arm, s, p)
            if not f.exists():
                raise S4Refused(f"{arm}: record missing {f.name}")
            recs[(s, p)] = json.loads(f.read_text())
    return recs


def induction_reading(out: Path, arm: str) -> dict:
    """Induction's dates on one toy: the half-rise of the max prefix-matching score on the arm's
    window, the previous-token and ICL scores' beside (the ICL score falls, so its negative)."""
    if arm.startswith("H_"):
        raise S4Refused("D1 (a′): no induction score of ours on H; its dates are HAPAX's published ones")
    g, win = grid(arm), window(arm)
    ind = [json.loads((out / arm / "induction" / f"step{s}.json").read_text()) for s in g]
    pm = [x["prefix_max"] for x in ind]
    pt = [x["prev_max"] for x in ind]
    tpm = [x["text"]["prefix_max"] for x in ind]
    icl = [-x["icl_mean"] for x in ind]
    return {"prefix": half_rise(pm, win, g), "prev": half_rise(pt, win, g), "icl_neg": half_rise(icl, win, g),
            "text_prefix": half_rise(tpm, win, g), "text_prefix_series": tpm,
            "text_prefix_argmax_last": ind[-1]["text"]["prefix_argmax"],
            "prefix_series": pm, "prev_series": pt, "icl_series": [x["icl_mean"] for x in ind],
            "loss_series": [x["loss_mean"] for x in ind],
            "prefix_argmax_last": ind[-1]["prefix_argmax"], "prev_argmax_last": ind[-1]["prev_argmax"]}


def induction(a) -> int:
    out = Path(a.out)
    for arm in a.arm:
        r = induction_reading(out, arm)
        (out / arm / "induction_reading.json").write_text(json.dumps(r, indent=1) + "\n")
        print(arm, {k: r[k]["step"] for k in ("prefix", "text_prefix", "prev", "icl_neg")},
              "plateau", round(r["prefix"]["plateau_median"], 3), "argmax", r["prefix_argmax_last"])
    return 0


#: H's induction dates (D1 (a′)): D4 (a) on HAPAX's published mean prefix-matching curve
#: (2511.05743v3, `prefix_induction_comparison_1b-triple_mean_visible.png`, sha256 41afea2f9e95…),
#: digitised by `s4_hapax_fig.py` at about 64 steps a pixel. Vanilla crosses in (721, 785]: 800.
#: Hapax's mean is a ramp with no transition; its crossing in (1172, 1236] is set by the window and
#: is not a formation date (`/challenge-pr` on #332), kept only as the record's entry. The gate's
#: steps on H use vanilla's date for both sides.
H_INDUCTION_PUBLISHED: Dict[str, Optional[int]] = {"H_vanilla": 800, "H_hapax": 1300}


def prompts_in(out: Path, arm: str) -> List[str]:
    """The prompts in V: every prompt for H; for a toy, those whose largest single-token share is
    ≤ ¼ (design "Prompts")."""
    if arm.startswith("H_"):
        return list(V1)
    m = json.loads((out / arm / "meta.json").read_text())
    return [p for p in V1 if m["top_share"][p] <= SHARE_BAR]


def gate(a) -> int:
    """(ii) C2 unit 3b's preflight on one arm at two steps: 3 fresh inits and the two steps; the
    plant-B dose, the null-matched and B-matched arms derived on the two steps pooled; each step
    gated on its own with those arms; the arm passes if both do."""
    arm = a.arm[0]
    if arm in TOYS and arm not in DATED_TOYS:
        raise S4Refused(f"{arm}: description only (Blocked 149 (i) (a)); no gate")
    torch, device = _dev()
    texts = prompt_texts()
    out = Path(a.out)
    g, win = grid(arm), window(arm)
    if arm == "R1":
        ind_step = None                 # one layer: no induction to date; the window's ends
    elif arm.startswith("H_"):
        ind_step = H_INDUCTION_PUBLISHED["H_vanilla"]
        if ind_step is None:
            raise S4Refused("H's published induction date is not entered (H_INDUCTION_PUBLISHED)")
        ind_step = max(s for s in g if s <= ind_step) if ind_step >= g[0] else g[0]
    else:
        ind_step = json.loads((out / arm / "induction_reading.json").read_text())["prefix"]["step"]
    steps = gate_steps(ind_step, g, win)
    prompts = prompts_in(out, arm)
    r = Reader(arm, texts, device, _shortformer(out, arm))
    rng = np.random.default_rng(GATE_SEED)
    dim = 2048 if arm.startswith("H_") else 256
    u = rng.standard_normal(dim)
    u /= np.linalg.norm(u)
    pops: Dict[str, dict] = {}
    names = [f"reinit:{k}" for k in range(N_REINIT)] + [f"step{s}" for s in steps]
    for name in names:
        if name.startswith("reinit"):
            st, _, _ = r.states_at(None, init_seed=int(name.split(":")[1]))
        else:
            st, _, _ = r.states_at(int(name[4:]))
        pops[name] = {p: {"base": cs._drops(st[p][0]), "plants": d._plants(st[p][0], u, rng)} for p in prompts}
        print(f"{arm} {name} done", flush=True)

    def flat(ns, w, key=None, dose=None):
        return [x for n in ns for p in prompts
                for x in (pops[n][p]["base"][w] if key is None else pops[n][p]["plants"][key][dose])]

    def units(ns):
        return {"base": {w: flat(ns, w) for w in ("u", "c")},
                "plants": {k: {f"{x:g}": flat(ns, None, k, f"{x:g}")
                               for x in (cs.DOSES_A if k[0] == "A" else d.DOSES_B)}
                           for k in ("A_u", "A_c", "B_u", "B_c")}}

    reinits, trained = names[:N_REINIT], names[N_REINIT:]
    nu, tr = units(reinits), units(trained)
    try:
        bd = d.b_dose(tr["base"]["u"], tr["plants"]["B_u"])
    except d.DoseRefused as e:
        raise S4Refused(f"{arm}: {e} (the uncentred count's gate refuses)")
    b_target = cs.share(cs.rate(tr["plants"]["B_u"][bd], REL_TOL), cs.rate(tr["base"]["u"], REL_TOL))
    try:
        bm = float(f"{cs.b_matched_tol(tr['base']['c'], tr['plants']['B_c'][bd], b_target):.3g}")
    except cs.SeedsRefused as e:
        bm, bm_err = None, str(e)
    else:
        bm_err = None
    nm = float(f"{cs.null_matched_tol(nu['base']['u'], nu['base']['c']):.3g}")
    arms = [nm, *d.FIXED_ARMS] + ([bm] if bm is not None else [])
    per_step = {n: d.gate_arms({"trained": units([n]), "reinit": nu}, arms, bd) for n in trained}
    gate_u = {n: {"null_rate": cs.rate(nu["base"]["u"], REL_TOL), "base_rate": cs.rate(units([n])["base"]["u"], REL_TOL),
                  "B_share": cs.share(cs.rate(units([n])["plants"]["B_u"][bd], REL_TOL),
                                      cs.rate(units([n])["base"]["u"], REL_TOL)),
                  "A_share": cs.share(cs.rate(units([n])["plants"]["A_u"][cs.A_DOSE], REL_TOL),
                                      cs.rate(units([n])["base"]["u"], REL_TOL))} for n in trained}
    for v in gate_u.values():           # the uncentred count's own gate: its null rate and plant B
        v["pass"] = v["null_rate"] <= cs.NULL_RATE and v["B_share"] >= d.B_SEEN_U
    rec = {"code": code_sha(), "meta": meta(torch, arm, texts), "arm": arm, "induction_step": ind_step,
           "steps": list(steps), "prompts": prompts, "b_dose": bd, "b_target": b_target,
           "b_matched_error": bm_err, "arms": arms, "per_step": per_step, "uncentred_at_1e-3": gate_u,
           "units_per_population": {"reinit": len(nu["base"]["c"]), **{n: len(units([n])["base"]["c"]) for n in trained}},
           "pass_by_arm": {f"{t:.3g}": all(per_step[n]["arms"][f"{t:.3g}"]["pass"] for n in trained) for t in arms},
           "pass_uncentred": all(v["pass"] for v in gate_u.values()),
           "pass": bm is not None and all(v["pass"] for v in per_step.values())}
    (out / arm / "gate.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / arm / "gate_units.json").write_text(json.dumps({"reinit": nu, **{n: units([n]) for n in trained}}) + "\n")
    print(json.dumps({k: rec[k] for k in ("arm", "induction_step", "steps", "b_dose", "arms", "pass_by_arm",
                                          "pass", "pass_uncentred", "uncentred_at_1e-3",
                                          "units_per_population", "b_matched_error")}, indent=1))
    return 0


def arm_reading(out: Path, arm: str) -> dict:
    """(iii) One arm: V's date (uncentred, REL_TOL) and the centred date on each gated arm, on the
    prompts in V; the per-prompt V beside. Refuses without a gate record."""
    gp = out / arm / "gate.json"
    if not gp.exists():
        raise S4Refused(f"{arm}: no gate record; no date is read before the gate")
    gt = json.loads(gp.read_text())
    g, win = grid(arm), window(arm)
    recs = load_recs(out, arm)
    prompts = gt["prompts"]
    vu = series(recs, "e1", REL_TOL, g, prompts)
    u = date_reading(vu, win, g)
    if not gt["pass_uncentred"]:        # refused by its gate: the series as description, no date
        u = {**u, "step": None, "refused": True}
    cent = {}
    for t in gt["arms"]:
        vc = series(recs, "e1c", t, g, prompts)
        cent[f"{t:.3g}"] = {**date_reading(vc, win, g), "gated": gt["pass_by_arm"][f"{t:.3g}"],
                            "series": vc.tolist()}
    cdates = sorted({x["step"] for x in cent.values() if x["gated"] and not x["no_break"]})
    beside = [p for p in V1 if p not in prompts]
    per_prompt = {p: date_reading(series(recs, "e1", REL_TOL, g, [p]), win, g)["step"] for p in V1}
    return {"arm": arm, "gate_pass": gt["pass"], "prompts": prompts, "beside_prompts": beside,
            "uncentred": {**u, "series": vu.tolist(), "gated": gt["pass_uncentred"]},
            "centred": {"arms": cent, "dates": cdates, "agree": len(cdates) == 1},
            "per_prompt_uncentred": per_prompt,
            "beside_series": {p: series(recs, "e1", REL_TOL, g, [p]).tolist() for p in beside}}


def arm_description(out: Path, arm: str) -> dict:
    """Blocked 149 (i) (a): a toy read as description only. V's series (uncentred, REL_TOL, the
    prompts in V), induction's reading and the held-out loss, labelled; no gate, no date."""
    g = grid(arm)
    recs = load_recs(out, arm)
    prompts = prompts_in(out, arm)
    ho = out / "heldout.json"
    return {"arm": arm, "label": "held-out loss rising", "description_only": True,
            "prompts": prompts, "grid": list(g),
            "uncentred_series": series(recs, "e1", REL_TOL, g, prompts).tolist(),
            "induction": json.loads((out / arm / "induction_reading.json").read_text()),
            "heldout": json.loads(ho.read_text())["loss"].get(arm) if ho.exists() else None}


def read(a) -> int:
    out = Path(a.out)
    res = {}
    for arm in a.arm:
        if arm in TOYS and arm not in DATED_TOYS:
            res[arm] = arm_description(out, arm)
            print(arm, "description only (Blocked 149 (i) (a))")
            continue
        try:
            r = arm_reading(out, arm)
        except S4Refused as e:
            print(f"{arm}: {e}")
            continue
        if arm.startswith("H_"):            # D1 (a′): HAPAX's published date, not ours
            r["induction"] = {"prefix": {"step": H_INDUCTION_PUBLISHED[arm]}, "source": "published"}
        else:
            r["induction"] = json.loads((out / arm / "induction_reading.json").read_text())
        res[arm] = r
        ind = r.get("induction", {})
        print(arm, "gate", r["gate_pass"], "| V", r["uncentred"]["step"], "nb" if r["uncentred"]["no_break"] else "",
              "plateau", r["uncentred"]["plateau"], "| centred", r["centred"]["dates"],
              {k: (v["step"], v["gated"]) for k, v in r["centred"]["arms"].items()},
              "| induction", ind.get("prefix", {}).get("step"), "prev", ind.get("prev", {}).get("step"))
    (out / "reading.json").write_text(json.dumps({"code": code_sha(), "arms": res}, indent=1) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cmd", choices=("fetch", "check", "run", "induction", "gate", "read", "heldout"))
    ap.add_argument("--arm", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return {"fetch": fetch, "check": check, "run": run, "induction": induction, "gate": gate,
            "read": read, "heldout": heldout}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
