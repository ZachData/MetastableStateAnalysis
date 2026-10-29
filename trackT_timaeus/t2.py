"""
trackT_timaeus/t2.py — T2: V on Timaeus's Dodecahedron (`trackT_timaeus/design-T2.md`, frozen
2026-10-29, Blocked 119: D1–D4 all (a)).

Dodecahedron (`timaeus/dodecahedron-32m`, `-67m`: Llama, RMSNorm, RoPE, GQA, tied embeddings,
FineWeb, 16k BPE) is trained under a linear 1000-step warmup to 3.5e-3, 11.7× Pythia's peak LR
over a shorter warmup, so 410m's two dates (C2 98 (c): E₁'s break, and the token-centred
cloud's) map to different Dodecahedron steps under each schedule (`to_dodeca`;
`tools/math_checks/t2_grid.py` T1–T6):

- **instantaneous**: equal LR level (t · p_p W_d / (p_d W_p) = 0.0599 t);
- **cumulative**: equal summed raw LR (t · √(p_p W_d / (p_d W_p)) = 0.245 t);
- **fraction**: the same fraction of the warmup done (t · 1000 / 1430);
- **step**: equal updates applied (t);
- **tokens**: equal tokens seen, at 4,194,304 a step (t / 2, leaning; beside, never scored) or at
  524,288 (4 t; scored only on D4 (a)'s condition, `TOKENS_524K`).

410m's dates are its 10 seeds' (`REF`, as C2 3b and 3c); a grid date s stands for a crossing in
(prev(s), s], and a schedule predicts every grid step whose interval meets a mapped 410m interval
(`predicted`). Per model and count the date is D4 (a)'s half-rise on two plateaus (D3 (a): the
early 1253–3738, or the late 5108–8158 when tokens is scored, decides readability; the other's
date joins the set when readable), the centred count over the gated arms as 3b's. The reading is
the scored schedules consistent with every readable date over 2 models × 2 counts (`joint`).
Model, data, tokenizer and depth all differ from Pythia's: description across models.

Tier 1, description (nothing here is registered).
    python -m trackT_timaeus.t2 gate --out <dir>                                  # planted effect, null
    HF_HOME=<main>/data/hf python -m trackT_timaeus.t2 known --out <dir>          # pythia-410m, A5 seed 0
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t2 fetch [--steps 1 9536]
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t2 preflight --model 32m --out <dir>
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t2 run --out <dir>
    python -m trackT_timaeus.t2 read --out <dir> --preflight <dir>
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force import a3
from trackA_force.a1a import DATA, REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import MAX_TOKENS, REL_TOL, TOKENIZER as P_TOKENIZER, n_viol
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_dose as d
from trackC_writeup import c2_seeds as cs

MODELS = {"32m": "timaeus/dodecahedron-32m", "67m": "timaeus/dodecahedron-67m"}
N_STATES = {"32m": 17, "67m": 21}               # every hidden state HF returns: L + 1, the last after the final norm
TOKENIZER_REV = "step9536"                      # tokenizer.json; unit 1 checks it is the same file at step 1 and on both
#: Dodecahedron's released steps (the card's 53); the reading's grid is 1–8158, 9536 the preflight's trained step
GRID = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 19, 22, 25, 30, 35, 40, 47, 55, 65, 75, 88, 103,
        121, 141, 165, 193, 225, 263, 308, 359, 420, 491, 574, 671, 785, 917, 1072, 1253, 1465,
        1713, 2002, 2340, 2735, 3198, 3738, 4369, 5108, 5970, 6979, 8158)
SCALE_STEP = 9536
PLATEAU_EARLY = tuple(s for s in GRID if 1253 <= s <= 3738)   # 410m's 1000–4000 under step, less 1072 (D3)
PLATEAU_LATE = tuple(s for s in GRID if 5108 <= s <= 8158)    # past every predicted date (D3; t2_grid T5)
EARLY = (1, 2)                                  # 92 (i)'s early check; every predicted date is ≥ 8
SUSTAIN_UNTIL = 4369
#: the Pythia side (pythia-410m; numbers.json cfg.lr_410m, cfg.warmup_70m) and Dodecahedron's (its card)
P_LR, P_WARMUP, P_TOKENS = 3e-4, 1430, 2_097_152
D_LR, D_WARMUP = 3.5e-3, 1000
BATCHES = {"tok4M": 4_194_304, "tok524k": 524_288}
REF = d.REF                                     # 410m's 10 seeds' grid dates: E₁ {256, 512}, centred {512, 1000}
HYPS = ("instantaneous", "cumulative", "fraction", "step", "tok4M", "tok524k")
BASE_SCORED = ("instantaneous", "cumulative", "fraction", "step")
EXCLUDED_BY_3B = ("instantaneous",)
READING_3C = "consistent with step"             # trackC_writeup/status-C.md "C2 unit 3c"; its margins there
#: D4 (a): None, or the date and `status-T.md` pointer of a reply from Timaeus confirming 524,288
#: tokens a step, quoted there before T2's first scored record. Only then is tok524k scored and
#: the late plateau primary. `run` records it; `read` refuses if it changed since.
TOKENS_524K: Optional[str] = None
FIXED_ARMS = cs.FIXED_ARMS                      # 1e-4, 3e-4
#: per model: null-matched, 1e-4, 3e-4, B-matched, from that model's preflight (3 s.f.), frozen
#: before any scored record. None: not frozen, `run` and `read` refuse.
TAU_ARMS: Dict[str, Optional[Tuple[float, ...]]] = {"32m": None, "67m": None}
DOSES_B = d.DOSES_B
N_REINIT = c.N_REINIT
ENERGY_TOL = c.ENERGY_TOL                       # placed (unit 3): |Δe1| / e1 against A5's seed-0 records
A5_SEED0 = c.A5_SEED0
KNOWN = {"zero_through": 128, "at": {256: 19, 512: 56}, "plateau": 57.0, "half_rise": 512}   # A5 seed 0 (design)
KNOWN_STEPS = tuple(s for s in STEPS if s <= 4000)
TRAINED_LOSS_MAX = 6.0                          # placed: mean next-token loss (nats) at 9536 below this; ln 16384 = 9.70
N_NULL = 20_000
SEED = 20261029


class T2Refused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t2.py", "trackC_writeup/c2_dose.py", "trackC_writeup/c2_seeds.py",
              "trackC_writeup/c2_aniso.py", "trackA_force/a5_seeds.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def tokens_scored() -> bool:
    return TOKENS_524K is not None


def scored_hyps(tok: Optional[bool] = None) -> Tuple[str, ...]:
    return BASE_SCORED + (("tok524k",) if (tokens_scored() if tok is None else tok) else ())


# ---------------------------------------------------------------- pure: the schedules

def to_dodeca(hyp: str, t: float) -> float:
    """The Dodecahedron step matching 410m's step t (t ≤ 1000, inside both warmups) under ``hyp``."""
    if hyp == "step":
        return t
    if hyp in BATCHES:
        return t * P_TOKENS / BATCHES[hyp]
    if hyp == "fraction":
        return t * D_WARMUP / P_WARMUP
    if hyp == "cumulative":
        return t * math.sqrt(P_LR * D_WARMUP / (D_LR * P_WARMUP))
    if hyp == "instantaneous":
        return t * P_LR * D_WARMUP / (D_LR * P_WARMUP)
    raise ValueError(hyp)


def prev(s: int) -> int:
    i = GRID.index(s)
    if i == 0:
        raise T2Refused("step 1 has no interval: a half-rise at 1 is unreadable")
    return GRID[i - 1]


@functools.lru_cache(maxsize=None)
def predicted(ref: Tuple[int, ...], hyp: str) -> FrozenSet[int]:
    """The grid steps a schedule predicts: each 410m date g stands for a crossing in (prev(g), g]
    on Pythia's grid; mapped, it is (m(prev(g)), m(g)], and the grid reads s when (prev(s), s]
    meets it."""
    out = set()
    for g in ref:
        lo, hi = to_dodeca(hyp, d.prev(g)), to_dodeca(hyp, g)
        out |= {s for s in GRID[1:] if s > lo and prev(s) < hi}
    return frozenset(out)


def band(kind: str, hyp: str) -> Tuple[float, float]:
    """The mapped crossing interval of 410m's whole date set for ``kind``."""
    return to_dodeca(hyp, d.prev(min(REF[kind]))), to_dodeca(hyp, max(REF[kind]))


def consistent(dates: Iterable[int], kind: str, hyps: Sequence[str]) -> FrozenSet[str]:
    """The schedules in ``hyps`` that predict at least one of ``dates`` (one model's dates: its two
    plateaus', or its arms', when they disagree)."""
    ds = set(dates)
    return frozenset(h for h in hyps if ds & predicted(REF[kind], h))


# ---------------------------------------------------------------- pure: the reading

def series(recs: Dict[Tuple[int, str], dict], key: str, rel_tol: float = REL_TOL,
           grid: Sequence[int] = GRID) -> np.ndarray:
    """V on ``grid`` for ``key``: violations summed over the 7 prompts."""
    return np.array([sum(n_viol(recs[(s, p)][key], rel_tol) for p in V1) for s in grid], float)


def half_rise(v: np.ndarray, window: Sequence[int], grid: Sequence[int] = GRID) -> dict:
    """D4 (a) on ``grid``: the first grid step with V ≥ half the median of V over ``window``."""
    v = np.asarray(v, float)
    p = float(np.median([v[grid.index(s)] for s in window]))
    if p <= 0:
        return {"plateau_median": p, "step": None}
    return {"plateau_median": p, "step": grid[int(np.argmax(v >= p / 2))]}


def sustained(v: np.ndarray, p: float, until: int = SUSTAIN_UNTIL) -> Optional[int]:
    """Beside (3c): the first grid step from which V stays ≥ p / 2 at every grid step to ``until``."""
    if p <= 0:
        return None
    first = None
    for i in reversed([i for i, s in enumerate(GRID) if s <= until]):
        if v[i] < p / 2:
            break
        first = GRID[i]
    return first


def date_reading(v: np.ndarray, window: Sequence[int], beside: bool = True) -> dict:
    """One count's half-rise on one plateau; unreadable if the plateau is 0 or V at an EARLY step
    is ≥ half of it."""
    v = np.asarray(v, float)
    r = half_rise(v, window)
    p = r["plateau_median"]
    early = {s: float(v[GRID.index(s)]) for s in EARLY}
    bad = p <= 0 or max(early.values()) >= p / 2
    out = {"step": None if bad else r["step"], "plateau": p, "early": early, "unreadable": bad}
    if beside:
        out["sustained"] = sustained(v, p)
    return out


def count_reading(v: np.ndarray, tok: Optional[bool] = None, beside: bool = True) -> dict:
    """D3 (a): both plateaus read; the primary (late if tokens is scored, else early) decides
    readability, and the date set is the union of the readable plateaus' dates."""
    tok = tokens_scored() if tok is None else tok
    rs = {"early": date_reading(v, PLATEAU_EARLY, beside), "late": date_reading(v, PLATEAU_LATE, beside)}
    primary = "late" if tok else "early"
    bad = rs[primary]["unreadable"]
    dates = [] if bad else sorted({r["step"] for r in rs.values() if not r["unreadable"]})
    return {"primary": primary, "plateaus": rs, "dates": dates, "unreadable": bad}


def tokens_beside(dates: Sequence[int], kind: str, hyps: Sequence[str]) -> Dict[str, bool]:
    """Which tokens columns not scored the dates also fit."""
    return {h: bool(set(dates) & predicted(REF[kind], h)) for h in ("tok4M", "tok524k") if h not in hyps}


def model_reading(vu: np.ndarray, vcs: Dict[str, np.ndarray], gate: Dict[str, bool],
                  tok: Optional[bool] = None, beside: bool = True) -> dict:
    """One model, as `c2_dose.size_reading`: the uncentred dates; the centred dates over the arms
    that passed their gate and are readable; each count's consistent scored schedules."""
    tok = tokens_scored() if tok is None else tok
    hyps = scored_hyps(tok)
    u = count_reading(vu, tok, beside)
    arms = {k: {**count_reading(v, tok, beside), "gated": gate[k]} for k, v in vcs.items()}
    cdates = sorted({x for a in arms.values() if a["gated"] and not a["unreadable"] for x in a["dates"]})
    out = {"uncentred": {**u, "consistent": None, "tokens_beside": None},
           "centred": {"dates": cdates, "arms": arms, "agree": len(cdates) == 1, "unreadable": not cdates,
                       "consistent": None, "tokens_beside": None}}
    for kind, k, ds in (("uncentred", "u", u["dates"]), ("centred", "c", cdates)):
        if ds:
            out[kind]["consistent"] = sorted(consistent(ds, k, hyps), key=HYPS.index)
            out[kind]["tokens_beside"] = tokens_beside(ds, k, hyps)
    return out


def joint(per_model: Dict[str, dict], tok: Optional[bool] = None) -> dict:
    """The scored schedules consistent with every readable date over the models and both counts;
    beside, the same less the schedule 3b excluded, and 3c's reading."""
    hyps = scored_hyps(tok)
    sets, unread = [], []
    for m, r in per_model.items():
        for kind in ("uncentred", "centred"):
            if r[kind]["consistent"] is None:
                unread.append(f"{m} {kind}")
            else:
                sets.append(frozenset(r[kind]["consistent"]))
    if not sets:
        return {"survivors": [], "label": "unreadable", "readable_dates": 0, "unreadable": unread,
                "with_3b": [], "scored": list(hyps), "reading_3c": READING_3C}
    surv = frozenset(hyps).intersection(*sets)
    label = ("not told apart" if surv == frozenset(hyps) else "neither" if not surv
             else "consistent with " + " and ".join(h for h in hyps if h in surv))
    return {"survivors": [h for h in hyps if h in surv], "label": label, "readable_dates": len(sets),
            "unreadable": unread, "with_3b": [h for h in hyps if h in surv and h not in EXCLUDED_BY_3B],
            "scored": list(hyps), "reading_3c": READING_3C}


# ---------------------------------------------------------------- pure: the calibration gate

def synth(x: float, plateau: float = 40.0) -> np.ndarray:
    """A V series on GRID rising through half ``plateau`` at step ``x`` (a steep logistic in log
    step): its half-rise on either plateau is the first grid step ≥ x."""
    s = np.asarray(GRID, float)
    return plateau / (1.0 + (x / s) ** 8)


def _one_arm(v: np.ndarray) -> Dict[str, np.ndarray]:
    return {"arm": v}


def planted_effect(tok: Optional[bool] = None) -> dict:
    """Per scored schedule and count: V crossing at the geometric middle of the schedule's band
    reads the first grid step past it on the primary plateau, and a date set whose consistent
    schedules contain it; the same on all 4 cells keeps it in `joint`. (With tokens scored, tok524k's
    centred crossing sits inside the early window, which then reads an earlier date beside: 2002
    and 2340, both tok524k's.)"""
    tok = tokens_scored() if tok is None else tok
    hyps = scored_hyps(tok)
    out, ok = {}, True
    for h in hyps:
        cells = {}
        for kind, k in (("uncentred", "u"), ("centred", "c")):
            lo, hi = band(k, h)
            x = math.sqrt(lo * hi)
            r = count_reading(synth(x), tok, beside=False)
            want = min(s for s in GRID if s >= x)
            cons = sorted(consistent(r["dates"], k, hyps), key=HYPS.index)
            good = r["plateaus"][r["primary"]]["step"] == want and want in r["dates"] and h in cons
            ok &= good
            cells[kind] = {"x": x, "want": want, "dates": r["dates"], "consistent": cons,
                           "excluded": [g for g in hyps if g not in cons], "pass": good}
        xu, xc = (math.sqrt(a * b) for a, b in (band("u", h), band("c", h)))
        per = {m: model_reading(synth(xu), _one_arm(synth(xc)), {"arm": True}, tok, beside=False) for m in MODELS}
        j = joint(per, tok)
        ok &= h in j["survivors"]
        out[h] = {"cells": cells, "joint": j["label"], "joint_keeps": h in j["survivors"]}
    return {"per_schedule": out, "pass": bool(ok)}


def planted_null(rng: np.random.Generator, n: int = N_NULL, tok: Optional[bool] = None) -> dict:
    """Crossings uniform in log step over (1, 4369], drawn independently for every model × count,
    through `joint`: each reading's share by chance (not a p)."""
    tok = tokens_scored() if tok is None else tok
    labels: Dict[str, int] = {}
    keeps = {h: 0 for h in scored_hyps(tok)}
    for _ in range(n):
        xs = np.exp(rng.uniform(0.0, math.log(SUSTAIN_UNTIL), size=2 * len(MODELS)))
        per = {m: model_reading(synth(xs[2 * i]), _one_arm(synth(xs[2 * i + 1])), {"arm": True}, tok, beside=False)
               for i, m in enumerate(MODELS)}
        j = joint(per, tok)
        labels[j["label"]] = labels.get(j["label"], 0) + 1
        for h in j["survivors"]:
            keeps[h] += 1
    return {"n": n, "label_share": {k: v / n for k, v in sorted(labels.items(), key=lambda kv: -kv[1])},
            "keeps_share": {h: v / n for h, v in keeps.items()}}


def null_matched_tol(null_u: Sequence[float], null_c: Sequence[float]) -> float:
    """`c2_seeds.null_matched_tol`, refusing when the re-inits' uncentred rate at 1e-3 is 0: the
    quantile is then the null's largest centred drop, set by one unit (3c's Parked 0, carried to
    the next unit that derives a centred arm; `trackC_writeup/status-C.md` "C2 unit 3c"), and when
    the quantile is ≤ 0, a tolerance that counts energy rises as violations (found on `-67m`'s
    first preflight, −3.05e-4)."""
    if cs.rate(null_u, REL_TOL) <= 0:
        raise T2Refused("the re-inits' uncentred rate at 1e-3 is 0: the null-matched arm is one unit's drop")
    t = cs.null_matched_tol(null_u, null_c)
    if t <= 0:
        raise T2Refused(f"the null-matched tolerance is {t:.3g} ≤ 0: it counts energy rises as violations")
    return t


# ---------------------------------------------------------------- model passes

def load(repo: str, revision: str, device):
    """A causal LM at ``revision``, float32, eager; a Llama checked against the fields its
    transformers-5 config carries that 4.44 does not read."""
    import torch
    from transformers import AutoModelForCausalLM
    from core.models import from_pretrained_eager
    m = from_pretrained_eager(AutoModelForCausalLM, repo, revision=revision,
                              torch_dtype=torch.float32).to(device).eval()
    if next(m.parameters()).dtype != torch.float32:
        raise T2Refused(f"{repo} {revision}: weights are not float32")
    if m.config.model_type == "llama":
        check_llama_config(m.config)
    return m


def check_llama_config(cfg) -> None:
    """4.44's Llama takes head_dim = hidden / heads and rope_theta's default 10000; refuse unless
    the stored head_dim and rope_parameters say the same."""
    hd = getattr(cfg, "head_dim", None)
    rp = getattr(cfg, "rope_parameters", None) or {}
    bad = []
    if hd is not None and hd != cfg.hidden_size // cfg.num_attention_heads:
        bad.append(f"head_dim {hd}")
    if rp and (rp.get("rope_type", "default") != "default" or float(rp.get("rope_theta", cfg.rope_theta)) != cfg.rope_theta):
        bad.append(f"rope_parameters {rp}")
    if getattr(cfg, "rope_scaling", None):
        bad.append(f"rope_scaling {cfg.rope_scaling}")
    if bad:
        raise T2Refused(f"config not as 4.44's Llama reads it: {bad}")


def tokenizer_json(model: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(MODELS[model], "tokenizer.json", revision=TOKENIZER_REV))


def tokenizer(model: str):
    """Dodecahedron's BPE from its tokenizer.json, saved by tokenizers ≥ 0.20 (merges as pairs);
    the installed 0.19 reads merges as "a b", so each pair is joined, refusing a piece with a space.
    No BOS is added (its post-processor is the sequence alone), as on Pythia."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    j = json.loads(tokenizer_json(model).read_text())
    merges = j["model"]["merges"]
    if merges and isinstance(merges[0], list):
        if any(" " in a or " " in b for a, b in merges):
            raise T2Refused(f"{model}: a merge piece holds a space; joining pairs would change them")
        j["model"]["merges"] = [f"{a} {b}" for a, b in merges]
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(j)),
                                   model_input_names=["input_ids", "attention_mask"])


def p_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(P_TOKENIZER[0], revision=P_TOKENIZER[1])


def reinit(cfg, seed: int):
    """A LlamaForCausalLM of ``cfg`` drawn by HF's Llama init (Linear and Embedding N(0,
    initializer_range), RMSNorm 1) under ``torch.manual_seed(seed)``; float32 (Dodecahedron stores
    float32), eval mode, eager attention."""
    import torch
    from transformers import LlamaForCausalLM
    cfg._attn_implementation = "eager"
    torch.manual_seed(int(seed))
    m = LlamaForCausalLM(cfg).to(torch.float32).eval()
    check_llama_config(m.config)
    return m


def weight_sds(model) -> Dict[str, float]:
    return {n: float(p.detach().float().std()) for n, p in model.named_parameters() if p.dim() > 1}


def convention(model, tok, device) -> dict:
    """Unit 1's check that HF returns L + 1 states, the first the embeddings and the last the
    final norm of the last block's output (as GPT-NeoX and GPT-2), and the mean next-token loss
    over the 7 prompts."""
    import torch
    from core.config import PROMPTS
    got = {}
    h = model.model.layers[-1].register_forward_hook(lambda mod, i, o: got.__setitem__("last", o[0]))
    losses = []
    try:
        for k, p in enumerate(V1):
            enc = tok(PROMPTS[p], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            ids = enc["input_ids"].to(device)
            with torch.no_grad():
                o = model(input_ids=ids, labels=ids, output_hidden_states=True)
            losses.append(float(o.loss))
            if k == 0:
                hs = o.hidden_states
                n_layers = model.config.num_hidden_layers
                with torch.no_grad():
                    d_last = float((hs[-1] - model.model.norm(got["last"])).abs().max())
                    d_emb = float((hs[0] - model.model.embed_tokens(ids)).abs().max())
                first = {"n_states": len(hs), "n_layers": n_layers, "last_is_final_norm_maxdiff": d_last,
                         "first_is_embedding_maxdiff": d_emb, "n_tokens": int(ids.shape[1])}
    finally:
        h.remove()
    ok = first["n_states"] == first["n_layers"] + 1 and first["last_is_final_norm_maxdiff"] == 0.0 \
        and first["first_is_embedding_maxdiff"] == 0.0
    return {**first, "loss_per_prompt": losses, "loss_mean": float(np.mean(losses)), "pass": ok}


def rec_path(out: Path, model: str, step: int, prompt: str) -> Path:
    return out / model / "records" / f"step{step}_{prompt}.json"


def meta(torch) -> dict:
    return {"design": "trackT_timaeus/design-T2.md", "code": code_sha(), "tau_arms": TAU_ARMS,
            "tokens_524k": TOKENS_524K, "scored": list(scored_hyps()), "models": MODELS,
            "grid": list(GRID), "plateau_early": list(PLATEAU_EARLY), "plateau_late": list(PLATEAU_LATE),
            "early": list(EARLY), "ref": REF, "prompts_hash": prompts_hash(), "prompts": list(V1),
            "tokenizer_rev": TOKENIZER_REV, "max_tokens": MAX_TOKENS, "dtype": "float32",
            "device": torch.cuda.get_device_name(0), "tf32": False, "torch": torch.__version__,
            "hf_home": os.environ.get("HF_HOME")}


def _frozen(model: str) -> Tuple[float, ...]:
    if TAU_ARMS.get(model) is None:
        raise T2Refused(f"TAU_ARMS[{model!r}] is not set: the rule is not frozen")
    return TAU_ARMS[model]


def fetch(a) -> int:
    """Download config.json, model.safetensors and tokenizer.json at the steps asked, for the
    models asked; nothing is read."""
    from huggingface_hub import snapshot_download
    t0 = time.time()
    for m in a.models:
        for s in a.steps:
            snapshot_download(MODELS[m], revision=f"step{s}",
                              allow_patterns=["config.json", "model.safetensors", "tokenizer.json"])
        print(f"{m}: {len(a.steps)} steps ({time.time() - t0:.0f} s)", flush=True)
    return 0


def known(a) -> int:
    """The known answer: this producer's path (AutoModel, `c2_aniso.states`, `c2_dose.record`,
    `series`) on pythia-410m at A5's steps 0–4000 reproduces seed 0's records (tokens, e1 within
    ENERGY_TOL) and its E₁ counts (0 through 128, 19 at 256, 56 at 512, plateau 57, half-rise 512)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = p_tokenizer()
    a5 = DATA / A5_SEED0 / "records"
    recs, worst = {}, 0.0
    for step in KNOWN_STEPS:
        model = load(c.MODEL, f"step{step}", device)
        for p in V1:
            hs, ids = c.states(model, tok, p, device)
            rec = d.record(hs, ids, p, model.config.num_hidden_layers + 1)
            ref = json.loads((a5 / f"step{step}_{p}.json").read_text())
            if rec["ids_sha"] != ref["ids_sha"]:
                raise T2Refused(f"410m step {step} {p}: tokens differ from A5's record")
            dev = float(np.max(np.abs(np.subtract(rec["e1"], ref["e1"])) / np.asarray(ref["e1"])))
            worst = max(worst, dev)
            recs[(step, p)] = rec
        del model
        torch.cuda.empty_cache()
        print(f"410m step {step} done", flush=True)
    v = series(recs, "e1", grid=KNOWN_STEPS)
    a5v = series({k: json.loads((a5 / f"step{k[0]}_{k[1]}.json").read_text()) for k in recs}, "e1", grid=KNOWN_STEPS)
    hr = half_rise(v, (1000, 2000, 4000), KNOWN_STEPS)
    checks = {"e1_within_tol": worst <= ENERGY_TOL,
              "zero_through_128": all(v[KNOWN_STEPS.index(s)] == 0 for s in KNOWN_STEPS if s <= KNOWN["zero_through"]),
              **{f"V_at_{s}": v[KNOWN_STEPS.index(s)] == n for s, n in KNOWN["at"].items()},
              "plateau": hr["plateau_median"] == KNOWN["plateau"], "half_rise": hr["step"] == KNOWN["half_rise"],
              "a3_d4a_agrees": a3.d4a(np.concatenate([v, np.zeros(len(STEPS) - len(v))]))["step"] == hr["step"],
              "V_equals_A5_records": bool(np.array_equal(v, a5v))}
    checks = {k: bool(x) for k, x in checks.items()}
    res = {"code": code_sha(), "repo": c.MODEL, "a5_records": str(A5_SEED0), "steps": list(KNOWN_STEPS),
           "V_u": v.tolist(), "half_rise": hr, "e1_max_rel_dev": worst, "energy_tol": ENERGY_TOL,
           "checks": checks, "pass": all(checks.values())}
    (out / "known_answer.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


def gate(a) -> int:
    """The pure part of the calibration gate: planted effect, planted null."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"code": code_sha(), "tokens_524k": TOKENS_524K, "scored": list(scored_hyps()), "seed": SEED,
           "planted_effect": planted_effect(), "planted_null": planted_null(np.random.default_rng(SEED))}
    res["pass"] = res["planted_effect"]["pass"]
    (out / "gate.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


def preflight(a) -> int:
    """As C2 3b's and 3c's, on one model: its step 1 (weight SDs, a populated record), 3 re-inits
    of its config (HF's Llama init) and step 9536 (the trained population, and the state
    convention and loss checks); plants A and B through both counts; the null-matched and B-matched
    arms derived; each arm gated. Writes the design-preflight tool's inputs for both counts."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoConfig
    m_ = a.model
    repo = MODELS[m_]
    out = Path(a.out) / m_
    out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(m_)
    tok_sha = hashlib.sha256(tokenizer_json(m_).read_bytes()).hexdigest()
    cfg = AutoConfig.from_pretrained(repo, revision=f"step{SCALE_STEP}")
    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(cfg.hidden_size)
    u /= np.linalg.norm(u)
    checks: Dict[str, dict] = {}
    step1 = load(repo, "step1", device)
    sd1 = weight_sds(step1)
    checks["step1"] = convention(step1, tok, device)
    hs, ids = c.states(step1, tok, V1[0], device)
    r1 = d.record(hs, ids, V1[0], N_STATES[m_])
    checks["step1"]["record"] = {"n": r1["n"], "states": len(r1["e1"]), "e1[0]": r1["e1"][0], "e1[-1]": r1["e1"][-1]}
    del step1
    pops: Dict[str, dict] = {}
    names = [f"reinit:{s}" for s in range(N_REINIT)] + [f"step{SCALE_STEP}"]
    init_check = {}
    for name in names:
        if name.startswith("reinit"):
            model = reinit(cfg, int(name.split(":")[1])).to(device)
            sds = weight_sds(model)
            r = {k: sds[k] / sd1[k] for k in sd1}
            init_check[name] = {"min_ratio": min(r.values()), "max_ratio": max(r.values()),
                                "worst": max(r, key=lambda k: abs(math.log(r[k])))}
        else:
            model = load(repo, f"step{SCALE_STEP}", device)
            checks[name] = convention(model, tok, device)
        if len(c.states(model, tok, V1[0], device)[0]) != N_STATES[m_]:
            raise T2Refused(f"{m_} {name}: not {N_STATES[m_]} states")
        per = {}
        for p in V1:
            hs, _ = c.states(model, tok, p, device)
            per[p] = {"base": cs._drops(hs), "plants": d._plants(hs, u, rng)}
        pops[name] = {"per_prompt": per}
        del model
        torch.cuda.empty_cache()
        print(f"{m_} {name} done", flush=True)
    tr_name = names[-1]
    conv_ok = checks["step1"]["pass"] and checks[tr_name]["pass"] and checks[tr_name]["loss_mean"] < TRAINED_LOSS_MAX

    reinits, trained = names[:-1], names[-1:]

    def flat(ns, w, key=None, dose=None):
        return [x for n in ns for p in V1
                for x in (pops[n]["per_prompt"][p]["base"][w] if key is None
                          else pops[n]["per_prompt"][p]["plants"][key][dose])]

    units = {pop: {"base": {w: flat(ns, w) for w in ("u", "c")},
                   "plants": {k: {f"{x:g}": flat(ns, None, k, f"{x:g}")
                                  for x in (cs.DOSES_A if k[0] == "A" else DOSES_B)}
                              for k in ("A_u", "A_c", "B_u", "B_c")}}
             for pop, ns in (("reinit", reinits), ("trained", trained))}
    (out / "preflight_units.json").write_text(json.dumps(units) + "\n")
    tr, nu = units["trained"], units["reinit"]
    bd = d.b_dose(tr["base"]["u"], tr["plants"]["B_u"])
    b_target = cs.share(cs.rate(tr["plants"]["B_u"][bd], REL_TOL), cs.rate(tr["base"]["u"], REL_TOL))
    derived: Dict[str, object] = {"b_dose": bd, "b_matched": cs.b_matched_tol(tr["base"]["c"], tr["plants"]["B_c"][bd], b_target)}
    try:
        derived["null_matched"] = null_matched_tol(nu["base"]["u"], nu["base"]["c"])
    except T2Refused as e:
        derived["null_matched"], derived["null_matched_refused"] = None, str(e)
    again = tuple(([float(f"{derived['null_matched']:.3g}")] if derived["null_matched"] is not None else [])
                  + list(FIXED_ARMS) + [float(f"{derived['b_matched']:.3g}")])
    frozen = TAU_ARMS.get(m_)
    if frozen and tuple(frozen) != again:
        raise T2Refused(f"{m_}: the preflight derives {again}, not the frozen {frozen}")
    arms = frozen or again
    g = d.gate_arms(units, arms, bd)
    g["pass"] = g["pass"] and derived["null_matched"] is not None
    gate_u = {"null_rate": cs.rate(nu["base"]["u"], REL_TOL), "base_rate": cs.rate(tr["base"]["u"], REL_TOL),
              "B_share": b_target, "b_dose": bd,
              "A_share": cs.share(cs.rate(tr["plants"]["A_u"][cs.A_DOSE], REL_TOL), cs.rate(tr["base"]["u"], REL_TOL))}
    for w, key in (("c", "B_c"), ("u", "B_u")):
        tool_in = {"statistic": f"{'centred' if w == 'c' else 'uncentred'} E_1 relative drop per (prompt, "
                                f"transition), (E_t − E_t+1) / |E_t|, dodecahedron-{m_} (t2)",
                   "direction": "greater", "nulls": {"reinit": nu["base"][w]}, "moved_marks": [],
                   "trained": tr["base"][w], "trained_step": str(SCALE_STEP),
                   "scored_steps": [str(s) for s in GRID],
                   "planted": {f"{x:g}": nu["plants"][key][f"{x:g}"] for x in DOSES_B},
                   "bounds": None, "decision_p": 0.05}
        (out / f"preflight_input_{w}.json").write_text(json.dumps(tool_in) + "\n")
    rec = {"code": code_sha(), "meta": meta(torch), "model": m_, "repo": repo, "tokenizer_sha256": tok_sha,
           "derived": derived, "b_target": b_target, "arms": list(arms), "frozen": frozen is not None, "gate": g,
           "uncentred_at_1e-3": gate_u, "units_per_population": {p: len(units[p]["base"]["c"]) for p in units},
           "scales": {pop: {"s_u": cs.scale(units[pop]["base"]["u"]), "s_c": cs.scale(units[pop]["base"]["c"])}
                      for pop in units},
           "doses_A": list(cs.DOSES_A), "doses_B": list(DOSES_B), "u_seed": SEED, "init_check": init_check,
           "checks": checks, "checks_pass": bool(conv_ok), "trained_loss_max": TRAINED_LOSS_MAX}
    (out / "preflight_plants.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"derived": derived, "arms": list(arms), "gate": g, "uncentred": gate_u,
                      "init_check": init_check, "checks": checks, "checks_pass": conv_ok}, indent=1))
    return 0 if g["pass"] and conv_ok else 1


def run(a) -> int:
    for m_ in a.models:
        _frozen(m_)
    os.environ["HF_HUB_OFFLINE"] = "1"            # fetched before; nothing is downloaded here
    torch, device = c._torch_setup()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mt = meta(torch)
    plan = out / "plan.json"
    if plan.exists() and json.loads(plan.read_text())["tokens_524k"] != TOKENS_524K:
        raise T2Refused("TOKENS_524K changed after records were written (D4 (a))")
    plan.write_text(json.dumps({**mt, "run_models": list(a.models)}, indent=1) + "\n")
    first, t0 = True, time.time()
    for m_ in a.models:
        tok = tokenizer(m_)
        rec_path(out, m_, 1, V1[0]).parent.mkdir(parents=True, exist_ok=True)
        for step in GRID:
            todo = [p for p in V1 if not rec_path(out, m_, step, p).exists()]
            if not todo:
                continue
            model = load(MODELS[m_], f"step{step}", device)
            for p in todo:
                hs, ids = c.states(model, tok, p, device)
                rec = d.record(hs, ids, p, N_STATES[m_])
                rec.update(model=m_, repo=MODELS[m_], step=step, revision=f"step{step}", code=mt["code"])
                rec_path(out, m_, step, p).write_text(json.dumps(rec) + "\n")
                if first:
                    print(f"first record populated: {m_} step {step} {p}, n {rec['n']}, "
                          f"states {len(rec['e1'])}, e1[0] {rec['e1'][0]:.6f}, e1c[0] {rec['e1c'][0]:.6f}, "
                          f"cos[-1] {rec['cos'][-1]:.4f}", flush=True)
                    first = False
            del model
            torch.cuda.empty_cache()
        print(f"{m_} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ---------------------------------------------------------------- reading

def load_model_recs(dr: Path, model: str) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in GRID:
        for p in V1:
            f = dr / f"step{s}_{p}.json"
            if not f.exists():
                raise T2Refused(f"{f}: missing")
            r = json.loads(f.read_text())
            if r.get("model") != model:
                raise T2Refused(f"{f}: model {r.get('model')}, not {model}")
            recs[(s, p)] = r
    return recs


def read(a) -> int:
    out = Path(a.out)
    plan = json.loads((out / "plan.json").read_text())
    if plan["tokens_524k"] != TOKENS_524K:
        raise T2Refused(f"TOKENS_524K is {TOKENS_524K!r}, the run's was {plan['tokens_524k']!r} (D4 (a))")
    per, beside, codes = {}, {}, {}
    for m_ in MODELS:
        arms = _frozen(m_)
        pf = json.loads((Path(a.preflight) / m_ / "preflight_plants.json").read_text())
        if [float(f"{t:.3g}") for t in pf["arms"]] != list(arms):
            raise T2Refused(f"{m_}: the preflight's arms {pf['arms']} are not the frozen {arms}")
        gate_ = {f"{t:.3g}": pf["gate"]["arms"][f"{t:.3g}"]["pass"] for t in pf["arms"]}
        recs = load_model_recs(out / m_ / "records", m_)
        codes[m_] = sorted({r["code"] for r in recs.values()})
        vu = series(recs, "e1")
        vcs = {f"{t:.3g}": series(recs, "e1c", t) for t in arms}
        per[m_] = model_reading(vu, vcs, gate_)
        beside[m_] = {"V_u": vu.tolist(), "V_c": {k: v.tolist() for k, v in vcs.items()},
                      "V_c_at_1e-3": series(recs, "e1c", 1e-3).tolist(),
                      "V_raw_centred": {k: series(recs, "e1r", t).tolist() for k, t in zip(vcs, arms)},
                      "mean_cos_layer_mean": [float(np.mean([recs[(st, p)]["cos"] for p in V1])) for st in GRID]}
    if len({tuple(v) for v in codes.values()}) != 1:
        raise T2Refused(f"records from more than one code version: {codes}")
    j = joint(per)
    null = planted_null(np.random.default_rng(SEED))
    res = {"code": code_sha(), "record_codes": codes, "tau_arms": TAU_ARMS, "tokens_524k": TOKENS_524K,
           "ref": REF, "grid": list(GRID),
           "predicted": {kind: {h: sorted(predicted(REF[kind[0]], h)) for h in HYPS} for kind in ("uncentred", "centred")},
           "early_first": {m: {"uncentred": r["uncentred"]["plateaus"][r["uncentred"]["primary"]]["early"],
                               "centred": {k: x["plateaus"][x["primary"]]["early"] for k, x in r["centred"]["arms"].items()}}
                           for m, r in per.items()},
           "per_model": per, "joint": j, "chance_share_of_this_label": null["label_share"].get(j["label"], 0.0),
           "planted_null": null, "beside": beside}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["early_first"]))
    print(json.dumps({m: {"u": r["uncentred"]["dates"], "c": r["centred"]["dates"],
                          "u_cons": r["uncentred"]["consistent"], "c_cons": r["centred"]["consistent"]}
                      for m, r in per.items()}))
    print(json.dumps(j))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("gate", "known", "fetch", "preflight", "run", "read"):
        sp = sub.add_parser(cmd)
        if cmd != "fetch":
            sp.add_argument("--out", required=True)
        if cmd in ("fetch", "run"):
            sp.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
        if cmd == "fetch":
            sp.add_argument("--steps", nargs="+", type=int, default=list(GRID) + [SCALE_STEP])
        if cmd == "preflight":
            sp.add_argument("--model", choices=list(MODELS), required=True)
        if cmd == "read":
            sp.add_argument("--preflight", required=True)
    a = ap.parse_args(argv)
    return {"gate": gate, "known": known, "fetch": fetch, "preflight": preflight, "run": run,
            "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
