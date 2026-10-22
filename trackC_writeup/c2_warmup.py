"""
trackC_writeup/c2_warmup.py — C2 unit 3c: the warmup arm on Stanford CRFM Mistral GPT-2 small
(`design-C2.md` Blocked 92 (ii), 98 (c); "Unit 3c: the warmup arm" there).

Five seeds of GPT-2 small (12 × 768, OpenWebText, 524 288 tokens a step) trained under a
**logarithmic** 4000-step warmup to 6e-4: the logged LR in every seed's `trainer_state.json` is
6e-4 · ln(n) / ln 4000 with n = t − K, K = 15–18 early fp16-overflow skips (DeepSpeed's log
warmup; `lit-C2-unit1.md` §5 read it as linear, corrected here). Pythia's (1430 steps, linear,
2 097 152 tokens a step) differs in its shape, its length and its tokens per step, so 410m's two
dates (98 (c): E₁'s break, and the token-centred cloud's) map to different GPT-2 steps under each
of five schedules (`MAPS`; `tools/math_checks/c2_warmup_grid.py`):

- **step**: equal Adam updates applied (t + K);
- **cumulative**: equal summed LR (a dose: how far the weights have moved);
- **instantaneous**: equal LR level (excluded on Pythia by 3b; kept so 3c is read as 3b was);
- **fraction**: the same fraction of the warmup done (K + t · 4000 / 1430);
- **tokens**: equal tokens seen (4 t).

410m's dates are its 10 seeds' (`REF`, as 3b); a GPT-2 grid date s stands for a crossing in
(prev(s), s]; a schedule predicts every grid step whose interval meets a mapped 410m interval,
for any K in `KS` (`predicted`). The reading is the set of schedules consistent with every
readable date over the 5 seeds and both counts (`joint`); `with_3b` drops instantaneous (3b
excluded it; on Pythia's sizes step, fraction and tokens are one prediction, 3b's "step").
Model, data, tokenizer and the 1/layer attention scaling all differ: description.

The centred count uses unit 3b's procedure on `alias`: arms null-matched and B-matched derived on
3 re-inits of the config (HF's GPT-2 init, as Mistral's step 0) and its off-grid step 20000, beside
1e-4 and 3e-4; each arm gated as unit 3b's (plant B at the smallest dose the uncentred count sees
at 0.4, Blocked 100 (a)). Records as unit 3b's (e1, e1c, e1r, cos per state; 13 states).

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_warmup fetch
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_warmup preflight --out <dir>
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_warmup run --out <dir> [--seeds ...]
    python -m trackC_writeup.c2_warmup read --out <dir> --preflight <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import MAX_TOKENS, REL_TOL, n_viol
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_dose as d
from trackC_writeup import c2_seeds as cs

SEEDS = {"alias": "stanford-crfm/alias-gpt2-small-x21", "battlestar": "stanford-crfm/battlestar-gpt2-small-x49",
         "caprica": "stanford-crfm/caprica-gpt2-small-x81", "darkmatter": "stanford-crfm/darkmatter-gpt2-small-x343",
         "expanse": "stanford-crfm/expanse-gpt2-small-x777"}
TOKENIZER = (SEEDS["alias"], "checkpoint-0")
#: GPT-2 small's checkpoint tags read (every one present on all five seeds; expanse lacks 30 and 300)
GRID = (0, 10, 20, 40, 60, 80, 100, 150, 200, 250, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000,
        1100, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200, 3600, 4000, 6000, 8000, 12000, 16000)
PLATEAU = tuple(s for s in GRID if 4000 <= s <= 16000)   # 410m's 1000–4000 under tokens; past every map's dates
EARLY = (0, 10)                                 # below every schedule's earliest predicted crossing (≥ 16)
PLATEAU_BESIDE = tuple(s for s in GRID if 1000 <= s <= 4000)
SCALE_STEP = 20000                              # off the grid, past the plateau; the preflight's trained step
#: the Pythia side (pythia-410m; numbers.json cfg.lr_410m, cfg.warmup_70m) and the GPT-2 side
#: (each seed's trainer_state.json; batch 512 × 1024 by its epoch count, `design-C2.md` "Unit 3c")
P_LR, P_WARMUP, P_TOKENS = 3e-4, 1430, 2_097_152
G_LR, G_WARMUP, G_TOKENS = 6e-4, 4000, 524_288
KS = (15, 16, 17, 18)                           # skipped steps by step 4000, over the 5 seeds (logged LR)
REF = d.REF                                     # 410m's 10 seeds' grid dates: E₁ {256, 512}, centred {512, 1000}
HYPS = ("instantaneous", "cumulative", "step", "fraction", "tokens")
EXCLUDED_BY_3B = ("instantaneous",)
FIXED_ARMS = cs.FIXED_ARMS                      # 1e-4, 3e-4
#: null-matched, 1e-4, 3e-4, B-matched, from the preflight (3 s.f.), frozen before any scored
#: record. None: not frozen, `run` and `read` refuse.
TAU_ARMS: Optional[Tuple[float, ...]] = (2.45e-3, 1e-4, 3e-4, 3.42e-3)
DOSES_B = d.DOSES_B
N_REINIT = c.N_REINIT
SEED = 20261025


class WarmupRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c2_warmup.py", "trackC_writeup/c2_dose.py", "trackC_writeup/c2_seeds.py",
              "trackC_writeup/c2_aniso.py", "trackA_force/a5_seeds.py", "trackA_force/a3.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- pure: the schedules

def g_cum(n: float) -> float:
    """GPT-2's summed LR after n scheduler iterations (continuous: ∫₁ⁿ p ln x / ln W dx)."""
    return 0.0 if n <= 1 else G_LR / math.log(G_WARMUP) * (n * math.log(n) - n + 1)


def p_cum(t: float) -> float:
    """410m's summed LR after t steps inside its warmup (continuous: p t² / 2W)."""
    return P_LR * t * t / (2 * P_WARMUP)


def _inv(f: Callable[[float], float], y: float, lo: float = 1.0, hi: float = 1e7) -> float:
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < y else (lo, mid)
    return hi


def to_gpt2(hyp: str, t: float, k: int) -> float:
    """The GPT-2 tag matching 410m's step t (t ≤ 1000 < both warmups) under ``hyp``, k skips."""
    if hyp == "step":
        return t + k
    if hyp == "tokens":
        return t * P_TOKENS / G_TOKENS
    if hyp == "fraction":
        return k + t * G_WARMUP / P_WARMUP
    if hyp == "instantaneous":                      # p_p t / W_p = p_g ln n / ln W_g
        return k + G_WARMUP ** (P_LR * t / P_WARMUP / G_LR)
    if hyp == "cumulative":
        return k + _inv(g_cum, p_cum(t))
    raise ValueError(hyp)


def prev(s: int) -> int:
    i = GRID.index(s)
    if i == 0:
        raise WarmupRefused("step 0 has no interval: a half-rise at 0 is unreadable")
    return GRID[i - 1]


def predicted(ref: Iterable[int], hyp: str, ks: Sequence[int] = KS) -> FrozenSet[int]:
    """The GPT-2 grid steps a schedule predicts: each 410m date g stands for a crossing in
    (prev(g), g] on Pythia's grid; mapped, it is (m(prev(g)), m(g)], and the GPT-2 grid reads s
    when (prev(s), s] meets it; the union over the skip counts ``ks``."""
    out = set()
    for k in ks:
        for g in ref:
            lo, hi = to_gpt2(hyp, d.prev(g), k), to_gpt2(hyp, g, k)
            out |= {s for s in GRID[1:] if s > lo and prev(s) < hi}
    return frozenset(out)


def consistent(dates: Iterable[int], kind: str) -> FrozenSet[str]:
    """The schedules that predict at least one of ``dates`` (one seed's date, or its arms' dates
    when they disagree)."""
    ds = set(dates)
    return frozenset(h for h in HYPS if ds & predicted(REF[kind], h))


# ---------------------------------------------------------------- pure: the reading

def series(recs: Dict[Tuple[int, str], dict], key: str, rel_tol: float = REL_TOL) -> np.ndarray:
    """V on the grid for ``key``: violations summed over the 7 prompts."""
    return np.array([sum(n_viol(recs[(s, p)][key], rel_tol) for p in V1) for s in GRID], float)


def half_rise(v: np.ndarray, window: Sequence[int] = PLATEAU) -> dict:
    """D4 (a) on GPT-2's grid: the first grid step with V ≥ half the median of V over ``window``."""
    v = np.asarray(v, float)
    p = float(np.median([v[GRID.index(s)] for s in window]))
    if p <= 0:
        return {"plateau_median": p, "step": None}
    return {"plateau_median": p, "step": GRID[int(np.argmax(v >= p / 2))]}


def sustained(v: np.ndarray, p: float, until: int = 4000) -> Optional[int]:
    """Beside: the first grid step from which V stays ≥ p / 2 at every grid step to ``until``."""
    if p <= 0:
        return None
    upto = [i for i, s in enumerate(GRID) if s <= until]
    for i in upto:
        if all(v[j] >= p / 2 for j in upto if j >= i):
            return GRID[i]
    return None


def date_reading(v: np.ndarray) -> dict:
    """One count's half-rise on its own plateau; unreadable if the plateau is 0 or an EARLY count
    is ≥ half of it (92 (i), on GPT-2's grid)."""
    v = np.asarray(v, float)
    r = half_rise(v)
    p = r["plateau_median"]
    early = {s: float(v[GRID.index(s)]) for s in EARLY}
    bad = p <= 0 or max(early.values()) >= p / 2
    return {"step": None if bad else r["step"], "plateau": p, "early": early, "unreadable": bad,
            "beside": {"sustained": sustained(v, p), "half_rise_on_1000_4000": half_rise(v, PLATEAU_BESIDE)}}


def seed_reading(vu: np.ndarray, vcs: Dict[str, np.ndarray], gate: Dict[str, bool]) -> dict:
    """One seed, as `c2_dose.size_reading`: the uncentred date; the centred dates of the arms
    that passed their gate and are readable; each date's consistent schedules."""
    u = date_reading(vu)
    arms = {k: {**date_reading(v), "gated": gate[k]} for k, v in vcs.items()}
    cdates = sorted({a["step"] for a in arms.values() if a["gated"] and not a["unreadable"]})
    out = {"uncentred": {**u, "consistent": None}, "centred": {"dates": cdates, "arms": arms,
                                                               "agree": len(cdates) == 1,
                                                               "unreadable": not cdates,
                                                               "consistent": None}}
    if not u["unreadable"]:
        out["uncentred"]["consistent"] = sorted(consistent([u["step"]], "u"))
    if cdates:
        out["centred"]["consistent"] = sorted(consistent(cdates, "c"))
    return out


def joint(per_seed: Dict[str, dict]) -> dict:
    """The schedules consistent with every readable date over the seeds and both counts; and the
    same without the schedule 3b excluded."""
    sets, unread = [], []
    for s, r in per_seed.items():
        for kind in ("uncentred", "centred"):
            if r[kind]["consistent"] is None:
                unread.append(f"{s} {kind}")
            else:
                sets.append(frozenset(r[kind]["consistent"]))
    if not sets:
        return {"survivors": [], "label": "unreadable", "readable_dates": 0, "unreadable": unread,
                "with_3b": []}
    surv = frozenset(HYPS).intersection(*sets)
    label = ("not told apart" if surv == frozenset(HYPS) else "neither" if not surv
             else "consistent with " + " and ".join(h for h in HYPS if h in surv))
    return {"survivors": [h for h in HYPS if h in surv], "label": label, "readable_dates": len(sets),
            "unreadable": unread, "with_3b": [h for h in HYPS if h in surv and h not in EXCLUDED_BY_3B]}


# ---------------------------------------------------------------- model passes

def load(seed: str, step: int, device):
    import torch
    from transformers import GPT2LMHeadModel
    from core.models import from_pretrained_eager
    m = from_pretrained_eager(GPT2LMHeadModel, SEEDS[seed], revision=f"checkpoint-{step}",
                              torch_dtype=torch.float32).to(device).eval()
    if next(m.parameters()).dtype != torch.float32:
        raise WarmupRefused(f"{seed} step {step}: weights are not float32")
    return m


def reinit(cfg, seed: int):
    """A GPT2LMHeadModel of ``cfg`` drawn by HF's GPT-2 init (N(0, 0.02); c_proj N(0, 0.02/√24);
    LayerNorm (1, 0); biases 0) under ``torch.manual_seed(seed)``, each value rounded to float16
    and held in float32 (Mistral stores float16); eval mode, eager attention."""
    import torch
    from transformers import GPT2LMHeadModel
    cfg._attn_implementation = "eager"
    torch.manual_seed(int(seed))
    m = GPT2LMHeadModel(cfg).to(torch.float32)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.half().float())
    return m.eval()


def weight_sds(model) -> Dict[str, float]:
    return {n: float(p.detach().float().std()) for n, p in model.named_parameters() if p.dim() > 1}


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])


def rec_path(out: Path, seed: str, step: int, prompt: str) -> Path:
    return out / seed / "records" / f"step{step}_{prompt}.json"


def meta(torch) -> dict:
    return {"design": "trackC_writeup/design-C2.md", "code": code_sha(), "tau_arms": TAU_ARMS,
            "seeds": SEEDS, "grid": list(GRID), "plateau": list(PLATEAU), "early": list(EARLY), "ref": REF,
            "ks": list(KS), "prompts_hash": prompts_hash(), "prompts": list(V1), "tokenizer": list(TOKENIZER),
            "max_tokens": MAX_TOKENS, "dtype": "float32", "device": torch.cuda.get_device_name(0),
            "tf32": False, "torch": torch.__version__, "hf_home": os.environ.get("HF_HOME")}


def _frozen() -> Tuple[float, ...]:
    if TAU_ARMS is None:
        raise WarmupRefused("TAU_ARMS is not set: the rule is not frozen")
    return TAU_ARMS


def fetch(a) -> int:
    """Download config.json and pytorch_model.bin at every grid tag (and the preflight's step) for
    the seeds asked, and the tokenizer; nothing is read."""
    from huggingface_hub import snapshot_download
    snapshot_download(TOKENIZER[0], revision=TOKENIZER[1],
                      allow_patterns=["config.json", "vocab.json", "merges.txt", "tokenizer_config.json",
                                      "special_tokens_map.json"])
    t0 = time.time()
    for seed in a.seeds:
        steps = GRID + ((SCALE_STEP,) if seed == "alias" else ())
        for s in steps:
            snapshot_download(SEEDS[seed], revision=f"checkpoint-{s}",
                              allow_patterns=["config.json", "pytorch_model.bin"])
        print(f"{seed}: {len(steps)} tags ({time.time() - t0:.0f} s)", flush=True)
    return 0


def run(a) -> int:
    _frozen()
    os.environ["HF_HUB_OFFLINE"] = "1"            # fetched before; nothing is downloaded here
    torch, device = c._torch_setup()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = meta(torch)
    (out / "plan.json").write_text(json.dumps({**m, "run_seeds": list(a.seeds)}, indent=1) + "\n")
    tok = tokenizer()
    first, t0 = True, time.time()
    for seed in a.seeds:
        rec_path(out, seed, 0, V1[0]).parent.mkdir(parents=True, exist_ok=True)
        for step in GRID:
            todo = [p for p in V1 if not rec_path(out, seed, step, p).exists()]
            if not todo:
                continue
            model = load(seed, step, device)
            n_states = model.config.n_layer + 1
            for p in todo:
                hs, ids = c.states(model, tok, p, device)
                rec = d.record(hs, ids, p, n_states)
                rec.update(seed=seed, repo=SEEDS[seed], step=step, revision=f"checkpoint-{step}", code=m["code"])
                rec_path(out, seed, step, p).write_text(json.dumps(rec) + "\n")
                if first:
                    print(f"first record populated: {seed} step {step} {p}, n {rec['n']}, "
                          f"states {n_states}, e1[0] {rec['e1'][0]:.6f}, e1c[0] {rec['e1c'][0]:.6f}, "
                          f"cos[-1] {rec['cos'][-1]:.4f}", flush=True)
                    first = False
            del model
            torch.cuda.empty_cache()
        print(f"{seed} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


# ---------------------------------------------------------------- reading

def load_seed(dr: Path, seed: str) -> Dict[Tuple[int, str], dict]:
    recs = {}
    for s in GRID:
        for p in V1:
            f = dr / f"step{s}_{p}.json"
            if not f.exists():
                raise WarmupRefused(f"{f}: missing")
            r = json.loads(f.read_text())
            if r.get("seed") != seed:
                raise WarmupRefused(f"{f}: seed {r.get('seed')}, not {seed}")
            recs[(s, p)] = r
    return recs


def read(a) -> int:
    out = Path(a.out)
    arms = _frozen()
    pf = json.loads((Path(a.preflight) / "preflight_plants.json").read_text())
    if [float(f"{t:.3g}") for t in pf["arms"]] != list(arms):
        raise WarmupRefused(f"the preflight's arms {pf['arms']} are not the frozen {arms}")
    gate = {f"{t:.3g}": pf["gate"]["arms"][f"{t:.3g}"]["pass"] for t in pf["arms"]}
    per, beside, codes = {}, {}, {}
    for seed in SEEDS:
        recs = load_seed(out / seed / "records", seed)
        codes[seed] = sorted({r["code"] for r in recs.values()})
        vu = series(recs, "e1")
        vcs = {f"{t:.3g}": series(recs, "e1c", t) for t in arms}
        per[seed] = seed_reading(vu, vcs, gate)
        beside[seed] = {"V_u": vu.tolist(), "V_c": {k: v.tolist() for k, v in vcs.items()},
                        "V_c_at_1e-3": series(recs, "e1c", 1e-3).tolist(),
                        "V_raw_centred": {k: series(recs, "e1r", t).tolist() for k, t in zip(vcs, arms)},
                        "mean_cos_layer_mean": [float(np.mean([recs[(st, p)]["cos"] for p in V1])) for st in GRID]}
    if len({tuple(v) for v in codes.values()}) != 1:
        raise WarmupRefused(f"records from more than one code version: {codes}")
    res = {"code": code_sha(), "record_codes": codes, "tau_arms": TAU_ARMS, "ref": REF, "grid": list(GRID),
           "predicted": {kind: {h: sorted(predicted(REF[kind[0]], h)) for h in HYPS}
                         for kind in ("uncentred", "centred")},
           "early_first": {s: {"uncentred": r["uncentred"]["early"],
                               "centred": {k: x["early"] for k, x in r["centred"]["arms"].items()}}
                           for s, r in per.items()},
           "per_seed": per, "joint": joint(per), "beside": beside}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["early_first"]))
    print(json.dumps({s: {"u": r["uncentred"]["step"], "c": r["centred"]["dates"],
                          "u_cons": r["uncentred"]["consistent"], "c_cons": r["centred"]["consistent"]}
                      for s, r in per.items()}))
    print(json.dumps(res["joint"]))
    return 0


# ---------------------------------------------------------------- the preflight (before the freeze)

def preflight(a) -> int:
    """As unit 3b's, on `alias`: 3 re-inits of the config and its off-grid step 20000; plants A and B
    through both counts; the null-matched and B-matched arms derived; each arm gated. The re-inits'
    weight SDs are set beside step 0's (the init they stand for). Writes the design-preflight tool's
    inputs for both counts."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoConfig
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    cfg = AutoConfig.from_pretrained(SEEDS["alias"], revision=f"checkpoint-{SCALE_STEP}")
    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(cfg.n_embd)
    u /= np.linalg.norm(u)
    step0 = weight_sds(load("alias", 0, "cpu"))
    pops: Dict[str, dict] = {}
    names = [f"reinit:{s}" for s in range(N_REINIT)] + [f"step{SCALE_STEP}"]
    init_check = {}
    for name in names:
        if name.startswith("reinit"):
            model = reinit(cfg, int(name.split(":")[1])).to(device)
            sds = weight_sds(model)
            r = {k: sds[k] / step0[k] for k in step0}
            init_check[name] = {"min_ratio": min(r.values()), "max_ratio": max(r.values()),
                                "worst": max(r, key=lambda k: abs(math.log(r[k])))}
        else:
            model = load("alias", SCALE_STEP, device)
        per = {}
        for p in V1:
            hs, _ = c.states(model, tok, p, device)
            per[p] = {"base": cs._drops(hs), "plants": d._plants(hs, u, rng)}
        pops[name] = {"per_prompt": per}
        del model
        torch.cuda.empty_cache()
        print(f"{name} done", flush=True)

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
    derived = {"b_dose": bd,
               "null_matched": cs.null_matched_tol(nu["base"]["u"], nu["base"]["c"]),
               "b_matched": cs.b_matched_tol(tr["base"]["c"], tr["plants"]["B_c"][bd], b_target)}
    again = (float(f"{derived['null_matched']:.3g}"), *FIXED_ARMS, float(f"{derived['b_matched']:.3g}"))
    if TAU_ARMS and tuple(TAU_ARMS) != again:
        raise WarmupRefused(f"the preflight derives {again}, not the frozen {TAU_ARMS}")
    arms = TAU_ARMS or again
    gate = d.gate_arms(units, arms, bd)
    gate_u = {"null_rate": cs.rate(nu["base"]["u"], REL_TOL), "base_rate": cs.rate(tr["base"]["u"], REL_TOL),
              "B_share": b_target, "b_dose": bd,
              "A_share": cs.share(cs.rate(tr["plants"]["A_u"][cs.A_DOSE], REL_TOL), cs.rate(tr["base"]["u"], REL_TOL))}
    for w, key in (("c", "B_c"), ("u", "B_u")):
        tool_in = {"statistic": f"{'centred' if w == 'c' else 'uncentred'} E_1 relative drop per (prompt, "
                                f"transition), (E_t − E_t+1) / |E_t|, Mistral GPT-2 small alias (c2_warmup)",
                   "direction": "greater", "nulls": {"reinit": nu["base"][w]}, "moved_marks": [],
                   "trained": tr["base"][w], "trained_step": str(SCALE_STEP),
                   "scored_steps": [str(s) for s in GRID],
                   "planted": {f"{x:g}": nu["plants"][key][f"{x:g}"] for x in DOSES_B},
                   "bounds": None, "decision_p": 0.05}
        (out / f"preflight_input_{w}.json").write_text(json.dumps(tool_in) + "\n")
    rec = {"code": code_sha(), "meta": meta(torch), "seed": "alias", "repo": SEEDS["alias"], "derived": derived,
           "b_target": b_target, "arms": list(arms), "frozen": TAU_ARMS is not None, "gate": gate,
           "uncentred_at_1e-3": gate_u, "units_per_population": {p: len(units[p]["base"]["c"]) for p in units},
           "scales": {pop: {"s_u": cs.scale(units[pop]["base"]["u"]), "s_c": cs.scale(units[pop]["base"]["c"])}
                      for pop in units},
           "doses_A": list(cs.DOSES_A), "doses_B": list(DOSES_B), "u_seed": SEED, "init_check": init_check}
    (out / "preflight_plants.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"derived": derived, "arms": list(arms), "gate": gate, "uncentred": gate_u,
                      "init_check": init_check}, indent=1))
    return 0 if gate["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("fetch", "preflight", "run", "read"):
        sp = sub.add_parser(cmd)
        if cmd != "fetch":
            sp.add_argument("--out", required=True)
        if cmd in ("fetch", "run"):
            sp.add_argument("--seeds", nargs="+", choices=list(SEEDS), default=list(SEEDS))
        if cmd == "read":
            sp.add_argument("--preflight", required=True)
    a = ap.parse_args(argv)
    return {"fetch": fetch, "preflight": preflight, "run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
