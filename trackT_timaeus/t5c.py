"""
trackT_timaeus/t5c.py — T5 U2: the cloud's trajectories through training (`design-T5.md` "U2" and
"U2's details", fixed 2026-10-30 before any U2 output; Blocked 125 (a)).

Every checkpoint is read as U1 reads a model (`t5.state_stats`: the 7 v1 prompts, its own
tokenizer, ≤ 512 tokens, float32, eager, GPU), keeping per prompt ρ_ℓ and ρ_ℓ without position 0
so the reading can bootstrap the prompts. Trajectories: `dodecahedron-32m`, `-67m` at their 53
steps; `pythia-70m`, `-160m`, `-410m` at C2's 18; `timaeus/gpt-2-small` at its init and 19
log-spaced checkpoints.

"Spreads" for one trajectory: Δ = ρ_mid(first) − min_t ρ_mid(t), with a 7-prompt bootstrap (one
resample used at every step, the minimum retaken in each). Spreads if Δ ≥ 0.15 and the 5th
percentile > 0.05; does not spread if the 95th < 0.15; else not told apart (checked in that order).

U5 (`design-T5.md` "U5's details"; Blocked 135 (a)): the same statistics on `pythia-410m` seeds 1–9 at
the 18 steps and on Mistral GPT-2 small (`alias-gpt2-small-x21`, `battlestar-gpt2-small-x49`) from
checkpoint 0, described with no rule: `u5` gates (seed 0 against U2's records, seeds against U4a's floor
records, the Mistral prompt loss), then writes the series, the shapes and the ten-seed table.

Tier 1, description (nothing here is registered).
    python -m trackT_timaeus.t5c fetch [--hf-root <HDD_1TB>/mets_data/hf]
    METS_DATA=<main>/data python -m trackT_timaeus.t5c run --out <dir>
    METS_DATA=<main>/data python -m trackT_timaeus.t5c read --out <dir>
    METS_DATA=<main>/data scripts/t5u5.sh                      # U5: run, then `t5c u5 --out <dir>`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, STEPS, V1, prompts_hash
from trackC_writeup import c2_aniso as c
from trackT_timaeus import t2, t5

PROMPTS_HASH = t5.PROMPTS_HASH
GPT2 = "timaeus/gpt-2-small"
GPT2_STEPS = (0, 100, 200, 300, 400, 500, 800, 1100, 1500, 2100, 3000, 4300, 6000, 8400, 11800, 16600,
              23400, 32900, 46200, 65000)
DODECA_STEPS = tuple(t2.GRID) + (t2.SCALE_STEP,)
#: name → (repo, steps, tokenizer); the first step is the trajectory's first checkpoint
TRAJ = {
    "dodecahedron-32m": (t2.MODELS["32m"], DODECA_STEPS, "32m"),
    "dodecahedron-67m": (t2.MODELS["67m"], DODECA_STEPS, "67m"),
    "pythia-70m": ("EleutherAI/pythia-70m", STEPS, "neox"),
    "pythia-160m": ("EleutherAI/pythia-160m", STEPS, "neox"),
    "pythia-410m": ("EleutherAI/pythia-410m", STEPS, "neox"),
    "gpt-2-small": (GPT2, GPT2_STEPS, "neox"),
}
SCORED = ("dodecahedron-32m", "dodecahedron-67m", "gpt-2-small")
#: the GPT-NeoX fields a `gpt-2-small` checkpoint must have (Pythia-160m's shape, T1 Q4, U0)
GPT2_SHAPE = {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "rotary_pct": 0.25,
              "use_parallel_residual": True, "vocab_size": 50304, "tie_word_embeddings": False}
# the rule (design-T5.md "U2"; fixed before output)
SPREAD_MIN = 0.15
BOOT_P5_MIN = 0.05
N_BOOT = 1000
BOOT_SEED = 20261030
H1_INIT_MIN = 0.75
# the gate (design-T5.md "Calibration gate", U2, and "U2's details")
KNOWN_TOL = 1e-6
KNOWN = {"pythia-410m": ("410m", (0, 4000)), "dodecahedron-32m": ("32m", (1, 9536))}
SYN_MU, SYN_PLANT, SYN_BOTTOM = 0.90, 0.15, 1072
SYN_DRAWS = 1000
SYN_PLANTED_MIN, SYN_FLAT_MAX = 0.90, 0.05
SPREAD_STEP = 9536                  # the step whose 7-prompt spread sets the synthetic's s
DEFAULT_ROOTS = t5.DEFAULT_ROOTS
HDD_ROOT = DEFAULT_ROOTS[1]

# ---------------------------------------------------------------- U5, the controls (design-T5.md "U5's details";
# Blocked 135 (a), the user's, 2026-11-06; fixed before any U5 output). Described, with no rule.
#: Mistral GPT-2 small: every checkpoint branch on disk (`HDD_1TB`), from checkpoint 0
MISTRAL_STEPS = (0, 10, 20, 40, 60, 80, 100, 150, 200, 250, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000,
                 1100, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200, 3600, 4000, 6000, 8000, 12000, 16000)
U5_SEEDS = tuple(range(1, 10))
U5_TRAJ = {
    **{f"pythia-410m-seed{k}": (f"EleutherAI/pythia-410m-seed{k}", STEPS, "neox") for k in U5_SEEDS},
    "mistral-alias": ("stanford-crfm/alias-gpt2-small-x21", MISTRAL_STEPS + (20000,), "mistral"),
    "mistral-battlestar": ("stanford-crfm/battlestar-gpt2-small-x49", MISTRAL_STEPS, "mistral"),
}
ALL_TRAJ = {**TRAJ, **U5_TRAJ}
MISTRAL = ("mistral-alias", "mistral-battlestar")
#: the GPT-2 BPE both runs train on, from alias's checkpoint 0 (battlestar's branches hold no tokenizer files)
MISTRAL_TOKENIZER = ("stanford-crfm/alias-gpt2-small-x21", "checkpoint-0")
U5_DIR = "trackT/t5u5/2026-11-06"
#: the gate: seed 0 re-read against U2's records (all 18 steps), seeds 1–9 against U4a's floor records
U2_DIR = "trackT/t5u2/2026-10-30"
FLOOR_DIR = Path("/run/media/system/HDD_1TB/mets_data/trackT/t5u4/floor")
FLOOR_STEPS = (4000, 143000)
#: the Mistral load check: checkpoint 0's prompt loss within LOSS_INIT_TOL nats of ln(vocab); the last
#: checkpoint's at least LOSS_DROP_MIN nats below checkpoint 0's (a mis-loaded model does not learn)
LOSS_INIT_TOL = 1.0
LOSS_DROP_MIN = 3.0
#: the shape's early-peak window, (0, PEAK_MAX_STEP] (U2's peaks: 16–47 on Pythia and Dodecahedron, 100 on gpt-2-small)
PEAK_MAX_STEP = 1000


class T5cRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t5c.py", "trackT_timaeus/t5.py", "trackT_timaeus/t2.py", "trackC_writeup/c2_aniso.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the rule (pure)

def mid_series(P: np.ndarray) -> np.ndarray:
    """ρ_mid per step from per-prompt ρ_ℓ, ``P`` (steps, prompts, states): the prompts' mean, then
    the median over the middle states 1 … L−1. Extra leading axes are kept."""
    return np.median(P.mean(-2)[..., 1:-1], -1)


def spread(P: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """Δ = first − min of ρ_mid, and its prompt bootstrap (one resample at every step)."""
    P = np.asarray(P, np.float64)
    s = mid_series(P)
    delta = float(s[0] - s.min())
    n = P.shape[1]
    idx = rng.integers(0, n, (n_boot, n))
    B = mid_series(P[:, idx, :])                         # (steps, n_boot)
    db = B[0] - B.min(0)
    p5, p95 = (float(x) for x in np.percentile(db, [5, 95]))
    return {"delta": delta, "p5": p5, "p95": p95, "argmin": int(np.argmin(s)), "first": float(s[0]),
            "min": float(s.min()), "label": label(delta, p5, p95)}


def label(delta: float, p5: float, p95: float) -> str:
    if delta >= SPREAD_MIN and p5 > BOOT_P5_MIN:
        return "spreads"
    if p95 < SPREAD_MIN:
        return "does not spread"
    return "not told apart"


def scalar_as_states(X: np.ndarray) -> np.ndarray:
    """A synthetic per-prompt ρ_mid series (steps, prompts) as (steps, prompts, 3) states whose one
    middle state is the value, so the synthetic goes through `spread` unchanged."""
    Z = np.zeros(X.shape + (3,))
    Z[..., 1] = X
    return Z


def planted_mu(steps: Sequence[int], plant: float) -> np.ndarray:
    t = np.asarray(steps, np.float64)
    return SYN_MU - plant * np.minimum(1.0, np.log(t) / math.log(SYN_BOTTOM))


def synthetic(s: float, plant: float, rng: np.random.Generator, draws: int = SYN_DRAWS,
              steps: Sequence[int] = DODECA_STEPS, n_prompts: int = len(V1)) -> dict:
    """The rule on synthetic trajectories: per prompt and step μ(t) + N(0, s²), i.i.d."""
    mu = planted_mu(steps, plant) if plant else np.full(len(steps), SYN_MU)
    labels = []
    for _ in range(draws):
        X = mu[:, None] + rng.normal(0.0, s, (len(steps), n_prompts))
        labels.append(spread(scalar_as_states(X), rng)["label"])
    return {"s": s, "plant": plant, "draws": draws,
            **{k: labels.count(k) / draws for k in ("spreads", "does not spread", "not told apart")}}


# ---------------------------------------------------------------- models

def load_gpt2(step: int, roots: Sequence[str], device):
    """`gpt-2-small` at ``step``: the init repo at 0 (as U1), else `checkpoint-<step>` with its own
    config, checked against Pythia-160m's shape, its stored weights upcast to float32."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    if step == 0:
        repo, rev = t5.REAL["gpt-2-small-init"][:2]
        m = t5.load_real(repo, rev, roots, device)
        return m, {"commit": t2.commit_of(m), "stored_dtypes": ["torch.float32"]}
    cache = t5.locate(GPT2, "main", roots)
    sub = f"checkpoint-{step}"
    cfg_path = hf_hub_download(GPT2, f"{sub}/config.json", cache_dir=cache)
    cfg = GPTNeoXConfig.from_json_file(cfg_path)
    bad = {k: getattr(cfg, k, None) for k, v in GPT2_SHAPE.items() if getattr(cfg, k, None) != v}
    if bad:
        raise T5cRefused(f"{GPT2} {sub}: config differs from Pythia-160m's shape: {bad}")
    cfg._attn_implementation = "eager"
    cfg.use_cache = False
    sd = load_file(hf_hub_download(GPT2, f"{sub}/model.safetensors", cache_dir=cache))
    m = GPTNeoXForCausalLM(cfg)
    missing, unexpected = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    buffers = (".attention.bias", ".attention.masked_bias", "inv_freq")
    missing = [k for k in missing if not k.endswith(buffers)]
    unexpected = [k for k in unexpected if not k.endswith(buffers)]
    if missing or unexpected:
        raise T5cRefused(f"{GPT2} {sub}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return m.to(torch.float32).to(device).eval(), {"commit": Path(cfg_path).parents[1].name,
                                                    "stored_dtypes": sorted({str(v.dtype) for v in sd.values()})}


def load(traj: str, step: int, roots: Sequence[str], device):
    repo = ALL_TRAJ[traj][0]
    if traj == "gpt-2-small":
        return load_gpt2(step, roots, device)
    m = t5.load_real(repo, revision_of(traj, step), roots, device)
    return m, {"commit": t2.commit_of(m)}


def revision_of(traj: str, step: int) -> str:
    return f"checkpoint-{step}" if traj in MISTRAL else f"step{step}"


def tokenizer_of(tk: str, roots: Sequence[str]):
    """'neox' → Pythia's; 'mistral' → the GPT-2 BPE from alias's checkpoint 0 (no BOS added); else Dodecahedron's."""
    if tk == "neox":
        return t2.p_tokenizer()
    if tk == "mistral":
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        repo, rev = MISTRAL_TOKENIZER
        vocab = hf_hub_download(repo, "vocab.json", revision=rev, cache_dir=t5.locate(repo, rev, roots))
        return AutoTokenizer.from_pretrained(str(Path(vocab).parent))
    return t2.tokenizer(tk)


def prompt_loss(model, tok, device) -> List[float]:
    """Mean next-token cross-entropy (nats) per v1 prompt, ≤ 512 tokens, as `c.states` tokenizes: the
    Mistral load check, beside elsewhere."""
    import torch
    from core.config import PROMPTS
    out = []
    for p in V1:
        enc = tok(PROMPTS[p], return_tensors="pt", truncation=True, max_length=c.MAX_TOKENS)
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out.append(float(model(input_ids=ids, labels=ids).loss))
    return out


def read_model(model, tok, device) -> dict:
    """U1's per-state statistics, averaged over the prompts, with ρ_ℓ and ρ_ℓ⁻⁰ kept per prompt."""
    per, ntok = [], []
    for p in V1:
        hs, ids = c.states(model, tok, p, device)
        per.append([t5.state_stats(h) for h in hs])
        ntok.append(len(ids))
    keys = ("rho", "rho_no0", "m", "m_no0", "n0")
    rec = {k: np.mean([[s[k] for s in row] for row in per], 0).tolist() for k in keys}
    rec["rho_per_prompt"] = [[s["rho"] for s in row] for row in per]
    rec["rho_no0_per_prompt"] = [[s["rho_no0"] for s in row] for row in per]
    rec["n_tokens"] = ntok
    rec["n_states"] = len(per[0])
    if rec["n_states"] != model.config.num_hidden_layers + 1 or not all(np.all(np.isfinite(rec[k])) for k in keys):
        raise T5cRefused(f"record not populated: {rec['n_states']} states")
    rec["rho_mid"], rec["rho_15"] = t5.rho_mid(rec["rho"]), t5.rho_15(rec["rho"])
    rec["rho_mid_no0"] = t5.rho_mid(rec["rho_no0"])
    return rec


# ---------------------------------------------------------------- commands

def fetch(a) -> int:
    """Download what U2 reads and is not on disk; nothing is read."""
    from huggingface_hub import hf_hub_download, snapshot_download
    cache = str(Path(a.hf_root) / "hub")
    t0 = time.time()
    for name in a.trajs:
        repo, steps, _ = TRAJ[name]
        for s in steps:
            if name == "gpt-2-small":
                if s == 0:
                    continue                      # the init repo, on disk since U0
                for f in ("config.json", "model.safetensors"):
                    hf_hub_download(repo, f"checkpoint-{s}/{f}", cache_dir=cache)
            else:
                snapshot_download(repo, revision=f"step{s}", cache_dir=cache,
                                  allow_patterns=["config.json", "model.safetensors", "*.bin", "tokenizer.json"]
                                  if name.startswith("dodeca") else ["config.json", "*.safetensors", "*.bin"])
        print(json.dumps({"fetched": name, "steps": len(steps), "t": round(time.time() - t0)}), flush=True)
    return 0


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(path)


def run(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if prompts_hash() != PROMPTS_HASH:
        raise T5cRefused(f"prompts_hash {prompts_hash()} != {PROMPTS_HASH}")
    torch, device = c._torch_setup()
    import transformers
    roots = a.hf_roots
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u5 = a.unit == "U5"
    meta = {"design": "trackT_timaeus/design-T5.md", "unit": a.unit, "code": code_sha(), "prompts_hash": prompts_hash(),
            "prompts": list(V1), "max_tokens": t2.MAX_TOKENS, "dtype": "float32", "tf32": False,
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "hf_roots": list(roots),
            "trajectories": {k: {"repo": v[0], "steps": list(v[1]), "tokenizer": v[2]} for k, v in ALL_TRAJ.items()
                             if k in a.trajs}}
    if (out / "meta.json").exists():                      # a later call into the same dir adds its trajectories
        prev = _load(out / "meta.json")
        if (prev["code"], prev["unit"]) != (meta["code"], meta["unit"]):
            raise T5cRefused(f"{out}: meta.json is from code {prev['code']} {prev['unit']}, not {meta['code']} {a.unit}")
        meta["trajectories"] = {**prev["trajectories"], **meta["trajectories"]}
    _write(out / "meta.json", meta)
    toks = {}
    t0 = time.time()
    for name in a.trajs:
        repo, steps, tk = ALL_TRAJ[name]
        if tk not in toks:
            toks[tk] = tokenizer_of(tk, roots)
        for s in (a.steps or steps):
            if s not in steps:
                raise T5cRefused(f"{name}: step {s} is not on its frozen list")
            p = out / name / f"step{s}.json"
            if p.exists() and not a.force:
                continue
            m, info = load(name, s, roots, device)
            rec = read_model(m, toks[tk], device)
            if u5:
                rec["loss_per_prompt"] = prompt_loss(m, toks[tk], device)
                rec["loss"] = float(np.mean(rec["loss_per_prompt"]))
            rec.update({"trajectory": name, "model": repo, "step": s, **info, "code": meta["code"],
                        "wall_s": round(time.time() - t0, 1)})
            _write(p, rec)
            print(json.dumps({"wrote": f"{name}/step{s}", "rho_mid": round(rec["rho_mid"], 4), "t": rec["wall_s"]}),
                  flush=True)
            del m
            torch.cuda.empty_cache()
    print(json.dumps({"done": True, "wall_s": round(time.time() - t0, 1)}))
    return 0


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def collect(out: Path) -> dict:
    recs = {}
    for name, (_, steps, _) in TRAJ.items():
        missing = [s for s in steps if not (out / name / f"step{s}.json").exists()]
        if missing:
            raise T5cRefused(f"{name}: {len(missing)} of {len(steps)} steps missing, e.g. {missing[:3]}")
        recs[name] = [_load(out / name / f"step{s}.json") for s in steps]
    codes = {r["code"] for v in recs.values() for r in v}
    if len(codes) != 1:
        raise T5cRefused(f"records from more than one code version: {codes}")
    return {"recs": recs, "code": codes.pop()}


def per_prompt(rs: List[dict], key: str = "rho_per_prompt") -> np.ndarray:
    return np.asarray([r[key] for r in rs], np.float64)          # (steps, prompts, states)


def prompt_spread(rec: dict) -> float:
    return float(np.std(np.median(np.asarray(rec["rho_per_prompt"])[:, 1:-1], 1), ddof=1))


def gate(recs: Dict[str, List[dict]], known_dir: Path) -> dict:
    g = {}
    diffs = {}
    for name, (mk, steps) in KNOWN.items():
        ref = _load(known_dir / f"diag_{mk}.json")["populations"]
        for s in steps:
            got = recs[name][TRAJ[name][1].index(s)]["rho"]
            diffs[f"{mk}@step{s}"] = float(np.max(np.abs(np.asarray(got) - np.asarray(ref[f"step{s}"]["mean_cos"]))))
    g["known_answer"] = {"max_diff": diffs, "pass": max(diffs.values()) <= KNOWN_TOL}
    s = prompt_spread(recs["dodecahedron-32m"][DODECA_STEPS.index(SPREAD_STEP)])
    rng = np.random.default_rng(BOOT_SEED)
    planted = synthetic(s, SYN_PLANT, rng)
    flat = synthetic(s, 0.0, rng)
    g["rule"] = {"s": s, "planted": planted, "flat": flat,
                 "pass": planted["spreads"] >= SYN_PLANTED_MIN and flat["spreads"] <= SYN_FLAT_MAX}
    s1 = prompt_spread(recs["dodecahedron-32m"][0])
    g["rule_beside"] = {"s_step1": s1, "planted_at_s_step1": synthetic(s1, SYN_PLANT, rng),
                        "flat_at_s_step1": synthetic(s1, 0.0, rng),
                        "planted_0.10": synthetic(s, 0.10, rng), "planted_0.20": synthetic(s, 0.20, rng)}
    g["pass"] = g["known_answer"]["pass"] and g["rule"]["pass"]
    return g


def reading(recs: Dict[str, List[dict]]) -> dict:
    res = {}
    for name, rs in recs.items():
        steps = TRAJ[name][1]
        rng = np.random.default_rng(BOOT_SEED)
        sp = spread(per_prompt(rs), rng)
        sp0 = spread(per_prompt(rs, "rho_no0_per_prompt"), rng)
        sp["argmin_step"], sp0["argmin_step"] = steps[sp["argmin"]], steps[sp0["argmin"]]
        res[name] = {"scored": name in SCORED, "rho_mid": sp, "rho_mid_no0_beside": sp0,
                     "position0_disagrees": sp["label"] != sp0["label"],
                     "series": {"step": list(steps), "rho_mid": [r["rho_mid"] for r in rs],
                                "rho_mid_no0": [r["rho_mid_no0"] for r in rs], "rho_15": [r["rho_15"] for r in rs],
                                "m_mid": [float(np.median(r["m"][1:-1])) for r in rs],
                                "n0_mid": [float(np.median(r["n0"][1:-1])) for r in rs]}}
    init = res["gpt-2-small"]["series"]["rho_mid"][0]
    lab = res["gpt-2-small"]["rho_mid"]["label"]
    if init < H1_INIT_MIN:
        h1 = f"neither branch applies: init rho_mid {init:.3f} < {H1_INIT_MIN}"
    elif lab == "spreads":
        h1 = "a flat init does not keep the cloud concentrated in a NeoX with biases"
    elif lab == "does not spread":
        h1 = "H1 consistent through training"
    else:
        h1 = "not told apart"
    res["H1_through_training"] = h1
    return res


def read(a) -> int:
    out = Path(a.out)
    col = collect(out)
    g = gate(col["recs"], Path(a.known_dir))
    res = {"code": col["code"], "reader": code_sha(), "gate": g}
    if g["pass"]:
        res["reading"] = reading(col["recs"])
    else:
        res["refused"] = "the gate failed: no reading (design-T5.md 'Calibration gate'; back to the user)"
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    show = {"gate": {"known_answer": g["known_answer"]["max_diff"], "s": g["rule"]["s"],
                     "planted_spreads": g["rule"]["planted"]["spreads"], "flat_spreads": g["rule"]["flat"]["spreads"],
                     "pass": g["pass"]}}
    if g["pass"]:
        show["labels"] = {k: {kk: (round(v[kk]["delta"], 4), round(v[kk]["p5"], 4), round(v[kk]["p95"], 4),
                                   v[kk]["label"], v[kk]["argmin_step"]) for kk in ("rho_mid", "rho_mid_no0_beside")}
                          for k, v in res["reading"].items() if isinstance(v, dict)}
        show["H1_through_training"] = res["reading"]["H1_through_training"]
    print(json.dumps(show, indent=1))
    return 0 if g["pass"] else 1


# ---------------------------------------------------------------- U5's reader (described; no rule, no label)

MISTRAL_VOCAB = 50257
GATE_KEYS = ("rho", "rho_no0", "m", "m_no0", "n0")


def u5_names() -> List[str]:
    return ["pythia-410m", *(f"pythia-410m-seed{k}" for k in U5_SEEDS), *MISTRAL]


def collect_u5(out: Path) -> dict:
    recs = {}
    for name in u5_names():
        steps = ALL_TRAJ[name][1]
        missing = [s for s in steps if not (out / name / f"step{s}.json").exists()]
        if missing:
            raise T5cRefused(f"{name}: {len(missing)} of {len(steps)} steps missing, e.g. {missing[:3]}")
        recs[name] = [_load(out / name / f"step{s}.json") for s in steps]
    codes = {r["code"] for v in recs.values() for r in v}
    if len(codes) != 1:
        raise T5cRefused(f"records from more than one code version: {codes}")
    return {"recs": recs, "code": codes.pop()}


def rec_diff(got: dict, ref: dict, keys: Sequence[str] = GATE_KEYS) -> float:
    """max over the per-state arrays of |got − ref| / max(1, |ref|): absolute below 1 (ρ, m), relative above (n0)."""
    d = 0.0
    for k in keys:
        a, b = np.asarray(got[k], np.float64), np.asarray(ref[k], np.float64)
        if a.shape != b.shape:
            return math.inf
        d = max(d, float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))))
    return d


def u5_gate(recs: Dict[str, List[dict]], u2_dir: Path, floor_dir: Path) -> dict:
    """design-T5.md "U5's details": seed 0 against U2's records at the 18 steps; seeds 1–9 against U4a's
    floor records at 4000 and 143000 (≤ KNOWN_TOL); the Mistral load check on the prompt loss."""
    g = {}
    d0 = {str(s): rec_diff(r, _load(u2_dir / "pythia-410m" / f"step{s}.json"))
          for s, r in zip(STEPS, recs["pythia-410m"])}
    g["seed0_vs_u2"] = {"max_diff": d0, "pass": max(d0.values()) <= KNOWN_TOL}
    df = {}
    for k in U5_SEEDS:
        rs = recs[f"pythia-410m-seed{k}"]
        for s in FLOOR_STEPS:
            df[f"seed{k}@{s}"] = rec_diff(rs[STEPS.index(s)], _load(floor_dir / f"seed{k}_step{s}.json"))
    g["seeds_vs_floor"] = {"max_diff": df, "pass": max(df.values()) <= KNOWN_TOL}
    ln_v = math.log(MISTRAL_VOCAB)
    lm = {}
    for name in MISTRAL:
        first, last = recs[name][0]["loss"], recs[name][-1]["loss"]
        lm[name] = {"loss_first": first, "loss_last": last, "ln_vocab": ln_v,
                    "pass": abs(first - ln_v) <= LOSS_INIT_TOL and last <= first - LOSS_DROP_MIN}
    g["mistral_load"] = {**lm, "pass": all(v["pass"] for v in lm.values())}
    g["pass"] = g["seed0_vs_u2"]["pass"] and g["seeds_vs_floor"]["pass"] and g["mistral_load"]["pass"]
    return g


def series_of(rs: List[dict]) -> Dict[str, list]:
    out = {"step": [r["step"] for r in rs], "rho_mid": [r["rho_mid"] for r in rs],
           "rho_mid_no0": [r["rho_mid_no0"] for r in rs], "rho_15": [r["rho_15"] for r in rs],
           "m_mid": [float(np.median(r["m"][1:-1])) for r in rs],
           "n0_mid": [float(np.median(r["n0"][1:-1])) for r in rs]}
    if all("loss" in r for r in rs):
        out["loss"] = [r["loss"] for r in rs]
    return out


def shape(steps: Sequence[int], series: Sequence[float]) -> dict:
    """U2's shape, fixed: the early peak is the maximum over steps in (0, PEAK_MAX_STEP]; the trough the
    minimum at or after it; the end the last step. Ties take the first step."""
    t, s = np.asarray(steps), np.asarray(series, np.float64)
    win = np.flatnonzero((t > 0) & (t <= PEAK_MAX_STEP))
    if not len(win):
        raise T5cRefused("no step in the early-peak window")
    ip = int(win[np.argmax(s[win])])
    it = ip + int(np.argmin(s[ip:]))
    return {"first": float(s[0]), "first_step": int(t[0]), "peak": float(s[ip]), "peak_step": int(t[ip]),
            "trough": float(s[it]), "trough_step": int(t[it]), "end": float(s[-1]), "end_step": int(t[-1]),
            "fall": float(s[ip] - s[it]), "late_rise": float(s[-1] - s[it])}


def seeds_table(series: Dict[str, Dict[str, list]], key: str = "rho_mid") -> dict:
    """Per step over the ten 410m seeds (0 = `pythia-410m`): mean, SD (ddof 1), min, max, the seeds at
    the ends, and seed 0's rank from the low end."""
    names = ["pythia-410m", *(f"pythia-410m-seed{k}" for k in U5_SEEDS)]
    X = np.asarray([series[n][key] for n in names], np.float64)          # (10, 18)
    return {"key": key, "step": list(STEPS), "mean": X.mean(0).tolist(), "sd": X.std(0, ddof=1).tolist(),
            "min": X.min(0).tolist(), "max": X.max(0).tolist(),
            "argmin_seed": X.argmin(0).tolist(), "argmax_seed": X.argmax(0).tolist(),
            "seed0_rank_from_low": [int((X[:, j] < X[0, j]).sum()) for j in range(X.shape[1])]}


def describe_u5(recs: Dict[str, List[dict]]) -> dict:
    series = {n: series_of(rs) for n, rs in recs.items()}
    shapes = {n: {k: shape(v["step"], v[k]) for k in ("rho_mid", "rho_mid_no0", "rho_15")} for n, v in series.items()}
    seeds = [n for n in u5_names() if n.startswith("pythia-410m")]
    feats = ("first", "peak", "peak_step", "trough", "trough_step", "end", "fall", "late_rise")
    over_seeds = {f: {"min": min(shapes[n]["rho_mid"][f] for n in seeds),
                      "max": max(shapes[n]["rho_mid"][f] for n in seeds),
                      "per_seed": [shapes[n]["rho_mid"][f] for n in seeds]} for f in feats}
    a, b = (dict(zip(series[n]["step"], series[n]["rho_mid"])) for n in MISTRAL)
    common = sorted(set(a) & set(b))
    diff = [abs(a[s] - b[s]) for s in common]
    return {"series": series, "shapes": shapes,
            "pythia_410m_seeds": {"rho_mid": seeds_table(series), "rho_mid_no0": seeds_table(series, "rho_mid_no0"),
                                  "n0_mid": seeds_table(series, "n0_mid"), "shape_over_seeds": over_seeds},
            "mistral_pair": {"common_steps": common, "abs_diff_rho_mid": diff, "max_abs_diff": max(diff),
                             "at_step": common[int(np.argmax(diff))]}}


def u5(a) -> int:
    out = Path(a.out)
    if a.seed0_only:                    # the known answer first (scripts/t5u5.sh), before the other 92 reads
        d0 = {str(s): rec_diff(_load(out / "pythia-410m" / f"step{s}.json"),
                               _load(Path(a.u2_dir) / "pythia-410m" / f"step{s}.json")) for s in STEPS}
        ok = max(d0.values()) <= KNOWN_TOL
        print(json.dumps({"seed0_vs_u2_max_diff": max(d0.values()), "pass": ok}))
        return 0 if ok else 1
    col = collect_u5(out)
    g = u5_gate(col["recs"], Path(a.u2_dir), Path(a.floor_dir))
    res = {"code": col["code"], "reader": code_sha(), "unit": "U5", "gate": g}
    if g["pass"]:
        res["description"] = describe_u5(col["recs"])
    else:
        res["refused"] = "the gate failed: no description (design-T5.md 'U5's details'; back to the user)"
    (out / "u5.json").write_text(json.dumps(res, indent=1) + "\n")
    show = {"gate": {"seed0_vs_u2": max(g["seed0_vs_u2"]["max_diff"].values()),
                     "seeds_vs_floor": max(g["seeds_vs_floor"]["max_diff"].values()),
                     "mistral_load": {k: (round(v["loss_first"], 3), round(v["loss_last"], 3))
                                      for k, v in g["mistral_load"].items() if isinstance(v, dict)},
                     "pass": g["pass"]}}
    if g["pass"]:
        d = res["description"]
        show["shapes_rho_mid"] = {n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s["rho_mid"].items()}
                                  for n, s in d["shapes"].items()}
        show["mistral_pair_max_abs_diff"] = (round(d["mistral_pair"]["max_abs_diff"], 4), d["mistral_pair"]["at_step"])
    print(json.dumps(show, indent=1))
    return 0 if g["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("fetch")
    sp.add_argument("--trajs", nargs="+", default=["dodecahedron-32m", "dodecahedron-67m", "gpt-2-small"],
                    choices=list(TRAJ))
    sp.add_argument("--hf-root", default=HDD_ROOT)
    sp = sub.add_parser("run")
    sp.add_argument("--out", required=True)
    sp.add_argument("--trajs", nargs="+", default=list(TRAJ), choices=list(ALL_TRAJ))
    sp.add_argument("--steps", nargs="*", type=int, default=None)
    sp.add_argument("--unit", choices=("U2", "U5"), default="U2", help="U5 adds the prompt loss to each record")
    sp.add_argument("--hf-roots", nargs="+", default=list(DEFAULT_ROOTS))
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("read")
    sp.add_argument("--out", required=True)
    sp.add_argument("--known-dir", default=str(DATA / t5.KNOWN_DIR))
    sp = sub.add_parser("u5", help="U5's gate and description (no rule)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--u2-dir", default=str(DATA / U2_DIR))
    sp.add_argument("--floor-dir", default=str(FLOOR_DIR))
    sp.add_argument("--seed0-only", action="store_true", help="only the known answer, seed 0 against U2")
    a = ap.parse_args(argv)
    return {"fetch": fetch, "run": run, "read": read, "u5": u5}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
