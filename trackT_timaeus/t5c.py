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

Tier 1, description (nothing here is registered).
    python -m trackT_timaeus.t5c fetch [--hf-root <HDD_1TB>/mets_data/hf]
    METS_DATA=<main>/data python -m trackT_timaeus.t5c run --out <dir>
    METS_DATA=<main>/data python -m trackT_timaeus.t5c read --out <dir>
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
    repo = TRAJ[traj][0]
    if traj == "gpt-2-small":
        return load_gpt2(step, roots, device)
    m = t5.load_real(repo, f"step{step}", roots, device)
    return m, {"commit": t2.commit_of(m)}


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
    meta = {"design": "trackT_timaeus/design-T5.md", "unit": "U2", "code": code_sha(), "prompts_hash": prompts_hash(),
            "prompts": list(V1), "max_tokens": t2.MAX_TOKENS, "dtype": "float32", "tf32": False,
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "hf_roots": list(roots),
            "trajectories": {k: {"repo": v[0], "steps": list(v[1]), "tokenizer": v[2]} for k, v in TRAJ.items()}}
    _write(out / "meta.json", meta)
    toks = {}
    t0 = time.time()
    for name in a.trajs:
        repo, steps, tk = TRAJ[name]
        if tk not in toks:
            toks[tk] = t2.p_tokenizer() if tk == "neox" else t2.tokenizer(tk)
        for s in (a.steps or steps):
            if s not in steps:
                raise T5cRefused(f"{name}: step {s} is not on its frozen list")
            p = out / name / f"step{s}.json"
            if p.exists() and not a.force:
                continue
            m, info = load(name, s, roots, device)
            rec = read_model(m, toks[tk], device)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("fetch")
    sp.add_argument("--trajs", nargs="+", default=["dodecahedron-32m", "dodecahedron-67m", "gpt-2-small"],
                    choices=list(TRAJ))
    sp.add_argument("--hf-root", default=HDD_ROOT)
    sp = sub.add_parser("run")
    sp.add_argument("--out", required=True)
    sp.add_argument("--trajs", nargs="+", default=list(TRAJ), choices=list(TRAJ))
    sp.add_argument("--steps", nargs="*", type=int, default=None)
    sp.add_argument("--hf-roots", nargs="+", default=list(DEFAULT_ROOTS))
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("read")
    sp.add_argument("--out", required=True)
    sp.add_argument("--known-dir", default=str(DATA / t5.KNOWN_DIR))
    a = ap.parse_args(argv)
    return {"fetch": fetch, "run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
