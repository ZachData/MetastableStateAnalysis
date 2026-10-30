"""
trackT_timaeus/t5.py — T5 U1: the unit cloud at step 0 and on random weights (H1 at init;
`trackT_timaeus/design-T5.md`, frozen 2026-10-30, Blocked 123: D1–D3 all (a)).

Every model is read the same way: the 7 v1 prompts, its own tokenizer (GPT-NeoX's for a NeoX,
Dodecahedron's for a Llama), ≤ 512 tokens, float32, eager, GPU, through T2's `c.states`. Per state
ℓ, ρ_ℓ is the mean pairwise cosine of the unit token vectors averaged over the prompts (T2's
`diag`); ρ_mid is its median over the middle states 1 … L−1, ρ_1:5 its mean over states 1 … 5.
Beside, per state: ρ without position 0, m (‖x̄‖² / mean ‖x_i‖², with and without position 0), n0
(‖x_0‖ / median ‖x_i‖).

Reads (`design-T5.md` "U1"): R1a `pythia-70m` step 0, R1b `timaeus/gpt-2-small-init-seed-5`,
`pythia-160m` / `-410m` step 0 and `dodecahedron-32m` step 1 beside; R2 the factorial, shapes
{70m, 160m, 410m, 32m} × {NeoX, Llama} × init {P, F} × biases {on, off} × 20 seeds (640 models);
the gate's controls (attention only, zeroed output projections). Each record is one model.

Seeds: each factorial model draws every weight from its own CUDA generator, seeded from a hash of
(cell, seed index), so the cells are independent draws and E_bias is a null in distribution.

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t5 run --out <dir> [--parts known real controls factorial]
    python -m trackT_timaeus.t5 read --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import DATA, REPO, V1, prompts_hash
from trackC_writeup import c2_aniso as c
from trackT_timaeus import t2

PROMPTS_HASH = "976b62d677bc"
#: (d, L, heads, KV heads); the Llama MLP is 8d/3 rounded up to 64 (32m's 1024 is Dodecahedron's)
SHAPES = {"70m": (512, 6, 8, 8), "160m": (768, 12, 12, 12), "410m": (1024, 24, 16, 16), "32m": (384, 16, 6, 2)}
ARCHS = ("neox", "llama")
INITS = ("P", "F")
BIASES = (True, False)
N_SEEDS = 20
FLAT = 0.02
SEED = 20261030
#: the real checkpoints and the factorial cell each stands in
REAL = {
    "pythia-70m@step0": ("EleutherAI/pythia-70m", "step0", ("70m", "neox", "P", True)),
    "pythia-160m@step0": ("EleutherAI/pythia-160m", "step0", ("160m", "neox", "P", True)),
    "pythia-410m@step0": ("EleutherAI/pythia-410m", "step0", ("410m", "neox", "P", True)),
    "dodecahedron-32m@step1": ("timaeus/dodecahedron-32m", "step1", ("32m", "llama", "F", False)),
    "gpt-2-small-init": ("timaeus/gpt-2-small-init-seed-5", "main", ("160m", "neox", "F", True)),
}
GPT2_INIT_CONFIG = ("EleutherAI/pythia-160m", "step0")    # the init repo holds weights only; its run's config is 160m's
#: the known answer: T2's diag records (code 4fd84525dd37)
KNOWN_DIR = "trackT/t2u1/2026-10-29-diag"
KNOWN = {"410m": ("EleutherAI/pythia-410m", ("step0", "step4000")),
         "32m": ("timaeus/dodecahedron-32m", ("step1", "step9536"))}
KNOWN_TOL = 1e-6
# the rules (design-T5.md "U1"; fixed before output)
R1_SUPPORTED, R1_FALSIFIED = 0.60, 0.30
E_MIN, TAU_MIN = 0.1, 0.02
DOSE3_SHAPES = ("160m", "410m", "32m")
# the gate (design-T5.md "Calibration gate", U1)
SD_TOL = 0.02
RANGE_TOL = 0.03
NULL_MAX = 0.05
COLLAPSE_MIN = 0.99                 # median over the 20 seeds of ρ at the last state
ZERO_TOL = 1e-6
DEFAULT_ROOTS = (str(DATA / "hf"), "/run/media/system/HDD_1TB/mets_data/hf")


class T5Refused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t5.py", "trackT_timaeus/t2.py", "trackC_writeup/c2_aniso.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- statistics (pure)

def rho_mid(rho: Sequence[float]) -> float:
    """Median of ρ_ℓ over the middle states 1 … L−1 (state 0 the embeddings, state L the final norm)."""
    return float(np.median(np.asarray(rho)[1:-1]))


def rho_15(rho: Sequence[float]) -> float:
    """Mean of ρ_ℓ over states 1 … 5."""
    return float(np.mean(np.asarray(rho)[1:6]))


def state_stats(h: np.ndarray) -> dict:
    """One state (n, d): ρ, ρ without position 0, m, m without position 0, n0."""
    X = np.asarray(h, np.float64)
    nrm = np.linalg.norm(X, axis=1)
    def m_of(Y):
        mu = Y.mean(0)
        return float(mu @ mu / np.mean(np.sum(Y * Y, 1)))
    return {"rho": c.mean_cos(c.unit(X)), "rho_no0": c.mean_cos(c.unit(X[1:])),
            "m": m_of(X), "m_no0": m_of(X[1:]), "n0": float(nrm[0] / np.median(nrm))}


def r1_label(x: float) -> str:
    return "supported" if x >= R1_SUPPORTED else "falsified" if x <= R1_FALSIFIED else "not told apart"


def small_init(d: int) -> float:
    return math.sqrt(2 / (5 * d))


def wang_init(d: int, L: int) -> float:
    return 2 / (L * math.sqrt(d))


def cell_key(shape: str, arch: str, init: str, bias: bool) -> str:
    return f"{shape}-{arch}-{init}-{'b' if bias else 'nob'}"


def factorial_cells() -> List[Tuple[str, str, str, bool]]:
    return [(s, a, i, b) for s in SHAPES for a in ARCHS for i in INITS for b in BIASES]


def seed_of(key: str, s: int) -> int:
    return int(hashlib.sha256(f"{SEED}|{key}|{s}".encode()).hexdigest()[:8], 16)


def r2_reading(med: Dict[str, float]) -> dict:
    """R2 from the cell medians of ρ_mid (`med[cell_key]`, 20 seeds each) and the pooled medians
    `med["<shape>-<arch>-<init>"]` (biases pooled, 40 seeds)."""
    e_bias = {f"{s}-{a}-{i}": med[cell_key(s, a, i, True)] - med[cell_key(s, a, i, False)]
              for s in SHAPES for a in ARCHS for i in INITS}
    tau = max(TAU_MIN, max(abs(v) for v in e_bias.values()))
    e_init = {f"{s}-{a}": med[f"{s}-{a}-F"] - med[f"{s}-{a}-P"] for s in SHAPES for a in ARCHS}
    bar = max(E_MIN, 2 * tau)
    size = {k: e_init[f"{s}-{a}"] >= bar for s in DOSE3_SHAPES for a in ARCHS for k in [f"{s}-{a}"]}
    order = {a: all(e_init[f"70m-{a}"] < e_init[f"{s}-{a}"] for s in DOSE3_SHAPES) for a in ARCHS}
    lf, lp, np_ = med["32m-llama-F"], med["32m-llama-P"], med["32m-neox-P"]
    gap = lf - np_
    frac = (lf - lp) / gap if gap > 0 else None
    halfway = frac is not None and frac >= 0.5
    failed = [f"size {k}" for k, v in size.items() if not v] + [f"order {a}" for a, v in order.items() if not v] \
        + ([] if halfway else ["halfway (Llama-32m P toward NeoX-32m P)"])
    if e_init["32m-llama"] < 2 * tau:
        label = "not supported"
    elif not failed:
        label = "supported"
    else:
        label = "partly"
    arch = {s: med[f"{s}-neox"] - med[f"{s}-llama"] for s in SHAPES}
    return {"e_bias": e_bias, "tau": tau, "bar": bar, "e_init": e_init, "size": size, "order_70m_smallest": order,
            "halfway": {"llama_F": lf, "llama_P": lp, "neox_P": np_, "fraction": frac, "pass": halfway},
            "failed": failed, "label": label, "arch_neox_minus_llama_beside": arch}


# ---------------------------------------------------------------- models

def locate(repo: str, revision: str, roots: Sequence[str]) -> str:
    """The HF cache (`<root>/hub`) holding ``repo`` at ``revision``; refuse if none does."""
    for r in roots:
        if (Path(r) / "hub" / f"models--{repo.replace('/', '--')}" / "refs" / revision).exists():
            return str(Path(r) / "hub")
    raise T5Refused(f"{repo}@{revision} in no cache of {list(roots)}")


def load_real(repo: str, revision: str, roots: Sequence[str], device):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    if repo == REAL["gpt-2-small-init"][0]:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cache = locate(repo, revision, roots)
        path = hf_hub_download(repo, "model.safetensors", revision=revision, cache_dir=cache)
        sd = load_file(path)
        cfg = AutoConfig.from_pretrained(GPT2_INIT_CONFIG[0], revision=GPT2_INIT_CONFIG[1],
                                         cache_dir=locate(*GPT2_INIT_CONFIG, roots))
        cfg._attn_implementation = "eager"
        m = GPTNeoXForCausalLM(cfg)
        missing, unexpected = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        buffers = (".attention.bias", ".attention.masked_bias", "inv_freq")
        missing = [k for k in missing if not k.endswith(buffers)]
        unexpected = [k for k in unexpected if not k.endswith(buffers)]
        if missing or unexpected:
            raise T5Refused(f"{repo}: missing {missing[:5]}, unexpected {unexpected[:5]}")
        m.config._commit_hash = Path(path).parent.name
        return m.to(torch.float32).to(device).eval()
    m = from_pretrained_eager(AutoModelForCausalLM, repo, revision=revision, torch_dtype=torch.float32,
                              cache_dir=locate(repo, revision, roots)).to(device).eval()
    if next(m.parameters()).dtype != torch.float32:
        raise T5Refused(f"{repo} {revision}: weights are not float32")
    if m.config.model_type == "llama":
        t2.check_llama_config(m.config)
    return m


def config_of(shape: str, arch: str, bias: bool, dodeca_cfg):
    d, L, h, kv = SHAPES[shape]
    if arch == "neox":
        from transformers import GPTNeoXConfig
        cfg = GPTNeoXConfig(vocab_size=50304, hidden_size=d, num_hidden_layers=L, num_attention_heads=h,
                            intermediate_size=4 * d, hidden_act="gelu", rotary_pct=0.25, rotary_emb_base=10000,
                            max_position_embeddings=2048, layer_norm_eps=1e-5, use_parallel_residual=True,
                            tie_word_embeddings=False, attention_bias=bias, initializer_range=FLAT)
    else:
        from transformers import LlamaConfig
        ff = dodeca_cfg.intermediate_size if shape == "32m" else int(math.ceil(8 * d / 3 / 64) * 64)
        cfg = LlamaConfig(vocab_size=dodeca_cfg.vocab_size, hidden_size=d, num_hidden_layers=L,
                          num_attention_heads=h, num_key_value_heads=kv, intermediate_size=ff, hidden_act="silu",
                          max_position_embeddings=2048, rms_norm_eps=dodeca_cfg.rms_norm_eps,
                          rope_theta=dodeca_cfg.rope_theta, tie_word_embeddings=True, attention_bias=bias,
                          mlp_bias=bias, initializer_range=FLAT)
    cfg._attn_implementation = "eager"
    cfg.use_cache = False
    return cfg


def weight_class(name: str) -> Optional[str]:
    """Each 2-D weight's role: emb, qkv, attn_out, mlp_in, mlp_out, unembed (None: not a weight)."""
    if not name.endswith(".weight"):
        return None
    for key, cls in (("embed_in", "emb"), ("embed_tokens", "emb"), ("query_key_value", "qkv"),
                     ("q_proj", "qkv"), ("k_proj", "qkv"), ("v_proj", "qkv"), ("attention.dense", "attn_out"),
                     ("o_proj", "attn_out"), ("dense_h_to_4h", "mlp_in"), ("gate_proj", "mlp_in"),
                     ("up_proj", "mlp_in"), ("dense_4h_to_h", "mlp_out"), ("down_proj", "mlp_out"),
                     ("embed_out", "unembed"), ("lm_head", "unembed")):
        if key in name:
            return cls
    return None


def init_sd(cls: str, init: str, d: int, L: int) -> float:
    if init == "F":
        return FLAT
    return wang_init(d, L) if cls in ("attn_out", "mlp_out") else small_init(d)


def build(shape: str, arch: str, init: str, bias: bool, seed: int, dodeca_cfg, device):
    """A random model of the cell: every 2-D weight N(0, init_sd), biases 0 (absent when off), norm
    gains 1 and LayerNorm shifts 0, from one CUDA generator seeded ``seed``."""
    import torch
    from transformers import GPTNeoXForCausalLM, LlamaForCausalLM
    cfg = config_of(shape, arch, bias, dodeca_cfg)
    with torch.device(device):
        m = (GPTNeoXForCausalLM if arch == "neox" else LlamaForCausalLM)(cfg)
    if arch == "neox" and not bias:
        for layer in m.gpt_neox.layers:
            layer.mlp.dense_h_to_4h.bias = None
            layer.mlp.dense_4h_to_h.bias = None
    d, L = SHAPES[shape][:2]
    g = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        for n, p in m.named_parameters():
            cls = weight_class(n)
            if cls is not None:
                p.normal_(0.0, init_sd(cls, init, d, L), generator=g)
            elif n.endswith(".bias"):
                p.zero_()
            elif "norm" in n:
                p.fill_(1.0)
            else:
                raise T5Refused(f"{n}: no init rule")
    lin_bias = [n for n, _ in m.named_parameters() if n.endswith(".bias") and "norm" not in n]
    if bool(lin_bias) != bias:
        raise T5Refused(f"{cell_key(shape, arch, init, bias)}: linear biases {len(lin_bias)}")
    return m.to(torch.float32).eval()


def class_sds(model) -> Dict[str, float]:
    """RMS of every weight of each role, pooled over layers."""
    acc: Dict[str, List[float]] = {}
    for n, p in model.named_parameters():
        cls = weight_class(n)
        if cls is not None:
            x = p.detach().double()
            acc.setdefault(cls, [0.0, 0.0])
            acc[cls][0] += float((x * x).sum())
            acc[cls][1] += x.numel()
    return {k: math.sqrt(s / n) for k, (s, n) in acc.items()}


def bias_max(model) -> float:
    xs = [float(p.detach().abs().max()) for n, p in model.named_parameters() if n.endswith(".bias")]
    return max(xs) if xs else 0.0


def attention_only(model) -> None:
    """Llama's blocks replaced by x ← attention(RMSNorm(x)): no MLP, no residual, the norms kept."""
    def fwd(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
            output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kw):
        out = self.self_attn(hidden_states=self.input_layernorm(hidden_states), attention_mask=attention_mask,
                             position_ids=position_ids, past_key_value=None, output_attentions=False,
                             use_cache=False, cache_position=cache_position, position_embeddings=position_embeddings)
        return (out[0],)
    for layer in model.model.layers:
        layer.forward = types.MethodType(fwd, layer)


def zero_outputs(model) -> None:
    import torch
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()


def read_model(model, tok, device) -> dict:
    per = []
    ntok = []
    for p in V1:
        hs, ids = c.states(model, tok, p, device)
        per.append([state_stats(h) for h in hs])
        ntok.append(len(ids))
    keys = ("rho", "rho_no0", "m", "m_no0", "n0")
    rec = {k: np.mean([[s[k] for s in row] for row in per], 0).tolist() for k in keys}
    rec["rho_per_prompt"] = [[s["rho"] for s in row] for row in per]
    rec["n_tokens"] = ntok
    rec["n_states"] = len(per[0])
    if rec["n_states"] != model.config.num_hidden_layers + 1 or not all(np.all(np.isfinite(rec[k])) for k in keys):
        raise T5Refused(f"record not populated: {rec['n_states']} states")
    rec["rho_mid"], rec["rho_15"] = rho_mid(rec["rho"]), rho_15(rec["rho"])
    rec["rho_mid_no0"] = rho_mid(rec["rho_no0"])
    return rec


# ---------------------------------------------------------------- commands

def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(path)


def run(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if prompts_hash() != PROMPTS_HASH:
        raise T5Refused(f"prompts_hash {prompts_hash()} != {PROMPTS_HASH}")
    torch, device = c._torch_setup()
    import transformers
    from transformers import AutoConfig
    roots = a.hf_roots
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"design": "trackT_timaeus/design-T5.md", "code": code_sha(), "prompts_hash": prompts_hash(),
            "prompts": list(V1), "max_tokens": t2.MAX_TOKENS, "dtype": "float32", "tf32": False,
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "transformers": transformers.__version__, "hf_roots": list(roots),
            "seed": SEED, "n_seeds": N_SEEDS, "shapes": SHAPES}
    _write(out / "meta.json", meta)
    toks = {"neox": t2.p_tokenizer(), "llama": t2.tokenizer("32m")}
    dodeca_cfg = AutoConfig.from_pretrained(KNOWN["32m"][0], revision="step1", cache_dir=locate(KNOWN["32m"][0], "step1", roots))
    t0 = time.time()

    def done(p: Path) -> bool:
        return p.exists() and not a.force

    def emit(path: Path, rec: dict) -> None:
        rec.update({"code": meta["code"], "wall_s": round(time.time() - t0, 1)})
        _write(path, rec)
        print(json.dumps({"wrote": str(path.relative_to(out)), "rho_mid": round(rec.get("rho_mid", float("nan")), 4),
                          "t": rec["wall_s"]}), flush=True)

    if "known" in a.parts:
        for mk, (repo, revs) in KNOWN.items():
            for rev in revs:
                p = out / "known" / f"{mk}@{rev}.json"
                if done(p):
                    continue
                m = load_real(repo, rev, roots, device)
                rec = read_model(m, toks["neox" if mk == "410m" else "llama"], device)
                emit(p, {"model": repo, "revision": rev, **rec})
                del m
                torch.cuda.empty_cache()
    if "real" in a.parts:
        for name, (repo, rev, cell) in REAL.items():
            p = out / "real" / f"{name}.json"
            if done(p):
                continue
            m = load_real(repo, rev, roots, device)
            rec = read_model(m, toks[cell[1]], device)
            emit(p, {"model": repo, "revision": rev, "commit": t2.commit_of(m), "cell": cell_key(*cell),
                     "sds": class_sds(m), "bias_max": bias_max(m), **rec})
            del m
            torch.cuda.empty_cache()
    if "controls" in a.parts:
        key = cell_key("32m", "llama", "F", False)
        for kind in ("attn_only", "zeroed"):
            for s in range(N_SEEDS):
                p = out / "controls" / kind / f"s{s:02d}.json"
                if done(p):
                    continue
                m = build("32m", "llama", "F", False, seed_of(key, s), dodeca_cfg, device)
                (attention_only if kind == "attn_only" else zero_outputs)(m)
                rec = read_model(m, toks["llama"], device)
                emit(p, {"control": kind, "cell": key, "seed_index": s, "seed": seed_of(key, s), **rec})
                del m
                torch.cuda.empty_cache()
    if "factorial" in a.parts:
        for cell in factorial_cells():
            key = cell_key(*cell)
            if a.cells and key not in a.cells:
                continue
            for s in range(a.n_seeds):
                p = out / "factorial" / key / f"s{s:02d}.json"
                if done(p):
                    continue
                m = build(*cell, seed_of(key, s), dodeca_cfg, device)
                rec = read_model(m, toks[cell[1]], device)
                sds = class_sds(m) if s == 0 else None
                emit(p, {"cell": key, "seed_index": s, "seed": seed_of(key, s), "sds": sds, **rec})
                del m
                torch.cuda.empty_cache()
    print(json.dumps({"done": True, "wall_s": round(time.time() - t0, 1)}))
    return 0


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def gate(out: Path, known_dir: Path, recs: dict) -> dict:
    """The U1 gate, in the design's order; every check reports, `pass` is their conjunction."""
    g = {}
    # known answer
    diffs = {}
    for mk, (_, revs) in KNOWN.items():
        ref = _load(known_dir / f"diag_{mk}.json")["populations"]
        for rev in revs:
            got = _load(out / "known" / f"{mk}@{rev}.json")["rho"]
            diffs[f"{mk}@{rev}"] = float(np.max(np.abs(np.asarray(got) - np.asarray(ref[rev]["mean_cos"]))))
    g["known_answer"] = {"max_diff": diffs, "pass": max(diffs.values()) <= KNOWN_TOL}
    # the init implementation: per role, the random cell's seed-0 SDs against the real checkpoint's
    sd = {}
    for name, (_, _, cell) in REAL.items():
        real = recs["real"][name]
        rnd = recs["factorial"][cell_key(*cell)][0]["sds"]
        rel = {k: rnd[k] / real["sds"][k] - 1 for k in ("emb", "qkv", "attn_out", "mlp_in", "mlp_out")}
        sd[name] = {"rel": rel, "bias_max_real": real["bias_max"], "pass": max(abs(v) for v in rel.values()) <= SD_TOL}
    g["init_sds"] = {"by_checkpoint": sd, "pass": all(v["pass"] for v in sd.values())}
    rng = {}
    for name, (_, _, cell) in REAL.items():
        xs = [r["rho_mid"] for r in recs["factorial"][cell_key(*cell)]]
        x = recs["real"][name]["rho_mid"]
        rng[name] = {"real": x, "lo": min(xs), "hi": max(xs), "pass": min(xs) - RANGE_TOL <= x <= max(xs) + RANGE_TOL}
    g["real_in_range"] = {"by_checkpoint": rng, "pass": all(v["pass"] for v in rng.values())}
    # planted null and τ
    med = medians(recs["factorial"])
    r2 = r2_reading(med)
    g["planted_null"] = {"e_bias": r2["e_bias"], "max_abs": max(abs(v) for v in r2["e_bias"].values()),
                         "tau": r2["tau"], "pass": max(abs(v) for v in r2["e_bias"].values()) <= NULL_MAX}
    # positive control: pure attention collapses
    last = [r["rho"][-1] for r in recs["controls"]["attn_only"]]
    g["collapse"] = {"median_last": float(np.median(last)), "min_last": float(np.min(last)),
                     "pass": float(np.median(last)) >= COLLAPSE_MIN}
    # zeroed output projections: every state the embeddings; the contrast at the rule's size
    zr = recs["controls"]["zeroed"]
    eq = max(float(np.max(np.abs(np.asarray(r["rho"]) - r["rho"][0]))) for r in zr)
    e_z = med[cell_key("32m", "llama", "F", False)] - float(np.median([r["rho_mid"] for r in zr]))
    g["zeroed"] = {"max_state_minus_state0": eq, "E_F_minus_zeroed": e_z, "bar": r2["bar"],
                   "pass": eq <= ZERO_TOL and e_z >= r2["bar"]}
    g["pass"] = all(v["pass"] for v in g.values() if isinstance(v, dict))
    return g


def medians(fac: Dict[str, List[dict]]) -> Dict[str, float]:
    med = {k: float(np.median([r["rho_mid"] for r in v])) for k, v in fac.items()}
    for s in SHAPES:
        for a in ARCHS:
            for i in INITS:
                med[f"{s}-{a}-{i}"] = float(np.median([r["rho_mid"] for b in BIASES for r in fac[cell_key(s, a, i, b)]]))
            med[f"{s}-{a}"] = float(np.median([r["rho_mid"] for i in INITS for b in BIASES
                                               for r in fac[cell_key(s, a, i, b)]]))
    return med


def collect(out: Path) -> dict:
    fac = {}
    for cell in factorial_cells():
        key = cell_key(*cell)
        rs = [_load(p) for p in sorted((out / "factorial" / key).glob("s*.json"))]
        if len(rs) != N_SEEDS:
            raise T5Refused(f"{key}: {len(rs)} of {N_SEEDS} seeds")
        fac[key] = rs
    ctl = {k: [_load(p) for p in sorted((out / "controls" / k).glob("s*.json"))] for k in ("attn_only", "zeroed")}
    if any(len(v) != N_SEEDS for v in ctl.values()):
        raise T5Refused(f"controls: {[len(v) for v in ctl.values()]} of {N_SEEDS}")
    real = {n: _load(out / "real" / f"{n}.json") for n in REAL}
    codes = {r["code"] for v in fac.values() for r in v} | {r["code"] for v in ctl.values() for r in v} \
        | {r["code"] for r in real.values()}
    if len(codes) != 1:
        raise T5Refused(f"records from more than one code version: {codes}")
    return {"factorial": fac, "controls": ctl, "real": real, "code": codes.pop()}


def read(a) -> int:
    out = Path(a.out)
    recs = collect(out)
    g = gate(out, Path(a.known_dir), recs)
    res = {"code": recs["code"], "reader": code_sha(), "gate": g}
    if g["pass"]:
        re = recs["real"]
        med = medians(recs["factorial"])
        res["R1a"] = {"rho_15": re["pythia-70m@step0"]["rho_15"], "label": r1_label(re["pythia-70m@step0"]["rho_15"]),
                      "beside_160m_step0_rho_15": re["pythia-160m@step0"]["rho_15"]}
        res["R1b"] = {"rho_15": re["gpt-2-small-init"]["rho_15"], "label": r1_label(re["gpt-2-small-init"]["rho_15"]),
                      "beside_rho_mid": re["gpt-2-small-init"]["rho_mid"],
                      "beside_160m_step0_rho_mid": re["pythia-160m@step0"]["rho_mid"]}
        res["R2"] = r2_reading(med)
        res["cells"] = {k: {"median_rho_mid": med[k],
                            "range": [min(r["rho_mid"] for r in v), max(r["rho_mid"] for r in v)],
                            "median_rho_15": float(np.median([r["rho_15"] for r in v])),
                            "median_rho_mid_no0": float(np.median([r["rho_mid_no0"] for r in v]))}
                        for k, v in recs["factorial"].items()}
        res["real"] = {n: {k: r[k] for k in ("rho_mid", "rho_15", "rho_mid_no0", "rho", "m", "m_no0", "n0")}
                       for n, r in re.items()}
    else:
        res["refused"] = "the gate failed: no reading (design-T5.md 'Calibration gate'; back to the user)"
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    show = {"gate": {k: (v["pass"] if isinstance(v, dict) else v) for k, v in g.items()}}
    if g["pass"]:
        show.update({k: res[k] for k in ("R1a", "R1b")})
        show["R2"] = {k: res["R2"][k] for k in ("tau", "bar", "e_init", "order_70m_smallest", "halfway", "failed", "label")}
    print(json.dumps(show, indent=1))
    return 0 if g["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("run")
    sp.add_argument("--out", required=True)
    sp.add_argument("--parts", nargs="+", default=["known", "real", "controls", "factorial"],
                    choices=["known", "real", "controls", "factorial"])
    sp.add_argument("--cells", nargs="*", default=None)
    sp.add_argument("--n-seeds", type=int, default=N_SEEDS)
    sp.add_argument("--hf-roots", nargs="+", default=list(DEFAULT_ROOTS))
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("read")
    sp.add_argument("--out", required=True)
    sp.add_argument("--known-dir", default=str(DATA / KNOWN_DIR))
    a = ap.parse_args(argv)
    return {"run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
