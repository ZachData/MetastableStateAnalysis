"""
trackT_timaeus/t5h.py — T5 U4a and U4b: the causal arm (`design-T5.md` "U4, the causal arm, designed",
frozen 2026-11-03 on Blocked 132; implementation details fixed before any arm output, "U4a's details").

Dodecahedron's recipe trained from scratch at 524,288 tokens a step (2048 × 256) under its steps and
LR schedule (linear from 0 to 3.5e-3 over 1000 steps, then cosine toward 9536, floor 0), stopped at
4369; AdamW β 0.9 / 0.95, weight decay 0.01 on matrices, clip 1.0, bf16 autocast over float32 master
weights. The model is `t5.build`'s Llama-32m (U1's code; 31,469,952 parameters). The data is FineWeb
`sample-10BT` (revision pinned) in Dodecahedron's tokenizer, packed into 2048-token sequences and
shuffled once; every arm reads the same stream in the same order. Checkpoints at step 0 and every
`t2.GRID` step ≤ 4369 are read through U2's per-state code (`t5c.read_model`).

Parts (commands):
    shard   tokenize and pack the stream (CPU; `<root>/data/`)
    theta   θ, tokens/s on A0's configuration: 50 timed steps after 10 warm-up steps (discarded)
    train   one arm, resumable (`<root>/<arm>/`)
    read    U2's statistics at each saved checkpoint of an arm
    known   the reader's known answer: released `-32m` steps 1 and 9536, saved and reloaded through
            this file's checkpoint format, against U2's records
    floor   the noise floor from outside: ρ_mid over `pythia-410m` seeds 0–9 at 4000 and 143000
    gate    U4a's gate on an arm's records (stage `start`, `trough` or `late`)

The rule (pure: `late_level`, `trough`, `arm_label`, `delay_label`, `hyp_label`, `interaction`) is
written here before any arm output and tested on planted series.

Tier 1, description (nothing here is registered).
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t5h shard
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t5h theta
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t5h train --arm A0 --until 1253
    HF_HOME=<HDD_1TB>/mets_data/hf METS_DATA=<main>/data python -m trackT_timaeus.t5h read --arm A0
    METS_DATA=<main>/data python -m trackT_timaeus.t5h gate --arm A0 --stage trough
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

from trackA_force.a1a import DATA, REPO, prompts_hash
from trackT_timaeus import t2, t5

# ---------------------------------------------------------------- the recipe (design-T5.md "U4", fixed)
SHAPE = "32m"
N_PARAMS = 31_469_952                    # B5
SEQ = 2048
BATCH_SEQ = 256
TOKENS_STEP = SEQ * BATCH_SEQ            # 524,288
PEAK, WARMUP, TOTAL = 3.5e-3, 1000, 9536
READ_STEP, CONT_TO = 4369, 5970
BETAS, EPS, WD, CLIP = (0.9, 0.95), 1e-8, 0.01, 1.0
CKPT_STEPS = (0,) + tuple(s for s in t2.GRID if s <= READ_STEP)          # 48 grid steps and step 0
CONT_STEPS = tuple(s for s in t2.GRID if READ_STEP < s <= CONT_TO)       # 5108, 5970 (the delay check)
#: arm → (init, generator seed, biases); A1 is A0 plus zero biases, A3 is A2 plus zero biases
#: A0pp is A0″, the second replicate seed (Blocked 133 (i) (d), the user's, 2026-11-03), trained to 5970
ARMS = {"A0": ("F", 0, False), "A0p": ("F", 1, False), "A0pp": ("F", 2, False), "A1": ("F", 0, True),
        "A2": ("P", 0, False), "A3": ("P", 0, True)}
REPLICATES = ("A0", "A0p", "A0pp")       # τ from the largest of their three pairs
# the data (design "data"; details "U4a's details" (d1)–(d5))
FINEWEB = "HuggingFaceFW/fineweb"
FINEWEB_REV = "9bb295ddab0e05d785b879661af7260fed5140fc"
FINEWEB_DIR = "sample/10BT"
DATA_SEED = 20261102                     # the shuffle of the 4369 × 256 sequences (design)
TAIL_SEED = 20261103                     # the continuation's 1601 × 256 (details (d4))
N_MAIN = READ_STEP * BATCH_SEQ           # 1,118,464 sequences, 2.29B tokens
N_TAIL = (CONT_TO - READ_STEP) * BATCH_SEQ
EOT = "<|endoftext|>"
# θ (design "Throughput"; B9, at the 4 GPU-day cap)
THETA_WARM, THETA_TIMED = 10, 50
N_MICRO, COMPILE = 64, True              # details (t12): micro-batch 4, compiled; every arm, every segment
THETA_ALL, THETA_ARMS = 51_911, 39_768  # B11: six arms + five continuations (A0, A0″, A1, A2, A3); six arms
# the gate (design "Gate"; fixed)
KNOWN_TOL = 1e-6                         # placed (U2's known-answer bar)
SD_TOL = 0.02                            # placed (U1's init bar)
BAND = {"F": (0.787, 0.881), "P": (0.415, 0.544)}
FLOOR_STEPS = (4000, 143000)
FLOOR_SEEDS = tuple(range(10))
FLOOR_MAX = 0.2                          # Blocked 133 (i) (d): the largest pair difference at 143000 (0.1996);
                                         # 2 × SD and the largest pair difference both ≤ it (first frozen 0.1, 2 × SD only)
TROUGH_BY, TROUGH_MAX = 1253, 0.25
LATE_MIN = 0.60                          # placed (above every Pythia end level read, 0.37–0.44)
LOSS_FALL = (range(1, 11), range(91, 101))   # mean training loss over steps 91–100 below 1–10 (details (t6))
# the rule (design "The statistic and the rule"; fixed)
L_STEPS = (3738, 4369)
LP_STEPS = (5108, 5970)
T_FROM = 100
TAU_MIN, BAR_MIN = 0.02, 0.2            # bar's floor raised from 0.1 by Blocked 133 (i) (d), before any arm output
N_BOOT, BOOT_SEED = 1000, 20261030
U2_DIR = "trackT/t5u2/2026-10-30"
HDD = Path("/run/media/system/HDD_1TB/mets_data")
ROOT = HDD / "trackT" / "t5u4"


class T5hRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackT_timaeus/t5h.py", "trackT_timaeus/t5c.py", "trackT_timaeus/t5.py", "trackT_timaeus/t2.py",
              "trackC_writeup/c2_aniso.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the schedule (pure)

def lr_at(step: int, peak: float = PEAK, warmup: int = WARMUP, total: int = TOTAL) -> float:
    """The LR of update ``step`` (1-indexed: update t makes checkpoint t), B3's: peak·t/warmup to
    the warmup's end, then peak·(1 + cos(π (t − warmup)/(total − warmup)))/2, floor 0."""
    if step < 1:
        raise T5hRefused(f"update {step}: updates are 1-indexed")
    if step <= warmup:
        return peak * step / warmup
    return peak * (1 + math.cos(math.pi * (step - warmup) / (total - warmup))) / 2


def batch_rows(step: int) -> Tuple[str, int, int]:
    """Update ``step``'s sequences: (part, first row, end row) of the shuffled shard."""
    if 1 <= step <= READ_STEP:
        return "main", (step - 1) * BATCH_SEQ, step * BATCH_SEQ
    if READ_STEP < step <= CONT_TO:
        k = step - READ_STEP - 1
        return "tail", k * BATCH_SEQ, (k + 1) * BATCH_SEQ
    raise T5hRefused(f"update {step}: past the stream ({CONT_TO})")


def decays(name: str, p) -> bool:
    """Weight decay on matrices only: not on norm gains or biases (design "optimizer")."""
    return p.ndim == 2


# ---------------------------------------------------------------- the rule (pure)

def at(series: Dict[int, float], steps: Sequence[int]) -> float:
    miss = [s for s in steps if s not in series]
    if miss:
        raise T5hRefused(f"series lacks steps {miss}")
    return float(np.mean([series[s] for s in steps]))


def late_level(series: Dict[int, float]) -> float:
    """L: mean ρ_mid at 3738 and 4369."""
    return at(series, L_STEPS)


def late_level_cont(series: Dict[int, float]) -> float:
    """L′: mean ρ_mid at 5108 and 5970 (the delay check)."""
    return at(series, LP_STEPS)


def trough(series: Dict[int, float], until: int = READ_STEP) -> Tuple[float, int]:
    """T: min ρ_mid over the grid steps T_FROM … ``until``, with its step."""
    ss = [s for s in series if T_FROM <= s <= until]
    if not ss:
        raise T5hRefused(f"no grid step in {T_FROM} … {until}")
    s = min(ss, key=lambda k: series[k])
    return float(series[s]), int(s)


def tau_bar(*levels: float) -> Tuple[float, float]:
    """τ = max(0.02, the largest pair difference among the replicates' L: A0, A0′, A0″);
    bar = max(0.2, 2τ) (Blocked 133 (i) (d))."""
    if len(levels) != len(REPLICATES):
        raise T5hRefused(f"tau_bar takes the {len(REPLICATES)} replicates' L, got {len(levels)}")
    tau = max(TAU_MIN, max(levels) - min(levels))
    return tau, max(BAR_MIN, 2 * tau)


def arm_label(dl: float, r_a0: float, bar: float) -> str:
    """At 4369: lowers (ΔL ≥ R_A0/2), partly (bar ≤ ΔL < R_A0/2), does not (|ΔL| < bar), raises
    (ΔL ≤ −bar). If bar ≥ R_A0/2 the design's seed-noise check fails and 'partly' cannot occur;
    the caller reports that (``readable``)."""
    if dl >= r_a0 / 2:
        return "lowers"
    if dl >= bar:
        return "partly"
    if dl <= -bar:
        return "raises"
    return "does not"


def readable(r_a0: float, bar: float) -> bool:
    """The seed-noise gate: bar < R_A0 / 2."""
    return bar < r_a0 / 2


def delay_label(dl: float, dlp: float, bar: float) -> str:
    """The delay check: holds if ΔL′ ≥ bar and ΔL′ ≥ ΔL / 2, else delays."""
    return "holds" if dlp >= bar and dlp >= dl / 2 else "delays"


def hyp_label(arm: str, delay: Optional[str]) -> str:
    """H1 or H2 from its arm's label and delay check (None: no continuation in the cap)."""
    if arm in ("does not", "raises"):
        return "not supported"
    if delay is None:
        return "lowers the late level at 4369; removes or delays not told apart"
    if delay == "delays":
        return "not told apart (delays the rise)"
    return "supported" if arm == "lowers" else "partly"


def interaction(l: Dict[str, float], bar: float) -> Tuple[float, str]:
    """I = ΔL_A3 − ΔL_A1 − ΔL_A2 = L_A1 + L_A2 − L_A0 − L_A3, at bar_I = √2 · bar."""
    i = l["A1"] + l["A2"] - l["A0"] - l["A3"]
    b = math.sqrt(2) * bar
    if i >= b:
        return i, "together lower it more than the sum"
    if i <= -b:
        return i, "less than the sum"
    return i, "no interaction seen at this noise"


# ---------------------------------------------------------------- the data shard

def fineweb_files(hf_cache: Optional[str] = None):
    """The sample's parquet files, in name order, at the pinned revision; each downloaded only when
    the stream reaches it."""
    from huggingface_hub import HfApi, hf_hub_download
    names = sorted(s.rfilename for s in HfApi().dataset_info(FINEWEB, revision=FINEWEB_REV).siblings
                   if s.rfilename.startswith(FINEWEB_DIR + "/") and s.rfilename.endswith(".parquet"))
    for n in names:
        yield Path(hf_hub_download(FINEWEB, n, repo_type="dataset", revision=FINEWEB_REV, cache_dir=hf_cache))


def shard(a) -> int:
    """(d1) each document's `text` encoded by Dodecahedron's tokenizer with no special tokens (it
    adds none), followed by `<|endoftext|>`; (d2) documents concatenated in file order and row order,
    cut into consecutive 2048-token sequences, no attention mask across documents; (d3) the first
    4369 × 256 sequences permuted by `default_rng(20261102)` and stored in that order; (d4) the next
    1601 × 256 (the delay check's) permuted by `default_rng(20261103)`; (d5) uint16, `<root>/data/`."""
    import pyarrow.parquet as pq
    out = Path(a.root) / "data"
    out.mkdir(parents=True, exist_ok=True)
    tj = t2.tokenizer_json(SHAPE)
    ref = t2.tokenizer(SHAPE)                           # the reader's (merges re-joined for tokenizers 0.19)
    tk = ref.backend_tokenizer
    eot = tk.token_to_id(EOT)
    if eot is None or tk.get_vocab_size() != 16384:
        raise T5hRefused(f"tokenizer: {EOT} id {eot}, vocab {tk.get_vocab_size()}")
    probe = "Hello world.  This is   a test of the tokenizer: 3.14159, naïve café, 東京."
    if tk.encode(probe, add_special_tokens=False).ids != ref(probe)["input_ids"]:
        raise T5hRefused("the tokenizers library and t2.tokenizer disagree on the probe")
    need = N_MAIN + N_TAIL
    raw = Path(a.tmp_dir) / "t5u4_stream.tmp.bin"      # on the SSD: the permutation reads it at random
    raw.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_seq, n_docs, buf, files_used = 0, 0, np.zeros(0, np.uint16), []
    with raw.open("wb") as f:
        for path in fineweb_files(a.hf_cache):
            if n_seq >= need:
                break
            pf = pq.ParquetFile(path)
            files_used.append({"file": path.name, "rows": pf.metadata.num_rows})
            for rb in pf.iter_batches(batch_size=8192, columns=["text"]):
                texts = rb.column(0).to_pylist()
                enc = tk.encode_batch(texts, add_special_tokens=False)
                lst: List[int] = []
                for e in enc:
                    lst.extend(e.ids)
                    lst.append(eot)
                ids = np.asarray(lst, np.uint16)
                n_docs += len(texts)
                buf = np.concatenate([buf, ids])
                k = min(len(buf) // SEQ, need - n_seq)
                if k:
                    f.write(buf[:k * SEQ].tobytes())
                    buf = buf[k * SEQ:]
                    n_seq += k
                if n_seq >= need:
                    files_used[-1]["last_batch_docs_through"] = n_docs
                    break
            print(json.dumps({"file": path.name, "sequences": n_seq, "docs": n_docs,
                              "t": round(time.time() - t0)}), flush=True)
    if n_seq < need:
        raise T5hRefused(f"the sample ran out at {n_seq} sequences of {need}")
    stream = np.memmap(raw, np.uint16, "r", shape=(need, SEQ))
    if int(stream.max()) >= 16384:
        raise T5hRefused("a token id ≥ 16384")
    meta = {"design": "trackT_timaeus/design-T5.md", "unit": "U4a", "code": code_sha(), "fineweb": FINEWEB,
            "revision": FINEWEB_REV, "files": files_used, "docs_read": n_docs, "seq": SEQ, "eot_id": eot,
            "tokenizer_json_sha256": hashlib.sha256(tj.read_bytes()).hexdigest()}
    for part, lo, n, seed in (("main", 0, N_MAIN, DATA_SEED), ("tail", N_MAIN, N_TAIL, TAIL_SEED)):
        perm = np.random.default_rng(seed).permutation(n)
        dst = np.memmap(out / f"{part}.bin", np.uint16, "w+", shape=(n, SEQ))
        for i in range(0, n, 65536):
            dst[i:i + 65536] = stream[lo + perm[i:i + 65536]]
        dst.flush()
        h = hashlib.sha256()
        for i in range(0, n, 65536):
            h.update(np.ascontiguousarray(dst[i:i + 65536]).tobytes())
        meta[part] = {"sequences": n, "seed": seed, "sha256": h.hexdigest(),
                      "eot_fraction": float(np.mean(dst[:4096] == eot))}
        del dst
    del stream
    raw.unlink()
    meta["wall_s"] = round(time.time() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"done": True, **{k: meta[k] for k in ("docs_read", "wall_s")}}))
    return 0


def open_shard(root: Path) -> Dict[str, np.ndarray]:
    d = Path(root) / "data"
    meta = json.loads((d / "meta.json").read_text())
    return {p: np.memmap(d / f"{p}.bin", np.uint16, "r", shape=(meta[p]["sequences"], SEQ)) for p in ("main", "tail")}


def batch_of(shard_: Dict[str, np.ndarray], step: int) -> np.ndarray:
    part, lo, hi = batch_rows(step)
    return np.asarray(shard_[part][lo:hi])


# ---------------------------------------------------------------- the model

def dodeca_cfg(roots: Sequence[str]):
    from transformers import AutoConfig
    repo = t2.MODELS[SHAPE]
    return AutoConfig.from_pretrained(repo, revision="step1", cache_dir=t5.locate(repo, "step1", roots))


def build_arm(arm: str, dcfg, device, attn: str = "sdpa"):
    """The arm's step-0 model (`t5.build`, U1's), float32, in train mode, attention ``attn``."""
    init, seed, bias = ARMS[arm]
    def cfg_edit(cfg):
        cfg._attn_implementation = attn             # the attention class is chosen at construction
    m = t5.build(SHAPE, "llama", init, bias, seed, dcfg, device, cfg_edit=cfg_edit)
    t2.check_llama_config(m.config)
    n = sum(p.numel() for p in m.parameters())
    if n != N_PARAMS + (16 * 3456 if bias else 0):
        raise T5hRefused(f"{arm}: {n} parameters")
    kinds = {type(layer.self_attn).__name__ for layer in m.model.layers}
    if len(kinds) != 1 or (attn == "sdpa") != ("Sdpa" in kinds.pop()):
        raise T5hRefused(f"{arm}: attention modules not {attn}")
    return m.train()


def save_weights(model, path: Path) -> None:
    """Float32 safetensors without the tied `lm_head.weight` (it is `embed_tokens.weight`)."""
    from safetensors.torch import save_file
    sd = {k: v.detach().to("cpu", copy=True).float().contiguous() for k, v in model.state_dict().items()
          if k != "lm_head.weight"}
    tmp = path.with_suffix(".tmp")
    save_file(sd, str(tmp))
    tmp.replace(path)


def load_weights(path: Path, arm: str, dcfg, device):
    """A saved checkpoint as the reader reads a model: eager, float32, eval."""
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaForCausalLM
    cfg = t5.config_of(SHAPE, "llama", ARMS[arm][2], dcfg)
    with torch.device(device):
        m = LlamaForCausalLM(cfg)
    missing, unexpected = m.load_state_dict(load_file(str(path), device=str(device)), strict=False)
    if [k for k in missing if k != "lm_head.weight"] or unexpected:
        raise T5hRefused(f"{path}: missing {missing[:4]}, unexpected {unexpected[:4]}")
    if m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr():
        raise T5hRefused(f"{path}: lm_head is not tied")
    return m.to(torch.float32).eval()


# ---------------------------------------------------------------- the loop

def make_opt(model, lr: float = PEAK):
    import torch
    dec = [p for n, p in model.named_parameters() if decays(n, p)]
    nod = [p for n, p in model.named_parameters() if not decays(n, p)]
    return torch.optim.AdamW([{"params": dec, "weight_decay": WD}, {"params": nod, "weight_decay": 0.0}],
                             lr=lr, betas=BETAS, eps=EPS, foreach=True)


def step_once(model, opt, tokens, step: int, n_micro: int, amp: bool = True, sched=lr_at) -> dict:
    """One update on ``tokens`` (B, SEQ) int64 on the model's device: next-token loss over each
    sequence's 2047 predictions, averaged over the batch through ``n_micro`` equal micro-batches;
    global-norm clip at 1.0; AdamW at ``sched(step)``."""
    import torch
    import torch.nn.functional as F
    lr = sched(step)
    for g in opt.param_groups:
        g["lr"] = lr
    B = tokens.shape[0]
    if B % n_micro:
        raise T5hRefused(f"batch {B} not divisible into {n_micro}")
    mb = B // n_micro
    tot = 0.0
    for i in range(n_micro):
        x = tokens[i * mb:(i + 1) * mb]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(input_ids=x).logits
        lg = logits[:, :-1] if logits.dtype == torch.float64 else logits[:, :-1].float()
        loss = F.cross_entropy(lg.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        (loss / n_micro).backward()
        tot += float(loss.detach()) / n_micro
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return {"step": step, "lr": lr, "loss": tot, "grad_norm": gn}


def _torch(seed_dev: bool = True):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False      # float32 matmuls stay float32; bf16 is autocast's
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device")
    return torch, torch.device("cuda")


def _jdump(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(path)


def train(a) -> int:
    """One arm from step 0 (or its last resume state) to ``--until``, saving the grid checkpoints,
    a resume state every ``--resume-every`` updates and at every grid step, and the optimizer at 4369."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = _torch()
    import transformers
    root, arm = Path(a.root), a.arm
    if a.until not in CKPT_STEPS + CONT_STEPS:
        raise T5hRefused(f"--until {a.until} is not a checkpoint step")
    d = root / arm
    (d / "ckpt").mkdir(parents=True, exist_ok=True)
    sh = open_shard(root)
    dcfg = dodeca_cfg(a.hf_roots)
    model = build_arm(arm, dcfg, device)
    opt = make_opt(model)
    state, start = d / "resume.pt", 0
    if state.exists():
        st = torch.load(state, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = int(st["step"])
        if st["code"] != code_sha() and not a.allow_code_change:
            raise T5hRefused(f"resume state from code {st['code']}, now {code_sha()}")
    if (a.n_micro, a.compile) != (N_MICRO, COMPILE) and not a.allow_config_change:
        raise T5hRefused(f"configuration (n_micro {a.n_micro}, compile {a.compile}) is not (t12)'s "
                         f"({N_MICRO}, {COMPILE})")
    meta_p = d / "meta.json"
    if meta_p.exists():
        m0 = json.loads(meta_p.read_text())
        if (m0["n_micro"], m0["compile"]) != (a.n_micro, a.compile):
            raise T5hRefused(f"{arm} began with n_micro {m0['n_micro']}, compile {m0['compile']}; "
                             f"a resumed segment may not change them")
    else:
        _jdump(meta_p, {"design": "trackT_timaeus/design-T5.md", "arm": arm, "init": ARMS[arm][0],
                        "seed": ARMS[arm][1], "biases": ARMS[arm][2], "code": code_sha(), "n_micro": a.n_micro,
                        "compile": a.compile, "attn": "sdpa", "amp": "bf16 autocast, float32 master",
                        "tf32": False, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "transformers": transformers.__version__, "shard": json.loads(
                            (root / "data" / "meta.json").read_text())["main"]["sha256"],
                        "recipe": {"peak": PEAK, "warmup": WARMUP, "total": TOTAL, "betas": BETAS, "eps": EPS,
                                   "wd": WD, "clip": CLIP, "tokens_step": TOKENS_STEP}})
    if start == 0 and not (d / "ckpt" / "step0.safetensors").exists():
        save_weights(model, d / "ckpt" / "step0.safetensors")
    fwd = torch.compile(model) if a.compile else model
    log = (d / "train_log.jsonl").open("a")
    log.write(json.dumps({"segment": {"from": start, "until": a.until, "code": code_sha(), "n_micro": a.n_micro,
                                      "compile": a.compile, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}}) + "\n")
    t0, last_save = time.time(), start
    def save_state(s):
        tmp = state.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "code": code_sha()}, tmp)
        tmp.replace(state)
    for s in range(start + 1, a.until + 1):
        x = torch.from_numpy(batch_of(sh, s).astype(np.int64)).to(device, non_blocking=True)
        r = step_once(fwd, opt, x, s, a.n_micro)
        if not math.isfinite(r["loss"]):
            raise T5hRefused(f"{arm} step {s}: loss {r['loss']}")
        r["wall_s"] = round(time.time() - t0, 2)
        log.write(json.dumps(r) + "\n")
        log.flush()
        if s in CKPT_STEPS or s in CONT_STEPS:
            save_weights(model, d / "ckpt" / f"step{s}.safetensors")
        if s == READ_STEP:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "code": code_sha()},
                       d / f"state{s}.pt")
        if s in CKPT_STEPS or s in CONT_STEPS or s - last_save >= a.resume_every or s == a.until:
            save_state(s)
            last_save = s
        if s % 50 == 0 or s == a.until:
            print(json.dumps({"arm": arm, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()},
                              "tok_s": round((s - start) * TOKENS_STEP / (time.time() - t0))}), flush=True)
    log.close()
    print(json.dumps({"done": arm, "until": a.until, "wall_s": round(time.time() - t0)}))
    return 0


def theta(a) -> int:
    """θ on A0's configuration: the shard's first 60 updates, 10 warm-up then 50 timed (the run discarded)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = _torch()
    sh = open_shard(Path(a.root))
    model = build_arm("A0", dodeca_cfg(a.hf_roots), device)
    opt = make_opt(model)
    fwd = torch.compile(model) if a.compile else model
    times, losses = [], []
    for s in range(1, THETA_WARM + THETA_TIMED + 1):
        x = torch.from_numpy(batch_of(sh, s).astype(np.int64)).to(device)
        torch.cuda.synchronize()
        t = time.time()
        r = step_once(fwd, opt, x, s, a.n_micro)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        losses.append(r["loss"])
    timed = times[THETA_WARM:]
    th = THETA_TIMED * TOKENS_STEP / sum(timed)
    rows = "six arms and every continuation" if th >= THETA_ALL else \
        "six arms; continuations as the cap allows" if th >= THETA_ARMS else "A0 alone; the rest back to the user"
    rec = {"theta_tokens_s": th, "row": rows, "n_micro": a.n_micro, "compile": a.compile,
           "step_s": {"median": float(np.median(timed)), "min": min(timed), "max": max(timed)},
           "warm_s": times[:THETA_WARM], "loss_first_last": [losses[0], losses[-1]],
           "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "code": code_sha(),
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "hours_A0_to_4369": READ_STEP * TOKENS_STEP / th / 3600}
    if a.out:
        _jdump(Path(a.out), rec)
    print(json.dumps(rec))
    return 0


# ---------------------------------------------------------------- the reader

def read_one(model, tok, device) -> dict:
    from trackT_timaeus import t5c
    rec = t5c.read_model(model, tok, device)
    conv = t2.convention(model, tok, device)
    if not conv["pass"]:
        raise T5hRefused(f"convention check failed: {conv}")
    rec["v1_loss_per_prompt"], rec["v1_loss_mean"] = conv["loss_per_prompt"], conv["loss_mean"]
    return rec


def read(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from trackC_writeup import c2_aniso as c
    torch, device = c._torch_setup()
    if prompts_hash() != t5.PROMPTS_HASH:
        raise T5hRefused(f"prompts_hash {prompts_hash()} != {t5.PROMPTS_HASH}")
    d = Path(a.root) / a.arm
    dcfg = dodeca_cfg(a.hf_roots)
    tok = t2.tokenizer(SHAPE)
    t0 = time.time()
    for s in CKPT_STEPS + CONT_STEPS:
        ck = d / "ckpt" / f"step{s}.safetensors"
        p = d / "read" / f"step{s}.json"
        if not ck.exists() or (p.exists() and not a.force):
            continue
        m = load_weights(ck, a.arm, dcfg, device)
        rec = read_one(m, tok, device)
        if s == 0:
            rec["class_sds"] = t5.class_sds(m)
        rec.update({"arm": a.arm, "step": s, "code": code_sha(), "ckpt_sha256": hashlib.sha256(
            ck.read_bytes()).hexdigest()[:16]})
        _jdump(p, rec)
        print(json.dumps({"read": f"{a.arm}/step{s}", "rho_mid": round(rec["rho_mid"], 4),
                          "v1_loss": round(rec["v1_loss_mean"], 3), "t": round(time.time() - t0)}), flush=True)
        del m
        torch.cuda.empty_cache()
    return 0


def known(a) -> int:
    """Released `-32m` steps 1 and 9536 through save_weights → load_weights → read_one, against U2."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    if prompts_hash() != t5.PROMPTS_HASH:
        raise T5hRefused(f"prompts_hash {prompts_hash()} != {t5.PROMPTS_HASH}")
    from trackC_writeup import c2_aniso as c
    torch, device = c._torch_setup()
    dcfg = dodeca_cfg(a.hf_roots)
    tok = t2.tokenizer(SHAPE)
    u2 = Path(os.environ.get("METS_DATA", str(DATA))) / U2_DIR / "dodecahedron-32m"
    tmp = Path(a.root) / "known"
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in (1, 917, 4369, 9536):
        m0 = t5.load_real(t2.MODELS[SHAPE], f"step{s}", a.hf_roots, device)
        save_weights(m0, tmp / f"step{s}.safetensors")
        del m0
        m = load_weights(tmp / f"step{s}.safetensors", "A0", dcfg, device)
        rec = read_one(m, tok, device)
        ref = json.loads((u2 / f"step{s}.json").read_text())
        diff = max(abs(x - y) for k in ("rho", "rho_no0") for x, y in zip(rec[k], ref[k]))
        out[s] = {"max_abs_diff": diff, "rho_mid": rec["rho_mid"], "u2_rho_mid": ref["rho_mid"],
                  "v1_loss_mean": rec["v1_loss_mean"], "gated": s in (1, 9536)}
        print(json.dumps({"step": s, **out[s]}), flush=True)
    ok = all(v["max_abs_diff"] <= KNOWN_TOL for v in out.values() if v["gated"])
    _jdump(tmp / "known.json", {"steps": out, "tol": KNOWN_TOL, "pass": ok, "code": code_sha(),
                                "prompts_hash": prompts_hash()})
    print(json.dumps({"known_pass": ok}))
    return 0 if ok else 3


def floor(a) -> int:
    """ρ_mid over `pythia-410m` seeds 0–9 at 4000 and 143000 (U2's reader), and the gate: 2·SD and the
    largest pair difference ≤ 0.2 at both steps (Blocked 133 (i) (d))."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    if prompts_hash() != t5.PROMPTS_HASH:
        raise T5hRefused(f"prompts_hash {prompts_hash()} != {t5.PROMPTS_HASH}")
    from trackC_writeup import c2_aniso as c
    from trackT_timaeus import t5c
    torch, device = c._torch_setup()
    tok = t2.p_tokenizer()
    out = Path(a.root) / "floor"
    vals: Dict[int, List[float]] = {s: [] for s in FLOOR_STEPS}
    for k in FLOOR_SEEDS:
        repo = "EleutherAI/pythia-410m" + (f"-seed{k}" if k else "")
        for s in FLOOR_STEPS:
            p = out / f"seed{k}_step{s}.json"
            if not p.exists():
                m = t5.load_real(repo, f"step{s}", a.hf_roots, device)
                rec = t5c.read_model(m, tok, device)
                rec.update({"model": repo, "step": s, "commit": t2.commit_of(m), "code": code_sha(),
                            "prompts_hash": prompts_hash()})
                _jdump(p, rec)
                del m
                torch.cuda.empty_cache()
            rec = json.loads(p.read_text())
            vals[s].append(rec["rho_mid"])
            print(json.dumps({"seed": k, "step": s, "rho_mid": round(rec["rho_mid"], 4)}), flush=True)
    u2 = Path(os.environ.get("METS_DATA", str(DATA))) / U2_DIR / "pythia-410m"
    seed0 = {s: abs(json.loads((u2 / f"step{s}.json").read_text())["rho_mid"] - vals[s][0]) for s in FLOOR_STEPS}
    sd = {s: float(np.std(v, ddof=1)) for s, v in vals.items()}
    res = {"rho_mid": {str(s): v for s, v in vals.items()}, "sd": {str(s): v for s, v in sd.items()},
           "two_sd": {str(s): 2 * v for s, v in sd.items()}, "max": FLOOR_MAX,
           "pass": all(2 * v <= FLOOR_MAX for v in sd.values())
           and all(max(v) - min(v) <= FLOOR_MAX for v in vals.values()),
           "seed0_vs_u2_abs_diff": {str(s): v for s, v in seed0.items()}, "code": code_sha(),
           "max_pair_diff": {str(s): float(max(v) - min(v)) for s, v in vals.items()}}
    _jdump(out / "floor.json", res)
    print(json.dumps(res))
    return 0 if res["pass"] else 3


# ---------------------------------------------------------------- the gate (U4a's stages)

def records(root: Path, arm: str) -> Dict[int, dict]:
    d = Path(root) / arm / "read"
    return {int(p.stem[4:]): json.loads(p.read_text()) for p in d.glob("step*.json")}


def train_log(root: Path, arm: str) -> Dict[int, dict]:
    out = {}
    for line in (Path(root) / arm / "train_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "segment" in r:
            continue
        out[r["step"]] = r                          # a resumed run's later line for a step wins
    return out


def gate_start(recs: Dict[int, dict], log: Dict[int, dict], arm: str) -> dict:
    init = ARMS[arm][0]
    g = {}
    sds = recs[0]["class_sds"]
    d, L = t5.SHAPES[SHAPE][:2]
    g["init_sd"] = {k: v / t5.init_sd(k, init, d, L) - 1 for k, v in sds.items()}
    g["init_sd_pass"] = all(abs(v) <= SD_TOL for v in g["init_sd"].values())
    lo, hi = BAND[init]
    g["step0_rho_mid"], g["band"] = recs[0]["rho_mid"], [lo, hi]
    g["step0_pass"] = lo <= recs[0]["rho_mid"] <= hi
    g["first_output_pass"] = all(s in recs and np.all(np.isfinite(recs[s]["rho"])) for s in (1, 10))
    early, late = (float(np.mean([log[s]["loss"] for s in r])) for r in LOSS_FALL)
    g["loss_1_10"], g["loss_91_100"] = early, late
    g["loss_falls_pass"] = late < early
    g["pass"] = all(g[k] for k in ("init_sd_pass", "step0_pass", "first_output_pass", "loss_falls_pass"))
    return g


def gate_trough(series: Dict[int, float]) -> dict:
    need = [s for s in CKPT_STEPS if T_FROM <= s <= TROUGH_BY]
    miss = [s for s in need if s not in series]
    if miss:
        raise T5hRefused(f"trough stage: steps {miss} unread")
    t, ts = trough(series, TROUGH_BY)
    return {"T_by_1253": t, "at": ts, "max": TROUGH_MAX, "pass": t <= TROUGH_MAX}


def gate_late(series: Dict[int, float]) -> dict:
    l = late_level(series)
    t, ts = trough(series)
    return {"L": l, "T": t, "T_step": ts, "R": l - t, "min": LATE_MIN, "pass": l >= LATE_MIN}


def gate(a) -> int:
    root = Path(a.root)
    recs = records(root, a.arm)
    log = train_log(root, a.arm)
    series = {s: r["rho_mid"] for s, r in recs.items()}
    res = {"arm": a.arm, "stage": a.stage, "code": code_sha()}
    res["start"] = gate_start(recs, log, a.arm)
    ok = res["start"]["pass"]
    if a.stage in ("trough", "late"):
        res["trough"] = gate_trough(series)
        ok = ok and res["trough"]["pass"]
    if a.stage == "late":
        res["late"] = gate_late(series)
        ok = ok and res["late"]["pass"]
    res["pass"] = bool(ok)
    res["series"] = {str(s): series[s] for s in sorted(series)}
    res["v1_loss"] = {str(s): recs[s]["v1_loss_mean"] for s in sorted(recs)}
    _jdump(root / a.arm / f"gate_{a.stage}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("series", "v1_loss")}))
    return 0 if ok else 3


# ---------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="t5h")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--root", default=str(ROOT))
        p.add_argument("--hf-roots", nargs="+", default=list(t5.DEFAULT_ROOTS))
        return p
    p = common(sub.add_parser("shard"))
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--tmp-dir", default=str(DATA / "trackT"))
    p = common(sub.add_parser("theta"))
    p.add_argument("--n-micro", type=int, default=N_MICRO)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--out", default=None)
    p = common(sub.add_parser("train"))
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--until", type=int, required=True)
    p.add_argument("--n-micro", type=int, default=N_MICRO)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument("--resume-every", type=int, default=200)
    p.add_argument("--allow-code-change", action="store_true")
    p = common(sub.add_parser("read"))
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--force", action="store_true")
    common(sub.add_parser("known"))
    common(sub.add_parser("floor"))
    p = common(sub.add_parser("gate"))
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--stage", required=True, choices=("start", "trough", "late"))
    a = ap.parse_args(argv)
    return {"shard": shard, "theta": theta, "train": train, "read": read, "known": known, "floor": floor,
            "gate": gate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
