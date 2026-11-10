"""
trackC_writeup/c4_train.py — C4 U1: the harness that trains `pythia-70m` from seed 0's init on
Pythia's data order (`design-C4.md`, frozen on Blocked 155 all (a)).

Two arms, the same init, data, optimizer and dense checkpoint grid; they differ only before W:
- **A0**, Pythia's own schedule: NeoX `AnnealingLR` (`c6_schedule.lr`): iteration i (0-based) runs
  at 1e-3 · i / 1430 to i = 1430, then cosine to 1e-4 at 143 000. Iteration 0 runs at rate 0 (so
  released step 1 equals step 0) but still enters Adam's moments, as NeoX's does.
- **F**, flat: 1e-3 from iteration 0 to 1430, then the same cosine.

Checkpoint t holds iterations 0 … t − 1; iteration i reads rows [1024 i, 1024 (i + 1)) of
`EleutherAI/pile-standard-pythia-preshuffled` (the batch `batch_viewer.py` gives for step i), each
row 2049 tokens: inputs its first 2048, labels its last 2048 (NeoX's packing; no loss mask, no
attention reset). The rows are read through `document.idx`'s pointers, and the first 4000 × 1024
rows are checked to be 2049 long and contiguous in shard 0.

Recipe, from `data/trackC/configs/pythia-70m.yml` (gpt-neox's 70M config): Adam β (0.9, 0.95),
ε 1e-8, decoupled weight decay **0.1** on every parameter but biases and norms (NeoX's
`get_params_for_weight_decay_optimization`; `design-C4.md` said 0.01, corrected in U1 from the
config), global-norm clip 1.0, batch 1024 × 2048 predictions, dropout 0. Precision: bf16
autocast over float32 master weights, tf32 off (Pythia ran fp16 with loss scaling, so this is
not bit-equal; H3, H4 and R1 are the guard). Attention `sdpa` in training; the reader loads eager.

Parts (commands):
    verify  the shard and index against the Hub's LFS hashes; the first rows' lengths and pointers (H2)
    theta   tokens/s on A0's configuration: 3 warm-up updates, then 5 timed (the run discarded)
    train   one arm, resumable (`<root>/<arm>/`), saving the dense grid (HF format, float32)
    h1      the saved step 0 against the released step 0, tensor by tensor (bit-equal)
    h3      the sign agreement of the zero-initialised biases' increments against the released
            run at steps 2 and 4 (≥ 0.95); beside, 8 and 16, and every parameter's step-2 sign
    h4      the loss on 64 held-out rows (never trained on to 4000) against the released run's

Tier 1 until the user registers the reading (D4 (a)); nothing here reads a count.
    python -m trackC_writeup.c4_train verify
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_train theta
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_train train --arm A0 --until 1000
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_train h3 --arm A0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO

# ---------------------------------------------------------------- the recipe (design-C4.md, fixed)
PEAK, MIN_LR, W, DECAY = 1e-3, 1e-4, 1430, 143000        # gpt-neox configs/pythia/70M.yml
BETAS, EPS, WD, CLIP = (0.9, 0.95), 1e-8, 0.1, 1.0       # WD from the config (the design's 0.01 was wrong)
ROW = 2049                                               # tokens a row: 2048 inputs, 2048 labels
BATCH = 1024                                             # rows an iteration
TOKENS_STEP = BATCH * (ROW - 1)                          # 2,097,152 predictions an update
HORIZON = 4000
#: the dense grid, shared by both arms (design "The arms"); Pythia's grid to 4000 is a subset
GRID = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768,
        1000, 1250, 1500, 2000, 3000, 4000)
OFF_GRID = (3500,)                                       # V's gate step on F (design "V's first check")
ARMS = ("A0", "F")
BASE = "EleutherAI/pythia-70m"
INIT_REV = "step0"
# the data (design "Cost and data")
PILE = "EleutherAI/pile-standard-pythia-preshuffled"
PILE_REV = "bac79b6820adb34e451f9a02cc1dc7cd920febf0"
SHARD, IDX = "document-00000-of-00020.bin", "document.idx"
HDD = Path("/run/media/system/HDD_1TB/mets_data")
ROOT = HDD / "c4"
N_ROWS = HORIZON * BATCH                                 # rows read by either arm to 4000
HELD_OUT = (N_ROWS, N_ROWS + 64)                         # H4's rows: iteration 4000's first 64
# the harness's bars (design "The harness's known answers")
H3_BAR = 0.95
H4_TOL = 0.02                                            # placed (fp16 against bf16)
DIVERGE = (2.0, 50)                                      # loss > 2× the running median for 50 steps


class C4Refused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4_train.py",):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the schedules (pure)

def lr_pythia(i: int) -> float:
    """NeoX `AnnealingLR` at iteration ``i`` (0-based), Pythia-70M's: linear from 0 over W, then
    cosine from the peak to MIN_LR at DECAY (`c6_schedule.lr`, restated so this file is whole)."""
    if i < 0:
        raise C4Refused(f"iteration {i}")
    if i <= W:
        return PEAK * i / W
    return MIN_LR + (PEAK - MIN_LR) / 2 * (math.cos(math.pi * (i - W) / (DECAY - W)) + 1)


def lr_flat(i: int) -> float:
    """F: the peak from iteration 0 to W, then Pythia's cosine (equal to A0's from W on)."""
    if i < 0:
        raise C4Refused(f"iteration {i}")
    return PEAK if i <= W else lr_pythia(i)


SCHEDULES = {"A0": lr_pythia, "F": lr_flat}


def summed(arm: str, t: int) -> float:
    """The summed rate over iterations 0 … t − 1 (to checkpoint t)."""
    return float(sum(SCHEDULES[arm](i) for i in range(t)))


def decays(name: str, p) -> bool:
    """Decay on weights only: not on biases or layer norms (NeoX's grouping; every GPT-NeoX
    weight that is not a norm is 2-D)."""
    return p.ndim == 2 and "layernorm" not in name and not name.endswith(".bias")


def diverged(losses: Sequence[float], factor: float = DIVERGE[0], run: int = DIVERGE[1]) -> Optional[int]:
    """The first iteration ending ``run`` consecutive losses above ``factor`` × the running median
    of every loss before it (design "Readings": a refusal); None if none."""
    k = 0
    for i, x in enumerate(losses):
        if i == 0:
            continue
        med = float(np.median(losses[:i]))
        k = k + 1 if (not math.isfinite(x) or x > factor * med) else 0
        if k >= run:
            return i
    return None


# ---------------------------------------------------------------- the data

def read_idx_header(path: Path) -> Tuple[int, int, int]:
    """Megatron `MMapIndexedDataset` index: (dtype code, rows, docs)."""
    with path.open("rb") as f:
        magic = f.read(9)
        if magic != b"MMIDIDX\x00\x00":
            raise C4Refused(f"{path}: magic {magic!r}")
        version = int(np.frombuffer(f.read(8), "<u8")[0])
        code = int(np.frombuffer(f.read(1), "<u1")[0])
        n = int(np.frombuffer(f.read(8), "<u8")[0])
        docs = int(np.frombuffer(f.read(8), "<u8")[0])
    if version != 1 or code != 8:                         # 8: uint16 in Megatron's table
        raise C4Refused(f"{path}: version {version}, dtype code {code}")
    return code, n, docs


def idx_arrays(path: Path, rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """The first ``rows`` sizes and pointers."""
    _, n, _ = read_idx_header(path)
    off = 9 + 8 + 1 + 8 + 8
    sizes = np.memmap(path, dtype="<i4", mode="r", offset=off, shape=(n,))
    ptrs = np.memmap(path, dtype="<i8", mode="r", offset=off + 4 * n, shape=(n,))
    return np.asarray(sizes[:rows]), np.asarray(ptrs[:rows])


def check_rows(sizes: np.ndarray, ptrs: np.ndarray, shard_bytes: int) -> dict:
    """Every row ROW long and contiguous from byte 0 (so row r is bytes [2·ROW·r, 2·ROW·(r+1))),
    and inside shard 0."""
    r = np.arange(len(sizes), dtype=np.int64)
    ok_len = bool(np.all(sizes == ROW))
    ok_ptr = bool(np.all(ptrs == r * 2 * ROW))
    end = int(ptrs[-1]) + 2 * int(sizes[-1])
    if not (ok_len and ok_ptr and end <= shard_bytes):
        raise C4Refused(f"rows: lengths {ok_len}, contiguous {ok_ptr}, end {end} ≤ {shard_bytes}")
    return {"rows": int(len(sizes)), "all_2049": ok_len, "contiguous": ok_ptr, "end_byte": end}


def open_rows(root: Path = ROOT) -> np.ndarray:
    """Shard 0 as (rows, ROW) uint16, read only."""
    p = root / "pile" / SHARD
    n = p.stat().st_size // (2 * ROW)
    return np.memmap(p, dtype="<u2", mode="r", shape=(n, ROW))


def batch_of(rows: np.ndarray, i: int) -> np.ndarray:
    """Iteration ``i``'s 1024 rows (0-based; checkpoint i + 1 holds it)."""
    if not 0 <= i < HORIZON:
        raise C4Refused(f"iteration {i} outside [0, {HORIZON})")
    return np.asarray(rows[i * BATCH:(i + 1) * BATCH])


def lfs_sha256(fname: str) -> str:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(PILE, revision=PILE_REV, files_metadata=True)
    s = next(s for s in info.siblings if s.rfilename == fname)
    return s.lfs.sha256 if s.lfs else None


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def verify(a) -> int:
    """H2 and the download: both files' sha256 against the Hub's LFS record; the first N_ROWS +
    64 rows (training and H4's held-out) of the index are 2049 long and contiguous in shard 0."""
    root = Path(a.root)
    out = {"code": code_sha(), "repo": PILE, "revision": PILE_REV}
    for f in (IDX, SHARD):
        p = root / "pile" / f
        want = lfs_sha256(f)
        got = sha256_file(p)
        out[f] = {"bytes": p.stat().st_size, "sha256": got, "lfs_sha256": want, "equal": got == want}
        if got != want:
            raise C4Refused(f"{f}: sha256 {got} ≠ the Hub's {want}")
        print(json.dumps({f: out[f]}), flush=True)
    _, n, docs = read_idx_header(root / "pile" / IDX)
    sizes, ptrs = idx_arrays(root / "pile" / IDX, HELD_OUT[1])
    out["index"] = {"rows_total": n, "docs": docs,
                    **check_rows(sizes, ptrs, (root / "pile" / SHARD).stat().st_size)}
    rows = open_rows(root)
    b0 = batch_of(rows, 0)
    out["batch0"] = {"shape": list(b0.shape), "sha256": hashlib.sha256(b0.tobytes()).hexdigest(),
                     "first_tokens": b0[0, :16].tolist(), "max_token": int(b0.max())}
    if int(b0.max()) >= 50304:
        raise C4Refused(f"batch 0 has token {int(b0.max())} ≥ the vocabulary")
    _jdump(root / "data_meta.json", out)
    print(json.dumps({k: out[k] for k in ("index", "batch0")}), flush=True)
    return 0


# ---------------------------------------------------------------- the model and the loop

def _torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False      # float32 stays float32; bf16 is autocast's
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device")
    return torch, torch.device("cuda")


def build(device, attn: str = "sdpa"):
    """The released step 0, float32, train mode, dropout 0 (checked)."""
    import torch
    from transformers import GPTNeoXForCausalLM
    m = GPTNeoXForCausalLM.from_pretrained(BASE, revision=INIT_REV, torch_dtype=torch.float32,
                                           attn_implementation=attn).to(device)
    c = m.config
    if (getattr(c, "attention_dropout", 0.0), getattr(c, "hidden_dropout", 0.0)) != (0.0, 0.0):
        raise C4Refused(f"dropout {c.attention_dropout}, {c.hidden_dropout}")
    if (c.num_hidden_layers, c.hidden_size, c.num_attention_heads, c.rotary_pct, c.use_parallel_residual,
            c.tie_word_embeddings, c.vocab_size) != (6, 512, 8, 0.25, True, False, 50304):
        raise C4Refused("not Pythia-70M's config")
    return m.train()


def make_opt(model):
    import torch
    dec = [p for n, p in model.named_parameters() if decays(n, p)]
    nod = [p for n, p in model.named_parameters() if not decays(n, p)]
    return torch.optim.AdamW([{"params": dec, "weight_decay": WD}, {"params": nod, "weight_decay": 0.0}],
                             lr=0.0, betas=BETAS, eps=EPS, foreach=True)


def step_once(model, opt, rows, lr: float, n_micro: int, amp: bool = True) -> dict:
    """One update on ``rows`` (B, ROW) int64 on the model's device: the mean next-token loss over
    each row's 2048 predictions, averaged through ``n_micro`` equal micro-batches; global-norm
    clip at CLIP; AdamW at ``lr`` (an update at lr 0 still moves Adam's moments)."""
    import torch
    import torch.nn.functional as F
    for g in opt.param_groups:
        g["lr"] = lr
    B = rows.shape[0]
    if B % n_micro:
        raise C4Refused(f"batch {B} not divisible into {n_micro}")
    mb, tot = B // n_micro, 0.0
    for k in range(n_micro):
        r = rows[k * mb:(k + 1) * mb]
        x, y = r[:, :-1], r[:, 1:]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(input_ids=x).logits
        lg = logits if logits.dtype == torch.float64 else logits.float()
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
        (loss / n_micro).backward()
        tot += float(loss.detach()) / n_micro
    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return {"lr": lr, "loss": tot, "grad_norm": gn}


def _jdump(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    tmp.replace(path)


def ckpt_dir(root: Path, arm: str, step: int) -> Path:
    return root / arm / "ckpt" / f"step{step}"


def save_ckpt(model, d: Path) -> None:
    """HF format (config + float32 safetensors), so the reader loads it as it loads the Hub's."""
    tmp = d.with_name(d.name + ".tmp")
    model.save_pretrained(str(tmp), safe_serialization=True)
    if d.exists():
        raise C4Refused(f"{d} exists")
    tmp.rename(d)


SAVE_STEPS = tuple(sorted(set(GRID) | set(OFF_GRID)))


def train(a) -> int:
    """One arm from step 0 (or its resume state) to ``--until``: every SAVE_STEPS checkpoint, a
    resume state at each and every ``--resume-every`` updates; the per-update log."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = _torch()
    import transformers
    root, arm = Path(a.root), a.arm
    if arm not in ARMS:
        raise C4Refused(f"arm {arm}")
    if a.until not in SAVE_STEPS:
        raise C4Refused(f"--until {a.until} is not a checkpoint step")
    dm = root / "data_meta.json"
    if not dm.exists():
        raise C4Refused(f"{dm}: run `verify` first")
    d = root / arm
    (d / "ckpt").mkdir(parents=True, exist_ok=True)
    rows = open_rows(root)
    model = build(device)
    opt = make_opt(model)
    state, start = d / "resume.pt", 0
    if state.exists():
        st = torch.load(state, map_location=device, weights_only=False)
        if st["code"] != code_sha() and not a.allow_code_change:
            raise C4Refused(f"resume state from code {st['code']}, now {code_sha()}")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start = int(st["step"])
    if a.until <= start:
        print(json.dumps({"arm": arm, "until": a.until, "resume_step": start, "nothing_to_train": True}))
        return 0
    meta_p = d / "meta.json"
    if meta_p.exists():
        m0 = json.loads(meta_p.read_text())
        if m0["n_micro"] != a.n_micro or m0["compile"] != a.compile:
            raise C4Refused(f"{arm} began with n_micro {m0['n_micro']}, compile {m0['compile']}")
    else:
        _jdump(meta_p, {"design": "trackC_writeup/design-C4.md", "arm": arm, "code": code_sha(),
                        "n_micro": a.n_micro, "compile": a.compile, "attn": "sdpa",
                        "amp": "bf16 autocast, float32 master", "tf32": False,
                        "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "transformers": transformers.__version__, "init": [BASE, INIT_REV],
                        "data": json.loads(dm.read_text())["batch0"]["sha256"],
                        "recipe": {"peak": PEAK, "min_lr": MIN_LR, "warmup": W, "decay": DECAY,
                                   "schedule": SCHEDULES[arm].__name__, "betas": BETAS, "eps": EPS,
                                   "wd": WD, "clip": CLIP, "batch": BATCH, "row": ROW}})
    if start == 0 and not ckpt_dir(root, arm, 0).exists():
        save_ckpt(model, ckpt_dir(root, arm, 0))
    fwd = torch.compile(model) if a.compile else model
    log = (d / "train_log.jsonl").open("a")
    log.write(json.dumps({"segment": {"from": start, "until": a.until, "code": code_sha(),
                                      "started": time.strftime("%Y-%m-%dT%H:%M:%S")}}) + "\n")
    t0, last = time.time(), start
    def save_state(s):
        tmp = state.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "code": code_sha()}, tmp)
        tmp.replace(state)
    for s in range(start + 1, a.until + 1):                 # update s is iteration s − 1
        i = s - 1
        x = torch.from_numpy(batch_of(rows, i).astype(np.int64)).to(device, non_blocking=True)
        r = step_once(fwd, opt, x, SCHEDULES[arm](i), a.n_micro)
        if not math.isfinite(r["loss"]):
            raise C4Refused(f"{arm} step {s}: loss {r['loss']}")
        r = {"step": s, **r, "wall_s": round(time.time() - t0, 2)}
        log.write(json.dumps(r) + "\n")
        log.flush()
        if s in SAVE_STEPS:
            save_ckpt(model, ckpt_dir(root, arm, s))
        if s in SAVE_STEPS or s - last >= a.resume_every or s == a.until:
            save_state(s)
            last = s
        if s % 10 == 0 or s == a.until or s <= 4:
            print(json.dumps({"arm": arm, "step": s, "lr": r["lr"], "loss": round(r["loss"], 5),
                              "grad_norm": round(r["grad_norm"], 4),
                              "tok_s": round((s - start) * TOKENS_STEP / (time.time() - t0))}), flush=True)
    log.close()
    print(json.dumps({"done": arm, "until": a.until, "wall_s": round(time.time() - t0)}), flush=True)
    return 0


def theta(a) -> int:
    """Throughput on A0's configuration: the first 8 iterations, 3 warm-up then 5 timed."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = _torch()
    rows = open_rows(Path(a.root)) if (Path(a.root) / "pile" / SHARD).exists() else None
    model = build(device)
    opt = make_opt(model)
    fwd = torch.compile(model) if a.compile else model
    g = np.random.default_rng(0)
    times, losses = [], []
    for i in range(8):
        b = batch_of(rows, i) if rows is not None else g.integers(0, 50277, (BATCH, ROW))
        x = torch.from_numpy(b.astype(np.int64)).to(device)
        torch.cuda.synchronize()
        t = time.time()
        r = step_once(fwd, opt, x, lr_pythia(i), a.n_micro)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        losses.append(r["loss"])
    tok_s = TOKENS_STEP * 5 / sum(times[3:])
    out = {"code": code_sha(), "n_micro": a.n_micro, "compile": a.compile, "real_data": rows is not None,
           "s_per_update": sum(times[3:]) / 5, "tok_s": tok_s, "losses": losses,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
           "hours_to_4000": HORIZON * sum(times[3:]) / 5 / 3600}
    print(json.dumps(out), flush=True)
    if a.out:
        _jdump(Path(a.out), out)
    return 0


# ---------------------------------------------------------------- the harness checks

def released_sd(step: int) -> Dict[str, np.ndarray]:
    """The released run's floating tensors at ``step`` as float32 (fp16 on the Hub, so exact)."""
    from trackC_writeup.c6_schedule import sd
    return sd(BASE, step)


def local_sd(root: Path, arm: str, step: int) -> Dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    return {k: v.astype(np.float32) for k, v in load_file(str(ckpt_dir(root, arm, step) / "model.safetensors")).items()}


def h1(a) -> int:
    """The saved step 0 against the released step 0: the same keys, every tensor bit-equal."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    rel, loc = released_sd(0), local_sd(Path(a.root), a.arm, 0)
    common = sorted(set(rel) & set(loc))
    only = {"released": sorted(set(rel) - set(loc)), "local": sorted(set(loc) - set(rel))}
    neq = [k for k in common if not np.array_equal(rel[k], loc[k])]
    out = {"arm": a.arm, "tensors": len(common), "not_equal": neq, "only": only,
           "pass": not neq and len(common) > 0}
    print(json.dumps(out), flush=True)
    _jdump(Path(a.root) / a.arm / "h1.json", out)
    return 0 if out["pass"] else 1


def sign_agree(da: np.ndarray, db: np.ndarray) -> dict:
    m = (da != 0) & (db != 0)
    return {"n_both": int(m.sum()), "n": int(da.size),
            "agree": float(np.mean(np.sign(da[m]) == np.sign(db[m]))) if m.any() else None}


def h3_reading(rel: Dict[int, dict], loc: Dict[int, dict], steps: Sequence[int] = (2, 4, 8, 16)) -> dict:
    """Per step, the zero-initialised biases' increments from step 0 (released against local),
    sign agreement where both moved; every parameter's step-2 increment beside. Pass: ≥ H3_BAR
    at steps 2 and 4."""
    zb = [k for k, v in rel[0].items() if k.endswith(".bias") and "layernorm" not in k and not np.any(v)]
    keys = sorted(set(rel[0]) & set(loc[0]))
    out = {"zero_biases": len(zb), "steps": {}}
    for s in steps:
        dr = np.concatenate([(rel[s][k] - rel[0][k]).ravel() for k in zb])
        dl = np.concatenate([(loc[s][k] - loc[0][k]).ravel() for k in zb])
        out["steps"][str(s)] = sign_agree(dr, dl)
    dr = np.concatenate([(rel[2][k] - rel[0][k]).ravel() for k in keys])
    dl = np.concatenate([(loc[2][k] - loc[0][k]).ravel() for k in keys])
    out["all_params_step2"] = sign_agree(dr, dl)
    out["pass"] = all((out["steps"][str(s)]["agree"] or 0) >= H3_BAR for s in (2, 4))
    return out


def h3(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    root = Path(a.root)
    steps = (0, 2, 4, 8, 16)
    rel = {s: released_sd(s) for s in steps}
    loc = {s: local_sd(root, a.arm, s) for s in steps}
    out = {"arm": a.arm, "code": code_sha(), **h3_reading(rel, loc)}
    print(json.dumps(out), flush=True)
    _jdump(root / a.arm / "h3.json", out)
    return 0 if out["pass"] else 1


def held_loss(model, rows: np.ndarray, device) -> float:
    import torch
    import torch.nn.functional as F
    tot = 0.0
    x_all = torch.from_numpy(np.asarray(rows[HELD_OUT[0]:HELD_OUT[1]]).astype(np.int64)).to(device)
    with torch.no_grad():
        for k in range(0, len(x_all), 8):
            r = x_all[k:k + 8]
            lg = model(input_ids=r[:, :-1]).logits.float()
            tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), r[:, 1:].reshape(-1))) * len(r)
    return tot / len(x_all)


def h4(a) -> int:
    """Float32 eager loss on the 64 held-out rows, local against released, at ``--steps``."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = _torch()
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    root = Path(a.root)
    rows = open_rows(root)
    out = {"arm": a.arm, "code": code_sha(), "rows": list(HELD_OUT), "tol": H4_TOL, "steps": {}}
    for s in a.steps:
        rel = from_pretrained_eager(GPTNeoXForCausalLM, BASE, revision=f"step{s}",
                                    torch_dtype=torch.float32).to(device).eval()
        loc = from_pretrained_eager(GPTNeoXForCausalLM, str(ckpt_dir(root, a.arm, s)),
                                    torch_dtype=torch.float32).to(device).eval()
        lr_, ll = held_loss(rel, rows, device), held_loss(loc, rows, device)
        out["steps"][str(s)] = {"released": lr_, "local": ll, "rel_diff": (ll - lr_) / lr_,
                                "pass": abs(ll - lr_) / lr_ <= H4_TOL}
        del rel, loc
        torch.cuda.empty_cache()
    out["pass"] = all(v["pass"] for v in out["steps"].values())
    print(json.dumps(out), flush=True)
    _jdump(root / a.arm / f"h4_{'_'.join(map(str, a.steps))}.json", out)
    return 0 if out["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(q):
        q.add_argument("--root", default=str(ROOT))
    q = sub.add_parser("verify"); common(q)
    q = sub.add_parser("theta"); common(q)
    q.add_argument("--n-micro", type=int, default=512)
    q.add_argument("--no-compile", dest="compile", action="store_false")
    q.add_argument("--out", default=None)
    q = sub.add_parser("train"); common(q)
    q.add_argument("--arm", required=True, choices=ARMS)
    q.add_argument("--until", type=int, required=True)
    q.add_argument("--n-micro", type=int, default=512)
    q.add_argument("--no-compile", dest="compile", action="store_false")
    q.add_argument("--resume-every", type=int, default=25)
    q.add_argument("--allow-code-change", action="store_true")
    for name in ("h1", "h3"):
        q = sub.add_parser(name); common(q)
        q.add_argument("--arm", required=True, choices=ARMS)
    q = sub.add_parser("h4"); common(q)
    q.add_argument("--arm", required=True, choices=ARMS)
    q.add_argument("--steps", type=int, nargs="+", required=True)
    a = p.parse_args(argv)
    return {"verify": verify, "theta": theta, "train": train, "h1": h1, "h3": h3, "h4": h4}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
