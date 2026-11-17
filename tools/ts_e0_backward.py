"""E0's backward pass, with §9.5 Parked 1 beside (`MATH_TRAINING_SYMMETRY.md` §10; tier 1, description).

WHY THIS EXISTS
---------------
The last layer's non-rotary key bias (KBn) gets zero gradient in exact arithmetic (a softmax row's
gradient sums to zero over keys) and is not decayed, yet on Pythia it reaches 0.75 of Adam's top
speed with one sign (§8.4, §9.3). §10 (frozen at `2cc19b5` before any output) asks which graph
gives it a systematic gradient: float64 / float32 eager, float16 eager (HF's), float16 through the
FlashAttention-2 kernel (δ = rowsum(dO ∘ O) from the fp16 O; Qiu & Yao 2510.04212). Beside it, the
gradients at the init on iterations 0 and 1 predict Adam's step-2 update (Parked 1).

PARTS
-----
    grad  --model M --step S --arm {R64,R32,H16,F16} [--plant]   one backward pass, gradients kept
          per half of the batch (A = rows 0-511, B = 512-1023; R64: 32 + 32), for every layer's
          QKV bias and the last block's writer biases, plus the global norm
    init  --model M         R32 on iterations 0 and 1 at step 0 (Parked 1), and H16's overflow at 2^12
    read                    every label of §10 from the saved gradients and the released weights
    beside                  C4's A0 and F: the last layer's KBn rms over tau_1 (weights only)

    python tools/ts_e0_backward.py grad --model pythia-70m --step 256 --arm R32
Records: `<main>/data/ts/e0/`. Batches: Pythia's order, shard 0 (`c4_train.py`'s reader); step
256 reads iteration 256's rows, step 143000 iteration 4000's (§10), step 0 iterations 0 and 1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MAIN = Path("/run/media/system/WDS_500/Mets")
HUB = MAIN / "data/hf/hub"
OUT = MAIN / "data/ts/e0"
SHARD = Path("/run/media/system/HDD_1TB/mets_data/c4/pile/document-00000-of-00020.bin")
C4 = Path("/run/media/system/HDD_1TB/mets_data/c4")
ROW, BATCH, ROT = 2049, 1024, 16          # tokens a row; rows an iteration; rotary dims a head
EPS, BETAS, CLIP = 1e-8, (0.9, 0.95), 1.0
SCALE0 = 2.0 ** 12                         # DeepSpeed's initial loss scale (Pythia's configs)
BATCH_IT = {256: 256, 143000: 4000, 64: 64, 128: 128}  # step -> iteration whose rows are read (§10; 64, 128 post hoc, #356 finding 1)
MOTION = {("pythia-70m", 256): 512, ("pythia-410m", 256): 512,
          ("pythia-70m", 143000): 64000, ("pythia-410m", 143000): 120000,
          ("pythia-70m", 64): 128, ("pythia-70m", 128): 256}   # post hoc (#356 finding 1)
PEAK = {"pythia-70m": 1e-3, "pythia-410m": 3e-4}
WARM = 1430
MICRO = {("pythia-70m", "R64"): 1, ("pythia-70m", "R32"): 2, ("pythia-70m", "H16"): 4,
         ("pythia-70m", "F16"): 4, ("pythia-410m", "R32"): 1, ("pythia-410m", "H16"): 2,
         ("pythia-410m", "F16"): 2,
         ("pythia-70m", "FB16"): 4}      # FB16: post hoc, beside (§10.1; not in the frozen family)                 # rows a micro-batch (the logits bound it)
#: §10's budget rule: 410m's arms timed at ~100 min on 1024 rows (> 1.5 h), so 256 rows
ROWS = {"pythia-70m": BATCH, "pythia-410m": 256}
BAR_SYS, BAR_RAND, BAR_MOVE, G0_BAR, G1_BAR = 0.65, 0.6, 0.6, 1e-8, 0.95


def rows_of(it: int, n: int = BATCH) -> np.ndarray:
    m = np.memmap(SHARD, dtype="<u2", mode="r")
    m = m[: (m.size // ROW) * ROW].reshape(-1, ROW)
    return np.asarray(m[it * BATCH: it * BATCH + n]).astype(np.int64)


def snap(model: str, step: int) -> Path:
    d = HUB / f"models--EleutherAI--{model}"
    ref = (d / "refs" / f"step{step}").read_text().strip()
    return d / "snapshots" / ref


def tensors(model: str, step: int, names) -> dict:
    """Named tensors of a released checkpoint as float64 numpy (fp16 on the Hub, so exact)."""
    import torch
    p = snap(model, step)
    st = p / "model.safetensors"
    if st.exists():
        from safetensors import safe_open
        with safe_open(str(st), "pt") as f:
            return {n: f.get_tensor(n).double().numpy() for n in names}
    sd = torch.load(p / "pytorch_model.bin", map_location="cpu", weights_only=True)
    return {n: sd[n].double().numpy() for n in names}


def load(model: str, step: int, arm: str, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    dt = {"R64": torch.float64, "R32": torch.float32, "H16": torch.float16, "F16": torch.float16,
          "FB16": torch.bfloat16}[arm]
    attn = "sdpa" if arm in ("F16", "FB16") else "eager"
    m = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{model}", revision=f"step{step}",
                                           cache_dir=str(HUB), local_files_only=True,
                                           torch_dtype=dt, attn_implementation=attn).to(device)
    c = m.config
    assert (c.attention_dropout, c.hidden_dropout) == (0.0, 0.0) and c.rotary_pct == 0.25
    m.train()
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m


def kept(model_obj) -> dict:
    """Parameters whose gradients are saved: every QKV bias, the last block's writer biases."""
    L = model_obj.config.num_hidden_layers
    out = {}
    for n, p in model_obj.named_parameters():
        if n.endswith("attention.query_key_value.bias") or (
                f"layers.{L - 1}." in n and n.endswith(("attention.dense.bias", "mlp.dense_4h_to_h.bias"))):
            out[n] = p
    return out


def half_grads(m, rows: np.ndarray, arm: str, mb: int, scale: float, plant=None) -> dict:
    """Gradient of the mean CE over ``rows`` (float32 sums over micro-batches), the kept tensors and
    the global sum of squares. ``plant`` = (param, w, kappa) adds kappa <w, param[KBn]> to the loss."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    dev = next(m.parameters()).device
    params = [p for p in m.parameters()]
    acc = [torch.zeros_like(p, dtype=torch.float64 if arm == "R64" else torch.float32) for p in params]
    n = rows.shape[0] // mb
    nonfinite, loss_tot = False, 0.0
    def ctx():                                 # F16: the flash kernel or an error, never a fallback
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if arm in ("F16", "FB16") else sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    for k in range(n):
        r = torch.from_numpy(rows[k * mb:(k + 1) * mb]).to(dev)
        x, y = r[:, :-1], r[:, 1:]
        with ctx():
            lg = m(input_ids=x).logits
            lg = lg if lg.dtype == torch.float64 else lg.float()
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
            tot = loss / n
            if plant is not None:
                p, w, kap = plant
                kb = p.view(-1, 3, 64)[:, 1, ROT:]
                tot = tot + kap * (w.to(kb.dtype) * kb).sum() / n
            (tot * scale).backward()
        loss_tot += float(loss.detach()) / n
        for a, p in zip(acc, params):
            if p.grad is not None:
                g = p.grad.to(a.dtype)
                if not torch.isfinite(g).all():
                    nonfinite = True
                a.add_(g)
                p.grad = None
    for a in acc:
        a.div_(scale)
    names = [nm for nm, _ in m.named_parameters()]
    keep = set(kept(m))
    out = {nm: a.double().cpu().numpy() for nm, a in zip(names, acc) if nm in keep}
    ss = float(sum(float((a.double() ** 2).sum()) for a in acc))
    return {"grads": out, "sumsq": ss, "loss": loss_tot, "nonfinite": nonfinite, "acc": acc}


def run_grad(model: str, step: int, arm: str, plant: bool = False, rows=None, tag=None,
             zero_kbn: bool = False) -> Path:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    # HF 4.44's eager path builds its causal mask with torch.full in the default dtype
    torch.set_default_dtype(torch.float64 if arm == "R64" else torch.float32)
    t0 = time.time()
    if rows is None:
        n = 64 if arm == "R64" else ROWS.get(model, BATCH)
        rows = rows_of(BATCH_IT[step], n)
    halves = (rows[: len(rows) // 2], rows[len(rows) // 2:])
    m = load(model, step, arm, dev)
    if zero_kbn:                                 # post hoc (#356 finding 1): the last layer's KBn set to 0
        import torch as _t
        L = m.config.num_hidden_layers
        with _t.no_grad():
            dict(m.named_parameters())[f"gpt_neox.layers.{L - 1}.attention.query_key_value.bias"].view(-1, 3, 64)[:, 1, ROT:] = 0
    scale = SCALE0 if arm in ("H16", "F16") else 1.0
    pl = None
    if plant:
        L = m.config.num_hidden_layers
        p = dict(m.named_parameters())[f"gpt_neox.layers.{L - 1}.attention.query_key_value.bias"]
        H = m.config.num_attention_heads
        w = torch.from_numpy(np.random.default_rng(0).choice([-1.0, 1.0], size=(H, 64 - ROT))).to(dev)
        pl = (p, w, float(plant))
    res, scales = [], []
    for h in halves:
        s = scale
        while True:
            r = half_grads(m, h, arm, MICRO[(model, arm)], s, pl)
            if not r["nonfinite"] or s <= 1.0:
                break
            s /= 2.0                                 # DeepSpeed would skip and halve
        res.append(r)
        scales.append(s)
    full_sumsq = float(sum(float((((a + b) / 2).double() ** 2).sum()) for a, b in zip(res[0]["acc"], res[1]["acc"])))
    for r in res:
        del r["acc"]
    rec = {"model": model, "step": step, "arm": arm, "rows": int(len(rows)), "scales": scales,
           "nonfinite_at_first_scale": [r["nonfinite"] for r in res] if arm in ("H16", "F16") else None,
           "loss": [r["loss"] for r in res], "sumsq_halves": [r["sumsq"] for r in res],
           "global_norm": math.sqrt(full_sumsq),
           "wall_s": round(time.time() - t0, 1), "torch": torch.__version__,
           "plant": None if pl is None else float(pl[2]), "zero_kbn": zero_kbn}
    import transformers
    rec["transformers"] = transformers.__version__
    OUT.mkdir(parents=True, exist_ok=True)
    stem = tag or f"{model}_{step}_{arm}{'_plant' if plant else ''}{'_zkb' if zero_kbn else ''}"
    np.savez(OUT / f"{stem}.npz", **{f"A|{k}": v for k, v in res[0]["grads"].items()},
             **{f"B|{k}": v for k, v in res[1]["grads"].items()})
    (OUT / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("model", "step", "arm", "scales", "loss", "wall_s")}), flush=True)
    del m
    torch.cuda.empty_cache()
    return OUT / f"{stem}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("grad")
    g.add_argument("--model", required=True)
    g.add_argument("--step", type=int, required=True)
    g.add_argument("--arm", required=True, choices=("R64", "R32", "H16", "F16", "FB16"))
    g.add_argument("--plant", type=float, default=0.0)
    g.add_argument("--zero-kbn", action="store_true")
    i = sp.add_parser("init")
    i.add_argument("--model", required=True)
    sp.add_parser("read")
    sp.add_parser("beside")
    a = ap.parse_args(argv)
    if a.cmd == "grad":
        run_grad(a.model, a.step, a.arm, plant=a.plant or False, zero_kbn=a.zero_kbn)
    elif a.cmd == "init":
        for it in (0, 1):
            run_grad(a.model, 0, "R32", rows=rows_of(it, ROWS[a.model]), tag=f"{a.model}_0_R32_it{it}")
        run_grad(a.model, 0, "H16", rows=rows_of(0, ROWS[a.model]), tag=f"{a.model}_0_H16_it0")
    elif a.cmd == "read":
        from tools.ts_e0_read import read_all  # noqa: E402  (kept apart: no torch needed)
        read_all()
    elif a.cmd == "beside":
        from tools.ts_e0_read import beside
        beside()
    return 0


if __name__ == "__main__":
    sys.exit(main())
