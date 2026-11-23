"""
trackC_writeup/c4pdf_train.py — C4′ (d)+(f)'s arms (`design-C4pdf.md` "The arms"; frozen on
Blocked 172, 2026-11-23).

C4's recipe, called unchanged (`c4_train.build`, `make_opt`, `step_once`, `save_ckpt`,
`open_rows`, `batch_of`), so `c4_train`'s hash stays `1c1e1d2df083`. One thing changes per arm:

| arm | start | rate at iteration i | states saved |
|---|---|---|---|
| F1 | seed 0's init | 1e-3 (F's) | grid D ≤ 128 |
| Fh | seed 0's init | 0.5e-3 | grid D (to 1000) |
| Ft | seed 0's init | 0.1e-3 | grid D (to 1000) |
| BP | A0's step-256 weights, fresh AdamW | 1e-3 | 256 + offsets |
| BO | A0's step-256 weights, fresh AdamW | `lr_pythia(i)` | 256 + offsets |

Iteration i reads C4's batch i (`c4_train.batch_of`), so a branch's first update reads A0's
batch 256. States are saved at absolute steps (a branch's offset k is step 256 + k).

**Guard** (D3): Fh and Ft refuse to take an update until the d1 and d2 entries are registered or
declined (`registration`). F1 and the branches are tier 1 and are not guarded.

**H-d** (`hd`): F1's weights equal F's at every F state ≤ 128, bit for bit. **H-f** (`hf`): a
branch's state at 256 equals A0's step-256 file, and its batch k is A0's batch 256 + k.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from trackA_force.a1a import REPO
from trackC_writeup import c4_train as ct
from trackC_writeup import c4pdf_rules as rl

ROOT = ct.HDD / "c4pdf"
C4_ROOT = ct.ROOT
ARMS = ("F1", "Fh", "Ft", "BP", "BO")
FACTOR = {"F1": 1.0, "Fh": rl.RATES["Fh"], "Ft": rl.RATES["Ft"]}
UNTIL = {"F1": 128, "Fh": rl.HORIZON, "Ft": rl.HORIZON,
         "BP": rl.BRANCH_FROM + rl.OFFSETS[-1], "BO": rl.BRANCH_FROM + rl.OFFSETS[-1]}
GUARDED = ("Fh", "Ft")
REGISTRY = REPO / "claims/registry.json"
ENTRIES = REPO / "trackC_writeup/c4pdf_entries.json"   # the user's, committed: {"d1": id | "declined", "d2": …}
DIVERGE_EVERY = 50


class C4pdfRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4pdf_train.py", "trackC_writeup/c4pdf_rules.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def lr(arm: str, i: int) -> float:
    """The rate at iteration ``i`` (0-based; checkpoint i + 1 holds it)."""
    if arm in FACTOR:
        return FACTOR[arm] * ct.PEAK
    if arm == "BP":
        return ct.PEAK
    if arm == "BO":
        return ct.lr_pythia(i)
    raise C4pdfRefused(f"arm {arm}")


def start_step(arm: str) -> int:
    return rl.BRANCH_FROM if arm in ("BP", "BO") else 0


def save_steps(arm: str) -> tuple:
    if arm in ("BP", "BO"):
        return tuple(rl.BRANCH_FROM + k for k in rl.OFFSETS)
    return tuple(s for s in rl.GRID_D if s <= UNTIL[arm])


def committed(path: Path, repo: Path = REPO) -> bool:
    import subprocess
    rel = str(path.resolve().relative_to(repo.resolve()))
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel],
                           capture_output=True).returncode == 0
    return tracked and clean


def registration(entries: Path = ENTRIES, registry: Path = REGISTRY, check_git: bool = True) -> Dict[str, str]:
    """Per reading (d1, d2): 'registered as <id>' (the id in the registry), 'declined', or 'pending'.
    ``entries`` is the user's committed file naming each reading's id or "declined"."""
    if not entries.exists():
        return {"d1": "pending", "d2": "pending"}
    if check_git and not committed(entries):
        raise C4pdfRefused(f"{entries}: counts only when committed (tracked and clean)")
    e = json.loads(entries.read_text())
    ids = {p.get("id") for p in json.loads(registry.read_text()).get("predictions", [])}
    out = {}
    for k in ("d1", "d2"):
        v = e.get(k)
        out[k] = ("declined" if v == "declined" else f"registered as {v}" if v in ids else "pending")
    return out


def require_decided(**kw) -> Dict[str, str]:
    st = registration(**kw)
    pend = [k for k, v in st.items() if v == "pending"]
    if pend:
        raise C4pdfRefused(f"{pend} neither registered nor declined: no update on F½ or F⅒ before the user's decision")
    return st


def _load_a0_256(device):
    """C4's model (`c4_train.build`: sdpa, train mode, dropout 0) with A0's step-256 weights."""
    import torch
    m = ct.build(device)
    sd = {k: torch.from_numpy(v) for k, v in _sd(ct.ckpt_dir(C4_ROOT, "A0", rl.BRANCH_FROM)).items()}
    m.load_state_dict(sd, strict=True)          # every key, no extra (checked 2026-11-23)
    return m.train()


def train(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    root, arm = Path(a.root), a.arm
    if arm not in ARMS:
        raise C4pdfRefused(f"arm {arm}")
    st = require_decided() if arm in GUARDED else {"tier": 1}
    until = a.until or UNTIL[arm]
    saves = save_steps(arm)
    if until not in saves:
        raise C4pdfRefused(f"--until {until} is not a saved state of {arm}")
    torch, device = ct._torch()
    import transformers
    d = root / arm
    (d / "ckpt").mkdir(parents=True, exist_ok=True)
    rows = ct.open_rows(C4_ROOT)
    s0 = start_step(arm)
    model = _load_a0_256(device) if s0 else ct.build(device)
    opt = ct.make_opt(model)
    state, start = d / "resume.pt", s0
    codes = {"c4pdf_train": code_sha(), "c4_train": ct.code_sha(), "c4pdf_rules": rl.code_sha()}
    if state.exists():
        r0 = torch.load(state, map_location=device, weights_only=False)
        if r0["code"] != codes and not a.allow_code_change:
            raise C4pdfRefused(f"resume state from code {r0['code']}, now {codes}")
        model.load_state_dict(r0["model"])
        opt.load_state_dict(r0["opt"])
        start = int(r0["step"])
    if until <= start:
        print(json.dumps({"arm": arm, "until": until, "resume_step": start, "nothing_to_train": True}))
        return 0
    meta_p = d / "meta.json"
    if meta_p.exists():
        m0 = json.loads(meta_p.read_text())
        if m0["n_micro"] != a.n_micro or m0["compile"] != a.compile:
            raise C4pdfRefused(f"{arm} began with n_micro {m0['n_micro']}, compile {m0['compile']}")
    else:
        ct._jdump(meta_p, {"design": "trackC_writeup/design-C4pdf.md", "arm": arm, "code": codes,
                           "registration": st, "n_micro": a.n_micro, "compile": a.compile, "attn": "sdpa",
                           "amp": "bf16 autocast, float32 master", "tf32": False,
                           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "init": [ct.BASE, ct.INIT_REV] if not s0 else ["A0", rl.BRANCH_FROM, "fresh AdamW"],
                           "rate": {"factor": FACTOR.get(arm), "schedule": "flat" if arm != "BO" else "lr_pythia"},
                           "saves": list(saves)})
    if start == s0 and not ct.ckpt_dir(root, arm, s0).exists():
        ct.save_ckpt(model, ct.ckpt_dir(root, arm, s0))
    fwd = torch.compile(model) if a.compile else model
    log = (d / "train_log.jsonl").open("a")
    log.write(json.dumps({"segment": {"from": start, "until": until, "code": codes,
                                      "started": time.strftime("%Y-%m-%dT%H:%M:%S")}}) + "\n")
    t0, last, losses = time.time(), start, []

    def save_state(s):
        tmp = state.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "code": codes}, tmp)
        tmp.replace(state)
    for s in range(start + 1, until + 1):
        i = s - 1
        x = torch.from_numpy(ct.batch_of(rows, i).astype(np.int64)).to(device, non_blocking=True)
        r = ct.step_once(fwd, opt, x, lr(arm, i), a.n_micro)
        if not math.isfinite(r["loss"]):
            raise C4pdfRefused(f"{arm} step {s}: loss {r['loss']}")
        r = {"step": s, **r, "wall_s": round(time.time() - t0, 2)}
        log.write(json.dumps(r) + "\n")
        log.flush()
        losses.append(r["loss"])
        if s in saves:
            ct.save_ckpt(model, ct.ckpt_dir(root, arm, s))
        if s in saves or s - last >= a.resume_every or s == until:
            save_state(s)
            last = s
        if s % DIVERGE_EVERY == 0 or s == until:
            dv = ct.diverged(losses)
            if dv is not None:
                save_state(s)
                log.close()
                raise C4pdfRefused(f"{arm} diverged at update {dv} of this segment: stopped at {s}")
        if s % 10 == 0 or s == until or s - s0 <= 4:
            print(json.dumps({"arm": arm, "step": s, "lr": r["lr"], "loss": round(r["loss"], 5),
                              "grad_norm": round(r["grad_norm"], 4)}), flush=True)
    log.close()
    print(json.dumps({"done": arm, "until": until, "wall_s": round(time.time() - t0)}), flush=True)
    return 0


def _sd(path: Path) -> Dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    return load_file(str(path / "model.safetensors"))


def bit_equal(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> dict:
    if set(a) != set(b):
        return {"equal": False, "why": "keys differ"}
    diff = {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in a
            if not np.array_equal(a[k], b[k])}
    return {"equal": not diff, "n_tensors": len(a), "n_differ": len(diff),
            "max_abs": max(diff.values()) if diff else 0.0}


def hd(a) -> int:
    """H-d: F1 against F at every F state ≤ 128."""
    root = Path(a.root)
    steps = [s for s in ct.SAVE_STEPS if s <= UNTIL["F1"]]
    res = {str(s): bit_equal(_sd(ct.ckpt_dir(root, "F1", s)), _sd(ct.ckpt_dir(C4_ROOT, "F", s))) for s in steps}
    out = {"gate": "H-d", "pass": all(v["equal"] for v in res.values()), "per_step": res, "code": code_sha()}
    ct._jdump(Path(a.out) / "hd.json", out)
    print(json.dumps({"H-d": out["pass"], "n": len(res)}))
    return 0


def hf(a) -> int:
    """H-f: each branch's state 256 is A0's file; batch k is A0's batch 256 + k (row hashes)."""
    root = Path(a.root)
    a0 = _sd(ct.ckpt_dir(C4_ROOT, "A0", rl.BRANCH_FROM))
    res = {}
    for arm in ("BP", "BO"):
        p = ct.ckpt_dir(root, arm, rl.BRANCH_FROM)
        res[arm] = bit_equal(_sd(p), a0) if p.exists() else {"equal": None, "why": "not saved yet"}
    rows = ct.open_rows(C4_ROOT)
    h = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    batches = {str(k): h(ct.batch_of(rows, rl.BRANCH_FROM + k)) for k in (0, 1, 255)}
    out = {"gate": "H-f", "state": res, "batches_256_plus_k": batches,
           "pass": all(v["equal"] for v in res.values()), "code": code_sha()}
    ct._jdump(Path(a.out) / "hf.json", out)
    print(json.dumps({"H-f": out["pass"]}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="c4pdf_train")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("train")
    q.add_argument("--arm", required=True, choices=ARMS)
    q.add_argument("--root", default=str(ROOT))
    q.add_argument("--until", type=int, default=None)
    q.add_argument("--n-micro", type=int, default=512)
    q.add_argument("--no-compile", dest="compile", action="store_false")
    q.add_argument("--resume-every", type=int, default=25)
    q.add_argument("--allow-code-change", action="store_true")
    for name in ("hd", "hf"):
        q = sub.add_parser(name)
        q.add_argument("--root", default=str(ROOT))
        q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"train": train, "hd": hd, "hf": hf}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
