"""
trackC_writeup/c8_train.py — C8 U2: the three re-runs that save Adam's second moment
(`design-C8.md` "U2"; D1 (a), Blocked 182, the user's; Blocked 183 (a), the user's).

C4's recipe, called unchanged (`c4_train.build`, `make_opt`, `step_once`, `save_ckpt`,
`open_rows`, `batch_of`; `c4pdf_train._load_a0_256`; `c4pdf_bk._moments`), so the training code's
hashes stay those of F¹, B_P and B_K. One thing is added: at every saved state the optimiser's
``exp_avg_sq`` (v, not bias-corrected) and its step count are written beside the weights, so
`c8_sharp` can build Adam's preconditioner there.

| re-run | as | start | optimiser | rate | states saved |
|---|---|---|---|---|---|
| F2 | F¹ | seed 0's init (`c4_train.build`) | fresh AdamW | 1e-3 flat | grid D ≤ 128 (33) |
| BP2 | B_P | A0's step-256 weights | fresh AdamW | 1e-3 flat | 256 + offsets ≤ 128 (14) |
| BK2 | B_K | A0's step-256 weights | the rebuilt moments at 256 (`c4pdf_bk._moments`) | 1e-3 flat | the same (14) |

Iteration i reads C4's batch i, as the originals did. **These are new trajectories** (the harness
is not bit-reproducible past one update; `/challenge-pr` on #387, finding 4), so each is read
against its own rule-T onset, computed by `c8_sharp reference` before any λ.

``hcheck`` (before any λ): each re-run's first logged loss (before its first update, so a
function of the start state and batch only) equals its original's to every digit; BK2's loaded
optimiser is at step 256; each saved v file names its step.

    python -m trackC_writeup.c8_train train --arm F2
    python -m trackC_writeup.c8_train hcheck --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from trackA_force.a1a import REPO
from trackC_writeup import c4_train as ct
from trackC_writeup import c4pdf_bk as bk
from trackC_writeup import c4pdf_rules as rl
from trackC_writeup import c4pdf_train as pt

ROOT = Path("/run/media/system/WDS_500/Mets_hdd_storage/c8/u2")
ARMS = ("F2", "BP2", "BK2")
AS = {"F2": "F1", "BP2": "BP", "BK2": "BK"}                # the original each re-runs
ORIG = {"F2": pt.ROOT / "F1", "BP2": pt.ROOT / "BP", "BK2": pt.ROOT / "BK"}
HORIZON = 128                                              # updates from each start (design "U2")


class C8TrainRefused(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackC_writeup/c8_train.py").read_bytes()).hexdigest()[:12]


def start_step(arm: str) -> int:
    return 0 if arm == "F2" else rl.BRANCH_FROM


def save_steps(arm: str) -> tuple:
    if arm == "F2":
        return tuple(s for s in rl.GRID_D if s <= HORIZON)
    if arm in ("BP2", "BK2"):
        return tuple(rl.BRANCH_FROM + k for k in rl.OFFSETS if k <= HORIZON)
    raise C8TrainRefused(f"arm {arm}")


def v_path(root: Path, arm: str, step: int) -> Path:
    return root / arm / "moments" / f"step{step}.safetensors"


def save_v(model, opt, p: Path) -> int:
    """v (``exp_avg_sq``) per named parameter, and the optimiser's step, which every parameter shares."""
    import torch
    from safetensors.torch import save_file
    out, steps = {}, set()
    for n, prm in model.named_parameters():
        st = opt.state.get(prm, {})
        if "exp_avg_sq" not in st:
            return -1                                       # no update yet: nothing to save
        out[n] = st["exp_avg_sq"].detach().to("cpu", torch.float32).contiguous()
        steps.add(int(float(st["step"])))
    if len(steps) != 1:
        raise C8TrainRefused(f"optimiser steps differ across parameters: {sorted(steps)}")
    k = steps.pop()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    save_file(out, str(tmp), metadata={"opt_step": str(k), "betas": json.dumps(ct.BETAS), "eps": str(ct.EPS)})
    tmp.replace(p)
    return k


def train(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    root, arm = Path(a.root), a.arm
    if arm not in ARMS:
        raise C8TrainRefused(f"arm {arm}")
    torch, device = ct._torch()
    import transformers
    d = root / arm
    (d / "ckpt").mkdir(parents=True, exist_ok=True)
    rows = ct.open_rows(ct.ROOT)
    s0 = start_step(arm)
    model = ct.build(device) if arm == "F2" else pt._load_a0_256(device)
    opt = ct.make_opt(model)
    saves, until = save_steps(arm), s0 + HORIZON
    codes = {"c8_train": code_sha(), "c4_train": ct.code_sha(), "c4pdf_train": pt.code_sha(),
             "c4pdf_bk": bk.code_sha(), "c4pdf_rules": rl.code_sha()}
    state, start = d / "resume.pt", s0
    if state.exists():
        r0 = torch.load(state, map_location=device, weights_only=False)
        if r0["code"] != codes and not a.allow_code_change:
            raise C8TrainRefused(f"resume state from code {r0['code']}, now {codes}")
        model.load_state_dict(r0["model"])
        opt.load_state_dict(r0["opt"])
        start = int(r0["step"])
    elif arm == "BK2":
        opt.load_state_dict(bk._moments(device))
        if bk._opt_step(opt) != {rl.BRANCH_FROM}:
            raise C8TrainRefused(f"loaded optimiser at steps {bk._opt_step(opt)}")
    if until <= start:
        print(json.dumps({"arm": arm, "resume_step": start, "nothing_to_train": True}))
        return 0
    meta_p = d / "meta.json"
    if meta_p.exists():
        m0 = json.loads(meta_p.read_text())
        if m0["n_micro"] != a.n_micro or m0["compile"] != a.compile:
            raise C8TrainRefused(f"{arm} began with n_micro {m0['n_micro']}, compile {m0['compile']}")
    else:
        ct._jdump(meta_p, {"design": "trackC_writeup/design-C8.md U2", "arm": arm, "re_runs": AS[arm], "code": codes,
                           "n_micro": a.n_micro, "compile": a.compile, "attn": "sdpa",
                           "amp": "bf16 autocast, float32 master", "tf32": False,
                           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "init": ([ct.BASE, ct.INIT_REV] if arm == "F2" else
                                    ["A0", rl.BRANCH_FROM, "fresh AdamW" if arm == "BP2" else
                                     "moments from " + str(bk.REBUILD / "A0/resume.pt")]),
                           "rate": "peak flat", "saves": list(saves), "v_saved": "exp_avg_sq at every saved state"})
    if start == s0:
        if not ct.ckpt_dir(root, arm, s0).exists():
            ct.save_ckpt(model, ct.ckpt_dir(root, arm, s0))
        if arm == "BK2" and not v_path(root, arm, s0).exists():
            save_v(model, opt, v_path(root, arm, s0))
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
        r = ct.step_once(fwd, opt, x, ct.PEAK, a.n_micro)
        if not math.isfinite(r["loss"]):
            raise C8TrainRefused(f"{arm} step {s}: loss {r['loss']}")
        r = {"step": s, **r, "wall_s": round(time.time() - t0, 2)}
        log.write(json.dumps(r) + "\n")
        log.flush()
        losses.append(r["loss"])
        if s in saves:
            ct.save_ckpt(model, ct.ckpt_dir(root, arm, s))
            k = save_v(model, opt, v_path(root, arm, s))
            if k != (s if arm == "F2" else s - rl.BRANCH_FROM + (rl.BRANCH_FROM if arm == "BK2" else 0)):
                raise C8TrainRefused(f"{arm} state {s}: optimiser at step {k}")
        if s in saves or s - last >= a.resume_every or s == until:
            save_state(s)
            last = s
        if s % pt.DIVERGE_EVERY == 0 or s == until:
            dv = ct.diverged(losses)
            if dv is not None:
                save_state(s)
                log.close()
                raise C8TrainRefused(f"{arm} diverged at update {dv} of this segment: stopped at {s}")
        if s % 10 == 0 or s == until or s - s0 <= 4:
            print(json.dumps({"arm": arm, "step": s, "loss": round(r["loss"], 5),
                              "grad_norm": round(r["grad_norm"], 4)}), flush=True)
    log.close()
    print(json.dumps({"done": arm, "until": until, "wall_s": round(time.time() - t0)}), flush=True)
    return 0


def _first_loss(p: Path) -> tuple:
    for ln in p.read_text().splitlines():
        r = json.loads(ln)
        if "step" in r:
            return r["step"], r["loss"]
    raise C8TrainRefused(f"{p}: no update logged")


def hcheck(a) -> int:
    from safetensors import safe_open
    root = Path(a.root)
    res = {}
    for arm in ARMS:
        s1, l1 = _first_loss(root / arm / "train_log.jsonl")
        o1, lo = _first_loss(ORIG[arm] / "train_log.jsonl")
        ks = {}
        for s in save_steps(arm):
            p = v_path(root, arm, s)
            if p.exists():
                with safe_open(str(p), "np") as f:
                    ks[s] = int(f.metadata()["opt_step"])
        want = {s: (s if arm == "F2" else s - rl.BRANCH_FROM + (rl.BRANCH_FROM if arm == "BK2" else 0))
                for s in ks}
        res[arm] = {"first_step": s1, "first_loss": l1, "orig_first_step": o1, "orig_first_loss": lo,
                    "v_steps_ok": ks == want, "n_v": len(ks), "n_ckpt": sum(ct.ckpt_dir(root, arm, s).exists()
                                                                          for s in save_steps(arm))}
        res[arm]["pass"] = bool(s1 == o1 and l1 == lo and ks == want and res[arm]["n_ckpt"] == len(save_steps(arm)))
    out = {"gate": "H-rerun", "per_arm": res, "pass": all(v["pass"] for v in res.values()), "code": code_sha()}
    ct._jdump(Path(a.out) / "hrerun.json", out)
    print(json.dumps({"H-rerun": out["pass"], **{k: v["pass"] for k, v in res.items()}}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="c8_train")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("train")
    q.add_argument("--arm", required=True, choices=ARMS)
    q.add_argument("--root", default=str(ROOT))
    q.add_argument("--n-micro", type=int, default=512)
    q.add_argument("--no-compile", dest="compile", action="store_false")
    q.add_argument("--resume-every", type=int, default=25)
    q.add_argument("--allow-code-change", action="store_true")
    q = sub.add_parser("hcheck")
    q.add_argument("--root", default=str(ROOT))
    q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"train": train, "hcheck": hcheck}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
