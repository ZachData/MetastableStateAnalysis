"""
trackC_writeup/c4pdf_bk.py — C4′ (d)+(f) U3b: the kept-moment branch B_K (STATE Blocked 174 (a),
taken by Claude on the user's "continue on the next 5 items", 2026-11-24; `status-C.md`
"C4′ (d)+(f) unit 3b"). A new file, so `c4pdf_rules` (the hash the d1, d2 drafts name) and
`c4pdf_train` keep their hashes. Frozen, with the rule K below, before any output of the rebuild
or of B_K exists.

**Why.** f read "not the early state's": a fresh AdamW at the peak (B_P) makes F's transient on
A0's trained step-256 state, the same restart at A0's own rate (B_O) none. B_P jumps the rate
5.6× *and* restarts Adam's moments, so it cannot tell which does it (#371, finding 3).

**The arms** (both from A0's **saved** step-256 weights, the file B_P and B_O start from):

| arm | optimiser state | rate at iteration i | states |
|---|---|---|---|
| BK | A0's moments at 256, **rebuilt** (A0 re-trained 0 → 256 by `c4_train` itself, unchanged) | 1e-3 (the peak) | 256 + offsets ≤ 128 |
| BKO | the same rebuilt moments | `lr_pythia(i)` (A0's own) | the same |

B_K and B_P then differ only in the optimiser state. **BKO is the null B_K needs** (added beside
Blocked 174 (a), which named B_K alone): the moments are a near-replicate's (H-d: the harness is
not bit-reproducible past one update), laid on the saved weights. If that mismatch alone disturbs
the state, BKO shows it at A0's own rate and B_K is not read.

**The rebuild** (`rebuild-root`, then `c4_train train --arm A0 --until 256 --root <REBUILD>`):
A0's own code (hash `1c1e1d2df083`), own data (the `pile/` shard linked), own grid; its
`resume.pt` at 256 holds the model and the moments. `gap` records how far it is from A0's saved
files at every state both saved (≤ 256) and how far its losses are from A0's log, before B_K trains.

**Rule K** (`k_label`), on rule T (`c4pdf_rules.transient`, unchanged) at each gated τ arm:

| B_K | BKO | K reads |
|---|---|---|
| transient | none | **the jump's**: the jump to the peak makes the transient with the moments kept |
| none | none | **the restart's**: with the moments kept the jump makes none; B_P's needs the fresh moments' first updates |
| any | transient | **unreadable**: the rebuilt moments on the saved weights disturb the state at A0's own rate |

Joint over the gated τ arms: one row everywhere, else "mixed".

**Gate G-K** (on arms already read; it checks only that the horizon 384 does not hide the effect):
B_P's states ≤ 384 under rule T: a transient at every gated arm (effect); B_O's ≤ 384: none (null).
`read-k` refuses unless G-K passes, H-K passes and the f gate is the one f was read on.

**H-K**: each arm's saved k = 0 state equals A0's step-256 file bit for bit; the loaded optimiser
state is at step 256; step 257's loss (before the first update) equals A0's own log's to every digit.

**Beside, read never:** B_K's onset in updates against B_P's; the training loss of B_K, BKO and
B_P at every update 257–384.
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
from trackC_writeup import c4pdf_train as pt

REBUILD = pt.ROOT / "a0r"                 # c4_train's root for A0's rebuild (A0/ under it)
ARMS = ("BK", "BKO")
UNTIL = rl.BRANCH_FROM + 128              # 384, fixed in Blocked 174 (a)
SAVES = tuple(rl.BRANCH_FROM + k for k in rl.OFFSETS if k <= 128)
K_TABLE = {(True, False): "the jump's", (False, False): "the restart's",
           (True, True): "unreadable (BKO has a transient)", (False, True): "unreadable (BKO has a transient)"}


class C4pdfBKRefused(RuntimeError):
    pass


def code_sha() -> str:
    return hashlib.sha256((REPO / "trackC_writeup/c4pdf_bk.py").read_bytes()).hexdigest()[:12]


def lr(arm: str, i: int) -> float:
    if arm == "BK":
        return ct.PEAK
    if arm == "BKO":
        return ct.lr_pythia(i)
    raise C4pdfBKRefused(f"arm {arm}")


def save_steps(arm: str) -> tuple:
    if arm not in ARMS:
        raise C4pdfBKRefused(f"arm {arm}")
    return SAVES


# ---- the rule (pure) -------------------------------------------------------------------------

def k_label(tr_k: dict, tr_ko: dict) -> str:
    return K_TABLE[(bool(tr_k["present"]), bool(tr_ko["present"]))]


def k_joint(labels: Dict[str, str]) -> str:
    v = set(labels.values())
    return v.pop() if len(v) == 1 else "mixed"


def cut(mat: np.ndarray, states: Sequence[int], until: int = UNTIL):
    st = [s for s in states if s <= until]
    return np.asarray(mat)[:len(st)], st


def g_k(mats_bp: Dict[str, np.ndarray], st_bp, mats_bo: Dict[str, np.ndarray], st_bo, arms) -> dict:
    per = {}
    for k in arms:
        mp, sp = cut(mats_bp[k], st_bp)
        mo, so = cut(mats_bo[k], st_bo)
        per[k] = {"BP_to_384": rl.transient(mp, sp), "BO_to_384": rl.transient(mo, so)}
    ok = all(v["BP_to_384"]["present"] and not v["BO_to_384"]["present"] for v in per.values())
    return {"per_arm": per, "pass": bool(ok)}


# ---- the rebuild ------------------------------------------------------------------------------

def rebuild_root(a) -> int:
    """REBUILD/ with C4's shard linked and its data_meta.json copied (what `c4_train train` reads)."""
    REBUILD.mkdir(parents=True, exist_ok=True)
    pile = REBUILD / "pile"
    if not pile.exists():
        pile.symlink_to(ct.ROOT / "pile", target_is_directory=True)
    dm = REBUILD / "data_meta.json"
    if not dm.exists():
        dm.write_bytes((ct.ROOT / "data_meta.json").read_bytes())
    if (REBUILD / "A0" / "resume.pt").exists():
        raise C4pdfBKRefused(f"{REBUILD / 'A0'} already holds a rebuild")
    print(json.dumps({"root": str(REBUILD), "pile": str(pile.resolve()), "shard_ok": (pile / ct.SHARD).exists()}))
    return 0


def _log(path: Path) -> Dict[int, float]:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "step" in r:
            out[int(r["step"])] = float(r["loss"])
    return out


def gap(a) -> int:
    """The rebuild against A0's saved files (every state both saved, ≤ 256) and A0's losses."""
    rows = {}
    for s in (x for x in ct.SAVE_STEPS if x <= rl.BRANCH_FROM):
        p, q = ct.ckpt_dir(REBUILD, "A0", s), ct.ckpt_dir(ct.ROOT, "A0", s)
        if not p.exists():
            raise C4pdfBKRefused(f"{p}: missing")
        x, y = pt._sd(p), pt._sd(q)
        be = pt.bit_equal(x, y)
        num = math.sqrt(sum(float(np.sum((x[k].astype(np.float64) - y[k].astype(np.float64)) ** 2)) for k in x))
        den = math.sqrt(sum(float(np.sum(y[k].astype(np.float64) ** 2)) for k in y))
        rows[str(s)] = {**be, "rel_l2": num / den}
    la, lr_ = _log(ct.ROOT / "A0" / "train_log.jsonl"), _log(REBUILD / "A0" / "train_log.jsonl")
    steps = list(range(1, rl.BRANCH_FROM + 1))
    if any(s not in lr_ for s in steps):
        raise C4pdfBKRefused("the rebuild's log does not hold every update 1–256")
    d = [abs(lr_[s] - la[s]) for s in steps]
    first = next((s for s in steps if lr_[s] != la[s]), None)
    import torch
    st = torch.load(REBUILD / "A0" / "resume.pt", map_location="cpu", weights_only=False)
    out = {"rebuild": str(REBUILD / "A0"), "resume_step": int(st["step"]), "resume_code": st["code"],
           "c4_train": ct.code_sha(), "per_state": rows,
           "loss": {"first_differing_update": first, "max_abs": max(d), "at_256": d[-1]},
           "code": code_sha()}
    if out["resume_step"] != rl.BRANCH_FROM or out["resume_code"] != ct.code_sha():
        raise C4pdfBKRefused(f"resume.pt at {out['resume_step']}, code {out['resume_code']}")
    ct._jdump(Path(a.out) / "gap.json", out)
    print(json.dumps({"rel_l2_256": rows[str(rl.BRANCH_FROM)]["rel_l2"], "max_abs_256": rows[str(rl.BRANCH_FROM)]["max_abs"],
                      "first_loss_diff": first, "loss_max_abs": max(d)}))
    return 0


# ---- the branches -----------------------------------------------------------------------------

def _moments(device):
    import torch
    p = REBUILD / "A0" / "resume.pt"
    st = torch.load(p, map_location=device, weights_only=False)
    if int(st["step"]) != rl.BRANCH_FROM or st["code"] != ct.code_sha():
        raise C4pdfBKRefused(f"{p}: step {st['step']}, code {st['code']}")
    return st["opt"]


def _opt_step(opt) -> set:
    return {int(float(v["step"])) for v in opt.state_dict()["state"].values()}


def train(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    arm, root = a.arm, Path(a.root)
    torch, device = ct._torch()
    import transformers
    d = root / arm
    (d / "ckpt").mkdir(parents=True, exist_ok=True)
    rows = ct.open_rows(ct.ROOT)
    model = pt._load_a0_256(device)
    opt = ct.make_opt(model)
    codes = {"c4pdf_bk": code_sha(), "c4pdf_train": pt.code_sha(), "c4_train": ct.code_sha(), "c4pdf_rules": rl.code_sha()}
    state, start = d / "resume.pt", rl.BRANCH_FROM
    if state.exists():
        r0 = torch.load(state, map_location=device, weights_only=False)
        if r0["code"] != codes and not a.allow_code_change:
            raise C4pdfBKRefused(f"resume state from code {r0['code']}, now {codes}")
        model.load_state_dict(r0["model"])
        opt.load_state_dict(r0["opt"])
        start = int(r0["step"])
    else:
        opt.load_state_dict(_moments(device))
        if _opt_step(opt) != {rl.BRANCH_FROM}:
            raise C4pdfBKRefused(f"loaded optimiser at steps {_opt_step(opt)}")
    if UNTIL <= start:
        print(json.dumps({"arm": arm, "resume_step": start, "nothing_to_train": True}))
        return 0
    meta_p = d / "meta.json"
    if meta_p.exists():
        m0 = json.loads(meta_p.read_text())
        if m0["n_micro"] != a.n_micro or m0["compile"] != a.compile:
            raise C4pdfBKRefused(f"{arm} began with n_micro {m0['n_micro']}, compile {m0['compile']}")
    else:
        ct._jdump(meta_p, {"design": "trackC_writeup/c4pdf_bk.py (STATE Blocked 174 (a))", "arm": arm, "code": codes,
                           "n_micro": a.n_micro, "compile": a.compile, "attn": "sdpa",
                           "amp": "bf16 autocast, float32 master", "tf32": False,
                           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "init": ["A0", rl.BRANCH_FROM, "saved weights", "moments from " + str(REBUILD / "A0/resume.pt")],
                           "rate": "peak flat" if arm == "BK" else "lr_pythia", "saves": list(SAVES)})
    if start == rl.BRANCH_FROM and not ct.ckpt_dir(root, arm, start).exists():
        ct.save_ckpt(model, ct.ckpt_dir(root, arm, start))
    fwd = torch.compile(model) if a.compile else model
    log = (d / "train_log.jsonl").open("a")
    log.write(json.dumps({"segment": {"from": start, "until": UNTIL, "code": codes,
                                      "started": time.strftime("%Y-%m-%dT%H:%M:%S")}}) + "\n")
    t0, last, losses = time.time(), start, []

    def save_state(s):
        tmp = state.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "code": codes}, tmp)
        tmp.replace(state)
    for s in range(start + 1, UNTIL + 1):
        i = s - 1
        x = torch.from_numpy(ct.batch_of(rows, i).astype(np.int64)).to(device, non_blocking=True)
        r = ct.step_once(fwd, opt, x, lr(arm, i), a.n_micro)
        if not math.isfinite(r["loss"]):
            raise C4pdfBKRefused(f"{arm} step {s}: loss {r['loss']}")
        r = {"step": s, **r, "wall_s": round(time.time() - t0, 2)}
        log.write(json.dumps(r) + "\n")
        log.flush()
        losses.append(r["loss"])
        if s in SAVES:
            ct.save_ckpt(model, ct.ckpt_dir(root, arm, s))
        if s in SAVES or s - last >= a.resume_every or s == UNTIL:
            save_state(s)
            last = s
        if s % pt.DIVERGE_EVERY == 0 or s == UNTIL:
            dv = ct.diverged(losses)
            if dv is not None:
                save_state(s)
                log.close()
                raise C4pdfBKRefused(f"{arm} diverged at update {dv} of this segment: stopped at {s}")
        if s % 10 == 0 or s == UNTIL or s - rl.BRANCH_FROM <= 4:
            print(json.dumps({"arm": arm, "step": s, "lr": r["lr"], "loss": round(r["loss"], 5),
                              "grad_norm": round(r["grad_norm"], 4)}), flush=True)
    log.close()
    print(json.dumps({"done": arm, "until": UNTIL, "wall_s": round(time.time() - t0)}), flush=True)
    return 0


def hk(a) -> int:
    """H-K: k = 0 is A0's step-256 file; step 257's loss equals A0's own log's to every digit."""
    root = Path(a.root)
    a0 = pt._sd(ct.ckpt_dir(ct.ROOT, "A0", rl.BRANCH_FROM))
    l257 = _log(ct.ROOT / "A0" / "train_log.jsonl")[rl.BRANCH_FROM + 1]
    res = {}
    for arm in ARMS:
        lg = _log(root / arm / "train_log.jsonl")
        res[arm] = {"state": pt.bit_equal(pt._sd(ct.ckpt_dir(root, arm, rl.BRANCH_FROM)), a0),
                    "loss_257": lg.get(rl.BRANCH_FROM + 1), "a0_loss_257": l257}
        res[arm]["pass"] = bool(res[arm]["state"]["equal"] and res[arm]["loss_257"] == l257)
    out = {"gate": "H-K", "per_arm": res, "pass": all(v["pass"] for v in res.values()), "code": code_sha()}
    ct._jdump(Path(a.out) / "hk.json", out)
    print(json.dumps({"H-K": out["pass"], **{k: v["loss_257"] for k, v in res.items()}}))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="c4pdf_bk")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rebuild-root")
    q = sub.add_parser("gap")
    q.add_argument("--out", required=True)
    q = sub.add_parser("train")
    q.add_argument("--arm", required=True, choices=ARMS)
    q.add_argument("--root", default=str(pt.ROOT))
    q.add_argument("--n-micro", type=int, default=512)
    q.add_argument("--no-compile", dest="compile", action="store_false")
    q.add_argument("--resume-every", type=int, default=25)
    q.add_argument("--allow-code-change", action="store_true")
    q = sub.add_parser("hk")
    q.add_argument("--root", default=str(pt.ROOT))
    q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return {"rebuild-root": rebuild_root, "gap": gap, "train": train, "hk": hk}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
