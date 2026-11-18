"""
trackC_writeup/c4pc_train.py — C4′ (c), U1: the replicate pair's harness (`design-C4pc.md`, frozen
on Blocked 166 all (a), 2026-11-18).

Two arms, A0′ (`A0p`, Pythia's schedule) and F′ (`Fp`, flat to W), from seed 0's init (the
released step 0) on **order π**: a seeded permutation of C4's 4 096 000 training rows (D2 (a)).
Every other part of the recipe is C4's: `c4_train.build`, `make_opt`, `step_once`, `lr_pythia`,
`lr_flat`, `save_ckpt`, `SAVE_STEPS`, called unchanged, so `c4_train`'s hash (`1c1e1d2df083`, the
C4 arms') stays and is recorded beside this file's.

**π's file** (`write-perm`): row k of `perm_rows.bin` is shard 0's row π(k), π =
`np.random.default_rng(PERM_SEED).permutation(N_ROWS)`; iteration i reads the file's rows
[1024 i, 1024 (i + 1)), as C4 read the shard's. Written in two sequential passes through buckets
on the NVMe disk (the shard and the file are on the HDD; random reads there would take hours).

**H2′** (`verify`): the file is N_ROWS × ROW uint16; at 4096 seeded k its row equals the shard's
row π(k); its sha256 and π's are recorded (`perm_meta.json`). Under π = identity the batches
equal `c4_train.batch_of`'s (`tests/test_trackC_c4pc.py`, CPU).

**The registration guard** (finding 3 on #361): `train` refuses until each of the five entries
(`ENTRY_IDS`) is in `claims/registry.json` or listed in `trackC_writeup/c4pc_declined.json`, which
counts only when tracked and committed (the user's commit; finding 2 on #362). So neither arm takes
an update before the user's decision, and a decline leaves a record in git.

**H3′, divergence** (finding 4 on #362): every `DIVERGE_EVERY` updates the loop runs
`c4_train.diverged` (C4's rule) on the whole run's losses; a divergence saves the state and stops
the arm, a refusal.

**H4′** (`h4p`): A0′'s loss on C4's 64 held-out rows (`c4_train.HELD_OUT`, never in π's file)
against A0's at 1000 and 4000, within `h4_tol()` = max(2 %, the largest |seed − pythia-70m| /
pythia-70m on the same rows at 1000 and 4000, `c4_read h4-controls`'s records). Outside: a refusal.

    python -m trackC_writeup.c4pc_train write-perm --tmp <nvme dir>
    python -m trackC_writeup.c4pc_train verify
    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4pc_train train --arm A0p --until 1000
    HF_HOME=… python -m trackC_writeup.c4pc_train h1 --arm A0p
    HF_HOME=… python -m trackC_writeup.c4pc_train h4p --steps 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import numpy as np

from trackA_force.a1a import DATA, REPO
from trackC_writeup import c4_train as ct

PERM_SEED = 20261118
N_ROWS, ROW, BATCH, HORIZON = ct.N_ROWS, ct.ROW, ct.BATCH, ct.HORIZON
ROOT = ct.HDD / "c4pc"
PERM_NAME = "perm_rows.bin"
ARMS = ("A0p", "Fp")
SCHEDULES = {"A0p": ct.lr_pythia, "Fp": ct.lr_flat}
C4_ARM = {"A0p": "A0", "Fp": "F"}
N_BUCKETS = 8
H2_SAMPLES = 4096
H4_BASE = 0.02
H4_CONTROLS = {1000: DATA / "trackC/c4/2026-11-09-u1/h4_controls_1000.json",
               4000: DATA / "trackC/c4pc/2026-11-18-u1/h4_controls_4000.json"}
REGISTRY = REPO / "claims/registry.json"
DECLINED = REPO / "trackC_writeup/c4pc_declined.json"   # the user's, committed; absent until then
DIVERGE_EVERY = 50                                       # the loop checks `c4_train.diverged` this often
#: the five entries' ids (L7, L8, L9, L9h, L10); spelled in parts because `tools/check_registry.py`
#: fails on any unregistered id it finds in the tree
ENTRY_IDS = tuple("P-SCH" + str(n) for n in range(2, 7))
ENTRY_OF = dict(zip(("L7", "L8", "L9", "L9h", "L10"), ENTRY_IDS))


class C4pcRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4pc_train.py",):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- π (pure)

def perm(seed: int = PERM_SEED, n: int = N_ROWS) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n).astype(np.int64)


def perm_sha(p: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(p, dtype="<i8").tobytes()).hexdigest()


def source_rows(p: np.ndarray, i: int) -> np.ndarray:
    """The shard rows iteration ``i`` reads under π (0-based; checkpoint i + 1 holds it)."""
    if not 0 <= i < HORIZON:
        raise C4pcRefused(f"iteration {i} outside [0, {HORIZON})")
    return p[i * BATCH:(i + 1) * BATCH]


def batch_of(rows: np.ndarray, i: int) -> np.ndarray:
    """Iteration ``i``'s 1024 rows of π's file (``c4_train.batch_of``'s form)."""
    if not 0 <= i < HORIZON:
        raise C4pcRefused(f"iteration {i} outside [0, {HORIZON})")
    return np.asarray(rows[i * BATCH:(i + 1) * BATCH])


def bucket_plan(p: np.ndarray, n_buckets: int = N_BUCKETS) -> np.ndarray:
    """inv[r] = the file position of shard row r (π⁻¹); the bucket of r is inv[r] // block."""
    if len(p) % n_buckets:
        raise C4pcRefused(f"{len(p)} rows not divisible into {n_buckets} buckets")
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p), dtype=p.dtype)
    return inv


def write_permuted(src: np.ndarray, dst: np.ndarray, p: np.ndarray, tmp: Path,
                   n_buckets: int = N_BUCKETS, chunk: int = 32768, log=print) -> None:
    """dst[k] = src[p[k]] for every k, by two sequential passes: src in order into buckets by
    destination block, then each bucket placed in RAM and written in order."""
    n = len(p)
    blk = n // n_buckets
    inv = bucket_plan(p, n_buckets)
    tmp.mkdir(parents=True, exist_ok=True)
    fh = [open(tmp / f"b{b}.rows", "wb") for b in range(n_buckets)]
    fk = [open(tmp / f"b{b}.pos", "wb") for b in range(n_buckets)]
    t0 = time.time()
    for r0 in range(0, n, chunk):
        x = np.asarray(src[r0:r0 + chunk])
        dest = inv[r0:r0 + len(x)]
        b = dest // blk
        for j in range(n_buckets):
            m = b == j
            if m.any():
                fh[j].write(np.ascontiguousarray(x[m]).tobytes())
                fk[j].write(np.ascontiguousarray(dest[m], dtype="<i8").tobytes())
        if (r0 // chunk) % 16 == 0:
            log(json.dumps({"pass": 1, "rows": r0 + len(x), "of": n, "s": round(time.time() - t0)}))
    for f in fh + fk:
        f.close()
    for j in range(n_buckets):
        pos = np.fromfile(tmp / f"b{j}.pos", dtype="<i8") - j * blk
        rows = np.fromfile(tmp / f"b{j}.rows", dtype=src.dtype).reshape(-1, src.shape[1])
        if len(pos) != blk or not np.array_equal(np.sort(pos), np.arange(blk)):
            raise C4pcRefused(f"bucket {j}: positions are not exactly its block")
        out = np.empty((blk, src.shape[1]), dtype=src.dtype)
        out[pos] = rows
        dst[j * blk:(j + 1) * blk] = out
        del out, rows
        (tmp / f"b{j}.rows").unlink()
        (tmp / f"b{j}.pos").unlink()
        log(json.dumps({"pass": 2, "bucket": j, "of": n_buckets, "s": round(time.time() - t0)}))
    if hasattr(dst, "flush"):
        dst.flush()


def sample_check(dst: np.ndarray, src: np.ndarray, p: np.ndarray, n: int = H2_SAMPLES,
                 seed: int = PERM_SEED + 1) -> dict:
    ks = np.sort(np.random.default_rng(seed).choice(len(p), n, replace=False))
    bad = [int(k) for k in ks if not np.array_equal(np.asarray(dst[k]), np.asarray(src[p[k]]))]
    return {"samples": n, "seed": seed, "mismatches": len(bad), "first_bad": bad[:5]}


# ---------------------------------------------------------------- registration and tolerance (pure)

def committed(path: Path, repo: Path = REPO) -> bool:
    """``path`` is tracked by git and equal to HEAD's copy (a decline leaves a record in git)."""
    import subprocess
    rel = str(path.resolve().relative_to(repo.resolve()))
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel],
                           capture_output=True).returncode == 0
    return tracked and clean


def registration(registry: Path = REGISTRY, declined: Path = DECLINED, check_git: bool = True) -> Dict[str, str]:
    """Per entry id: 'registered' (in the registry), 'declined' (in the tracked, committed
    ``declined`` file, the user's commit; `/challenge-pr` on #362, finding 2), or 'pending'."""
    reg = json.loads(registry.read_text())
    ids = {e.get("id") for e in reg.get("predictions", [])}
    dec = set()
    if declined.exists():
        if check_git and not committed(declined):
            raise C4pcRefused(f"{declined}: a decline counts only when committed (tracked and clean)")
        dec = set(json.loads(declined.read_text()).get("declined", []))
    return {i: "registered" if i in ids else "declined" if i in dec else "pending" for i in ENTRY_IDS}


def require_decided(registry: Path = REGISTRY, declined: Path = DECLINED, check_git: bool = True) -> Dict[str, str]:
    st = registration(registry, declined, check_git)
    pend = [i for i, s in st.items() if s == "pending"]
    if pend:
        raise C4pcRefused(f"entries {pend} neither registered nor declined: no update before the user's decision")
    return st


def prior_losses(log: Path, start: int) -> list:
    """Losses of steps 1 … ``start`` from an arm's log (a resumed step's last entry wins), so the
    divergence check sees the whole run."""
    by = {}
    if log.exists():
        for ln in log.read_text().splitlines():
            d = json.loads(ln)
            if "step" in d and "loss" in d:
                by[int(d["step"])] = float(d["loss"])
    if sorted(k for k in by if k <= start) != list(range(1, start + 1)):
        raise C4pcRefused(f"{log}: steps 1..{start} not all logged")
    return [by[s] for s in range(1, start + 1)]


def h4_tol(controls: Dict[int, dict]) -> dict:
    """max(H4_BASE, the largest |seed − pythia-70m| / pythia-70m over the seeds at each step)."""
    gaps = {}
    for s, rec in controls.items():
        rd = rec["rel_diff"]
        gaps[str(s)] = max(abs(v) for k, v in rd.items() if "-seed" in k)
    return {"gaps": gaps, "base": H4_BASE, "tol": max([H4_BASE] + list(gaps.values()))}


# ---------------------------------------------------------------- I/O

def perm_path(root: Path = ROOT) -> Path:
    return root / "pile" / PERM_NAME


def open_perm(root: Path = ROOT, mode: str = "r") -> np.ndarray:
    return np.memmap(perm_path(root), dtype="<u2", mode=mode, shape=(N_ROWS, ROW))


def write_perm(a) -> int:
    root = Path(a.root)
    pp = perm_path(root)
    if pp.exists():
        raise C4pcRefused(f"{pp} exists")
    if not (ct.ROOT / "data_meta.json").exists():
        raise C4pcRefused("C4's shard is not verified (c4_train verify)")
    src = ct.open_rows(ct.ROOT)[:N_ROWS]
    p = perm()
    pp.parent.mkdir(parents=True, exist_ok=True)
    part = pp.with_suffix(".part")
    dst = np.memmap(part, dtype="<u2", mode="w+", shape=(N_ROWS, ROW))
    write_permuted(src, dst, p, Path(a.tmp), log=lambda s: print(s, flush=True))
    del dst
    part.rename(pp)
    print(json.dumps({"written": str(pp), "bytes": pp.stat().st_size}), flush=True)
    return 0


def verify(a) -> int:
    root = Path(a.root)
    pp = perm_path(root)
    want = N_ROWS * ROW * 2
    if pp.stat().st_size != want:
        raise C4pcRefused(f"{pp}: {pp.stat().st_size} bytes, want {want}")
    p = perm()
    if not np.array_equal(np.sort(p), np.arange(N_ROWS)):
        raise C4pcRefused("π is not a permutation of C4's rows")
    src, dst = ct.open_rows(ct.ROOT), open_perm(root)
    if p.max() >= ct.HELD_OUT[0]:
        raise C4pcRefused("π reaches H4's held-out rows")
    sc = sample_check(dst, src, p)
    b0 = batch_of(dst, 0)
    out = {"code": code_sha(), "code_c4_train": ct.code_sha(), "perm_seed": PERM_SEED, "perm_sha256": perm_sha(p),
           "file": str(pp), "bytes": pp.stat().st_size, "file_sha256": ct.sha256_file(pp), "h2": sc,
           "c4_data_meta": json.loads((ct.ROOT / "data_meta.json").read_text())["batch0"]["sha256"],
           "batch0": {"sha256": hashlib.sha256(b0.tobytes()).hexdigest(), "max_token": int(b0.max())},
           "held_out_untouched": True, "pass": sc["mismatches"] == 0 and int(b0.max()) < 50304}
    ct._jdump(root / "perm_meta.json", out)
    print(json.dumps(out), flush=True)
    return 0 if out["pass"] else 1


def train(a) -> int:
    """C4's loop (`c4_train.train`) on π's file, for A0p / Fp, after the registration guard."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    root, arm = Path(a.root), a.arm
    st = require_decided()
    pm = root / "perm_meta.json"
    if not pm.exists() or not json.loads(pm.read_text())["pass"]:
        raise C4pcRefused(f"{pm}: run `verify` first (H2′)")
    if a.until not in ct.SAVE_STEPS:
        raise C4pcRefused(f"--until {a.until} is not a checkpoint step")
    torch, device = ct._torch()
    import transformers
    d = root / arm
    (d / "ckpt").mkdir(parents=True, exist_ok=True)
    rows = open_perm(root)
    model = ct.build(device)
    opt = ct.make_opt(model)
    state, start = d / "resume.pt", 0
    codes = {"c4pc_train": code_sha(), "c4_train": ct.code_sha()}
    if state.exists():
        s0 = torch.load(state, map_location=device, weights_only=False)
        if s0["code"] != codes and not a.allow_code_change:
            raise C4pcRefused(f"resume state from code {s0['code']}, now {codes}")
        model.load_state_dict(s0["model"])
        opt.load_state_dict(s0["opt"])
        start = int(s0["step"])
    if a.until <= start:
        print(json.dumps({"arm": arm, "until": a.until, "resume_step": start, "nothing_to_train": True}))
        return 0
    meta_p = d / "meta.json"
    if meta_p.exists():
        m0 = json.loads(meta_p.read_text())
        if m0["n_micro"] != a.n_micro or m0["compile"] != a.compile:
            raise C4pcRefused(f"{arm} began with n_micro {m0['n_micro']}, compile {m0['compile']}")
    else:
        ct._jdump(meta_p, {"design": "trackC_writeup/design-C4pc.md", "arm": arm, "code": codes,
                           "registration": st, "n_micro": a.n_micro, "compile": a.compile, "attn": "sdpa",
                           "amp": "bf16 autocast, float32 master", "tf32": False,
                           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                           "transformers": transformers.__version__, "init": [ct.BASE, ct.INIT_REV],
                           "data": {k: json.loads(pm.read_text())[k] for k in ("perm_seed", "perm_sha256", "file_sha256")},
                           "recipe": {"schedule": SCHEDULES[arm].__name__, "as_c4_arm": C4_ARM[arm]}})
    if start == 0 and not ct.ckpt_dir(root, arm, 0).exists():
        ct.save_ckpt(model, ct.ckpt_dir(root, arm, 0))
    fwd = torch.compile(model) if a.compile else model
    losses = prior_losses(d / "train_log.jsonl", start)
    log = (d / "train_log.jsonl").open("a")
    log.write(json.dumps({"segment": {"from": start, "until": a.until, "code": codes,
                                      "started": time.strftime("%Y-%m-%dT%H:%M:%S")}}) + "\n")
    t0, last = time.time(), start

    def save_state(s):
        tmp = state.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "code": codes}, tmp)
        tmp.replace(state)
    for s in range(start + 1, a.until + 1):
        i = s - 1
        x = torch.from_numpy(batch_of(rows, i).astype(np.int64)).to(device, non_blocking=True)
        r = ct.step_once(fwd, opt, x, SCHEDULES[arm](i), a.n_micro)
        if not math.isfinite(r["loss"]):
            raise C4pcRefused(f"{arm} step {s}: loss {r['loss']}")
        r = {"step": s, **r, "wall_s": round(time.time() - t0, 2)}
        log.write(json.dumps(r) + "\n")
        log.flush()
        losses.append(r["loss"])
        if s in ct.SAVE_STEPS:
            ct.save_ckpt(model, ct.ckpt_dir(root, arm, s))
        if s in ct.SAVE_STEPS or s - last >= a.resume_every or s == a.until:
            save_state(s)
            last = s
        if s % DIVERGE_EVERY == 0 or s == a.until:
            dv = ct.diverged(losses)
            if dv is not None:                     # C4's refusal, caught while training (finding 4 on #362)
                save_state(s)
                log.close()
                raise C4pcRefused(f"{arm} diverged at step {dv} by c4_train.diverged: stopped at {s}")
        if s % 10 == 0 or s == a.until or s <= 4:
            print(json.dumps({"arm": arm, "step": s, "lr": r["lr"], "loss": round(r["loss"], 5),
                              "grad_norm": round(r["grad_norm"], 4)}), flush=True)
    log.close()
    print(json.dumps({"done": arm, "until": a.until, "wall_s": round(time.time() - t0)}), flush=True)
    return 0


def init(a) -> int:
    """Step 0 of an arm saved with no update (the registration guard is not needed: nothing has
    trained), then H1′ on it. `train` finds it and does not re-save."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = ct._torch()
    root = Path(a.root)
    d = ct.ckpt_dir(root, a.arm, 0)
    if not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        ct.save_ckpt(ct.build(device), d)
    return h1(a)


def h1(a) -> int:
    return ct.h1(SimpleNamespace(root=a.root, arm=a.arm))


def h4p(a) -> int:
    """H4′: A0′'s held-out loss against A0's (float32 eager), at ``--steps``, within h4_tol()."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = ct._torch()
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    root = Path(a.root)
    tol = h4_tol({s: json.loads(f.read_text()) for s, f in H4_CONTROLS.items()})
    rows = ct.open_rows(ct.ROOT)
    out = {"code": code_sha(), "rows": list(ct.HELD_OUT), "tol": tol, "steps": {}}
    for s in a.steps:
        ls = {}
        for nm, r_, arm in (("A0", ct.ROOT, "A0"), ("A0p", root, "A0p")):
            mdl = from_pretrained_eager(GPTNeoXForCausalLM, str(ct.ckpt_dir(r_, arm, s)),
                                        torch_dtype=torch.float32).to(device).eval()
            ls[nm] = ct.held_loss(mdl, rows, device)
            del mdl
            torch.cuda.empty_cache()
        rd = (ls["A0p"] - ls["A0"]) / ls["A0"]
        out["steps"][str(s)] = {**ls, "rel_diff": rd, "pass": abs(rd) <= tol["tol"]}
    out["pass"] = all(v["pass"] for v in out["steps"].values())
    ct._jdump(root / "A0p" / f"h4p_{'_'.join(map(str, a.steps))}.json", out)
    print(json.dumps(out), flush=True)
    return 0 if out["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--root", default=str(ROOT))
    q = sub.add_parser("write-perm"); common(q)
    q.add_argument("--tmp", required=True)
    q = sub.add_parser("verify"); common(q)
    q = sub.add_parser("train"); common(q)
    q.add_argument("--arm", required=True, choices=ARMS)
    q.add_argument("--until", type=int, required=True)
    q.add_argument("--n-micro", type=int, default=512)
    q.add_argument("--no-compile", dest="compile", action="store_false")
    q.add_argument("--resume-every", type=int, default=25)
    q.add_argument("--allow-code-change", action="store_true")
    for name in ("init", "h1"):
        q = sub.add_parser(name); common(q)
        q.add_argument("--arm", required=True, choices=ARMS)
    q = sub.add_parser("h4p"); common(q)
    q.add_argument("--steps", type=int, nargs="+", required=True)
    a = p.parse_args(argv)
    return {"write-perm": write_perm, "verify": verify, "train": train, "init": init, "h1": h1,
            "h4p": h4p}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
