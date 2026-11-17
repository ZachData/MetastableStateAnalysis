"""
trackC_writeup/c4p_path.py — C4′ (b), U2: is F A0 on another clock? (`design-C4p.md` "(b) Is F
A0 on another clock?" and "(b) as U2 runs it"; Blocked 163 (a), the user's, 2026-11-17).

A new module, so `c4p_read`'s hash (U1's records, `f5b158635210`), `c4_read`'s and `c4_train`'s
stay.

**The statistic.** K(X, Y) = the mean over the 64 held-out rows' 131 072 positions
(`c4_train.HELD_OUT`) of KL(p_X ‖ p_Y), nats. float32 forward (eager, no TF32) and float32
log-softmax; the sum Σ_v p_X (log p_X − log p_Y) and the mean over positions in float64.
For each pseudo-F step s, match(s) = the reference state minimising K(F_s, ·), r(s) = that
minimum, δ(g) = min K(g, neighbour) on the reference grid; s is on the path iff
r(s) ≤ δ(match(s)). Consecutive bit-equal reference states are one state.

**Two phases**, because one model's float32 log-probs on one row are 412 MB:
`cache` runs each model once over the 64 rows and stores its final hidden state (after the
final layer norm) and its unembedding; `k` rebuilds every model's log-probs per chunk of 64
positions on the GPU and accumulates K for every pair in a job, in float64.

Jobs: `null` (the released run against A0 rounded to fp16), `seed1`..`seed9` (each seed
against A0, with the released run beside), `F` (F against A0; refused before `gate.json`).

    export HF_HOME=/run/media/system/HDD_1TB/mets_data/hf HF_HUB_OFFLINE=1 METS_DATA=<main>/data
    python -m trackC_writeup.c4p_path cache --out <dir> --job null seed1 … seed9
    python -m trackC_writeup.c4p_path k --out <dir> --job null seed1 … seed9
    python -m trackC_writeup.c4p_path gate --out <dir>
    python -m trackC_writeup.c4p_path cache --out <dir> --job F
    python -m trackC_writeup.c4p_path k --out <dir> --job F
    python -m trackC_writeup.c4p_path read --out <dir>
    python -m trackC_writeup.c4p_path theta --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trackA_force.a1a import REPO
from trackC_writeup import c4_train as ct

A0_STEPS = tuple(ct.SAVE_STEPS)                                  # 29, with 0 and 3500
F_STEPS = tuple(s for s in ct.SAVE_STEPS if s >= 1)              # 28
REL_STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000)   # the null's 13
SEED_STEPS = REL_STEPS[1:]                                       # 12: no step 1 cached
SEEDS = tuple(f"seed{k}" for k in range(1, 10))
REPOS = {"rel": ct.BASE, **{f"seed{k}": f"EleutherAI/pythia-70m-seed{k}" for k in range(1, 10)}}
EARLY_MAX, LATE_MIN = 256, 320
N_POS = (ct.HELD_OUT[1] - ct.HELD_OUT[0]) * (ct.ROW - 1)         # 131 072
CHUNK, SUB = 64, 8                                               # positions per chunk, per float64 batch
BUDGET_H = 3.0
GATE_SEEDS = 8                                                   # of 9 read "another path"
READINGS = ("one path", "another path", "mixed")


class C4pPathRefused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("trackC_writeup/c4p_path.py", "trackC_writeup/c4_train.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the model sets and jobs (pure)

def key(kind: str, step: int) -> str:
    return f"{kind}@{step}"


def parse(k: str) -> Tuple[str, int]:
    kind, s = k.split("@")
    return kind, int(s)


def job_keys(job: str) -> List[str]:
    if job == "null":
        return [key("A0h", t) for t in A0_STEPS] + [key("rel", s) for s in REL_STEPS]
    if job in SEEDS:
        return ([key("A0", t) for t in A0_STEPS] + [key("rel", s) for s in SEED_STEPS]
                + [key(job, s) for s in SEED_STEPS])
    if job == "F":
        return [key("A0", t) for t in A0_STEPS] + [key("F", s) for s in F_STEPS]
    raise C4pPathRefused(f"job {job}")


JOBS = ("null",) + SEEDS + ("F",)
#: per job: the pseudo-F kind, its steps, the reference kind
ROLES = {"null": ("rel", REL_STEPS, "A0h"), "F": ("F", F_STEPS, "A0"),
         **{s: (s, SEED_STEPS, "A0") for s in SEEDS}}


# ---------------------------------------------------------------- the reading (pure)

def merge_states(steps: Sequence[int], hashes: Sequence[str]) -> List[Tuple[int, ...]]:
    """Consecutive grid steps with bit-equal weights form one state."""
    out: List[List[int]] = []
    last = None
    for s, h in zip(steps, hashes):
        if out and h == last:
            out[-1].append(int(s))
        else:
            out.append([int(s)])
        last = h
    return [tuple(g) for g in out]


def deltas(Kref: np.ndarray) -> np.ndarray:
    """δ(g) = min(K(g, g−1), K(g, g+1)) over the states that exist, on the state matrix."""
    n = Kref.shape[0]
    if n < 2:
        raise C4pPathRefused("one reference state: no resolution")
    d = np.empty(n)
    for g in range(n):
        nb = [Kref[g, h] for h in (g - 1, g + 1) if 0 <= h < n]
        d[g] = min(nb)
    return d


def path_rows(Kps: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pseudo-F step: match (state index), r, on the path. ``Kps`` is (steps, states)."""
    m = np.argmin(Kps, axis=1)
    r = Kps[np.arange(len(m)), m]
    return m, r, r <= delta[m]


def monotone(idx: Sequence[int]) -> bool:
    """Each match's state index ≥ the running maximum of the earlier ones − 1."""
    top = None
    for i in idx:
        if top is not None and i < top - 1:
            return False
        top = i if top is None else max(top, i)
    return True


def window_reading(idx: Sequence[int], on: Sequence[bool]) -> str:
    n, n_on = len(idx), int(np.sum(on))
    if n == 0:
        raise C4pPathRefused("empty window")
    if 3 * n_on >= 2 * n and monotone(idx):
        return "one path"
    if 3 * n_on < n:
        return "another path"
    return "mixed"


def windows(steps: Sequence[int]) -> Dict[str, List[int]]:
    """Positions in ``steps`` of each window."""
    return {"early": [i for i, s in enumerate(steps) if 1 <= s <= EARLY_MAX],
            "late": [i for i, s in enumerate(steps) if s >= LATE_MIN]}


def read_pseudo(K: np.ndarray, keys: Sequence[str], pkind: str, psteps: Sequence[int], rkind: str,
                hashes: Dict[str, str]) -> dict:
    """One pseudo-F against one reference path, from a job's K matrix (row = p, column = q)."""
    pos = {k: i for i, k in enumerate(keys)}
    states = merge_states(A0_STEPS, [hashes[key(rkind, t)] for t in A0_STEPS])
    rep = [pos[key(rkind, g[0])] for g in states]
    Kref = K[np.ix_(rep, rep)]
    delta = deltas(Kref)
    prow = [pos[key(pkind, s)] for s in psteps]
    Kps = K[np.ix_(prow, rep)]
    m, r, on = path_rows(Kps, delta)
    rev = K[np.ix_(rep, prow)].T                                   # K(ref, pseudo), beside
    ident = [s in states[g] for s, g in zip(psteps, m)]
    same = []
    for i, s in enumerate(psteps):
        g = next((j for j, st in enumerate(states) if s in st), None)
        same.append(float(Kps[i, g]) if g is not None else None)
    out = {"states": [list(g) for g in states], "delta": delta.tolist(), "steps": list(psteps),
           "match": [list(states[g]) for g in m], "match_idx": m.tolist(), "r": r.tolist(),
           "on": [bool(x) for x in on], "identity": ident,
           "beside": {"reverse_match": [list(states[g]) for g in np.argmin(rev, 1)],
                      "reverse_r": rev.min(1).tolist(), "K_to_same_step": same},
           "windows": {}}
    for w, ix in windows(psteps).items():
        out["windows"][w] = {"steps": [int(psteps[i]) for i in ix], "n": len(ix),
                             "n_on": int(sum(on[i] for i in ix)), "n_identity": int(sum(ident[i] for i in ix)),
                             "monotone": monotone([int(m[i]) for i in ix]),
                             "reading": window_reading([int(m[i]) for i in ix], [bool(on[i]) for i in ix])}
    return out


def gate_of(null: dict, seeds: Dict[str, dict], need: int = GATE_SEEDS) -> dict:
    """A window is readable iff the null reads "one path" with match(s) = s at ≥ ⅔ of its steps
    and ≥ ``need`` of the seeds read "another path"."""
    res = {}
    for w in ("early", "late"):
        nw = null["windows"][w]
        null_ok = nw["reading"] == "one path" and 3 * nw["n_identity"] >= 2 * nw["n"]
        n_another = sum(s["windows"][w]["reading"] == "another path" for s in seeds.values())
        res[w] = {"null_reading": nw["reading"], "null_identity": f"{nw['n_identity']} of {nw['n']}",
                  "null_ok": bool(null_ok), "seeds_another": int(n_another), "seeds_of": len(seeds),
                  "seed_readings": {k: s["windows"][w]["reading"] for k, s in seeds.items()},
                  "readable": bool(null_ok and n_another >= need)}
    return res


def summed_match(s: int) -> Optional[float]:
    """The A0 step with F's summed rate at checkpoint s (exact sums, linear between steps); None
    past A0's horizon."""
    f = sum(ct.lr_flat(i) for i in range(s))
    c = 0.0
    for t in range(ct.HORIZON):
        nxt = c + ct.lr_pythia(t)
        if nxt >= f:
            return t + (f - c) / (nxt - c) if nxt > c else float(t)
        c = nxt
    return None


# ---------------------------------------------------------------- forward passes

def _torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device")
    return torch, torch.device("cuda")


def load(k: str, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    kind, s = parse(k)
    if kind in ("A0", "A0h", "F"):
        arm = "F" if kind == "F" else "A0"
        m = from_pretrained_eager(GPTNeoXForCausalLM, str(ct.ckpt_dir(ct.ROOT, arm, s)), torch_dtype=torch.float32)
    else:
        m = from_pretrained_eager(GPTNeoXForCausalLM, REPOS[kind], revision=f"step{s}", torch_dtype=torch.float32)
    m = m.to(device).eval()
    if kind == "A0h":
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(p.half().float())
    if any(p.dtype != torch.float32 for p in m.parameters()):
        raise C4pPathRefused(f"{k}: not float32")
    return m


def weights_sha(m) -> str:
    h = hashlib.sha256()
    for n, p in m.named_parameters():
        h.update(n.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _rows():
    rows = ct.open_rows(ct.ROOT)
    x = np.asarray(rows[ct.HELD_OUT[0]:ct.HELD_OUT[1]]).astype(np.int64)
    if x.shape != (64, ct.ROW):
        raise C4pPathRefused(f"held-out rows {x.shape}")
    return x


def cache_dir(out: Path) -> Path:
    return out / "cache"


def cache(a) -> int:
    torch, device = _torch()
    import torch.nn.functional as Fn
    out = Path(a.out)
    cd = cache_dir(out)
    cd.mkdir(parents=True, exist_ok=True)
    x = torch.from_numpy(_rows()).to(device)
    want = []
    for j in a.job:
        if j == "F" and not (out / "gate.json").exists():
            raise C4pPathRefused("F is cached after the gate: gate.json missing")
        want += [k for k in job_keys(j) if k not in want]
    for k in want:
        d = cd / k
        if (d / "meta.json").exists():
            continue
        t0 = time.time()
        m = load(k, device)
        W = m.embed_out.weight.detach()
        if m.embed_out.bias is not None:
            raise C4pPathRefused(f"{k}: embed_out has a bias")
        H = np.empty((64, ct.ROW - 1, W.shape[1]), np.float32)
        nll, dmax = 0.0, None
        with torch.no_grad():
            for r in range(64):
                h = m.gpt_neox(input_ids=x[r:r + 1, :-1]).last_hidden_state[0]
                lg = Fn.linear(h, W)
                if r == 0:                                  # the known answer: the model's own logits
                    dmax = float((m(input_ids=x[:1, :-1]).logits[0] - lg).abs().max())
                    if dmax > 1e-4:
                        raise C4pPathRefused(f"{k}: h·Wᵀ differs from the model's logits by {dmax}")
                nll += float(Fn.cross_entropy(lg, x[r, 1:], reduction="sum"))
                H[r] = h.cpu().numpy()
        if not np.isfinite(H).all():
            raise C4pPathRefused(f"{k}: hidden states not finite")
        tmp = cd / (k + ".tmp")
        tmp.mkdir(exist_ok=True)
        np.save(tmp / "h.npy", H)
        np.save(tmp / "W.npy", W.cpu().numpy())
        meta = {"key": k, "code": code_sha(), "weights_sha256": weights_sha(m), "held_loss": nll / N_POS,
                "logits_check_maxabs": dmax, "seconds": round(time.time() - t0, 2)}
        (tmp / "meta.json").write_text(json.dumps(meta) + "\n")
        tmp.rename(d)
        print(json.dumps(meta), flush=True)
        del m, W
        torch.cuda.empty_cache()
    return 0


def _meta(out: Path, k: str) -> dict:
    f = cache_dir(out) / k / "meta.json"
    if not f.exists():
        raise C4pPathRefused(f"{f}: missing (run cache first)")
    return json.loads(f.read_text())


def k_job(a) -> int:
    torch, device = _torch()
    out = Path(a.out)
    total_m2 = sum(len(job_keys(j)) ** 2 for j in JOBS)
    for job in a.job:
        if job == "F" and not (out / "gate.json").exists():
            raise C4pPathRefused("F's K is computed after the gate: gate.json missing")
        f = out / f"K_{job}.npz"
        if f.exists():
            continue
        keys = job_keys(job)
        M = len(keys)
        metas = [_meta(out, k) for k in keys]
        if {m["code"] for m in metas} != {code_sha()}:
            raise C4pPathRefused(f"{job}: cache codes {sorted({m['code'] for m in metas})} ≠ {code_sha()}")
        W = [torch.from_numpy(np.load(cache_dir(out) / k / "W.npy")).to(device) for k in keys]
        H = [np.load(cache_dir(out) / k / "h.npy", mmap_mode="r") for k in keys]
        V = W[0].shape[0]
        per_row = np.zeros((64, M, M))
        t0 = time.time()
        with torch.no_grad():
            L = torch.empty((M, CHUNK, V), dtype=torch.float32, device=device)
            for r in range(64):
                acc = torch.zeros((M, M), dtype=torch.float64, device=device)
                hr = [torch.from_numpy(np.ascontiguousarray(H[i][r])).to(device) for i in range(M)]
                for c0 in range(0, ct.ROW - 1, CHUNK):
                    for i in range(M):
                        L[i] = torch.log_softmax(torch.nn.functional.linear(hr[i][c0:c0 + CHUNK], W[i]), -1)
                    for b0 in range(0, CHUNK, SUB):
                        L64 = L[:, b0:b0 + SUB].double().transpose(0, 1).contiguous()      # (SUB, M, V)
                        C = torch.bmm(L64.exp(), L64.transpose(1, 2))                   # C[i, j] = Σ p_i log p_j
                        acc += (torch.diagonal(C, dim1=1, dim2=2).unsqueeze(2) - C).sum(0)
                    if r == 0 and c0 == (ct.ROW - 1) // 2 - CHUNK:                     # one half-row timed
                        torch.cuda.synchronize()
                        half = time.time() - t0
                        proj = half * 2 * 64 * total_m2 / (M * M) / 3600
                        print(json.dumps({"job": job, "half_row_s": round(half, 2), "projected_all_jobs_h": round(proj, 2)}), flush=True)
                        if proj > BUDGET_H:
                            raise C4pPathRefused(f"projected {proj:.2f} h > {BUDGET_H} h: stopping for the user")
                per_row[r] = acc.cpu().numpy()
                if r % 16 == 15:
                    print(json.dumps({"job": job, "row": r + 1, "s": round(time.time() - t0, 1)}), flush=True)
        Ksum = per_row.sum(0) / N_POS
        if not np.isfinite(Ksum).all() or np.any(np.diag(Ksum) != 0):
            raise C4pPathRefused(f"{job}: K not finite or its diagonal non-zero")
        np.savez(f, K=Ksum, per_row=per_row / (N_POS / 64), keys=np.array(keys),
                 code=code_sha(), seconds=time.time() - t0)
        print(json.dumps({"job": job, "M": M, "s": round(time.time() - t0, 1), "K_min": float(Ksum.min()),
                          "K_max": float(Ksum.max())}), flush=True)
        del W, H, L
        torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- no forward pass

def _job(out: Path, job: str) -> Tuple[np.ndarray, List[str], Dict[str, str], dict]:
    z = np.load(out / f"K_{job}.npz")
    if str(z["code"]) != code_sha():
        raise C4pPathRefused(f"K_{job}: code {z['code']} ≠ {code_sha()}")
    keys = [str(k) for k in z["keys"]]
    metas = {k: _meta(out, k) for k in keys}
    return z["K"], keys, {k: m["weights_sha256"] for k, m in metas.items()}, metas


def _brief(rd: dict) -> dict:
    return {w: (v["reading"], f"{v['n_on']}/{v['n']} on", f"{v['n_identity']} identity", v["monotone"])
            for w, v in rd["windows"].items()}


def gate(a) -> int:
    out = Path(a.out)
    if (out / "K_F.npz").exists():
        raise C4pPathRefused("K_F.npz exists: the gate is read before F")
    K, keys, hs, _ = _job(out, "null")
    null = read_pseudo(K, keys, "rel", REL_STEPS, "A0h", hs)
    seeds, scale = {}, {}
    for sd in SEEDS:
        K, keys, hs, _ = _job(out, sd)
        seeds[sd] = read_pseudo(K, keys, sd, SEED_STEPS, "A0", hs)
        pos = {k: i for i, k in enumerate(keys)}
        scale[sd] = {str(s): float(K[pos[key("rel", s)], pos[key(sd, s)]]) for s in SEED_STEPS}
    g = gate_of(null, seeds)
    res = {"code": code_sha(), "gate": g, "null": null, "seeds": seeds,
           "beside": {"K_released_seed_same_step": scale}}
    (out / "gate.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(g, indent=1))
    print("null", json.dumps(_brief(null)))
    for sd in SEEDS:
        print(sd, json.dumps(_brief(seeds[sd])))
    return 0


def _train_loss(arm: str) -> Dict[int, float]:
    out = {}
    for line in (ct.ROOT / arm / "train_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "step" in r:
            out[int(r["step"])] = float(r["loss"])
    return out


def read(a) -> int:
    out = Path(a.out)
    g = json.loads((out / "gate.json").read_text())
    if g["code"] != code_sha():
        raise C4pPathRefused(f"gate code {g['code']} ≠ {code_sha()}")
    K, keys, hs, metas = _job(out, "F")
    rd = read_pseudo(K, keys, "F", F_STEPS, "A0", hs)
    reading = {w: (rd["windows"][w]["reading"] if g["gate"][w]["readable"] else None) for w in ("early", "late")}
    tl = {arm: _train_loss(arm) for arm in ("A0", "F")}
    rows = []
    for s, mt, r, on in zip(F_STEPS, rd["match"], rd["r"], rd["on"]):
        t = mt[0]
        rows.append({"s": s, "match": mt, "r": r, "on": on,
                     "held_loss_F": metas[key("F", s)]["held_loss"], "held_loss_A0_match": metas[key("A0", t)]["held_loss"],
                     "train_loss_F": tl["F"].get(s + 1), "train_loss_A0_match": tl["A0"].get(t + 1),
                     "summed_rate_A0_step": summed_match(s)})
    res = {"code": code_sha(), "readable": {w: g["gate"][w]["readable"] for w in g["gate"]}, "reading": reading,
           "F": rd, "per_step": rows,
           "beside": {"held_loss_A0": {str(t): metas[key("A0", t)]["held_loss"] for t in A0_STEPS},
                      "K_A0_F_same_step": {str(s): float(K[keys.index(key("F", s)), keys.index(key("A0", s))]) for s in F_STEPS}}}
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"readable": res["readable"], "reading": reading}))
    print("F", json.dumps(_brief(rd)))
    for row in rows:
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}))
    return 0


def group(name: str) -> str:
    if "embed_in" in name:
        return "embed_in"
    if "embed_out" in name:
        return "embed_out"
    if "layernorm" in name or "layer_norm" in name:
        return "norm"
    if ".attention." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    return "other"


def disp(sd: Dict[str, np.ndarray], sd0: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    gs: Dict[str, List[np.ndarray]] = {}
    for k in sorted(sd0):
        if np.issubdtype(sd0[k].dtype, np.floating):
            gs.setdefault(group(k), []).append((sd[k].astype(np.float64) - sd0[k]).ravel())
    out = {g: np.concatenate(v) for g, v in gs.items()}
    out["all"] = np.concatenate([out[g] for g in sorted(gs)])
    return out


def theta(a) -> int:
    """Beside: per parameter group ‖θ_s − θ_0‖ on both arms, the A0 step of nearest displacement,
    and cos(θ^F_s − θ_0, θ^A0_match(s) − θ_0)."""
    out = Path(a.out)
    rd = json.loads((out / "reading.json").read_text())
    sd0 = {k: v.astype(np.float64) for k, v in ct.local_sd(ct.ROOT, "A0", 0).items()}
    norms = {"A0": {}, "F": {}}
    for t in A0_STEPS:
        d = disp(ct.local_sd(ct.ROOT, "A0", t), sd0)
        norms["A0"][str(t)] = {g: float(np.linalg.norm(v)) for g, v in d.items()}
    res_rows = []
    for row in rd["per_step"]:
        s, t = row["s"], row["match"][0]
        dF = disp(ct.local_sd(ct.ROOT, "F", s), sd0)
        dA = disp(ct.local_sd(ct.ROOT, "A0", t), sd0)
        nF = {g: float(np.linalg.norm(v)) for g, v in dF.items()}
        norms["F"][str(s)] = nF
        cos = {g: (float(dF[g] @ dA[g] / (np.linalg.norm(dF[g]) * np.linalg.norm(dA[g])))
                   if np.linalg.norm(dF[g]) > 0 and np.linalg.norm(dA[g]) > 0 else None) for g in dF}
        eq = {g: int(min(A0_STEPS, key=lambda u: abs(norms["A0"][str(u)][g] - nF[g]))) for g in nF}
        res_rows.append({"s": s, "match": t, "cos_to_match": cos, "A0_step_equal_disp": eq})
        print(json.dumps({"s": s, "match": t, "cos_all": round(cos["all"] or 0, 4), "eq_all": eq["all"],
                          "normF_all": round(nF["all"], 3), "normA0match_all": round(norms["A0"][str(t)]["all"], 3)}), flush=True)
    res = {"code": code_sha(), "norms": norms, "rows": res_rows}
    (out / "theta.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("cache", "k", "gate", "read", "theta"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        if cmd in ("cache", "k"):
            sp.add_argument("--job", nargs="+", choices=JOBS, required=True)
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return {"cache": cache, "k": k_job, "gate": gate, "read": read, "theta": theta}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
