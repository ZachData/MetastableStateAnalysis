"""
prompt_library/l2.py — the library's U2: T5 U2's trajectories re-read on the library
(`prompt_library/design-L.md` "U2, fixed now (under D5 (a))" and "U2's details", fixed before
any U2 output; D5 (a), Blocked 140, the user's).

Every checkpoint of T5 U2's six trajectories (`trackT_timaeus/t5c.py` `TRAJ`, same steps, same
loaders, same tokenizers) is read on the first k = 1024 `val` items of the library
(`core.library.load("val", 1024)`), one item per forward pass as `c2_aniso.states` reads a prompt
(the model's own tokenizer, no special tokens, truncated at 512 of its own tokens, float32, eager,
TF32 off), and T5 U1's per-state statistics are kept per item (ρ_ℓ, ρ_ℓ⁻⁰, m, m⁻⁰, n0; computed
on the GPU in float64 by the same formulas as `t5.state_stats`, which the known answer checks).

The rule is T5 U2's, unchanged (`t5c.spread`, `t5c.label`): Δ = ρ_mid(first) − min_t ρ_mid(t),
spreads if Δ ≥ 0.15 and the bootstrap's 5th percentile > 0.05, does not spread if its 95th < 0.15,
else not told apart; the bootstrap resamples items (1000 resamples, one per resample used at
every step, the minimum retaken). Only its arithmetic changes: a resample's mean over items is
taken as a count-weighted mean (`count_spread`, equal to `t5c.spread` to float rounding on the
same draws, `tests/test_l2.py`), since `t5c.spread`'s fancy index would hold
53 × 1000 × 1024 × 17 float64 at k = 1024.

Gate, before any Δ is read on the library (`design-L.md`): the known answer (this reader fed the
7 v1 texts reproduces U2's records at 410m 0, 4000 and 32m 1, 9536 within 1e-6) and the rule's
synthetic at k = 1024 (planted 0.15 "spreads" ≥ 90 %, flat ≤ 5 %, s = the library's own spread at
32m step 9536).

Recorded beside T5 U2, not as a correction to it. Tier 1, description.
    METS_DATA=<main>/data python -m prompt_library.l2 known --out <dir>
    METS_DATA=<main>/data python -m prompt_library.l2 run --out <dir> [--trajs ...] [--steps ...]
    METS_DATA=<main>/data python -m prompt_library.l2 read --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from core import library
from trackA_force.a1a import DATA, REPO, V1
from trackT_timaeus import t5c

SPLIT = "val"
K = 1024
MAX_TOKENS = 512                     # U2's: ≤ 512 of the model's own tokens (t2.MAX_TOKENS)
KEYS = ("rho", "rho_no0", "m", "m_no0", "n0")
U2_DIR = "trackT/t5u2/2026-10-30"
KNOWN = {"pythia-410m": (0, 4000), "dodecahedron-32m": (1, 9536)}
KNOWN_TOL = t5c.KNOWN_TOL            # 1e-6
SUBSET_MIN = 32                      # design-L.md D3: a subset is read on its own at ≥ 32 items
MIN_NORM = 1e-12                     # c2_aniso.MIN_NORM's role: a row with no direction refuses


class L2Refused(RuntimeError):
    pass


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("prompt_library/l2.py", "core/library.py", "trackT_timaeus/t5c.py", "trackT_timaeus/t5.py",
              "trackT_timaeus/t2.py", "trackC_writeup/c2_aniso.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- the rule's arithmetic (pure)

def counts_of(idx: np.ndarray, n: int) -> np.ndarray:
    """(n_boot, n) resample indices → (n_boot, n) counts of each item in each resample."""
    C = np.zeros((idx.shape[0], n), np.float64)
    np.add.at(C, (np.repeat(np.arange(idx.shape[0]), idx.shape[1]), idx.ravel()), 1.0)
    return C


def count_spread(P: np.ndarray, rng: np.random.Generator, n_boot: int = t5c.N_BOOT) -> dict:
    """`t5c.spread` on ``P`` (steps, items, states), the same rng draws, each resample's item mean
    taken as C @ P / n instead of P[:, idx].mean(): equal to float rounding, O(steps·items·states·n_boot)
    flops and no (steps, n_boot, items, states) array."""
    P = np.asarray(P, np.float64)
    s = t5c.mid_series(P)
    delta = float(s[0] - s.min())
    n = P.shape[1]
    idx = rng.integers(0, n, (n_boot, n))
    C = counts_of(idx, n) / n                                        # (n_boot, n)
    M = np.einsum("bn,tns->tbs", C, P, optimize=True)                # (steps, n_boot, states)
    B = np.median(M[..., 1:-1], -1)                                  # (steps, n_boot)
    db = B[0] - B.min(0)
    p5, p95 = (float(x) for x in np.percentile(db, [5, 95]))
    return {"delta": delta, "p5": p5, "p95": p95, "argmin": int(np.argmin(s)), "first": float(s[0]),
            "min": float(s.min()), "label": t5c.label(delta, p5, p95)}


def synthetic(s: float, plant: float, rng: np.random.Generator, n_items: int, draws: int = t5c.SYN_DRAWS,
              steps: Sequence[int] = t5c.DODECA_STEPS) -> dict:
    """`t5c.synthetic` at n_items through `count_spread` (one middle state per item, as
    `t5c.scalar_as_states`)."""
    mu = t5c.planted_mu(steps, plant) if plant else np.full(len(steps), t5c.SYN_MU)
    labels = []
    for _ in range(draws):
        X = mu[:, None] + rng.normal(0.0, s, (len(steps), n_items))
        labels.append(count_spread(t5c.scalar_as_states(X), rng)["label"])
    return {"s": s, "plant": plant, "draws": draws, "n_items": n_items,
            **{k: labels.count(k) / draws for k in ("spreads", "does not spread", "not told apart")}}


def item_spread(R: np.ndarray) -> float:
    """`t5c.prompt_spread` on one checkpoint's (items, states) ρ: SD (ddof 1) of each item's middle median."""
    return float(np.std(np.median(np.asarray(R)[:, 1:-1], 1), ddof=1))


# ---------------------------------------------------------------- reading a model

def stats_torch(h) -> Dict[str, float]:
    """`t5.state_stats` on one state ``h`` (n, d), on its device in float64; the median is numpy's
    (the mean of the two middle values; torch.median would take the lower)."""
    import torch
    X = h.to(torch.float64)
    nrm = torch.linalg.vector_norm(X, dim=1)
    if float(nrm.min()) < MIN_NORM:
        raise L2Refused(f"a row of norm {float(nrm.min()):.3g}: no direction")
    U = X / nrm[:, None]

    def mc(V):
        s = V.sum(0)
        n = V.shape[0]
        return float((s @ s - n) / (n * (n - 1)))

    def m_of(Y):
        mu = Y.mean(0)
        return float(mu @ mu / (Y * Y).sum(1).mean())
    return {"rho": mc(U), "rho_no0": mc(U[1:]), "m": m_of(X), "m_no0": m_of(X[1:]),
            "n0": float(nrm[0] / torch.quantile(nrm, 0.5))}


def check_tokenizer(tok, text: str) -> None:
    """The library's slots are the model's own tokens: refuse a tokenizer that adds a start or end
    token (`status-L.md` Parked, carried to U2; Pythia and Dodecahedron add none, t2.tokenizer)."""
    a = tok(text)["input_ids"]
    b = tok(text, add_special_tokens=False)["input_ids"]
    if a != b:
        raise L2Refused(f"{type(tok).__name__} adds special tokens ({len(a)} against {len(b)} ids)")


def read_text(model, tok, text: str, device) -> tuple:
    """One text as `c2_aniso.states` reads a prompt: (per-state stats, n_tokens, truncated)."""
    import torch
    full = len(tok(text, add_special_tokens=False)["input_ids"])
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        o = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        st = [stats_torch(h[0]) for h in o.hidden_states]
    return st, int(enc["input_ids"].shape[1]), full > MAX_TOKENS


def read_texts(model, tok, texts: Sequence[str], device) -> dict:
    per, ntok, trunc = [], [], []
    for t in texts:
        st, n, tr = read_text(model, tok, t, device)
        per.append(st)
        ntok.append(n)
        trunc.append(tr)
    L = model.config.num_hidden_layers + 1
    arr = {k: np.asarray([[s[k] for s in row] for row in per], np.float64) for k in KEYS}   # (items, states)
    if arr["rho"].shape != (len(texts), L) or not all(np.all(np.isfinite(v)) for v in arr.values()):
        raise L2Refused(f"record not populated: {arr['rho'].shape}, expected ({len(texts)}, {L})")
    return {**arr, "n_tokens": np.asarray(ntok, np.int64), "truncated": np.asarray(trunc, bool)}


# ---------------------------------------------------------------- commands

def _setup():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from trackC_writeup import c2_aniso as c
    return c._torch_setup()


def known(a) -> int:
    """The known answer: the 7 v1 texts through this reader against U2's step records."""
    from core.config import PROMPTS
    torch, device = _setup()
    u2 = Path(a.u2_dir)
    texts = [PROMPTS[p] for p in V1]
    diffs = {}
    for name, steps in KNOWN.items():
        tok = t5c.tokenizer_of(t5c.TRAJ[name][2], a.hf_roots)
        check_tokenizer(tok, texts[0])
        for s in steps:
            m, _ = t5c.load(name, s, a.hf_roots, device)
            got = read_texts(m, tok, texts, device)
            ref = json.loads((u2 / name / f"step{s}.json").read_text())
            mean = {k: got[k].mean(0).tolist() for k in KEYS}
            d = t5c.rec_diff(mean, ref)
            d = max(d, float(np.max(np.abs(got["rho"] - np.asarray(ref["rho_per_prompt"])))),
                    float(np.max(np.abs(got["rho_no0"] - np.asarray(ref["rho_no0_per_prompt"])))))
            if got["n_tokens"].tolist() != ref["n_tokens"]:
                d = math.inf
            diffs[f"{name}@{s}"] = d
            del m
            torch.cuda.empty_cache()
    ok = max(diffs.values()) <= KNOWN_TOL
    res = {"code": code_sha(), "max_diff": diffs, "tol": KNOWN_TOL, "pass": ok}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "known.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if ok else 1


def _write_npz(path: Path, arrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    tmp.replace(path)


def run(a) -> int:
    torch, device = _setup()
    import transformers
    items = library.load(SPLIT, a.k)
    texts = [it.text for it in items]
    out = Path(a.out)
    meta = {"design": "prompt_library/design-L.md", "unit": "U2", "code": code_sha(), "split": SPLIT, "k": a.k,
            "library_hash": library.library_hash(SPLIT, a.k), "max_tokens": MAX_TOKENS, "dtype": "float32",
            "stats_dtype": "float64", "tf32": False, "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "transformers": transformers.__version__, "hf_roots": list(a.hf_roots)}
    same = ("code", "library_hash", "k", "device", "torch", "transformers")   # a resume keeps one environment
    if (out / "meta.json").exists():
        prev = json.loads((out / "meta.json").read_text())
        diff = {k: (prev.get(k), meta[k]) for k in same if prev.get(k) != meta[k]}
        if diff:
            raise L2Refused(f"{out}: meta.json differs from this run in {diff}")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "meta.json.tmp"                     # atomic: a cut write leaves the old meta.json
    tmp.write_text(json.dumps(meta, indent=1) + "\n")
    tmp.replace(out / "meta.json")
    t0 = time.time()
    toks = {}
    for name in a.trajs:
        repo, steps, tk = t5c.TRAJ[name]
        if tk not in toks:
            toks[tk] = t5c.tokenizer_of(tk, a.hf_roots)
            check_tokenizer(toks[tk], texts[0])
        for s in (a.steps or steps):
            if s not in steps:
                raise L2Refused(f"{name}: step {s} is not on U2's list")
            p = out / name / f"step{s}.npz"
            if p.exists() and p.with_suffix(".json").exists() and not a.force:
                continue                            # both files, or the step is re-read
            m, info = t5c.load(name, s, a.hf_roots, device)
            t1 = time.time()
            rec = read_texts(m, toks[tk], texts, device)
            _write_npz(p, rec)
            (out / name / f"step{s}.json").write_text(json.dumps(
                {"trajectory": name, "model": repo, "step": s, **info, "code": meta["code"],
                 "library_hash": meta["library_hash"], "read_s": round(time.time() - t1, 1),
                 "rho_mid": t5c.t5.rho_mid(rec["rho"].mean(0)), "n_truncated": int(rec["truncated"].sum()),
                 "n_tokens_min": int(rec["n_tokens"].min()), "n_tokens_max": int(rec["n_tokens"].max())},
                indent=1) + "\n")
            print(json.dumps({"wrote": f"{name}/step{s}", "rho_mid": round(t5c.t5.rho_mid(rec["rho"].mean(0)), 4),
                              "read_s": round(time.time() - t1, 1), "t": round(time.time() - t0)}), flush=True)
            del m
            torch.cuda.empty_cache()
    print(json.dumps({"done": True, "wall_s": round(time.time() - t0, 1)}))
    return 0


def collect(out: Path, k: int) -> Dict[str, dict]:
    meta = json.loads((out / "meta.json").read_text())
    if meta["library_hash"] != library.library_hash(SPLIT, k) or meta["k"] != k:
        raise L2Refused(f"{out}: records are on library {meta['library_hash'][:12]} k {meta['k']}, not k {k}")
    recs = {}
    for name, (_, steps, _) in t5c.TRAJ.items():
        missing = [s for s in steps if not (out / name / f"step{s}.npz").exists()]
        if missing:
            raise L2Refused(f"{name}: {len(missing)} of {len(steps)} steps missing, e.g. {missing[:3]}")
        zs = [np.load(out / name / f"step{s}.npz") for s in steps]
        js = [json.loads((out / name / f"step{s}.json").read_text()) for s in steps]
        if {j["code"] for j in js} != {meta["code"]}:
            raise L2Refused(f"{name}: records from another code version")
        recs[name] = {"steps": list(steps), **{key: np.stack([z[key] for z in zs]) for key in KEYS},
                      "n_tokens": np.stack([z["n_tokens"] for z in zs]),
                      "truncated": np.stack([z["truncated"] for z in zs])}
    return recs


def gate(recs: Dict[str, dict], known_path: Path, k: int, code: str) -> dict:
    """``code``: the records' code; the known answer counts only if it ran on the same code
    (`/challenge-pr` on #322, finding 3)."""
    g = {}
    kn = json.loads(known_path.read_text()) if known_path.exists() else {"pass": False, "missing": str(known_path)}
    if kn.get("code") != code:
        kn = {**kn, "pass": False, "refused": f"known.json is from code {kn.get('code')}, the records from {code}"}
    g["known_answer"] = kn
    s = item_spread(recs["dodecahedron-32m"]["rho"][t5c.DODECA_STEPS.index(t5c.SPREAD_STEP)])
    rng = np.random.default_rng(t5c.BOOT_SEED)
    planted = synthetic(s, t5c.SYN_PLANT, rng, k)
    flat = synthetic(s, 0.0, rng, k)
    g["rule"] = {"s": s, "planted": planted, "flat": flat,
                 "pass": planted["spreads"] >= t5c.SYN_PLANTED_MIN and flat["spreads"] <= t5c.SYN_FLAT_MAX}
    g["pass"] = bool(kn.get("pass")) and g["rule"]["pass"]
    return g


def _label(P: np.ndarray, steps: Sequence[int]) -> dict:
    sp = count_spread(P, np.random.default_rng(t5c.BOOT_SEED))
    sp["argmin_step"] = steps[sp["argmin"]]
    return sp


def reading(recs: Dict[str, dict], subsets: List[str], u2_reading: dict) -> dict:
    res = {}
    big = sorted({x for x in subsets if subsets.count(x) >= SUBSET_MIN}, key=lambda x: -subsets.count(x))
    sub_arr = np.asarray(subsets)
    for name, r in recs.items():
        steps = r["steps"]
        sp, sp0 = _label(r["rho"], steps), _label(r["rho_no0"], steps)
        per_subset = {x: {"n": int((sub_arr == x).sum()), **_label(r["rho"][:, sub_arr == x], steps)} for x in big}
        no_web = ~np.isin(sub_arr, sorted(library.WEB_SUBSETS))
        mid_item = np.median(r["rho"][..., 1:-1], -1)                 # (steps, items)
        u2 = u2_reading[name]
        res[name] = {"scored": name in t5c.SCORED, "rho_mid": sp, "rho_mid_no0_beside": sp0,
                     "u2_label_7": u2["rho_mid"]["label"], "u2_delta_7": u2["rho_mid"]["delta"],
                     "beside": {"per_subset": per_subset,
                                "without_web": {"n": int(no_web.sum()), **_label(r["rho"][:, no_web], steps)},
                                "n_truncated": int(r["truncated"][0].sum()),
                                "n_tokens_range": [int(r["n_tokens"].min()), int(r["n_tokens"].max())]},
                     "series": {"step": steps, "rho_mid": t5c.mid_series(r["rho"]).tolist(),
                                "rho_mid_no0": t5c.mid_series(r["rho_no0"]).tolist(),
                                "rho_mid_7": u2["series"]["rho_mid"],
                                "item_mid_p10_p50_p90": np.percentile(mid_item, [10, 50, 90], axis=1).T.tolist(),
                                "m_mid": np.median(r["m"].mean(1)[:, 1:-1], -1).tolist(),
                                "n0_mid": np.median(r["n0"].mean(1)[:, 1:-1], -1).tolist()}}
    return res


def read(a) -> int:
    out = Path(a.out)
    recs = collect(out, a.k)
    code = json.loads((out / "meta.json").read_text())["code"]
    g = gate(recs, out / "known.json", a.k, code)
    res = {"code": code, "reader": code_sha(),
           "library_hash": library.library_hash(SPLIT, a.k), "k": a.k, "gate": g}
    if g["pass"]:
        subsets = [e[1] for e in library.manifest()["splits"][SPLIT]["items"][:a.k]]
        u2 = json.loads((Path(a.u2_dir) / "reading.json").read_text())["reading"]
        res["reading"] = reading(recs, subsets, u2)
    else:
        res["refused"] = "the gate failed: no Δ is read (design-L.md 'U2'; back to the user)"
    (out / "reading.json").write_text(json.dumps(res, indent=1) + "\n")
    show = {"gate": {"known": g["known_answer"].get("max_diff"), "s": g["rule"]["s"],
                     "planted": g["rule"]["planted"]["spreads"], "flat": g["rule"]["flat"]["spreads"],
                     "pass": g["pass"]}}
    if g["pass"]:
        show["labels"] = {n: (round(v["rho_mid"]["delta"], 4), round(v["rho_mid"]["p5"], 4),
                              round(v["rho_mid"]["p95"], 4), v["rho_mid"]["label"], v["rho_mid"]["argmin_step"],
                              "U2: " + v["u2_label_7"]) for n, v in res["reading"].items()}
    print(json.dumps(show, indent=1))
    return 0 if g["pass"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("known", "run", "read"):
        sp = sub.add_parser(cmd)
        sp.add_argument("--out", required=True)
        sp.add_argument("--hf-roots", nargs="+", default=list(t5c.DEFAULT_ROOTS))
        sp.add_argument("--u2-dir", default=str(DATA / U2_DIR))
        sp.add_argument("--k", type=int, default=K)
        if cmd == "run":
            sp.add_argument("--trajs", nargs="+", default=list(t5c.TRAJ), choices=list(t5c.TRAJ))
            sp.add_argument("--steps", nargs="*", type=int, default=None)
            sp.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return {"known": known, "run": run, "read": read}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
