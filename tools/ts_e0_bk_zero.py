"""Does the large key bias matter? b_K zeroed at 143000 (`MATH_TRAINING_SYMMETRY.md` §11; tier 1).

WHY THIS EXISTS
---------------
The last layers' non-rotary key bias (KBn, dims 16-63 of each head's key bias) grew to rms 54
(70m) / 16.5 (410m) on Pythia although the loss cannot see it in exact arithmetic (§8.4, §10).
It adds a per-row constant to every score, so a softmax ignores it; a finite-precision graph
may not (the stored key and the score are rounded at b_K's size). §11, frozen at `932f195`
before any output, asks whether zeroing KBn in every layer moves the output more than each
arm's own rounding does: R32 (the float32 eager graph every reading here uses), H16 (fp16
eager), F16 (fp16 flash), each against R64 (float64 eager, the exact reference).

PARTS
-----
    run  --model M [--rows N]   one cell (step 143000): every arm intact and zeroed, the gates'
                                R64 passes, the description beside; record data/ts/e0bk/<M>.json
    read                        §11's labels and the answer to Blocked 162 (b) -> read.json

    python tools/ts_e0_bk_zero.py run --model pythia-70m
Input: Pythia's iteration 4000, rows 0..N-1 of shard 0 (N = 64), the first 513 tokens a row.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ts_e0_backward import MAIN, HUB, ROT, rows_of, tensors  # noqa: E402

OUT = MAIN / "data/ts/e0bk"
STEP, IT, T = 143000, 4000, 512
ARMS = ("R32", "H16", "F16")
G0_KL, G0_H, G1_FACTOR = 1e-12, 1e-10, 10.0
COSMETIC, MATTERS = 1.0, 10.0


def load(model: str, arm: str, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    dt = {"R64": torch.float64, "R32": torch.float32, "H16": torch.float16, "F16": torch.float16}[arm]
    m = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model}", revision=f"step{STEP}", cache_dir=str(HUB), local_files_only=True,
        torch_dtype=dt, attn_implementation="sdpa" if arm == "F16" else "eager").to(device).eval()
    assert m.config.rotary_pct == 0.25 and m.config.hidden_size // m.config.num_attention_heads == 64
    return m


def kbias(m) -> list:
    """Each layer's QKV bias viewed (heads, 3, 64); index 1 on the middle axis is the key."""
    return [l.attention.query_key_value.bias.view(-1, 3, 64) for l in m.gpt_neox.layers]


class Zero:
    """Context: zero a slice of every layer's key bias (``part`` 'n' = dims 16-63, 'r' = 0-15)."""
    def __init__(self, m, part: str):
        self.b, self.sl = kbias(m), (slice(ROT, None) if part == "n" else slice(0, ROT))
        self.saved = None

    def __enter__(self):
        import torch
        with torch.no_grad():
            self.saved = [b[:, 1, self.sl].clone() for b in self.b]
            for b in self.b:
                b[:, 1, self.sl] = 0
        return self

    def __exit__(self, *exc):
        import torch
        with torch.no_grad():
            for b, s in zip(self.b, self.saved):
                b[:, 1, self.sl] = s


def fwd(m, arm: str, x, attn: bool):
    """Log-probs (float64), hidden states (float64), attentions (eager arms only) for one row."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.set_default_dtype(torch.float64 if arm == "R64" else torch.float32)  # HF 4.44 builds its mask in it
    kern = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if arm == "F16" else sdpa_kernel(
        [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    with torch.no_grad(), kern:
        o = m(input_ids=x, output_hidden_states=True, output_attentions=attn and arm != "F16")
    lg = o.logits[0].double()
    fin = bool(torch.isfinite(lg).all())
    lp = torch.log_softmax(lg, -1)
    hs = [h[0].double() for h in o.hidden_states]
    at = [a[0].double().cpu() for a in o.attentions] if (attn and arm != "F16") else None  # CPU: 410m's fp64 maps OOM the 3080
    del o
    return {"lp": lp, "hs": hs, "at": at, "finite": fin}


def kl(a, b):
    """Per-token KL(p_a || p_b) from log-probs (float64)."""
    import torch
    return (a.exp() * (a - b)).sum(-1).clamp_min(0)


def rel(a_hs, b_hs):
    """Per layer, per token ||a - b|| / ||a||."""
    return [((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-300)) for a, b in zip(a_hs, b_hs)]


def tv(a_at, b_at):
    """Per layer, the attention rows' total variation, mean over heads and queries."""
    return [float(0.5 * (a - b).abs().sum(-1).mean()) for a, b in zip(a_at, b_at)]


def ce(lp, y):
    return float(-lp.gather(-1, y[:, None]).mean())


def offsets(m, hs) -> list:
    """Per layer, |q_n . b_n| / sqrt(64) at every (token, head) from R64's residual stream."""
    import torch
    out = []
    for l, h in zip(m.gpt_neox.layers, hs):
        qkv = l.attention.query_key_value(l.input_layernorm(h)).view(h.shape[0], -1, 3, 64)
        b = l.attention.query_key_value.bias.view(-1, 3, 64)[:, 1, ROT:]
        out.append((qkv[:, :, 0, ROT:] * b).sum(-1).abs().div(8.0).flatten())
    return out


def run(model: str, n_rows: int) -> Path:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    t0 = time.time()
    rows = torch.from_numpy(rows_of(IT, n_rows)[:, : T + 1]).to(dev)
    m64 = load(model, "R64", dev)
    L = m64.config.num_hidden_layers
    acc = {a: {"kl_I": [], "kl_Z": [], "d": [], "ce_I": [], "ce_Z": [], "fin_I": [], "fin_Z": [],
               "mI": [[] for _ in range(L + 1)], "mZ": [[] for _ in range(L + 1)],
               "mZR": [[] for _ in range(L + 1)], "tvI": [], "tvZ": [], "tvZR": []} for a in ARMS}
    g = {"d0": [], "m0": [[] for _ in range(L + 1)], "drot": [], "mrot": [[] for _ in range(L + 1)],
         "ce64": [], "ce_rot": [], "tv_rot": []}
    offs = [[] for _ in range(L)]
    timing = []
    for ai, arm in enumerate(ARMS):
        ma = load(model, arm, dev)
        for r in range(n_rows):
            tr = time.time()
            x, y = rows[r: r + 1, :T], rows[r, 1: T + 1]
            ref = fwd(m64, "R64", x, attn=True)
            if ai == 0:                           # the gates and the description, once a cell
                with Zero(m64, "n"):
                    z0 = fwd(m64, "R64", x, attn=False)
                g["d0"].append(kl(ref["lp"], z0["lp"]).cpu().numpy())
                for l, v in enumerate(rel(ref["hs"], z0["hs"])):
                    g["m0"][l].append(v.cpu().numpy())
                with Zero(m64, "r"):
                    zr = fwd(m64, "R64", x, attn=True)
                g["drot"].append(kl(ref["lp"], zr["lp"]).cpu().numpy())
                for l, v in enumerate(rel(ref["hs"], zr["hs"])):
                    g["mrot"][l].append(v.cpu().numpy())
                g["ce64"].append(ce(ref["lp"], y))
                g["ce_rot"].append(ce(zr["lp"], y))
                g["tv_rot"].append(tv(ref["at"], zr["at"]))
                with torch.no_grad():
                    for l, v in enumerate(offsets(m64, ref["hs"][:L])):
                        offs[l].append(v.float().cpu().numpy())
                del z0, zr
            aI = fwd(ma, arm, x, attn=True)
            with Zero(ma, "n"):
                aZ = fwd(ma, arm, x, attn=True)
            A = acc[arm]
            A["kl_I"].append(kl(ref["lp"], aI["lp"]).cpu().numpy())
            A["kl_Z"].append(kl(ref["lp"], aZ["lp"]).cpu().numpy())
            A["d"].append(kl(aI["lp"], aZ["lp"]).cpu().numpy())
            A["ce_I"].append(ce(aI["lp"], y))
            A["ce_Z"].append(ce(aZ["lp"], y))
            A["fin_I"].append(aI["finite"])
            A["fin_Z"].append(aZ["finite"])
            for l, (vi, vz) in enumerate(zip(rel(ref["hs"], aI["hs"]), rel(aI["hs"], aZ["hs"]))):
                A["mI"][l].append(vi.cpu().numpy())
                A["mZ"][l].append(vz.cpu().numpy())
            for l, v in enumerate(rel(ref["hs"], aZ["hs"])):   # beside, post hoc (after 70m was seen)
                A["mZR"][l].append(v.cpu().numpy())
            if aI["at"] is not None:
                A["tvI"].append(tv(ref["at"], aI["at"]))
                A["tvZ"].append(tv(aI["at"], aZ["at"]))
                A["tvZR"].append(tv(ref["at"], aZ["at"]))
            del ref, aI, aZ
            timing.append(time.time() - tr)
            if r == 0:
                print(json.dumps({"arm": arm, "row0_s": round(timing[-1], 1),
                                  "row0_d": float(A["d"][0].mean()), "row0_klI": float(A["kl_I"][0].mean()),
                                  "finite": [A["fin_I"][0], A["fin_Z"][0]]}), flush=True)
        del ma
        torch.cuda.empty_cache()

    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    med = lambda xs: float(np.median(cat(xs)))  # noqa: E731
    rec = {"model": model, "step": STEP, "input": {"shard": "document-00000-of-00020.bin",
           "iteration": IT, "rows": n_rows, "tokens_in": T}, "design": "932f195",
           "torch": torch.__version__, "wall_s": round(time.time() - t0, 1)}
    import transformers
    rec["transformers"] = transformers.__version__
    rec["gates"] = {"G0_d": float(cat(g["d0"]).mean()), "G0_d_max": float(cat(g["d0"]).max()),
                    "G0_m_max": max(med(v) for v in g["m0"]),
                    "G0_m_max_token": max(float(cat(v).max()) for v in g["m0"]),
                    "G1_drot": float(cat(g["drot"]).mean()),
                    "G1_m_rot": [med(v) for v in g["mrot"]],
                    "ce_R64": float(np.mean(g["ce64"])), "ce_R64_rotzero": float(np.mean(g["ce_rot"])),
                    "tv_rot": list(np.mean(g["tv_rot"], 0))}
    for arm in ARMS:
        A = acc[arm]
        out = {"e_I": float(cat(A["kl_I"]).mean()), "e_Z": float(cat(A["kl_Z"]).mean()),
               "d": float(cat(A["d"]).mean()), "d_median_token": float(np.median(cat(A["d"]))),
               "ce_I": float(np.mean(A["ce_I"])), "ce_Z": float(np.mean(A["ce_Z"])),
               "finite_I": all(A["fin_I"]), "finite_Z": all(A["fin_Z"]),
               "m_floor": [med(v) for v in A["mI"]], "m_zero": [med(v) for v in A["mZ"]],
               "m_zeroed_vs_exact": [med(v) for v in A["mZR"]]}   # beside, post hoc
        if A["tvI"]:
            out["tv_floor"] = list(np.mean(A["tvI"], 0))
            out["tv_zero"] = list(np.mean(A["tvZ"], 0))
            out["tv_zeroed_vs_exact"] = list(np.mean(A["tvZR"], 0))   # beside, post hoc
        rec[arm] = out
    W = tensors(model, STEP, [f"gpt_neox.layers.{l}.attention.query_key_value.bias" for l in range(L)])
    besd = []
    for l in range(L):
        o = cat(offs[l]).astype(np.float64)
        q50, mx = float(np.median(o)), float(o.max())
        besd.append({"layer": l, "kbn_norm": float(np.linalg.norm(
            W[f"gpt_neox.layers.{l}.attention.query_key_value.bias"].reshape(-1, 3, 64)[:, 1, ROT:])),
            "offset_median": q50, "offset_max": mx,
            "fp16_spacing_at_median": float(np.spacing(np.float16(q50))) if q50 < 65504 else math.inf,
            "fp32_spacing_at_median": float(np.spacing(np.float32(q50)))})
    rec["beside"] = besd
    rec["row_s_median"] = float(np.median(timing))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{model}.json"
    p.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"wrote": str(p), "wall_s": rec["wall_s"]}), flush=True)
    return p


def label(c: dict) -> dict:
    """§11's rule for one (cell, arm); ``c`` is the arm's record, ``g1`` whether G1 passed."""
    r = c["d"] / c["e_I"] if c["e_I"] > 0 else math.inf
    rh = max((z / f if f > 0 else (math.inf if z > 0 else 0.0))
             for z, f in zip(c["m_zero"], c["m_floor"]))
    if c["finite_I"] != c["finite_Z"] or r >= MATTERS or rh >= MATTERS:
        lab = "matters"
    elif r <= COSMETIC and rh <= COSMETIC:
        lab = "cosmetic"
    else:
        lab = "comparable"
    return {"r": r, "r_h": rh, "label": lab, "e_Z_over_e_I": c["e_Z"] / c["e_I"] if c["e_I"] > 0 else None,
            "dCE": c["ce_Z"] - c["ce_I"]}


def read() -> Path:
    cells = {}
    for model in ("pythia-70m", "pythia-410m"):
        p = OUT / f"{model}.json"
        if not p.exists():
            raise SystemExit(f"refuse: {p} missing (both cells are in the family)")
        rec = json.loads(p.read_text())
        gt = rec["gates"]
        g0 = gt["G0_d"] <= G0_KL and gt["G0_m_max"] <= G0_H
        g1 = gt["G1_drot"] >= G1_FACTOR * rec["R32"]["e_I"]
        out = {"G0": g0, "G1": g1, "G0_d": gt["G0_d"], "G0_m_max": gt["G0_m_max"],
               "G1_drot": gt["G1_drot"], "R32_e_I": rec["R32"]["e_I"], "rows": rec["input"]["rows"]}
        for arm in ARMS:
            lab = label(rec[arm])
            if not g0:
                lab["label"] = "refused (G0)"
            elif not g1 and lab["label"] == "cosmetic":
                lab["label"] = "unreadable (G1)"
            lab["d_over_drot"] = rec[arm]["d"] / gt["G1_drot"] if gt["G1_drot"] > 0 else None
            out[arm] = lab
        cells[model] = out
    r32 = [cells[m]["R32"]["label"] for m in cells]
    if any(l.startswith("refused") or l.startswith("unreadable") for l in r32):
        ans = "not answered: " + ", ".join(r32)
    elif any(l in ("matters", "comparable") for l in r32):
        ans = "matters to the readings here"
    else:
        ans = "cosmetic for the readings here"
    res = {"cells": cells, "blocked_162b": ans, "design": "932f195"}
    p = OUT / "read.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--rows", type=int, default=64)
    sp.add_parser("read")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.model, a.rows)
    else:
        read()
    return 0


if __name__ == "__main__":
    sys.exit(main())
