"""Does b_K's floor reach a reported number? A5's V at 143000, KBn zeroed (`MATH_TRAINING_SYMMETRY.md` §12; tier 1).

WHY THIS EXISTS
---------------
§11 found that in the float32 eager graph every reading here uses, the non-rotary key bias (KBn,
dims 16-63 of each head's key bias) is most of the rounding error in the top layers (≈ 1e-4
relative), although it is invisible in exact arithmetic. It did not re-run any reading. §12,
frozen at `a7c7cda` before any output, re-runs one recorded threshold count: A5's seed-0 record
at 143000 (`trackA_force/a5_seeds.py` `one_pass` on `pythia-410m`, v1's 7 prompts), whose V at
rel_tol 1e-3 is the panel's `p1_energy_nviol_b1` (70). Arms: R32,I (the record, run twice),
R32,Z (KBn = 0), R64,I / R64,Z (float64 throughout), and R32,s (KBn × s, the planted effect).

PARTS
-----
    run     every arm; gate K refuses early; records data/ts/e0bk_a5/arms.json
    read    §12's gates, classes, s* and the answer to Blocked 163 (b) -> read.json

    python tools/ts_e0_bk_reading.py run && python tools/ts_e0_bk_reading.py read
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ts_e0_backward import MAIN, HUB  # noqa: E402
from tools.ts_e0_bk_zero import kbias  # noqa: E402
from tools.ts_e0_backward import ROT  # noqa: E402

OUT = MAIN / "data/ts/e0bk_a5"
REC = MAIN / "data/trackA/a5/2026-10-20-u1/seed0/records"
REPO_ID, STEP = "EleutherAI/pythia-410m", 143000
TAUS = (1e-3, 1e-4, 1e-2)                          # primary first; A5's SENS_TOLS beside
LADDER = (-1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
K_E1, G0_X = 1e-6, 1e-12
S_DEFAULT = 4                                      # s* <= 4: 70m's KBn is 3.3x 410m's


class Scale:
    """Context: KBn (dims ROT.. of every layer's key bias) multiplied by ``s`` (0 zeroes it)."""
    def __init__(self, m, s: float):
        self.b, self.s, self.saved = kbias(m), s, None

    def __enter__(self):
        import torch
        with torch.no_grad():
            self.saved = [b[:, 1, ROT:].clone() for b in self.b]
            for b in self.b:
                b[:, 1, ROT:] *= self.s
        return self

    def __exit__(self, *exc):
        import torch
        with torch.no_grad():
            for b, sv in zip(self.b, self.saved):
                b[:, 1, ROT:] = sv
            assert all(torch.equal(b[:, 1, ROT:], sv) for b, sv in zip(self.b, self.saved))


def load(dtype):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    m = from_pretrained_eager(GPTNeoXForCausalLM, REPO_ID, revision=f"step{STEP}", cache_dir=str(HUB),
                              local_files_only=True, torch_dtype=dtype).to("cuda").eval()
    assert next(m.parameters()).dtype == dtype and m.config.rotary_pct == 0.25
    return m


def ln_of(m) -> dict:
    return {"w": [l.input_layernorm.weight.detach().double().cpu().numpy() for l in m.gpt_neox.layers],
            "b": [l.input_layernorm.bias.detach().double().cpu().numpy() for l in m.gpt_neox.layers],
            "eps": float(m.gpt_neox.layers[0].input_layernorm.eps)}


def pass64(model, tok, prompt: str, device) -> dict:
    """`a5_seeds.one_pass` with float64 kept after the model (no float32 cast; E₁ in float64);
    e35 is not read here."""
    import hashlib
    import torch
    from core.battery_structure import induction_candidates
    from core.config import PROMPTS
    from core.metrics import interaction_energies_batched
    from trackA_force.a5_seeds import MAX_TOKENS
    enc = tok(PROMPTS[prompt], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    ids = enc["input_ids"][0].tolist()
    with torch.no_grad():
        o = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True,
                  output_attentions=True)
    e1 = []
    for h in o.hidden_states:
        U = torch.nn.functional.normalize(h[0].double(), p=2, dim=-1).cpu().numpy()
        e1.append(interaction_energies_batched(U @ U.T, [1.0])[1.0])
    A = torch.stack([a[0] for a in o.attentions]).double()
    n = A.shape[-1]
    i = torch.arange(1, n, device=A.device)
    prev = A[:, :, i, i - 1].mean(-1)
    pairs = induction_candidates(ids, strict=True)
    q = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=A.device)
    k = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=A.device)
    ind = A[:, :, q, k].sum(-1) if len(pairs) else torch.zeros(A.shape[:2], dtype=A.dtype)
    return {"prompt": prompt, "n": n, "ids_sha": hashlib.sha256(str(ids).encode()).hexdigest()[:12],
            "e1": e1, "prev": prev.cpu().numpy().tolist(), "ind": ind.cpu().numpy().tolist(),
            "n_ind": len(pairs)}


def slim(rec: dict) -> dict:
    return {k: rec[k] for k in ("prompt", "n", "ids_sha", "e1", "prev", "ind", "n_ind")}


# ---------------------------------------------------------------- pure

def xs(arm: dict) -> np.ndarray:
    """(7, 24): the relative drops (E_prev - E_curr) / |E_prev| the count thresholds."""
    from trackA_force.a1a import V1
    out = []
    for p in V1:
        e = np.asarray(arm[p]["e1"], np.float64)
        out.append(-np.diff(e) / np.maximum(np.abs(e[:-1]), 1e-12))
    return np.array(out)


def v_of(arm: dict, tau: float) -> int:
    from trackA_force.a1a import V1
    from trackA_force.a5_seeds import n_viol
    return int(sum(n_viol(arm[p]["e1"], tau) for p in V1))


def picks_of(arm: dict) -> dict:
    from trackA_force.a5_seeds import head_scores, pick
    sc = head_scores({(STEP, p): r for p, r in arm.items()})
    return {k: pick(sc[k]) for k in ("prev", "ind")}


def klass(a_i, a_z, a_64) -> str:
    if a_i == a_z == a_64:
        return "holds"
    if a_z != a_64:
        return "floor-fragile"
    return "b_K reaches it"


# ---------------------------------------------------------------- run

def run_arm(m, tok, ln, scale, device) -> dict:
    from trackA_force.a1a import V1
    from trackA_force.a5_seeds import check_populated, one_pass
    ctx = Scale(m, scale) if scale is not None else None
    out = {}
    if ctx:
        ctx.__enter__()
    try:
        for p in V1:
            r = one_pass(m, tok, p, ln, device)
            r["step"] = STEP
            check_populated(r)
            out[p] = slim(r)
    finally:
        if ctx:
            ctx.__exit__(None, None, None)
    return out


def run(_a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer
    from trackA_force.a1a import V1
    from trackA_force.a5_seeds import TOKENIZER
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device")
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    arms, secs = {}, {}

    m = load(torch.float32)
    ln = ln_of(m)
    for name, s in [("R32,I", None), ("R32,I#2", None)]:
        t = time.time()
        arms[name] = run_arm(m, tok, ln, s, dev)
        secs[name] = time.time() - t
        if name == "R32,I":                         # gate K, early: refuse before spending more
            rec = {p: json.loads((REC / f"step{STEP}_{p}.json").read_text()) for p in V1}
            e_rel = max(float(np.max(np.abs(np.asarray(arms[name][p]["e1"]) - rec[p]["e1"])
                                     / np.abs(rec[p]["e1"]))) for p in V1)
            print(f"first arm populated: V_1e-3 {v_of(arms[name], 1e-3)} (recorded "
                  f"{v_of(rec, 1e-3)}), E1 max rel diff vs record {e_rel:.2e}", flush=True)
            if e_rel > K_E1 or any(v_of(arms[name], t) != v_of(rec, t) for t in TAUS):
                raise SystemExit("refusing: gate K, R32,I is not the recorded graph")
    for name, s in [("R32,Z", 0.0)] + [(f"R32,{s}", float(s)) for s in LADDER]:
        t = time.time()
        arms[name] = run_arm(m, tok, ln, s, dev)
        secs[name] = time.time() - t
        print(f"{name}: V_1e-3 {v_of(arms[name], 1e-3)} ({secs[name]:.0f} s)", flush=True)
    del m
    torch.cuda.empty_cache()

    torch.set_default_dtype(torch.float64)          # HF 4.44 builds its mask in it (as §11's R64)
    m = load(torch.float64)
    for name, s in [("R64,I", None), ("R64,Z", 0.0), ("R64,1024", 1024.0)]:
        t = time.time()
        ctx = Scale(m, s) if s is not None else None
        if ctx:
            ctx.__enter__()
        try:
            arms[name] = {p: pass64(m, tok, p, dev) for p in V1}
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
        secs[name] = time.time() - t
        print(f"{name}: V_1e-3 {v_of(arms[name], 1e-3)} ({secs[name]:.0f} s)", flush=True)
    torch.set_default_dtype(torch.float32)

    meta = {"design": "MATH_TRAINING_SYMMETRY.md §12 (frozen a7c7cda)", "repo": REPO_ID, "step": STEP,
            "record": str(REC.relative_to(MAIN)), "prompts": list(V1), "tokenizer": list(TOKENIZER),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "device": torch.cuda.get_device_name(0), "tf32": False, "ladder": list(LADDER),
            "seconds": secs, "gpu_seconds": sum(secs.values())}
    (OUT / "arms.json").write_text(json.dumps({"meta": meta, "arms": arms}) + "\n")
    print(f"wrote {OUT / 'arms.json'} ({sum(secs.values()):.0f} s of passes)", flush=True)
    return 0


# ---------------------------------------------------------------- read

def read(_a) -> int:
    from trackA_force.a1a import V1
    d = json.loads((OUT / "arms.json").read_text())
    A = d["arms"]
    rec = {p: json.loads((REC / f"step{STEP}_{p}.json").read_text()) for p in V1}
    x = {k: xs(v) for k, v in A.items()}
    V = {k: {f"{t:g}": v_of(v, t) for t in TAUS} for k, v in A.items()}
    P = {k: picks_of(v) for k, v in A.items()}
    Vrec, Prec = {f"{t:g}": v_of(rec, t) for t in TAUS}, picks_of(rec)

    e_rel = max(float(np.max(np.abs(np.asarray(A["R32,I"][p]["e1"]) - rec[p]["e1"]) / np.abs(rec[p]["e1"])))
                for p in V1)
    ids_ok = all(A["R32,I"][p]["ids_sha"] == rec[p]["ids_sha"] for p in V1)
    heads = lambda pk: {k: v["head"] for k, v in pk.items()}  # noqa: E731
    K = {"ids_equal": ids_ok, "e1_max_rel_diff": e_rel, "tol": K_E1,
         "V_equal": V["R32,I"] == Vrec, "picks_equal": heads(P["R32,I"]) == heads(Prec),
         "twice_bit_identical": bool(np.array_equal(x["R32,I"], x["R32,I#2"])), "V_recorded": Vrec}
    K["pass"] = bool(ids_ok and e_rel <= K_E1 and K["V_equal"] and K["picks_equal"] and K["twice_bit_identical"])
    g0_dx = float(np.max(np.abs(x["R64,Z"] - x["R64,I"])))
    G0 = {"max_dx": g0_dx, "tol": G0_X, "V_equal": V["R64,Z"] == V["R64,I"],
          "picks_equal": heads(P["R64,Z"]) == heads(P["R64,I"])}
    G0["pass"] = bool(g0_dx <= G0_X and G0["V_equal"] and G0["picks_equal"])
    moved = [s for s in LADDER if s >= 2 and V[f"R32,{s}"]["0.001"] != V["R32,I"]["0.001"]]
    G1 = {"pass": bool(moved), "s_star": (min(moved) if moved else None),
          "ladder_V": {str(s): V[f"R32,{s}"] for s in LADDER},
          "R64_1024_vs_R64_I_max_dx": float(np.max(np.abs(x["R64,1024"] - x["R64,I"]))),
          "R64_1024_V": V["R64,1024"]}

    dZ = float(np.max(np.abs(x["R32,Z"] - x["R32,I"])))
    dI = float(np.max(np.abs(x["R32,I"] - x["R64,I"])))
    dZ64 = float(np.max(np.abs(x["R32,Z"] - x["R64,I"])))
    dflip = float(np.max(np.abs(x["R32,-1"] - x["R64,I"])))
    per_tau = {}
    for t in TAUS:
        k = f"{t:g}"
        mu = float(np.min(np.abs(x["R32,I"] - t)))
        flips = [{"prompt": V1[i], "transition": int(j) + 1, "x_R32I": float(x["R32,I"][i, j]),
                  "x_R32Z": float(x["R32,Z"][i, j]), "x_R64I": float(x["R64,I"][i, j])}
                 for i, j in zip(*np.nonzero(((x["R32,I"] > t) != (x["R32,Z"] > t))
                                             | ((x["R32,I"] > t) != (x["R64,I"] > t))))]
        per_tau[k] = {"V": {a: V[a][k] for a in ("R32,I", "R32,Z", "R64,I", "R32,-1")},
                      "class": klass(V["R32,I"][k], V["R32,Z"][k], V["R64,I"][k]),
                      "mu": mu, "mu_over_dI": mu / dI if dI else None, "mu_over_dZ": mu / dZ if dZ else None,
                      "flips": flips}
    pk = {}
    for kind in ("prev", "ind"):
        a = [P[n][kind]["head"] for n in ("R32,I", "R32,Z", "R64,I")]
        gap = P["R32,I"][kind]["score"] - P["R32,I"][kind]["runner_up_score"]
        pk[kind] = {"heads": dict(zip(("R32,I", "R32,Z", "R64,I"), a)), "class": klass(*a),
                    "gap_top_runner_up": gap, "runner_up": P["R32,I"][kind]["runner_up"],
                    "score_dZ": abs(P["R32,Z"][kind]["score"] - P["R32,I"][kind]["score"]),
                    "score_dI": abs(P["R32,I"][kind]["score"] - P["R64,I"][kind]["score"])}

    if not K["pass"]:
        answer = "refused: gate K (not the recorded graph, or not deterministic)"
    elif not G0["pass"]:
        answer = "refused: gate G0 (the zeroing touched a visible coordinate)"
    else:
        c = per_tau["0.001"]["class"]
        answer = {"holds": "below the reported number here", "b_K reaches it": "reaches a reported number",
                  "floor-fragile": "the count is float32-sensitive whatever b_K does"}[c]
        if c == "holds" and not G1["pass"]:
            answer += " (holds, and insensitive to b_K at x1024)"
    s_star = G1["s_star"]
    default = bool(K["pass"] and G0["pass"] and (per_tau["0.001"]["class"] == "b_K reaches it"
                                                  or (s_star is not None and s_star <= S_DEFAULT)))
    res = {"meta": d["meta"], "K": K, "G0": G0, "G1": G1,
           "x": {"dZ": dZ, "dI": dI, "eZ_over_eI": dZ64 / dI if dI else None, "flip_vs_exact": dflip},
           "per_tau": per_tau, "picks": pk, "answer_163b": answer, "zero_KBn_by_default": default}
    (OUT / "read.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("K", "G0", "x", "answer_163b", "zero_KBn_by_default")}, indent=1))
    print("s*:", s_star, "| per tau:", {k: (v["class"], v["V"]) for k, v in per_tau.items()})
    print("picks:", {k: (v["class"], v["heads"]["R32,I"], f"gap {v['gap_top_runner_up']:.3g}") for k, v in pk.items()})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(f=run)
    sub.add_parser("read").set_defaults(f=read)
    a = ap.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    raise SystemExit(main())
