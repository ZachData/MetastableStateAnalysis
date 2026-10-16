"""
trackA_force/a2_batch.py — A2 unit 4: the batch over 18 steps and the ranking (`design-A2.md`
"Unit 4"; descriptive, tests nothing). STATE Blocked 58, the user's: (a1) "few" is read against the
init band, (b1) the field at A1a's ``fe_full`` layer βs (M1's beside at 143000 only), (c1) 64 nodes
and a refusal on G0.

- ``band``: G2's redraws of Pythia's init (50 seeds + a 51st, `a2_gate.g2`) read at **step 0's own
  layer βs** per prompt (A1a's records), every set statistic's 5–95 % range. A step's set is
  **few** if its size is below the band's 5 % (*placed* by 58 (a1)); ``|S| ≤ 43`` beside.
- ``run``: per step (143000 first) and v1 prompt, one hooked pass (`a2_comp`), the field at A1a's
  layer βs for that (step, prompt); at 143000 a second read at unit 2's M1 layer βs (beside).
  Refuses unless the 64-node gate passed on this code and the band was made by it; refuses a pass
  whose rows differ from the stored run, whose split or reconstruction is off, or (in the pass) a
  block whose path integral is off the chord.
- ``report``: per step, over v1's 7 prompts' median per component: ``S⁻``, ``S⁺``, ``S⁰`` and their
  reads against the band; and beside, the items carried to A2 (attention against the MLP, block 23
  at 143000, attention's drift-removed step, the sink and top-key parts, the off carrier).

    python -m trackA_force.a2_batch band   --out <dir> --gate <a2_gate.json>
    python -m trackA_force.a2_batch run    --out <dir> --gate <a2_gate.json> [--steps ...]
    python -m trackA_force.a2_batch report --out <dir>

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import a2_comp as ac
from .a1a import DATA, N_BLOCKS, STEPS, T_BLOCKS, V1, prompts_hash, run_dir

A1A_DIR = DATA / "trackA/a1a_batch_2026-10-15/records"
M1_FILE = DATA / "trackA/a2_beta_2026-10-16/refit_step143000.json"
FIRST = 143000
M1_STEP = 143000
RECON_TOL = 1e-4                    # the counted parts against the hidden states (the gate read ≤ 8.1e-8)
WARMUP = (512, 1000, 2000)          # LR warmup overlaps these (design "Fences")
ATTN = tuple(range(ac.N_HEADS)) + (ac.N_HEADS,)       # heads + b_O within a block's 18
MLP = len(ac.COMPS) - 1


def a1a_betas(step: int) -> Dict[str, List[float]]:
    out = {}
    for p in V1:
        f = A1A_DIR / f"step{step}_{p}.json"
        if not f.exists():
            raise SystemExit(f"refusing: no A1a record {f}")
        lb = json.loads(f.read_text())["beta"]["layer"]
        if len(lb) != N_BLOCKS or not np.all(np.isfinite(lb)):
            raise SystemExit(f"refusing: {f}: layer β not 24 finite values")
        out[p] = [float(b) for b in lb]
    return out


def m1_betas() -> Dict[str, List[float]]:
    r = json.loads(M1_FILE.read_text())
    out = {c["prompt"]: [float(b) for b in c["refit"]["m1/all"]["layer"]] for c in r["cells"]}
    if sorted(out) != sorted(V1):
        raise SystemExit(f"refusing: {M1_FILE} has prompts {sorted(out)}")
    return out


def check_gate(path: Path, code: str) -> Dict:
    g = json.loads(path.read_text())
    if g.get("code") != code or g.get("nodes") != ac.NODES or g.get("quick"):
        raise SystemExit(f"refusing: gate {path} is code {g.get('code')}, nodes {g.get('nodes')}, quick "
                         f"{g.get('quick')}; this is {code}, {ac.NODES} nodes")
    if not g["verdict"]["pass"]:
        raise SystemExit(f"refusing: gate {path} did not pass: {g['verdict']}")
    return g


def _model(step: int):
    import torch
    from core.models import load_model
    model, _ = load_model(f"pythia-410m-step{step}")
    if next(model.parameters()).dtype != torch.float32:
        raise SystemExit("refusing: model is not float32")
    return model


def _setup(device: str):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("refusing: no CUDA device")


# ---------------------------------------------------------------- band

def band(a) -> int:
    from . import a2_gate
    _setup(a.device)
    code = ac.code_sha()
    check_gate(Path(a.gate), code)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    prompts = V1[:1] if a.quick else V1
    betas = a1a_betas(0)
    g = a2_gate.g2(a.device, prompts, 3 if a.quick else a2_gate.N_SEEDS, {p: betas[p] for p in prompts})
    rec = {"code": code, "gate_code": a2_gate.gate_sha(), "prompts_hash": prompts_hash(), "quick": a.quick,
           "beta": "A1a fe_full layer βs at step 0, per prompt", "betas": betas, "nodes": ac.NODES,
           "seed": a2_gate.SEED, **g, "seconds": time.monotonic() - t0}
    (out / "a2_band.json").write_text(json.dumps(rec) + "\n")
    print({k: (v["lo"], v["hi"], v["step0"]) for k, v in g["range"].items()}, flush=True)
    return 0


def check_band(path: Path, code: str) -> Dict:
    from .a2_gate import gate_sha
    b = json.loads(path.read_text())
    if b.get("code") != code or b.get("gate_code") != gate_sha() or b.get("quick"):
        raise SystemExit(f"refusing: band {path} is code {b.get('code')} / {b.get('gate_code')}, quick "
                         f"{b.get('quick')}; this is {code} / {gate_sha()}")
    return b


# ---------------------------------------------------------------- run

def _read(hk, ids, ln, betas, rd: Path) -> Dict:
    from . import a2_gate
    from p1e_energy_field.u2_attn import MATCH_TOL
    hs, rows = hk.run_read(ids, ln, betas)
    match = a2_gate.stored_match(hs, rd)
    if match > MATCH_TOL["v1"]:
        raise SystemExit(f"refusing: {rd}: pass against stored unit rows {match:.1e} > {MATCH_TOL['v1']}")
    recon = ac.check_recon(hs, rows)
    if recon > RECON_TOL:
        raise SystemExit(f"refusing: {rd}: counted parts against the hidden states {recon:.1e}")
    blocks = {}
    for L in range(N_BLOCKS):
        r = rows[L]
        blocks[str(L)] = {k: np.asarray(r[k]).tolist() for k in ("proj", "proj_df", "off", "size", "disp")}
        blocks[str(L)].update({k: r[k] for k in ("g0", "total", "split_rel", "jstar", "jstar_tok",
                                                  "jstar_mass", "jstar_over_unif")})
    chk = {"match_unit": match, "recon": recon, "g0_max": max(r["g0"] for r in rows.values()),
           "split_max": max(r["split_rel"] for r in rows.values())}
    return {"betas": list(betas), "checks": chk, "blocks": blocks}


def check_populated(rec: Dict) -> None:
    b = rec["fe"]["blocks"]
    if sorted(map(int, b)) != list(range(N_BLOCKS)):
        raise SystemExit(f"refusing: blocks {sorted(b)}")
    P = np.array([b[str(L)]["proj"] for L in range(N_BLOCKS)])
    if P.shape != (N_BLOCKS, len(ac.COMPS) + len(ac.BESIDE)) or not np.all(np.isfinite(P)) or np.all(P == 0):
        raise SystemExit(f"refusing: proj {P.shape}, finite {np.isfinite(P).mean():.2f}, all zero {np.all(P == 0)}")
    off = np.array([sum(b[str(L)]["off"][:len(ac.COMPS)]) for L in range(N_BLOCKS)])
    if np.abs(off - 1).max() > 1e-6:
        raise SystemExit(f"refusing: off shares do not sum to 1 ({np.abs(off - 1).max():.1e})")


def run(a) -> int:
    import torch
    from transformers import AutoTokenizer
    from core.holdout import refuse_held_out
    from p1e_energy_field.u2_attn import ln_from, token_ids
    _setup("cuda")
    code, ph = ac.code_sha(), prompts_hash()
    check_gate(Path(a.gate), code)
    check_band(Path(a.out) / "a2_band.json", code)
    steps = [int(s) for s in a.steps] if a.steps else [FIRST] + [s for s in STEPS if s != FIRST]
    refuse_held_out([run_dir(s, p) for s in steps for p in V1], context="a2_batch")
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    m1 = m1_betas()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000")
    first = True
    for step in steps:
        todo = []
        for p in V1:
            f = out / "records" / f"step{step}_{p}.json"
            if f.exists():
                if json.loads(f.read_text()).get("code") != code:
                    raise SystemExit(f"refusing to resume: {f} was written by other code")
            else:
                todo.append(p)
        if not todo:
            print(f"have step {step}", flush=True)
            first = False
            continue
        t0 = time.monotonic()
        betas = a1a_betas(step)
        model = _model(step)
        ln = ln_from(model)
        hk = ac.comp_hooked(model)
        dev = next(model.parameters()).device
        for p in todo:
            rd = run_dir(step, p)
            man = json.loads((rd / "manifest.json").read_text())
            if man.get("hf_revision") != f"step{step}":
                raise SystemExit(f"refusing: {rd} manifest revision {man.get('hf_revision')}")
            ids = torch.tensor([token_ids(tok, rd)], device=dev)
            rec = {"step": step, "prompt": p, "run": str(rd), "code": code, "prompts_hash": ph,
                   "nodes": ac.NODES, "pass": "cuda:float32", "fe": _read(hk, ids, ln, betas[p], rd)}
            if step == M1_STEP:
                rec["m1"] = _read(hk, ids, ln, m1[p], rd)
            if first:
                check_populated(rec)
                print(f"first record populated; checks {rec['fe']['checks']}", flush=True)
                first = False
            f = out / "records" / f"step{step}_{p}.json"
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec) + "\n")
            tmp.rename(f)
            print(f"done step{step}_{p} {rec['fe']['checks']}", flush=True)
            del rec, ids
            gc.collect()
            torch.cuda.empty_cache()
        hk.close()
        del model, hk
        torch.cuda.empty_cache()
        print(f"step {step}: {time.monotonic() - t0:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------- report

def _arr(blocks: Dict, key: str, sl=slice(None)) -> np.ndarray:
    """(24, parts) for ``key``."""
    return np.array([blocks[str(L)][key][sl] for L in range(N_BLOCKS)], dtype=np.float64)


def _rows(blocks: Dict) -> Dict[int, Dict]:
    return {L: {"off": blocks[str(L)]["off"], "total": blocks[str(L)]["total"]} for L in range(N_BLOCKS)}


def step_summary(recs: Sequence[Dict], arm: str, band: Dict) -> Dict:
    """One step's reads over its prompts (median per component) for ``arm`` ∈ {fe, m1}."""
    k = len(ac.COMPS)
    proj = np.stack([_arr(r[arm]["blocks"], "proj", slice(0, k)).ravel() for r in recs])
    pdf = np.stack([_arr(r[arm]["blocks"], "proj_df", slice(0, k)).ravel() for r in recs])
    off = np.stack([ac.off_global(_rows(r[arm]["blocks"])) for r in recs])
    med = np.median(proj, 0)
    st = ac.set_stats(med, np.median(off, 0))
    rng = band["range"]
    for s in ("S_minus", "S_plus", "S_zero"):
        st[f"{s}_few"] = bool(st[s] < rng[s]["lo"])
        st[f"{s}_le43"] = bool(st[s] <= ac.FEW)
    T = list(T_BLOCKS)
    tb = lambda X, cols: float(np.median(X.reshape(len(recs), N_BLOCKS, k)[:, T][:, :, cols].sum((1, 2))))  # noqa: E731
    out = {"stats": st,
           "top_minus": [(ac.comp_name(i), float(med[i])) for i in np.argsort(med)[:5]],
           "top_plus": [(ac.comp_name(i), float(med[i])) for i in np.argsort(-med)[:5]],
           "top_off": [(ac.comp_name(i), float(v)) for i, v in
                       sorted(enumerate(np.median(off, 0)), key=lambda t: -t[1])[:5]],
           "attn_proj": tb(proj, list(ATTN)), "mlp_proj": tb(proj, [MLP]), "block_proj": tb(proj, list(range(k))),
           "attn_proj_df": tb(pdf, list(ATTN)), "mlp_proj_df": tb(pdf, [MLP]),
           "mlp_off": float(np.median(off.reshape(len(recs), N_BLOCKS, k)[:, :, MLP].sum(1))),
           "attn_off": float(np.median(off.reshape(len(recs), N_BLOCKS, k)[:, :, list(ATTN)].sum((1, 2))))}
    # the beside parts: sink (position 0) and top key, summed over heads, blocks 0–22
    H = ac.N_HEADS
    sink = np.stack([_arr(r[arm]["blocks"], "proj", slice(k, k + H)) for r in recs])
    top = np.stack([_arr(r[arm]["blocks"], "proj", slice(k + H, k + 2 * H)) for r in recs])
    sink_sz = np.stack([_arr(r[arm]["blocks"], "size", slice(k, k + H)).sum(1) for r in recs])
    tot_sz = np.stack([[r[arm]["blocks"][str(L)]["total"]["size"] for L in range(N_BLOCKS)] for r in recs])
    share = sink_sz / tot_sz                                                # Σ_h mean|sink_h| / mean|v|: an upper bound
    out.update({"sink_proj": float(np.median(sink[:, T].sum((1, 2)))),
                "topkey_proj": float(np.median(top[:, T].sum((1, 2)))),
                "sink_share_max": float(np.median(share.max(1))), "sink_share_med": float(np.median(np.median(share, 1)))})
    toks = [t for r in recs for L in range(N_BLOCKS) for t in r[arm]["blocks"][str(L)]["jstar_tok"]]
    vals, cnt = np.unique(toks, return_counts=True)
    out["topkey_mode"] = (int(vals[cnt.argmax()]), float(cnt.max() / len(toks)))
    b23 = np.median(proj.reshape(len(recs), N_BLOCKS, k)[:, 23], 0)
    out["block23"] = {"total": float(b23.sum()),
                      "top_minus": [(ac.COMPS[i], float(b23[i])) for i in np.argsort(b23)[:3]]}
    out["g0_max"] = max(r[arm]["checks"]["g0_max"] for r in recs)
    out["recon_max"] = max(r[arm]["checks"]["recon"] for r in recs)
    out["match_max"] = max(r[arm]["checks"]["match_unit"] for r in recs)
    return out


def _tokstr(tid: int) -> str:
    try:
        from transformers import AutoTokenizer
        return repr(AutoTokenizer.from_pretrained("EleutherAI/pythia-410m", revision="step143000").decode([tid]))
    except Exception:                                                       # noqa: BLE001
        return "?"


def report(out: Path) -> int:
    code = ac.code_sha()
    band = check_band(out / "a2_band.json", code)
    by: Dict[int, List[Dict]] = {}
    for f in (out / "records").glob("step*.json"):
        r = json.loads(f.read_text())
        if r.get("code") != code:
            raise SystemExit(f"refusing: {f} is code {r.get('code')}, this is {code}")
        by.setdefault(r["step"], []).append(r)
    steps = [s for s in STEPS if s in by]
    for s in steps:
        if sorted(r["prompt"] for r in by[s]) != sorted(V1):
            raise SystemExit(f"refusing: step {s} has {len(by[s])} prompts")
    summ = {s: step_summary(by[s], "fe", band) for s in steps}
    if M1_STEP in by:
        summ_m1 = step_summary(by[M1_STEP], "m1", band)
    rng = band["range"]
    L = [f"# A2 unit 4 (code {code}; band at step 0's βs, 5–95 %: "
         + ", ".join(f"{k} [{rng[k]['lo']:.3g}, {rng[k]['hi']:.3g}]" for k in ("S_minus", "S_plus", "S_zero", "top43_minus"))
         + f"; real step 0 in the band's pass: |S-| {rng['S_minus']['step0']})", "",
         "step | |S-| few(<band 5%) ≤43 | |S+| few | |S0| few | top43- | n<0 | gini | sum proj B0-22 (attn / mlp) | "
         "attn proj_df | off share (mlp / attn) | sink proj, share max | g0 max | warmup",
         "---|---|---|---|---|---|---|---|---|---|---|---|---"]
    for s in steps:
        x, st = summ[s], summ[s]["stats"]
        L.append(f"{s} | {st['S_minus']} {'few' if st['S_minus_few'] else 'coop'} {'y' if st['S_minus_le43'] else 'n'} | "
                 f"{st['S_plus']} {'few' if st['S_plus_few'] else 'coop'} | {st['S_zero']} {'few' if st['S_zero_few'] else 'coop'} | "
                 f"{st['top43_minus']:.3f} | {st['n_negative']} | {st['gini']:.2f} | "
                 f"{x['block_proj']:+.3f} ({x['attn_proj']:+.3f} / {x['mlp_proj']:+.3f}) | {x['attn_proj_df']:+.3f} | "
                 f"{x['mlp_off']:.2f} / {x['attn_off']:.2f} | {x['sink_proj']:+.4f}, {x['sink_share_max']:.3f} | "
                 f"{x['g0_max']:.1e} | {'warmup' if s in WARMUP else ''}")
    L += ["", "## Top of each set (median over prompts; proj per block step)", ""]
    for s in steps:
        x = summ[s]
        L.append(f"- {s}: against {', '.join(f'{n} {v:+.3f}' for n, v in x['top_minus'])}; "
                 f"along {', '.join(f'{n} {v:+.3f}' for n, v in x['top_plus'][:3])}; "
                 f"off {', '.join(f'{n} {v:.3f}' for n, v in x['top_off'][:3])}; "
                 f"top key mode {_tokstr(x['topkey_mode'][0])} ({x['topkey_mode'][1]:.2f} of head×block), "
                 f"top-key proj {x['topkey_proj']:+.4f}")
    if FIRST in summ:
        b = summ[FIRST]["block23"]
        L += ["", f"Block 23 at 143000: total {b['total']:+.3f}; most against "
              + ", ".join(f"{n} {v:+.3f}" for n, v in b["top_minus"])]
    if M1_STEP in by:
        st = summ_m1["stats"]
        L += ["", f"Beside, 143000 at M1's βs: |S-| {st['S_minus']}, |S+| {st['S_plus']}, |S0| {st['S_zero']}, "
              f"top43- {st['top43_minus']:.3f}, attn / mlp proj {summ_m1['attn_proj']:+.3f} / {summ_m1['mlp_proj']:+.3f}; "
              f"against {', '.join(f'{n} {v:+.3f}' for n, v in summ_m1['top_minus'][:3])}"]
    txt = "\n".join(L) + "\n"
    (out / "a2_batch_report.txt").write_text(txt)
    (out / "a2_batch_summary.json").write_text(json.dumps(
        {"code": code, "fe": {str(s): summ[s] for s in steps},
         "m1_143000": summ_m1 if M1_STEP in by else None}) + "\n")
    print(txt)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("band", "run", "report"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", nargs="*")
    ap.add_argument("--quick", action="store_true", help="band only: one prompt, 3 seeds (a check)")
    a = ap.parse_args(argv)
    if a.cmd == "band":
        return band(a)
    if a.cmd == "run":
        return run(a)
    return report(Path(a.out))


if __name__ == "__main__":
    raise SystemExit(main())
