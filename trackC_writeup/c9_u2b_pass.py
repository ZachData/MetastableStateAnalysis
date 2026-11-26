"""C9 unit 2b's producer: one GPU pass per (run, step) for E3, E4, E5 and, on 410m and the harness
arms, E6u, E6b, E6b′ (`design-C9.md` "U2b as it runs"). Its own module so the records' code hash
covers the producer alone (`/challenge-pr` on #382, finding 2); the reading is `c9_u2b.py`.

Per record (one step of one run):
- **E3** (D2 (a)): C4's induction instrument (`c4_close.heads`: `IND_N` = 8 random sequences of
  2 × `IND_LEN` tokens from `IND_SEED`, ids 1000–50000, one batch), every head's score **per
  sequence** (`c4_read.induction_scores`), so the bootstrap has units (C4's `heads_*.json` keep
  only the mean). Units = sequences, statistic = max over heads of their mean.
- **E4** (D2 (a)): per v1 prompt and head, the mean attention from i to i − 1, i ≥ 1 (A3's
  ``prev``, every head). Units = prompts, statistic = max over heads of their mean.
- **E5**: per v1 prompt, the minimum over hidden states of `core.metrics.effective_rank` of the
  states on the sphere (Phase 1's ``MinRank``, `effective_rank_normed`); every state's value kept.
- **E6** (410m, A0, A0′ only; D3 (a)): A3's `a3_ngram` KL(q ‖ p_model) on its 64 blocks, with A3
  unit 4's table (E6u, E6b) and table set′ (E6b′, and E6u′ beside), and the model's loss.

The prompts are tokenised as every energy record was (`c2_aniso.states`: truncation at 512), and
each prompt's ids hash is stored so the known answer can compare it with the run's records.
GPU, float32 weights, TF32 off; SVDs for E5 in float64 on the GPU.

    HF_HOME=... METS_DATA=<main>/data python -m trackC_writeup.c9_u2b_pass --run alias --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from trackA_force.a1a import DATA, REPO, V1

PROMPT_MAX = 512
A3NG = DATA / "trackA" / "a3" / "2026-10-20-A3u4-ngram"
A3NG_FULL = DATA / "trackA" / "a3" / "2026-10-20-A3u5-ngram-full"
CORPUS = DATA / "trackA" / "a3" / "corpus" / "val.jsonl.zst"
E6_RUNS = ("410m", "A0", "A0p")


class PassRefused(RuntimeError):
    pass


def code_sha() -> str:
    """This file and the producers it calls (not the reading)."""
    h = hashlib.sha256()
    for f in ("trackC_writeup/c9_u2b_pass.py", "trackA_force/a3_ngram.py", "core/metrics.py",
              "trackC_writeup/c4_read.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def steps_of(run: str):
    from trackC_writeup.c9_u2 import grid_of
    return grid_of(run).steps


def tokenizer(run: str):
    from transformers import AutoTokenizer
    if run in ("410m", "A0", "A0p"):
        from tools.run.behavioural import TOKENIZER_ID
        return AutoTokenizer.from_pretrained(TOKENIZER_ID)
    from trackC_writeup.c2_warmup import tokenizer as tg
    return tg()


def load(run: str, step: int, device):
    import torch
    from trackC_writeup import c9_u2
    if run == "410m":
        from trackB_circuit.b0_gate import load as l410
        return l410(step, str(device), torch.float32)[0]
    if run in c9_u2.GPT2_SEEDS:
        from trackC_writeup.c2_warmup import load as lg
        return lg(run, step, device)
    from trackC_writeup.c4_read import load_local
    root = {"A0": c9_u2.HDD / "c4", "A0p": c9_u2.HDD / "c4pc"}[run]
    return load_local(root / run / "ckpt" / f"step{step}", device)


def ids_sha(ids) -> str:
    return hashlib.sha256(str(list(ids)).encode()).hexdigest()[:12]


def rank_normed(h, torch) -> float:
    """`core.metrics.effective_rank(·, "normed")` on the GPU in float64."""
    x = torch.nn.functional.normalize(h.double(), p=2, dim=-1)
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    sv2 = torch.linalg.svdvals(x) ** 2
    tot = float(sv2.sum())
    if tot < 1e-12:
        return 1.0
    p = torch.clamp(sv2 / tot, min=1e-12)
    return float(torch.exp(-(p * torch.log(p)).sum()))


def ngram_setup(device, torch):
    from trackA_force.a3_ngram import prepare
    tok = tokenizer("410m")
    out = {}
    for name, (d, scored) in {"t": (A3NG, None), "full": (A3NG_FULL, A3NG)}.items():
        P = prepare(CORPUS, d, tok, scored)
        t = P["t"]
        out[name] = {"blocks": P["blocks"], "t": t, "q_u": torch.from_numpy(t["q_u"]).to(device),
                     "negH_b": np.array([[t["negH_b"][int(v)] for v in b] for b in P["blocks"]])}
    if not np.array_equal(out["t"]["blocks"], out["full"]["blocks"]):
        raise PassRefused("table set′'s blocks are not unit 4's")
    return out


def e6(m, ng, device, torch) -> dict:
    from trackA_force.a3_ngram import BLOCK, VOCAB, cross_terms
    blocks = ng["t"]["blocks"]
    rec = {"kl_u": [], "kl_b": [], "kl_u_full": [], "kl_b_full": [], "loss": []}
    with torch.no_grad():
        for k, b in enumerate(blocks):
            ids = torch.from_numpy(b).to(device)[None]
            logits = m(ids).logits[0]
            if logits.shape[-1] != VOCAB:
                raise PassRefused(f"{logits.shape[-1]} logits, A3's tables have {VOCAB}")
            logp = torch.log_softmax(logits.double(), dim=-1)
            for tag, T in (("", ng["t"]), ("_full", ng["full"])):
                cu, cb = cross_terms(logp, b, T["t"], T["q_u"])
                rec["kl_u" + tag].append(float((T["t"]["negH_u"] - cu.cpu().numpy()).mean()))
                rec["kl_b" + tag].append(float((T["negH_b"][k] - cb.cpu().numpy()).mean()))
            rec["loss"].append(float(-logp[torch.arange(BLOCK - 1, device=device), ids[0, 1:]].mean()))
    return rec


def run(run_id: str, out: Path) -> None:
    import torch
    from core.config import PROMPTS
    from trackC_writeup import c4_read as cr
    from trackC_writeup.c2_aniso import _torch_setup
    torch, device = _torch_setup()
    code = code_sha()
    d = out / "pass" / run_id
    d.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(run_id)
    encs = {p: tok(PROMPTS[p], return_tensors="pt", truncation=True, max_length=PROMPT_MAX) for p in V1}
    g = np.random.default_rng(cr.IND_SEED)
    half = g.integers(1000, 50000, (cr.IND_N, cr.IND_LEN))
    seqs = torch.from_numpy(np.concatenate([half, half], 1)).to(device)
    ng = ngram_setup(device, torch) if run_id in E6_RUNS else None
    for s in steps_of(run_id):
        f = d / f"step{s}.json"
        if f.exists():
            if json.loads(f.read_text()).get("code") != code:
                raise PassRefused(f"refusing to resume: {f} was written by other code")
            continue
        t0 = time.time()
        m = load(run_id, s, device)
        rec = {"run": run_id, "step": s, "code": code, "device": torch.cuda.get_device_name(0),
               "ids_sha": {}, "e4": {}, "e5": {}, "e5_states": {}}
        with torch.no_grad():
            att = m(input_ids=seqs, output_attentions=True).attentions
            A = torch.stack(att, 1).float().cpu().numpy()                 # (N, L, H, n, n)
            rec["e3"] = [cr.induction_scores(A[i], cr.IND_LEN).tolist() for i in range(cr.IND_N)]
            del att, A
            for p in V1:
                enc = {k: v.to(device) for k, v in encs[p].items()}
                o = m(**enc, output_hidden_states=True, output_attentions=True)
                rec["ids_sha"][p] = ids_sha(encs[p]["input_ids"][0].tolist())
                a = torch.stack([x[0] for x in o.attentions]).double()     # (L, H, n, n)
                n = a.shape[-1]
                i = torch.arange(1, n, device=device)
                rec["e4"][p] = a[:, :, i, i - 1].mean(-1).cpu().numpy().tolist()
                ranks = [rank_normed(h[0].float(), torch) for h in o.hidden_states]
                rec["e5_states"][p], rec["e5"][p] = ranks, float(min(ranks))
                del o, a
            if ng is not None:
                rec["e6"] = e6(m, ng, device, torch)
        vals = [np.ravel(rec["e3"]), np.ravel([rec["e4"][p] for p in V1]), list(rec["e5"].values())]
        if not all(np.all(np.isfinite(v)) for v in vals) or (ng and not np.all(np.isfinite(rec["e6"]["kl_b"]))):
            raise PassRefused(f"{run_id} step {s}: a value is not finite")
        del m
        torch.cuda.empty_cache()
        f.write_text(json.dumps(rec) + "\n")
        print(json.dumps({"run": run_id, "step": s, "s": round(time.time() - t0, 1)}), flush=True)
    print("done", flush=True)


def main(argv=None) -> int:
    from trackC_writeup.c9_u2 import RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=RUNS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    run(a.run, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
