"""P-I1's relay-count null over the registered 19-step grid (PROJECT.md §3.4).

The sibling of `curve.py` and `behavioural.py`. Builds each battery prompt's
`PromptNullContext` ONCE — it depends only on tokenisation, not on the
checkpoint — then, for the axis PROJECT.md §3.1 registers (heads that carry a
relay anywhere in the raw 19-step sweep, "forming" heads under
`P_I1_RELAY_OWNER`), runs `p7_motifs.relay_count_null.null_envelope` at every
checkpoint and writes the per-head null mean/sd and the above-null EXCESS
(raw - null mean, clipped at 0) to `data/analysis/relay_null_series.json`.

PATHS ARE DERIVED, as in `curve.py` and `behavioural.py`: `METS_REPO` and
`METS_DATA` override. `METS_NULL_REPLICATES` overrides the replicate count
(default 100); each replicate reshuffles the whole battery table and reruns
`find_relays`, so the wall time scales linearly in it -- printed per step so a
run can be judged and killed early if the estimate is off.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.battery_structure import tokenize_prompt
from core.changepoint_colocation import REGISTERED_P_I1_SWEEP
from core.config import PROMPTS
from core.interactions import InteractionTable
from p7_motifs.formation_gate import P_I1_RELAY_OWNER
from p7_motifs.relay_count_null import build_prompt_context, null_envelope

STEPS = list(REGISTERED_P_I1_SWEEP)
N_REPLICATES = int(os.environ.get("METS_NULL_REPLICATES", "100"))
SEED = 20260904
TOKENIZER_ID = "EleutherAI/pythia-410m"


def _tokens_txt(run_dir: Path) -> list:
    """See `tools/run/behavioural.py`'s identical helper: p1_io.py's writer
    is f"{i:3d}  {tok}\\n", so `line[5:]` is the token verbatim."""
    return [line[5:] for line in (run_dir / "tokens.txt").read_text().splitlines()]


def run_dir_for(step: int, prompt: str) -> Path:
    hits = [
        p.parent
        for p in DATA.glob(f"phase12/*/pythia-410m-step{step}_{prompt}/attentions.npz")
        if not p.parent.parent.name.startswith("p2_eigenspectra_")
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"step {step} prompt {prompt!r}: {len(hits)} Phase 1 runs with an "
            f"attentions.npz, need exactly 1")
    return hits[0]


def prompt_ids(tokenizer, prompt: str, n_tokens: int, run_dir: Path) -> list:
    ids = tokenize_prompt(tokenizer, PROMPTS[prompt])["ids"]
    if len(ids) < n_tokens:
        raise SystemExit(
            f"{prompt!r}: tokeniser gave {len(ids)} ids, need >= {n_tokens}")
    ids = ids[:n_tokens]
    got = list(tokenizer.convert_ids_to_tokens(ids))
    want = _tokens_txt(run_dir)
    if got != want:
        raise SystemExit(
            f"{prompt!r}: re-tokenisation does not match {run_dir}/tokens.txt")
    return ids


def main() -> None:
    from transformers import AutoTokenizer

    series_path = DATA / "analysis" / "formation_series.json"
    raw = json.loads(series_path.read_text())
    if [int(s) for s in raw["steps"]] != STEPS:
        raise SystemExit(f"{series_path} is not on the registered sweep")
    raw_series = {
        tuple(int(x) for x in k.split(",")): [float(v) for v in vals]
        for k, vals in raw["series"][P_I1_RELAY_OWNER].items()
    }
    # PROJECT.md §3.1: the pre-filtered axis is every head that carries a
    # relay anywhere in the raw sweep, under the registered relay_owner.
    forming_heads = sorted(raw_series)
    print(f"{len(forming_heads)} forming heads under "
          f"relay_owner={P_I1_RELAY_OWNER!r}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    t0 = InteractionTable.load(DATA / "phase7" / f"step{STEPS[0]}" / "interaction_table.npz")
    table_prompts = sorted(set(t0.columns["prompt_key"].tolist()))
    del t0

    anchor_step = STEPS[-1]
    contexts = {}
    for p in table_prompts:
        rd = run_dir_for(anchor_step, p)
        with np.load(rd / "attentions.npz", allow_pickle=False) as z:
            n_tokens = int(z["attentions"].shape[-1])
        ids = prompt_ids(tokenizer, p, n_tokens, rd)
        contexts[p] = build_prompt_context(p, ids)
    print(f"built PromptNullContext for {sorted(contexts)} "
          f"(pool sizes {[contexts[p].pool_size for p in sorted(contexts)]})",
          flush=True)

    null_mean = {h: [0.0] * len(STEPS) for h in forming_heads}
    null_sd = {h: [0.0] * len(STEPS) for h in forming_heads}
    excess = {h: [0.0] * len(STEPS) for h in forming_heads}
    per_step = {}

    for i, s in enumerate(STEPS):
        path = DATA / "phase7" / f"step{s}" / "interaction_table.npz"
        t = InteractionTable.load(path)
        t0 = time.time()
        env = null_envelope(t, contexts, relay_owner=P_I1_RELAY_OWNER,
                            n_replicates=N_REPLICATES, seed=SEED + i,
                            heads=forming_heads)
        dt = time.time() - t0
        del t
        means = []
        for h in forming_heads:
            m = env.get(h, {"mean": 0.0, "sd": 0.0})
            null_mean[h][i] = m["mean"]
            null_sd[h][i] = m["sd"]
            excess[h][i] = max(raw_series[h][i] - m["mean"], 0.0)
            means.append(m["mean"])
        per_step[str(s)] = {"elapsed_seconds": round(dt, 1),
                            "null_mean_avg_over_heads": float(np.mean(means))}
        print(f"step{s:<7d} {dt:6.1f}s  raw_total={sum(raw_series[h][i] for h in forming_heads):>10.0f}  "
              f"null_mean_total={sum(means):>10.2f}  "
              f"excess_total={sum(excess[h][i] for h in forming_heads):>10.2f}",
              flush=True)

    out = {
        "_what_this_is": "P-I1's relay-count null (PROJECT.md §3.4) over "
                         "REGISTERED_P_I1_SWEEP, on the forming axis "
                         "(PROJECT.md §3.1): heads carrying a relay anywhere "
                         "in the raw sweep, under the registered relay_owner. "
                         "null_mean/null_sd are per replicate-set moments "
                         "from p7_motifs.relay_count_null.null_envelope; "
                         "above_null_excess = max(raw - null_mean, 0.0).",
        "steps": STEPS,
        "relay_owner": P_I1_RELAY_OWNER,
        "n_replicates": N_REPLICATES,
        "seed": SEED,
        "forming_heads": [f"{h[0]},{h[1]}" for h in forming_heads],
        "per_step": per_step,
        "null_mean": {f"{h[0]},{h[1]}": null_mean[h] for h in forming_heads},
        "null_sd": {f"{h[0]},{h[1]}": null_sd[h] for h in forming_heads},
        "above_null_excess": {f"{h[0]},{h[1]}": excess[h] for h in forming_heads},
    }
    outdir = Path(os.environ.get("METS_SCRATCH", str(DATA / "analysis")))
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "relay_null_series.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nWROTE {dest}")


if __name__ == "__main__":
    main()
