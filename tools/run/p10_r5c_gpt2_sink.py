"""R5c: does 5c's attention flip on `gpt2-large` survive with T1–T2 out of the means?
(`p10_cluster_function/design-10.md` "R5c"; `status-10.md` §1.19.) On each stored run's own
HDBSCAN labels (the partition 5c read), R3's sink split per (prompt, layer):

  all      every position (5c as published)
  drop     T1–T2 out of the means, attention untouched (primary, raw)
  drop+T4  T1–T2 out of the means, their columns dropped and rows renormalised

T1–T2 per prompt: position 0 ∪ positions whose norm exceeds 10× the layer median at any
hidden-state layer 2–32, union over the trained and random runs. The prompt is the unit: a
prompt's gap is its mean over readable layers. Beside: the drop arm's corrected gap in R3's
position bins. No permutations: sign counts over prompts. Run:
    python tools/run/p10_r5c_gpt2_sink.py --trained <run batch> --random <run batch> --out <file>
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

import numpy as np

from core.parking import mask_corrected_received, population_enrichment, received_attention, \
    relative_to_layer_mean
from tools.run.p10_attention_baseline import _load_attentions
from tools.run.p10_r3_position_bins import gaps as binned_gaps
from tools.run.p10_r3_sink_split import ARMS, run as sink_split
from tools.run.p10_token_composition import OUTSIDE

MASSIVE_LAYERS = tuple(range(2, 33))  # Pythia's 2–20 of 24 hidden states, scaled to 36 blocks
N_PROMPTS, SURVIVES_AT = 21, 16       # one-sided sign test: P(≥ 16 of 21 | 1/2) = 0.013
KINDS = ("raw", "corrected")


def sign_p(k: int, n: int) -> float:
    """One-sided P(X ≥ k), X ~ Binomial(n, 1/2)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def sink_positions(norms_by_arm) -> list:
    """Position 0 ∪ every arm's massive positions (`move_text.massive_positions`, ratio 10)."""
    from p1d_cluster_ensemble.move_text import massive_positions  # needs hdbscan; runtime only
    out = {0}
    for norms in norms_by_arm:
        out |= set(massive_positions(np.asarray(norms), layers=MASSIVE_LAYERS))
    return sorted(int(p) for p in out)


def enrichments(run_dir, labels_by_layer) -> list:
    """5c's two ratios, every position, raw: [unclustered, clustered] per readable layer."""
    attn = _load_attentions(Path(run_dir))
    out = []
    for L, lab in sorted(labels_by_layer.items()):
        if L >= attn.shape[0]:
            continue
        lab = np.asarray(lab)
        if (lab == -1).all() or (lab != -1).all():
            continue
        r = relative_to_layer_mean(received_attention(attn[L], zero_diagonal=True))
        out.append([population_enrichment(r, lab == -1), population_enrichment(r, lab != -1)])
    return out


def binned(run_dir, labels_by_layer, dropped) -> list:
    """Drop arm, corrected: (pooled, binned) noise − clustered per readable layer."""
    attn = _load_attentions(Path(run_dir))
    out = []
    for L, lab in sorted(labels_by_layer.items()):
        if L >= attn.shape[0]:
            continue
        lab = np.asarray(lab).copy()
        lab[dropped] = OUTSIDE
        dom = lab[lab != OUTSIDE]
        if (dom == -1).all() or (dom != -1).all():
            continue
        out.append(binned_gaps(mask_corrected_received(attn[L]), lab))
    return out


def unit(run_dir, dropped, key):
    labels = {int(k): v for k, v in json.loads((Path(run_dir) / "hdbscan_labels.json").read_text()).items()}
    return {"key": key, "layers": sink_split(run_dir, labels, dropped, key),
            "enrichments": enrichments(run_dir, labels), "binned": binned(run_dir, labels, dropped)}


def prompt_gap(layers, arm: str, kind: str):
    """A prompt's gap: the mean over its readable layers, or ``None`` if none is."""
    i = KINDS.index(kind)
    v = [r[arm][i] for r in layers if r[arm] is not None]
    return float(np.mean(v)) if v else None


def read_arm(trained: dict, random: dict, arm: str, kind: str) -> dict:
    """Per-prompt gaps → the pre-stated reading. ``trained``/``random``: prompt → layer rows."""
    if set(trained) != set(random):
        raise ValueError(f"prompts differ: {sorted(set(trained) ^ set(random))}")
    t = {k: prompt_gap(v, arm, kind) for k, v in trained.items()}
    r = {k: prompt_gap(v, arm, kind) for k, v in random.items()}
    keys = sorted(k for k in t if t[k] is not None and r[k] is not None)
    n = len(keys)
    t_pos = sum(t[k] > 0 for k in keys)
    d_pos = sum(t[k] - r[k] > 0 for k in keys)
    t_mean, r_mean = float(np.mean([t[k] for k in keys])), float(np.mean([r[k] for k in keys]))
    return {"n": n, "trained_mean": round(t_mean, 4), "random_mean": round(r_mean, 4),
            "trained_positive": t_pos, "trained_minus_random_positive": d_pos,
            "p_trained": round(sign_p(t_pos, n), 4), "p_paired": round(sign_p(d_pos, n), 4),
            "verdict": verdict(n, t_mean, t_pos, d_pos, r_mean),
            "per_prompt": {k: [round(t[k], 4), round(r[k], 4)] for k in keys}}


def verdict(n: int, t_mean: float, t_pos: int, d_pos: int, r_mean: float) -> str:
    if n != N_PROMPTS:
        raise ValueError(f"the rule is fixed for {N_PROMPTS} prompts, got {n}")
    if t_mean > 0 and t_pos >= SURVIVES_AT and d_pos >= SURVIVES_AT:
        return "sign-flipped as 5c stated" if r_mean < 0 else "survives"
    return "does not survive"


def by_third(rows: dict, arm: str, kind: str, n_layers: int = 36) -> list:
    """Mean gap over every (prompt, layer) unit in each third of depth."""
    i, out = KINDS.index(kind), []
    for lo in range(0, n_layers, n_layers // 3):
        v = [r[arm][i] for rs in rows.values() for r in rs
             if r[arm] is not None and lo <= r["layer"] < lo + n_layers // 3]
        out.append(round(float(np.mean(v)), 4) if v else None)
    return out


def prompt_dirs(batch: Path, model: str) -> dict:
    out = {p.name[len(model) + 1:]: p for p in sorted(batch.glob(f"{model}_*")) if p.is_dir()}
    if not out:
        raise SystemExit(f"refusing: no {model}_* run dirs in {batch}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trained", type=Path, required=True)
    ap.add_argument("--random", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    td, rd = prompt_dirs(a.trained, "gpt2-large"), prompt_dirs(a.random, "gpt2-large-random")
    if set(td) != set(rd) or len(td) != N_PROMPTS:
        raise SystemExit(f"refusing: prompt sets differ or are not {N_PROMPTS}: {sorted(td)} / {sorted(rd)}")
    dropped, inputs = {}, {}
    for k in sorted(td):
        toks = [(d / "tokens.txt").read_bytes() for d in (td[k], rd[k])]
        if toks[0] != toks[1]:
            raise SystemExit(f"refusing: {k}'s tokens differ between the arms")
        dropped[k] = sink_positions([np.load(d / "activations.npz")["norms"] for d in (td[k], rd[k])])
        inputs[k] = {arm: {f: hashlib.sha256((d / f).read_bytes()).hexdigest()[:12]
                           for f in ("hdbscan_labels.json", "attentions.npz")}
                     for arm, d in (("trained", td[k]), ("random", rd[k]))}
    jobs = [(str(d[k]), dropped[k], f"{m}|{k}") for m, d in (("trained", td), ("random", rd)) for k in sorted(d)]
    with ProcessPoolExecutor(a.jobs) as ex:
        units = list(ex.map(unit, *zip(*jobs)))
    rows = {m: {u["key"].split("|")[1]: u["layers"] for u in units if u["key"].startswith(m + "|")}
            for m in ("trained", "random")}
    reading = {f"{arm}|{kind}": read_arm(rows["trained"], rows["random"], arm, kind)
               for arm in ARMS for kind in KINDS}
    extra = {}
    for m in ("trained", "random"):
        us = [u for u in units if u["key"].startswith(m + "|")]
        e = np.array([x for u in us for x in u["enrichments"]])
        b = np.array([x for u in us for x in u["binned"] if x[1] is not None])
        lr = [r for u in us for r in u["layers"]]
        extra[m] = {"all_raw_enrichment": [round(float(x), 4) for x in e.mean(0)],
                    "t1_t2_all_noise_share": round(float(np.mean([r["t1_t2_all_noise"] for r in lr])), 4),
                    "drop_raw_by_third": by_third(rows[m], "drop", "raw"),
                    "drop_corrected_pooled_vs_binned": [round(float(x), 4) for x in b.mean(0)],
                    "n_units": len(lr), "n_binned": int(len(b))}
    a.out.write_text(json.dumps({"trained": str(a.trained), "random": str(a.random),
                                 "massive_layers": [MASSIVE_LAYERS[0], MASSIVE_LAYERS[-1]],
                                 "t1_t2": dropped, "inputs": inputs, "reading": reading,
                                 "beside": extra, "rows": rows}, indent=1))
    for k, v in reading.items():
        print(k, {x: v[x] for x in v if x != "per_prompt"})
    print(json.dumps(extra, indent=1))


if __name__ == "__main__":
    main()
