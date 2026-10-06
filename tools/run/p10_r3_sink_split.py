"""R3 beside A0's ladder: which half of c0 → c1 removes the raw flip, on c0's own labels
(`p10_cluster_function/status-10.md` §1.17). Per (step, prompt, layer), the sweep-pooled gap
(rest − members, raw and mask-corrected) on the stored labels:

  all      every position, no T4 (c0 as published)
  drop     T1–T2 positions out of the means, attention untouched
  drop+T4  T1–T2 out of the means, and their columns dropped with rows renormalised (T4)

No permutations: sizes only. Run:
    python tools/run/p10_r3_sink_split.py --labels <R0 labels> --out <file> [--jobs 14]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))

import numpy as np

from core.parking import mask_corrected_received, received_attention, relative_to_layer_mean
from tools.run.p10_attention_baseline import _load_attentions, noise_enrichment, clustered_enrichment, \
    t1_t2_positions, t4_vectors

ARMS = ("all", "drop", "drop+T4")


def gap(values, labels):
    return noise_enrichment(values, labels) - clustered_enrichment(values, labels)


def run(run_dir, labels_by_layer, dropped, key):
    attn = _load_attentions(Path(run_dir))
    out = []
    for layer, lab in sorted(labels_by_layer.items()):
        if layer >= attn.shape[0]:
            continue
        A, lab = attn[layer], np.asarray(lab)
        keep = np.ones(lab.size, bool)
        keep[dropped] = False
        raw = relative_to_layer_mean(received_attention(A, zero_diagonal=True))
        cor = mask_corrected_received(A)
        raw4, cor4 = t4_vectors(A, dropped)
        sink_noise = bool(all(lab[d] == -1 for d in dropped))
        row = {"key": key, "layer": int(layer), "t1_t2_all_noise": sink_noise}
        for arm, (r, c, m) in {"all": (raw, cor, np.ones_like(keep)), "drop": (raw, cor, keep),
                               "drop+T4": (raw4, cor4, keep)}.items():
            if (lab[m] == -1).all() or (lab[m] != -1).all():
                row[arm] = None
            else:
                row[arm] = [round(gap(r[m], lab[m]), 4), round(gap(c[m], lab[m]), 4)]
        out.append(row)
    return out


def main():
    from tools.run.p10_label_source import reader_input
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    dropped, ts = t1_t2_positions(a.labels)
    src = reader_input(a.labels, "c0")
    keys = sorted(k for k, v in src["labels"].items() if v)
    jobs = [(src["runs"][k], src["labels"][k], dropped[k[1]], f"{k[0]}|{k[1]}") for k in keys]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for rs in ex.map(run, *zip(*jobs)) for r in rs]
    by = {}
    for r in rows:
        by.setdefault(int(r["key"].split("|")[0]), []).append(r)

    def pooled(rs, arm):
        v = np.array([r[arm] for r in rs if r[arm] is not None])
        return [round(float(x), 4) for x in v.mean(0)] if v.size else None
    summ = {"sweep": {arm: pooled(rows, arm) for arm in ARMS},
            "by_step": {s: {arm: pooled(rs, arm) for arm in ARMS} for s, rs in sorted(by.items())},
            "t1_t2_all_noise_share": round(float(np.mean([r["t1_t2_all_noise"] for r in rows])), 4)}
    a.out.write_text(json.dumps({"labels": str(a.labels), "token_sets": ts, "t4_dropped": dropped,
                                 "summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ["sweep"]), "T1-T2 all noise in", summ["t1_t2_all_noise_share"], "of units")
    for s, v in summ["by_step"].items():
        print(s, v)


if __name__ == "__main__":
    main()
