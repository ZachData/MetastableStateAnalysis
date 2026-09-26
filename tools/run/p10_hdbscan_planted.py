"""Known-answer check: does HDBSCAN cluster exact duplicates planted in Gaussian noise?

Parked in `p10_cluster_function/handoff-10.md` ("How much of 'copy count' is
HDBSCAN's"), from `/challenge-pr` on #90. `status-10.md` §1.7 found that at
step 0 layer 0, 26 of 205 2-copy tokens are noise although their twin is the
identical vector, while every 3-5-copy group is clustered. That is a property
of the instrument if planted duplicates in structureless noise do the same.

THE PLANT
---------
``n_bg`` background points i.i.d. N(0, I_d), plus groups of exact copies of
further N(0, I_d) draws: ``GROUPS[c]`` groups of c copies each. Step 0's
embeddings are a random Gaussian init, so at layer 0 this is close to the real
input, not only a toy. The partition is the exact call
`p1_mstate_tracking.clustering.cluster_count_sweep` makes: L2-normalise, cosine
distance clipped at 0, ``hdbscan.HDBSCAN(min_cluster_size=2,
metric="precomputed")``.

Per copy count c, over seeds x groups:
- ``whole``: every copy has the same non-noise label and no other point has it
  (the group is its own cluster: the known answer).
- ``noise``: every copy is noise.
- ``merged``: all copies share one non-noise label that other points also carry.
- ``split``: copies carry more than one label (ties between identical points).
A second arm plants nothing (``N_BG + copies`` background points only), for the
count HDBSCAN makes of pure noise. Plus the background's clustered rate (the
known answer is 0: no structure),
and each cluster's make-up: how many planted groups it holds (0 = a cluster of
background points only).

WHAT IT DOES NOT SHOW: why a real run's twins are noise at a deeper layer or a
trained step, where twins are no longer identical. It only says whether
identical twins in noise are noise at the rate §1.7 saw.

TIER 1, EXPLORATORY, NOT REGISTERED. Needs the `hdbscan` package; refuses
without it (the §1.2 outage wrote empty partitions and silent zeros).

Run (conda `mets` env, which reproduces the sweep's partitions):
    python tools/run/p10_hdbscan_planted.py
"""
import argparse
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[2])))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))

import numpy as np
from core.metrics import cosine_distance_matrix

N_BG = 250
D = 1024
GROUPS = {2: 40, 3: 10, 4: 6, 5: 4}       # 80 + 30 + 24 + 20 = 154 copies; ~404 points
PARAMS = {"min_cluster_size": 2, "metric": "precomputed"}  # clustering.py's call


def plant(rng, n_bg: int = N_BG, d: int = D, groups: dict = GROUPS):
    """(points, group_id per point: -1 background, else the group index; copies per group)."""
    pts, gid, sizes = [rng.standard_normal((n_bg, d))], [np.full(n_bg, -1)], []
    for c, k in sorted(groups.items()):
        for _ in range(k):
            pts.append(np.repeat(rng.standard_normal((1, d)), c, axis=0))
            gid.append(np.full(c, len(sizes)))
            sizes.append(c)
    X, g = np.vstack(pts), np.concatenate(gid)
    perm = rng.permutation(len(g))             # order must not matter; do not let it help
    return X[perm], g[perm], sizes


def partition(X: np.ndarray) -> np.ndarray:
    import hdbscan
    return hdbscan.HDBSCAN(**PARAMS).fit_predict(cosine_distance_matrix(X))


def score(labels: np.ndarray, g: np.ndarray, sizes: list) -> dict:
    """{copies: {whole, noise, merged, split}} counts, background [n, n_clustered], planted groups per cluster."""
    out = {c: {"whole": 0, "noise": 0, "merged": 0, "split": 0} for c in sorted(set(sizes))}
    for i, c in enumerate(sizes):
        lab = labels[g == i]
        if (lab == -1).all():
            out[c]["noise"] += 1
        elif len(set(lab)) > 1:
            out[c]["split"] += 1
        elif (labels == lab[0]).sum() == c:
            out[c]["whole"] += 1
        else:
            out[c]["merged"] += 1
    bg = labels[g == -1]
    groups_per_cluster = [len(set(g[labels == k]) - {-1}) for k in set(labels) - {-1}]
    return {"groups": out, "background": [int(len(bg)), int((bg != -1).sum())],
            "groups_per_cluster": groups_per_cluster}


def run(seeds: int, n_bg: int = N_BG, d: int = D, groups: dict = GROUPS) -> dict:
    tot = {c: {"whole": 0, "noise": 0, "merged": 0, "split": 0} for c in groups}
    bg = [0, 0]
    make_up = {}
    n_clusters = []
    for s in range(seeds):
        X, g, sizes = plant(np.random.default_rng(s), n_bg, d, groups)
        labels = partition(X)
        r = score(labels, g, sizes)
        for c, v in r["groups"].items():
            for k in v:
                tot[c][k] += v[k]
        bg[0] += r["background"][0]
        bg[1] += r["background"][1]
        for k in r["groups_per_cluster"]:
            make_up[k] = make_up.get(k, 0) + 1
        n_clusters.append(int(len(set(labels)) - (1 if -1 in labels else 0)))
    return {"by_copies": {str(c): {**v, "n": sum(v.values()),
                                   "noise_rate": v["noise"] / sum(v.values())}
                          for c, v in sorted(tot.items())},
            "background": {"n": bg[0], "n_clustered": bg[1]},
            "clusters_by_planted_groups_held": {str(k): v for k, v in sorted(make_up.items())},
            "n_clusters": {"mean": float(np.mean(n_clusters)), "min": min(n_clusters),
                           "max": max(n_clusters), "n_groups_planted": sum(groups.values())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(DATA / "analysis" / "p10_hdbscan_planted.json"))
    args = ap.parse_args()
    try:
        hv = version("hdbscan")
    except Exception as e:                     # refuse, never write an empty partition
        sys.exit(f"hdbscan not installed ({e}); use the conda `mets` env")

    t0 = time.time()
    res = run(args.seeds)
    n_all = N_BG + sum(c * k for c, k in GROUPS.items())
    res_none = run(args.seeds, n_bg=n_all, groups={})
    record = {
        "schema": "p10_hdbscan_planted/1",
        "row": "known answer: exact duplicates planted in Gaussian noise, HDBSCAN as clustering.py calls it",
        "tier": "1 (exploratory, unregistered, descriptive)",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {"n_bg": N_BG, "d": D, "groups": GROUPS, "seeds": list(range(args.seeds)),
                  "rng": "numpy default_rng(seed)", "params": PARAMS},
        "toolchain": {"python": sys.version.split()[0], "hdbscan": hv,
                      "sklearn": version("scikit-learn"), "numpy": np.__version__},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],
        "result": res,
        "result_no_groups": {"n_bg": n_all, **res_none},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}  ({time.time() - t0:.0f}s, hdbscan {hv})")
    for c, v in res["by_copies"].items():
        print(f"  {c} copies: n={v['n']:4d}  whole {v['whole']:4d}  noise {v['noise']:4d}  "
              f"merged {v['merged']:4d}  split {v['split']:3d}  noise rate {v['noise_rate']:.3f}")
    b = res["background"]
    print(f"  background: {b['n_clustered']}/{b['n']} clustered;  clusters/run "
          f"{res['n_clusters']['mean']:.1f} (planted groups {res['n_clusters']['n_groups_planted']})")
    print(f"  clusters by planted groups held: {res['clusters_by_planted_groups_held']}")
    b = res_none["background"]
    print(f"  no groups, {n_all} points: clusters/run {res_none['n_clusters']['mean']:.1f} "
          f"(min {res_none['n_clusters']['min']}, max {res_none['n_clusters']['max']}); "
          f"{b['n_clustered']}/{b['n']} clustered")


if __name__ == "__main__":
    main()
