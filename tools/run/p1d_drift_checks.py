"""Two checks on the drift record (`status-1d.md` "Float-noise drift"), after
/challenge-pr on #101. Tier 1, no null.

1. ``--fine``: agglomerative's pick per record. k, singleton share, and the
   share of its non-singleton clusters that are a single repeated token (all
   members the same token string). If the fine partitions are mostly singletons
   plus same-token groups, their stability is trivial.
2. ``--swap STEP PROMPT LAYER``: rebuild the consensus from sweep A's families,
   swapping in one family's labels and weight from sweep B, and report
   ARI(rebuilt, A's consensus) for each family swapped alone. This names which
   family's drift the consensus carries.
3. ``--baseline``: Phase 10 §3's stored-label comparison, Stage 0 against the
   pilot, per prompt, counted both ways: label vectors not identical (§3's
   rule, all layers) and ARI < 1 at layers >= 1. Reads only
   ``hdbscan_labels.json``; roots are ignored.
4. ``--matched PROMPT``: centroid and linkage families forced to HDBSCAN's k
   on both sweeps (the Parked item from /challenge-pr on #101).
5. ``--ties PROMPT``: near-duplicate pairs and nearest-neighbour ties per record,
   and where HDBSCAN's A-vs-B co-membership disagreements sit in distance.

    python -m tools.run.p1d_drift_checks <root_a> <root_b> --fine
    python -m tools.run.p1d_drift_checks <root_a> <root_b> --swap 32 repeated_tokens 6
    METS_DATA=<main>/data python -m tools.run.p1d_drift_checks - - --baseline
    python -m tools.run.p1d_drift_checks <root_a> <root_b> --matched repeated_tokens
    python -m tools.run.p1d_drift_checks <root_a> <root_b> --ties repeated_tokens
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

from p1d_cluster_ensemble import ensemble


def _res(root: Path, run: str) -> dict:
    return json.loads((root / run / "p1d_results.json").read_text())


def _tokens(res: dict) -> list[str]:
    # tokens.txt is "<index, right-aligned>  <token>"; the token keeps its own spaces.
    lines = (Path(res["run_dir"]) / "tokens.txt").read_text().split("\n")
    return [re.match(r"^\s*\d+  (.*)$", ln).group(1) for ln in lines if ln.strip()]


def fine(a: Path) -> None:
    print(f"{'step':>6s} {'prompt':16s} {'L':>2s} {'n':>4s} {'k':>4s} {'single':>6s} {'same-token':>10s}  (share of non-singleton clusters)")
    ks = []
    for run in sorted(p.name for p in a.iterdir() if (p / "p1d_results.json").exists()):
        res = _res(a, run)
        tok = _tokens(res)
        for L, pl in sorted(res["per_layer"].items(), key=lambda t: int(t[0])):
            s = pl["selection"]["agglomerative"]
            if s.get("selected") is None:
                continue
            lab = np.asarray(s["selected_labels"])
            size = Counter(lab.tolist())
            multi = [c for c, m in size.items() if m > 1]
            same = sum(len({tok[i] for i in np.flatnonzero(lab == c)}) == 1 for c in multi)
            single = sum(m == 1 for m in size.values()) / len(lab)
            ks.append((len(size), single, same / max(1, len(multi))))
            print(f"{run.split('step')[1].split('_')[0]:>6s} {run.split('_', 2)[2][:16]:16s} {L:>2s} "
                  f"{len(lab):4d} {len(size):4d} {single:6.2f} {same / max(1, len(multi)):10.2f}")
    k = np.array(ks)
    print(f"\n{len(k)} records: k median {np.median(k[:, 0]):.0f}, singleton share median {np.median(k[:, 1]):.2f}, "
          f"same-token share of multi-member clusters median {np.median(k[:, 2]):.2f}")


def swap(a: Path, b: Path, step: str, prompt: str, layer: str) -> None:
    run = f"pythia-410m-step{step}_{prompt}"
    pa, pb = (_res(r, run)["per_layer"][layer] for r in (a, b))
    lab = lambda pl: {f: np.asarray(s["selected_labels"]) for f, s in pl["selection"].items()
                      if s.get("selected") is not None and f in pl["ensemble"]["families"]}
    la, lb = lab(pa), lab(pb)
    wa, wb = pa["ensemble"]["weights"], pb["ensemble"]["weights"]
    base = ensemble.build(la, weights=wa)["consensus"]["labels"]
    other = ensemble.build(lb, weights=wb)["consensus"]["labels"]
    print(f"{run} L{layer}: ARI(A rebuilt, B rebuilt) = {ari(base, other):.3f}")
    for f in la:
        for what, (l2, w2) in {"labels+weight": ({**la, f: lb[f]}, {**wa, f: wb[f]}),
                               "weight only": (la, {**wa, f: wb[f]})}.items():
            c = ensemble.build(l2, weights=w2)["consensus"]["labels"]
            print(f"  swap {f:17s} {what:13s} ARI vs A {ari(base, c):.3f}  vs B {ari(other, c):.3f}  "
                  f"(weight {wa[f]:.3f} -> {wb[f]:.3f}, labels ARI {ari(la[f], lb[f]):.3f})")


PILOT = Path("/run/media/system/HDD_1TB/Mets_archive/2026-08-12_05-01-35")


def _k(lab: np.ndarray) -> int:
    return len(set(lab.tolist()) - {-1})


def matched(a: Path, b: Path, prompt: str) -> None:
    """Every record of PROMPT: fit the centroid and linkage families at the k
    of A's tuned HDBSCAN and of A's shipped HDBSCAN, on both sweeps, and report
    ARI(A, B) beside HDBSCAN's own. HDBSCAN is read from the stored labels
    (noise kept as a label, as §3 does); its ARI on the tokens both sweeps
    assign is printed too, to tell moved clusters from moved noise."""
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from p1d_cluster_ensemble import methods, p1d_io
    fits = {
        "kmeans": lambda d, k: KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(d.normed),
        "sph_kmeans": lambda d, k: methods.spherical_kmeans(d.normed, k, seed=0),
        "agglo_avg": lambda d, k: AgglomerativeClustering(
            n_clusters=k, linkage="average", metric="precomputed").fit_predict(d.cos_dist),
        "ward": lambda d, k: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(d.normed),
    }
    runs = sorted(p.name for p in a.iterdir() if p.name.endswith(prompt) and (p / "p1d_results.json").exists())
    print(f"{'step':>6s} {'L':>2s} {'maxdD':>7s} {'ref':8s} {'kA/kB':>7s} {'HDB':>6s} {'HDB asg':>7s} "
          + " ".join(f"{f:>10s}" for f in fits))
    for run in runs:
        ra, rb = _res(a, run), _res(b, run)
        la, lb = (p1d_io.load_run(Path(r["run_dir"])) for r in (ra, rb))
        for L in sorted(ra["per_layer"], key=int):
            da, db = (methods.LayerData.from_normed(p1d_io.layer_activations(x, int(L))) for x in (la, lb))
            dD = float(np.abs(da.cos_dist - db.cos_dist).max())
            refs = {"tuned": [np.asarray(r["per_layer"][L]["selection"]["hdbscan"]["selected_labels"])
                              for r in (ra, rb)],
                    "shipped": [np.asarray(x["shipped_hdbscan"][int(L)]) for x in (la, lb)]}
            for name, (ha, hb) in refs.items():
                if ha.ndim == 0 or hb.ndim == 0:  # tuned HDBSCAN abstained on a sweep
                    print(f"{run.split('step')[1].split('_')[0]:>6s} {L:>2s} {dD:7.1e} {name:8s} abstained")
                    continue
                both = (ha >= 0) & (hb >= 0)
                k = max(2, _k(ha))
                row = [ari(fits[f](da, k), fits[f](db, k)) for f in fits]
                print(f"{run.split('step')[1].split('_')[0]:>6s} {L:>2s} {dD:7.1e} {name:8s} "
                      f"{_k(ha):3d}/{_k(hb):<3d} {ari(ha, hb):6.3f} {ari(ha[both], hb[both]):7.3f} "
                      + " ".join(f"{v:10.3f}" for v in row), flush=True)


def ties(a: Path, b: Path, prompt: str, near: float = 1e-4, tie: float = 1e-6) -> None:
    """Per record of PROMPT, on sweep A: the share of token pairs closer than
    NEAR (cosine distance), and of tokens whose first and second nearest
    neighbours differ by less than TIE (the float noise's size). Then, for
    shipped and tuned HDBSCAN, the token pairs whose co-membership differs
    between A and B (noise is never a co-member), and how many of them are
    closer than NEAR. Descriptive: it locates the disagreement, it does not
    show that the ties cause it."""
    from p1d_cluster_ensemble import methods, p1d_io
    for run in sorted(p.name for p in a.iterdir() if p.name.endswith(prompt) and (p / "p1d_results.json").exists()):
        ra, rb = _res(a, run), _res(b, run)
        la, lb = (p1d_io.load_run(Path(r["run_dir"])) for r in (ra, rb))
        for L in sorted(ra["per_layer"], key=int):
            d = methods.LayerData.from_normed(p1d_io.layer_activations(la, int(L))).cos_dist
            iu = np.triu_indices_from(d, 1)
            dd = d[iu]
            nn = np.sort(d + 9 * np.eye(len(d)), axis=1)[:, :2]
            head = f"{run.split('step')[1].split('_')[0]:>6s} L{L:<2s}"
            print(f"{head} pairs<{near:g} {np.mean(dd < near):.2f}  NN ties<{tie:g} "
                  f"{np.mean(nn[:, 1] - nn[:, 0] < tie):.2f}  median d {np.median(dd):.1e}")
            refs = {"shipped": (la["shipped_hdbscan"][int(L)], lb["shipped_hdbscan"][int(L)]),
                    "tuned": tuple(np.asarray(r["per_layer"][L]["selection"]["hdbscan"]["selected_labels"])
                                   for r in (ra, rb))}
            for name, (ha, hb) in refs.items():
                if ha.ndim == 0 or hb.ndim == 0:
                    continue
                co = lambda h: (h[:, None] == h[None, :]) & (h[:, None] >= 0)
                dis = (co(ha) != co(hb))[iu]
                if dis.any():
                    print(f"{head}   {name:8s} disagreeing pairs {dis.sum():5d}, "
                          f"share <{near:g} {np.mean(dd[dis] < near):.2f}, median d {np.median(dd[dis]):.1e}")
                else:
                    print(f"{head}   {name:8s} no disagreeing pairs")


def baseline() -> None:
    import os
    from core.holdout import V1_PROMPT_KEYS
    keys = V1_PROMPT_KEYS - {"short_heterogeneous"}
    idx = json.loads((Path(os.environ["METS_DATA"]) / "phase12/stage0_logs/stage0_index.json").read_text())["runs"]
    per = {}
    for k, s0 in sorted(idx.items()):
        step, key = k.split("|")
        pd = PILOT / f"pythia-410m-step{step}_{key}" / "hdbscan_labels.json"
        if key not in keys or not pd.exists():
            continue
        a = json.loads((Path(s0) / "hdbscan_labels.json").read_text())
        b = json.loads(pd.read_text())
        for L in set(a) & set(b):
            x, y = np.asarray(a[L]), np.asarray(b[L])
            per.setdefault(key, []).append((int(L), bool((x != y).any()), float(ari(x, y))))
    print(f"{'prompt':16s} {'not identical / all layers':>26s} {'ARI<1 / layers>=1':>18s} {'min ARI':>8s} {'p5 ARI (L>=1)':>13s}")
    tot = np.zeros(4, int)
    for key, r in sorted(per.items()):
        ne = sum(d for _, d, _ in r); v = np.array([a for L, _, a in r if L >= 1])
        m = int((v < 1).sum()); tot += [ne, len(r), m, len(v)]
        print(f"{key:16s} {ne:12d} / {len(r):<11d} {m:8d} / {len(v):<7d} {v.min():8.3f} {np.percentile(v, 5):13.3f}")
    print(f"{'all 8':16s} {tot[0]:12d} / {tot[1]:<11d} {tot[2]:8d} / {tot[3]:<7d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--fine", action="store_true")
    p.add_argument("--swap", nargs=3, metavar=("STEP", "PROMPT", "LAYER"))
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--matched", metavar="PROMPT")
    p.add_argument("--ties", metavar="PROMPT")
    args = p.parse_args()
    if args.matched:
        matched(args.a, args.b, args.matched)
    if args.ties:
        ties(args.a, args.b, args.ties)
    if args.fine:
        fine(args.a)
    if args.swap:
        swap(args.a, args.b, *args.swap)
    if args.baseline:
        baseline()


if __name__ == "__main__":
    main()
