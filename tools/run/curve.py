"""The P-I1 formation curve over the registered 19-step grid.

Answers HANDOFF.md §6's first question: with the five log-spaced fills inside
(1000, 54000) on disk, does the per-head change centroid still take ONE value?

PATHS ARE DERIVED. The first version opened with
`sys.path.insert(0, "/mnt/mets")` and read `/mnt/vm_storage/...`; both mounts
were gone by 2026-09-03, and the tree then moved under the repo. METS_REPO and
METS_DATA override.

THE PER-PROMPT WORKAROUND IS GONE. It called `find_relays(t.filter(prompt_key=p))`
because `find_relays` used to join rows across prompts (HANDOFF §3.1). Commit
f7e95bc confined the join to its context, so the whole-table call is now the
same answer for a fraction of the time.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
# One root. The generated tree moved under the repo on 2026-09-03; the old
# METS_VOL named a VM's scratch volume, which is the class of path that broke
# this script and sweep.sh when the mounts changed.
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

import numpy as np

from core.changepoint_colocation import (
    ColocationRefused,
    REGISTERED_P_I1_SWEEP,
    change_profile,
    interval_midpoints,
)
from core.interactions import InteractionTable
from p7_motifs.motif_alphabet import find_relays

# The registry owns the grid and the owner; restating either here is how a
# second copy drifts from the one the gate reads.
STEPS = list(REGISTERED_P_I1_SWEEP)
from p7_motifs.formation_gate import P_I1_RELAY_OWNER
OWNERS = ("tag_writer", "matcher", "both")

out = {"steps": STEPS, "relay_owner_registered": P_I1_RELAY_OWNER, "per_step": {}}
series = {o: {} for o in OWNERS}          # owner -> (layer,head) -> [v per step]

for i, s in enumerate(STEPS):
    path = DATA / "phase7" / f"step{s}" / "interaction_table.npz"
    t = InteractionTable.load(path)
    # ONE find_relays pass, not four. The previous version called
    # find_relays(t.filter(prompt_key=p)) for each of 8 prompts and then
    # per_head_relay_strength(t, o) for each of 3 owners -- and that helper
    # calls find_relays(t) itself. Every number below is a projection of the
    # same relay list: RelayInstance carries prompt_key, (layer_1, head_1) and
    # (layer_2, head_2), which is the prompt, the tag writer and the matcher.
    # The 8 filters also each allocated a copy of a slice of a 5.5 GB table.
    relays = find_relays(t)
    rec = {"n_edges": len(t), "per_prompt": {}}
    for p in sorted(set(np.unique(t.columns["prompt_key"]).tolist())):
        rec["per_prompt"][p] = 0        # a prompt with no relays must read 0,
    for r in relays:                    # not be missing from the record
        rec["per_prompt"][r.prompt_key] = rec["per_prompt"].get(r.prompt_key, 0) + 1
    rec["total"] = sum(rec["per_prompt"].values())
    rec["total_no_repeated"] = sum(
        v for k, v in rec["per_prompt"].items() if k != "repeated_tokens")

    strengths = {o: {} for o in OWNERS}
    for r in relays:
        w, m = (r.layer_1, r.head_1), (r.layer_2, r.head_2)
        strengths["tag_writer"][w] = strengths["tag_writer"].get(w, 0.0) + 1.0
        strengths["matcher"][m] = strengths["matcher"].get(m, 0.0) + 1.0
        strengths["both"][w] = strengths["both"].get(w, 0.0) + 1.0
        strengths["both"][m] = strengths["both"].get(m, 0.0) + 1.0

    rec["owner"] = {}
    for o in OWNERS:
        strength = strengths[o]
        rec["owner"][o] = {"n_heads_with_relays": len(strength),
                           "total": float(sum(strength.values()))}
        for k, v in strength.items():
            series[o].setdefault(k, [0.0] * len(STEPS))[i] = float(v)
    out["per_step"][str(s)] = rec
    print(f"step{s:<7d} edges={rec['n_edges']:>10,d} relays={rec['total']:>10,d}  "
          f"ex-repeated={rec['total_no_repeated']:>9,d}  "
          f"heads({P_I1_RELAY_OWNER})={rec['owner'][P_I1_RELAY_OWNER]['n_heads_with_relays']:>4d}",
          flush=True)
    del t

# ---- the degeneracy question -------------------------------------------
# A head whose series never rises has no location to measure; change_profile
# refuses it rather than reporting a uniform profile, so those are counted
# separately and NOT folded in as a centroid of zero.
out["centroids"] = {}
for o in OWNERS:
    cents, refused = {}, 0
    for k, v in series[o].items():
        try:
            prof = change_profile(STEPS, v, "rise")
        except ColocationRefused:
            refused += 1
            continue
        cents[f"{k[0]},{k[1]}"] = {
            "centroid_log_step": prof["centroid_log_step"],
            "centroid_step": prof["centroid_step"],
            "dispersion_log_step": prof["dispersion_log_step"],
            "concentration": prof["concentration"],
            "noise_mass_share_estimate": prof["noise_mass_share_estimate"],
        }
    vals = np.array([c["centroid_log_step"] for c in cents.values()])
    # Distinct at the scale a float64 mean over n_intervals terms can resolve,
    # which is what HANDOFF §3.2 found the equality test could not do.
    distinct = len(np.unique(np.round(vals, 9))) if vals.size else 0
    out["centroids"][o] = {
        "n_heads_scored": len(cents), "n_heads_refused_no_rise": refused,
        "n_distinct_centroids": int(distinct),
        "centroid_min": float(vals.min()) if vals.size else None,
        "centroid_max": float(vals.max()) if vals.size else None,
        "centroid_sd": float(vals.std()) if vals.size else None,
        "per_head": cents,
    }
    print(f"\n{o}: {len(cents)} heads scored, {refused} refused (no rise), "
          f"{distinct} DISTINCT centroids"
          + (f", span {vals.min():.4f}..{vals.max():.4f} log-step, sd {vals.std():.4g}"
             if vals.size else ""))

out["interval_midpoints"] = interval_midpoints(STEPS).tolist()
outdir = Path(os.environ.get("METS_SCRATCH", str(DATA / "analysis")))
dest = outdir / "curve.json"
json.dump(out, open(dest, "w"), indent=1)
print(f"\nWROTE {dest}")

# The per-head SERIES, written beside the centroids and not into curve.json.
# curve.json is the artifact HANDOFF §4.6 diffs after every change to the
# storage layer, and a file whose content is diffed is not the place to add a
# key. The centroids alone determine `paired_colocation_arm`'s statistic, but
# they cannot be handed to it: the arm takes series and computes the profiles
# itself, so anything that runs the REAL arm rather than a re-implementation of
# it needs these.
sdest = outdir / "formation_series.json"
json.dump({"steps": STEPS,
           "relay_owner_registered": P_I1_RELAY_OWNER,
           "series": {o: {f"{k[0]},{k[1]}": v for k, v in series[o].items()}
                      for o in OWNERS}},
          open(sdest, "w"), indent=1)
print(f"WROTE {sdest}")
