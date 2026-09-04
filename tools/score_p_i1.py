"""tools/score_p_i1.py — P-I1's p-value, for the first time with a real null.

Everything before this file supplied one piece: `tools/run/curve.py` the raw
relay series, `tools/run/behavioural.py` the real behavioural series,
`tools/run/relay_null.py` the relay-count null (PROJECT.md §3.4) on the
pre-filtered forming axis (§3.1), and `p7_motifs/formation_gate.py`'s
`skip_no_rise` the arm-side half of §3.1's fix. This is the first script that
calls `p_value_p_i1` with all four in place.

It does NOT adjudicate. `adjudicate_p_i1(..., adjudicate=True)` writes an
entry to `claims/adjudications/`, which stays empty on purpose until the
author decides this result should be registered there -- printing a number
is not that decision, and this script does not make it for them.

Usage
-----
    python3 -m tools.run.relay_null      # writes relay_null_series.json first
    python3 -m tools.score_p_i1
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("METS_REPO", "/run/media/system/WDS_500/Mets"))
DATA = Path(os.environ.get("METS_DATA", str(REPO / "data")))
sys.path.insert(0, str(REPO))

from core.changepoint_colocation import REGISTERED_P_I1_SWEEP
from p7_motifs.formation_gate import P_I1_RELAY_OWNER, p_value_p_i1

STEPS = list(REGISTERED_P_I1_SWEEP)


def main() -> int:
    null_path = DATA / "analysis" / "relay_null_series.json"
    behav_path = DATA / "analysis" / "behavioural_series.json"
    if not null_path.exists():
        print(f"{null_path} is missing; run `python3 -m tools.run.relay_null` "
              f"first (~40 min over the real sweep).")
        return 1
    if not behav_path.exists():
        print(f"{behav_path} is missing; run "
              f"`python3 -m tools.run.behavioural --write` first.")
        return 1

    null = json.loads(null_path.read_text())
    behav = json.loads(behav_path.read_text())

    if [int(s) for s in null["steps"]] != STEPS:
        print(f"{null_path} is not on the registered sweep")
        return 1
    if null["relay_owner"] != P_I1_RELAY_OWNER:
        print(f"{null_path} was scored under relay_owner="
              f"{null['relay_owner']!r}, not the registered "
              f"{P_I1_RELAY_OWNER!r}")
        return 1

    forming_heads = null["forming_heads"]          # ["l,h", ...]
    excess = null["above_null_excess"]
    behav_series = behav["series_excl_repeated"]
    missing = [h for h in forming_heads if h not in behav_series]
    if missing:
        print(f"{len(missing)} forming heads have no behavioural series "
              f"(e.g. {missing[:3]}); the two artifacts do not agree on the "
              f"head axis")
        return 1

    relay_strength = [excess[h] for h in forming_heads]
    induction_score = [behav_series[h] for h in forming_heads]

    res = p_value_p_i1(STEPS, relay_strength, induction_score,
                       skip_no_rise=True)

    print(f"P-I1, relay_owner={P_I1_RELAY_OWNER!r}, "
          f"{null['n_replicates']} null replicates, "
          f"{len(forming_heads)} heads on the forming axis\n")
    print(f"p_value       = {res['p_value']}")
    print(f"p_reciprocal  = {res['p_reciprocal']}")
    print(f"verdict       = {res['verdict']}")
    print(f"reason        = {res['reason']}")
    if res.get("arms"):
        arm = res["arms"][0]
        print(f"n_units       = {arm['n_units']}  "
              f"(n_skipped_no_rise = {arm.get('n_skipped_no_rise')})")
        print(f"attainable_floor = {arm['attainable_floor']}")
        print(f"mean_distance_log_step = {arm['mean_distance_log_step']}")
    if res.get("endpoint_flags"):
        print(f"\nendpoint_flags (reported, no p-value): "
              f"{res['endpoint_flags']}")
    print(f"\nNOT adjudicated: claims/adjudications/ is untouched by this "
          f"script. adjudicate_p_i1(..., adjudicate=True) is the author's "
          f"call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
