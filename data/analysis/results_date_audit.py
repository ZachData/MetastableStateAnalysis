#!/usr/bin/env python3
"""Audit the provenance of every on-disk result under data/phase12 and data/phase7:
which git_sha and timestamp produced it, grouped, so stale (pre-refactor) runs
are visible."""
import json, os, glob, subprocess, collections

REPO = "/run/media/system/WDS_500/Mets"
os.chdir(REPO)

def sha_date(sha):
    try:
        out = subprocess.run(["git", "show", "-s", "--format=%ci|%s", sha],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            d, s = out.stdout.strip().split("|", 1)
            return d[:19], s
    except Exception:
        pass
    return "NOT-IN-REPO", "?"

# ---------------- phase12 ----------------
rows = []
for m in glob.glob("data/phase12/**/manifest.json", recursive=True):
    try:
        d = json.load(open(m))
    except Exception:
        continue
    rows.append(dict(
        path=m.replace("data/phase12/", ""),
        step=d.get("checkpoint_step"),
        phase=d.get("phase", "?"),
        sha=(d.get("git_sha") or "?")[:12],
        ts=(d.get("timestamp") or "?")[:19],
        sublayer=d.get("sublayer_semantics"),
        parallel=d.get("parallel_residual"),
        wdtype=(d.get("config") or {}).get("weight_dtype"),
    ))

print(f"=== data/phase12 : {len(rows)} run manifests ===\n")

by_sha = collections.defaultdict(list)
for r in rows:
    by_sha[r["sha"]].append(r)

for sha, rs in sorted(by_sha.items(), key=lambda kv: min(x["ts"] for x in kv[1])):
    gdate, gsubj = sha_date(sha)
    phases = collections.Counter(x["phase"] for x in rs)
    steps = sorted({x["step"] for x in rs}, key=lambda v: (v is None, v))
    tsrange = (min(x["ts"] for x in rs), max(x["ts"] for x in rs))
    print(f"git_sha {sha}  commit_date {gdate}   ({gsubj[:60]})")
    print(f"   runs written {tsrange[0]} .. {tsrange[1]}")
    print(f"   {dict(phases)}  |  {len(rs)} runs  |  steps {steps}")
    print(f"   sublayer_semantics={sorted({str(x['sublayer']) for x in rs})}  "
          f"parallel_residual={sorted({str(x['parallel']) for x in rs})}  "
          f"weight_dtype={sorted({str(x['wdtype']) for x in rs})}")
    print()

# ---------------- phase7 ----------------
print("\n=== data/phase7 : interaction tables ===\n")
for man in sorted(glob.glob("data/phase7/**/*.json", recursive=True)) + \
          sorted(glob.glob("data/phase7/**/manifest*", recursive=True)):
    print("  ", man)
for npz in sorted(glob.glob("data/phase7/step*/interaction_table.npz")):
    st = os.stat(npz)
    import datetime
    print(f"  {npz}   size={st.st_size/1e6:.1f}MB  mtime={datetime.datetime.fromtimestamp(st.st_mtime).isoformat()[:19]}")

# ---------------- analysis json provenance ----------------
print("\n=== data/analysis/*.json : self-reported provenance ===\n")
for j in sorted(glob.glob("data/analysis/*.json")):
    try:
        d = json.load(open(j))
    except Exception:
        continue
    keys = {k: d[k] for k in ("generated_utc", "generated", "git_sha", "n_replicates",
                              "_what_this_is") if k in d}
    st = os.stat(j)
    import datetime
    mt = datetime.datetime.fromtimestamp(st.st_mtime).isoformat()[:19]
    print(f"  {os.path.basename(j):<34} mtime={mt}  {list(keys.items())[:3]}")
