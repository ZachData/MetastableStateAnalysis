"""What a size-profile-preserving null does and does not buy for the ARI --
because two Phase 9/10 documents asked for one to remove a bias the statistic
already removes.

WHY THIS EXISTS
---------------
`plan-9.md` Sec 2.3 and `p10_cluster_function/notes-10.md` Sec 4.4 both say that
before any ARI between two cluster labellings is quoted, it must be run against
"label permutations that preserve the observed cluster-size profile", on the
grounds that two labellings dominated by one giant cluster "agree at high ARI for
reasons that have nothing to do with content".

The premise is right about the RAND index and wrong about the ADJUSTED one. The
adjustment in ARI is exactly the expectation under the permutation model with
both size profiles held fixed (the generalized hypergeometric):

    ARI = ( sum_ij C(n_ij,2) - E ) / ( (1/2)[sum_i C(a_i,2) + sum_j C(b_j,2)] - E )
    E   = [ sum_i C(a_i,2) * sum_j C(b_j,2) ] / C(N,2)

so E[ARI] = 0 under that null BY CONSTRUCTION, whatever the size profiles are.

WHAT THIS CHANGES
-----------------
The null is still needed -- for the VARIANCE, not the centering. "ARI = 0.18" is
already bias-corrected; whether 0.18 is far from zero depends on N and on the
size profiles, and only a null says so. So the instruction stands but its reason
changes: build the permutation null to get a p-value, not to de-bias.

WHAT IT DOES NOT PROVE
----------------------
- Mean-zero is checked by Monte Carlo at particular (N, profile) settings, not
  proved here; the closed form above is the proof and this is its test.
- It says nothing about the OTHER concern in those sections, which is real: a
  partition that is mostly noise labels (-1) is not a partition, and
  `adjusted_rand_index(ignore_noise=...)` changes what N even means. That is a
  definition choice and no adjustment fixes it.
- HDBSCAN noise is not a cluster. Treating -1 as one cluster versus dropping
  those points gives different ARIs on the same data, and the design must fix
  which before looking.

Run: python3 tools/math_checks/ari_size_profile_null.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                    # pragma: no cover - entrypoint
    sys.path.insert(0, str(ROOT))

from core.functional_distance import adjusted_rand_index  # noqa: E402

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"      {detail}")
    results.append(ok)


rng = np.random.default_rng(20260920)


def labels_from_sizes(sizes):
    return np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])


# Three regimes, including the one the documents worried about.
REGIMES = {
    "balanced (10 clusters of 26)": ([26] * 10, [26] * 10),
    "one giant cluster (200 + 8x8)": ([200] + [8] * 8, [200] + [8] * 8),
    "very unequal profiles (giant vs balanced)": ([230] + [17] * 2, [33] * 8),
}

DRAWS = 4000
for name, (sizes_a, sizes_b) in REGIMES.items():
    a = labels_from_sizes(sizes_a)
    b = labels_from_sizes(sizes_b)
    assert len(a) == len(b), name
    vals = np.empty(DRAWS)
    for t in range(DRAWS):
        vals[t] = adjusted_rand_index(rng.permutation(a), b)
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    se = sd / np.sqrt(DRAWS)
    record(f"E[ARI] = 0 under the size-profile-preserving null -- {name}",
           abs(mean) < 4 * se,
           f"mean {mean:+.5f} (SE {se:.5f}), sd {sd:.5f}, "
           f"95th pct {np.percentile(vals, 95):+.4f}")

# The point the documents were reaching for, restated correctly: the SPREAD
# under the null is what differs between regimes, so the same ARI means
# different things.
spreads = {}
for name, (sizes_a, sizes_b) in REGIMES.items():
    a, b = labels_from_sizes(sizes_a), labels_from_sizes(sizes_b)
    vals = np.array([adjusted_rand_index(rng.permutation(a), b) for _ in range(DRAWS)])
    spreads[name] = float(np.percentile(vals, 95))
lo, hi = min(spreads.values()), max(spreads.values())
record("the null's 95th percentile varies several-fold across size profiles",
       hi > 2 * lo,
       "; ".join(f"{k}: {v:+.4f}" for k, v in spreads.items())
       + "  -- so a fixed ARI threshold is not comparable across layers or models, "
         "which is the real content of the warning")

# Sanity: identical labellings give exactly 1, and the statistic is symmetric.
a = labels_from_sizes([200] + [8] * 8)
record("ARI(a, a) = 1", abs(adjusted_rand_index(a, a) - 1.0) < 1e-12)
b = labels_from_sizes([33] * 8)
record("ARI is symmetric",
       abs(adjusted_rand_index(a, b) - adjusted_rand_index(b, a)) < 1e-12)

print()
print(f"{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
