"""The structural baselines a causally-masked attention matrix has BEFORE any
content enters, and what they imply for three quantities this project reports.

WHY THIS EXISTS
---------------
`p10_cluster_function/attention-10.md` audits the attention flip -- trained
models routing ~1.6x layer-average attention to unclustered tokens and ~0.5x to
clustered ones. The statistic is

    received[key] = sum over heads, sum over queries of attn[h, query, key]
    reported as   received[population].mean() / received.mean()

with the diagonal zeroed and NOTHING ELSE divided out. Under a causal mask a
token at position j is visible to only n - j queries, so `received` has a
built-in tilt toward early positions before content enters at all. This derives
that tilt in closed form so the audit compares against a number rather than an
intuition.

It also derives the row-side baselines for attention entropy and for the
partition function Z_beta,i, because the same mask acts on them in the OPPOSITE
direction -- which bears on an identification `math-1.md` Sec 1A.6 asserts and
nobody has measured.

WHAT IT DOES NOT PROVE
----------------------
Everything here is the CONTENT-FREE baseline: attention uniform within the
causal triangle, and (for Z) all pairwise inner products equal. That is the
null, not the model. It says what the statistic reports when the network does
nothing, which is exactly what an enrichment ratio needs and exactly what the
flip's measurement lacks. It says nothing about whether the observed deviation
from the baseline is large, and nothing about any particular checkpoint.

The Z results additionally assume the concentration regime (Theorem 6.9's
common inner product). Outside it they are a leading-order reading, not an
identity.

Run: python3 tools/math_checks/causal_mask_attention_baseline.py
"""
import sympy as sp

results = []


def record(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"      {detail}")
    results.append(ok)


i, j, k, n = sp.symbols("i j k n", integer=True, positive=True)

# ---------------------------------------------------------------------------
# 1. Received attention under a content-free causal mask
# ---------------------------------------------------------------------------
# Positions 0..n-1. Query i attends over keys 0..i inclusive (causal, with
# self-attention present in the model). Content-free means uniform within the
# row: a[i, j] = 1/(i+1) for j <= i, else 0.
#
#   received(j) = sum_{i=j}^{n-1} 1/(i+1) = sum_{m=j+1}^{n} 1/m = H_n - H_j
#
# H_0 = 0, so received(0) = H_n.
received = sp.harmonic(n) - sp.harmonic(j)

# Verify against the explicit sum at several n, for every j.
ok = True
for nn in (5, 12, 40):
    for jj in range(nn):
        explicit = sum(sp.Rational(1, ii + 1) for ii in range(jj, nn))
        closed = received.subs({n: nn, j: jj})
        if sp.simplify(explicit - closed) != 0:
            ok = False
record("received(j) = H_n - H_j   [closed form vs explicit sum, n in {5,12,40}, all j]", ok)

# Total mass is n (n rows, each summing to 1), so the LAYER MEAN IS EXACTLY 1
# and `received(j)` already IS the "x layer average" quantity the flip reports.
ok = True
for nn in (5, 12, 40, 264):
    total = sum(received.subs({n: nn, j: jj}) for jj in range(nn))
    if sp.simplify(total - nn) != 0:
        ok = False
record("sum_j received(j) = n, so the layer mean is exactly 1 and received(j) IS the ratio",
       ok, "the flip's 'x layer average' and this baseline are directly comparable")

# ---------------------------------------------------------------------------
# 2. The size of the structural tilt, at the project's battery length
# ---------------------------------------------------------------------------
N_BATTERY = 264  # the ~264-token battery core/sink_audit.py sizes its baseline on
H = lambda m: float(sp.harmonic(m))
first = H(N_BATTERY)                      # position 0
last = H(N_BATTERY) - H(N_BATTERY - 1)    # position n-1
median = H(N_BATTERY) - H(N_BATTERY // 2)
record("content-free tilt at n=264 spans three orders of magnitude",
       first > 6.0 and last < 0.005,
       f"pos 0 = {first:.3f}x, median pos = {median:.3f}x, last pos = {last:.5f}x layer average")

# The observed contrast (1.6x vs 0.5x) is far inside that span. Which positions
# does the CONTENT-FREE baseline already assign those values to?
def position_with_ratio(target, nn=N_BATTERY):
    """Smallest j whose baseline received(j) is <= target."""
    Hn = H(nn)
    for jj in range(nn):
        if Hn - H(jj) <= target:
            return jj
    return nn - 1

p16 = position_with_ratio(1.6)
p05 = position_with_ratio(0.5)
record("the observed 1.6x / 0.5x are both attainable with zero content",
       0 < p16 < p05 < N_BATTERY,
       f"baseline hits 1.6x at position ~{p16} and 0.5x at position ~{p05} (n=264): "
       "a population split at those mean positions reproduces the flip with no learned behaviour")

# Consistency: a two-population split of the SAME tokens must average to 1.
# f * 1.6 + (1 - f) * 0.5 = 1  =>  f = 0.5/1.1 ~= 0.4545, i.e. ~45% unclustered.
f = sp.Rational(5, 11)
record("the reported 1.6x / 0.5x pin the unclustered fraction to ~45%",
       sp.simplify(f * sp.Rational(8, 5) + (1 - f) * sp.Rational(1, 2) - 1) == 0,
       f"f = {float(f):.4f}; status-5c reports ~40-50% unclustered -- the numbers are "
       "internally consistent, which is a consistency check, not evidence of content")

# ---------------------------------------------------------------------------
# 3. Row-side baseline: attention entropy
# ---------------------------------------------------------------------------
# core/metrics.py::attention_entropy averages row entropies. Content-free,
# row i is uniform over i+1 keys, so its entropy is log(i+1) exactly.
ok = True
for nn in (5, 12, 40):
    per_row = [sp.log(ii + 1) for ii in range(nn)]
    mean_entropy = sum(per_row) / nn
    # sum_{i=0}^{n-1} log(i+1) = log(n!)
    if sp.simplify(mean_entropy - sp.log(sp.factorial(nn)) / nn) != 0:
        ok = False
record("content-free attention entropy: row i has entropy log(i+1), layer mean log(n!)/n",
       ok, f"at n=264 the mean is {float(sp.log(sp.factorial(264))/264):.3f} nats; "
           "a per-row deviation from log(i+1) is the content, the raw value is mostly position")

# ---------------------------------------------------------------------------
# 4. Z_beta,i, and the direction the mask pushes it
# ---------------------------------------------------------------------------
# Z_beta,i = sum_j exp(beta <x_i, x_j>) is particle i's softmax denominator --
# a ROW quantity, and (math-1.md Sec 1A.6) the per-token weight in the metric
# under which (SA) is a gradient flow: a high-Z token is expensive to move.
#
# In the concentration regime every inner product is a common gamma, so:
beta, gamma = sp.symbols("beta gamma", real=True)
Z_unmasked = n * sp.exp(beta * gamma)          # row i sums over all n
Z_masked = (i + 1) * sp.exp(beta * gamma)      # row i sums over j <= i only

record("unmasked: Z_beta,i is position-independent under concentration",
       sp.simplify(sp.diff(Z_unmasked, i)) == 0,
       "so any spread in Z is content -- e.g. a high-norm sink, whose inner products "
       "with everything are large. This is the regime Sec 1A.6's 'high-Z = sink' reading assumes")

record("masked: Z_beta,i grows linearly in position, and position 0 is the MINIMUM",
       sp.simplify(sp.diff(Z_masked, i) - sp.exp(beta * gamma)) == 0,
       "Z_masked(0) = exp(beta*gamma) is the smallest value any row can take")

# The consequence worth carrying: under the mask the two structural baselines
# point in OPPOSITE directions in position.
# received(j) - received(j+1) = H_{j+1} - H_j = 1/(j+1) > 0, so received
# DECREASES with position. sympy leaves harmonic() unevaluated under simplify,
# so expand_func is what turns the difference into a rational function.
received_step = sp.expand_func(
    (sp.harmonic(n) - sp.harmonic(j)) - (sp.harmonic(n) - sp.harmonic(j + 1))
)
record("the two mask baselines are anti-aligned in position",
       sp.simplify(received_step - 1 / (j + 1)) == 0,
       "received(j) falls as 1/(j+1) per step while Z_masked(i) rises linearly. "
       "So the sink (position 0) is simultaneously the LARGEST received-attention and the "
       "SMALLEST Z -- i.e. under a causal mask the 'high-Z = sink' identification REVERSES. "
       "It is an unmasked-model statement, and Pythia is masked")

print()
print(f"{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
