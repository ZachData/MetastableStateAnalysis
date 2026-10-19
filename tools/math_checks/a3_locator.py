"""A3's half-rise locator on the 18-step grid: its null floor and its power (design-A3.md §4).

Synthetic only (numpy, seeded). Noise is iid N(0, sigma) at every step, step 0 included, so a
Delta carries step 0's noise as well. For the null, it finds the floor c (in sigma) at which a
flat series reads "no change" in >= 95 % of draws. For each grid gap up to 4000, it finds the
smallest amplitude (in sigma) of a logistic step in log2(step) centred mid-gap, of width W
(logistic scale, in octaves), that the rule places at the step just past the centre in >= 90 %.

Not proved here: that real rows have iid noise (prompts, heads and layers are not independent
draws across steps, and weights-only rows have no sampling noise at all), or that a logistic
is the shape a real row moves in.
"""
import numpy as np

STEPS = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000]
WIN = [s for s in STEPS if s <= 4000]          # step 0 and 12 after it
X = np.log2(np.maximum(WIN, 1)); X[0] = -1.0   # step 0 one octave below step 1, as the panel draws it
N = 20000
W = 0.25                                       # placed: logistic scale, octaves
rng = np.random.default_rng(0)


def locate(x, c):
    """x: (draws, len(WIN)), noise unit sigma. Returns index of change step, or -1 for no change."""
    d = x[:, 1:] - x[:, :1]
    ref = np.abs(d).max(axis=1)
    hit = np.abs(d) >= ref[:, None] / 2
    idx = hit.argmax(axis=1) + 1
    idx[ref < c] = -1
    return idx


null = rng.standard_normal((N, len(WIN)))
ref_null = np.abs(null[:, 1:] - null[:, :1]).max(axis=1)
c95 = float(np.quantile(ref_null, 0.95))
print(f"null: P(no change) at 3 sigma = {np.mean(ref_null < 3):.3f}; floor for 95 % = {c95:.2f} sigma")

print(f"planted logistic, width {W} oct, floor {c95:.2f}: smallest amplitude (sigma) placed right in >= 90 %")
amps = np.arange(3.0, 30.01, 0.5)
for i in range(1, len(WIN) - 1):
    centre = (X[i] + X[i + 1]) / 2
    shape = 1 / (1 + np.exp(-(X - centre) / W))
    shape = shape - shape[0]
    need = None
    for a in amps:
        x = rng.standard_normal((4000, len(WIN))) + a * shape
        if np.mean(locate(x, c95) == i + 1) >= 0.90:
            need = a
            break
    print(f"  gap {WIN[i]:>5}-{WIN[i + 1]:<5}: {need if need is not None else '> 30'}")
