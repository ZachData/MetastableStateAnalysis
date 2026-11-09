"""
trackS_subspace/s4_hapax_fig.py — H's induction dates read off HAPAX's published figure (S4 U1,
D1 (a′): no induction score of ours runs on `vanilla_1b` / `hapax_1b`).

The figure is 2511.05743v3's `prefix_induction_comparison_1b-triple_mean_visible.png` (the
prefix-matching score across training, Vanilla / Hapax / Thresholded-Hapax and their means). Its
fourth panel, "Mean", is digitised by colour: the axes frame (x 1138–1476 px, y 27–238 px for
1.0–0.0) and the x limits from matplotlib's 5 % margin about the data's 100–19,900 (ticks at 0
and 20,000 land on 1152 and 1462 px, checked). About 64 steps a pixel. Rows above 95 px are the
legend and are skipped (both means stay below 0.68 there). Hapax's curve is hidden under the
Thresholded-Hapax line before about step 980, so its early part is not read. Then D4 (a) on H's
window (1000, 2000, 4000): the first digitised point at or above half the window's median, rounded
up to H's 100-step grid. The mean is over the panel's heads (the paper's), not the max over heads
the toys use.

    python -I trackS_subspace/s4_hapax_fig.py <figure.png>
"""
import hashlib
import json
import sys

import numpy as np

X0, X1, Y0, Y1 = 1138, 1476, 27, 238
XLIM = (100 - 0.05 * 19800, 19900 + 0.05 * 19800)
COLOURS = {"vanilla": (0, 114, 178), "hapax": (230, 159, 0)}
WINDOW = (1000, 2000, 4000)


def step(x: float) -> float:
    return XLIM[0] + (x - X0) / (X1 - X0) * (XLIM[1] - XLIM[0])


def value(y: float) -> float:
    return (Y1 - y) / (Y1 - Y0)


def digitise(a: np.ndarray) -> dict:
    out = {}
    for k, c in COLOURS.items():
        pts = []
        for x in range(X0 + 2, X1 - 1):
            ys = [y for y in range(95, Y1) if np.abs(a[y, x] - c).sum() < 40]
            if ys:
                pts.append((round(step(x)), round(float(value(np.mean(ys))), 3)))
        if len(pts) < 200:              # the panel has 338 columns; a colour seen in few is not the curve
            raise SystemExit(f"REFUSE: {k}'s colour found in {len(pts)} columns only")
        out[k] = pts
    return out


def date(pts) -> dict:
    def at(s):
        return min(pts, key=lambda q: abs(q[0] - s))[1]
    w = [at(s) for s in WINDOW]
    m = float(np.median(w))
    first = next(q for q in pts if q[1] >= m / 2)
    return {"window_values": w, "median": m, "crossing": first, "grid_date": int(-(-first[0] // 100) * 100)}


def main(path: str) -> int:
    from PIL import Image
    raw = open(path, "rb").read()
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    pts = digitise(a)
    res = {"figure_sha256": hashlib.sha256(raw).hexdigest(), "points": pts,
           "dates": {k: date(p) for k, p in pts.items()}}
    print(json.dumps(res["dates"], indent=1))
    with open(path + ".digitised.json", "w") as f:
        json.dump(res, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
