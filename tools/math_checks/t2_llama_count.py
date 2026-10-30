"""Checks the closed-form claims in trackT_timaeus/design-T2.md "Blocked 120 (b)" (a centred count
fit to Llama on Dodecahedron, proposed 2026-10-30).

h_i (i = 1 … n) the raw hidden states of one prompt at one state, h̄ their mean over tokens, u a
unit vector. e1c normalises each row, then centres over tokens and re-normalises (C2's centred
count); e1r centres the raw rows over tokens, then normalises (C2's "raw-centred", beside in 3a–3c).

L1  Plant A (every row moved by the same vector c) cancels exactly from the raw-centred rows:
    (h_i + c) − mean(h + c) = h_i − h̄, so plant A's leak through e1r is 0 whatever the cloud
    (sympy, n = 3, d = 2, every entry a symbol).
L2  Plant A_s (every row moved by a ‖h_i‖ u, i.e. a common direction on the sphere) leaves, after
    raw centring, (h_i − h̄) + a (‖h_i‖ − mean ‖h‖) u: its leak is set by the spread of token
    norms and vanishes when every norm is equal (sympy, norms as free positive symbols).
L3  For n unit rows with mean off-diagonal cosine ρ, the mean row ‖x̄‖² = (1 + (n − 1) ρ) / n and
    the token-centred rows' mean squared norm is 1 − ‖x̄‖² = (n − 1)(1 − ρ) / n: at ρ = 0.9 e1c
    re-normalises a residual of squared norm about 0.1, at 410m's 0.17 about 0.83 (sympy identity;
    the values printed for n = 512).
L4  At equal token norms (h_i = r x_i, ‖x_i‖ = 1) the raw-centred rows are r (x_i − x̄), a positive
    multiple of e1c's centred rows, so e1r = e1c exactly: e1r re-normalises the same residual L3
    sizes, and differs from e1c only through the spread of token norms (sympy; added after
    `/challenge-pr` on #287, finding 1, which found D5's first reason false).

NOT proved: that a small residual is why unit 1's centred arms leaked plant A (the leak is
measured, `status-T.md` "T2 unit 1"; L3 says only how much of each row centring leaves); that
e1r passes A_s, B or the re-init bar on any model (L2 says A_s can leak through e1r only through
the norm spread, which on Dodecahedron at 9536 is position 0 at 13.9–23.8× the median); that
e1r on Dodecahedron is the event e1r or e1c dates on Pythia.
"""

import sympy as sp


def l1() -> None:
    n, d = 3, 2
    h = sp.Matrix(n, d, lambda i, j: sp.Symbol(f"h{i}{j}"))
    c = sp.Matrix(1, d, lambda i, j: sp.Symbol(f"c{j}"))
    hp = h + sp.ones(n, 1) * c

    def centre(m):
        return m - sp.ones(n, 1) * (sp.ones(1, n) * m) / n

    assert sp.simplify(centre(hp) - centre(h)) == sp.zeros(n, d)
    print("L1 ok: plant A cancels exactly from the raw-centred rows")


def l2() -> None:
    n, d = 3, 2
    h = sp.Matrix(n, d, lambda i, j: sp.Symbol(f"h{i}{j}"))
    r = [sp.Symbol(f"r{i}", positive=True) for i in range(n)]       # ‖h_i‖, free
    a = sp.Symbol("a")
    u = sp.Matrix(1, d, lambda i, j: sp.Symbol(f"u{j}"))
    hp = h + sp.Matrix(n, 1, r) * a * u

    def centre(m):
        return m - sp.ones(n, 1) * (sp.ones(1, n) * m) / n

    rbar = sum(r) / n
    want = centre(h) + sp.Matrix(n, 1, [ri - rbar for ri in r]) * a * u
    assert sp.simplify(centre(hp) - want) == sp.zeros(n, d)
    eq = {ri: r[0] for ri in r[1:]}
    assert sp.simplify((centre(hp) - centre(h)).subs(eq)) == sp.zeros(n, d)
    print("L2 ok: A_s leaves a (‖h_i‖ − mean‖h‖) u after raw centring; 0 at equal norms")


def l3() -> None:
    n, rho = sp.symbols("n rho", positive=True)
    # Σ_ij ⟨x_i, x_j⟩ = n + n (n − 1) ρ for unit rows with mean off-diagonal cosine ρ
    s2 = n + n * (n - 1) * rho
    xbar2 = s2 / n**2
    assert sp.simplify(xbar2 - (1 + (n - 1) * rho) / n) == 0
    # mean_i ‖x_i − x̄‖² = 1 − ‖x̄‖² for unit rows
    resid = 1 - xbar2
    assert sp.simplify(resid - (n - 1) * (1 - rho) / n) == 0
    for r in (0.9, 0.91, 0.84, 0.47, 0.17):
        print(f"L3   n 512, ρ {r}: centred rows' mean squared norm {float(resid.subs({n: 512, rho: r})):.4f}")
    print("L3 ok: ‖x̄‖² = (1 + (n − 1) ρ) / n; centred residual (n − 1)(1 − ρ) / n")


def l4() -> None:
    n, d = 3, 2
    x = sp.Matrix(n, d, lambda i, j: sp.Symbol(f"x{i}{j}"))      # rows of norm 1 (not needed below)
    r = sp.Symbol("r", positive=True)

    def centre(m):
        return m - sp.ones(n, 1) * (sp.ones(1, n) * m) / n

    # raw rows r x_i: centred raw rows equal r × the centred unit rows, a positive scalar multiple,
    # so each row normalises to the same unit vector and the two clouds' energies are equal
    assert sp.simplify(centre(r * x) - r * centre(x)) == sp.zeros(n, d)
    print("L4 ok: at equal norms the raw-centred rows are r × e1c's centred rows; e1r = e1c")


if __name__ == "__main__":
    l1()
    l2()
    l3()
    l4()
