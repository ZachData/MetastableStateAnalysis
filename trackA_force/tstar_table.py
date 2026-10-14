"""t*(n, beta, g0): time for eq. (6.9) to carry gamma from g0 to 0.9 (`design-A.md` "t* before any output").

No model, no data: the ODE of `p1c_frames/gamma_ode.py` (its RHS, `gamma_rhs_sa`) integrated
with an adaptive solver, because the module's fixed-step RK4 at t_max = 12 returns inf from
beta ~ 7 (the time is finite but longer than 12). Agrees with `collapse_time` where both finish
(n = 467, beta = 3.5: 4.332).

    python trackA_force/tstar_table.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from p1c_frames.gamma_ode import gamma_rhs_sa

NS = (467, 512, 1032, 2041)
BETAS = (0.0, 1.6, 3.5, 5.6, 7.0, 8.0, 10.0)
G0S = (0.0, 0.3, 0.6)


def tstar(n: int, beta: float, g0: float = 0.0, target: float = 0.9) -> float:
    """Refuses a start at or above the target or at or below the simplex point -1/(n-1),
    where the event never fires and inf would read as "T is below t*" (#201, finding 4)."""
    if not (-1.0 / (n - 1) < g0 < target):
        raise ValueError(f"g0 = {g0} outside (-1/(n-1), {target})")
    ev = lambda t, g: g[0] - target
    ev.terminal = True
    s = solve_ivp(lambda t, g: [gamma_rhs_sa(g[0], n, beta)], (0.0, 1e7), [g0],
                  events=ev, rtol=1e-10, atol=1e-12, method="LSODA")
    return float(s.t_events[0][0]) if len(s.t_events[0]) else float("inf")


def main() -> None:
    for g0 in G0S:
        print(f"g0 = {g0}")
        print("n     " + "".join(f"b={b:<6}" for b in BETAS))
        for n in NS:
            print(f"{n:<6}" + "".join(f"{tstar(n, b, g0):<8.2f}" for b in BETAS))


if __name__ == "__main__":
    main()
