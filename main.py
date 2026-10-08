"""Run the full Lotka-Volterra study and save figures to figures/."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lv.analysis import summarize, sweep  # noqa: E402
from lv.model import (  # noqa: E402
    LVParams,
    conserved_quantity,
    equilibrium,
    jacobian,
    linear_period,
    make_rhs,
    simulate,
)
from lv.plots import (  # noqa: E402
    make_phase_portrait,
    make_sweep,
    make_sweep_summary,
    make_time_series,
    save_figure,
)
from lv.rk4 import rk4_solve  # noqa: E402

FIG = ROOT / "figures"

PARAMS = LVParams()
SWEEP_VALUES = {
    "a": [0.5, 0.75, 1.0, 1.5, 2.0],
    "b": [0.05, 0.075, 0.1, 0.15, 0.2],
    "c": [0.75, 1.0, 1.5, 2.0, 3.0],
    "d": [0.0375, 0.05, 0.075, 0.1, 0.15],
}


def check_convergence() -> None:
    print("RK4 order check on y' = y, y(0) = 1, t in [0, 1]:")
    prev = None
    for h in (0.1, 0.05, 0.025, 0.0125):
        _, y = rk4_solve(lambda t, y: y, [1.0], 0.0, 1.0, h)
        err = abs(y[-1, 0] - np.e)
        ratio = f"  ratio={prev / err:6.2f}" if prev else ""
        print(f"  h={h:<7g} error={err:.3e}{ratio}")
        prev = err


def check_stationary_points(p: LVParams) -> None:
    f = make_rhs(p)
    Vs, Ps = equilibrium(p)
    print(f"\nStationary points for {p.label()}:")
    for name, pt in (("(0, 0)", (0.0, 0.0)), ("(c/d, a/b)", (Vs, Ps))):
        rhs = f(0.0, np.array(pt))
        ev = np.linalg.eigvals(jacobian(p, *pt))
        print(f"  {name:<11} = ({pt[0]:g}, {pt[1]:g})  f = {rhs}  eigenvalues = {np.round(ev, 6)}")
    print(f"  small-oscillation period 2*pi/sqrt(ac) = {linear_period(p):.4f}")


def main() -> None:
    check_convergence()
    check_stationary_points(PARAMS)

    sol = simulate(PARAMS)
    H = conserved_quantity(PARAMS, sol.V, sol.P)
    s = summarize(sol)
    print(f"\nBase run: h={sol.h:g}, {len(sol.t)} points")
    print(f"  V in [{s['V_min']:.4f}, {s['V_max']:.4f}], P in [{s['P_min']:.4f}, {s['P_max']:.4f}]")
    print(f"  period (numerical) = {s['T_num']:.4f}, linear = {s['T_lin']:.4f}")
    print(f"  max relative drift of H = {np.max(np.abs(H - H[0]) / abs(H[0])):.2e}")

    paths = [
        save_figure(make_time_series(sol), FIG / "time_series.png"),
        save_figure(make_phase_portrait(PARAMS), FIG / "phase_portrait.png"),
    ]

    sweeps = {}
    header = f"{'':>3} {'value':>7} {'V*':>7} {'P*':>7} {'Vmin':>7} {'Vmax':>7} " \
             f"{'Pmin':>7} {'Pmax':>7} {'T_num':>7} {'T_lin':>7} {'h':>6}"
    print("\nParameter sweeps (others at default, V0=10, P0=5):")
    print(header)
    for name, values in SWEEP_VALUES.items():
        sweeps[name] = sweep(PARAMS, name, values)
        paths.append(save_figure(make_sweep(PARAMS, name, sweeps[name]), FIG / f"sweep_{name}.png"))
        for sl in sweeps[name]:
            r = summarize(sl)
            print(f"{name:>3} {r[name]:>7g} {r['V*']:>7.2f} {r['P*']:>7.2f} "
                  f"{r['V_min']:>7.2f} {r['V_max']:>7.2f} {r['P_min']:>7.2f} "
                  f"{r['P_max']:>7.2f} {r['T_num']:>7.2f} {r['T_lin']:>7.2f} {r['h']:>6g}")
    paths.append(save_figure(make_sweep_summary(sweeps), FIG / "sweep_summary.png"))

    print("\nSaved figures:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
