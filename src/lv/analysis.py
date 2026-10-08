"""Post-processing of trajectories: period and amplitude estimates."""
from __future__ import annotations

import numpy as np

from .model import LVParams, Solution, equilibrium, linear_period, simulate


def estimate_period(t: np.ndarray, x: np.ndarray) -> float:
    """Mean period from upward crossings of the signal's mean level.

    Crossing times are found by linear interpolation between grid points.
    Returns nan if fewer than two crossings are found.
    """
    level = 0.5 * (x.min() + x.max())
    s = x - level
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(idx) < 2:
        return float("nan")
    tc = t[idx] - s[idx] * (t[idx + 1] - t[idx]) / (s[idx + 1] - s[idx])
    return float(np.mean(np.diff(tc)))


def summarize(sol: Solution) -> dict:
    p = sol.params
    Vs, Ps = equilibrium(p)
    return {
        "a": p.a, "b": p.b, "c": p.c, "d": p.d,
        "V*": Vs, "P*": Ps,
        "V_min": sol.V.min(), "V_max": sol.V.max(),
        "P_min": sol.P.min(), "P_max": sol.P.max(),
        "T_num": estimate_period(sol.t, sol.V),
        "T_lin": linear_period(p),
        "h": sol.h,
    }


def sweep(base: LVParams, name: str, values) -> list[Solution]:
    """Vary one coefficient, keeping the others and the initial state fixed."""
    return [simulate(base.with_(**{name: v})) for v in values]
