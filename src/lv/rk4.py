"""Generic classical 4th-order Runge-Kutta integrator.

Knows nothing about any particular model: it integrates y' = f(t, y)
for a vector state y.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(f: RHS, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Advance y(t) to y(t + h) with one classical RK4 step."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_solve(f: RHS, y0, t0: float, t1: float, h: float):
    """Integrate y' = f(t, y) from t0 to t1 with fixed step h.

    The last step is shortened so that the grid ends exactly at t1.
    Returns (t, y) where t has shape (n,) and y has shape (n, len(y0)).
    """
    if h <= 0:
        raise ValueError("step h must be positive")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    n_full = int(np.floor((t1 - t0) / h + 1e-9))
    t = t0 + h * np.arange(n_full + 1)
    if t1 - t[-1] > 1e-12 * max(1.0, abs(t1)):
        t = np.append(t, t1)
    else:
        t[-1] = t1

    y = np.empty((len(t), np.size(y0)))
    y[0] = np.asarray(y0, dtype=float)
    for i in range(len(t) - 1):
        y[i + 1] = rk4_step(f, t[i], y[i], t[i + 1] - t[i])
    return t, y
