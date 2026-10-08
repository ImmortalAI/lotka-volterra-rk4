"""Lotka-Volterra predator-prey model.

    dV/dt = (a - b*P) * V      V - prey
    dP/dt = (-c + d*V) * P     P - predators

Stationary points (set both right-hand sides to zero):
    (a - bP) V = 0  ->  V = 0  or  P = a/b
    (-c + dV) P = 0 ->  P = 0  or  V = c/d
giving
    (0, 0)            - saddle, Jacobian eigenvalues a > 0 and -c < 0
    (c/d, a/b)        - center, eigenvalues +/- i*sqrt(a*c)

Along every trajectory H = d*V - c*ln V + b*P - a*ln P is conserved,
so orbits around (c/d, a/b) are closed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .rk4 import rk4_solve


@dataclass(frozen=True)
class LVParams:
    a: float = 1.0      # prey birth rate
    b: float = 0.1      # prey death rate per encounter with a predator
    c: float = 1.5      # predator death rate without food
    d: float = 0.075    # predator reproduction per eaten prey
    V0: float = 10.0    # initial prey
    P0: float = 5.0     # initial predators
    h: float = 0.01     # RK4 step
    t0: float = 0.0
    t1: float = 50.0

    def with_(self, **changes) -> "LVParams":
        return replace(self, **changes)

    def label(self) -> str:
        return f"a={self.a:g}, b={self.b:g}, c={self.c:g}, d={self.d:g}"


@dataclass(frozen=True)
class Solution:
    t: np.ndarray
    V: np.ndarray
    P: np.ndarray
    h: float           # step actually used
    params: LVParams


def make_rhs(p: LVParams):
    a, b, c, d = p.a, p.b, p.c, p.d

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        V, P = y
        return np.array([(a - b * P) * V, (-c + d * V) * P])

    return rhs


def equilibrium(p: LVParams) -> tuple[float, float]:
    """Non-trivial stationary point (V*, P*) = (c/d, a/b)."""
    return p.c / p.d, p.a / p.b


def jacobian(p: LVParams, V: float, P: float) -> np.ndarray:
    return np.array([
        [p.a - p.b * P, -p.b * V],
        [p.d * P, -p.c + p.d * V],
    ])


def conserved_quantity(p: LVParams, V, P):
    V = np.asarray(V, dtype=float)
    P = np.asarray(P, dtype=float)
    return p.d * V - p.c * np.log(V) + p.b * P - p.a * np.log(P)


def linear_period(p: LVParams) -> float:
    """Period of small oscillations around (c/d, a/b)."""
    return 2 * np.pi / np.sqrt(p.a * p.c)


def simulate(p: LVParams, max_halvings: int = 12) -> Solution:
    """Solve the model with RK4.

    If the step is too coarse and drives a population to zero or below,
    the step is halved and the integration repeated (no clamping).
    """
    h = p.h
    rhs = make_rhs(p)
    for _ in range(max_halvings + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            t, y = rk4_solve(rhs, [p.V0, p.P0], p.t0, p.t1, h)
        if np.all(np.isfinite(y)) and np.all(y > 0):
            return Solution(t=t, V=y[:, 0], P=y[:, 1], h=h, params=p)
        h /= 2
    raise RuntimeError(f"populations became non-positive even with h={h * 2:g}")
