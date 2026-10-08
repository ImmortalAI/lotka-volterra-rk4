"""Figures: time series, phase portrait, parameter sweeps."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from .analysis import summarize
from .model import LVParams, Solution, equilibrium, simulate

# Reference palette (dataviz skill): categorical slots 1-2, ordinal blue ramp, ink.
PREY = "#2a78d6"
PRED = "#eb6834"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8980"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

PARAM_NAMES = {
    "a": "a (рождаемость жертв)",
    "b": "b (гибель жертв при встрече)",
    "c": "c (убыль хищников без еды)",
    "d": "d (размножение хищников от еды)",
}

matplotlib.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "figure.dpi": 110,
})


def _ramp(n: int) -> list[str]:
    idx = np.linspace(0, len(RAMP) - 1, n).round().astype(int)
    return [RAMP[i] for i in idx]


def save_figure(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def make_time_series(sol: Solution) -> Figure:
    p = sol.params
    Vs, Ps = equilibrium(p)
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(sol.t, sol.V, color=PREY, label="V(t) — жертвы")
    ax.plot(sol.t, sol.P, color=PRED, label="P(t) — хищники")
    ax.axhline(Vs, color=PREY, lw=1, ls=":", alpha=0.8)
    ax.axhline(Ps, color=PRED, lw=1, ls=":", alpha=0.8)
    ax.text(sol.t[-1], Vs, f" V*={Vs:g}", va="center", color=INK_2, fontsize=9)
    ax.text(sol.t[-1], Ps, f" P*={Ps:g}", va="center", color=INK_2, fontsize=9)
    ax.set_xlabel("время t")
    ax.set_ylabel("численность")
    ax.set_xlim(sol.t[0], sol.t[-1])
    ax.set_ylim(0, max(sol.V.max(), sol.P.max()) * 1.18)
    ax.set_title(
        "Динамика численности жертв и хищников (RK4)\n"
        f"{p.label()}, V0={p.V0:g}, P0={p.P0:g}, h={sol.h:g}"
    )
    ax.legend(loc="upper right", ncols=2)
    return fig


def make_phase_portrait(p: LVParams, scales=(0.2, 0.4, 0.6, 0.8, 1.0)) -> Figure:
    """Several orbits, starting at (V*, s*P0') for a range of distances from the center."""
    Vs, Ps = equilibrium(p)
    fig = Figure(figsize=(10.5, 6.5), layout="constrained")
    ax = fig.subplots()

    # Initial points on the segment from the center toward (V0, P0).
    sols = []
    for s in scales:
        V0 = Vs + s * (p.V0 - Vs)
        P0 = Ps + s * (p.P0 - Ps)
        sols.append(simulate(p.with_(V0=V0, P0=P0)))

    vmax = max(s.V.max() for s in sols) * 1.08
    pmax = max(s.P.max() for s in sols) * 1.08
    VV, PP = np.meshgrid(np.linspace(0.5, vmax, 22), np.linspace(0.5, pmax, 22))
    dV = (p.a - p.b * PP) * VV
    dP = (-p.c + p.d * VV) * PP
    norm = np.hypot(dV, dP) + 1e-12
    ax.quiver(VV, PP, dV / norm, dP / norm, color=AXIS, angles="xy",
              scale=32, width=0.0025, headwidth=4, zorder=1)

    for sol, color, s in zip(sols, _ramp(len(sols)), scales):
        is_main = np.isclose(s, 1.0)
        ax.plot(sol.V, sol.P, color=color, lw=2.4 if is_main else 1.6, zorder=2,
                label=f"V0={sol.params.V0:.3g}, P0={sol.params.P0:.3g}"
                + (" (основная)" if is_main else ""))
        ax.plot(sol.V[0], sol.P[0], "o", ms=5, color=color, mec=SURFACE, mew=1.2, zorder=3)

    ax.axvline(Vs, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.axhline(Ps, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.plot(Vs, Ps, "*", ms=15, color=INK, mec=SURFACE, mew=1.2, zorder=4,
            label=f"стационарная точка (c/d, a/b) = ({Vs:g}, {Ps:g})")
    ax.plot(0, 0, "X", ms=10, color=INK, mec=SURFACE, mew=1.2, zorder=4, clip_on=False,
            label="тривиальная точка (0, 0) — седло")
    ax.annotate("V = c/d", (Vs, pmax), xytext=(4, -12), textcoords="offset points",
                color=INK_2, fontsize=9)
    ax.annotate("P = a/b", (0, Ps), xytext=(4, 4), textcoords="offset points",
                color=INK_2, fontsize=9)

    ax.set_xlim(0, vmax)
    ax.set_ylim(0, pmax)
    ax.set_xlabel("V — жертвы")
    ax.set_ylabel("P — хищники")
    ax.set_title(f"Фазовый портрет (RK4, h={p.h:g})\n{p.label()}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    return fig


def make_sweep(base: LVParams, name: str, sols: list[Solution]) -> Figure:
    """Phase orbits + V(t) + P(t) for several values of one coefficient."""
    colors = _ramp(len(sols))
    fig = Figure(figsize=(13, 7.5), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
    ax_ph = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1], sharex=ax_v)

    for sol, color in zip(sols, colors):
        val = getattr(sol.params, name)
        Vs, Ps = equilibrium(sol.params)
        lbl = f"{name}={val:g}  (V*={Vs:.3g}, P*={Ps:.3g})"
        ax_ph.plot(sol.V, sol.P, color=color, lw=1.8, label=lbl)
        ax_ph.plot(Vs, Ps, "o", ms=8, color=color, mec=INK, mew=0.8, zorder=3)
        ax_v.plot(sol.t, sol.V, color=color, lw=1.6, label=f"{name}={val:g}")
        ax_p.plot(sol.t, sol.P, color=color, lw=1.6, label=f"{name}={val:g}")

    ax_ph.plot(base.V0, base.P0, "s", ms=7, color=INK, zorder=4,
               label=f"общее начальное состояние ({base.V0:g}, {base.P0:g})")
    ax_ph.set_xlim(left=0)
    ax_ph.set_ylim(bottom=0)
    ax_ph.set_xlabel("V — жертвы")
    ax_ph.set_ylabel("P — хищники")
    ax_ph.set_title("Фазовые траектории (● — равновесие)")
    ax_ph.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncols=2)

    ax_v.set_ylabel("V(t) — жертвы")
    ax_v.set_title("Жертвы V(t)")
    ax_v.set_ylim(bottom=0)
    ax_p.set_ylabel("P(t) — хищники")
    ax_p.set_xlabel("время t")
    ax_p.set_title("Хищники P(t)")
    ax_p.set_ylim(bottom=0)
    ax_p.set_xlim(base.t0, base.t1)

    others = ", ".join(f"{k}={getattr(base, k):g}" for k in "abcd" if k != name)
    fig.suptitle(f"Влияние коэффициента {PARAM_NAMES[name]}; остальные: {others}",
                 color=INK, fontsize=13)
    return fig


def make_sweep_summary(sweeps: dict[str, list[Solution]]) -> Figure:
    """Equilibrium, period and amplitude vs each coefficient (one column per coefficient)."""
    fig = Figure(figsize=(4 * len(sweeps), 10), layout="constrained")
    axes = fig.subplots(3, len(sweeps), sharey="row", squeeze=False)
    for j, (name, sols) in enumerate(sweeps.items()):
        rows = [summarize(s) for s in sols]
        x = np.array([r[name] for r in rows])
        col = axes[:, j]

        col[0].plot(x, [r["V*"] for r in rows], "-o", color=PREY, ms=6, label="V* = c/d")
        col[0].plot(x, [r["P*"] for r in rows], "-o", color=PRED, ms=6, label="P* = a/b")
        col[0].set_title(f"Равновесие vs {name}")

        col[1].plot(x, [r["T_num"] for r in rows], "-o", color=INK, ms=6,
                    label="T (RK4, данная орбита)")
        col[1].plot(x, [r["T_lin"] for r in rows], "--", color=MUTED,
                    label="2π/√(ac) (малые колебания)")
        col[1].set_title(f"Период vs {name}")

        col[2].plot(x, [r["V_max"] - r["V_min"] for r in rows], "-o", color=PREY, ms=6,
                    label="размах V")
        col[2].plot(x, [r["P_max"] - r["P_min"] for r in rows], "-o", color=PRED, ms=6,
                    label="размах P")
        col[2].set_title(f"Амплитуда (max−min) vs {name}")
        col[2].set_xlabel(PARAM_NAMES[name])

    axes[0, 0].set_ylabel("численность")
    axes[1, 0].set_ylabel("период T")
    axes[2, 0].set_ylabel("max − min")
    for i in range(3):
        axes[i, 0].legend(fontsize=8)
        for ax in axes[i]:
            ax.set_ylim(bottom=0)
    fig.suptitle("Сводка параметрических прогонов (один коэффициент меняется, остальные "
                 "по умолчанию; начальное состояние V0=10, P0=5)", color=INK, fontsize=13)
    return fig
