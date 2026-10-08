"""GUI-independent logic for the desktop app: input parsing, computation, formatting."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .analysis import summarize
from .model import LVParams, Solution, conserved_quantity, equilibrium, make_rhs, simulate

# Input fields shown in the form, in display order, with Russian labels.
FIELDS = {
    "a": "a — рождаемость жертв",
    "b": "b — гибель жертв при встрече",
    "c": "c — убыль хищников без еды",
    "d": "d — размножение хищников",
    "V0": "V0 — жертвы в начале",
    "P0": "P0 — хищники в начале",
    "h": "h — шаг RK4",
    "t1": "t1 — конец интервала",
}

MAX_STEPS = 2_000_000
SWEEP_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)


def default_inputs() -> dict[str, str]:
    p = LVParams()
    return {name: f"{getattr(p, name):g}" for name in FIELDS}


def parse_params(raw: dict[str, str]) -> LVParams:
    """Parse form strings into LVParams; raise ValueError listing every bad field."""
    values: dict[str, float] = {}
    errors: list[str] = []
    for name, label in FIELDS.items():
        text = raw.get(name, "").strip().replace(",", ".")
        try:
            x = float(text)
        except ValueError:
            errors.append(f"{label}: «{raw.get(name, '')}» — не число")
            continue
        if not math.isfinite(x):
            errors.append(f"{label}: должно быть конечным числом")
        elif x <= 0:
            errors.append(f"{label}: должно быть > 0")
        else:
            values[name] = x

    if not errors:
        if values["h"] >= values["t1"]:
            errors.append(f"{FIELDS['h']}: должен быть меньше t1")
        elif values["t1"] / values["h"] > MAX_STEPS:
            errors.append(
                f"{FIELDS['h']}: слишком мелкий шаг — больше {MAX_STEPS:,} шагов".replace(",", " ")
            )
    if errors:
        raise ValueError("\n".join(errors))
    known = {f.name for f in fields(LVParams)}
    return LVParams(**{k: v for k, v in values.items() if k in known})


@dataclass(frozen=True)
class Result:
    solution: Solution
    summary: dict
    H_drift: float
    residual: float   # max |f| at (c/d, a/b)


def compute(p: LVParams) -> Result:
    sol = simulate(p)
    H = conserved_quantity(p, sol.V, sol.P)
    residual = float(np.max(np.abs(make_rhs(p)(0.0, np.array(equilibrium(p))))))
    return Result(
        solution=sol,
        summary=summarize(sol),
        H_drift=float(np.max(np.abs(H - H[0]) / abs(H[0]))),
        residual=residual,
    )


def _g(x: float) -> str:
    return f"{x:.4g}"


def summary_rows(res: Result) -> list[tuple[str, str]]:
    s, p = res.summary, res.solution.params
    w = math.sqrt(p.a * p.c)
    T = s["T_num"]
    period = (f"{T:.4f}" if math.isfinite(T)
              else "не определён — увеличьте t1 (нужно ≥ 2 периодов)")
    return [
        ("Тривиальная точка (0, 0)", f"седло, λ = {_g(p.a)}, {_g(-p.c)}"),
        ("Нетривиальная точка (c/d, a/b)",
         f"({_g(s['V*'])}, {_g(s['P*'])}) — центр, λ = ±{w:.4f}·i"),
        ("|f| в точке (c/d, a/b)", f"{res.residual:.1e}"),
        ("Период (численно, RK4)", period),
        ("Период малых колебаний 2π/√(ac)", f"{s['T_lin']:.4f}"),
        ("Жертвы V: min … max", f"{_g(s['V_min'])} … {_g(s['V_max'])}"),
        ("Хищники P: min … max", f"{_g(s['P_min'])} … {_g(s['P_max'])}"),
        ("Относительный дрейф H", f"{res.H_drift:.1e}"),
        ("Шаг h (использован)", f"{res.solution.h:g}"
         + (" — уменьшен, чтобы V, P > 0" if res.solution.h < p.h else "")),
        ("Число точек", f"{len(res.solution.t)}"),
    ]


def sweep_values(p: LVParams, name: str) -> list[float]:
    base = getattr(p, name)
    return [base * k for k in SWEEP_FACTORS]


def write_csv(sol: Solution, path: Path) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "V", "P"])
        for row in zip(sol.t, sol.V, sol.P):
            w.writerow([repr(float(x)) for x in row])
    return path
