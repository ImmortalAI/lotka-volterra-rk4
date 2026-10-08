# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project goal

Numerically study the Lotka–Volterra predator–prey model (source: `Task.pdf`, in Russian):

```
dV/dt = (a − b·P)·V
dP/dt = (−c + d·V)·P
```

| Symbol | Meaning |
|--------|---------|
| `V` | prey population (жертвы) |
| `P` | predator population (хищники) |
| `t` | time |
| `a` | prey birth rate |
| `b` | prey death rate per encounter with a predator |
| `c` | predator death rate when food is scarce |
| `d` | rate at which eaten prey sustains predator reproduction |

> Note: the PDF's item 2 says "predators V(t) and prey P(t)". That's a typo. By the definitions and the equations, **V is prey and P is predators**. Use those definitions everywhere.

## Required deliverables

1. **RK4 solver.** Solve the system with a hand-written classical 4th-order Runge–Kutta method. Don't use library integrators (`scipy.integrate.solve_ivp`, `odeint`, etc.) for the main solution. They may be used only in tests, as a reference to cross-check against.
2. **Time plots.** Plot `V(t)` (prey) and `P(t)` (predators) on a shared time axis, with a legend and axis labels.
3. **Phase portrait.** Plot prey `V` on the X axis and predators `P` on the Y axis. Mark the stationary point. Ideally show several trajectories from different initial conditions.
4. **Stationary points (general form).** Derive them analytically and state them in the code or report:
   - trivial point: `(V, P) = (0, 0)`, a saddle
   - non-trivial point: `(V*, P*) = (c/d, a/b)`, a center (closed orbits around it)

   Check numerically that both derivatives vanish at `(c/d, a/b)`.
5. **Conclusions.** Describe how each coefficient `a`, `b`, `c`, `d` affects the system: oscillation amplitude, period, and where the equilibrium sits. Back this up with parameter sweeps, varying one coefficient while holding the others fixed, and plotting the results. Key points:
   - `a` ↑ → `P*` rises (more predators at equilibrium), prey grow faster
   - `b` ↑ → `P*` falls, prey are hit harder by each encounter
   - `c` ↑ → `V*` rises, predators die out faster without food
   - `d` ↑ → `V*` falls, predators convert prey into offspring more efficiently
   - Small-oscillation period ≈ `2π / √(a·c)`

## Suggested stack and layout

No language is mandated. Python is the default unless the user says otherwise:

- Python 3.10+, `numpy`, `matplotlib` (put dependencies in `requirements.txt`)
- `pytest` for tests

Suggested structure:

```
src/lv/model.py       # right-hand side f(t, y; a, b, c, d), equilibrium()
src/lv/rk4.py         # generic rk4_step / rk4_solve(f, y0, t0, t1, h)
src/lv/plots.py       # time series, phase portrait, parameter sweeps
main.py               # runs everything, saves figures to figures/
tests/                # unit tests
REPORT.md             # results, stationary-point derivation, conclusions (in Russian, like the task)
```

## Implementation guidelines

- Keep the RK4 integrator generic (`f(t, y) -> dy/dt` over vector `y`). It shouldn't know about the Lotka–Volterra model.
- Put the parameters and initial conditions in one place (a dataclass or config dict). Use a sensible default such as `a=1.0, b=0.1, c=1.5, d=0.075, V0=10, P0=5, h=0.01, t∈[0, 50]`.
- Populations must stay non-negative. If the step size breaks that, reduce `h` rather than clamping the values.
- Save figures to `figures/` as PNG, with titles, axis labels, legends, and parameter values in the title or caption.
- Plot labels may be in Russian to match the task. Keep code identifiers and comments in English.

## Testing / verification

- RK4 on `dy/dt = y` with `y(0)=1` should match `e^t`. Halving `h` should cut the global error by about 16×, which confirms 4th order.
- The right-hand side should be zero at `(c/d, a/b)` and at `(0, 0)`.
- The conserved quantity `H = d·V − c·ln V + b·P − a·ln P` should stay nearly constant along a trajectory. Its relative drift is a good accuracy check.
- Run `pytest` and `python main.py` before saying the work is done.
