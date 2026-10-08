import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lv.model import (
    LVParams,
    conserved_quantity,
    equilibrium,
    jacobian,
    make_rhs,
    simulate,
)


@pytest.fixture
def p():
    return LVParams()


def test_defaults(p):
    assert (p.a, p.b, p.c, p.d) == (1.0, 0.1, 1.5, 0.075)
    assert (p.V0, p.P0, p.h, p.t0, p.t1) == (10.0, 5.0, 0.01, 0.0, 50.0)


def test_equilibrium_general_form(p):
    assert equilibrium(p) == pytest.approx((p.c / p.d, p.a / p.b))
    assert equilibrium(p) == pytest.approx((20.0, 10.0))


def test_rhs_vanishes_at_both_stationary_points(p):
    f = make_rhs(p)
    np.testing.assert_allclose(f(0.0, np.array(equilibrium(p))), 0.0, atol=1e-12)
    np.testing.assert_allclose(f(0.0, np.array([0.0, 0.0])), 0.0, atol=0)


def test_rhs_vanishes_at_equilibrium_for_random_params():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b, c, d = rng.uniform(0.05, 3.0, size=4)
        q = LVParams(a=a, b=b, c=c, d=d)
        np.testing.assert_allclose(
            make_rhs(q)(0.0, np.array(equilibrium(q))), 0.0, atol=1e-12
        )


def test_origin_is_saddle_and_coexistence_is_center(p):
    ev0 = np.linalg.eigvals(jacobian(p, 0.0, 0.0))
    assert sorted(ev0.real) == pytest.approx([-p.c, p.a])
    ev1 = np.linalg.eigvals(jacobian(p, *equilibrium(p)))
    assert ev1.real == pytest.approx([0.0, 0.0], abs=1e-12)
    assert sorted(abs(ev1.imag)) == pytest.approx([np.sqrt(p.a * p.c)] * 2)


def test_conserved_quantity_drift_is_tiny(p):
    sol = simulate(p)
    H = conserved_quantity(p, sol.V, sol.P)
    assert np.max(np.abs(H - H[0]) / abs(H[0])) < 1e-8


def test_populations_stay_positive(p):
    sol = simulate(p)
    assert sol.V.min() > 0 and sol.P.min() > 0


def test_simulate_reduces_step_if_populations_go_negative():
    # h=1.5 is far too coarse here and makes RK4 overshoot below zero
    q = LVParams(h=1.5, t1=30.0)
    sol = simulate(q)
    assert sol.h < 1.5
    assert sol.V.min() > 0 and sol.P.min() > 0


def test_matches_scipy_reference(p):
    sol = simulate(p)
    ref = solve_ivp(
        make_rhs(p), (p.t0, p.t1), [p.V0, p.P0],
        t_eval=sol.t, rtol=1e-11, atol=1e-11, method="DOP853",
    )
    np.testing.assert_allclose(sol.V, ref.y[0], rtol=1e-6)
    np.testing.assert_allclose(sol.P, ref.y[1], rtol=1e-6)


def test_small_oscillation_period_matches_linear_theory(p):
    Vs, Ps = equilibrium(p)
    q = LVParams(V0=Vs * 1.01, P0=Ps, t1=40.0, h=0.005)
    sol = simulate(q)
    from lv.analysis import estimate_period

    assert estimate_period(sol.t, sol.V) == pytest.approx(
        2 * np.pi / np.sqrt(p.a * p.c), rel=1e-3
    )
