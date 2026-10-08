import numpy as np
import pytest

from lv.rk4 import rk4_solve, rk4_step


def exp_rhs(t, y):
    return y


def test_single_step_matches_taylor_to_fourth_order():
    h = 0.1
    y1 = rk4_step(exp_rhs, 0.0, np.array([1.0]), h)
    taylor = 1 + h + h**2 / 2 + h**3 / 6 + h**4 / 24
    assert y1[0] == pytest.approx(taylor, abs=1e-15)


def test_exponential_growth_matches_exact_solution():
    t, y = rk4_solve(exp_rhs, [1.0], 0.0, 1.0, 0.01)
    assert t[0] == 0.0 and t[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(y[:, 0], np.exp(t), rtol=1e-9)


def test_global_error_is_fourth_order():
    errors = []
    for h in (0.1, 0.05, 0.025):
        _, y = rk4_solve(exp_rhs, [1.0], 0.0, 1.0, h)
        errors.append(abs(y[-1, 0] - np.e))
    ratios = [errors[i] / errors[i + 1] for i in range(2)]
    for r in ratios:
        assert 15.0 < r < 17.0


def test_vector_system_harmonic_oscillator():
    # y'' = -y  ->  [y, y']' = [y', -y]
    def rhs(t, y):
        return np.array([y[1], -y[0]])

    t, y = rk4_solve(rhs, [1.0, 0.0], 0.0, 2 * np.pi, 0.001)
    np.testing.assert_allclose(y[:, 0], np.cos(t), atol=1e-10)


def test_end_point_is_hit_exactly_when_interval_not_multiple_of_h():
    t, y = rk4_solve(exp_rhs, [1.0], 0.0, 1.0, 0.3)
    assert t[-1] == pytest.approx(1.0)
    assert y[-1, 0] == pytest.approx(np.e, rel=1e-3)


def test_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        rk4_solve(exp_rhs, [1.0], 0.0, 1.0, 0.0)
