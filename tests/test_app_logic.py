import csv
import math

import numpy as np
import pytest

from lv.app_logic import (
    FIELDS,
    compute,
    default_inputs,
    parse_params,
    summary_rows,
    sweep_values,
    write_csv,
)
from lv.model import LVParams


def test_default_inputs_round_trip():
    assert parse_params(default_inputs()) == LVParams()


def test_parse_accepts_comma_decimal_and_spaces():
    raw = default_inputs() | {"a": " 1,5 ", "h": "0,005"}
    p = parse_params(raw)
    assert p.a == 1.5 and p.h == 0.005


@pytest.mark.parametrize("field,value", [
    ("a", "abc"), ("b", ""), ("c", "-1"), ("d", "0"),
    ("V0", "0"), ("P0", "-5"), ("h", "0"), ("t1", "-10"),
    ("a", "nan"), ("t1", "inf"),
])
def test_parse_rejects_invalid_values(field, value):
    with pytest.raises(ValueError) as exc:
        parse_params(default_inputs() | {field: value})
    assert FIELDS[field] in str(exc.value)


def test_parse_reports_all_bad_fields_at_once():
    with pytest.raises(ValueError) as exc:
        parse_params(default_inputs() | {"a": "x", "h": "-1"})
    msg = str(exc.value)
    assert FIELDS["a"] in msg and FIELDS["h"] in msg


def test_parse_rejects_step_not_smaller_than_interval():
    with pytest.raises(ValueError):
        parse_params(default_inputs() | {"h": "60", "t1": "50"})


def test_parse_rejects_too_many_steps():
    with pytest.raises(ValueError):
        parse_params(default_inputs() | {"h": "1e-6", "t1": "1000"})


def test_compute_and_summary_rows():
    res = compute(LVParams())
    assert res.solution.V.min() > 0
    rows = dict(summary_rows(res))
    assert "(20, 10)" in rows["Нетривиальная точка (c/d, a/b)"]
    assert "центр" in rows["Нетривиальная точка (c/d, a/b)"]
    assert "седло" in rows["Тривиальная точка (0, 0)"]
    assert rows["Период малых колебаний 2π/√(ac)"].startswith("5.130")
    assert rows["Период (численно, RK4)"].startswith("5.479")
    assert rows["Жертвы V: min … max"] == "7.926 … 40.6"
    assert float(rows["Относительный дрейф H"]) < 1e-8


def test_summary_handles_undetermined_period():
    res = compute(LVParams(t1=2.0))
    assert math.isnan(res.summary["T_num"])
    assert "не определён" in dict(summary_rows(res))["Период (численно, RK4)"]


def test_sweep_values_are_centered_on_current_value():
    vals = sweep_values(LVParams(c=2.0), "c")
    assert vals == pytest.approx([1.0, 1.5, 2.0, 3.0, 4.0])


def test_write_csv(tmp_path):
    res = compute(LVParams(t1=1.0, h=0.25))
    path = tmp_path / "out.csv"
    write_csv(res.solution, path)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["t", "V", "P"]
    assert len(rows) == 1 + len(res.solution.t)
    np.testing.assert_allclose(float(rows[-1][1]), res.solution.V[-1])
