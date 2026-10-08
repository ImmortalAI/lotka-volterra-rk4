import pytest

tk = pytest.importorskip("tkinter")

from app import App  # noqa: E402
from lv.app_logic import FIELDS  # noqa: E402


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    root.withdraw()
    a = App(root)
    yield a
    root.destroy()


def test_initial_state(app):
    assert app.vars["a"].get() == "1"
    assert str(app.csv_button["state"]) == "disabled"


def test_calculate_fills_results_and_plots(app):
    app.calculate()
    assert app.status_var.get().startswith("Готово")
    rows = {app.table.item(i)["values"][0]: app.table.item(i)["values"][1]
            for i in app.table.get_children()}
    assert "центр" in str(rows["Нетривиальная точка (c/d, a/b)"])
    assert set(app.canvases) == {"time", "phase", "sweep"}
    assert str(app.csv_button["state"]) == "normal"


def test_invalid_input_shows_error_and_keeps_previous_results(app):
    app.calculate()
    n_rows = len(app.table.get_children())
    app.vars["a"].set("abc")
    app.vars["h"].set("-1")
    app.calculate()
    msg = app.status_var.get()
    assert FIELDS["a"] in msg and FIELDS["h"] in msg
    assert len(app.table.get_children()) == n_rows


def test_changing_sweep_coefficient_redraws(app):
    app.calculate()
    old = app.canvases["sweep"].figure
    app.sweep_var.set("d")
    app.on_sweep_change()
    fig = app.canvases["sweep"].figure
    assert fig is not old
    assert "коэффициента d" in fig._suptitle.get_text()


def test_reset_restores_defaults(app):
    app.vars["a"].set("7")
    app.reset()
    assert app.vars["a"].get() == "1"
