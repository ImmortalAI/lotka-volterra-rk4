from matplotlib.figure import Figure

from lv.analysis import sweep
from lv.model import LVParams, simulate
from lv.plots import (
    make_phase_portrait,
    make_sweep,
    make_sweep_summary,
    make_time_series,
    save_figure,
)

P = LVParams(t1=10.0, h=0.02)


def test_make_functions_return_figures_with_axes():
    sols = sweep(P, "a", [0.5, 1.0])
    figs = [
        make_time_series(simulate(P)),
        make_phase_portrait(P),
        make_sweep(P, "a", sols),
        make_sweep_summary({"a": sols}),
    ]
    for fig in figs:
        assert isinstance(fig, Figure)
        assert fig.axes


def test_save_figure(tmp_path):
    path = save_figure(make_time_series(simulate(P)), tmp_path / "sub" / "ts.png")
    assert path.exists() and path.stat().st_size > 0
