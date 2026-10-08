"""Desktop UI for the Lotka-Volterra RK4 study.

Run:  .venv/bin/python app.py
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

from lv.analysis import sweep  # noqa: E402
from lv.app_logic import (  # noqa: E402
    FIELDS,
    compute,
    default_inputs,
    parse_params,
    summary_rows,
    sweep_values,
    write_csv,
)
from lv.plots import (  # noqa: E402
    PARAM_NAMES,
    make_phase_portrait,
    make_sweep,
    make_time_series,
)

ERROR_COLOR = "#b3261e"
OK_COLOR = "#52514e"


class App(ttk.Frame):
    TABS = {
        "time": "Динамика V(t), P(t)",
        "phase": "Фазовый портрет",
        "sweep": "Влияние коэффициента",
    }

    def __init__(self, root: tk.Tk):
        super().__init__(root, padding=10)
        self.root = root
        self.result = None
        self.params = None
        self.canvases: dict[str, FigureCanvasTkAgg] = {}
        self.toolbars: dict[str, NavigationToolbar2Tk] = {}

        root.title("Модель Лотки–Вольтерры — метод Рунге–Кутты 4-го порядка")
        root.geometry("1400x860")
        root.minsize(1000, 650)
        self.pack(fill=tk.BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_tabs()
        root.bind("<Return>", lambda _e: self.calculate())

    # ----- layout -------------------------------------------------------

    def _build_sidebar(self) -> None:
        side = ttk.Frame(self)
        side.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        eq = ttk.Label(side, justify=tk.LEFT, font=("TkFixedFont", 11),
                       text="dV/dt = (a − b·P)·V\ndP/dt = (−c + d·V)·P")
        eq.pack(anchor="w", pady=(0, 8))

        form = ttk.LabelFrame(side, text="Параметры", padding=8)
        form.pack(fill=tk.X)
        self.vars: dict[str, tk.StringVar] = {}
        for i, (name, label) in enumerate(FIELDS.items()):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar()
            ttk.Entry(form, textvariable=var, width=10).grid(row=i, column=1, sticky="e", pady=2,
                                                             padx=(8, 0))
            self.vars[name] = var
        self.reset()

        buttons = ttk.Frame(side)
        buttons.pack(fill=tk.X, pady=8)
        ttk.Button(buttons, text="Рассчитать", command=self.calculate).pack(fill=tk.X)
        row = ttk.Frame(buttons)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(row, text="По умолчанию", command=self.reset).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        self.csv_button = ttk.Button(row, text="Сохранить CSV…", command=self.save_csv,
                                     state=tk.DISABLED)
        self.csv_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        self.status_var = tk.StringVar(value="Введите параметры и нажмите «Рассчитать».")
        self.status = ttk.Label(side, textvariable=self.status_var, wraplength=330,
                                justify=tk.LEFT, foreground=OK_COLOR)
        self.status.pack(fill=tk.X, pady=(0, 8))

        res = ttk.LabelFrame(side, text="Результаты", padding=4)
        res.pack(fill=tk.BOTH, expand=True)
        self.table = ttk.Treeview(res, columns=("name", "value"), show="headings", height=11)
        self.table.heading("name", text="Величина")
        self.table.heading("value", text="Значение")
        self.table.column("name", width=190, anchor="w")
        self.table.column("value", width=170, anchor="w")
        self.table.pack(fill=tk.BOTH, expand=True)

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=1, sticky="nsew")
        self.tab_frames: dict[str, ttk.Frame] = {}
        for key, title in self.TABS.items():
            outer = ttk.Frame(self.notebook)
            self.notebook.add(outer, text=title)
            if key == "sweep":
                bar = ttk.Frame(outer, padding=(4, 4))
                bar.pack(fill=tk.X)
                ttk.Label(bar, text="Коэффициент:").pack(side=tk.LEFT)
                self.sweep_var = tk.StringVar(value="a")
                box = ttk.Combobox(bar, textvariable=self.sweep_var, state="readonly",
                                   values=list(PARAM_NAMES), width=4)
                box.pack(side=tk.LEFT, padx=6)
                box.bind("<<ComboboxSelected>>", lambda _e: self.on_sweep_change())
                ttk.Label(bar, foreground=OK_COLOR,
                          text="значения: ×0.5, ×0.75, ×1, ×1.5, ×2 от текущего; "
                               "остальные параметры и начальное состояние — из формы").pack(
                    side=tk.LEFT, padx=6)
            frame = ttk.Frame(outer)
            frame.pack(fill=tk.BOTH, expand=True)
            ttk.Label(frame, text="Графики появятся после расчёта.",
                      foreground=OK_COLOR).pack(expand=True)
            self.tab_frames[key] = frame

    # ----- actions ------------------------------------------------------

    def reset(self) -> None:
        for name, value in default_inputs().items():
            self.vars[name].set(value)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_var.set(text)
        self.status.configure(foreground=ERROR_COLOR if error else OK_COLOR)

    def calculate(self) -> None:
        try:
            params = parse_params({k: v.get() for k, v in self.vars.items()})
        except ValueError as e:
            self._set_status(f"Ошибка ввода:\n{e}", error=True)
            return

        self._set_status("Считаю…")
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            result = compute(params)
            figs = {
                "time": make_time_series(result.solution),
                "phase": make_phase_portrait(params),
                "sweep": self._sweep_figure(params),
            }
        except (RuntimeError, FloatingPointError, OverflowError) as e:
            self._set_status(f"Ошибка расчёта: {e}", error=True)
            return
        finally:
            self.root.configure(cursor="")

        self.params, self.result = params, result
        self.table.delete(*self.table.get_children())
        for name, value in summary_rows(result):
            self.table.insert("", tk.END, values=(name, value))
        for key, fig in figs.items():
            self._show(key, fig)
        self.csv_button.configure(state=tk.NORMAL)
        self._set_status(f"Готово: {len(result.solution.t)} точек, h = {result.solution.h:g}.")

    def _sweep_figure(self, params):
        name = self.sweep_var.get()
        return make_sweep(params, name, sweep(params, name, sweep_values(params, name)))

    def on_sweep_change(self) -> None:
        if self.params is None:
            return
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            self._show("sweep", self._sweep_figure(self.params))
        except (RuntimeError, FloatingPointError, OverflowError) as e:
            self._set_status(f"Ошибка расчёта: {e}", error=True)
        finally:
            self.root.configure(cursor="")

    def _show(self, key: str, fig) -> None:
        frame = self.tab_frames[key]
        for child in frame.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        self.canvases[key] = canvas
        self.toolbars[key] = toolbar

    def save_csv(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Сохранить решение", defaultextension=".csv",
            initialfile="lotka_volterra.csv", filetypes=[("CSV", "*.csv")])
        if path:
            write_csv(self.result.solution, Path(path))
            self._set_status(f"Сохранено: {path}")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
