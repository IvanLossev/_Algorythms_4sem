import threading
import traceback
import tkinter as tk
from contextlib import suppress
from tkinter import ttk, messagebox

import matplotlib

# Важно: backend должен быть TkAgg для встраивания в Tkinter.
# Если pyplot уже импортирован (например, GUI стартует из `GA.py`), то backend
# мог быть уже выбран — поэтому используем force=True.
with suppress(Exception):
    matplotlib.use("TkAgg", force=True)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


KNOWN_MINIMA = [
    (1.34941, -1.34941),
    (1.34941, 1.34941),
    (-1.34941, 1.34941),
    (-1.34941, -1.34941),
]


def get_algorithms():
    """
    Когда GUI запускается из `GA.py` как скрипта, алгоритмы уже находятся в `__main__`.
    Тогда повторно импортировать `GA.py` не нужно (иначе файл выполнится дважды).
    """
    try:
        from __main__ import Genetic_algo, ParticleSwarm  # type: ignore

        return Genetic_algo, ParticleSwarm
    except Exception:
        # Подхватываем реализации из `GA_PSO.py`, чтобы `GA.py` можно было удалить.
        from GA_PSO import Genetic_algo, ParticleSwarm  # type: ignore

        return Genetic_algo, ParticleSwarm


class SearchGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Поиск минимума функции (GA / PSO)")
        self.geometry("980x620")

        # Верхняя панель управления
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(ctrl, text="Алгоритм:").pack(side=tk.LEFT)

        self.algo_var = tk.StringVar(value="GA")
        self.algo_combo = ttk.Combobox(
            ctrl,
            textvariable=self.algo_var,
            values=["GA", "PSO"],
            state="readonly",
            width=10,
        )
        self.algo_combo.pack(side=tk.LEFT, padx=(8, 18))

        ttk.Label(ctrl, text="Границы (min,max):").pack(side=tk.LEFT)
        self.bounds_var = tk.StringVar(value="-5,5")
        bounds_entry = ttk.Entry(ctrl, textvariable=self.bounds_var, width=12)
        bounds_entry.pack(side=tk.LEFT, padx=(8, 18))

        ttk.Label(ctrl, text="Популяция/частицы:").pack(side=tk.LEFT)
        self.n_var = tk.StringVar(value="100")
        n_entry = ttk.Entry(ctrl, textvariable=self.n_var, width=6)
        n_entry.pack(side=tk.LEFT, padx=(8, 18))

        ttk.Label(ctrl, text="Поколения/итерации:").pack(side=tk.LEFT)
        self.gen_var = tk.StringVar(value="200")
        gen_entry = ttk.Entry(ctrl, textvariable=self.gen_var, width=6)
        gen_entry.pack(side=tk.LEFT, padx=(8, 18))

        self.run_btn = ttk.Button(ctrl, text="Запустить", command=self.on_run)
        self.run_btn.pack(side=tk.RIGHT)

        # Основная часть: график + текст результата
        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.result_text = tk.Text(main, height=6, wrap="word")
        self.result_text.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        self.result_text.configure(state="disabled")

        fig = Figure(figsize=(10.5, 5.2), dpi=100)
        self.ax_history = fig.add_subplot(111)

        self.ax_history.set_title("Сходимость (лучшее значение f)")
        self.ax_history.set_xlabel("Шаг / поколение")
        self.ax_history.set_ylabel("min f")
        self.ax_history.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(fig, master=main)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def set_result(self, text: str):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state="disabled")

    def parse_bounds(self):
        # Ожидаем формат: "-5,5"
        s = self.bounds_var.get().strip()
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError("Неверный формат границ. Пример: -5,5")
        lo = float(parts[0].strip())
        hi = float(parts[1].strip())
        return (lo, hi)

    def on_run(self):
        self.run_btn.configure(state="disabled")
        self.set_result("Запуск... считаю. Это может занять немного времени.\n")
        self.ax_history.clear()
        self.ax_history.grid(True, alpha=0.3)
        self.canvas.draw_idle()

        algo = self.algo_var.get()
        try:
            bounds = self.parse_bounds()
            n = int(self.n_var.get())
            generations = int(self.gen_var.get())
        except Exception as e:
            self.run_btn.configure(state="normal")
            messagebox.showerror("Ошибка ввода", str(e))
            return

        def worker():
            try:
                Genetic_algo, ParticleSwarm = get_algorithms()
                if algo == "GA":
                    # GA использует pop_size и generations
                    ga = Genetic_algo(
                        population_size=n,
                        generations=generations,
                        answer_bounds=bounds,
                        elite_ratio=0.5,
                        mutation_rate=0.2,
                    )
                    best, best_f, history, calls = ga.run()

                    positions = None
                else:
                    # PSO использует n_particles и generations
                    pso = ParticleSwarm(
                        bounds=bounds,
                        n_particles=max(2, n // 1),
                        generations=generations,
                        w=0.7,
                        c1=1.5,
                        c2=1.5,
                        Vmax=1.0,
                    )
                    best, best_f, history, calls = pso.run()

                    positions = pso.positions.copy()

                result = {
                    "algo": algo,
                    "best": best,
                    "best_f": best_f,
                    "history": history,
                    "calls": calls,
                    "bounds": bounds,
                    "positions": positions,
                }
                self.after(0, lambda: self.on_finished(result))
            except Exception as e:
                err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                self.after(0, lambda: self.on_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def on_error(self, err: str):
        self.run_btn.configure(state="normal")
        self.set_result("Ошибка:\n" + err[-4000:])
        messagebox.showerror("Ошибка", "Случилась ошибка при запуске. См. текст внизу окна.")

    def on_finished(self, result: dict):
        self.run_btn.configure(state="normal")

        algo = result["algo"]
        best = result["best"]
        best_f = result["best_f"]
        history = result["history"]
        calls = result["calls"]
        bounds = result.get("bounds", (-5.0, 5.0))
        positions = result.get("positions")

        a = abs(KNOWN_MINIMA[0][0])  # точки минимума имеют вид (+-a, +-a)

        self.set_result(
            f"Алгоритм: {algo}\n"
            f"Лучшее решение: x = {best[0]:.6f}, y = {best[1]:.6f}\n"
            f"Значение функции: f = {best_f:.6f}\n"
            f"Вызовов функции: {calls}\n"
            f"Точки минимума: +/- {a:.5f}, +/- {a:.5f}\n"
            f"Ожидаемый результат: -2.06261\n"
        )

        # 1) График сходимости
        self.ax_history.clear()

        self.ax_history.grid(True, alpha=0.3)
        self.ax_history.set_title(f"Сходимость ({algo})")
        self.ax_history.set_xlabel("Шаг / поколение")
        self.ax_history.set_ylabel("min f")

        self.ax_history.plot(history, linewidth=2)
        self.ax_history.axhline(
            y=-2.06261,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Ожидаемый минимум",
        )
        self.ax_history.legend()

        self.canvas.draw_idle()


def main():
    app = SearchGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

