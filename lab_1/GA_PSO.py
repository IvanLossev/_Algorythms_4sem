"""
Лабораторная: вариант 9 (функция f(x,y)), ГА с модификацией «кроссовер»,
РА с модификацией «ограничение скорости».

Один файл, несколько классов. Модификации включаются полем use_modifications
(ветвление внутри методов), а не отдельным классом и не дублированием run().
"""

from __future__ import annotations

import math
import random
import threading
import traceback
import tkinter as tk
from contextlib import suppress
from tkinter import messagebox, ttk

import matplotlib
import numpy as np

with suppress(Exception):
    matplotlib.use("TkAgg", force=True)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def function_var9(x: float, y: float) -> float:
    return -0.0001 * (
        abs(
            math.sin(x)
            * math.sin(y)
            * math.exp(abs(100 - (math.sqrt(x**2 + y**2) / math.pi)))
        )
        + 1
    ) ** 0.1


KNOWN_MINIMA = [
    (1.34941, -1.34941),
    (1.34941, 1.34941),
    (-1.34941, 1.34941),
    (-1.34941, -1.34941),
]


class GeneticAlgorithmReal:

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 200,
        answer_bounds: tuple[float, float] = (-5, 5),
        elite_ratio: float = 0.5,
        mutation_rate: float = 0.1,
        use_modifications: bool = True,
    ):
        self.pop_size = population_size
        self.generations = generations
        self.bounds = answer_bounds
        self.remains = elite_ratio
        self.mutation_rate = mutation_rate
        self.use_modifications = use_modifications
        self.func_calls = 0

    def evaluate(self, x: float, y: float) -> float:
        self.func_calls += 1
        return function_var9(x, y)

    def init_population(self) -> list[list[float]]:
        lo, hi = self.bounds
        return [[random.uniform(lo, hi), random.uniform(lo, hi)] for _ in range(self.pop_size)]

    def crossover(self, parent1: list[float], parent2: list[float]) -> tuple[list[float], list[float]]:
        x1, y1 = parent1
        x2, y2 = parent2
        if self.use_modifications:
            alpha = random.choice([0.3, 0.4, 0.5])
            c1 = [alpha * x1 + (1 - alpha) * x2, alpha * y1 + (1 - alpha) * y2]
            c2 = [alpha * x2 + (1 - alpha) * x1, alpha * y2 + (1 - alpha) * y1]
            return c1, c2
        return [x1, y1], [x2, y2]

    def mutate(self, individual: list[float]) -> list[float]:
        """Мутация: с вероятностью mutation_rate изменяем координаты"""
        if random.random() < self.mutation_rate:
            x, y = individual
            lo, hi = self.bounds
            sigma = 0.1 * (hi - lo)
            x += random.gauss(0, sigma)
            y += random.gauss(0, sigma)
            x = max(lo, min(hi, x))
            y = max(lo, min(hi, y))
            return [x, y]
        return individual.copy()

    def run(self) -> tuple[list[float] | None, float, list[float], int]:
        population = self.init_population()
        best_result = float("inf")
        best_vals: list[float] | None = None
        history: list[float] = []
        lo, hi = self.bounds

        for gen in range(self.generations):
            fitness = [self.evaluate(ind[0], ind[1]) for ind in population]
            sorted_indices = np.argsort(fitness)
            sorted_pop = [population[i] for i in sorted_indices]
            sorted_fit = [fitness[i] for i in sorted_indices]

            best_f = sorted_fit[0]
            best_ind = sorted_pop[0]

            if best_f < best_result:
                best_result = best_f
                best_vals = list(best_ind)

            history.append(best_f)

            elite_size = int(self.pop_size * self.remains)
            elites = sorted_pop[:elite_size]
            parents = sorted_pop[:elite_size]
            random.shuffle(parents)

            children: list[list[float]] = []
            for i in range(0, len(parents) - 1, 2):
                if len(children) < self.pop_size - elite_size:
                    c1, c2 = self.crossover(parents[i], parents[i + 1])
                    c1 = self.mutate(c1)
                    c2 = self.mutate(c2)
                    children.append(c1)
                if len(children) < self.pop_size - elite_size:
                    children.append(c2)

            while len(children) < self.pop_size - elite_size:
                children.append([random.uniform(lo, hi), random.uniform(lo, hi)])

            population = elites + children

            if (gen + 1) % 20 == 0:
                print(
                    f"Поколение {gen+1}: лучший f = {best_f:.6f} в ({best_ind[0]:.5f}, {best_ind[1]:.5f})"
                )

        return best_vals, best_result, history, self.func_calls


class GeneticAlgorithmBinary:
    # Хромосома: 2 * bits_per_var бит (x, затем y)

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 200,
        answer_bounds: tuple[float, float] = (-5, 5),
        elite_ratio: float = 0.5,
        mutation_rate: float = 0.02,
        bits_per_var: int = 16,
        use_modifications: bool = True,
    ):
        self.pop_size = population_size
        self.generations = generations
        self.bounds = answer_bounds
        self.remains = elite_ratio
        self.mutation_rate = mutation_rate
        self.bits_per_var = bits_per_var
        self.chromo_len = 2 * bits_per_var
        self.use_modifications = use_modifications
        self.func_calls = 0

    def evaluate(self, x: float, y: float) -> float:
        self.func_calls += 1
        return function_var9(x, y)

    def _decode_axis(self, bits: list[int]) -> float:
        lo, hi = self.bounds
        n = len(bits)
        if n == 0:
            return lo
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        denom = (1 << n) - 1
        return lo + (val / denom) * (hi - lo) if denom else lo

    def decode(self, chromosome: list[int]) -> tuple[float, float]:
        h = self.bits_per_var
        x = self._decode_axis(chromosome[:h])
        y = self._decode_axis(chromosome[h : h + h])
        return x, y

    def random_chromosome(self) -> list[int]:
        return [random.randint(0, 1) for _ in range(self.chromo_len)]

    def init_population(self) -> list[list[int]]:
        return [self.random_chromosome() for _ in range(self.pop_size)]

    def mutate(self, chromo: list[int]) -> None:
        for i in range(len(chromo)):
            if random.random() < self.mutation_rate:
                chromo[i] ^= 1

    def crossover(
        self, p1: list[int], p2: list[int]
    ) -> tuple[list[int], list[int]]:
        if not self.use_modifications:
            return p1[:], p2[:]
        if self.chromo_len < 2:
            return p1[:], p2[:]
        pt = random.randint(1, self.chromo_len - 1)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
        return c1, c2

    def run(self) -> tuple[list[float] | None, float, list[float], int]:
        population = self.init_population()
        best_result = float("inf")
        best_vals: list[float] | None = None
        history: list[float] = []

        for gen in range(self.generations):
            phenotypes = [self.decode(ch) for ch in population]
            fitness = [self.evaluate(x, y) for x, y in phenotypes]
            sorted_indices = np.argsort(fitness)
            sorted_pop = [population[i] for i in sorted_indices]
            sorted_fit = [fitness[i] for i in sorted_indices]

            best_f = sorted_fit[0]
            best_ind = sorted_pop[0]
            bx, by = self.decode(best_ind)

            if best_f < best_result:
                best_result = best_f
                best_vals = [bx, by]

            history.append(best_f)

            elite_size = int(self.pop_size * self.remains)
            elites = [c[:] for c in sorted_pop[:elite_size]]
            parents = [c[:] for c in sorted_pop[:elite_size]]
            random.shuffle(parents)

            children: list[list[int]] = []
            for i in range(0, len(parents) - 1, 2):
                if len(children) < self.pop_size - elite_size:
                    c1, c2 = self.crossover(parents[i], parents[i + 1])
                    self.mutate(c1)
                    self.mutate(c2)
                    children.append(c1)
                if len(children) < self.pop_size - elite_size:
                    children.append(c2)

            while len(children) < self.pop_size - elite_size:
                ch = self.random_chromosome()
                self.mutate(ch)
                children.append(ch)

            population = elites + children

            if (gen + 1) % 20 == 0:
                print(f"Поколение {gen+1}: лучший f = {best_f:.6f} в ({bx:.5f}, {by:.5f})")

        return best_vals, best_result, history, self.func_calls


class ParticleSwarmReal:

    def __init__(
        self,
        bounds: tuple[float, float] = (-5, 5),
        n_particles: int = 30,
        generations: int = 200,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        Vmax: float = 1.0,
        use_modifications: bool = True,
    ):
        self.bounds = bounds
        self.dim = 2
        self.n_particles = n_particles
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.Vmax = Vmax
        self.use_modifications = use_modifications
        self.func_calls = 0

        self.positions: np.ndarray | None = None
        self.velocities: np.ndarray | None = None
        self.best_positions: np.ndarray | None = None
        self.best_coords: np.ndarray | None = None
        self.global_best: np.ndarray | None = None
        self.global_best_coords = float("inf")

    def evaluate(self, x: float, y: float) -> float:
        self.func_calls += 1
        return function_var9(x, y)

    def init_population(self) -> None:
        lo, hi = self.bounds
        self.positions = np.array(
            [[random.uniform(lo, hi) for _ in range(self.dim)] for _ in range(self.n_particles)]
        )
        self.velocities = np.zeros((self.n_particles, self.dim))
        self.best_positions = self.positions.copy()
        self.best_coords = np.array([self.evaluate(p[0], p[1]) for p in self.positions])
        best_idx = int(np.argmin(self.best_coords))
        self.global_best = self.best_positions[best_idx].copy()
        self.global_best_coords = float(self.best_coords[best_idx])

    def update_velocity(self, i: int) -> None:
        assert self.positions is not None
        assert self.velocities is not None
        assert self.best_positions is not None
        assert self.global_best is not None

        r1 = random.random()
        r2 = random.random()
        cognitive = self.c1 * r1 * (self.best_positions[i] - self.positions[i])
        social = self.c2 * r2 * (self.global_best - self.positions[i])
        self.velocities[i] = self.w * self.velocities[i] + cognitive + social

        if self.use_modifications:
            for j in range(self.dim):
                if self.velocities[i][j] > self.Vmax:
                    self.velocities[i][j] = self.Vmax
                elif self.velocities[i][j] < -self.Vmax:
                    self.velocities[i][j] = -self.Vmax

    def update_position(self, i: int) -> None:
        assert self.positions is not None
        assert self.velocities is not None
        self.positions[i] += self.velocities[i]
        lo, hi = self.bounds
        for j in range(self.dim):
            if self.positions[i][j] < lo:
                self.positions[i][j] = lo
            elif self.positions[i][j] > hi:
                self.positions[i][j] = hi

    def run(self) -> tuple[np.ndarray, float, list[float], int]:
        self.init_population()
        assert self.global_best is not None
        history = [self.global_best_coords]

        print("Начало работы роевого алгоритма...")
        print(
            f"Начальный gbest: {self.global_best_coords:.6f} в точке ({self.global_best[0]:.5f}, {self.global_best[1]:.5f})"
        )

        for gen in range(self.generations):
            for i in range(self.n_particles):
                assert self.positions is not None
                fitness = self.evaluate(self.positions[i][0], self.positions[i][1])
                assert self.best_coords is not None
                assert self.best_positions is not None

                if fitness < self.best_coords[i]:
                    self.best_coords[i] = fitness
                    self.best_positions[i] = self.positions[i].copy()

                if fitness < self.global_best_coords:
                    self.global_best_coords = fitness
                    self.global_best = self.positions[i].copy()

                self.update_velocity(i)
                self.update_position(i)

            history.append(self.global_best_coords)

            if (gen + 1) % 20 == 0:
                assert self.global_best is not None
                print(
                    f"Итерация {gen+1}: лучший f = {self.global_best_coords:.6f} в ({self.global_best[0]:.5f}, {self.global_best[1]:.5f})"
                )

        assert self.global_best is not None
        return self.global_best, self.global_best_coords, history, self.func_calls


class SearchGUI(tk.Tk):
    ALGO_GA_REAL = "ГА (вещественный)"
    ALGO_GA_BIN = "ГА (двоичный)"
    ALGO_PSO = "PSO (вещественный)"

    def __init__(self) -> None:
        super().__init__()
        self.title("Вариант 9: ГА / РА")
        self.geometry("920x580")

        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Label(ctrl, text="Алгоритм:").pack(side=tk.LEFT)
        self.algo_var = tk.StringVar(value=self.ALGO_GA_REAL)

        # ИСПРАВЛЕНО: убираем command, используем bind
        combo = ttk.Combobox(
            ctrl,
            textvariable=self.algo_var,
            values=[self.ALGO_GA_REAL, self.ALGO_GA_BIN, self.ALGO_PSO],
            state="readonly",
            width=28,
        )
        combo.bind("<<ComboboxSelected>>", self.on_algo_change)
        combo.pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(ctrl, text="Границы (min,max):").pack(side=tk.LEFT)
        self.bounds_var = tk.StringVar(value="-5,5")
        ttk.Entry(ctrl, textvariable=self.bounds_var, width=10).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="Популяция / частицы:").pack(side=tk.LEFT)
        self.n_var = tk.StringVar(value="100")
        ttk.Entry(ctrl, textvariable=self.n_var, width=5).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="Поколения / итерации:").pack(side=tk.LEFT)
        self.gen_var = tk.StringVar(value="200")
        ttk.Entry(ctrl, textvariable=self.gen_var, width=5).pack(side=tk.LEFT, padx=(4, 12))

        self.w_frame = ttk.Frame(ctrl)
        ttk.Label(self.w_frame, text="Коэффициент инерции (w):").pack(side=tk.LEFT)
        self.w_var = tk.StringVar(value="0.7")
        ttk.Entry(self.w_frame, textvariable=self.w_var, width=5).pack(side=tk.LEFT)
        # По умолчанию скрыто, так как начальный алгоритм - ГА
        self.w_frame.pack_forget()

        self.run_btn = ttk.Button(ctrl, text="Запустить", command=self.on_run)
        self.run_btn.pack(side=tk.RIGHT)

        mod_row = ttk.Frame(self)
        mod_row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 4))
        self.use_mod_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            mod_row,
            text="Использовать модификации (ГА: кроссовер; РА: ограничение скорости)",
            variable=self.use_mod_var,
        ).pack(side=tk.LEFT)

        main = ttk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.result_text = tk.Text(main, height=7, wrap="word")
        self.result_text.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self.result_text.configure(state="disabled")

        fig = Figure(figsize=(9, 4.2), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Сходимость")
        self.ax.set_xlabel("Поколение / итерация")
        self.ax.set_ylabel("лучший f")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(fig, master=main)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def on_algo_change(self, event=None) -> None:
        if self.algo_var.get() == self.ALGO_PSO:
            self.w_frame.pack(side=tk.LEFT, padx=(4, 12))
        else:
            self.w_frame.pack_forget()

    def _set_result(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state="disabled")

    def _parse_bounds(self) -> tuple[float, float]:
        parts = self.bounds_var.get().strip().split(",")
        if len(parts) != 2:
            raise ValueError("Границы: формат min,max например -5,5")
        return float(parts[0].strip()), float(parts[1].strip())

    def on_run(self) -> None:
        self.run_btn.configure(state="disabled")
        self._set_result("Считаю…\n")
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

        use_mod = self.use_mod_var.get()
        algo = self.algo_var.get()
        try:
            bounds = self._parse_bounds()
            n = int(self.n_var.get())
            gens = int(self.gen_var.get())
            w = float(self.w_var.get()) if algo == self.ALGO_PSO else 0.7
        except Exception as e:
            self.run_btn.configure(state="normal")
            messagebox.showerror("Ввод", str(e))
            return

        def worker() -> None:
            try:
                if algo == self.ALGO_GA_REAL:
                    ga = GeneticAlgorithmReal(
                        population_size=n,
                        generations=gens,
                        answer_bounds=bounds,
                        elite_ratio=0.5,
                        mutation_rate=0.2,
                        use_modifications=use_mod,
                    )
                    best, best_f, history, calls = ga.run()
                elif algo == self.ALGO_GA_BIN:
                    gb = GeneticAlgorithmBinary(
                        population_size=n,
                        generations=gens,
                        answer_bounds=bounds,
                        elite_ratio=0.5,
                        mutation_rate=0.02,
                        bits_per_var=16,
                        use_modifications=use_mod,
                    )
                    best, best_f, history, calls = gb.run()
                else:
                    pso = ParticleSwarmReal(
                        bounds=bounds,
                        n_particles=max(2, n),
                        generations=gens,
                        w=w,
                        c1=1.5,
                        c2=1.5,
                        Vmax=1.0,
                        use_modifications=use_mod,
                    )
                    best, best_f, history, calls = pso.run()
                    best = [float(best[0]), float(best[1])]

                a = abs(KNOWN_MINIMA[0][0])
                assert best is not None
                text = (
                    f"Режим: {algo}\n"
                    f"Модификации: {'да' if use_mod else 'нет'}\n"
                    f"Лучшее решение: x = {best[0]:.6f}, y = {best[1]:.6f}\n"
                    f"Значение функции: f = {best_f:.6f}\n"
                    f"Вызовов функции: {calls}\n"
                    f"Точки минимума: +/- {a:.5f}, +/- {a:.5f}\n"
                    f"Ожидаемый результат: -2.06261\n"
                )
                self.after(
                    0,
                    lambda: self._on_done(text, history, algo, use_mod),
                )
            except Exception as e:
                err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                self.after(0, lambda: self._on_err(err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_err(self, err: str) -> None:
        self.run_btn.configure(state="normal")
        self._set_result("Ошибка:\n" + err[-3500:])
        messagebox.showerror("Ошибка", "См. текст в окне.")

    def _on_done(self, text: str, history: list[float], algo: str, use_mod: bool) -> None:
        self.run_btn.configure(state="normal")
        self._set_result(text)
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        tag = "модиф." if use_mod else "без модиф."
        self.ax.set_title(f"Сходимость: {algo} ({tag})")
        self.ax.plot(history, linewidth=2)
        self.ax.axhline(y=-2.06261, color="red", linestyle="--", alpha=0.7, label="Ожидаемый минимум")
        self.ax.legend()
        self.canvas.draw_idle()


def main() -> None:
    app = SearchGUI()
    app.mainloop()


# Алиасы для совместимости со старыми именами классов
Genetic_algo = GeneticAlgorithmReal
ParticleSwarm = ParticleSwarmReal


if __name__ == "__main__":
    main()