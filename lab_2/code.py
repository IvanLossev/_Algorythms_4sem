import math
import os
import random
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


def build_small_directed_graph() -> tuple[list[str], dict[tuple[int, int], float]]:
    '''
    Построение малого графа из рис. 1
    Вершины: a=0, b=1, c=2, d=3, f=4, g=5
    '''
    names = ["a", "b", "c", "d", "f", "g"]
    idx = {n: i for i, n in enumerate(names)}
    edges_raw = [
        ("a", "b", 3), ("a", "f", 1),
        ("b", "a", 3), ("b", "c", 8), ("b", "g", 3),
        ("c", "b", 3), ("c", "d", 1), ("c", "g", 1),
        ("d", "c", 8), ("d", "f", 1),
        ("f", "a", 3), ("f", "d", 3),
        ("g", "a", 3), ("g", "b", 3), ("g", "c", 3), ("g", "d", 5), ("g", "f", 4),
    ]
    edges: dict[tuple[int, int], float] = {}
    for u, v, w in edges_raw:
        edges[(idx[u], idx[v])] = w
    return names, edges


def build_distance_matrix_from_edges(n: int, edges: dict[tuple[int, int], float], directed: bool = False) -> np.ndarray:
    '''матрица расстояний n×n.'''
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    for (i, j), w in edges.items():
        dist[i][j] = w
        if not directed:
            dist[j][i] = w
    return dist


def parse_stp_file(filepath: str) -> tuple[int, np.ndarray]:
    """Парсит STP-файл. Возвращает (число вершин, матрица расстояний)."""
    n = 0
    edges: dict[tuple[int, int], float] = {}
    in_graph = False
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Section Graph"):
                in_graph = True
                continue
            if line == "End" and in_graph:
                in_graph = False
                continue
            if in_graph:
                if line.startswith("Nodes"):
                    n = int(line.split()[1])
                elif line.startswith("E "):
                    parts = line.split()
                    u, v, w = int(parts[1]) - 1, int(parts[2]) - 1, float(parts[3])
                    edges[(u, v)] = w
    dist = build_distance_matrix_from_edges(n, edges, directed=False)
    return n, dist


def route_cost(route: list[int], dist: np.ndarray) -> float:
    '''Стоимость гамильтонова цикла.'''
    total = 0.0
    for i in range(len(route)):
        c = dist[route[i]][route[(i + 1) % len(route)]]
        if c == np.inf:
            return np.inf
        total += c
    return total


def nearest_neighbor_route(dist: np.ndarray, start: int = 0) -> list[int]:
    '''Жадный алгоритм ближайшего соседа.'''
    n = dist.shape[0]
    visited = [False] * n
    route = [start]
    visited[start] = True
    for _ in range(n - 1):
        cur = route[-1]
        best_next = -1
        best_dist = np.inf
        for j in range(n):
            if not visited[j] and dist[cur][j] < best_dist:
                best_dist = dist[cur][j]
                best_next = j
        if best_next == -1:
            for j in range(n):
                if not visited[j]:
                    best_next = j
                    break
        route.append(best_next)
        visited[best_next] = True
    return route


def format_route(route: list[int], names: Optional[list[str]] = None) -> str:
    '''Форматирование маршрута.'''
    if names:
        labels = [names[i] for i in route]
    else:
        labels = [str(i + 1) for i in route]
    if len(labels) > 20:
        return " → ".join(labels[:10]) + " → ... → " + " → ".join(labels[-5:]) + f" ({len(labels)} вершин)"
    return " → ".join(labels) + " → " + labels[0]


class SimulatedAnnealing:
    '''
    use_modification=False: базовый(Больцмановский) отжиг T(k) = T0 / ln(1+k)
    use_modification=True:  Отжиг Коши T(k) = T0 / (1+k) + генерация Коши
    '''

    def __init__(
        self,
        dist: np.ndarray,
        T0: float = 1000.0,
        T_min: float = 1e-3,
        max_iter: int = 100_000,
        use_modification: bool = True,
    ):
        self.dist = dist
        self.n = dist.shape[0]
        self.T0 = T0
        self.T_min = T_min
        self.max_iter = max_iter
        self.use_modification = use_modification
        self.func_calls = 0

    def _temperature(self, k: int) -> float:
        if self.use_modification:
            return self.T0 / (1 + k)
        else:
            return self.T0 / math.log(2 + k)

    def _neighbor(self, route: list[int]) -> list[int]:
        new_route = route[:]
        n = len(new_route)
        if self.use_modification:
            i = int(abs(np.random.standard_cauchy())) % n
            j = int(abs(np.random.standard_cauchy())) % n
            while i == j:
                j = int(abs(np.random.standard_cauchy())) % n
        else:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while i == j:
                j = random.randint(0, n - 1)
        if i > j:
            i, j = j, i
        new_route[i:j + 1] = reversed(new_route[i:j + 1])
        return new_route

    def _eval(self, route: list[int]) -> float:
        self.func_calls += 1
        return route_cost(route, self.dist)

    def run(self, callback=None) -> tuple[list[int], float, list[float], int, float]:
        '''
        callback(iteration, best_cost) — вызывается периодически для обновления GUI.
        '''
        start_time = time.time()
        self.func_calls = 0

        current = nearest_neighbor_route(self.dist, start=0)
        current_cost = self._eval(current)
        best = current[:]
        best_cost = current_cost
        history: list[float] = [best_cost]

        for k in range(1, self.max_iter + 1):
            T = self._temperature(k)
            if T < self.T_min:
                break

            candidate = self._neighbor(current)
            candidate_cost = self._eval(candidate)
            delta = candidate_cost - current_cost

            if delta < 0:
                current = candidate
                current_cost = candidate_cost
            else:
                if T > 0 and random.random() < math.exp(-delta / T):
                    current = candidate
                    current_cost = candidate_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

            step = max(1, self.max_iter // 500)
            if k % step == 0:
                history.append(best_cost)
                if callback and k % (step * 10) == 0:
                    callback(k, best_cost)

        elapsed = time.time() - start_time
        return best, best_cost, history, self.func_calls, elapsed


class AntColonyOptimization:
    '''
    use_modification=False: без модификации
    use_modification=True:  начальное расположение
    '''

    def __init__(
        self,
        dist: np.ndarray,
        n_ants: int = 30,
        n_iterations: int = 200,
        alpha: float = 1.0,
        beta: float = 3.0,
        rho: float = 0.5,
        Q: float = 100.0,
        use_modification: bool = True,
    ):
        self.dist = dist
        self.n = dist.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.use_modification = use_modification
        self.func_calls = 0

    def _initial_starts(self) -> list[int]:
        if self.use_modification:
            scores = []
            for i in range(self.n):
                reachable = [self.dist[i][j] for j in range(self.n)
                            if j != i and self.dist[i][j] < np.inf]
                if reachable:
                    scores.append((i, np.mean(reachable)))
                else:
                    scores.append((i, np.inf))
            scores.sort(key=lambda x: x[1])
            starts = []
            n_strategic = self.n_ants // 2
            top_vertices = [s[0] for s in scores[:max(1, self.n // 4)]]
            for k in range(n_strategic):
                starts.append(top_vertices[k % len(top_vertices)])
            for _ in range(self.n_ants - n_strategic):
                starts.append(random.randint(0, self.n - 1))
            return starts
        else:
            start_node = random.randint(0, self.n - 1)
            return [start_node] * self.n_ants

    def _eval(self, route: list[int]) -> float:
        self.func_calls += 1
        return route_cost(route, self.dist)

    def _construct_route(
        self, start: int, pheromone: np.ndarray, heuristic: np.ndarray
    ) -> list[int]:
        n = self.n
        visited = np.zeros(n, dtype=bool)
        route = [start]
        visited[start] = True
        pher_alpha = pheromone ** self.alpha if self.alpha != 1.0 else pheromone
        heur_beta = heuristic ** self.beta

        for _ in range(n - 1):
            cur = route[-1]
            mask = ~visited & (self.dist[cur] < np.inf)
            if not mask.any():
                unvisited = np.where(~visited)[0]
                if len(unvisited) > 0:
                    nxt = int(np.random.choice(unvisited))
                else:
                    break
            else:
                probs = np.where(mask, pher_alpha[cur] * heur_beta[cur], 0.0)
                total = probs.sum()
                if total == 0:
                    candidates = np.where(mask)[0]
                    nxt = int(np.random.choice(candidates))
                else:
                    probs /= total
                    nxt = int(np.random.choice(n, p=probs))
            route.append(nxt)
            visited[nxt] = True
        return route

    def run(self, callback=None) -> tuple[list[int], float, list[float], int, float]:
        start_time = time.time()
        self.func_calls = 0
        n = self.n

        tau0 = 1.0 / n
        pheromone = np.full((n, n), tau0)

        heuristic = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and self.dist[i][j] < np.inf and self.dist[i][j] > 0:
                    heuristic[i][j] = 1.0 / self.dist[i][j]

        best_route: list[int] = []
        best_cost = np.inf
        history: list[float] = []

        for iteration in range(self.n_iterations):
            starts = self._initial_starts()
            routes: list[list[int]] = []
            costs: list[float] = []

            for ant_idx in range(self.n_ants):
                route = self._construct_route(starts[ant_idx], pheromone, heuristic)
                cost = self._eval(route)
                routes.append(route)
                costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_route = route[:]

            pheromone *= (1 - self.rho)
            for ant_idx in range(self.n_ants):
                if costs[ant_idx] < np.inf:
                    deposit = self.Q / costs[ant_idx]
                    r = routes[ant_idx]
                    for k in range(len(r)):
                        i_node = r[k]
                        j_node = r[(k + 1) % len(r)]
                        pheromone[i_node][j_node] += deposit

            history.append(best_cost)
            if callback:
                callback(iteration + 1, best_cost)

        elapsed = time.time() - start_time
        return best_route, best_cost, history, self.func_calls, elapsed


BERLIN52_PATH = "D:/_Algorythms_4sem/lab_2/berlin52.stp"
WORLD666_PATH = "D:/_Algorythms_4sem/lab_2/world666.stp"


def load_graph(graph_name: str) -> tuple[int, np.ndarray, Optional[list[str]]]:
    """Загрузка графа по имени. Возвращает (n, dist, node_names)."""
    if graph_name == "Малый орграф (6 вершин)":
        names, edges = build_small_directed_graph()
        n = len(names)
        dist = build_distance_matrix_from_edges(n, edges, directed=True)
        return n, dist, names
    elif graph_name == "berlin52 (52 вершины)":
        n, dist = parse_stp_file(BERLIN52_PATH)
        return n, dist, None
    elif graph_name == "world666 (666 вершин)":
        n, dist = parse_stp_file(WORLD666_PATH)
        return n, dist, None
    else:
        raise ValueError(f"Неизвестный граф: {graph_name}")


DEFAULT_PARAMS = {
    "Малый орграф (6 вершин)": {
        "sa": {"T0": 100, "T_min": 0.01, "max_iter": 10_000},
        "aco": {"n_ants": 10, "n_iterations": 100, "alpha": 1.0, "beta": 3.0, "rho": 0.5, "Q": 50},
    },
    "berlin52 (52 вершины)": {
        "sa": {"T0": 5000, "T_min": 0.001, "max_iter": 200_000},
        "aco": {"n_ants": 40, "n_iterations": 200, "alpha": 1.0, "beta": 3.0, "rho": 0.5, "Q": 500},
    },
    "world666 (666 вершин)": {
        "sa": {"T0": 50000, "T_min": 0.1, "max_iter": 300_000},
        "aco": {"n_ants": 20, "n_iterations": 30, "alpha": 1.0, "beta": 5.0, "rho": 0.3, "Q": 5000},
    },
}


ALGORITHM_CHOICES = [
    "SA без модификации",
    "SA с модификацией (Отжиг Коши)",
    "ACO без модификации",
    "ACO с модификацией (Начальное расположение)",
]

# Маппинг: алгоритм → (класс, use_modification, тип параметров)
ALGO_MAP = {
    ALGORITHM_CHOICES[0]: ("SA", False),
    ALGORITHM_CHOICES[1]: ("SA", True),
    ALGORITHM_CHOICES[2]: ("ACO", False),
    ALGORITHM_CHOICES[3]: ("ACO", True),
}


class TSPApp:
    '''Графический интерфейс для решения TSP.'''

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Лабораторная работа №2")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)

        self._running = False

        self._build_ui()

    def _build_ui(self):
        top_frame = ttk.LabelFrame(self.root, text="Настройки", padding=10)
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(top_frame, text="Алгоритм:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.algo_var = tk.StringVar(value=ALGORITHM_CHOICES[0])
        self.algo_combo = ttk.Combobox(top_frame, textvariable=self.algo_var, values=ALGORITHM_CHOICES, state="readonly", width=48)
        self.algo_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algo_changed)

        graph_names = list(DEFAULT_PARAMS.keys())
        ttk.Label(top_frame, text="Граф:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.graph_var = tk.StringVar(value=graph_names[0])
        self.graph_combo = ttk.Combobox(
            top_frame, textvariable=self.graph_var,
            values=graph_names, state="readonly", width=30
        )
        self.graph_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        self.graph_combo.bind("<<ComboboxSelected>>", self._on_graph_changed)

        # Кнопка запуска
        self.run_btn = ttk.Button(top_frame, text="▶ Запустить", command=self._on_run)
        self.run_btn.grid(row=0, column=4, padx=(10, 0))

        # ─── Панель параметров ───
        params_frame = ttk.LabelFrame(self.root, text="Параметры алгоритма", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        # SA параметры
        self.sa_frame = ttk.Frame(params_frame)
        self.sa_frame.pack(fill=tk.X)

        ttk.Label(self.sa_frame, text="T₀:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.t0_var = tk.StringVar(value="100")
        ttk.Entry(self.sa_frame, textvariable=self.t0_var, width=12).grid(row=0, column=1, padx=(0, 15))

        ttk.Label(self.sa_frame, text="T_min:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.tmin_var = tk.StringVar(value="0.01")
        ttk.Entry(self.sa_frame, textvariable=self.tmin_var, width=12).grid(row=0, column=3, padx=(0, 15))

        ttk.Label(self.sa_frame, text="Макс. итераций:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.sa_maxiter_var = tk.StringVar(value="10000")
        ttk.Entry(self.sa_frame, textvariable=self.sa_maxiter_var, width=12).grid(row=0, column=5, padx=(0, 15))

        # ACO параметры
        self.aco_frame = ttk.Frame(params_frame)
        self.aco_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(self.aco_frame, text="Муравьёв:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.nants_var = tk.StringVar(value="10")
        ttk.Entry(self.aco_frame, textvariable=self.nants_var, width=8).grid(row=0, column=1, padx=(0, 10))

        ttk.Label(self.aco_frame, text="Итераций:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.aco_iter_var = tk.StringVar(value="100")
        ttk.Entry(self.aco_frame, textvariable=self.aco_iter_var, width=8).grid(row=0, column=3, padx=(0, 10))

        ttk.Label(self.aco_frame, text="α:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.alpha_var = tk.StringVar(value="1.0")
        ttk.Entry(self.aco_frame, textvariable=self.alpha_var, width=6).grid(row=0, column=5, padx=(0, 10))

        ttk.Label(self.aco_frame, text="β:").grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        self.beta_var = tk.StringVar(value="3.0")
        ttk.Entry(self.aco_frame, textvariable=self.beta_var, width=6).grid(row=0, column=7, padx=(0, 10))

        ttk.Label(self.aco_frame, text="ρ:").grid(row=0, column=8, sticky=tk.W, padx=(0, 5))
        self.rho_var = tk.StringVar(value="0.5")
        ttk.Entry(self.aco_frame, textvariable=self.rho_var, width=6).grid(row=0, column=9, padx=(0, 10))

        ttk.Label(self.aco_frame, text="Q:").grid(row=0, column=10, sticky=tk.W, padx=(0, 5))
        self.q_var = tk.StringVar(value="50")
        ttk.Entry(self.aco_frame, textvariable=self.q_var, width=8).grid(row=0, column=11, padx=(0, 10))

        # ─── Средняя часть: лог + график ───
        middle_frame = ttk.Frame(self.root)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Лог
        log_frame = ttk.LabelFrame(middle_frame, text="Результаты", padding=5)
        log_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # График
        chart_frame = ttk.LabelFrame(middle_frame, text="График сходимости", padding=5)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.fig, self.ax = plt.subplots(figsize=(5, 3.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ─── Статусная строка ───
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Инициализация параметров
        self._on_algo_changed()
        self._on_graph_changed()

    def _on_algo_changed(self, event=None):
        '''Показать/скрыть параметры в зависимости от выбранного алгоритма.'''
        algo_name = self.algo_var.get()
        algo_type, _ = ALGO_MAP[algo_name]

        if algo_type == "SA":
            for child in self.sa_frame.winfo_children():
                child.configure(state="normal") if hasattr(child, 'configure') else None
            for child in self.aco_frame.winfo_children():
                if isinstance(child, ttk.Entry):
                    child.configure(state="disabled")
        else:
            for child in self.sa_frame.winfo_children():
                if isinstance(child, ttk.Entry):
                    child.configure(state="disabled")
            for child in self.aco_frame.winfo_children():
                child.configure(state="normal") if hasattr(child, 'configure') else None

        # Обновить значения параметров по умолчанию
        self._update_default_params()

    def _on_graph_changed(self, event=None):
        # Обновить параметры по умолчанию при смене графа.
        self._update_default_params()

    def _update_default_params(self):
        # Заполнить поля параметров значениями по умолчанию.
        graph_name = self.graph_var.get()
        if graph_name not in DEFAULT_PARAMS:
            return
        params = DEFAULT_PARAMS[graph_name]

        # SA
        sa = params["sa"]
        self.t0_var.set(str(sa["T0"]))
        self.tmin_var.set(str(sa["T_min"]))
        self.sa_maxiter_var.set(str(sa["max_iter"]))

        # ACO
        aco = params["aco"]
        self.nants_var.set(str(aco["n_ants"]))
        self.aco_iter_var.set(str(aco["n_iterations"]))
        self.alpha_var.set(str(aco["alpha"]))
        self.beta_var.set(str(aco["beta"]))
        self.rho_var.set(str(aco["rho"]))
        self.q_var.set(str(aco["Q"]))

    def _log(self, text: str):
        """Добавить текст в лог (потокобезопасно)."""
        self.root.after(0, self._log_impl, text)

    def _log_impl(self, text: str):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def _set_status(self, text: str):
        self.root.after(0, lambda: self.status_var.set(text))

    def _on_run(self):
        if self._running:
            messagebox.showwarning("Внимание", "Алгоритм уже выполняется!")
            return

        self._running = True
        self.run_btn.configure(state="disabled")
        self.log_text.delete("1.0", tk.END)
        self.ax.clear()
        self.canvas.draw()

        thread = threading.Thread(target=self._run_algorithm, daemon=True)
        thread.start()

    def _run_algorithm(self):
        try:
            random.seed(42)
            np.random.seed(42)

            algo_name = self.algo_var.get()
            graph_name = self.graph_var.get()
            algo_type, use_mod = ALGO_MAP[algo_name]

            self._log(f"{'=' * 60}")
            self._log(f"  Алгоритм: {algo_name}")
            self._log(f"  Граф: {graph_name}")
            self._log(f"{'=' * 60}")

            # Загрузка графа
            self._set_status(f"Загрузка графа {graph_name}...")
            self._log(f"\n  Загрузка графа...")
            n, dist, node_names = load_graph(graph_name)
            self._log(f"  Граф загружен: {n} вершин")

            # Callback для обновления статуса
            def progress_cb(iteration, best_cost):
                self._set_status(f"Итерация {iteration}: лучшая стоимость = {best_cost:.2f}")

            # Запуск алгоритма
            self._set_status("Выполнение алгоритма...")
            self._log(f"\n  Запуск алгоритма...\n")

            if algo_type == "SA":
                T0 = float(self.t0_var.get())
                T_min = float(self.tmin_var.get())
                max_iter = int(self.sa_maxiter_var.get())

                self._log(f"  Параметры SA:")
                self._log(f"    T₀ = {T0}")
                self._log(f"    T_min = {T_min}")
                self._log(f"    Макс. итераций = {max_iter}")
                self._log(f"    Модификация: {'Отжиг Коши' if use_mod else 'Больцмановский (базовый)'}")
                self._log("")

                solver = SimulatedAnnealing(
                    dist, T0=T0, T_min=T_min, max_iter=max_iter,
                    use_modification=use_mod
                )
                route, cost, history, calls, elapsed = solver.run(callback=progress_cb)

            else:  # ACO
                n_ants = int(self.nants_var.get())
                n_iterations = int(self.aco_iter_var.get())
                alpha = float(self.alpha_var.get())
                beta = float(self.beta_var.get())
                rho = float(self.rho_var.get())
                Q = float(self.q_var.get())

                self._log(f"  Параметры ACO:")
                self._log(f"    Муравьёв = {n_ants}")
                self._log(f"    Итераций = {n_iterations}")
                self._log(f"    α = {alpha}, β = {beta}, ρ = {rho}, Q = {Q}")
                self._log(f"    Модификация: {'Начальное расположение' if use_mod else 'Блуждающая колония (базовый)'}")
                self._log("")

                solver = AntColonyOptimization(
                    dist, n_ants=n_ants, n_iterations=n_iterations,
                    alpha=alpha, beta=beta, rho=rho, Q=Q,
                    use_modification=use_mod
                )
                route, cost, history, calls, elapsed = solver.run(callback=progress_cb)

            # Вывод результатов
            route_str = format_route(route, node_names)
            self._log(f"\n{'─' * 50}")
            self._log(f"  РЕЗУЛЬТАТЫ")
            self._log(f"{'─' * 50}")
            self._log(f"  Лучший маршрут: {route_str}")
            self._log(f"  Стоимость маршрута: {cost:.2f}")
            self._log(f"  Вызовов целевой функции: {calls}")
            self._log(f"  Время выполнения: {elapsed:.3f} с")
            self._log(f"{'─' * 50}")

            # Построение графика
            self.root.after(0, self._draw_chart, history, algo_name)

            self._set_status(f"Готово! Стоимость: {cost:.2f}, Время: {elapsed:.3f} с")

        except Exception as e:
            self._log(f"\n  ОШИБКА: {e}")
            self._set_status(f"Ошибка: {e}")
            import traceback
            self._log(traceback.format_exc())

        finally:
            self._running = False
            self.root.after(0, lambda: self.run_btn.configure(state="normal"))

    def _draw_chart(self, history: list[float], algo_name: str):
        self.ax.clear()
        self.ax.plot(history, linewidth=1.5, color="#2196F3")
        self.ax.set_title(f"Сходимость: {algo_name}", fontsize=10)
        self.ax.set_xlabel("Шаг / Итерация", fontsize=9)
        self.ax.set_ylabel("Лучшая стоимость", fontsize=9)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()


def run_experiment(
    name: str,
    dist: np.ndarray,
    node_names: Optional[list[str]],
    sa_params: dict,
    aco_params: dict,
    output_dir: str = "results",
):
    '''Запуск всех 4 режимов (SA±mod, ACO±mod) на одном графе.'''
    matplotlib.use("Agg")
    os.makedirs(output_dir, exist_ok=True)
    n = dist.shape[0]
    print(f"\n{'=' * 70}")
    print(f"  Граф: {name} ({n} вершин)")
    print(f"{'=' * 70}")

    results = {}

    for key, label, cls, mod, params in [
        ("SA_base", "SA без модиф. (Больцмановский)", SimulatedAnnealing, False, sa_params),
        ("SA_mod", "SA с модиф. (Коши)", SimulatedAnnealing, True, sa_params),
        ("ACO_base", "ACO без модиф. (Блужд. колония)", AntColonyOptimization, False, aco_params),
        ("ACO_mod", "ACO с модиф. (Нач. расположение)", AntColonyOptimization, True, aco_params),
    ]:
        print(f"\n[{label}]")
        solver = cls(dist, use_modification=mod, **params)
        route, cost, hist, calls, elapsed = solver.run()
        results[key] = {"route": route, "cost": cost, "history": hist,
                        "calls": calls, "time": elapsed}
        route_str = format_route(route, node_names)
        print(f"  Лучший маршрут: {route_str}")
        print(f"  Стоимость: {cost:.2f}")
        print(f"  Вызовов ЦФ: {calls}, Время: {elapsed:.3f} с")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(results["SA_base"]["history"], label="SA без модиф. (Больцмановский)", linewidth=1.5)
    ax.plot(results["SA_mod"]["history"], label="SA с модиф. (Коши)", linewidth=1.5, linestyle="--")
    ax.set_title(f"Сходимость SA — {name}")
    ax.set_xlabel("Шаг записи")
    ax.set_ylabel("Лучшая стоимость")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results["ACO_base"]["history"], label="ACO без модиф. (Блужд. колония)", linewidth=1.5)
    ax.plot(results["ACO_mod"]["history"], label="ACO с модиф. (Нач. расположение)", linewidth=1.5, linestyle="--")
    ax.set_title(f"Сходимость ACO — {name}")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Лучшая стоимость")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"convergence_{name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  График сохранён: {fig_path}")

    print(f"\n  {'Режим':<40} {'Стоимость':>12} {'Вызовов ЦФ':>12} {'Время, с':>10}")
    print(f"  {'-' * 74}")
    for key, label in [
        ("SA_base", "SA без модиф. (Больцмановский)"),
        ("SA_mod", "SA с модиф. (Коши)"),
        ("ACO_base", "ACO без модиф. (Блужд. колония)"),
        ("ACO_mod", "ACO с модиф. (Нач. расположение)"),
    ]:
        r = results[key]
        print(f"  {label:<40} {r['cost']:>12.2f} {r['calls']:>12} {r['time']:>10.3f}")

    return results


def console_main():
    '''Консольный запуск всех экспериментов (для генерации отчёта).'''
    random.seed(42)
    np.random.seed(42)
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # 1. Малый граф
    names, edges = build_small_directed_graph()
    n_small = len(names)
    dist_small = build_distance_matrix_from_edges(n_small, edges, directed=True)

    print("\nМатрица расстояний малого орграфа:")
    print("    ", "  ".join(f"{n:>3}" for n in names))
    for i in range(n_small):
        row = []
        for j in range(n_small):
            if dist_small[i][j] == np.inf:
                row.append("inf")
            else:
                row.append(f"{dist_small[i][j]:3.0f}")
        print(f"  {names[i]}: {'  '.join(row)}")

    from itertools import permutations
    print("\n  Полный перебор гамильтоновых циклов малого орграфа:")
    best_perm_cost = np.inf
    best_perm_route = None
    count_valid = 0
    for perm in permutations(range(n_small)):
        c = route_cost(list(perm), dist_small)
        if c < np.inf:
            count_valid += 1
            if c < best_perm_cost:
                best_perm_cost = c
                best_perm_route = list(perm)
    print(f"  Допустимых циклов: {count_valid}")
    if best_perm_route is not None:
        print(f"  Оптимальный маршрут: {format_route(best_perm_route, names)}")
        print(f"  Оптимальная стоимость: {best_perm_cost:.2f}")

    all_results["small"] = run_experiment(
        "small_graph", dist_small, names,
        sa_params={"T0": 100, "T_min": 0.01, "max_iter": 10_000},
        aco_params={"n_ants": 10, "n_iterations": 100, "alpha": 1.0, "beta": 3.0, "rho": 0.5, "Q": 50},
        output_dir=output_dir,
    )

    # 2. berlin52
    if os.path.exists(BERLIN52_PATH):
        n_b, dist_b = parse_stp_file(BERLIN52_PATH)
        all_results["berlin52"] = run_experiment(
            "berlin52", dist_b, None,
            sa_params={"T0": 5000, "T_min": 0.001, "max_iter": 200_000},
            aco_params={"n_ants": 40, "n_iterations": 200, "alpha": 1.0, "beta": 3.0, "rho": 0.5, "Q": 500},
            output_dir=output_dir,
        )

    # 3. world666
    if os.path.exists(WORLD666_PATH):
        print("\n  Загрузка world666...")
        n_w, dist_w = parse_stp_file(WORLD666_PATH)
        all_results["world666"] = run_experiment(
            "world666", dist_w, None,
            sa_params={"T0": 50000, "T_min": 0.1, "max_iter": 300_000},
            aco_params={"n_ants": 20, "n_iterations": 30, "alpha": 1.0, "beta": 5.0, "rho": 0.3, "Q": 5000},
            output_dir=output_dir,
        )

    # Сводный график
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    graph_order = ["small", "berlin52", "world666"]
    graph_titles = ["Малый граф (6 вершин)", "berlin52 (52 вершины)", "world666 (666 вершин)"]
    for idx, (gname, gtitle) in enumerate(zip(graph_order, graph_titles)):
        if gname not in all_results:
            continue
        ax = axes[idx]
        res = all_results[gname]
        for key, label, ls in [
            ("SA_base", "SA базовый", "-"),
            ("SA_mod", "SA Коши", "--"),
            ("ACO_base", "ACO базовый", "-"),
            ("ACO_mod", "ACO нач. расп.", "--"),
        ]:
            ax.plot(res[key]["history"], label=label, linestyle=ls, linewidth=1.2)
        ax.set_title(gtitle)
        ax.set_xlabel("Шаг / Итерация")
        ax.set_ylabel("Лучшая стоимость")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_convergence.png"), dpi=150)
    plt.close()
    print("\nГотово!")


def main():
    import sys
    if "--console" in sys.argv:
        console_main()
    else:
        root = tk.Tk()
        app = TSPApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()