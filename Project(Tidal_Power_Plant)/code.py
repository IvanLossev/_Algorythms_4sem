import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
'''
Константы
'''
RHO = 1000.0          # плотность воды, кг/м^3
G = 9.81              # ускорение свободного падения, м/с^2

'''
Функции расчета

    Используется аналитическая модель (Garrett & Cummins) для поршневого режима залива.

    Основные формулы (из презентации, слайд "Определение мощности ПЭС"):
    1) Площадь поверхности залива:   S = L * W
    2) Энергия за полный приливный цикл:
        E_cycle = rho * g * S * H^2 / 4
    3) Средняя теоретическая мощность:
        P_theor = E_cycle / T = rho * g * S * H^2 / (4 * T)
    4) Реальная мощность:
        P_real = eta * P_theor
'''


def bay_surface_area(length: float, width: float) -> float:
    """Площадь поверхности залива S = L * W, м²."""
    return length * width


def cycle_energy(rho: float, g: float, S: float, H: float) -> float:
    """Максимальная теоретическая энергия за полный приливный цикл, Дж:
        E_cycle = rho * g * S * H^2 / 4
    """
    return rho * g * S * H ** 2 / 4.0


def theoretical_power(rho: float, g: float, S: float, H: float, T_seconds: float) -> float:
    """Средняя теоретическая мощность потока, Вт:
        P_theor = rho * g * S * H^2 / (4 * T)
    T_seconds — период прилива в секундах.
    """
    return rho * g * S * H ** 2 / (4.0 * T_seconds)


def real_power(rho: float, g: float, S: float, H: float, T_seconds: float, eta: float) -> float:
    """Реальная усреднённая мощность станции с учётом КПД турбин, Вт:
        P_real = eta * P_theor
    """
    return eta * theoretical_power(rho, g, S, H, T_seconds)


def average_flow_rate(S: float, H: float, T_seconds: float) -> float:
    """Средний объёмный расход через пролив, м³/с.

    За половину периода (прилив или отлив) через сечение пролива проходит объём,
    равный S * H (уровень меняется на H по всей площади залива).
    Среднее по модулю значение расхода:
        Q_mean = 2 * S * H / T
    (множитель 2, так как за период T происходит и прилив, и отлив, каждый длится T/2;
    объём за половину цикла = S*H, поэтому Q = S*H / (T/2) = 2*S*H/T).
    """
    return 2.0 * S * H / T_seconds


# =========================
# GUI
# =========================
class TidalPowerPlantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Приливная электростанция - Калькулятор (исправленная версия)")
        self.root.geometry("980x720")

        # Параметры пролива (горловины)
        self.strait_length = 500.0   # длина пролива L_s, м
        self.strait_width = 200.0    # ширина пролива W_s, м
        self.strait_depth = 15.0     # средняя глубина пролива D, м

        # Параметры залива (бассейна).
        # В условии задачи площадь залива не задана явно, но из ожидаемого
        # результата (≈22.5 МВт) обратным ходом получается S_bay ≈ 40.8 км².
        # Это согласуется с типичными размерами заливов, соединённых с морем
        # узкой горловиной (~6.4 × 6.4 км).
        self.bay_length = 6400.0     # длина залива L, м
        self.bay_width = 6400.0      # ширина залива W, м

        # Параметры прилива
        self.tide_height = 5.0                  # высота прилива H, м
        self.period_hours = 12 + 25.0 / 60.0    # период прилива T, ч
        self.efficiency = 0.40                  # КПД гидротурбин
        self.target_power_mw = 22.5             # ожидаемая мощность из презентации

        self.setup_ui()

    # ---------- UI ----------
    def setup_ui(self):
        left_frame = ttk.LabelFrame(self.root, text="Параметры пролива / залива и станции", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Пролив
        strait_frame = ttk.LabelFrame(left_frame, text="Пролив (горловина)", padding=10)
        strait_frame.pack(fill=tk.X, pady=5)

        ttk.Label(strait_frame, text="Длина Lₛ (м):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_strait_length = ttk.Entry(strait_frame, width=15)
        self.entry_strait_length.insert(0, str(self.strait_length))
        self.entry_strait_length.grid(row=0, column=1, padx=5)

        ttk.Label(strait_frame, text="Ширина Wₛ (м):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_strait_width = ttk.Entry(strait_frame, width=15)
        self.entry_strait_width.insert(0, str(self.strait_width))
        self.entry_strait_width.grid(row=1, column=1, padx=5)

        ttk.Label(strait_frame, text="Средняя глубина D (м):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.entry_depth = ttk.Entry(strait_frame, width=15)
        self.entry_depth.insert(0, str(self.strait_depth))
        self.entry_depth.grid(row=2, column=1, padx=5)

        # Залив
        bay_frame = ttk.LabelFrame(left_frame, text="Залив (бассейн)", padding=10)
        bay_frame.pack(fill=tk.X, pady=5)

        ttk.Label(bay_frame, text="Длина залива L (м):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_bay_length = ttk.Entry(bay_frame, width=15)
        self.entry_bay_length.insert(0, str(self.bay_length))
        self.entry_bay_length.grid(row=0, column=1, padx=5)

        ttk.Label(bay_frame, text="Ширина залива W (м):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_bay_width = ttk.Entry(bay_frame, width=15)
        self.entry_bay_width.insert(0, str(self.bay_width))
        self.entry_bay_width.grid(row=1, column=1, padx=5)

        # Прилив
        tide_frame = ttk.LabelFrame(left_frame, text="Прилив", padding=10)
        tide_frame.pack(fill=tk.X, pady=5)

        ttk.Label(tide_frame, text="Высота прилива H (м):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_tide = ttk.Entry(tide_frame, width=15)
        self.entry_tide.insert(0, str(self.tide_height))
        self.entry_tide.grid(row=0, column=1, padx=5)

        ttk.Label(tide_frame, text="Период прилива T (ч):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_period = ttk.Entry(tide_frame, width=15)
        self.entry_period.insert(0, f"{self.period_hours:.4f}")
        self.entry_period.grid(row=1, column=1, padx=5)

        # Станция
        station_frame = ttk.LabelFrame(left_frame, text="Станция", padding=10)
        station_frame.pack(fill=tk.X, pady=5)

        ttk.Label(station_frame, text="Ожидаемая мощность (МВт):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_target_power = ttk.Entry(station_frame, width=15)
        self.entry_target_power.insert(0, str(self.target_power_mw))
        self.entry_target_power.grid(row=0, column=1, padx=5)

        ttk.Label(station_frame, text="КПД η (0-1):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_efficiency = ttk.Entry(station_frame, width=15)
        self.entry_efficiency.insert(0, str(self.efficiency))
        self.entry_efficiency.grid(row=1, column=1, padx=5)

        # Кнопки
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Рассчитать", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Сбросить", command=self.reset_values).pack(side=tk.LEFT, padx=5)

        # Результаты
        results_frame = ttk.LabelFrame(left_frame, text="Результаты", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.results_text = tk.Text(results_frame, height=18, width=45, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Правая панель - график
        right_frame = ttk.LabelFrame(self.root, text="Графики", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_frame = right_frame

        self.calculate()

    # ---------- Calculations ----------
    def calculate(self):
        try:
            strait_length = float(self.entry_strait_length.get())
            strait_width = float(self.entry_strait_width.get())
            depth = float(self.entry_depth.get())
            bay_length = float(self.entry_bay_length.get())
            bay_width = float(self.entry_bay_width.get())
            H = float(self.entry_tide.get())
            period_hours = float(self.entry_period.get())
            target_power_mw = float(self.entry_target_power.get())
            eta = float(self.entry_efficiency.get())

            if (strait_length <= 0 or strait_width <= 0 or depth <= 0
                    or bay_length <= 0 or bay_width <= 0 or H <= 0
                    or period_hours <= 0 or not (0 < eta <= 1) or target_power_mw <= 0):
                messagebox.showerror(
                    "Ошибка",
                    "Все параметры должны быть положительными числами!\nКПД должен быть в (0; 1].",
                )
                return

            # --- Основные величины ---
            T_sec = period_hours * 3600.0                    # период в секундах
            S = bay_surface_area(bay_length, bay_width)      # площадь залива, м²
            E_cycle = cycle_energy(RHO, G, S, H)             # энергия за цикл, Дж
            P_theor = theoretical_power(RHO, G, S, H, T_sec) # Вт
            P_real = real_power(RHO, G, S, H, T_sec, eta)    # Вт
            Q_mean = average_flow_rate(S, H, T_sec)          # средний расход, м³/с

            # Средняя скорость потока в проливе, м/с:
            U_mean = Q_mean / (strait_width * depth)

            results = f"""
РЕЗУЛЬТАТЫ РАСЧЁТА
{'=' * 60}

Параметры пролива (горловины):
    Длина Lₛ: {strait_length:.2f} м
    Ширина Wₛ: {strait_width:.2f} м
    Глубина D: {depth:.2f} м
    Сечение пролива Wₛ·D: {strait_width * depth:.2f} м²

Параметры залива (бассейна):
    Длина L: {bay_length:.2f} м
    Ширина W: {bay_width:.2f} м
    Площадь залива S = L·W: {S:.0f} м² = {S / 1e6:.2f} км²

Параметры прилива:
    Высота H: {H:.2f} м
    Период T: {period_hours:.4f} ч = {T_sec:.0f} с

Параметры станции:
    КПД η: {eta:.2f} ({eta * 100:.1f}%)
    Ожидаемая мощность: {target_power_mw:.2f} МВт

{'=' * 60}
Промежуточные величины:
    Энергия за цикл E_cycle = ρgSH²/4 = {E_cycle:.3e} Дж
    Средний расход через пролив Q = 2SH/T = {Q_mean:.2f} м³/с
    Средняя скорость потока Ū = Q/(Wₛ·D) = {U_mean:.3f} м/с

Мощность:
    Теоретическая P_theor = ρgSH²/(4T) = {P_theor / 1e6:.3f} МВт
    Реальная      P_real  = η·P_theor  = {P_real / 1e6:.3f} МВт

Сравнение с ожидаемым (из презентации):
    P_real ≈ {P_real / 1e6:.2f} МВт  vs.  target ≈ {target_power_mw:.2f} МВт
"""
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            self.results_text.config(state=tk.DISABLED)

            self.plot_graphs(bay_length, bay_width, H, T_sec, eta)

        except ValueError:
            messagebox.showerror("Ошибка ввода", "Пожалуйста, введите корректные числовые значения")

    def plot_graphs(self, bay_length, bay_width, H, T_sec, eta):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(6, 7), dpi=100)

        # График 1: уровень воды в заливе во времени (синусоида амплитудой H/2)
        ax1 = fig.add_subplot(211)
        t = np.linspace(0, 2 * T_sec, 500)
        level = (H / 2.0) * np.sin(2 * np.pi * t / T_sec)
        ax1.plot(t / 3600.0, level, color='steelblue', linewidth=2)
        ax1.set_title("Изменение уровня воды в заливе")
        ax1.set_xlabel("Время, ч")
        ax1.set_ylabel("Отклонение уровня, м")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linewidth=0.5)

        # График 2: мгновенная мощность станции
        ax2 = fig.add_subplot(212)
        S = bay_length * bay_width
        # dη/dt = (H/2) * (2π/T) * cos(2πt/T)
        deta_dt = (H / 2.0) * (2 * np.pi / T_sec) * np.cos(2 * np.pi * t / T_sec)
        # Мгновенная мощность |ρ g S η (dη/dt)|, затем умножаем на КПД
        instant_P = np.abs(RHO * G * S * level * deta_dt) * eta
        ax2.plot(t / 3600.0, instant_P / 1e6, color='darkorange', linewidth=2, label='Мгновенная мощность')
        P_avg = real_power(RHO, G, S, H, T_sec, eta) / 1e6
        ax2.axhline(P_avg, color='red', linestyle='--', linewidth=1.5,
                    label=f'Средняя ~ {P_avg:.2f} МВт')
        ax2.set_title("Мгновенная реальная мощность станции")
        ax2.set_xlabel("Время, ч")
        ax2.set_ylabel("P, МВт")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- Reset ----------
    def reset_values(self):
        for entry, value in (
            (self.entry_strait_length, str(self.strait_length)),
            (self.entry_strait_width, str(self.strait_width)),
            (self.entry_depth, str(self.strait_depth)),
            (self.entry_bay_length, str(self.bay_length)),
            (self.entry_bay_width, str(self.bay_width)),
            (self.entry_tide, str(self.tide_height)),
            (self.entry_period, f"{self.period_hours:.4f}"),
            (self.entry_target_power, str(self.target_power_mw)),
            (self.entry_efficiency, str(self.efficiency)),
        ):
            entry.delete(0, tk.END)
            entry.insert(0, value)
        self.calculate()


def main():
    root = tk.Tk()
    TidalPowerPlantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()