import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Константы
RHO = 1000.0          # плотность воды, кг/м^3
G = 9.81              # ускорение свободного падения, м/с^2

# =========================
# Функции расчета
# =========================

def theoretical_power(rho, g, A, Q):
    """
    Теоретическая максимальная мощность:
    P = (1/4) * rho * g * A * Q
    """
    return 0.25 * rho * g * A * Q


def real_power(rho, g, A, Q, eta):
    """
    Реальная мощность с учетом КПД.
    """
    return eta * theoretical_power(rho, g, A, Q)


def calculate_flow_rate(width, depth, tide_height, period_hours):
    """
    Расчет среднего расхода воды через пролив по его поперечному сечению и периоду прилива.
    """
    strait_area = width * depth  # площадь поперечного сечения пролива
    period_seconds = period_hours * 3600
    
    # За полный цикл прилива-отлива уровень поднимается и опускается на tide_height.
    # Мы считаем усредненный расход по поперечному сечению.
    Q = strait_area * 4 * tide_height / period_seconds
    return Q


def required_flow_for_target_power(P_real_target, rho, g, A, eta):
    """
    Находит расход Q, необходимый для получения заданной реальной мощности
    при известном КПД и приливной амплитуде.
    """
    P_theoretical_target = P_real_target / eta
    Q_required = 4 * P_theoretical_target / (rho * g * A)
    return Q_required, P_theoretical_target


class TidalPowerPlantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Приливная электростанция - Калькулятор")
        self.root.geometry("900x700")
        
        # Параметры пролива
        self.strait_length = 500.0
        self.strait_width = 200.0
        self.strait_depth = 15.0
        
        # Остальные параметры
        self.tide_height = 5.0
        self.period_hours = 12 + 25/60
        self.efficiency = 0.40
        self.target_power_mw = 22.5
        
        self.setup_ui()
    
    def setup_ui(self):
        """Создание интерфейса"""
        # Левая панель - входные данные
        left_frame = ttk.LabelFrame(self.root, text="Параметры пролива и системы", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Параметры пролива
        strait_frame = ttk.LabelFrame(left_frame, text="Пролив", padding=10)
        strait_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(strait_frame, text="Длина (м):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_length = ttk.Entry(strait_frame, width=15)
        self.entry_length.insert(0, str(self.strait_length))
        self.entry_length.grid(row=0, column=1, padx=5)
        
        ttk.Label(strait_frame, text="Ширина (м):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_width = ttk.Entry(strait_frame, width=15)
        self.entry_width.insert(0, str(self.strait_width))
        self.entry_width.grid(row=1, column=1, padx=5)
        
        ttk.Label(strait_frame, text="Средняя глубина (м):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.entry_depth = ttk.Entry(strait_frame, width=15)
        self.entry_depth.insert(0, str(self.strait_depth))
        self.entry_depth.grid(row=2, column=1, padx=5)
        
        # Параметры прилива
        tide_frame = ttk.LabelFrame(left_frame, text="Прилив", padding=10)
        tide_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(tide_frame, text="Высота прилива (м):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_tide = ttk.Entry(tide_frame, width=15)
        self.entry_tide.insert(0, str(self.tide_height))
        self.entry_tide.grid(row=0, column=1, padx=5)
        
        ttk.Label(tide_frame, text="Период прилива (ч):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_period = ttk.Entry(tide_frame, width=15)
        self.entry_period.insert(0, f"{self.period_hours:.4f}")
        self.entry_period.grid(row=1, column=1, padx=5)
        
        # Параметры станции
        station_frame = ttk.LabelFrame(left_frame, text="Станция", padding=10)
        station_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(station_frame, text="Целевая мощность (МВт):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.entry_target_power = ttk.Entry(station_frame, width=15)
        self.entry_target_power.insert(0, str(self.target_power_mw))
        self.entry_target_power.grid(row=0, column=1, padx=5)
        
        ttk.Label(station_frame, text="КПД (0-1):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entry_efficiency = ttk.Entry(station_frame, width=15)
        self.entry_efficiency.insert(0, str(self.efficiency))
        self.entry_efficiency.grid(row=1, column=1, padx=5)
        
        # Кнопка расчета
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        calc_button = ttk.Button(button_frame, text="Рассчитать", command=self.calculate)
        calc_button.pack(side=tk.LEFT, padx=5)
        
        reset_button = ttk.Button(button_frame, text="Сбросить", command=self.reset_values)
        reset_button.pack(side=tk.LEFT, padx=5)
        
        # Результаты
        results_frame = ttk.LabelFrame(left_frame, text="Результаты", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = tk.Text(results_frame, height=15, width=40, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Правая панель - график
        right_frame = ttk.LabelFrame(self.root, text="График мощности", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas_frame = right_frame
        
        # Начальный расчет и построение графика
        self.calculate()
    
    def calculate(self):
        """Выполнить расчет"""
        try:
            # Чтение входных данных
            length = float(self.entry_length.get())
            width = float(self.entry_width.get())
            depth = float(self.entry_depth.get())
            tide_height = float(self.entry_tide.get())
            period_hours = float(self.entry_period.get())
            target_power_mw = float(self.entry_target_power.get())
            efficiency = float(self.entry_efficiency.get())
            
            # Валидация
            if length <= 0 or width <= 0 or depth <= 0 or tide_height <= 0 or period_hours <= 0 or not (0 < efficiency <= 1) or target_power_mw <= 0:
                messagebox.showerror("Ошибка", "Все параметры должны быть положительными числами!\nКПД должен быть между 0 и 1.")
                return
            
            # Расчет расхода через поперечное сечение пролива
            Q_geom = calculate_flow_rate(width, depth, tide_height, period_hours)
            P_geom_theor = theoretical_power(RHO, G, tide_height, Q_geom)
            P_geom_real = real_power(RHO, G, tide_height, Q_geom, efficiency)

            # Расчет необходимого расхода для заданной мощности
            P_real_target = target_power_mw * 1e6
            Q_required, P_theor_target = required_flow_for_target_power(P_real_target, RHO, G, tide_height, efficiency)
            
            results = f"""
РЕЗУЛЬТАТЫ РАСЧЕТА
{'='*60}

Параметры пролива:
  Длина: {length:.2f} м
  Ширина: {width:.2f} м
  Глубина: {depth:.2f} м
  Сечение: {width*depth:.2f} м²

Параметры прилива:
  Высота: {tide_height:.2f} м
  Период: {period_hours:.4f} ч

Параметры станции:
  Целевая мощность: {target_power_mw:.2f} МВт
  КПД: {efficiency:.2f} ({efficiency*100:.1f}%)

{'='*60}
Расчеты по геометрии пролива:
  Расход Q: {Q_geom:.2f} м³/с
  Теоретическая мощность: {P_geom_theor/1e6:.3f} МВт
  Реальная мощность: {P_geom_real/1e6:.3f} МВт

Расчеты для цели {target_power_mw:.2f} МВт:
  Необходимый расход Q: {Q_required:.2f} м³/с
  Теоретическая мощность при этом расходе: {P_theor_target/1e6:.3f} МВт
"""
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results)
            self.results_text.config(state=tk.DISABLED)
            
            # Построение графика
            self.plot_graph(RHO, G, tide_height, efficiency, Q_required)
            
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Пожалуйста, введите корректные числовые значения")
    
    def plot_graph(self, rho, g, A, eta, Q_required=None):
        """Построение графика зависимости мощности от расхода"""
        # Очистка предыдущего графика
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        Q_values = np.linspace(0, 7000, 300)
        P_theor_values = theoretical_power(rho, g, A, Q_values) / 1e6
        P_real_values = real_power(rho, g, A, Q_values, eta) / 1e6
        
        ax.plot(Q_values, P_theor_values, label="Теоретическая", linewidth=2)
        ax.plot(Q_values, P_real_values, label="Реальная", linewidth=2)
        
        if Q_required is not None:
            ax.axvline(Q_required, color='red', linestyle='--', label=f'Q для {Q_required:.0f} м³/с')
        
        ax.set_title(f"Мощность при высоте прилива {A} м")
        ax.set_xlabel("Расход Q, м³/с")
        ax.set_ylabel("Мощность, МВт")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def reset_values(self):
        """Сброс значений по умолчанию"""
        self.entry_length.delete(0, tk.END)
        self.entry_length.insert(0, str(self.strait_length))

        self.entry_width.delete(0, tk.END)
        self.entry_width.insert(0, str(self.strait_width))
        
        self.entry_depth.delete(0, tk.END)
        self.entry_depth.insert(0, str(self.strait_depth))
        
        self.entry_tide.delete(0, tk.END)
        self.entry_tide.insert(0, str(self.tide_height))
        
        self.entry_period.delete(0, tk.END)
        self.entry_period.insert(0, f"{self.period_hours:.4f}")
        
        self.entry_target_power.delete(0, tk.END)
        self.entry_target_power.insert(0, str(self.target_power_mw))
        
        self.entry_efficiency.delete(0, tk.END)
        self.entry_efficiency.insert(0, str(self.efficiency))
        
        self.calculate()


def main():
    root = tk.Tk()
    app = TidalPowerPlantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
