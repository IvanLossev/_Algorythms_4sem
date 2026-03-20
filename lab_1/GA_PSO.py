import math
import random
import numpy as np
import matplotlib.pyplot as plt

def function_var9(x, y):
    func = -0.0001 * (abs(math.sin(x)*math.sin(y)*math.exp(abs(100 - (math.sqrt(x**2 + y**2) / math.pi)))) + 1)**0.1
    return func

class Genetic_algo:
    def __init__(self, population_size=100, generations=200, answer_bounds=(-5, 5), elite_ratio=0.5, mutation_rate=0.1):
        self.pop_size = population_size
        self.generations = generations
        self.bounds = answer_bounds
        self.remains = elite_ratio
        self.func_calls = 0
        self.mutation_rate = mutation_rate
        
    def evaluate (self, x, y):
        self.func_calls += 1
        return function_var9(x, y)
    
    def init_population(self):
        return [[random.uniform(self.bounds[0], self.bounds[1]), 
                random.uniform(self.bounds[0], self.bounds[1])]
                for _ in range(self.pop_size)]
    
    def new_population(self, parent1, parent2):
        x1, y1 = parent1
        x2, y2 = parent2
        
        flag = random.choice([0.3, 0.4, 0.5])
        
        child1_x = flag * x1 + (1 - flag) * x2
        child1_y = flag * y1 + (1 - flag) * y2
        
        child2_x = flag * x2 + (1 - flag) * x1
        child2_y = flag * y2 + (1 - flag) * y1
        
        return [child1_x, child1_y], [child2_x, child2_y]
    
    def run(self):
        population = self.init_population()
        best_result = float('inf')
        best_vals = None
        history = []
        
        for gen in range (self.generations):
            fitness = [self.evaluate(ind[0], ind[1]) for ind in population]
            sorted_indices = np.argsort(fitness)
            sorted_pop = [population[i] for i in sorted_indices]
            sorted_fit = [fitness[i] for i in sorted_indices]
            
            best_f = sorted_fit[0]
            best_ind = sorted_pop[0]
            
            if best_f < best_result:
                best_result = best_f
                best_vals = best_ind
            
            history.append(best_f)
            
            elite_size = int(self.pop_size * self.remains)
            elites = sorted_pop[:elite_size]
            parents = sorted_pop[:elite_size]
            random.shuffle(parents)
            
            children = []
            
            for i in range (0, len(parents) - 1, 2):
                if len(children) < self.pop_size - elite_size:
                    p1 = parents[i]
                    p2 = parents[i+1]
                    c1, c2 = self.new_population(p1, p2)
                    
                    children.append(c1)
                    if len(children) < self.pop_size - elite_size:
                        children.append(c2)
                    
            # если к-во членов популяции нечетное
            while len(children) < self.pop_size - elite_size:
                x = random.uniform(self.bounds[0], self.bounds[1])
                y = random.uniform(self.bounds[0], self.bounds[1])
                children.append([x, y])
                
            population = elites + children
            
            #Каждые 20 поколений печатаем, как идут дела.
            if (gen + 1) % 5 == 0:
                print(f"Поколение {gen+1}: лучший f = {best_f:.6f} в ({best_ind[0]:.5f}, {best_ind[1]:.5f})")
            
        return best_vals, best_result, history, self.func_calls

class ParticleSwarm:
    def __init__(self, bounds=(-5, 5), n_particles=30, generations=200, w=0.7, c1=1.5, c2=1.5, Vmax=1.0):
        self.bounds = bounds
        self.dim = 2    #размерность пространства
        self.n_particles = n_particles
        self.generations = generations
        self.w = w
        self.c1 = c1    #собственне данные
        self.c2 = c2    #роевые данные
        self.Vmax = Vmax
        
        self.func_calls = 0
        
        #данные частиц
        self.positions = None                   # текущие позиции
        self.velocities = None                  # текущие скорости
        self.best_positions = None              # лучшие личные позиции
        self.best_coords = None                 # значения в лучших личных позициях
        self.global_best = None                 # глобальный лучший
        self.global_best_coords = float('inf')  # лучшее глобальное значение

    def evaluate(self, x, y):
        self.func_calls += 1
        return function_var9(x, y)

    def init_population(self):
        self.positions = np.array([[random.uniform(self.bounds[0], self.bounds[1]) 
                                   for _ in range(self.dim)]
                                   for _ in range(self.n_particles)])
        
        self.velocities = np.zeros((self.n_particles, self.dim))
        
        #личные лучшие позиции (пока совпадают с текущими)
        self.best_positions = self.positions.copy()
        
        #вычисляем фитнес для начальных позиций
        self.best_coords = np.array([self.evaluate(p[0], p[1]) for p in self.positions])
        
        # Находим глобальный лучший
        best_idx = np.argmin(self.best_coords)
        self.global_best = self.best_positions[best_idx].copy()
        self.global_best_coords = self.best_coords[best_idx]

    def update_velocity(self, i):
        
        #Обновление скорости частицы i с ограничением
        r1 = random.random()
        r2 = random.random()
        
        # Когнитивная составляющая (к личному лучшему)
        cognitive = self.c1 * r1 * (self.best_positions[i] - self.positions[i])
        
        # Социальная составляющая (к глобальному лучшему)
        social = self.c2 * r2 * (self.global_best - self.positions[i])
        
        # Новая скорость
        self.velocities[i] = self.w * self.velocities[i] + cognitive + social
        
        for j in range(self.dim):
            if self.velocities[i][j] > self.Vmax:
                self.velocities[i][j] = self.Vmax
            elif self.velocities[i][j] < -self.Vmax:
                self.velocities[i][j] = -self.Vmax

    def update_position(self, i):
        """
        Обновление позиции частицы i и проверка границ
        """
        self.positions[i] += self.velocities[i]
        
        # Проверка границ
        for j in range(self.dim):
            if self.positions[i][j] < self.bounds[0]:
                self.positions[i][j] = self.bounds[0]
                # Можно также отразить скорость
                # self.velocities[i][j] *= -0.5
            elif self.positions[i][j] > self.bounds[1]:
                self.positions[i][j] = self.bounds[1]
                # self.velocities[i][j] *= -0.5

    def run(self):
        self.init_population()
        history = [self.global_best_coords]
        
        print("Начало работы роевого алгоритма...")
        print(f"Начальный gbest: {self.global_best_coords:.6f} в точке ({self.global_best[0]:.5f}, {self.global_best[1]:.5f})")
        
        for gen in range(self.generations):
            for i in range(self.n_particles):
                # Вычисляем фитнес в текущей позиции
                fitness = self.evaluate(self.positions[i][0], self.positions[i][1])
                
                # Обновляем личный лучший (pbest)
                if fitness < self.best_coords[i]:
                    self.best_coords[i] = fitness
                    self.best_positions[i] = self.positions[i].copy()
                
                # Обновляем глобальный лучший (gbest)
                if fitness < self.global_best_coords:
                    self.global_best_coords = fitness
                    self.global_best = self.positions[i].copy()
                
                # Обновляем скорость и позицию (для следующего шага)
                self.update_velocity(i)
                self.update_position(i)
            
            # Сохраняем историю
            history.append(self.global_best_coords)
            
            # Вывод прогресса
            if (gen + 1) % 20 == 0:
                print(f"Итерация {gen+1}: лучший f = {self.global_best_coords:.6f} в ({self.global_best[0]:.5f}, {self.global_best[1]:.5f})")
        
        return self.global_best, self.global_best_coords, history, self.func_calls


# ---------- Запуск PSO ----------
if __name__ == "__main__":
    print("="*60)
    
    # Создаём и запускаем PSO
    pso = ParticleSwarm(
        bounds=(-5, 5),
        n_particles=30,      # количество частиц
        generations=200,      # число итераций
        w=0.7,                # инерция
        c1=1.5,               # когнитивный коэффициент
        c2=1.5,               # социальный коэффициент
        Vmax=1.0              # максимальная скорость (ограничение)
    )
    
    best, best_f, history, calls = pso.run()
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ PSO:")
    print(f"Лучшее решение: x = {best[0]:.6f}, y = {best[1]:.6f}")
    print(f"Значение функции: f = {best_f:.6f}")
    print(f"Вызовов функции: {calls}")
    print(f"Ожидаемый минимум: -2.06261")
    print("="*60)
    
    # Построение графика сходимости
    plt.figure(figsize=(12, 5))
    
    # График 1: Сходимость PSO
    plt.subplot(1, 2, 1)
    plt.plot(history, linewidth=2, color='green')
    plt.xlabel('Итерация', fontsize=12)
    plt.ylabel('Лучшее значение функции', fontsize=12)
    plt.title('Сходимость PSO', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=-2.06261, color='red', linestyle='--', alpha=0.7, label='Ожидаемый минимум')
    plt.legend()
    
    # График 2: Траектории частиц (последние позиции)
    plt.subplot(1, 2, 2)
    
    # Известные минимумы
    known_minima = [
        (1.34941, -1.34941),
        (1.34941, 1.34941),
        (-1.34941, 1.34941),
        (-1.34941, -1.34941)
    ]
    known_x = [p[0] for p in known_minima]
    known_y = [p[1] for p in known_minima]
    plt.scatter(known_x, known_y, c='red', marker='*', s=200, label='Известные минимумы', zorder=5)
    
    # Текущие позиции частиц
    plt.scatter(pso.positions[:, 0], pso.positions[:, 1], c='blue', marker='o', s=50, alpha=0.5, label='Частицы', zorder=3)
    
    # Лучшая найденная точка
    plt.scatter(best[0], best[1], c='green', marker='D', s=100, label='Найденный минимум', zorder=4)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Позиции частиц в конце поиска', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(pso.bounds[0], pso.bounds[1])
    plt.ylim(pso.bounds[0], pso.bounds[1])
    
    plt.tight_layout()
    plt.savefig('pso_results.png', dpi=150)
    plt.show()

'''
if __name__ == "__main__":
    ga = Genetic_algo(
        population_size=100,
        generations=200,
        answer_bounds=(-20, 20),
        elite_ratio=0.5,
        mutation_rate=0.2   # 20% мутаций
    )
    
    best, best_f, history, calls = ga.run()
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Лучшее решение: x = {best[0]:.6f}, y = {best[1]:.6f}")
    print(f"Значение функции: f = {best_f:.6f}")
    print("Ожидаемый минимум: -2.06261")
    print("="*50)
    
    plt.plot(history)
    plt.xlabel('Поколение')
    plt.ylabel('Лучшее f(x,y)')
    plt.title('Сходимость ГА (с мутациями 20%)')
    plt.grid(True)
    plt.show()
'''