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
