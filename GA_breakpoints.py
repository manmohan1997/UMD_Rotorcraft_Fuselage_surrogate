import numpy as np
# from scipy.optimize import least_squares
# from scipy.signal import argrelextrema
# from sklearn.metrics import mean_squared_error
# from numpy.polynomial.chebyshev import Chebyshev
# from scipy.optimize import curve_fit
from miscelene import *
from Fitting_Methods import *
from BreakPoint_Detector import *

# # import numpy as np
# import pymoo
# dir(pymoo)
from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.algorithms.soo.genetic_algorithm import GA
from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.survival import Survival
from pymoo.core.callback import Callback
from joblib import Parallel, delayed

from pymoo.optimize import minimize
from sklearn.metrics import mean_squared_error
class PrintBestFitnessCallback(Callback):
    def __init__(self, x_values):
        super().__init__()
        self.generation = 0
        self.x_values = x_values

    def notify(self, algorithm):
        self.generation += 1
        # Get the individual with the best fitness (lowest objective function value)
        best_index = algorithm.pop.get("F").argmin()
        best_solution = algorithm.pop.get("X")[best_index]
        
        # Extract indices of critical points
        critical_points_indices = np.where(best_solution.astype(bool))[0]
        
        # Print generation info
        print(f"Generation {self.generation}: Best Fitness = {algorithm.pop.get('F').min():.8f}, "
              f"Critical Points Indices = {critical_points_indices.tolist()}")


# Genetic algorithm using pymoo
class CriticalPointsSampling(Sampling):
    def __init__(self, min_critical_points, max_critical_points, x_values, initial_critical_points=None):
        super().__init__()
        self.min_critical_points = min_critical_points
        self.max_critical_points = max_critical_points
        self.x_values = x_values
        self.initial_critical_points = initial_critical_points

    def _do(self, problem, n_samples, **kwargs):
        population = []

        # Add initial critical points to the population if provided
        if self.initial_critical_points is not None:
            initial_critical_points_sorted = np.sort(self.initial_critical_points)
            binary_individual = np.zeros(len(self.x_values), dtype=bool)
            binary_individual[initial_critical_points_sorted] = True

            # Ensure it meets the constraints
            all_indices = [0] + list(initial_critical_points_sorted) + [len(self.x_values) - 1]
            segments = [(all_indices[i], all_indices[i + 1]) for i in range(len(all_indices) - 1)]
            if all(len(self.x_values[start:end + 1]) >= 4 for start, end in segments):
                population.append(binary_individual)

        # Add random individuals to fill the rest of the population
        while len(population) < n_samples:
            while True:
                ind_len = np.random.randint(self.min_critical_points, self.max_critical_points + 1)
                individual = np.sort(np.random.choice(len(self.x_values), ind_len, replace=False))
                all_indices = [0] + list(individual) + [len(self.x_values) - 1]
                segments = [(all_indices[i], all_indices[i + 1]) for i in range(len(all_indices) - 1)]

                if all(len(self.x_values[start:end + 1]) >= 4 for start, end in segments):
                    binary_individual = np.zeros(len(self.x_values), dtype=bool)
                    binary_individual[individual] = True
                    population.append(binary_individual)
                    break

        return np.array(population)
   
class CriticalPointsCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)  # 2 parents, 2 children

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, False)

        for i in range(n_matings):
            p1, p2 = X[0, i, :], X[1, i, :]

            # Convert binary parents to indices of critical points
            indices1 = np.where(p1)[0]
            indices2 = np.where(p2)[0]

            # Perform crossover (blend point indices from both parents)
            point = np.random.randint(1, min(len(indices1), len(indices2)))
            child1_indices = np.sort(np.unique(np.concatenate((indices1[:point], indices2[point:]))))
            child2_indices = np.sort(np.unique(np.concatenate((indices2[:point], indices1[point:]))))

            # Create binary offspring
            child1 = np.zeros(n_var, dtype=bool)
            child2 = np.zeros(n_var, dtype=bool)
            child1[child1_indices] = True
            child2[child2_indices] = True

            Y[0, i, :] = child1
            Y[1, i, :] = child2

        return Y

class CriticalPointsMutation(Mutation):
    def __init__(self, mutation_rate, min_critical_points, max_critical_points, x_values):
        super().__init__()
        self.mutation_rate = mutation_rate
        self.min_critical_points = min_critical_points
        self.max_critical_points = max_critical_points
        self.x_values = x_values

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            if np.random.rand() < self.mutation_rate:
                individual = np.where(X[i, :])[0]
                if len(individual) > self.min_critical_points and (
                    len(individual) < self.max_critical_points and np.random.rand() > 0.5
                    or len(individual) == self.max_critical_points
                ):
                    idx = np.random.randint(len(individual))
                    individual = np.delete(individual, idx)
                else:
                    new_point = np.random.randint(1, len(self.x_values) - 1)
                    individual = np.append(individual, new_point)
                binary_individual = np.zeros(len(self.x_values), dtype=bool)
                binary_individual[individual] = True
                X[i, :] = binary_individual
        return X
class CriticalPointsOptimization(Problem):
    def __init__(self, x_values, y_values_dict, slopes_dict, method, min_critical_points, max_critical_points):
        self.x_values = x_values
        self.y_values_dict = y_values_dict
        self.slopes_dict = slopes_dict
        self.method = method
        self.min_critical_points = min_critical_points
        self.max_critical_points = max_critical_points
        super().__init__(n_var=len(x_values), 
                         n_obj=1, 
                         xl=0, 
                         xu=1, 
                         type_var=np.bool_)

    # def _evaluate(self, x, out, *args, **kwargs):
    #     solutions = x.astype(bool)
    #     fitnesses = []

    #     for solution in solutions:
    #         critical_points_indices = np.where(solution)[0]
    #         fitness = self._fitness_function(critical_points_indices)
    #         fitnesses.append(fitness)

    #     out["F"] = np.array(fitnesses).reshape(-1, 1)

    #parralel implememntation 
    def _evaluate(self, x, out, *args, **kwargs):
        solutions = x.astype(bool)

        # Parallelize the computation of fitness values
        fitnesses = Parallel(n_jobs=-1)(  # -1 means use all available CPU cores
            delayed(self._fitness_function)(np.where(solution)[0]) for solution in solutions
        )

        # Store the computed fitnesses
        out["F"] = np.array(fitnesses).reshape(-1, 1)

    def _fitness_function(self, critical_points_indices):
        if len(critical_points_indices) < self.min_critical_points:
            return float('inf')

        all_indices = [0] + list(critical_points_indices) + [len(self.x_values) - 1]
        segments = [(all_indices[i], all_indices[i + 1]) for i in range(len(all_indices) - 1)]
        total_rms_error = 0

        for var_name in self.y_values_dict.keys():
            y_values = np.array(self.y_values_dict[var_name], dtype=np.float64)
            slopes = np.array(self.slopes_dict[var_name], dtype=np.float64)

            if self.method == 'hermite':
                fitted_segments = compute_hermite_segments(self.x_values, y_values, slopes, segments)
                y_fitted = evaluate_hermite_segments(self.x_values, fitted_segments)
            elif self.method == 'chebyshev':
                fitted_segments = compute_chebyshev_segments(self.x_values, y_values, segments, degree=3)
                y_fitted = evaluate_chebyshev_segments(self.x_values, self.x_values, fitted_segments, segments)

            if np.any(np.isnan(y_fitted)):
                return float('inf')

            rms_error = np.sqrt(mean_squared_error(y_values, y_fitted))
            total_rms_error += rms_error

        return total_rms_error / len(self.y_values_dict)

class ElitistSurvival(Survival):
    def __init__(self,n_survive):
        super().__init__(filter_infeasible=True)
        self.n_survive = n_survive
        self.best_individual = None
        self.best_fitness = np.inf

    def _do(self, problem, pop, n_survive, **kwargs):
        F = pop.get("F")

        # Update best individual seen so far
        for i, f in enumerate(F):
            if f[0] < self.best_fitness:
                self.best_fitness = f[0]
                self.best_individual = pop[i]

        # Sort population by fitness
        I = np.argsort(F[:, 0])
        survivors = pop[I[:self.n_survive]]

        # Check if best individual is still in the survivors
        is_elite_present = any(np.all(ind.X == self.best_individual.X) for ind in survivors)

        # If not, replace the worst with the best
        if not is_elite_present:
            survivors[-1] = self.best_individual

        return survivors
    
def genetic_algorithm_optimum_critical_points(x_values, y_values_dict, slopes_dict, initial_critical_points=None,
                                                     method='hermite', population_size=240, generations=50, 
                                                     mutation_rate=0.4, min_critical_points=3, max_critical_points=5):
    n_survive = population_size // 2 
    # Define the problem
    problem = CriticalPointsOptimization(x_values, y_values_dict, slopes_dict, method, 
                                         min_critical_points, max_critical_points)

    # Configure the genetic algorithm
    algorithm = GA(
        pop_size=population_size,
        sampling=CriticalPointsSampling(min_critical_points, max_critical_points, x_values, initial_critical_points),
        crossover=CriticalPointsCrossover(),
        mutation=CriticalPointsMutation(mutation_rate, min_critical_points, max_critical_points, x_values),
        survival=ElitistSurvival(n_survive)   # Enable elitism   
    )
    callback = PrintBestFitnessCallback(x_values)
    # Run optimization
    res = minimize(problem,
                algorithm,
                ('n_gen', generations),
                verbose=True,
                callback=callback)  

    # Extract best solution
    best_solution = res.X
    best_indices = np.where(best_solution.astype(bool))[0]
    best_rms_error = res.F[0]

    return best_rms_error, best_indices.tolist()



# ## GA for animation ####

# class PrintBestFitnessCallback(Callback):
#     def __init__(self, x_values):
#         super().__init__()
#         self.generation = 0
#         self.x_values = x_values

#     def notify(self, algorithm):
#         self.generation += 1
#         # Get the individual with the best fitness (lowest objective function value)
#         best_index = algorithm.pop.get("F").argmin()
#         best_solution = algorithm.pop.get("X")[best_index]
        
#         # Extract indices of critical points
#         critical_points_indices = np.where(best_solution.astype(bool))[0]
        
#         # Print generation info
#         print(f"Generation {self.generation}: Best Fitness = {algorithm.pop.get('F').min():.8f}, "
#               f"Critical Points Indices = {critical_points_indices.tolist()}")


# # Genetic algorithm using pymoo
# class CriticalPointsSampling(Sampling):
#     def __init__(self, min_critical_points, max_critical_points, x_values, initial_critical_points=None):
#         super().__init__()
#         self.min_critical_points = min_critical_points
#         self.max_critical_points = max_critical_points
#         self.x_values = x_values
#         self.initial_critical_points = initial_critical_points

#     def _do(self, problem, n_samples, **kwargs):
#         population = []

#         # Add initial critical points to the population if provided
#         if self.initial_critical_points is not None:
#             initial_critical_points_sorted = np.sort(self.initial_critical_points)
#             binary_individual = np.zeros(len(self.x_values), dtype=bool)
#             binary_individual[initial_critical_points_sorted] = True

#             # Ensure it meets the constraints
#             all_indices = [0] + list(initial_critical_points_sorted) + [len(self.x_values) - 1]
#             segments = [(all_indices[i], all_indices[i + 1]) for i in range(len(all_indices) - 1)]
#             if all(len(self.x_values[start:end + 1]) >= 4 for start, end in segments):
#                 population.append(binary_individual)

#         # Add random individuals to fill the rest of the population
#         while len(population) < n_samples:
#             while True:
#                 ind_len = np.random.randint(self.min_critical_points, self.max_critical_points + 1)
#                 individual = np.sort(np.random.choice(len(self.x_values), ind_len, replace=False))
#                 all_indices = [0] + list(individual) + [len(self.x_values) - 1]
#                 segments = [(all_indices[i], all_indices[i + 1]) for i in range(len(all_indices) - 1)]

#                 if all(len(self.x_values[start:end + 1]) >= 4 for start, end in segments):
#                     binary_individual = np.zeros(len(self.x_values), dtype=bool)
#                     binary_individual[individual] = True
#                     population.append(binary_individual)
#                     break

#         return np.array(population)
   
# class CriticalPointsCrossover(Crossover):
#     def __init__(self):
#         super().__init__(2, 2)  # 2 parents, 2 children

#     def _do(self, problem, X, **kwargs):
#         _, n_matings, n_var = X.shape
#         Y = np.full_like(X, False)

#         for i in range(n_matings):
#             p1, p2 = X[0, i, :], X[1, i, :]

#             # Convert binary parents to indices of critical points
#             indices1 = np.where(p1)[0]
#             indices2 = np.where(p2)[0]

#             # Perform crossover (blend point indices from both parents)
#             point = np.random.randint(1, min(len(indices1), len(indices2)))
#             child1_indices = np.sort(np.unique(np.concatenate((indices1[:point], indices2[point:]))))
#             child2_indices = np.sort(np.unique(np.concatenate((indices2[:point], indices1[point:]))))

#             # Create binary offspring
#             child1 = np.zeros(n_var, dtype=bool)
#             child2 = np.zeros(n_var, dtype=bool)
#             child1[child1_indices] = True
#             child2[child2_indices] = True

#             Y[0, i, :] = child1
#             Y[1, i, :] = child2

#         return Y

# class CriticalPointsMutation(Mutation):
#     def __init__(self, mutation_rate, min_critical_points, max_critical_points, x_values):
#         super().__init__()
#         self.mutation_rate = mutation_rate
#         self.min_critical_points = min_critical_points
#         self.max_critical_points = max_critical_points
#         self.x_values = x_values

#     def _do(self, problem, X, **kwargs):
#         for i in range(len(X)):
#             if np.random.rand() < self.mutation_rate:
#                 individual = np.where(X[i, :])[0]
#                 if len(individual) > self.min_critical_points and (
#                     len(individual) < self.max_critical_points and np.random.rand() > 0.5
#                     or len(individual) == self.max_critical_points
#                 ):
#                     idx = np.random.randint(len(individual))
#                     individual = np.delete(individual, idx)
#                 else:
#                     new_point = np.random.randint(1, len(self.x_values) - 1)
#                     individual = np.append(individual, new_point)
#                 binary_individual = np.zeros(len(self.x_values), dtype=bool)
#                 binary_individual[individual] = True
#                 X[i, :] = binary_individual
#         return X
# class CriticalPointsOptimization(Problem):
#     def __init__(self, x_values, y_values_dict, slopes_dict, method, min_critical_points, max_critical_points):
#         self.x_values = x_values
#         self.y_values_dict = y_values_dict
#         self.slopes_dict = slopes_dict
#         self.method = method
#         self.min_critical_points = min_critical_points
#         self.max_critical_points = max_critical_points
#         super().__init__(n_var=len(x_values), 
#                          n_obj=1, 
#                          xl=0, 
#                          xu=1, 
#                          type_var=np.bool_)

#     # def _evaluate(self, x, out, *args, **kwargs):
#     #     solutions = x.astype(bool)
#     #     fitnesses = []

#     #     for solution in solutions:
#     #         critical_points_indices = np.where(solution)[0]
#     #         fitness = self._fitness_function(critical_points_indices)
#     #         fitnesses.append(fitness)

#     #     out["F"] = np.array(fitnesses).reshape(-1, 1)

#     #parralel implememntation 
#     def _evaluate(self, x, out, *args, **kwargs):
#         solutions = x.astype(bool)

#         # Parallelize the computation of fitness values
#         fitnesses = Parallel(n_jobs=-1)(  # -1 means use all available CPU cores
#             delayed(self._fitness_function)(np.where(solution)[0]) for solution in solutions
#         )

#         # Store the computed fitnesses
#         out["F"] = np.array(fitnesses).reshape(-1, 1)

#     def _fitness_function(self, critical_points_indices):
#         if len(critical_points_indices) < self.min_critical_points:
#             return float('inf')

#         all_indices = [0] + list(critical_points_indices) + [len(self.x_values) - 1]
#         segments = [(all_indices[i], all_indices[i + 1]) for i in range(len(all_indices) - 1)]
#         total_rms_error = 0

#         for var_name in self.y_values_dict.keys():
#             y_values = np.array(self.y_values_dict[var_name], dtype=np.float64)
#             slopes = np.array(self.slopes_dict[var_name], dtype=np.float64)

#             if self.method == 'hermite':
#                 fitted_segments = compute_hermite_segments(self.x_values, y_values, slopes, segments)
#                 y_fitted = evaluate_hermite_segments(self.x_values, fitted_segments)
#             elif self.method == 'chebyshev':
#                 fitted_segments = compute_chebyshev_segments(self.x_values, y_values, segments, degree=3)
#                 y_fitted = evaluate_chebyshev_segments(self.x_values, self.x_values, fitted_segments, segments)

#             if np.any(np.isnan(y_fitted)):
#                 return float('inf')

#             rms_error = np.sqrt(mean_squared_error(y_values, y_fitted))
#             total_rms_error += rms_error

#         return total_rms_error / len(self.y_values_dict)

# class ElitistSurvival(Survival):
#     def __init__(self,n_survive):
#         super().__init__(filter_infeasible=True)
#         self.n_survive = n_survive
#         self.best_individual = None
#         self.best_fitness = np.inf

#     def _do(self, problem, pop, n_survive, **kwargs):
#         F = pop.get("F")

#         # Update best individual seen so far
#         for i, f in enumerate(F):
#             if f[0] < self.best_fitness:
#                 self.best_fitness = f[0]
#                 self.best_individual = pop[i]

#         # Sort population by fitness
#         I = np.argsort(F[:, 0])
#         survivors = pop[I[:self.n_survive]]

#         # Check if best individual is still in the survivors
#         is_elite_present = any(np.all(ind.X == self.best_individual.X) for ind in survivors)

#         # If not, replace the worst with the best
#         if not is_elite_present:
#             survivors[-1] = self.best_individual

#         return survivors

# class TrackGenerationDataCallback(Callback):
#     def __init__(self, x_values, y_values_dict, slopes_dict, method):
#         super().__init__()
#         self.generation = 0
#         self.x_values = x_values
#         self.y_values_dict = y_values_dict
#         self.slopes_dict = slopes_dict
#         self.method = method
#         self.generations_data = []  # Store data for each generation

#     def notify(self, algorithm):
#         self.generation += 1
#         # Get the individual with the best fitness
#         best_index = algorithm.pop.get("F").argmin()
#         best_solution = algorithm.pop.get("X")[best_index]

#         # Extract indices of critical points
#         critical_points_indices = np.where(best_solution.astype(bool))[0]

#         # Compute segments for visualization using `compute_hermite_segments`
#         fits = {}
#         for key, y_values in self.y_values_dict.items():
#             slopes = self.slopes_dict[key]
#             segments = [(critical_points_indices[i], critical_points_indices[i + 1]) 
#                         for i in range(len(critical_points_indices) - 1)]
            
#             # Compute the optimized Hermite segments
#             hermite_segments = compute_hermite_segments(self.x_values, y_values, slopes, segments, is_c1_continuous=True)
            
#             # Evaluate the fitted Hermite curve over the x-values
#             fitted_curve = evaluate_hermite_segments(self.x_values, hermite_segments)
#             fits[key] = fitted_curve

#         # Store generation data
#         self.generations_data.append({
#             'fitness': algorithm.pop.get('F').min(),
#             'critical_points': critical_points_indices.tolist(),
#             'fits': fits
#         })

#         # Print generation info (optional)
#         print(f"Generation {self.generation}: Best Fitness = {algorithm.pop.get('F').min():.8f}, "
#               f"Critical Points = {critical_points_indices.tolist()}")

# def genetic_algorithm_optimum_critical_points(x_values, y_values_dict, slopes_dict, initial_critical_points=None,
#                                               method='hermite', population_size=240, generations=100,
#                                               mutation_rate=0.4, min_critical_points=3, max_critical_points=5):
#     n_survive = population_size // 2
#     # Define the problem
#     problem = CriticalPointsOptimization(x_values, y_values_dict, slopes_dict, method, 
#                                          min_critical_points, max_critical_points)

#     # Configure the genetic algorithm
#     algorithm = GA(
#         pop_size=population_size,
#         sampling=CriticalPointsSampling(min_critical_points, max_critical_points, x_values, initial_critical_points),
#         crossover=CriticalPointsCrossover(),
#         mutation=CriticalPointsMutation(mutation_rate, min_critical_points, max_critical_points, x_values),
#         survival=ElitistSurvival(n_survive)  # Enable elitism
#     )

#     callback = TrackGenerationDataCallback(x_values, y_values_dict, slopes_dict, method)

#     res = minimize(problem,
#                 algorithm,
#                 ('n_gen', generations),
#                 verbose=True,
#                 callback=callback)


#     # Extract best solution
#     best_solution = res.X
#     best_indices = np.where(best_solution.astype(bool))[0]
#     best_rms_error = res.F[0]

#     # Return best results and the generation data
#     return best_rms_error, best_indices.tolist(), callback.generations_data
