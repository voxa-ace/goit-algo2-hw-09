import random
import math

# Sphere function definition
def sphere_function(x):
    """
    Computes the value of the Sphere function for a given input vector x.
    :param x: List of coordinates.
    :return: Value of the Sphere function.
    """
    return sum(xi ** 2 for xi in x)

# Hill Climbing
def hill_climbing(func, bounds, iterations=1000, epsilon=1e-6):
    """
    Hill Climbing algorithm to minimize a given function.
    :param func: Objective function to minimize.
    :param bounds: List of tuples defining variable bounds.
    :param iterations: Maximum number of iterations.
    :param epsilon: Convergence threshold.
    :return: Best solution and its function value.
    """
    current_solution = [random.uniform(b[0], b[1]) for b in bounds]
    current_value = func(current_solution)

    for _ in range(iterations):
        next_solution = [
            max(min(current_solution[i] + random.uniform(-0.1, 0.1), bounds[i][1]), bounds[i][0])
            for i in range(len(bounds))
        ]
        next_value = func(next_solution)

        if abs(next_value - current_value) < epsilon:
            break

        if next_value < current_value:
            current_solution, current_value = next_solution, next_value

    return current_solution, current_value

# Random Local Search
def random_local_search(func, bounds, iterations=1000, epsilon=1e-6):
    """
    Random Local Search algorithm to minimize a given function.
    :param func: Objective function to minimize.
    :param bounds: List of tuples defining variable bounds.
    :param iterations: Maximum number of iterations.
    :param epsilon: Convergence threshold.
    :return: Best solution and its function value.
    """
    best_solution = [random.uniform(b[0], b[1]) for b in bounds]
    best_value = func(best_solution)

    for _ in range(iterations):
        candidate_solution = [random.uniform(b[0], b[1]) for b in bounds]
        candidate_value = func(candidate_solution)

        if abs(candidate_value - best_value) < epsilon:
            break

        if candidate_value < best_value:
            best_solution, best_value = candidate_solution, candidate_value

    return best_solution, best_value

# Simulated Annealing
def simulated_annealing(func, bounds, iterations=1000, temp=1000, cooling_rate=0.95, epsilon=1e-6):
    """
    Simulated Annealing algorithm to minimize a given function.
    :param func: Objective function to minimize.
    :param bounds: List of tuples defining variable bounds.
    :param iterations: Maximum number of iterations.
    :param temp: Initial temperature.
    :param cooling_rate: Rate at which the temperature decreases.
    :param epsilon: Convergence threshold.
    :return: Best solution and its function value.
    """
    current_solution = [random.uniform(b[0], b[1]) for b in bounds]
    current_value = func(current_solution)
    best_solution, best_value = current_solution, current_value

    for _ in range(iterations):
        next_solution = [
            max(min(current_solution[i] + random.uniform(-1, 1), bounds[i][1]), bounds[i][0])
            for i in range(len(bounds))
        ]
        next_value = func(next_solution)

        if abs(next_value - current_value) < epsilon or temp < epsilon:
            break

        delta = next_value - current_value
        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temp):
            current_solution, current_value = next_solution, next_value

        if current_value < best_value:
            best_solution, best_value = current_solution, current_value

        temp *= cooling_rate

    return best_solution, best_value

if __name__ == "__main__":
    # Function bounds
    bounds = [(-5, 5), (-5, 5)]

    # Execute algorithms
    print("Hill Climbing:")
    hc_solution, hc_value = hill_climbing(sphere_function, bounds)
    print("Solution:", hc_solution, "Value:", hc_value)

    print("\nRandom Local Search:")
    rls_solution, rls_value = random_local_search(sphere_function, bounds)
    print("Solution:", rls_solution, "Value:", rls_value)

    print("\nSimulated Annealing:")
    sa_solution, sa_value = simulated_annealing(sphere_function, bounds)
    print("Solution:", sa_solution, "Value:", sa_value)
