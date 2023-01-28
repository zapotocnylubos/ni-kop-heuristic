import random
import math

from .max_weighted_cnf import MaxWeightedCNF


class SimulatedAnnealing:
    mwcnf: MaxWeightedCNF
    initial_temperature: int
    final_temperature: int
    num_iterations_per_temperature: int
    perturbation_flips: int
    cooling_factor: float
    penalty: int

    def __init__(self,
                 mwcnf: MaxWeightedCNF,
                 initial_temperature: int = 5000,
                 final_temperature: int = 1,
                 num_iterations_per_temperature: int = 10,
                 perturbation_flips: int = 1,
                 cooling_factor: float = .975,
                 penalty: int = -5000):
        self.mwcnf = mwcnf
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.num_iterations_per_temperature = num_iterations_per_temperature
        self.perturbation_flips = perturbation_flips
        self.cooling_factor = cooling_factor
        self.penalty = penalty

        if not penalty:
            self.penalty = -int(sum(self.mwcnf.weights) / len(self.mwcnf.weights))

    def random_assignment(self) -> tuple[bool]:
        return tuple(random.choices([True, False], k=self.mwcnf.variables_count))

    def objective_function(self, assignment: tuple[bool]):
        unsatisfied_clauses = self.mwcnf.unsatisfied_clauses(assignment)

        penalization = (1 - (unsatisfied_clauses / self.mwcnf.clauses_count)) * (unsatisfied_clauses * self.penalty)
        penalization *= self.mwcnf.variables_count / 8

        return sum([
            self.mwcnf.weights[index] for index, variable in enumerate(assignment) if variable
        ]) + penalization

    def perturb_solution(self, assignment: tuple[bool]) -> tuple[bool]:
        assignment = list(assignment)
        for _ in range(self.perturbation_flips):
            variable = random.randint(0, len(assignment) - 1)
            assignment[variable] = not assignment[variable]
        return tuple(assignment)

    @staticmethod
    def probability(delta: int, temperature: int):
        exponent = -delta / temperature
        if exponent > 1:
            return 1
        return math.exp(exponent)

    def cool_temperature(self, temperature: int):
        return self.cooling_factor * temperature

    def run(self, record_history=False):
        current_solution = self.random_assignment()
        current_objective = self.objective_function(current_solution)
        current_objective_history = []

        if record_history:
            current_objective_history.append(current_objective)

        best_solution = current_solution
        best_objective = current_objective
        best_objective_history = []

        if record_history:
            best_objective_history.append(best_objective)

        temperature = self.initial_temperature

        while temperature > self.final_temperature:
            for _ in range(self.num_iterations_per_temperature):
                new_solution = self.perturb_solution(current_solution)

                new_objective = self.objective_function(new_solution)
                delta = new_objective - current_objective

                if delta > 0 or self.probability(abs(delta), temperature) > random.uniform(0, 1):
                    current_solution = new_solution
                    current_objective = new_objective

                    if current_objective > best_objective:
                        best_solution = current_solution
                        best_objective = current_objective

            temperature = self.cool_temperature(temperature)

        return \
            self.mwcnf.file_path, \
            self.mwcnf.weight(best_solution), \
            self.mwcnf.satisfied_clauses(best_solution), \
            self.mwcnf.clauses_count, \
            len([variable for variable in best_solution if variable]), \
            self.mwcnf.variables_count, \
            current_objective_history, \
            best_objective_history


def run_simulated_annealing(sa: SimulatedAnnealing):
    return sa.run()
