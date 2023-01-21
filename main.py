import csv
import multiprocessing
import os
import random
import shutil
import time
from typing import Dict

import numpy as np
from tqdm import tqdm

np.seterr(over='ignore')

DATA_SUITES = (
    'wuf20-71',
    'wuf20-71R',
    'wuf20-91',
    'wuf20-91R',
    'wuf50-200',
    'wuf50-218',
    'wuf50-218R',
    'wuf75-325',
    'wuf100-430',
)

DATA_SUITES_VARIATIONS = (
    'M',
    'N',
    'Q',
    'R',
)

DATA_DIRECTORY = 'data'
OUTPUT_DIRECTORY = 'output'
BACKUP_DIRECTORY = 'backup'

DEBUG = True
RUNS = 1


def flatten(list):
    return [item for sublist in list for item in sublist]


class Clause:
    clause: tuple[int]

    def __init__(self, clause: tuple[int]):
        self.clause = clause

    def satisfied(self, assignment: tuple[bool]) -> bool:
        return any([
            not ((literal > 0) ^ assignment[abs(literal)]) for literal in self.clause
        ])


class MaxWeightedCNF:
    file_path: str
    clauses_count: int
    variables_count: int
    clauses: list[Clause]
    weights: tuple[int]

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.clauses_count = 0
        self.variables_count = 0
        self.clauses = []
        self.weights = tuple()
        self.extract_content()

    def extract_content(self):
        with open(self.file_path) as file:
            for line in file:
                if line.startswith('c'):
                    continue
                if line.startswith('p'):
                    self.variables_count = int(line.split(' ')[2]) + 1  # +1 for x0 (unused)
                    self.clauses_count = int(line.split(' ')[3])
                    continue
                if line.startswith('w'):
                    self.weights = tuple([0] + [int(weight) for weight in line.split(' ')[1:-1]])
                    continue
                self.clauses.append(
                    Clause(tuple([int(literal) for literal in line.lstrip().rstrip().split(' ')[:-1]]))
                )

    def satisfied_clauses(self, assignment: tuple[bool]) -> int:
        return sum([
            clause.satisfied(assignment) for clause in self.clauses
        ])

    def unsatisfied_clauses(self, assignment: tuple[bool]) -> int:
        return self.clauses_count - self.satisfied_clauses(assignment)

    def satisfied(self, assignment: tuple[bool]) -> bool:
        return self.satisfied_clauses(assignment) == self.clauses_count

    def weight(self, assignment: tuple[bool]) -> int:
        if len(assignment) != self.variables_count:
            raise ValueError(f'Invalid count of assignment variables ({len(assignment)} != {self.variables_count})')

        return sum([
            self.weights[index] for index, value in enumerate(assignment) if value
        ])


class Optimum:
    optimum: Dict[str, int] = {}

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extract_content()

    def extract_content(self) -> None:
        with open(self.file_path) as file:
            for line in file:
                line_split = line.split(' ')
                instance_name = line_split[0]
                weight = int(line_split[1])
                self.optimum[instance_name] = weight

    def get_optimal_weight(self, instance: str) -> int:
        return self.optimum[instance]


class SimulatedAnnealing:
    initial_temperature: int
    final_temperature: int
    num_iterations_per_temperature: int
    perturbation_flips: int
    cooling_factor: float

    def __init__(self,
                 initial_temperature: int = 5000,
                 final_temperature: int = 1,
                 num_iterations_per_temperature: int = 10,
                 perturbation_flips: int = 1,
                 cooling_factor: float = .995):

        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.num_iterations_per_temperature = num_iterations_per_temperature
        self.perturbation_flips = perturbation_flips
        self.cooling_factor = cooling_factor

    @staticmethod
    def random_assignment(mwcnf: MaxWeightedCNF) -> tuple[bool]:
        return tuple(random.choices([True, False], k=mwcnf.variables_count))

    @staticmethod
    def objective_function(mwcnf: MaxWeightedCNF, assignment: tuple[bool]):
        return sum([
            mwcnf.weights[variable] for variable in assignment if variable
        ])

    def perturb_solution(self, assignment: tuple[bool]) -> tuple[bool]:
        assignment = list(assignment)
        for _ in range(self.perturbation_flips):
            variable = random.randint(0, len(assignment) - 1)
            assignment[variable] = not assignment[variable]
        return tuple(assignment)

    @staticmethod
    def probability(delta: int, temperature: int):
        return np.exp(-delta / temperature)

    def cool_temperature(self, temperature: int):
        return self.cooling_factor * temperature

    def run(self, mwcnf: MaxWeightedCNF, record_history=False):
        current_solution = self.random_assignment(mwcnf)
        current_objective = self.objective_function(mwcnf, current_solution)
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

                new_objective = self.objective_function(mwcnf, new_solution)
                delta = new_objective - best_objective

                if delta > 0 or self.probability(delta, temperature) > random.uniform(0, 1):
                    current_solution = new_solution
                    current_objective = new_objective

                    if record_history:
                        current_objective_history.append(current_objective)

                    if current_objective > best_objective:
                        best_solution = current_solution
                        best_objective = current_objective

                        if record_history:
                            best_objective_history.append(best_objective)

            temperature = self.cool_temperature(temperature)

        return \
            mwcnf.file_path, \
            mwcnf.weight(best_solution), \
            mwcnf.satisfied_clauses(best_solution), \
            mwcnf.clauses_count, \
            len([variable for variable in best_solution if variable]), \
            mwcnf.variables_count, \
            current_objective_history, \
            best_objective_history


def run_sa(mwcnf: MaxWeightedCNF):
    return SimulatedAnnealing().run(mwcnf)


if __name__ == '__main__':
    if not os.path.exists(BACKUP_DIRECTORY):
        os.mkdir(BACKUP_DIRECTORY)
    BACKUP_TIMESTAMP = time.time().__str__()

    if os.path.exists(OUTPUT_DIRECTORY):
        shutil.copytree(OUTPUT_DIRECTORY, os.path.join(BACKUP_DIRECTORY, BACKUP_TIMESTAMP))
        shutil.rmtree(OUTPUT_DIRECTORY)
    os.mkdir(OUTPUT_DIRECTORY)

    for suite in DATA_SUITES:
        os.mkdir(os.path.join(OUTPUT_DIRECTORY, suite))

        for suite_variation in DATA_SUITES_VARIATIONS:
            print(suite, suite_variation)

            os.mkdir(os.path.join(OUTPUT_DIRECTORY, suite, suite_variation))
            sa_output = open(os.path.join(OUTPUT_DIRECTORY, suite, suite_variation, 'sa.csv'), 'w')

            sa_heading = ['instance', 'weight', 'satisfied_clauses', 'clauses',
                          'true_variables', 'variables', 'current_objective_history', 'best_objective_history']
            sa_writer = csv.writer(sa_output)

            sa_writer.writerow(sa_heading)

            optimum = None
            optimum_path = os.path.join(DATA_DIRECTORY, suite, f'{suite}-{suite_variation}-opt.dat')

            if os.path.exists(optimum_path):
                optimum = Optimum(optimum_path)

            for root, _, files in os.walk(os.path.join(DATA_DIRECTORY, suite, f'{suite}-{suite_variation}')):
                run_mwcnfs = flatten([[MaxWeightedCNF(os.path.join(root, file))] * RUNS for file in files])
                if DEBUG:
                    run_mwcnfs = [run_mwcnfs[0]]
                pool = multiprocessing.Pool(processes=8)

                sa_results = list(
                    tqdm(pool.imap(run_sa, run_mwcnfs), total=len(run_mwcnfs)))

                sa_results = [(result[0].split('/')[-1],
                               result[1], result[2], result[3], result[4],
                               result[5], result[6], result[7]) for result in sa_results]

                sa_results.sort(key=lambda result: result[0])

                sa_writer.writerows(sa_results)
