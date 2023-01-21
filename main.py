import random
import subprocess
import os
import re
import csv
import time
import shutil
import math
import itertools
import multiprocessing
import random
import numpy as np

from tqdm import tqdm

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


class Clause:
    clause: list[int]

    def __init__(self, clause: list[int]):
        self.clause = clause

    def satisfied(self, assignment: list[bool]) -> bool:
        return any([
            not ((literal > 0) ^ assignment[abs(literal)]) for literal in self.clause
        ])


class MaxWeightedCNF:
    file_path: str
    clauses_count: int = 0
    variables_count: int = 0
    clauses: list[Clause] = []
    weights: list[int] = []

    def __init__(self, file_path: str):
        self.file_path = file_path
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
                    self.weights = [0] + [int(weight) for weight in line.split(' ')[1:-1]]
                    continue
                self.clauses.append(
                    Clause([int(literal) for literal in line.lstrip().rstrip().split(' ')[:-1]])
                )

    def satisfied(self, assignment: list[bool]) -> bool:
        return all([
            clause.satisfied(assignment) for clause in self.clauses
        ])

    def weight(self, assignment: list[bool]) -> int:
        if len(assignment) != self.variables_count:
            raise ValueError(f'Invalid count of assignment variables ({len(assignment)} != {self.variables_count})')

        return sum([
            self.weights[index] for index, value in enumerate(assignment) if value
        ])


class SimulatedAnnealing:
    initial_temperature: int
    final_temperature: int
    num_iterations_per_temperature: int
    cooling_factor: float

    def __init__(self,
                 initial_temperature: int = 1000,
                 final_temperature: int = 1,
                 num_iterations_per_temperature: int = 10,
                 cooling_factor: float = .995):

        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.num_iterations_per_temperature = num_iterations_per_temperature
        self.cooling_factor = cooling_factor

    def random_assignment(self, mwcnf: MaxWeightedCNF):
        return random.choices([True, False], k=mwcnf.variables_count)

    def perturb_solution(self, assignment: list[bool]):
        random_index = random.randint(0, len(assignment) - 1)
        assignment[random_index] = not assignment[random_index]
        return assignment

    def probability(self, delta: int, temperature: int):
        return np.exp(-delta / temperature)

    def cool_temperature(self, temperature: int):
        return self.cooling_factor * temperature

    def run(self, mwcnf: MaxWeightedCNF):
        current_solution = self.random_assignment(mwcnf)
        current_weight = mwcnf.weight(current_solution)
        best_solution = current_solution
        best_weight = current_weight
        temperature = self.initial_temperature

        while temperature > self.final_temperature:
            for i in range(self.num_iterations_per_temperature):
                new_solution = self.perturb_solution(current_solution)
                new_weight = mwcnf.weight(new_solution)
                delta = new_weight - current_weight
                if delta > 0 or self.probability(delta, temperature) > random.uniform(0, 1):
                    current_solution = new_solution
                    current_weight = new_weight
                    if current_weight > best_weight:
                        best_solution = current_solution
                        best_weight = current_weight
            temperature = self.cool_temperature(temperature)

        return best_solution


if __name__ == '__main__':
    if not os.path.exists(BACKUP_DIRECTORY):
        os.mkdir(BACKUP_DIRECTORY)
    BACKUP_TIMESTAMP = time.time().__str__()

    if os.path.exists(OUTPUT_DIRECTORY):
        shutil.copytree(OUTPUT_DIRECTORY, os.path.join(BACKUP_DIRECTORY, BACKUP_TIMESTAMP))
        shutil.rmtree(OUTPUT_DIRECTORY)
    os.mkdir(OUTPUT_DIRECTORY)

    for suite in DATA_SUITES:
        for suite_variation in DATA_SUITES_VARIATIONS:
            print(suite, suite_variation)

            for root, _, files in os.walk(os.path.join(DATA_DIRECTORY, suite, f'{suite}-{suite_variation}')):
                for file in files:
                    file_path = os.path.join(root, file)
                    mwcnf = MaxWeightedCNF(file_path)

                    sa = SimulatedAnnealing()
                    solution = sa.run(mwcnf)
                    mx = mwcnf.weight(solution)

                    print(mx)
                    exit(0)
