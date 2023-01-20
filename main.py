import random
import subprocess
import os
import re
import csv
import time
import shutil
import itertools
import multiprocessing
from typing import List, Any, Tuple

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
    clause: tuple[int]

    def __init__(self, clause: tuple[int]):
        self.clause = clause

    def satisfied(self, assignment: tuple[bool]) -> bool:
        return any([
            not ((variable > 0) ^ assignment[abs(variable)]) for variable in self.clause
        ])

    def weight(self, weights: tuple[int], assignment: tuple[bool]) -> int:
        return sum([
            weights[abs(variable)] for variable in self.clause if assignment[abs(variable)]
        ])


# def test_clause_satisfiable():
#     assert not clause_satisfied([None, False, False, False], [1, 2, 3])
#     assert clause_satisfied([None, True, False, False], [1, 2, 3])
#     assert clause_satisfied([None, False, True, False], [1, 2, 3])
#     assert clause_satisfied([None, False, False, True], [1, 2, 3])
#     assert clause_satisfied([None, True, True, False], [1, 2, 3])
#     assert clause_satisfied([None, False, True, True], [1, 2, 3])
#     assert clause_satisfied([None, True, True, True], [1, 2, 3])
#     assert clause_satisfied([None, False, False, False], [1, -2, 3])
#     assert not clause_satisfied([None, False, True, False], [1, -2, 3])
#     assert clause_satisfied([None, False, True, True], [1, -2, 3])
#     assert not clause_satisfied([None, True, True, True], [-1, -2, -3])


# def test_clause_weight():
#     assert 1 == clause_weight([0, 1, 1, 1], [None, True, False, False], [1, 2, 3])
#     assert 1 == clause_weight([0, 1, 1, 1], [None, False, True, False], [1, 2, 3])
#     assert 1 == clause_weight([0, 1, 1, 1], [None, False, False, True], [1, 2, 3])
#     assert 2 == clause_weight([0, 1, 1, 1], [None, True, True, False], [1, 2, 3])
#     assert 2 == clause_weight([0, 1, 1, 1], [None, False, True, True], [1, 2, 3])
#     assert 3 == clause_weight([0, 1, 1, 1], [None, True, True, True], [1, 2, 3])
#     assert 8 == clause_weight([0, 4, 1, 3], [None, True, True, True], [1, 2, 3])
#     assert 2 == clause_weight([0, 1, 1, 1], [None, False, True, True], [-1, -2, 3])
#     # assert 6 == clause_weight([0, 2, 3, 4], [None, False, True, True], [-1, -2, 3])
#     # assert 9 == clause_weight([0, 2, 3, 4], [None, False, False, True], [-1, -2, 3])


class MaxWeightedCNF:
    file_path: str
    clauses_count: int = 0
    variables_count: int = 0
    clauses: tuple[Clause] = ()
    weights: tuple[int] = ()

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extract_content()

    def extract_content(self):
        with open(self.file_path) as file:
            for line in file:
                if line.startswith('c'):
                    continue
                if line.startswith('p'):
                    self.variables_count = int(line.split(' ')[2]) + 1
                    self.clauses_count = int(line.split(' ')[3])
                    continue
                if line.startswith('w'):
                    self.weights = (0,) + tuple([int(weight) for weight in line.split(' ')[1:-1]])
                    continue
                self.clauses = self.clauses + tuple([
                    Clause(tuple([int(literal) for literal in line.lstrip().rstrip().split(' ')[:-1]])),
                ])

    def satisfied(self, assignment: tuple[bool]) -> bool:
        return all([
            clause.satisfied(assignment) for clause in self.clauses
        ])

    def weight(self, assignment: tuple[bool]) -> int:
        return sum([
            clause.weight(self.weights, assignment) for clause in self.clauses
        ])


if __name__ == '__main__':
    # test_clause_satisfiable()
    # test_clause_weight()

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
                    mwsat = MaxWeightedCNF(file_path)
                    assignment = [True] * mwsat.variables_count

                    assignments = tuple(itertools.product([False, True], repeat=mwsat.variables_count))
                    satifiable_assignments = tuple([
                        assignment for assignment in assignments if mwsat.satisfied(assignment)
                    ])
                    mx = max([
                        mwsat.weight(assignment) for assignment in satifiable_assignments
                    ])
                    print(mx)
                    exit(0)
