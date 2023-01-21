from .clause import Clause


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
