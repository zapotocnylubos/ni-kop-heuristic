class Clause:
    clause: tuple[int]

    def __init__(self, clause: tuple[int]):
        self.clause = clause

    def satisfied(self, assignment: tuple[bool]) -> bool:
        return any([
            not ((literal > 0) ^ assignment[abs(literal)]) for literal in self.clause
        ])
