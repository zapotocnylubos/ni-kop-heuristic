from typing import Dict


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

    def optimal_weight(self, instance: str) -> int:
        return self.optimum[instance]
