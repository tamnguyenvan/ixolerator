from dataclasses import dataclass


@dataclass
class Metric:
    map50: float = 0
    map: float = 0
    recall: float = 0
    precision: float = 0
    f1: float = 0
