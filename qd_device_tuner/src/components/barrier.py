from .gate import *

class Barrier(Gate):
    def __init__(self,
                 name: str,
                 source: int,
                 unit: str,
                 bounds: tuple | float | int) -> None:
        super().__init__(name, source, unit, bounds)