from .ohmic import *

class Drain(Ohmic):
    def __init__(self,
                 name: str,
                 source: int,
                 scale: float,
                 offset: float,
                 unit: str,
                 bounds: tuple | float | int) -> None:
        super().__init__(name, source, scale, offset, unit, bounds)

    def get_current(self) -> float:
        pass