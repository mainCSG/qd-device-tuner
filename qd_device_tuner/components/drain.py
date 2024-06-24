from .ohmic import *

class Drain(Ohmic):
    def __init__(self,
                 name: str,
                 source: int,
                 scale: float,
                 offset: float,
                 unit: str = None,
                 bounds: tuple | float | int = None) -> None:
        super().__init__(name, source, scale, offset, unit, bounds)

    def get_current(self) -> float:
        return float(self.source() - self.offset) * self.scale