from .gate import *

class Barrier(Gate):
    def __init__(self,
                 name: str,
                 source: int,
                 step: float,
                 unit: str = None,
                 bounds: tuple | float | int = None) -> None:
        super().__init__(name, source, unit, step, bounds)

        self.pinch_off_voltage = None
        self.pinch_off_width = None