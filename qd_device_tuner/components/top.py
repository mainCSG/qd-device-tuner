from .gate import *

class Top(Gate):
    def __init__(self,  
                 name: str,
                 source: int,
                 step: float,
                 unit: str = None,
                 bounds: tuple | float = None) -> None:
        super().__init__(name, source, unit, step, bounds)

        self.pinch_off_voltage = None