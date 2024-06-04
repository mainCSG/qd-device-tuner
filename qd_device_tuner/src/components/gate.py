import numpy as np

class Gate:
    def __init__(self,
                 name: str,
                 source: int,
                 unit: str = "V",
                 bounds: tuple | float | int = (-np.inf, np.inf)) -> None:
        if isinstance(bounds, float) or isinstance(bounds, int):
            self.bounds = (-bounds, bounds)
        else:
            self.bounds = bounds
        self.name = name
        self.source = source
        self.unit = unit
    
    def set_voltage(self) -> None:
        pass

    def get_voltage(self) -> float:
        pass