import numpy as np
import qcodes as qc

class Gate:
    def __init__(self,
                 name: str,
                 source: qc.parameters.Parameter,
                 unit: str = "V",
                 step: float = 0.01,
                 bounds: tuple | float | int = (-np.inf, np.inf)) -> None:
        if isinstance(bounds, float) or isinstance(bounds, int):
            self.bounds = (-bounds, bounds)
        else:
            self.bounds = bounds
        self.name = name
        self.source = source
        self.step = step
        self.unit = unit
    
    def set_voltage(self, voltage: float) -> None:
        self.source(voltage)

    def get_voltage(self) -> float:
        return float(self.source())

    def print_info(self) -> str:
        print(vars(self))