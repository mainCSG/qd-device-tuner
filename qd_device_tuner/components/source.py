from .ohmic import *

class Source(Ohmic):
    def __init__(self,
                 name: str,
                 source: int,
                 scale: float,
                 step: float,
                 unit: str = None,
                 bounds: tuple | float = None) -> None:
        super().__init__(name, source, scale, 0, unit, bounds)
        self.step = step

    def get_device_voltage(self) -> float:
        self.source() * self.scale

    def get_voltage(self) -> float:
        self.source()

    def set_voltage(self, voltage: float) -> None:
        self.source(voltage)
