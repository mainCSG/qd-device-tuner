from .ohmic import *

class Source(Ohmic):
    def __init__(self,
                 name: str,
                 source: int,
                 scale: float,
                 unit: str,
                 bounds: tuple | float) -> None:
        super().__init__(name, source, scale, unit, bounds)