import numpy as np

class Ohmic:
    def __init__(self,
                 name: str,
                 source: int,
                 scale: float,
                 offset: float = 0.0,
                 unit: str = "V",
                 bounds: tuple | float | int = (-np.inf, np.inf)) -> None:
        if isinstance(bounds, float) or isinstance(bounds, int):
            self.bounds = (-bounds, bounds)
        else:
            self.bounds = bounds
        self.name = name
        self.source = source
        self.scale = scale
        self.offset = offset
        self.unit = unit