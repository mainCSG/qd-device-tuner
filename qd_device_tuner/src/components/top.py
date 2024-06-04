from .gate import *

class Top(Gate):
    def __init__(self,  
                 name: str,
                 source: int,
                 unit: str,
                 bounds: tuple | float) -> None:
        super().__init__(name, source, unit, bounds)