from ..utils.acquisition import *
from ..utils.analysis import *
from ..device import *

class LinearDotArray(Analysis, Acquisition):
    def __init__(self, device: Device) -> None:
        super().__init__()