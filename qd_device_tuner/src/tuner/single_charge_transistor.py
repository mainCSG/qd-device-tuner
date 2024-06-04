from ..utils.acquisition import *
from ..utils.analysis import *
from ..device import *

import pathlib
from pathlib import Path

class SingleChargeTransistor:
    def __init__(self, device: Device, save_dir: Path) -> None:\
        
        self.acquisition = Acquisition(device, save_dir)
        self.analysis = Analysis()