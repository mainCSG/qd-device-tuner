from .components.ohmic import Ohmic
from .components.gate import Gate
from .components.drain import Drain
from .components.source import Source
from .components.barrier import Barrier
from .components.plunger import Plunger
from .components.top import Top

import qcodes as qc

class Device:
    def __init__(self,
                 Station: qc.Station,
                 name: str,
                 charge_carrier: str,
                 operation_mode: str) -> None:
        self.Station = Station
        self.name = name
        self.charge_carrier = charge_carrier
        self.operation_mode = operation_mode

        self.ohmics = {}
        self.barriers = {}
        self.plungers = {}
        self.tops = {}
        self.drains = {}

    def add_ohmic(self, ohmic: Ohmic) -> None:
        self.ohmics[ohmic.name] = ohmic
        setattr(self, ohmic.name, ohmic)

    def add_plunger(self, plunger: Plunger) -> None:
        self.plungers[plunger.name] = plunger
        setattr(self, plunger.name, plunger)

    def add_barrier(self, barrier: Barrier) -> None:
        self.barriers[barrier.name] = barrier
        setattr(self, barrier.name, barrier)

    def add_top(self, top: Top) -> None:
        self.tops[top.name] = top
        setattr(self, top.name, top)

    def ground(self) -> None:
        pass