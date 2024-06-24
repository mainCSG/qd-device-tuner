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
                 name: str,
                 charge_carrier: str,
                 operation_mode: str,
                 minimum_current_threshold: float,
                 maximum_current_threshold: float) -> None:

        self.name = name
        self.charge_carrier = charge_carrier
        self.operation_mode = operation_mode

        if (self.charge_carrier, self.operation_mode) == ('e', 'acc'):
            self.polarity = +1
        if (self.charge_carrier, self.operation_mode) == ('e', 'dep'):
            self.polarity = -1
        if (self.charge_carrier, self.operation_mode) == ('h', 'acc'):
            self.polarity = -1
        if (self.charge_carrier, self.operation_mode) == ('h', 'dep'):
            self.polarity = +1

        self.sources = {}
        self.barriers = {}
        self.plungers = {}
        self.tops = {}
        self.drains = {}

        self.turn_on_voltage = None
        self.saturation_voltage = None
        self.saturation_current = None
        self.turns_on = None

        self.minimum_current_threshold = minimum_current_threshold
        self.maximum_current_threshold = maximum_current_threshold

    def add_source(self, source: Source) -> None:
        self.sources[source.name] = source
        setattr(self, source.name, source)

    def add_plunger(self, plunger: Plunger) -> None:
        self.plungers[plunger.name] = plunger
        setattr(self, plunger.name, plunger)

    def add_barrier(self, barrier: Barrier) -> None:
        self.barriers[barrier.name] = barrier
        setattr(self, barrier.name, barrier)

    def add_top(self, top: Top) -> None:
        self.tops[top.name] = top
        setattr(self, top.name, top)

    def add_drain(self, drain: Drain) -> None:
        self.drains[drain.name] = drain
        setattr(self, drain.name, drain)