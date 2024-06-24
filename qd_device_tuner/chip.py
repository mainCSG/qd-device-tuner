from .device import Device

class Chip:
    def __init__(self, name: str) -> None:
        self.name = name
        self.devices = {}

    def add_device(self, device: Device) -> None:
        self.devices[device.name] = device
        setattr(self, device.name, device)