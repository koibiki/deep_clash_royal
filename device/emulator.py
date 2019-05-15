from device.device import Device


class Emulator(Device):

    def __init__(self, device_id):
        super().__init__(device_id)
        self.offset_w = 0
        self.offset_h = 0
        self.scale = 0.5

    def tap_button(self, button):
        pass

    def swipe(self, action):
        pass
