"""Read amdxdna clock/power metadata via DRM_IOCTL_AMDXDNA_GET_INFO."""

import ctypes
import fcntl
import os

DRM_COMMAND_BASE = 0x40
AMDXDNA_GET_INFO = 7
DRM_IOCTL_GET_INFO = ((3 << 30) | (16 << 16) | (0x64 << 8) |
                      (DRM_COMMAND_BASE + AMDXDNA_GET_INFO))

CLOCK_METADATA = 3
SENSORS = 4
POWER_MODE = 9
RESOURCE_INFO = 12


class GetInfo(ctypes.Structure):
    _fields_ = [("param", ctypes.c_uint32),
                ("buffer_size", ctypes.c_uint32),
                ("buffer", ctypes.c_uint64)]


class Clock(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 16),
                ("freq_mhz", ctypes.c_uint32),
                ("pad", ctypes.c_uint32)]


class ClockMeta(ctypes.Structure):
    _fields_ = [("mp_npu_clock", Clock), ("h_clock", Clock)]


class Sensor(ctypes.Structure):
    _fields_ = [("label", ctypes.c_char * 64),
                ("input", ctypes.c_uint32),
                ("max", ctypes.c_uint32),
                ("average", ctypes.c_uint32),
                ("highest", ctypes.c_uint32),
                ("status", ctypes.c_char * 64),
                ("units", ctypes.c_char * 16),
                ("unitm", ctypes.c_int8),
                ("type", ctypes.c_uint8),
                ("pad", ctypes.c_uint8 * 6)]


class Resource(ctypes.Structure):
    _fields_ = [("npu_clk_max", ctypes.c_uint64),
                ("npu_tops_max", ctypes.c_uint64),
                ("npu_task_max", ctypes.c_uint64),
                ("npu_tops_curr", ctypes.c_uint64),
                ("npu_task_curr", ctypes.c_uint64)]


class PowerMode(ctypes.Structure):
    _fields_ = [("power_mode", ctypes.c_uint8), ("pad", ctypes.c_uint8 * 7)]


class Xdna:
    def __init__(self, path="/dev/accel/accel0"):
        self.fd = os.open(path, os.O_RDWR)

    def close(self):
        os.close(self.fd)

    def _info(self, param, payload):
        info = GetInfo(param, ctypes.sizeof(payload), ctypes.addressof(payload))
        buf = bytearray(ctypes.string_at(ctypes.byref(info),
                                         ctypes.sizeof(info)))
        fcntl.ioctl(self.fd, DRM_IOCTL_GET_INFO, buf, True)

    def clocks(self):
        payload = ClockMeta()
        self._info(CLOCK_METADATA, payload)
        return (payload.mp_npu_clock.freq_mhz, payload.h_clock.freq_mhz)

    def power_mw(self):
        payload = (Sensor * 16)()
        self._info(SENSORS, payload)
        for sensor in payload:
            label = sensor.label.decode(errors="replace").strip("\x00")
            if label == "Total Power":
                return sensor.input
        return None

    def power_mode(self):
        payload = PowerMode()
        self._info(POWER_MODE, payload)
        return payload.power_mode

    def resource(self):
        payload = Resource()
        self._info(RESOURCE_INFO, payload)
        return payload


if __name__ == "__main__":
    x = Xdna()
    mp, h = x.clocks()
    r = x.resource()
    print(f"mp_npu={mp} MHz h={h} MHz clk_max={r.npu_clk_max} "
          f"tops={r.npu_tops_curr}/{r.npu_tops_max} "
          f"power={x.power_mw()} mW pmode={x.power_mode()}")
    x.close()
