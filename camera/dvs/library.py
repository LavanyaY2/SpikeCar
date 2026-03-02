import ctypes
from pathlib import Path

import numpy as np


class BaselineState(ctypes.Structure):
    _fields_ = [
        ("accum_u16", ctypes.POINTER(ctypes.c_uint16)),
        ("image_u8", ctypes.POINTER(ctypes.c_uint8)),
        ("last_event_count", ctypes.c_uint32),
        ("tick_count", ctypes.c_uint32),
        ("gain", ctypes.c_float),
        ("decay", ctypes.c_float),
        ("dt_s", ctypes.c_float),
    ]


class Library:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        so_path = Path(__file__).resolve().parent.parent / "c" / "dvs.so"
        self.lib = ctypes.CDLL(str(so_path))

        self.lib.initialize.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        self.lib.initialize(width, height)

        self.lib.initializeBaseline.argtypes = [ctypes.c_float, ctypes.c_float]
        self.lib.stopBaseline.argtypes = []
        self.lib.baselineSetParams.argtypes = [ctypes.c_float, ctypes.c_float]
        self.lib.baselineTick.argtypes = [ctypes.c_float]

        self.lib.initialize_camera.restype = ctypes.c_int
        self.lib.start_camera.argtypes = []
        self.lib.update_camera.restype = ctypes.c_int
        self.lib.shutdown.argtypes = []

        self.active = False
        self.state_ptr = None
        self.state = None

        self.accum_raw = None
        self.image = None

    def start_baseline(self, gain=6.0, decay=0.0):
        if self.active:
            return

        self.lib.initializeBaseline(ctypes.c_float(gain), ctypes.c_float(decay))

        self.state_ptr = ctypes.POINTER(BaselineState).in_dll(self.lib, "baseline")
        self.state = self.state_ptr.contents

        self.accum_raw = np.ctypeslib.as_array(self.state.accum_u16, shape=(self.height, self.width))
        self.image = np.ctypeslib.as_array(self.state.image_u8, shape=(self.height, self.width))

        self.active = True

    def stop_baseline(self):
        if not self.active:
            return
        self.lib.stopBaseline()
        self.active = False
        self.state_ptr = None
        self.state = None
        self.accum_raw = None
        self.image = None

    def set_params(self, gain=None, decay=None):
        if not self.active:
            return
        s = self.state_ptr.contents
        if gain is None:
            gain = float(s.gain)
        if decay is None:
            decay = float(s.decay)
        self.lib.baselineSetParams(ctypes.c_float(gain), ctypes.c_float(decay))

    def tick(self, dt_s: float):
        if not self.active:
            return
        self.lib.baselineTick(ctypes.c_float(dt_s))

    def __del__(self):
        try:
            self.lib.shutdown()
        except Exception:
            pass
