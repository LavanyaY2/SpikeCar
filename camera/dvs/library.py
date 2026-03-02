import ctypes
from pathlib import Path

import numpy as np


class StructTimeSurface(ctypes.Structure):
    _fields_ = [
                ("magnitude", ctypes.POINTER(ctypes.c_float)),
                ("tau", ctypes.c_float),
               ]


class Library:
    def __init__(self, width, height):
        so_path = Path(__file__).resolve().parent.parent / 'c' / 'dvs.so'
        self.lib = ctypes.CDLL(str(so_path))
        self.width = width
        self.height = height
        self.lib.initialize(width, height)

        self.timesurface_active = False

    def stop_timesurface(self):
        self.lib.stopTimesurface()
        self.timesurface_active = False

    def init_timesurface(self, timesurface_tau=0.1):
        self.lib.initializeTimesurface(ctypes.c_float(timesurface_tau))
        self.struct_ts = ctypes.POINTER(StructTimeSurface).in_dll(self.lib, "ts")
        self.timesurface = np.ctypeslib.as_array(self.struct_ts.contents.magnitude, shape=(self.height, self.width))
        self.timesurface_active = True

    def __delete__(self):
        self.lib.shutdown()

    def process_events(self, t, x, y, p):
        N = len(t)
        self.lib.processEvents(N,
                t.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                p.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                )
