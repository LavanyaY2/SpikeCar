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

        # Adding raw data buffer stuff for later use
        # Declare get_event_buffer return/arg types
        self.lib.get_event_buffer.restype = ctypes.c_uint32
        self.lib.get_event_buffer.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int8)),
        ]

    def stop_timesurface(self):
        self.lib.stopTimesurface()
        self.timesurface_active = False

    def init_timesurface(self, timesurface_tau=0.1):
        self.lib.initializeTimesurface(ctypes.c_float(timesurface_tau))
        self.struct_ts = ctypes.POINTER(StructTimeSurface).in_dll(self.lib, "ts")
        self.timesurface = np.ctypeslib.as_array(self.struct_ts.contents.magnitude, shape=(self.height, self.width))
        self.timesurface_active = True

    def get_recent_events(self):
        """
        Flush the C-side ring buffer and return raw events as numpy arrays.
        Returns None if the buffer is empty.
        """
        p_t = ctypes.POINTER(ctypes.c_int32)()
        p_x = ctypes.POINTER(ctypes.c_uint16)()
        p_y = ctypes.POINTER(ctypes.c_uint16)()
        p_p = ctypes.POINTER(ctypes.c_int8)()

        count = self.lib.get_event_buffer(
            ctypes.byref(p_t),
            ctypes.byref(p_x),
            ctypes.byref(p_y),
            ctypes.byref(p_p),
        )

        if count == 0:
            return None

        return {
            't': np.ctypeslib.as_array(p_t, shape=(count,)).copy(),
            'x': np.ctypeslib.as_array(p_x, shape=(count,)).copy(),
            'y': np.ctypeslib.as_array(p_y, shape=(count,)).copy(),
            'p': np.ctypeslib.as_array(p_p, shape=(count,)).copy(),
        }

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
