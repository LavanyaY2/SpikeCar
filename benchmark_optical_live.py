#!/usr/bin/env python3
import csv
import ctypes
import math
import time
import numpy as np


def main():
    out_csv = "optical_live_latency.csv"
    n_windows = 200
    tick_ms = 50

    lib = ctypes.CDLL("./libbaseline.so")

    # camera
    lib.initialize_camera.restype = ctypes.c_int
    lib.start_camera.restype = None
    lib.stop_camera.restype = None
    lib.close_camera.restype = None
    lib.shutdown.restype = None

    # baseline init
    lib.initialize.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
    lib.initializeBaseline.argtypes = [
        ctypes.c_float,   # gain
        ctypes.c_float,   # decay
        ctypes.c_uint8,   # thresh
        ctypes.c_uint32,  # area_min
        ctypes.c_float,   # match_radius
        ctypes.c_float,   # ema_alpha
        ctypes.c_uint8,   # growth_needed
        ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16
    ]
    lib.baselineTick.argtypes = [ctypes.c_float]
    lib.baselineTick.restype = None

    lib.baselineGetCriticalTTC.restype = ctypes.c_float
    lib.baselineGetLastEventCount.restype = ctypes.c_uint32

    lib.update_camera.restype = ctypes.c_int

    # init
    lib.initialize(640, 480)
    rc = lib.initialize_camera()
    if rc != 0:
        raise RuntimeError("initialize_camera failed")

    lib.initializeBaseline(
        ctypes.c_float(20.0),   # gain
        ctypes.c_float(0.0),    # decay
        ctypes.c_uint8(20),     # thresh
        ctypes.c_uint32(220),   # area_min
        ctypes.c_float(50.0),   # match_radius
        ctypes.c_float(0.25),   # ema_alpha
        ctypes.c_uint8(4),      # growth_needed
        ctypes.c_uint16(0), ctypes.c_uint16(0),
        ctypes.c_uint16(639), ctypes.c_uint16(479)
    )
    lib.start_camera()

    rows = []

    print("Starting live optical-expansion benchmark...")
    print("Approach obstacle now.")

    try:
        for window_idx in range(n_windows):
            t0_ns = time.perf_counter_ns()

            # collect live events for 50 ms
            t_collect_end = time.perf_counter() + (tick_ms / 1000.0)
            total_events = 0
            while time.perf_counter() < t_collect_end:
                count = lib.update_camera()
                if count > 0:
                    total_events += count

            # process that 50 ms chunk
            lib.baselineTick(ctypes.c_float(0.05))
            ttc = float(lib.baselineGetCriticalTTC())
            if not math.isfinite(ttc):
                ttc = None

            t1_ns = time.perf_counter_ns()
            lat_ms = (t1_ns - t0_ns) / 1e6

            rows.append({
                "window_idx": window_idx,
                "event_count": total_events,
                "ttc": "" if ttc is None else ttc,
                "latency_ms": lat_ms,
            })

            print(f"[OPT] window={window_idx:03d} events={total_events:5d} "
                  f"ttc={ttc} latency_ms={lat_ms:.2f}")

    finally:
        lib.shutdown()

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["window_idx", "event_count", "ttc", "latency_ms"])
            writer.writeheader()
            writer.writerows(rows)

        lats = np.array([r["latency_ms"] for r in rows], dtype=np.float64)
        print("\nSaved:", out_csv)
        print(f"mean   = {lats.mean():.2f} ms")
        print(f"median = {np.median(lats):.2f} ms")
        print(f"p95    = {np.percentile(lats, 95):.2f} ms")


if __name__ == "__main__":
    main()