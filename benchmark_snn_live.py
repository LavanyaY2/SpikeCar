#!/usr/bin/env python3
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer

import dvs


# -----------------------------------------------------------------------------
# SNN model (same as training)
# -----------------------------------------------------------------------------
class SpikeCarSNN(nn.Module):
    def __init__(self, time_steps=5, alpha=2.0, dropout_rate=0.3, v_thresh=1.0):
        super().__init__()
        self.T = time_steps

        def make_plif():
            return neuron.ParametricLIFNode(
                init_tau=2.0,
                v_threshold=v_thresh,
                detach_reset=True,
                surrogate_function=surrogate.ATan(alpha=alpha),
                step_mode='s'
            )

        self.conv1 = layer.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.bn1   = layer.BatchNorm2d(16)
        self.lif1  = make_plif()

        self.conv2 = layer.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.bn2   = layer.BatchNorm2d(32)
        self.lif2  = make_plif()

        self.conv3 = layer.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn3   = layer.BatchNorm2d(64)
        self.lif3  = make_plif()

        self.conv4 = layer.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn4   = layer.BatchNorm2d(128)
        self.lif4  = make_plif()

        self.pool    = layer.MaxPool2d(2, 2)
        self.gap     = layer.AdaptiveAvgPool2d((1, 1))
        self.dropout = layer.Dropout(dropout_rate)

        self.fc1  = layer.Linear(128, 64, bias=False)
        self.lif5 = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        self.fc2  = layer.Linear(64, 1, bias=False)

        self.readout = neuron.LIFNode(
            tau=2.0,
            v_threshold=float('inf'),
            detach_reset=True,
            step_mode='s'
        )

    def forward(self, x):
        batch_size = x.shape[0]
        for t in range(self.T):
            x_t = x[:, t:t+1, :, :]
            out = self.pool(self.lif1(self.bn1(self.conv1(x_t))))
            out = self.pool(self.lif2(self.bn2(self.conv2(out))))
            out = self.pool(self.lif3(self.bn3(self.conv3(out))))
            out = self.pool(self.lif4(self.bn4(self.conv4(out))))
            out = self.gap(out).view(batch_size, -1)
            out = self.dropout(out)
            out = self.lif5(self.fc1(out))
            out = self.fc2(out)
            self.readout(out)
        return self.readout.v


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_timesurface(events, start_time, end_time, height, width, tau=0.03):
    ts = np.zeros((height, width), dtype=np.float32)
    mask = (events["t"] >= start_time) & (events["t"] < end_time)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return ts

    t_arr = events["t"][idx].astype(np.float64)
    x_arr = events["x"][idx].astype(np.int32)
    y_arr = events["y"][idx].astype(np.int32)
    p_arr = events["p"][idx]

    decay = np.exp(-(end_time - t_arr) / (tau * 1e6))
    signed_decay = np.where(p_arr > 0, decay, -decay).astype(np.float32)
    np.add.at(ts, (y_arr, x_arr), signed_decay)
    return ts


def build_sample_5x480x640(raw_events, tau=0.03):
    t_end = int(raw_events["t"][-1])
    t_start = t_end - 50_000
    bin_us = 10_000

    bins = []
    for i in range(5):
        b0 = t_start + i * bin_us
        b1 = b0 + bin_us
        bins.append(create_timesurface(raw_events, b0, b1, 480, 640, tau=tau))

    sample = np.stack(bins, axis=0).astype(np.float32)
    scale = np.abs(sample).max()
    if scale > 0:
        sample /= scale
    return sample


# -----------------------------------------------------------------------------
# Main benchmark
# -----------------------------------------------------------------------------
def main():
    weights_path = "best_snn_trial_25.pth"
    out_csv = "snn_live_latency.csv"
    tau = 0.03
    n_windows = 200
    tick_ms = 50

    lib = dvs.Library(width=640, height=480)
    lib.lib.initialize_camera()
    lib.lib.start_camera()

    device = get_device()
    model = SpikeCarSNN(time_steps=5).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []

    print("Starting live SNN benchmark...")
    print("Approach obstacle now.")

    try:
        for window_idx in range(n_windows):
            t0_ns = time.perf_counter_ns()

            # collect live events for one 50 ms window
            time.sleep(tick_ms / 1000.0)
            raw_events = lib.get_recent_events()

            ttc = None
            event_count = 0

            if raw_events is not None and len(raw_events["t"]) >= 2:
                event_count = len(raw_events["t"])

                sample = build_sample_5x480x640(raw_events, tau=tau)
                tensor = torch.from_numpy(sample).unsqueeze(0).to(device)
                tensor = F.interpolate(
                    tensor, size=(240, 320), mode="bilinear", align_corners=False
                )

                with torch.no_grad():
                    functional.reset_net(model)
                    pred = model(tensor)

                ttc = float(pred.item())
                ttc = max(0.0, min(ttc, 6.0))

            t1_ns = time.perf_counter_ns()
            lat_ms = (t1_ns - t0_ns) / 1e6

            rows.append({
                "window_idx": window_idx,
                "event_count": event_count,
                "ttc": "" if ttc is None else ttc,
                "latency_ms": lat_ms,
            })

            print(f"[SNN] window={window_idx:03d} events={event_count:5d} "
                  f"ttc={ttc} latency_ms={lat_ms:.2f}")

    finally:
        lib.lib.stop_camera()
        lib.lib.close_camera()

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