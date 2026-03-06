import argparse
import tkinter as tk
from tkinter import ttk
# from collections import deque
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional

import dvs


class SpikeCarSNN(nn.Module):
    def __init__(self, time_steps=5, alpha=2.0, dropout_rate=0.3, v_thresh=1.0):
        super().__init__()
        self.T = time_steps
        surrogate_gradient = surrogate.ATan(alpha=alpha)

        self.conv1 = layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = layer.BatchNorm2d(16)
        self.lif1 = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=v_thresh,
            detach_reset=True,
            surrogate_function=surrogate_gradient,
        )

        self.conv2 = layer.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = layer.BatchNorm2d(32)
        self.lif2 = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=v_thresh,
            detach_reset=True,
            surrogate_function=surrogate_gradient,
        )

        self.conv3 = layer.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm3 = layer.BatchNorm2d(64)
        self.lif3 = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=v_thresh,
            detach_reset=True,
            surrogate_function=surrogate_gradient,
        )

        self.conv4 = layer.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm4 = layer.BatchNorm2d(128)
        self.lif4 = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=v_thresh,
            detach_reset=True,
            surrogate_function=surrogate_gradient,
        )

        self.pool = layer.MaxPool2d(kernel_size=2, stride=2)
        self.global_average_pooling = layer.AdaptiveAvgPool2d((1, 1))
        self.dropout = layer.Dropout(dropout_rate)
        self.fc1 = layer.Linear(128, 64, bias=False)
        self.lif5 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.fc2 = layer.Linear(64, 1, bias=False)
        self.readout_lif = neuron.LIFNode(tau=2.0, v_threshold=float('inf'))

    def forward(self, x):
        batch_size = x.shape[0]
        for t in range(self.T):
            x_t = x[:, t:t + 1, :, :]
            out = self.lif1(self.batch_norm1(self.conv1(x_t)))
            out = self.lif2(self.batch_norm2(self.conv2(out)))
            out = self.lif3(self.batch_norm3(self.conv3(out)))
            out = self.lif4(self.batch_norm4(self.conv4(out)))
            out = self.pool(out)
            out = self.global_average_pooling(out)
            out = out.view(batch_size, -1)
            out = self.dropout(out)
            out = self.lif5(self.fc1(out))
            out = self.fc2(out)
            out = self.readout_lif(out)
        return self.readout_lif.v


class RealTimeTTCInferencer:
    def __init__(self, weights_path, collision_threshold):
        self.enabled = False
        self.error = None
        self.collision_threshold = collision_threshold

        if weights_path is None:
            self.error = 'No weights supplied. TTC inference disabled.'
            return

        weights = Path(weights_path)
        if not weights.exists():
            self.error = f'Weights not found: {weights}'
            return

        self.device = self._get_device()
        self.model = SpikeCarSNN(time_steps=5).to(self.device)

        try:
            state = torch.load(str(weights), map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.enabled = True
        except Exception as exc:
            self.error = f'Failed to load model: {exc}'

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def predict(self, bins_5x480x640):
        if not self.enabled:
            return None

        x = bins_5x480x640.astype(np.float32, copy=False)
        # scale = float(np.abs(x).max())
        # if scale > 0:
        #     x = x / scale

        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        tensor = F.interpolate(tensor, size=(240, 320), mode='bilinear', align_corners=False)

        with torch.no_grad():
            functional.reset_net(self.model)
            pred = self.model(tensor)
            ttc_s = float(pred.item())

        return ttc_s


THIS_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser('Samsung DVS Viewer + TTC Inference')
parser.add_argument('--source', metavar='FILE', type=str, default=None,
                    help='event file to view')
parser.add_argument('--timesurface_tau', default=0.01, type=float,
                    help='filter time for event image view (default=0.01)')
parser.add_argument('--timesurface_gain', default=1.0, type=float,
                    help='magnitude of event image view (default=1)')
parser.add_argument('--weights', type=str, default=str(THIS_DIR / 'best_snn_trial_25.pth'),
                    help='path to trained SNN weights')
parser.add_argument('--collision_ttc', type=float, default=2.5,
                    help='collision warning threshold in seconds')
parser.add_argument('--tick_ms', type=int, default=50,
                    help='processing/viewer tick period in milliseconds')

args = parser.parse_args()

lib = dvs.Library(width=640, height=480)

# for raw data stuff
def create_timesurface(events, start_time, end_time, height, width, tau=0.01):
    time_surface = np.zeros((height, width), dtype=np.float32)
    mask = (events['t'] >= start_time) & (events['t'] < end_time)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return time_surface
    t_arr = events['t'][indices].astype(np.float64)
    x_arr = events['x'][indices].astype(np.int32)
    y_arr = events['y'][indices].astype(np.int32)
    p_arr = events['p'][indices]
    decay = np.exp(-(end_time - t_arr) / (tau * 1e6))
    signed_decay = np.where(p_arr > 0, decay, -decay).astype(np.float32)
    np.add.at(time_surface, (y_arr, x_arr), signed_decay)
    return time_surface

class MainView:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Samsung DVS Viewer')

        self.root.geometry('1200x600')
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.columnconfigure(0, weight=1)

        self.pane1 = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        self.pane1.grid(row=0, column=0, sticky='news')
        self.frame1 = ttk.Frame(self.pane1, width=600, height=600)

        self.statusbar = ttk.Label(self.root, text='')
        self.statusbar.grid(row=1, column=0, sticky='news')

        self.has_image = False
        self.bind_keys(self.root)

        self.app_image = Timesurface(self, self.frame1)

        self.inferencer = RealTimeTTCInferencer(
            weights_path=args.weights,
            collision_threshold=args.collision_ttc,
        )
        # commenting these out
        # self.bin_buffer = deque(maxlen=5)
        # self.prev_ts_frame = None
        self.last_prediction = None
        self.smoothed_ttc = None
        self.last_event_count = 0
        self.last_tick_time = time.time()
        self.tick_fps = 0.0

        if self.inferencer.enabled:
            self.statusbar.configure(
                text=f'Model loaded on {self.inferencer.device}. Waiting for 5 bins...'
            )
        else:
            self.statusbar.configure(text=self.inferencer.error)

    def toggle_image(self, event=None):
        if not self.has_image:
            if len(self.pane1.panes()) == 0:
                self.pane1.add(self.frame1, weight=1)
            else:
                self.pane1.insert(0, self.frame1, weight=1)
            self.has_image = True
        else:
            self.pane1.forget(0)
            self.has_image = False

    def bind_keys(self, window):
        window.bind('<Up>', self.image_gain_up)
        window.bind('<Down>', self.image_gain_down)
        window.bind('<Escape>', self.quit)

    def image_gain_up(self, event=None):
        args.timesurface_gain *= 1.2

    def image_gain_down(self, event=None):
        args.timesurface_gain /= 1.2

    def quit(self, event):
        self.root.destroy()

    def set_source(self):
        lib.lib.initialize_camera()
        lib.lib.start_camera()
        self.root.after(10, self.pull_camera)
        self.root.after(args.tick_ms, self.tick)

    def pull_camera(self):
        counts = []
        for _ in range(3):
            counts.append(lib.lib.update_camera())
        self.last_event_count = int(np.sum([c for c in counts if c > 0]))
        self.root.after(10, self.pull_camera)

    def tick(self):
        now = time.time()
        dt = now - self.last_tick_time
        self.last_tick_time = now
        if dt > 0:
            self.tick_fps = 1.0 / dt

        if lib.timesurface_active and self.inferencer.enabled:
            raw_events = lib.get_recent_events()

            if raw_events is not None and len(raw_events['t']) >= 2:
                t_end   = int(raw_events['t'][-1])
                t_start = t_end - 50_000  # 50ms window
                bin_duration_us = 10_000

                bins = []
                for i in range(5):
                    bin_start = t_start + i * bin_duration_us
                    bin_end   = bin_start + bin_duration_us
                    ts_bin = create_timesurface(
                        raw_events, bin_start, bin_end, 480, 640, tau=0.01
                    )
                    bins.append(ts_bin)

                sample = np.stack(bins, axis=0)  # (5, 480, 640)

                scale = np.abs(sample).max()
                if scale > 0:
                    sample = sample / scale

                pred_ttc = self.inferencer.predict(sample)

                # ---- Exponential moving average smoothing ----
                ALPHA = 0.3
                if self.smoothed_ttc is None:
                    self.smoothed_ttc = pred_ttc
                else:
                    self.smoothed_ttc = ALPHA * pred_ttc + (1 - ALPHA) * self.smoothed_ttc

                self.last_prediction = self.smoothed_ttc
                # ----------------------------------------------

        self.update_status()
        self.root.after(args.tick_ms, self.tick)

    def update_status(self):
        if not self.inferencer.enabled:
            self.statusbar.configure(text=self.inferencer.error)
            return

        if self.last_prediction is None:
            self.statusbar.configure(
                text=(
                    f'fps: {self.tick_fps:.1f} | '
                    f'events: {self.last_event_count} | '
                    f'waiting for events...'
                )
            )
            return

        ttc = max(self.last_prediction, 0.0)
        state = 'COLLISION WARNING' if ttc <= self.inferencer.collision_threshold else 'safe'
        self.statusbar.configure(
            text=(
                f'fps: {self.tick_fps:.1f} | '
                f'events: {self.last_event_count} | '
                f'pred TTC: {ttc:.2f}s | '
                f'threshold: {self.inferencer.collision_threshold:.2f}s | '
                f'state: {state}'
            )
        )


class Timesurface:
    def __init__(self, app, frame):
        self.app = app

        self.image = ImageTk.PhotoImage(image=self.get_image())
        self.label = ttk.Label(frame, image=self.image, padding=0)
        self.label.grid(column=0, rowspan=1, row=0, sticky='news')
        frame.bind('<Configure>', self.resize_image)
        app.root.update()
        app.root.after(args.tick_ms, self.update)

    def update(self):
        if not self.app.has_image:
            if lib.timesurface_active:
                lib.stop_timesurface()
        else:
            if not lib.timesurface_active:
                lib.init_timesurface(timesurface_tau=args.timesurface_tau)

            img = self.get_image()
            img = img.resize((self.image.width(), self.image.height()))
            self.image = ImageTk.PhotoImage(image=img)
            self.label.config(image=self.image)
        self.app.root.after(args.tick_ms, self.update)

    def resize_image(self, event):
        img = self.get_image()
        img = img.resize((event.width, event.height))
        self.image = ImageTk.PhotoImage(image=img)
        self.label.config(image=self.image)

    def get_image(self):
        if not lib.timesurface_active:
            img = np.zeros((480, 640), dtype=np.uint8) + 128
        else:
            img = np.clip(lib.timesurface * args.timesurface_gain + 128, 0, 255).astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        return img


main = MainView()
main.set_source()
main.toggle_image()
main.root.mainloop()