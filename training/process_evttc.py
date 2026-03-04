#!/usr/bin/env python3
"""
Process EvTTC dataset into training samples for Samsung DVS camera

- takes the raw .hdf5 files from the EvTTC dataset
- converts them into numpy arrays that the SNN can train on
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import h5py
import pandas as pd
from pathlib import Path


def load_events_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'prophesee' in f:
            prophesee = f['prophesee']
            if 'event_cam_left' in prophesee:
                print("Using: /prophesee/event_cam_left")
                cam_group = prophesee['event_cam_left']
            elif 'event_cam_right' in prophesee:
                print("Using: /prophesee/event_cam_right")
                cam_group = prophesee['event_cam_right']
            else:
                raise KeyError("Cannot find event_cam_left or event_cam_right in /prophesee")

            events = {
                't': cam_group['t'][:].astype(np.float64),
                'x': cam_group['x'][:].astype(np.uint16),
                'y': cam_group['y'][:].astype(np.uint16),
                'p': cam_group['p'][:].astype(np.int8),
            }

            if 'calib' in cam_group and 'resolution' in cam_group['calib']:
                resolution = cam_group['calib']['resolution'][:]
                print(f"Camera resolution: {resolution[0]}x{resolution[1]} (width x height)")

            return events
        else:
            raise KeyError("Cannot find 'prophesee' group in HDF5 file")


def load_ttc_csv(csv_file):
    print(f"Loading TTC from {csv_file.name}...")
    df = pd.read_csv(csv_file, sep='\t', engine='python')

    print(f"Detected {df.shape[1]} columns")
    if df.shape[1] < 5:
        print(f"Warning: Expected 5 columns, got {df.shape[1]}")

    ttc_values = df.iloc[:, -1].values.astype(np.float32)

    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].values.astype(np.float32)
    elif 'timestamp (s)' in df.columns:
        timestamps = df['timestamp (s)'].values.astype(np.float32)
    elif df.shape[1] > 1:
        timestamps = df.iloc[:, 1].values.astype(np.float32)
    else:
        timestamps = None

    print(f"Loaded {len(ttc_values)} TTC ground truth values")
    print(f"TTC range: [{ttc_values.min():.2f}, {ttc_values.max():.2f}] seconds")

    return ttc_values, timestamps


def letterbox_events(events, target_width=640, target_height=480):
    width  = int(events['x'].max()) + 1
    height = int(events['y'].max()) + 1

    scale     = min(target_width / width, target_height / height)
    new_width  = int(np.floor(width  * scale))
    new_height = int(np.floor(height * scale))
    pad_x = (target_width  - new_width)  / 2.0
    pad_y = (target_height - new_height) / 2.0

    x = events['x'].astype(np.float32) * scale + pad_x
    y = events['y'].astype(np.float32) * scale + pad_y

    keep = (x >= 0) & (x < target_width) & (y >= 0) & (y < target_height)

    events2 = {k: v[keep] for k, v in events.items()}
    events2['x'] = np.floor(x[keep]).astype(np.uint16)
    events2['y'] = np.floor(y[keep]).astype(np.uint16)

    return events2, dict(scale=scale, pad=(pad_x, pad_y),
                         orig=(width, height), scaled=(new_width, new_height))


def load_and_resize_evttc(sequence_path, target_height=480, target_width=640):
    print(f"\n{'='*60}")
    print(f"Loading: {sequence_path.name}")
    print('='*60)

    events_file = sequence_path / "events.hdf5"
    if not events_file.exists():
        events_file = sequence_path / "events.h5"
        if not events_file.exists():
            print("=> No events file found")
            return None, None, None, None

    try:
        events = load_events_from_hdf5(events_file)
    except Exception as e:
        print(f"=> Error loading events: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    print(f"Loaded {len(events['t']):,} events")
    print(f"Time range: {events['t'][0]/1e6:.3f}s to {events['t'][-1]/1e6:.3f}s")
    print(f"Duration: {(events['t'][-1] - events['t'][0])/1e6:.2f} seconds")

    original_width  = int(events['x'].max()) + 1
    original_height = int(events['y'].max()) + 1
    print(f"Detected resolution from orig data: {original_width}x{original_height}")

    events, info = letterbox_events(events, target_width, target_height)
    print(f"Letterbox: scale={info['scale']:.4f}, pad={info['pad']}, "
          f"orig={info['orig']}, scaled={info['scaled']}")

    ttc_file = sequence_path / "ttc_groundtruth.csv"
    ttc_timestamps = None

    if not ttc_file.exists():
        print("=> No TTC ground truth CSV found")
        ttc_gt = None
    else:
        try:
            ttc_gt, ttc_timestamps = load_ttc_csv(ttc_file)
        except Exception as e:
            print(f"=> Error loading TTC: {e}")
            ttc_gt = None

    return events, ttc_gt, ttc_timestamps, (target_height, target_width)


def create_timesurface(events, start_time, end_time, height, width, tau=0.01):
    time_surface = np.zeros((height, width), dtype=np.float32)
    mask = (events['t'] >= start_time) & (events['t'] < end_time)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return time_surface

    # Vectorized — no Python loop over events
    t_arr = events['t'][indices].astype(np.float64)
    x_arr = events['x'][indices].astype(np.int32)
    y_arr = events['y'][indices].astype(np.int32)
    p_arr = events['p'][indices]

    decay = np.exp(-(end_time - t_arr) / (tau * 1e6))
    signed_decay = np.where(p_arr > 0, decay, -decay).astype(np.float32)
    np.add.at(time_surface, (y_arr, x_arr), signed_decay)

    return time_surface


def create_temporal_bins(events, start_time, height, width, n_bins=5, bin_duration_us=10000):
    bins = []
    for i in range(n_bins):
        bin_start = start_time + i * bin_duration_us
        bin_end   = bin_start  + bin_duration_us
        ts = create_timesurface(events, bin_start, bin_end, height, width)
        bins.append(ts)
    return np.stack(bins, axis=0)


def process_sequence(sequence_path, height=480, width=640):
    events, ttc_gt, ttc_timestamps, (h, w) = load_and_resize_evttc(
        sequence_path, height, width)

    if events is None:
        return [], []

    samples, labels = [], []
    t_start = events['t'][0]
    t_end   = events['t'][-1]
    duration = t_end - t_start

    window_duration_us = 50_000
    stride_us          = 25_000

    print(f"\n Processing:")
    print(f"Window: {window_duration_us/1000:.0f}ms, Stride: {stride_us/1000:.0f}ms")
    print(f"Expected samples: ~{int((duration - window_duration_us) / stride_us)}")

    t = int(t_start)
    sample_count   = 0
    filtered_count = 0

    while t + window_duration_us <= int(t_end):
        bins = create_temporal_bins(events, t, height, width,
                                    n_bins=5, bin_duration_us=10_000)

        if ttc_gt is not None:
            current_time_s = (t - t_start) / 1e6

            if ttc_timestamps is not None:
                ttc_idx = np.argmin(np.abs(ttc_timestamps - current_time_s))
            else:
                time_fraction = (t - t_start) / duration
                ttc_idx = int(time_fraction * len(ttc_gt))

            ttc_idx   = min(int(ttc_idx), len(ttc_gt) - 1)
            ttc_value = ttc_gt[ttc_idx]

            if 0.1 < ttc_value < 10.0:
                samples.append(bins)
                labels.append(ttc_value)
                sample_count += 1
            else:
                filtered_count += 1

        t += stride_us

    print(f"Generated: {sample_count} samples, filtered: {filtered_count}")
    return samples, labels


def load_slider_events(sequence_path):
    events_data = np.load(str(sequence_path / 'events.npz'))
    ttc_data    = np.load(str(sequence_path / 'ttc_gt.npz'))

    events = {
        't': events_data['t'].astype(np.float64),
        'x': events_data['x'].astype(np.uint16),
        'y': events_data['y'].astype(np.uint16),
        'p': events_data['p'].astype(np.int8),
    }
    ttc_gt         = ttc_data['ttc'].astype(np.float32)
    ttc_timestamps = ttc_data['t'].astype(np.float64) / 1e6  # μs → seconds

    return events, ttc_gt, ttc_timestamps


def process_slider_sequence(sequence_path, height=480, width=640):
    print(f"\n{'='*60}")
    print(f"Loading slider: {sequence_path.name}")
    print('='*60)

    events, ttc_gt, ttc_timestamps = load_slider_events(sequence_path)

    print(f"Loaded {len(events['t']):,} events")
    print(f"TTC msgs: {len(ttc_gt)}, range: [{ttc_gt.min():.2f}, {ttc_gt.max():.2f}]s")

    orig_w = int(events['x'].max()) + 1
    orig_h = int(events['y'].max()) + 1
    print(f"Detected resolution: {orig_w}x{orig_h}")

    if orig_w != width or orig_h != height:
        print(f"Resizing {orig_w}x{orig_h} → {width}x{height}")
        events, _ = letterbox_events(events, width, height)

    t_start = events['t'][0]
    t_end   = events['t'][-1]

    # Align TTC timestamps to event stream timebase
    # Both start independently at t=0 so offset is usually ~0,
    # but we compute it explicitly to be safe
    event_t_start_s = t_start / 1e6
    ttc_t_start     = ttc_timestamps[0]
    ttc_offset      = event_t_start_s - ttc_t_start
    ttc_timestamps_aligned = ttc_timestamps + ttc_offset

    print(f"TTC timestamp offset applied: {ttc_offset:.4f}s")
    print(f"Valid TTC entries (>0.1s): {(ttc_gt > 0.1).sum()} / {len(ttc_gt)}")

    window_duration_us = 50_000
    stride_us          = 25_000

    t = int(t_start)
    samples, labels = [], []
    sample_count   = 0
    filtered_count = 0

    while t + window_duration_us <= int(t_end):
        bins = create_temporal_bins(events, t, height, width,
                                    n_bins=5, bin_duration_us=10_000)

        current_time_s = (t - t_start) / 1e6
        ttc_idx   = int(np.argmin(np.abs(ttc_timestamps_aligned - current_time_s)))
        ttc_idx   = min(ttc_idx, len(ttc_gt) - 1)
        ttc_value = ttc_gt[ttc_idx]

        if 0.1 < ttc_value < 5.0:
            samples.append(bins)
            labels.append(ttc_value)
            sample_count += 1
        else:
            filtered_count += 1

        t += stride_us

    print(f"Generated: {sample_count} samples, filtered: {filtered_count}")
    return samples, labels


def main():
    print("\n" + "=" * 60)
    print("EvTTC Dataset Processing for Samsung DVS (640x480)")
    print("=" * 60)

    splits = {
        "train": [
            "CCRs-1-low-100%-ttc",
            "CCRs-1-medium-100%-ttc",
        ],
        "val": [
            "CCRs-2-low-100%-ttc",
        ],
        "test": [
            "CCRs-3-low-100%-ttc",
        ],
    }

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- EvTTC splits ---
    for split_name, sequences in splits.items():
        print(f"\n\n{'='*20} Processing {split_name.upper()} split {'='*20}")

        all_samples, all_labels = [], []

        for seq_name in sequences:
            seq_path = Path("data/evttc") / seq_name
            if seq_path.exists():
                samples, labels = process_sequence(seq_path)
                all_samples.extend(samples)
                all_labels.extend(labels)
            else:
                print(f"=> Folder not found: {seq_path}")

        if not all_samples:
            print(f"=> No valid samples for {split_name}. Skipping.")
            continue

        X = np.array(all_samples, dtype=np.float32)
        y = np.array(all_labels,  dtype=np.float32)

        print("Normalizing samples...")
        for i in range(len(X)):
            s_max = np.abs(X[i]).max()
            if s_max > 0:
                X[i] /= s_max

        np.save(output_dir / f"X_{split_name}.npy", X)
        np.save(output_dir / f"y_{split_name}.npy", y)

        print(f"\nSUCCESS! Saved {len(X):,} {split_name} samples")
        print(f"  Shape: {X.shape}")
        print(f"  Size:  {X.nbytes / (1024**2):.1f} MB")
        print(f"  TTC:   {y.mean():.2f}s ± {y.std():.2f}s")

    # --- Slider sequences (saved separately, combined in DataLoader) ---
    print("\n\n" + "="*20 + " Processing SLIDER sequences " + "="*20)

    slider_sequences = [
        Path("data/slider/Slider500"),
        Path("data/slider/Slider750"),
        Path("data/slider/Slider1000"),
    ]

    slider_samples, slider_labels = [], []
    for seq_path in slider_sequences:
        if seq_path.exists():
            s, l = process_slider_sequence(seq_path)
            slider_samples.extend(s)
            slider_labels.extend(l)
        else:
            print(f"=> Not found: {seq_path}")

    if slider_samples:
        # 5x oversample — slider dataset is small relative to EvTTC
        slider_samples = slider_samples * 5
        slider_labels  = slider_labels  * 5

        X_slider = np.array(slider_samples, dtype=np.float32)
        y_slider = np.array(slider_labels,  dtype=np.float32)

        print("Normalizing slider samples...")
        for i in range(len(X_slider)):
            s_max = np.abs(X_slider[i]).max()
            if s_max > 0:
                X_slider[i] /= s_max

        # Save separately — avoids OOM from concatenating 5GB arrays in RAM
        np.save(output_dir / 'X_slider.npy', X_slider)
        np.save(output_dir / 'y_slider.npy', y_slider)

        print(f"\nSlider saved: {len(X_slider)} samples (5x oversampled)")
        print(f"  Shape: {X_slider.shape}")
        print(f"  Size:  {X_slider.nbytes / (1024**2):.1f} MB")
        print(f"  TTC:   {y_slider.min():.2f}s – {y_slider.max():.2f}s")
    else:
        print("=> No slider samples generated.")


if __name__ == "__main__":
    main()
