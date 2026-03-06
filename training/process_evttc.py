#!/usr/bin/env python3
"""
Process EvTTC dataset into training samples for Samsung DVS camera
- takes the raw .hdf5 files from the EvTTC dataset
- converts them into numpy arrays that the SNN can train on
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


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

    scale      = min(target_width / width, target_height / height)
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

    ttc_file       = sequence_path / "ttc_groundtruth.csv"
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
    mask    = (events['t'] >= start_time) & (events['t'] < end_time)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return time_surface

    t_arr = events['t'][indices].astype(np.float64)
    x_arr = events['x'][indices].astype(np.int32)
    y_arr = events['y'][indices].astype(np.int32)
    p_arr = events['p'][indices]

    decay        = np.exp(-(end_time - t_arr) / (tau * 1e6))
    signed_decay = np.where(p_arr > 0, decay, -decay).astype(np.float32)
    np.add.at(time_surface, (y_arr, x_arr), signed_decay)

    return time_surface


def create_temporal_bins(events, start_time, height, width, n_bins=5, bin_duration_us=10000):
    bins = []
    for i in range(n_bins):
        bin_start = start_time + i * bin_duration_us
        bin_end   = bin_start  + bin_duration_us
        bins.append(create_timesurface(events, bin_start, bin_end, height, width))
    return np.stack(bins, axis=0)


def process_sequence(sequence_path, height=480, width=640):
    events, ttc_gt, ttc_timestamps, (h, w) = load_and_resize_evttc(
        sequence_path, height, width)

    if events is None:
        return [], []

    samples, labels = [], []
    t_start  = events['t'][0]
    t_end    = events['t'][-1]
    duration = t_end - t_start

    window_duration_us = 50_000
    stride_us          = 25_000

    print(f"\n Processing:")
    print(f"  Window: {window_duration_us/1000:.0f}ms, Stride: {stride_us/1000:.0f}ms")
    print(f"  Expected samples: ~{int((duration - window_duration_us) / stride_us)}")

    t              = int(t_start)
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

            # Replace with the two-tier approach
            if ttc_value <= 0 or np.isinf(ttc_value) or np.isnan(ttc_value):
                filtered_count += 1
            else:
                ttc_value = min(float(ttc_value), 10.0)
                samples.append(bins)
                labels.append(ttc_value)
                sample_count += 1

        t += stride_us

    print(f"  Generated: {sample_count} samples, filtered: {filtered_count}")
    return samples, labels


# CHANGED: added dataset statistics plot saved alongside the .npy files
def plot_label_distribution(y, split_name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Label Distribution — {split_name} split', fontweight='bold')

    axes[0].hist(y, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('TTC (s)'); axes[0].set_ylabel('Count')
    axes[0].set_title('TTC Histogram'); axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(y, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[1].set_ylabel('TTC (s)'); axes[1].set_title('TTC Boxplot')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f'label_dist_{split_name}.png', dpi=150)
    plt.close(fig)
    print(f"  Saved label distribution → {output_dir}/label_dist_{split_name}.png")


def main():
    
    print("\n" + "=" * 60)
    print("EvTTC Dataset Processing for Samsung DVS (640x480)")
    print("=" * 60)

    splits = {
        "train": [
            "CCRs-1-low-100%-ttc",
            "CCRs-1-medium-100%-ttc",
            "CCRs-side-low-ttc",
        ],
        "val": [
            "CCRs-2-low-100%-ttc",
            "CCRs-side-medium-ttc",
        ],
        "test": [
            "CCRs-3-low-100%-ttc",
            "CCRs-3-medium-100%-ttc",
            "CCRs-2-medium-100%-ttc",
        ],
    }

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

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
        s_max = np.abs(X).max(axis=(1, 2, 3), keepdims=True)
        s_max[s_max == 0] = 1.0
        X = X / s_max

        np.save(output_dir / f"X_{split_name}.npy", X)
        np.save(output_dir / f"y_{split_name}.npy", y)

        # CHANGED: print percentile breakdown so you can see coverage across TTC range
        print(f"\nSUCCESS! Saved {len(X):,} {split_name} samples")
        print(f"  Shape : {X.shape}")
        print(f"  Size  : {X.nbytes / (1024**2):.1f} MB")
        print(f"  TTC   : mean={y.mean():.2f}s  std={y.std():.2f}s  "
              f"min={y.min():.2f}s  max={y.max():.2f}s")
        print(f"  Percentiles — "
              f"p10={np.percentile(y,10):.2f}s  "
              f"p25={np.percentile(y,25):.2f}s  "
              f"p50={np.percentile(y,50):.2f}s  "
              f"p75={np.percentile(y,75):.2f}s  "
              f"p90={np.percentile(y,90):.2f}s")

        # CHANGED: save label distribution plot per split
        plot_label_distribution(y, split_name, output_dir)


if __name__ == "__main__":
    main()
