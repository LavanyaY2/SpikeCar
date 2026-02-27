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


# each event is a tuple (x, y, t, p)
def load_events_from_hdf5(file_path):
    """
    Load events from EvTTC HDF5 format
    Structure: /prophesee/event_cam_left/{x, y, t, p}
    """
    with h5py.File(file_path, 'r') as f:
        # EvTTC stores events under /prophesee/event_cam_left or /prophesee/event_cam_right
        # Use left camera (8mm lens) by default
        if 'prophesee' in f:
            prophesee = f['prophesee']
            
            # Try left camera first
            if 'event_cam_left' in prophesee:
                print("Using: /prophesee/event_cam_left")
                cam_group = prophesee['event_cam_left']
            # Fallback to right camera
            elif 'event_cam_right' in prophesee:
                print("Using: /prophesee/event_cam_right")
                cam_group = prophesee['event_cam_right']
            else:
                raise KeyError("Cannot find event_cam_left or event_cam_right in /prophesee")
            
            # Load separate arrays for x, y, t, p
            events = {
                't': cam_group['t'][:].astype(np.float64),
                'x': cam_group['x'][:].astype(np.uint16),
                'y': cam_group['y'][:].astype(np.uint16),
                'p': cam_group['p'][:].astype(np.int8)
            }
            
            # Load resolution from calibration
            if 'calib' in cam_group and 'resolution' in cam_group['calib']:
                resolution = cam_group['calib']['resolution'][:]
                print(f"Camera resolution: {resolution[0]}x{resolution[1]} (width x height)")
            
            return events
        else:
            raise KeyError("Cannot find 'prophesee' group in HDF5 file")


def load_ttc_csv(csv_file):
    """
    Load TTC ground truth from CSV/TSV file
    Format: Index, timestamp (s), distance (m), velocity (m/s), ttc (s)
    """
    print(f"Loading TTC from {csv_file.name}...")
    df = pd.read_csv(csv_file, sep='\t', engine='python')
    
    print(f"Detected {df.shape[1]} columns")
    if df.shape[1] < 5:
        print(f"Warning: Expected 5 columns, got {df.shape[1]}")
    
    # The CSV has columns: Index, timestamp, distance, velocity, ttc
    # We want the 'ttc' column (last column or column named 'ttc')
    # Assume last column is TTC
    ttc_values = df.iloc[:, -1].values.astype(np.float32)
    
    # Get timestamps if available
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


# Helper method for Letterbox resizing
def letterbox_events(events, target_width=640, target_height=480):
    
    # Original sensor resolution from max observed coordinates
    width = int(events['x'].max()) + 1
    height = int(events['y'].max()) + 1

    # Compute a single uniform scale - fits scaled content entirely in the target without cropping
    scale = min(target_width / width, target_height / height)

    new_width = int(np.floor(width * scale))
    new_height = int(np.floor(height * scale))

    # Compute remaining space to add padding
    # + Splits it on both sides to center the image
    pad_x = (target_width - new_width) / 2.0
    pad_y = (target_height - new_height) / 2.0

    # Transform every coordinate
    x = events['x'].astype(np.float32) * scale + pad_x
    y = events['y'].astype(np.float32) * scale + pad_y

    # Events tha are inside the target frame
    keep = (x >= 0) & (x < target_width) & (y >= 0) & (y < target_height)

    # Only keep events in the target frame
    events2 = {k: v[keep] for k, v in events.items()}
    events2['x'] = np.floor(x[keep]).astype(np.uint16)
    events2['y'] = np.floor(y[keep]).astype(np.uint16)

    # Return the resized and padded event stream
    return events2, dict(scale=scale, pad=(pad_x, pad_y), orig=(width, height), scaled=(new_width, new_height))


def load_and_resize_evttc(sequence_path, target_height=480, target_width=640):
    """
    Load EvTTC and resize to Samsung DVS Gen3 camera resolution (640x480)
    EvTTC data original resolution: 1280x720 (Prophesee EVK4)
    """
    print(f"\n{'='*60}")
    print(f"Loading: {sequence_path.name}")
    print('='*60)
    
    events_file = sequence_path / "events.hdf5"
    if not events_file.exists():
        events_file = sequence_path / "events.h5"
        if not events_file.exists():
            print(f"=> No events file found")
            return None, None, None, None
    
    # Load events from HDF5 file
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
    
    # Get original resolution from the data
    original_width = int(events['x'].max()) + 1
    original_height = int(events['y'].max()) + 1
    print(f"Detected resolution from orig data: {original_width}x{original_height}")
    
    events, info = letterbox_events(events, target_width, target_height)
    print(f"Letterbox: scale={info['scale']:.4f}, pad={info['pad']}, orig={info['orig']}, scaled={info['scaled']}")
    
    # Load TTC ground truth
    ttc_file = sequence_path / "ttc_groundtruth.csv"
    ttc_timestamps = None
    
    if not ttc_file.exists():
        print(f"=> No TTC ground truth CSV found")
        ttc_gt = None
    
    if ttc_file.exists():
        try:
            ttc_gt, ttc_timestamps = load_ttc_csv(ttc_file)
        except Exception as e:
            print(f"=> Error loading TTC: {e}")
            ttc_gt = None
    else:
        ttc_gt = None
    
    return events, ttc_gt, ttc_timestamps, (target_height, target_width)


def create_timesurface(events, start_time, end_time, height, width, tau=0.01):
    """
    Create a 2D time surface representation for a time window using exponential decay
    More recent events contribute larger values
    Each pixel in the output has a signed value representing recent activity there

    tau: time constant in seconds (adjusted for microsecond timestamps)
    """

    # Output image
    time_surface = np.zeros((height, width), dtype=np.float32)
    # Select events tht have timestamps that are in the time window
    mask = (events['t'] >= start_time) & (events['t'] < end_time)
    
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return time_surface
    
    for i in indices:
        t = events['t'][i]
        x = int(events['x'][i])
        y = int(events['y'][i])
        p = events['p'][i]
        
        # Exponential decay
        decay = np.exp(-(end_time - t) / (tau * 1e6))
        
        # Positive polarity adds, negative subtracts
        time_surface[y, x] += decay if p > 0 else -decay
    
    return time_surface


def create_temporal_bins(events, start_time, height, width, n_bins=5, bin_duration_us=10000):
    """
    Create multiple time surfaces to form a multi-channel tensor
    With the default bin and bin_duration values, we get a 50ms sample
    
    Returns bins stacked into shape (5, height, width)
    """
    bins = []
    for i in range(n_bins):
        bin_start = start_time + i * bin_duration_us
        bin_end = bin_start + bin_duration_us
        ts = create_timesurface(events, bin_start, bin_end, height, width)
        bins.append(ts)
    
    return np.stack(bins, axis=0)


def process_sequence(sequence_path, height=480, width=640):
    """
    Process a single EvTTC sequence into training samples
    Walk through entire event stream and generate samples using a sliding window
    """

    events, ttc_gt, ttc_timestamps, (h, w) = load_and_resize_evttc(sequence_path, height, width)
    
    if events is None:
        return [], []
    
    samples, labels = [], []
    t_start, t_end = events['t'][0], events['t'][-1]
    duration = t_end - t_start
    
    # Sliding window: 50ms window, 25ms stride => gives 50% overlap between windows
    window_duration_us = 50000  # 50ms
    stride_us = 25000  # 25ms
    
    print(f"\n Processing:")
    print(f"Window: {window_duration_us/1000:.0f}ms")
    print(f"Stride: {stride_us/1000:.0f}ms")
    print(f"Expected number of samples: ~{int((duration - window_duration_us) / stride_us)}")
    
    t = int(t_start)
    sample_count = 0
    filtered_count = 0

    while t + window_duration_us <= int(t_end):
        bins = create_temporal_bins(events, t, height, width, n_bins=5, bin_duration_us=10000)
        
        if ttc_gt is not None:
            # Map timestamp to groundtruth TTC index
            current_time_s = (t - t_start) / 1e6
            
            if ttc_timestamps is not None:
                # Find closest TTC timestamp
                ttc_idx = np.argmin(np.abs(ttc_timestamps - current_time_s))
            else:
                # Linear interpolation based on time fraction
                time_fraction = (t - t_start) / duration
                ttc_idx = int(time_fraction * len(ttc_gt))
            
            ttc_idx = min(ttc_idx, len(ttc_gt) - 1)
            ttc_value = ttc_gt[ttc_idx]
            
            # Filter: only include approaching collision scenarios (i.e. TTC between 0.1 and 10 seconds)
            if 0.1 < ttc_value < 10.0:
                samples.append(bins)
                labels.append(ttc_value)
                sample_count += 1
            else:
                filtered_count += 1
        
        t += stride_us
    
    print(f"Generated: {sample_count} samples")
    if filtered_count > 0:
        print(f"Filtered: {filtered_count} samples (TTC out of range)")
    
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
        ]
    }

    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, sequences in splits.items():
        print(f"\n\n{'=' * 20} Processing {split_name.upper()} split {'=' * 20}")

        all_samples, all_labels = [], []

        # Process all folders in a split
        for seq_name in sequences:
            seq_path = Path("data/evttc") / seq_name

            if seq_path.exists():
                samples, labels = process_sequence(seq_path)
                all_samples.extend(samples)
                all_labels.extend(labels)
            else:
                print(f"=> Folder not found: {seq_path}")
                
        if not all_samples:
            print(f"=> No valid samples generated for {split_name} split. Skipping.")
            continue
        
        # Convert to numpy arrays
        X = np.array(all_samples, dtype=np.float32)
        y = np.array(all_labels, dtype=np.float32)

        # Normalize each sample
        print("Normalizing samples...")
        for i in range(len(X)):
            sample_max = np.abs(X[i]).max()
            if sample_max > 0:
                X[i] = X[i] / sample_max

        # X_train - 4D numpy array containing all 50ms time surface images
        # y_train - 1S array containing the exact time to collision for every sample in X_train
        x_filename = f"X_{split_name}.npy"
        y_filename = f"y_{split_name}.npy"
        np.save(output_dir / x_filename, X)
        np.save(output_dir / y_filename, y)

        print("\nSUCCESS!")
        print(f"Saved {len(X):,} {split_name} samples")
        print(f"  Shape: {X.shape}")
        print(f"  Size: {X.nbytes / (1024**2):.1f} MB")
        print(f"  TTC mean: {y.mean():.2f}s ± {y.std():.2f}s")

if __name__ == "__main__":
    main()
