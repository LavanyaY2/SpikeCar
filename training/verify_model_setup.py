#!/usr/bin/env python3
"""
Verification Script for SNN Model Training Setup
Checks: Libraries, Data Loading, Model Instantiation, Forward Pass
"""

import numpy as np
import time
from pathlib import Path

def print_status(step, msg):
    print(f"[{step}] {msg}")

def check_libraries():
    print_status("1/4", "Checking libraries...")
    try:
        import torch
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        
        import spikingjelly
        print(f"  - SpikingJelly imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Error: {e}")
        return False

def check_data():
    print_status("2/4", "Checking processed data...")
    paths = {
        "X_train": "data/processed/X_train.npy",
        "y_train": "data/processed/y_train.npy",
        "X_val":   "data/processed/X_val.npy",
        "y_val":   "data/processed/y_val.npy"
    }
    
    all_exist = True
    for name, p in paths.items():
        path = Path(p)
        if not path.exists():
            print(f"  ❌ Missing: {p}")
            all_exist = False
        else:
            try:
                # Just peek at the shape without loading data
                data = np.load(p, mmap_mode='r')
                print(f"  - {name}: Found. Shape: {data.shape}")
            except Exception as e:
                print(f"  ❌ Error loading {p}: {e}")
                all_exist = False
                
    if all_exist:
        print("  ✅ Data check passed.")
        return True
    return False

def check_model_forward_pass():
    print_status("3/4", "Checking Model & Forward Pass...")
    try:
        import torch
        import torch.nn as nn
        from spikingjelly.activation_based import neuron, layer, functional, surrogate

        # Define a tiny verification SNN
        class VerificationSNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = layer.Conv2d(5, 4, kernel_size=3, padding=1, bias=False)
                self.bn1 = layer.BatchNorm2d(4)
                self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
                self.pool = layer.AdaptiveAvgPool2d((1, 1))
                self.fc = layer.Linear(4, 1, bias=False) 

            def forward(self, x):
                # Simple forward pass
                # x: (Batch, 5, 480, 640)
                # We'll treat channels as features for this dummy test
                out = self.lif1(self.bn1(self.conv1(x)))
                out = self.pool(out).flatten(1)
                out = self.fc(out)
                return out

        # Create dummy input (Batch=2, Channels=5, H=64, W=64) - smaller size for speed
        dummy_input = torch.randn(2, 5, 64, 64)
        model = VerificationSNN()
        
        print("  - Running dummy forward pass...")
        start = time.time()
        functional.reset_net(model)
        output = model(dummy_input)
        end = time.time()
        
        print(f"  - Output shape: {output.shape} (Expected: [2, 1])")
        
        if output.shape == (2, 1):
            print("  ✅ Model logic check passed.")
            return True
        else:
            print("  ❌ Model output shape incorrect.")
            return False

    except Exception as e:
        print(f"  ❌ Model check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("SNN Training Environment Verification")
    print("="*60)
    
    if not check_libraries():
        print("\n🛑 Critical: Library check failed.")
        return

    print("-" * 30)
    if not check_data():
        print("\n🛑 Critical: Data check failed.")
        return

    print("-" * 30)
    if not check_model_forward_pass():
        print("\n🛑 Critical: Model check failed.")
        return

    print("\n" + "="*60)
    print("✅ READY TO TRAIN!")
    print("   Run: python training/train_snn.py")
    print("="*60)

if __name__ == "__main__":
    main()
