#!/usr/bin/env python3
"""
Train a Spiking Neural Network (SNN) for TTC Estimation
Method: Direct Training (BPTT) with Surrogate Gradients
Framework: SpikingJelly + PyTorch
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import torch.nn.functional as F
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Hyperparameters — tuned for EvTTC TTC regression
# -----------------------------------------------------------------------------
CONFIG = {
    "lr":           1e-4,    # Adam learning rate
    "weight_decay": 1e-4,    # L2 regularization
    "alpha":        2.0,     # ATan surrogate gradient sharpness
    "dropout_rate": 0.3,     # Dropout before FC layers
    "v_thresh":     1.0,     # LIF spike threshold
    "huber_delta": 0.5,     # Huber loss delta (seconds) — transition point between L1 and L2
    "batch_size":   16,
    "epochs":       5,
    "time_steps":   5,
}

# -----------------------------------------------------------------------------
# 1. Dataset Class
# -----------------------------------------------------------------------------
class TTCDataset(Dataset):
    def __init__(self, X_path, y_path, target_h=240, target_w=320):
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path)
        self.target_h = target_h
        self.target_w = target_w

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        x = F.interpolate(x.unsqueeze(0), size=(240, 320), mode='bilinear', align_corners=False).squeeze(0)
        y = torch.tensor([self.y[idx]]).float()
        return x, y


# -----------------------------------------------------------------------------
# 2. SNN Architecture
# -----------------------------------------------------------------------------
class SpikeCarSNN(nn.Module):

    """
    Input:  (B, T=5, 240, 320)
    Output: scalar TTC prediction per sample in seconds

    Spatial flow:
      Block 1: (B, 1,   240, 320) -> pool -> (B, 16,  120, 160)
      Block 2: (B, 16,  120, 160) -> pool -> (B, 32,   60,  80)
      Block 3: (B, 32,   60,  80) -> pool -> (B, 64,   30,  40)
      Block 4: (B, 64,   30,  40) -> pool -> (B, 128,  15,  20)
      GAP:     (B, 128, 15,  20)  -> GAP  -> (B, 128)
      FC1:     (B, 128) -> (B, 64)
      FC2:     (B, 64)  -> (B, 1)
      Readout: accumulates membrane voltage over T steps -> scalar TTC
    """

    def __init__(self, time_steps=5, alpha=2.0, dropout_rate=0.3, v_thresh=1.0):
        super().__init__()
        self.T = time_steps

        def make_plif():
            # ParametricLIF: tau is a learnable parameter per neuron
            # step_mode='s' matches our explicit time loop in forward()
            return neuron.ParametricLIFNode(
                init_tau=2.0,
                v_threshold=v_thresh,
                detach_reset=True,
                surrogate_function=surrogate.ATan(alpha=alpha),
                step_mode='s'
            )

        self.conv1       = layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(16)
        self.lif1        = make_plif()

        self.conv2       = layer.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(32)
        self.lif2        = make_plif()

        self.conv3       = layer.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = layer.BatchNorm2d(64)
        self.lif3        = make_plif()

        self.conv4       = layer.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = layer.BatchNorm2d(128)
        self.lif4        = make_plif()

        self.pool = layer.MaxPool2d(kernel_size=2, stride=2)
        self.gap = layer.AdaptiveAvgPool2d((1, 1))

        self.dropout                = layer.Dropout(dropout_rate)

        self.fc1                    = layer.Linear(128, 64, bias=False)
        self.lif5                   = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode='s')
        
        self.fc2                    = layer.Linear(64, 1, bias=False)

        # Readout: v_thresh=inf means it NEVER fires — only accumulates voltage
        # This gives us a continuous analog output suitable for regression
        self.readout = neuron.LIFNode(
            tau=2.0,
            v_threshold=float('inf'),
            detach_reset=True,
            step_mode='s'
        )

    def forward(self, x):
        # x: (B, T, H, W)
        batch_size  = x.shape[0]

        for t in range(self.T):
            x_t = x[:, t:t+1, :, :] # (B, 1, H, W)

            # Each block: conv -> BN -> LIF -> pool
            out = self.pool(self.lif1(self.bn1(self.conv1(x_t)))) # (B, 16, 120, 160)
            out = self.pool(self.lif2(self.bn2(self.conv2(out))))   # (B, 32,  60,  80)
            out = self.pool(self.lif3(self.bn3(self.conv3(out))))   # (B, 64,  30,  40)
            out = self.pool(self.lif4(self.bn4(self.conv4(out))))   # (B, 128, 15,  20)

            out = self.gap(out).view(batch_size, -1)  # (B, 128)
            out = self.dropout(out)
            out = self.lif5(self.fc1(out))            # (B, 64)
            out = self.fc2(out)                       # (B, 1)

            # Feed into readout — side effect charges .v, return value (always 0) discarded
            self.readout(out)

        # Membrane voltage = TTC prediction accumulated over all T steps
        return self.readout.v  # (B, 1)


# -----------------------------------------------------------------------------
# 3. Device
# -----------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------------------------------------------------------------
# 4. Training
# -----------------------------------------------------------------------------
def train(config):
    device = get_device()
    print(f"Device     : {device}")
    print(f"Config     : {config}\n")
    Path("models").mkdir(exist_ok=True)

    # --- Data ---
    train_ds = TTCDataset('data/processed/X_train.npy', 'data/processed/y_train.npy')
    val_ds   = TTCDataset('data/processed/X_val.npy',   'data/processed/y_val.npy')

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}\n")

    # --- Model ---
    model = SpikeCarSNN(
        time_steps   = config["time_steps"],
        alpha        = config["alpha"],
        dropout_rate = config["dropout_rate"],
        v_thresh     = config["v_thresh"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params  : {total_params:,}\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # Cosine annealing: smoothly decays LR to near-zero over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    delta = config["huber_delta"]

    # --- Logging ---
    history = {
        "train_loss": [], "val_loss": [],
        "val_mae": [], "val_rel_err": [],
        "lr": []
    }
    best_val_loss = float('inf')
    epochs_no_improve = 0
    EARLY_STOP_PATIENCE = 8  # stop if val loss doesn't improve for 8 epochs

    print("=" * 75)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val MAE':>9} | {'Rel Err%':>9} | {'LR':>10}")
    print("=" * 75)

    total_start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        train_loss_sum = 0.0

        for x, y in tqdm(train_loader, desc=f"  [Train] Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)

            # Reset all LIF neuron states (membrane voltages + spike traces)
            # Must happen before EVERY forward pass — neurons are stateful
            functional.reset_net(model)

            # Forward Pass
            optimizer.zero_grad()  # clear gradients from previous batch
            pred = model(x)
            loss = F.huber_loss(pred, y, delta=delta)
            loss.backward()

            # Clip gradients — prevents exploding gradients through T BPTT steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss_sum += loss.item()

        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader)
        current_lr     = scheduler.get_last_lr()[0]

        # ---- Validate ----
        model.eval()
        val_loss_sum     = 0.0
        val_mae_sum      = 0.0
        val_rel_err_sum  = 0.0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"  [Val]   Epoch {epoch}", leave=False):
                x, y = x.to(device), y.to(device)
                functional.reset_net(model)   # reset here too — val batches are independent

                pred = model(x)

                val_loss_sum    += F.huber_loss(pred, y, delta=delta).item()
                val_mae_sum     += F.l1_loss(pred, y).item()
                val_rel_err_sum += (
                    torch.mean(torch.abs((y - pred) / y.clamp(min=0.1))).item() * 100
                )

        n             = len(val_loader)
        avg_val_loss  = val_loss_sum    / n
        avg_val_mae   = val_mae_sum     / n
        avg_val_rel   = val_rel_err_sum / n
        epoch_time    = time.time() - epoch_start

        # Log
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mae"].append(avg_val_mae)
        history["val_rel_err"].append(avg_val_rel)
        history["lr"].append(current_lr)

        # Flags for console readability
        improved_flag = " <<<" if avg_val_loss < best_val_loss else ""
        print(
            f"{epoch:>6} | {avg_train_loss:>10.4f} | {avg_val_loss:>9.4f} | "
            f"{avg_val_mae:>9.4f} | {avg_val_rel:>8.1f}% | {current_lr:>10.2e}"
            f"  [{epoch_time:.1f}s]{improved_flag}"
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     {k: v.cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             best_val_loss,
                    "val_mae":              avg_val_mae,
                    "config":               config,
                },
                "models/best_snn.pth"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping triggered — no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    total_time = time.time() - total_start
    print("=" * 75)
    print(f"Training complete in {total_time / 60:.1f} min")
    print(f"Best Val Loss : {best_val_loss:.4f}")

    # ---- Plot ----
    epochs_ran = len(history["train_loss"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(range(1, epochs_ran + 1), history["train_loss"], label="Train")
    axes[0].plot(range(1, epochs_ran + 1), history["val_loss"],   label="Val")
    axes[0].set_title("Huber Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(range(1, epochs_ran + 1), history["val_mae"],     color='orange', label="Val MAE")
    axes[1].plot(range(1, epochs_ran + 1), history["val_rel_err"], color='red',    label="Val Rel Err %")
    axes[1].set_title("Val MAE (s) & Rel Error (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(range(1, epochs_ran + 1), history["lr"], color='green')
    axes[2].set_title("Learning Rate (Cosine Decay)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("models/training_curves.png", dpi=150)
    plt.close()
    print("Training curves saved to models/training_curves.png")

    return history


# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train(CONFIG)