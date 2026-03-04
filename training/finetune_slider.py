#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import numpy as np
from pathlib import Path
from tqdm import tqdm


# -----------------------------------------------------------------------------
# SNN Architecture
# -----------------------------------------------------------------------------
class SpikeCarSNN(nn.Module):
    def __init__(self, time_steps=5, alpha=2.0, dropout_rate=0.3, v_thresh=1.0):
        super().__init__()
        self.T = time_steps
        surrogate_gradient = surrogate.ATan(alpha=alpha)

        self.conv1 = layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = layer.BatchNorm2d(16)
        self.lif1 = neuron.ParametricLIFNode(init_tau=2.0, v_threshold=v_thresh,
                        detach_reset=True, surrogate_function=surrogate_gradient)

        self.conv2 = layer.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = layer.BatchNorm2d(32)
        self.lif2 = neuron.ParametricLIFNode(init_tau=2.0, v_threshold=v_thresh,
                        detach_reset=True, surrogate_function=surrogate_gradient)

        self.conv3 = layer.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm3 = layer.BatchNorm2d(64)
        self.lif3 = neuron.ParametricLIFNode(init_tau=2.0, v_threshold=v_thresh,
                        detach_reset=True, surrogate_function=surrogate_gradient)

        self.conv4 = layer.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm4 = layer.BatchNorm2d(128)
        self.lif4 = neuron.ParametricLIFNode(init_tau=2.0, v_threshold=v_thresh,
                        detach_reset=True, surrogate_function=surrogate_gradient)

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
            x_t = x[:, t:t+1, :, :]
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


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class TTCDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.tensor([self.y[idx]]).float()
        x = F.interpolate(x.unsqueeze(0), size=(240, 320),
                          mode='bilinear', align_corners=False).squeeze(0)
        return x, y


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
def weighted_loss(pred, target):
    weights = 1.0 / target.clamp(min=0.3)
    return (weights * torch.abs(pred - target)).mean()


# -----------------------------------------------------------------------------
# Fine-tune
# -----------------------------------------------------------------------------
def finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Fine-tuning on: {device}")

    model = SpikeCarSNN(time_steps=5).to(device)
    weights_path = Path('camera/best_snn_trial_25.pth')
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    print(f"Loaded weights from {weights_path}")

    for name, param in model.named_parameters():
        is_head = any(k in name for k in ['fc1', 'fc2', 'lif5', 'readout_lif'])
        param.requires_grad = is_head

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params: {trainable}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    train_ds = ConcatDataset([
        TTCDataset('data/processed/X_train.npy',  'data/processed/y_train.npy'),
        TTCDataset('data/processed/X_slider.npy', 'data/processed/y_slider.npy'),
    ])
    val_ds = TTCDataset('data/processed/X_val.npy', 'data/processed/y_val.npy')

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2)

    print(f"\nTrain samples: {len(train_ds)} | Val samples: {len(val_ds)}\n")

    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)

    best_val = float('inf')

    # Outer progress bar tracking epochs
    epoch_bar = tqdm(range(1, 41), desc='Epochs', unit='epoch', position=0)

    for epoch in epoch_bar:
        # ── Train ──
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f'  E{epoch:02d} train',
                         unit='batch', leave=False, position=1)
        for xb, yb in train_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            functional.reset_net(model)
            pred = model(xb)
            loss = weighted_loss(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        # ── Validate ──
        model.eval()
        val_loss = 0.0

        val_bar = tqdm(val_loader, desc=f'  E{epoch:02d} val  ',
                       unit='batch', leave=False, position=1)
        with torch.no_grad():
            for xb, yb in val_bar:
                xb, yb = xb.to(device), yb.to(device)
                functional.reset_net(model)
                pred = model(xb)
                loss = weighted_loss(pred, yb)
                val_loss += loss.item()
                val_bar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        # Update epoch bar with summary
        saved = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / 'finetuned_slider.pth')
            saved = ' ✓ saved'

        epoch_bar.set_postfix(
            train=f'{train_loss:.4f}',
            val=f'{val_loss:.4f}',
            best=f'{best_val:.4f}'
        )
        tqdm.write(
            f"Epoch {epoch:3d}/40 | train: {train_loss:.4f} | "
            f"val: {val_loss:.4f} | best: {best_val:.4f}{saved}"
        )

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Weights saved to: models/finetuned_slider.pth")


if __name__ == '__main__':
    finetune()
