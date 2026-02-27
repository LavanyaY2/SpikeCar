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

# -----------------------------------------------------------------------------
# 1. Dataset Class
# To create a custom PyTorch dataset
# -----------------------------------------------------------------------------
class TTCDataset(Dataset):
    def __init__(self, X_path, y_path):
        # Load data
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path)
        
    def __len__(self):
        # To determine end of an epoch
        return len(self.y)
    
    def __getitem__(self, idx):
        # Load a sample and convert to tensor
        # Input shape: (5, 480, 640) -> represents 5 time steps
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.tensor([self.y[idx]]).float()
        return x, y

# -----------------------------------------------------------------------------
# 2. SNN Architecture
# -----------------------------------------------------------------------------
class SpikeCarSNN(nn.Module):
    def __init__(self, time_steps=5):
        super().__init__()
        self.T = time_steps
        
        # Surrogate gradient (allows backprop through spikes)
        """
        Why surrogate gradients?
        - Standard ANNs use smooth activation functions that have well-defined derivatives
        - In SNNs, neuron outputs are spikes, modeled by a heaviside step function
        - The derivative for that is 0 everywhere, undefined at the threshold
        - Therefore, the gradients vanish
        - A surrogate gradient is a smooth, fake derivative used during backprop
        - Shape of the surrogate (a tunable hyperparam) matters to make sure gradients don't vanish/explode
        """
        # Replaces spike with an inverse tangent during backprop
        # ATan has "heavy tails" - spikes with voltage far away from the threshold will still have non-zero gradient
        surrogate_gradient = surrogate.ATan()

        """
        General layer structure
            - Convolutional layer with a kernel size of 3 = basically a 3x3 filter to find basic patterns
            - BatchNorm to normalize values to prevent vanishing/exploding gradients
            - LIFNode (Leaky integrate-and-fire) - has continuous internal memory; collects scattered events and integrates them together
        """
        # Layer 1: 640x480 -> 240x320
        self.conv1 = layer.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm1 = layer.BatchNorm2d(16)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate_gradient)

        # Layer 2: 240x320 -> 120x160
        self.conv2 = layer.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm2 = layer.BatchNorm2d(32)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate_gradient)

        # Layer 3: 120x160 -> 60x80
        self.conv3 = layer.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm3 = layer.BatchNorm2d(64)
        self.lif3 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate_gradient)

        # Layer 4: 60x80 -> 30x40
        self.conv4 = layer.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm4 = layer.BatchNorm2d(128)
        self.lif4 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate_gradient)

        """
        Global Average Pooling Layer:
            - Reduces 2D image data into a 1D array
        """
        self.global_average_pooling = layer.AdaptiveAvgPool2d((1, 1))

        # Standard fully-connected layer
        self.fc1 = layer.Linear(128, 64, bias=False)

        # Final spikng neuron layer in the network
        self.lif5 = neuron.LIFNode(tau=2.0, detach_reset=True)

        # Final layer - shrinks 64 spikes from last layer into 1 output
        self.fc2 = layer.Linear(64, 1, bias=False)

        # Final output layer is NOT spiking because we need a continuous regression value (TTC)
        # We will accumulate the membrane potential or simply sum the output over time.
        

    def forward(self, x):
        # x shape: (Batch, T, Height, Width) -> (Batch, 5, 480, 640)
        # In the data, the 5 channels are the 5 time bins
        # We treat the channels as time steps (T=5)
        # We need to reshape to (T, Batch, 1, H, W) for SpikingJelly logic if using functional.multi_step_forward
        # But here we'll write the loop explicitly for clarity.
        
        batch_size = x.shape[0]
        # Keep a running total of the predicted TTC for each of the 5 time-steps
        output_sum = 0
        
        # Loop over time steps (Direct Training BPTT happens here automatically by PyTorch)
        for t in range(self.T):
            # Extract the t-th bin: (Batch, 1, 480, 640)
            x_t = x[:, t:t+1, :, :] 
            
            # Layer 1
            out = self.lif1(self.batch_norm1(self.conv1(x_t)))
            # Layer 2
            out = self.lif2(self.batch_norm2(self.conv2(out)))
            # Layer 3
            out = self.lif3(self.batch_norm3(self.conv3(out)))
            # Layer 4
            out = self.lif4(self.batch_norm4(self.conv4(out)))
            
            # Pooling & Flatten
            out = self.global_average_pooling(out) # (Batch, 128, 1, 1)
            out = out.view(batch_size, -1) # (Batch, 128)
            
            # FC 1
            out = self.lif5(self.fc1(out))
            
            # FC 2 (Output)
            # We don't spike here. We sum the continuous output over time.
            out = self.fc2(out) 
            output_sum += out
            
        # Average the output over time steps to get predicted TTC
        return output_sum / self.T

# -----------------------------------------------------------------------------
# 3. Training Loop
# -----------------------------------------------------------------------------
def train_model():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path("models").mkdir(exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 4 # Reduce if OOM
    LR = 1e-3
    EPOCHS = 50
    
    # Data Loaders
    train_ds = TTCDataset('data/processed/X_train.npy', 'data/processed/y_train.npy')
    val_ds = TTCDataset('data/processed/X_val.npy', 'data/processed/y_val.npy')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model Init
    model = SpikeCarSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Regression loss
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss_epoch = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Reset neuron states before every batch! Important for SNNs.
            functional.reset_net(model)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            
        avg_train_loss = train_loss_epoch / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation ---
        model.eval()
        # Initialize running totals for ALL metrics
        val_mse_sum = 0
        val_mae_sum = 0
        val_rel_error_sum = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                functional.reset_net(model) # Reset states
                
                pred = model(x)

                # 1. MSE Loss (Squashes big errors)
                mse = torch.nn.functional.mse_loss(pred, y)
                val_mse_sum += mse.item()
                
                # 2. MAE Loss (Average seconds off)
                mae = torch.nn.functional.l1_loss(pred, y)
                val_mae_sum += mae.item()
                
                # 3. Relative Error % (Percentage off from true TTC)
                rel = torch.mean(torch.abs((y - pred) / y)) * 100
                val_rel_error_sum += rel.item()

        # Calculate the average across all batches in the validation set
        num_batches = len(val_loader)
        avg_val_mse = val_mse_sum / num_batches
        avg_val_mae = val_mae_sum / num_batches
        avg_val_rel_error = val_rel_error_sum / num_batches
            
        val_losses.append(avg_val_mse) # Still plotting MSE for the graph
                
        # Logging - Now it prints all the metrics!
        print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_mse:.4f} | Val MAE: {avg_val_mae:.4f}s | Val Rel Error: {avg_val_rel_error:.1f}%")
        
        # Save Best Model (Based on MSE)
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            torch.save(model.state_dict(), "models/best_snn.pth")
            print("  >>> Saved Best Model")
            
    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.1f} minutes.")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    
    # Plot Training Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('SNN Training Curve')
    plt.legend()
    plt.savefig('models/training_curve.png')
    print("Training curve saved to models/training_curve.png")

if __name__ == "__main__":
    train_model()
