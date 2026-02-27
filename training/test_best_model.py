#!/usr/bin/env python3
"""
Evaluate Trained SNN on Test Data
"""
import torch
from torch.utils.data import DataLoader
from train_snn import SpikeCarSNN, TTCDataset  # Import from your training script
import numpy as np
import matplotlib.pyplot as plt

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    test_ds = TTCDataset('data/processed/X_test.npy', 'data/processed/y_test.npy')
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # Load Model
    model = SpikeCarSNN().to(device)
    model.load_state_dict(torch.load("models/best_snn.pth"))
    model.eval()
    
    predictions = []
    actuals = []
    
    print("Running evaluation on Test Set...")
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            # Reset states!
            from spikingjelly.activation_based import functional
            functional.reset_net(model)
            
            pred = model(x)
            
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())
            
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Metrics
    mse = np.mean((predictions - actuals)**2)
    mae = np.mean(np.abs(predictions - actuals))
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f} seconds (Average Error)")
    
    # Plot Logic
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([0, 10], [0, 10], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual TTC (s)')
    plt.ylabel('Predicted TTC (s)')
    plt.title(f'SNN Predictions (MAE: {mae:.2f}s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/test_results.png')
    print("Saved plot to models/test_results.png")

if __name__ == "__main__":
    evaluate()
