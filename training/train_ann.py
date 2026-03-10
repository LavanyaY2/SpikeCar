import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


CONFIG = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3,
    "huber_delta": 0.5,
    "batch_size": 16,
    "epochs": 20,
    "target_h": 240,
    "target_w": 320,
    "seed": 42,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TTCDataset(Dataset):
    def __init__(self, X_path, y_path, target_h=240, target_w=320):
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path)
        self.target_h = target_h
        self.target_w = target_w

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()  # (5, H, W)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.target_h, self.target_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        y = torch.tensor([self.y[idx]], dtype=torch.float32)
        return x, y


class TTCBaselineCNN(nn.Module):
    def __init__(self, in_channels=5, dropout_rate=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device, delta):
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    rel_err_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))

            pred = model(x)
            loss_sum += F.huber_loss(pred, y, delta=delta).item()
            mae_sum += F.l1_loss(pred, y).item()
            rel_err_sum += torch.mean(torch.abs((y - pred) / y.clamp(min=0.1))).item() * 100

    n = len(loader)
    return {
        "loss": loss_sum / n,
        "mae": mae_sum / n,
        "rel_err": rel_err_sum / n,
    }


def train(config):
    set_seed(config["seed"])
    device = get_device()
    print(f"Device     : {device}")
    print(f"Config     : {config}\n")

    Path("models").mkdir(exist_ok=True)
    pin_memory = (device.type == "cuda")

    train_ds = TTCDataset("data/processed/X_train.npy", "data/processed/y_train.npy",
                          target_h=config["target_h"], target_w=config["target_w"])
    val_ds = TTCDataset("data/processed/X_val.npy", "data/processed/y_val.npy",
                        target_h=config["target_h"], target_w=config["target_w"])
    test_ds = TTCDataset("data/processed/X_test.npy", "data/processed/y_test.npy",
                         target_h=config["target_h"], target_w=config["target_w"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=0, pin_memory=pin_memory)

    model = TTCBaselineCNN(in_channels=5, dropout_rate=config["dropout_rate"]).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    delta = config["huber_delta"]

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rel_err": [], "lr": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 8

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0

        for x, y in tqdm(train_loader, desc=f"[Train] Epoch {epoch}", leave=False):
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            pred = model(x)
            loss = F.huber_loss(pred, y, delta=delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()

        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader)
        val_metrics = evaluate(model, val_loader, device, delta)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rel_err"].append(val_metrics["rel_err"])
        history["lr"].append(current_lr)

        improved = val_metrics["loss"] < best_val_loss
        print(
            f"Epoch {epoch:02d} | "
            f"train {avg_train_loss:.4f} | "
            f"val {val_metrics['loss']:.4f} | "
            f"mae {val_metrics['mae']:.4f} | "
            f"rel {val_metrics['rel_err']:.1f}% | "
            f"lr {current_lr:.2e} | "
            f"{time.time() - epoch_start:.1f}s"
            + ("  <<<" if improved else "")
        )

        if improved:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["mae"],
                    "config": config,
                },
                "models/best_cnn_baseline.pth"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} stagnant epochs.")
                break

    checkpoint = torch.load("models/best_cnn_baseline.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, delta)

    print("\nFinal test metrics:")
    print(f"  Test Loss   : {test_metrics['loss']:.4f}")
    print(f"  Test MAE    : {test_metrics['mae']:.4f}")
    print(f"  Test RelErr : {test_metrics['rel_err']:.2f}%")

    return model, history, test_metrics


if __name__ == "__main__":
    train(CONFIG)