"""
PyTorch Deep Multiple Perceptron System for Wireless Power Transfer (WPT)
========================================================================
Implements a deep Multiple Perceptron System in PyTorch to predict Mutual Inductance
(Mutual_Inductance_H) and related electrical parameters featuring:
  1. Configurable Multi-Layer Perceptron (MLP) architecture.
  2. Multi-Head Perceptron Architecture (Shared representation + dedicated heads).
  3. Batch Normalization, Dropout, and ReLU activations.
  4. AdamW optimizer with Learning Rate Scheduling and Early Stopping.
  5. Automatic device selection (CUDA / CPU).
  6. Checkpointing (.pth) and loss curve generation.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "WPT_1000_Final_Dataset.csv")
DEFAULT_SAVE_DIR = os.path.join(SCRIPT_DIR, "models", "pytorch")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "pytorch")

DEFAULT_FEATURES = [
    'Frequency_kHz', 'L1_H', 'L2_H', 'k', 'R1_Ohm', 'R2_Ohm',
    'RL_Ohm', 'C1_F', 'C2_F', 'Vin_RMS_V'
]

DEFAULT_TARGETS = [
    'Mutual_Inductance_H'
]


if TORCH_AVAILABLE:
    class WPTDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class MultiplePerceptronNetwork(nn.Module):
        """
        Deep Multilayer Perceptron with configurable hidden layers,
        Batch Normalization, Dropout, and non-linear activations.
        """
        def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32], dropout_rate=0.05):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class MultiHeadPerceptronSystem(nn.Module):
        """
        Multi-Head Perceptron Architecture: Shared backbone feature extractor
        with dedicated perceptron heads for each WPT target metric.
        """
        def __init__(self, input_dim, target_names, shared_dims=[128, 64], head_dims=[32]):
            super().__init__()
            self.target_names = target_names

            # Shared feature extractor backbone
            backbone = []
            prev = input_dim
            for dim in shared_dims:
                backbone.append(nn.Linear(prev, dim))
                backbone.append(nn.BatchNorm1d(dim))
                backbone.append(nn.ReLU())
                prev = dim
            self.shared_backbone = nn.Sequential(*backbone)

            # Dedicated output heads
            self.heads = nn.ModuleDict()
            for name in target_names:
                head_layers = []
                h_prev = prev
                for h_dim in head_dims:
                    head_layers.append(nn.Linear(h_prev, h_dim))
                    head_layers.append(nn.ReLU())
                    h_prev = h_dim
                head_layers.append(nn.Linear(h_prev, 1))
                self.heads[name] = nn.Sequential(*head_layers)

        def forward(self, x):
            features = self.shared_backbone(x)
            outputs = [self.heads[name](features) for name in self.target_names]
            return torch.cat(outputs, dim=1)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train PyTorch Multiple Perceptron Model to predict Mutual Inductance on WPT Dataset.")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--architecture', type=str, choices=['mlp', 'multi_head'], default='mlp')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience.")
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def train_pytorch(args):
    if not TORCH_AVAILABLE:
        print("\n[NOTICE] PyTorch ('torch') is not installed in this Python environment.")
        print("To use the PyTorch Multiple Perceptron pipeline, install it via:")
        print("    pip install torch\n")
        print("You can run the primary scikit-learn training pipeline right now via:")
        print("    python train_mlp_system.py\n")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using compute device: {device}")

    # Load data
    data_path = args.data_path
    if not os.path.exists(data_path):
        fallback = os.path.join(SCRIPT_DIR, "data", "WPT_1000_Final_Dataset.csv")
        if os.path.exists(fallback):
            data_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if 'Simulation_Status' in df.columns:
        df = df[df['Simulation_Status'] == 'OK'].copy()

    X = df[DEFAULT_FEATURES].values
    y = df[DEFAULT_TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    train_dataset = WPTDataset(X_train_scaled, y_train_scaled)
    test_dataset = WPTDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    if args.architecture == 'multi_head':
        model = MultiHeadPerceptronSystem(input_dim=len(DEFAULT_FEATURES), target_names=DEFAULT_TARGETS)
    else:
        model = MultiplePerceptronNetwork(input_dim=len(DEFAULT_FEATURES), output_dim=len(DEFAULT_TARGETS))
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_loss = float('inf')
    best_weights = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"\n[INFO] Commencing PyTorch Training for Mutual Inductance ({args.architecture})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(batch_X)
        epoch_train_loss = running_train_loss / len(train_dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                running_val_loss += loss.item() * len(batch_X)
        epoch_val_loss = running_val_loss / len(test_dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{args.epochs}] - Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        preds_scaled = model(X_test_tensor).cpu().numpy()
        preds = scaler_y.inverse_transform(preds_scaled)

    # Metrics
    metrics = {}
    print("\n" + "=" * 90)
    print(f"{'Target Variable':<24} | {'R2 Score':<10} | {'RMSE':<14} | {'MAE':<14} | {'MAPE (%)':<10}")
    print("-" * 90)
    for idx, target in enumerate(DEFAULT_TARGETS):
        yt = y_test[:, idx]
        yp = preds[:, idx]
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae = mean_absolute_error(yt, yp)
        mape = mean_absolute_percentage_error(yt, yp) * 100.0
        metrics[target] = {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae), 'MAPE': float(mape)}
        print(f"{target:<24} | {r2:<10.4f} | {rmse:<14.4e} | {mae:<14.4e} | {mape:<10.2f}")
    print("=" * 90)

    # Plot loss curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(train_losses, label="Train Loss (MSE)", linewidth=1.5)
    plt.plot(val_losses, label="Validation Loss (MSE)", linewidth=1.5)
    plt.title("PyTorch Multiple Perceptron Loss Curve", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Normalized)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "pytorch_loss_curves.png"), dpi=300)
    plt.close()

    # Save checkpoint
    checkpoint_path = os.path.join(args.save_dir, "pytorch_mlp_system.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': args.architecture,
        'features': DEFAULT_FEATURES,
        'targets': DEFAULT_TARGETS,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'metrics': metrics
    }, checkpoint_path)
    print(f"[INFO] PyTorch model successfully saved to: {checkpoint_path}")


if __name__ == '__main__':
    args = parse_arguments()
    train_pytorch(args)
