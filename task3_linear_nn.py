"""
Task 3: Linear Regression implemented as a Neural Network
Dataset: California Housing (sklearn)
Protocol: pytorch_task_v1

Linear Regression as a NN:
  • Single nn.Linear layer
  • Loss: MSELoss 
  • Optimiser: SGD with momentum  
  • Metric: R² coefficient of determination
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class LinearRegressionNN(nn.Module):
    """Single-layer linear model: ŷ = W·x + b  (no activation)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)   # (B,)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


def evaluate_r2(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    return r2_score(torch.cat(targets), torch.cat(preds))

def main():
    EPOCHS      = 100
    BATCH_SIZE  = 512
    LR          = 0.01
    MOMENTUM    = 0.9
    TARGET_R2   = 0.60      # R² ≥ 0.60 → pass
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Task 3 – Linear Regression NN]  device={DEVICE}")

    # ── Data ──
    housing = fetch_california_housing()
    X, y    = housing.data, housing.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader  = DataLoader(
        TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE
    )

    # ── Model ──
    model     = LinearRegressionNN(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training ──
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_b.size(0)
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            r2 = evaluate_r2(model, test_loader, DEVICE)
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"  Epoch {epoch:>3}/{EPOCHS}  MSE={avg_loss:.4f}  test_R²={r2:.4f}")

    # ── Final evaluation ──
    final_r2 = evaluate_r2(model, test_loader, DEVICE)
    print(f"\n[Task 3 – Linear Regression NN]  Final test R²: {final_r2:.4f}  (target ≥ {TARGET_R2})")

    # Also print learned coefficients for interpretability
    w = model.linear.weight.detach().cpu().numpy().flatten()
    b = model.linear.bias.detach().cpu().item()
    print(f"  Learned weights : {np.round(w, 4)}")
    print(f"  Learned bias    : {b:.4f}")

    if final_r2 >= TARGET_R2:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
