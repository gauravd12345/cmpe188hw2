"""
Task 4: Logistic Regression implemented as a Neural Network
Dataset: Breast Cancer Wisconsin (sklearn)
Protocol: pytorch_task_v1

Logistic Regression as a NN:
  • Single nn.Linear layer + Sigmoid
  • Loss: BCEWithLogitsLoss
  • Optimiser: Adam with L2 regularisation 
  • Metric: Accuracy
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np

class LogisticRegressionNN(nn.Module):
    """
    Binary Logistic Regression as a NN.
    One linear layer; sigmoid applied outside (BCEWithLogitsLoss handles it).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)   # raw logits, shape (B,)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))

def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds  = (logits >= 0).long()
    acc    = (preds == labels).float().mean().item()
    proba  = torch.sigmoid(logits).numpy()
    auc    = roc_auc_score(labels.numpy(), proba)
    # Non-trivial check: model must predict both classes
    unique_preds = preds.unique().numel()
    return acc, auc, unique_preds

def main():
    EPOCHS        = 200
    BATCH_SIZE    = 64
    LR            = 1e-3
    WEIGHT_DECAY  = 1e-4       # L2 regularisation 
    TARGET_ACC    = 0.93
    TARGET_AUC    = 0.97
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Task 4 – Logistic Regression NN]  device={DEVICE}")

    # ── Data ──
    data  = load_breast_cancer()
    X, y  = data.data.astype(np.float32), data.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train)
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)),
            torch.from_numpy(y_test)
        ),
        batch_size=BATCH_SIZE
    )

    # ── Model ──
    model     = LogisticRegressionNN(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # ── Training ──
    best_auc = 0.0
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

        avg_loss = total_loss / len(train_loader.dataset)
        scheduler.step(avg_loss)

        if epoch % 40 == 0 or epoch == 1:
            acc, auc, _ = evaluate(model, test_loader, DEVICE)
            best_auc = max(best_auc, auc)
            print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={avg_loss:.4f}  "
                  f"test_acc={acc:.4f}  AUC-ROC={auc:.4f}")

    # ── Final evaluation ──
    final_acc, final_auc, n_unique = evaluate(model, test_loader, DEVICE)

    print(f"\n[Task 4 – Logistic Regression NN]")
    print(f"  Final accuracy : {final_acc:.4f}  (target ≥ {TARGET_ACC})")
    print(f"  Final AUC-ROC  : {final_auc:.4f}  (target ≥ {TARGET_AUC})")
    print(f"  Unique classes predicted: {n_unique}  (must be 2 — non-trivial boundary)")

    # Print top-5 most influential features
    w = model.linear.weight.detach().cpu().numpy().flatten()
    top5 = np.argsort(np.abs(w))[::-1][:5]
    print(f"  Top-5 feature indices by |weight|: {top5.tolist()}")
    print(f"    → {[data.feature_names[i] for i in top5]}")

    if final_acc >= TARGET_ACC and final_auc >= TARGET_AUC and n_unique == 2:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
