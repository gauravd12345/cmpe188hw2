"""
task_id : linreg_lvl1_raw_tensors_new
series  : Linear Regression (Neural Network edition)
level   : 1
algorithm: Linear Regression (nn.Linear, SGD, California Housing)

Description:
    Linear Regression implemented as a single nn.Linear layer (no activation),
    trained with MSELoss and SGD + momentum. Dataset: California Housing.
    Compared against sklearn LinearRegression. Demonstrates that a no-activation
    single-layer NN IS linear regression.

Math:
    Model:      y_hat = W * x + b    (h_theta(x) = theta_0 + theta_1*x_1 + ... + theta_n*x_n)

    MSE cost:   J(theta) = (1/N) * sum_i (y_hat_i - y_i)^2

    Gradient descent update (SGD with momentum):
        v     <- momentum * v + (1-momentum) * grad J(theta)
        theta <- theta - lr * v

    R^2 score:  R^2 = 1 - SS_res / SS_tot
                SS_res = sum(y - y_hat)^2
                SS_tot = sum(y - y_bar)^2

Protocol: pytorch_task_v1
Entrypoint: python tasks/linreg_lvl1_raw_tensors_new/task.py
"""

import os
import sys
import json
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Module-level setup ─────────────────────────────────────────────────────────
OUTPUT_DIR = 'tasks/linreg_lvl1_raw_tensors_new/artifacts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
#  1. Metadata
# ─────────────────────────────────────────────
def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id":    "linreg_lvl1_raw_tensors_new",
        "series":     "Linear Regression",
        "level":      1,
        "algorithm":  "Linear Regression (nn.Linear)",
        "dataset":    "California Housing",
        "frameworks": ["pytorch", "sklearn"],
        "metrics":    ["mse", "rmse", "r2"],
        "output_dir": OUTPUT_DIR,
    }


# ─────────────────────────────────────────────
#  2. Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
#  3. Device
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
#  4. Data
# ─────────────────────────────────────────────
def make_dataloaders(
    batch_size: int = 512,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Dict[str, Any]:
    """
    Loads California Housing, applies StandardScaler on train split.
    Returns dict with train_loader, val_loader, test_loader, and raw arrays for sklearn.
    """
    housing = fetch_california_housing()
    X = housing.data.astype(np.float32)
    y = housing.target.astype(np.float32)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_fraction / (1.0 - test_fraction),
        random_state=42,
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return {
        'train_loader': to_loader(X_train, y_train, True),
        'val_loader':   to_loader(X_val,   y_val,   False),
        'test_loader':  to_loader(X_test,  y_test,  False),
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        'scaler':  scaler,
        'n_features':    X_train.shape[1],
        'feature_names': list(housing.feature_names),
    }


# ─────────────────────────────────────────────
#  5. Model — sklearn-style API on the class
# ─────────────────────────────────────────────
class LinearRegressionNN(nn.Module):
    """
    Single nn.Linear layer — exactly OLS linear regression.
    No activation: y_hat = W * x + b

    Provides sklearn-style .fit() / .predict() / .save() / .load() methods.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 150,
        lr: float = 0.05,
        momentum: float = 0.9,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train with SGD + momentum; return history."""
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'train_losses': [], 'val_losses': [],
                   'train_r2':     [], 'val_r2':     []}

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            all_pred, all_true = [], []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = self(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                all_pred.append(pred.detach().cpu())
                all_true.append(y.cpu())
            scheduler.step()

            train_mse = total_loss / len(train_loader.dataset)
            train_r2  = _r2(torch.cat(all_true), torch.cat(all_pred))
            history['train_losses'].append(train_mse)
            history['train_r2'].append(train_r2)

            if val_loader is not None:
                val_m = evaluate(self, val_loader)
                history['val_losses'].append(val_m['mse'])
                history['val_r2'].append(val_m['r2'])
            else:
                history['val_losses'].append(None)
                history['val_r2'].append(None)

            if verbose and (epoch % 30 == 0 or epoch == 1):
                val_str = (f"val_MSE={history['val_losses'][-1]:.4f}  "
                           f"val_R2={history['val_r2'][-1]:.4f}"
                           if val_loader else "")
                print(f"  Epoch [{epoch}/{epochs}]  "
                      f"train_MSE={train_mse:.4f}  train_R2={train_r2:.4f}  {val_str}")

        return history

    def predict(self, x) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            return self(x.to(device)).cpu().numpy()

    def save(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath, map_location=device))


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1.0 - ss_res / ss_tot).item()


def build_model(input_dim: int = 8) -> LinearRegressionNN:
    return LinearRegressionNN(input_dim=input_dim)


# ─────────────────────────────────────────────
#  6. Train
# ─────────────────────────────────────────────
def train(
    model: LinearRegressionNN,
    dataloaders: Dict[str, Any],
    epochs: int = 150,
    lr: float = 0.05,
    verbose: bool = True,
) -> Dict[str, Any]:
    return model.fit(
        dataloaders['train_loader'],
        val_loader=dataloaders['val_loader'],
        epochs=epochs, lr=lr, verbose=verbose,
    )


# ─────────────────────────────────────────────
#  7. Evaluate
# ─────────────────────────────────────────────
def evaluate(model: LinearRegressionNN, loader: DataLoader) -> Dict[str, float]:
    """Returns dict: {mse, rmse, r2}."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu()
            all_pred.append(pred)
            all_true.append(y)

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    mse  = ((pred - true) ** 2).mean().item()
    rmse = mse ** 0.5
    r2   = _r2(true, pred)

    return {'mse': mse, 'rmse': rmse, 'r2': r2}


# ─────────────────────────────────────────────
#  8. Predict
# ─────────────────────────────────────────────
def predict(model: LinearRegressionNN, x) -> np.ndarray:
    return model.predict(x)


# ─────────────────────────────────────────────
#  9. Save artifacts
# ─────────────────────────────────────────────
def save_artifacts(
    model: LinearRegressionNN,
    history: Dict[str, Any],
    metrics: Dict[str, Any],
    sklearn_metrics: Dict[str, float],
) -> None:
    model.save(os.path.join(OUTPUT_DIR, 'model.pt'))

    w = model.linear.weight.detach().cpu().numpy().flatten().tolist()
    b = float(model.linear.bias.detach().cpu().item())
    all_out = {
        'metadata':        get_task_metadata(),
        'pytorch_metrics': metrics,
        'sklearn_metrics': sklearn_metrics,
        'comparison': {
            'r2_diff':   abs(metrics['val']['r2'] - sklearn_metrics['r2']),
            'rmse_diff': abs(metrics['val']['rmse'] - sklearn_metrics['rmse']),
        },
        'coefficients': {'weights': w, 'bias': b},
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(all_out, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Plot: training loss curve ──
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label='Train MSE')
    if history['val_losses'][0] is not None:
        plt.plot(history['val_losses'], label='Val MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Training History — Linear Regression NN')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg_lvl1_training_history.png'), dpi=150)
    plt.close()

    # ── Plot: R2 comparison bar ──
    labels_bar = ['Train (PyTorch)', 'Val (PyTorch)', 'Test (PyTorch)', 'Sklearn']
    values_bar = [
        metrics['train']['r2'],
        metrics['val']['r2'],
        metrics['test']['r2'],
        sklearn_metrics['r2'],
    ]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels_bar, values_bar, color=['steelblue', 'seagreen', 'darkorange', 'salmon'])
    plt.title('R2 Comparison: PyTorch Linear NN vs Sklearn LinearRegression')
    plt.ylabel('R2 Score')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{bar.get_height():.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg_lvl1_model_comparison.png'), dpi=150)
    plt.close()

    print(f"  Artifacts saved to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('Linear Regression (NN) vs Sklearn -- California Housing')
    print('=' * 60)

    set_seed(42)
    metadata = get_task_metadata()
    print(f"\nTask:    {metadata['task_id']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Device:  {device}")

    R2_THRESHOLD   = 0.60
    SKLEARN_DELTA  = 0.05
    EPOCHS         = 150

    # ── [1/5] Data ──
    print('\n[1/5] Loading and preprocessing data...')
    dataloaders = make_dataloaders()
    print(f"  Train samples : {len(dataloaders['X_train'])}")
    print(f"  Val   samples : {len(dataloaders['X_val'])}")
    print(f"  Test  samples : {len(dataloaders['X_test'])}")
    print(f"  Features      : {dataloaders['n_features']}")

    # ── [2/5] sklearn baseline ──
    print('\n[2/5] Training Sklearn LinearRegression...')
    sk_model = SklearnLR()
    sk_model.fit(dataloaders['X_train'], dataloaders['y_train'])
    sk_val_pred  = sk_model.predict(dataloaders['X_val'])
    sk_test_pred = sk_model.predict(dataloaders['X_test'])
    sklearn_metrics = {
        'mse':  float(mean_squared_error(dataloaders['y_val'], sk_val_pred)),
        'rmse': float(np.sqrt(mean_squared_error(dataloaders['y_val'], sk_val_pred))),
        'r2':   float(r2_score(dataloaders['y_val'], sk_val_pred)),
    }
    print(f"  Sklearn val  R2  : {sklearn_metrics['r2']:.4f}")
    print(f"  Sklearn val  RMSE: {sklearn_metrics['rmse']:.4f}")

    # ── [3/5] Build & train PyTorch model ──
    print('\n[3/5] Building and training Linear Regression NN...')
    model   = build_model(input_dim=dataloaders['n_features'])
    print(f"  Architecture: {model}")
    history = train(model, dataloaders, epochs=EPOCHS, lr=0.05, verbose=True)

    # ── [4/5] Evaluate on train AND val AND test ──
    print('\n[4/5] Evaluating on train and validation splits...')
    print('\n  --- Train split ---')
    train_metrics = evaluate(model, dataloaders['train_loader'])
    for k, v in train_metrics.items():
        print(f"    {k:8s}: {v:.6f}")

    print('\n  --- Validation split ---')
    val_metrics = evaluate(model, dataloaders['val_loader'])
    for k, v in val_metrics.items():
        print(f"    {k:8s}: {v:.6f}")

    print('\n  --- Test split ---')
    test_metrics = evaluate(model, dataloaders['test_loader'])
    for k, v in test_metrics.items():
        print(f"    {k:8s}: {v:.6f}")

    # ── [5/5] Save artifacts ──
    print('\n[5/5] Saving artifacts...')
    all_metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
    save_artifacts(model, history, all_metrics, sklearn_metrics)

    # ── Comparison ──
    r2_diff = abs(val_metrics['r2'] - sklearn_metrics['r2'])
    print(f"\n--- Comparison ---")
    print(f"  PyTorch val R2  : {val_metrics['r2']:.4f}")
    print(f"  Sklearn val R2  : {sklearn_metrics['r2']:.4f}")
    print(f"  R2 Difference   : {r2_diff:.4f}  (threshold < {SKLEARN_DELTA})")
    if r2_diff < SKLEARN_DELTA:
        print("  Both models perform similarly (R2 difference < 0.05)")
    else:
        print("  Significant performance difference between models")

    w = model.linear.weight.detach().cpu().numpy().flatten()
    b = model.linear.bias.detach().cpu().item()
    print(f"\n  Learned weights : {np.round(w, 4)}")
    print(f"  Learned bias    : {b:.4f}")

    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")
    print('=' * 60)

    try:
        assert val_metrics['r2']  > R2_THRESHOLD, \
            f"Val R2 {val_metrics['r2']:.4f} <= {R2_THRESHOLD}"
        assert test_metrics['r2'] > R2_THRESHOLD, \
            f"Test R2 {test_metrics['r2']:.4f} <= {R2_THRESHOLD}"
        assert r2_diff < SKLEARN_DELTA, \
            f"PyTorch vs sklearn R2 gap {r2_diff:.4f} > {SKLEARN_DELTA}"
        print('Task completed successfully!')
        print('=' * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f'\nFAIL -- {e}')
        sys.exit(1)
