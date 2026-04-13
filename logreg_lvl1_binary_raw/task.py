"""
task_id : logreg_lvl1_binary_raw
series  : Logistic Regression
level   : 1
algorithm: Logistic Regression (Binary, as Neural Network)
Protocol: pytorch_task_v1
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
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Module-level setup ─────────────────────────────────────────────────────────
OUTPUT_DIR = 'tasks/logreg_lvl1_binary_raw/artifacts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id":    "logreg_lvl1_binary_raw",
        "series":     "Logistic Regression",
        "level":      1,
        "algorithm":  "Logistic Regression (Binary, nn.Linear)",
        "dataset":    "Breast Cancer Wisconsin",
        "frameworks": ["pytorch", "sklearn"],
        "metrics":    ["accuracy", "auc", "mse", "r2"],
        "output_dir": OUTPUT_DIR,
    }

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(
    batch_size: int = 64,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Dict[str, Any]:
    """
    Loads Breast Cancer Wisconsin, standardises features.
    Returns dict with train_loader, val_loader, test_loader, and raw arrays for sklearn.
    """
    data = load_breast_cancer()
    X    = data.data.astype(np.float32)
    y    = data.target.astype(np.float32)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_fraction / (1.0 - test_fraction),
        random_state=42, stratify=y_tv,
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
        'feature_names': list(data.feature_names),
    }

class LogisticRegressionNN(nn.Module):
    """
    Binary Logistic Regression as a single nn.Linear layer.
    Raw logit output; sigmoid is applied inside BCEWithLogitsLoss (numerically stable).

    Provides sklearn-style .fit() / .predict() / .save() / .load() methods.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)   # raw logits (B,)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train with Adam + L2 regularisation; return history."""
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=30, factor=0.5
        )

        history = {'train_losses': [], 'val_losses': [],
                   'train_acc':    [], 'val_acc':    []}

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = correct = total = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = self(x)
                loss   = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                correct    += ((logits >= 0).float() == y).sum().item()
                total      += y.size(0)

            val_m = evaluate(self, val_loader) if val_loader else {'loss': None, 'accuracy': None}
            scheduler.step(val_m['loss'] if val_m['loss'] is not None else total_loss / total)
            history['train_losses'].append(total_loss / total)
            history['train_acc'].append(correct / total)
            history['val_losses'].append(val_m['loss'])
            history['val_acc'].append(val_m['accuracy'])

            if verbose and (epoch % 60 == 0 or epoch == 1):
                val_str = (f"val_loss={val_m['loss']:.4f}  val_acc={val_m['accuracy']:.4f}"
                           if val_loader else "")
                print(f"  Epoch [{epoch}/{epochs}]  "
                      f"train_loss={total_loss/total:.4f}  "
                      f"train_acc={correct/total:.4f}  {val_str}")

        return history

    def predict(self, x) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            logits = self(x.to(device))
        return (logits >= 0).long().cpu().numpy()

    def predict_proba(self, x) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            logits = self(x.to(device))
        return torch.sigmoid(logits).cpu().numpy()

    def save(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath, map_location=device))


def build_model(input_dim: int = 30) -> LogisticRegressionNN:
    return LogisticRegressionNN(input_dim=input_dim)

def train(
    model: LogisticRegressionNN,
    dataloaders: Dict[str, Any],
    epochs: int = 300,
    lr: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, Any]:
    return model.fit(
        dataloaders['train_loader'],
        val_loader=dataloaders['val_loader'],
        epochs=epochs, lr=lr, verbose=verbose,
    )

def evaluate(model: LogisticRegressionNN, loader: DataLoader) -> Dict[str, float]:
    """
    Returns dict: {loss, accuracy, mse, r2, auc, n_unique_preds}.
    MSE and R2 are computed between predicted probabilities and binary labels.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            total_loss  += criterion(logits, y).item() * x.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    n      = labels.size(0)

    avg_loss = total_loss / n
    probs    = torch.sigmoid(logits)
    preds    = (logits >= 0).float()
    accuracy = (preds == labels).float().mean().item()

    mse    = ((probs - labels) ** 2).mean().item()
    ss_res = ((probs - labels) ** 2).sum().item()
    ss_tot = ((labels - labels.mean()) ** 2).sum().item()
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)

    auc         = roc_auc_score(labels.numpy(), probs.numpy())
    n_unique    = int(preds.unique().numel())

    return {
        'loss':           avg_loss,
        'accuracy':       accuracy,
        'mse':            mse,
        'r2':             r2,
        'auc':            auc,
        'n_unique_preds': n_unique,
    }

def predict(model: LogisticRegressionNN, x) -> np.ndarray:
    return model.predict(x)

def save_artifacts(
    model: LogisticRegressionNN,
    history: Dict[str, Any],
    metrics: Dict[str, Any],
    sklearn_metrics: Dict[str, float],
    feature_names=None,
) -> None:
    model.save(os.path.join(OUTPUT_DIR, 'model.pt'))

    w    = model.linear.weight.detach().cpu().numpy().flatten().tolist()
    b    = float(model.linear.bias.detach().cpu().item())
    all_out = {
        'metadata':        get_task_metadata(),
        'pytorch_metrics': metrics,
        'sklearn_metrics': sklearn_metrics,
        'comparison': {
            'acc_diff': abs(metrics['val']['accuracy'] - sklearn_metrics['accuracy']),
            'auc_diff': abs(metrics['val']['auc']      - sklearn_metrics['auc']),
        },
        'coefficients': {'weights': w, 'bias': b},
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(all_out, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Plot: training curves ──
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    if history['val_losses'][0] is not None:
        plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('BCE Loss')
    plt.title('Training Loss -- Logistic Regression NN')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if history['val_acc'][0] is not None:
        plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training Accuracy -- Logistic Regression NN')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'logreg_lvl1_training_curves.png'), dpi=150)
    plt.close()

    # ── Plot: model comparison bar chart ──
    labels_bar = ['Train (PyTorch)', 'Val (PyTorch)', 'Test (PyTorch)', 'Sklearn']
    values_bar = [
        metrics['train']['accuracy'],
        metrics['val']['accuracy'],
        metrics['test']['accuracy'],
        sklearn_metrics['accuracy'],
    ]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels_bar, values_bar, color=['steelblue', 'seagreen', 'darkorange', 'salmon'])
    plt.ylim(0, 1.1)
    plt.title('Accuracy Comparison: PyTorch Logistic NN vs Sklearn')
    plt.ylabel('Accuracy')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{bar.get_height():.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'logreg_lvl1_model_comparison.png'), dpi=150)
    plt.close()

    # ── Print top feature weights ──
    if feature_names is not None:
        w_arr = np.array(w)
        top5  = np.argsort(np.abs(w_arr))[::-1][:5]
        print(f"  Top-5 features by |weight|: {[feature_names[i] for i in top5]}")

    print(f"  Artifacts saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    print('=' * 60)
    print('Logistic Regression (NN) vs Sklearn -- Breast Cancer')
    print('=' * 60)

    set_seed(42)
    metadata = get_task_metadata()
    print(f"\nTask:    {metadata['task_id']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Device:  {device}")

    ACC_THRESHOLD = 0.93
    AUC_THRESHOLD = 0.97
    SKLEARN_DELTA = 0.03
    EPOCHS        = 300

    # ── [1/5] Data ──
    print('\n[1/5] Loading and preprocessing data...')
    dataloaders = make_dataloaders()
    print(f"  Train samples : {len(dataloaders['X_train'])}")
    print(f"  Val   samples : {len(dataloaders['X_val'])}")
    print(f"  Test  samples : {len(dataloaders['X_test'])}")
    print(f"  Features      : {dataloaders['n_features']}")

    # ── [2/5] sklearn baseline ──
    print('\n[2/5] Training Sklearn LogisticRegression...')
    sk_model = SklearnLR(max_iter=1000, random_state=42)
    sk_model.fit(dataloaders['X_train'], dataloaders['y_train'])
    sk_val_pred  = sk_model.predict(dataloaders['X_val'])
    sk_val_proba = sk_model.predict_proba(dataloaders['X_val'])[:, 1]
    sklearn_metrics = {
        'accuracy': float(accuracy_score(dataloaders['y_val'], sk_val_pred)),
        'auc':      float(roc_auc_score(dataloaders['y_val'], sk_val_proba)),
    }
    print(f"  Sklearn val  accuracy : {sklearn_metrics['accuracy']:.4f}")
    print(f"  Sklearn val  AUC      : {sklearn_metrics['auc']:.4f}")

    # ── [3/5] Build & train PyTorch model ──
    print('\n[3/5] Building and training Logistic Regression NN...')
    model   = build_model(input_dim=dataloaders['n_features'])
    print(f"  Architecture: {model}")
    history = train(model, dataloaders, epochs=EPOCHS, lr=1e-3, verbose=True)

    # ── [4/5] Evaluate on train AND val AND test ──
    print('\n[4/5] Evaluating on train and validation splits...')
    print('\n  --- Train split ---')
    train_metrics = evaluate(model, dataloaders['train_loader'])
    for k, v in train_metrics.items():
        print(f"    {k:20s}: {v}")

    print('\n  --- Validation split ---')
    val_metrics = evaluate(model, dataloaders['val_loader'])
    for k, v in val_metrics.items():
        print(f"    {k:20s}: {v}")

    print('\n  --- Test split ---')
    test_metrics = evaluate(model, dataloaders['test_loader'])
    for k, v in test_metrics.items():
        print(f"    {k:20s}: {v}")

    # ── [5/5] Save artifacts ──
    print('\n[5/5] Saving artifacts...')
    all_metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
    save_artifacts(model, history, all_metrics, sklearn_metrics,
                   feature_names=dataloaders['feature_names'])

    # ── Comparison ──
    acc_diff = abs(val_metrics['accuracy'] - sklearn_metrics['accuracy'])
    print(f"\n--- Comparison ---")
    print(f"  PyTorch val accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  Sklearn val accuracy : {sklearn_metrics['accuracy']:.4f}")
    print(f"  Accuracy difference  : {acc_diff:.4f}")
    print(f"  PyTorch val AUC      : {val_metrics['auc']:.4f}")
    print(f"  Sklearn val AUC      : {sklearn_metrics['auc']:.4f}")
    print(f"  Non-trivial boundary : {val_metrics['n_unique_preds']} class(es) predicted (must be 2)")

    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")
    print('=' * 60)

    try:
        assert val_metrics['accuracy']      > ACC_THRESHOLD, \
            f"Val accuracy {val_metrics['accuracy']:.4f} <= {ACC_THRESHOLD}"
        assert val_metrics['auc']           > AUC_THRESHOLD, \
            f"Val AUC {val_metrics['auc']:.4f} <= {AUC_THRESHOLD}"
        assert val_metrics['n_unique_preds'] == 2, \
            f"Model predicts only {val_metrics['n_unique_preds']} class(es) -- degenerate boundary"
        assert test_metrics['accuracy']     > ACC_THRESHOLD, \
            f"Test accuracy {test_metrics['accuracy']:.4f} <= {ACC_THRESHOLD}"
        print('Task completed successfully!')
        print('=' * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f'\nFAIL -- {e}')
        sys.exit(1)
