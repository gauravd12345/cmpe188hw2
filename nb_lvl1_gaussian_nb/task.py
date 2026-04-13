"""
task_id : nb_lvl1_gaussian_nb
series  : Naive Bayes
level   : 1
algorithm: Gaussian Naive Bayes (as a Neural Network)

Description:
    Implements Gaussian Naive Bayes from first principles in PyTorch,
    expressed as a differentiable neural network layer. Parameters
    (class means, log-variances, log-priors) are learned via gradient
    descent on NLL loss. Compared against sklearn GaussianNB.

Math:
    Gaussian likelihood per feature f for class c:
        p(x_f | c) = (1 / sqrt(2*pi*sigma^2_cf)) * exp(-(x_f - mu_cf)^2 / (2*sigma^2_cf))

    Log-posterior (unnormalised):
        log P(c | x) = log pi_c + sum_f log p(x_f | c)
                     = log pi_c - 0.5 * sum_f [ log(2*pi) + log(sigma^2_cf) + (x_f-mu_cf)^2 / sigma^2_cf ]

    Classification: argmax_c log P(c | x)

    Accuracy within 3% of sklearn GaussianNB required.

Protocol: pytorch_task_v1
Entrypoint: python tasks/nb_lvl1_gaussian_nb/task.py
"""

import os
import sys
import json
import random
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Module-level setup ─────────────────────────────────────────────────────────
OUTPUT_DIR = 'tasks/nb_lvl1_gaussian_nb/artifacts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
#  1. Metadata
# ─────────────────────────────────────────────
def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id":    "nb_lvl1_gaussian_nb",
        "series":     "Naive Bayes",
        "level":      1,
        "algorithm":  "Gaussian Naive Bayes (as Neural Network)",
        "dataset":    "Iris",
        "framework":  "PyTorch",
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
    batch_size: int = 32,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Dict[str, Any]:
    """
    Loads Iris dataset, standardises features.
    Returns dict with train_loader, val_loader, test_loader, plus raw arrays for sklearn.
    """
    data = load_iris()
    X    = data.data.astype(np.float32)
    y    = data.target

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
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return {
        'train_loader':  to_loader(X_train, y_train, True),
        'val_loader':    to_loader(X_val,   y_val,   False),
        'test_loader':   to_loader(X_test,  y_test,  False),
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        'scaler':  scaler,
        'n_features':    X_train.shape[1],
        'num_classes':   len(np.unique(y)),
        'feature_names': list(data.feature_names),
        'class_names':   list(data.target_names),
        # raw (unscaled) for sklearn
        'X_tv_raw': X_tv, 'y_tv': y_tv,
    }


# ─────────────────────────────────────────────
#  5. Model — Gaussian NB as NN with sklearn-style API
# ─────────────────────────────────────────────
class GaussianNBNN(nn.Module):
    """
    Gaussian Naive Bayes expressed as a differentiable module.

    Learnable parameters per class c, per feature f:
        mu      : class means           shape (C, F)
        log_var : log class variances   shape (C, F)  — ensures sigma^2 > 0
        log_pi  : log class priors      shape (C,)

    Provides sklearn-style .fit() / .predict() / .save() / .load() methods.
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.mu      = nn.Parameter(torch.randn(num_classes, num_features) * 0.1)
        self.log_var = nn.Parameter(torch.zeros(num_classes, num_features))
        self.log_pi  = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, F)
        log P(c|x) proportional to log pi_c
            + sum_f [ -0.5*(log(2*pi) + log_var_cf + (x_f - mu_cf)^2 / var_cf) ]
        Returns log-posteriors shape (B, C).
        """
        var   = torch.exp(self.log_var)                         # (C, F)
        diff  = x.unsqueeze(1) - self.mu.unsqueeze(0)          # (B, C, F)
        log_p = -0.5 * (np.log(2 * np.pi) + self.log_var + diff ** 2 / var)
        return self.log_pi + log_p.sum(dim=2)                   # (B, C)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 300,
        lr: float = 5e-3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train via gradient descent on NLL loss."""
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

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
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)

            val_m = evaluate(self, val_loader) if val_loader else {'loss': None, 'accuracy': None}
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
        return logits.argmax(dim=1).cpu().numpy()

    def save(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.load_state_dict(torch.load(filepath, map_location=device))


def build_model(num_features: int = 4, num_classes: int = 3) -> GaussianNBNN:
    return GaussianNBNN(num_features, num_classes)


# ─────────────────────────────────────────────
#  6. Train
# ─────────────────────────────────────────────
def train(
    model: GaussianNBNN,
    dataloaders: Dict[str, Any],
    epochs: int = 300,
    lr: float = 5e-3,
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
def evaluate(model: GaussianNBNN, loader: DataLoader) -> Dict[str, float]:
    """
    Returns dict with keys: loss, accuracy, mse, r2.
    MSE and R2 computed between predicted probabilities and one-hot labels.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
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
    accuracy = (logits.argmax(1) == labels).float().mean().item()

    probs   = torch.softmax(logits, dim=1)
    one_hot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1.0)
    mse     = ((probs - one_hot) ** 2).mean().item()
    ss_res  = ((probs - one_hot) ** 2).sum().item()
    ss_tot  = ((one_hot - one_hot.mean(0)) ** 2).sum().item()
    r2      = 1.0 - ss_res / (ss_tot + 1e-12)

    return {'loss': avg_loss, 'accuracy': accuracy, 'mse': mse, 'r2': r2}


# ─────────────────────────────────────────────
#  8. Predict
# ─────────────────────────────────────────────
def predict(model: GaussianNBNN, x) -> np.ndarray:
    return model.predict(x)


# ─────────────────────────────────────────────
#  9. Save artifacts
# ─────────────────────────────────────────────
def save_artifacts(
    model: GaussianNBNN,
    history: Dict[str, Any],
    metrics: Dict[str, Any],
    sklearn_metrics: Dict[str, float],
) -> None:
    model.save(os.path.join(OUTPUT_DIR, 'model.pt'))

    all_out = {
        'metadata':       get_task_metadata(),
        'pytorch_metrics': metrics,
        'sklearn_metrics': sklearn_metrics,
        'comparison': {
            'acc_diff': abs(metrics['val']['accuracy'] - sklearn_metrics['accuracy']),
        },
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
    plt.xlabel('Epoch'); plt.ylabel('NLL Loss')
    plt.title('Training Loss — Gaussian NB NN')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if history['val_acc'][0] is not None:
        plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training Accuracy — Gaussian NB NN')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nb_lvl1_training_curves.png'), dpi=150)
    plt.close()

    # ── Plot: PyTorch vs sklearn accuracy comparison ──
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
    plt.title('Accuracy Comparison: PyTorch GNB vs Sklearn GNB')
    plt.ylabel('Accuracy')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{bar.get_height():.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nb_lvl1_model_comparison.png'), dpi=150)
    plt.close()

    print(f"  Artifacts saved to {OUTPUT_DIR}/")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('Gaussian Naive Bayes (NN) vs Sklearn — Iris Dataset')
    print('=' * 60)

    set_seed(42)
    metadata = get_task_metadata()
    print(f"\nTask:    {metadata['task_id']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Device:  {device}")

    ACC_THRESHOLD  = 0.90
    SKLEARN_DELTA  = 0.03
    EPOCHS         = 300

    # ── [1/5] Data ──
    print('\n[1/5] Loading and preprocessing data...')
    dataloaders = make_dataloaders()
    print(f"  Train samples : {len(dataloaders['X_train'])}")
    print(f"  Val   samples : {len(dataloaders['X_val'])}")
    print(f"  Test  samples : {len(dataloaders['X_test'])}")
    print(f"  Features      : {dataloaders['n_features']}")
    print(f"  Classes       : {dataloaders['class_names']}")

    # ── [2/5] sklearn baseline ──
    print('\n[2/5] Training Sklearn GaussianNB...')
    sk_model = GaussianNB()
    sk_model.fit(dataloaders['X_train'], dataloaders['y_train'])
    sk_val_acc  = accuracy_score(dataloaders['y_val'],  sk_model.predict(dataloaders['X_val']))
    sk_test_acc = accuracy_score(dataloaders['y_test'], sk_model.predict(dataloaders['X_test']))
    sklearn_metrics = {'accuracy': float(sk_test_acc), 'val_accuracy': float(sk_val_acc)}
    print(f"  Sklearn val  accuracy: {sk_val_acc:.4f}")
    print(f"  Sklearn test accuracy: {sk_test_acc:.4f}")

    # ── [3/5] Build & train PyTorch model ──
    print('\n[3/5] Building and training Gaussian NB NN...')
    model   = build_model(num_features=dataloaders['n_features'],
                          num_classes=dataloaders['num_classes'])
    history = train(model, dataloaders, epochs=EPOCHS, lr=5e-3, verbose=True)

    # ── [4/5] Evaluate on train AND val AND test ──
    print('\n[4/5] Evaluating on train and validation splits...')
    print('\n  --- Train split ---')
    train_metrics = evaluate(model, dataloaders['train_loader'])
    for k, v in train_metrics.items():
        print(f"    {k:12s}: {v:.6f}")

    print('\n  --- Validation split ---')
    val_metrics = evaluate(model, dataloaders['val_loader'])
    for k, v in val_metrics.items():
        print(f"    {k:12s}: {v:.6f}")

    print('\n  --- Test split ---')
    test_metrics = evaluate(model, dataloaders['test_loader'])
    for k, v in test_metrics.items():
        print(f"    {k:12s}: {v:.6f}")

    # ── [5/5] Save artifacts ──
    print('\n[5/5] Saving artifacts...')
    all_metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
    save_artifacts(model, history, all_metrics, sklearn_metrics)

    # ── Comparison ──
    acc_diff = abs(val_metrics['accuracy'] - sk_val_acc)
    print(f"\n--- Comparison ---")
    print(f"  PyTorch val  accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  Sklearn val  accuracy : {sk_val_acc:.4f}")
    print(f"  Difference            : {acc_diff:.4f}  (threshold < {SKLEARN_DELTA})")

    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")
    print('=' * 60)

    try:
        assert val_metrics['accuracy']  > ACC_THRESHOLD, \
            f"Val accuracy {val_metrics['accuracy']:.4f} <= {ACC_THRESHOLD}"
        assert test_metrics['accuracy'] > ACC_THRESHOLD, \
            f"Test accuracy {test_metrics['accuracy']:.4f} <= {ACC_THRESHOLD}"
        assert acc_diff < SKLEARN_DELTA, \
            f"Gap from sklearn {acc_diff:.4f} > {SKLEARN_DELTA}"
        print('Task completed successfully!')
        print('=' * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f'\nFAIL -- {e}')
        sys.exit(1)
