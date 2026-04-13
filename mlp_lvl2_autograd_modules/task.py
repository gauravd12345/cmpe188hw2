"""
task_id : mlp_lvl2_autograd_modules
series  : Neural Networks (MLP)
level   : 2
algorithm: MLP (nn.Module + Autograd)

Description:
    General MLP classifier with Dropout and BatchNorm trained on MNIST.
    Uses torch.nn.Module + autograd (no manual gradient computation).

Protocol: pytorch_task_v1
Entrypoint: python tasks/mlp_lvl2_autograd_modules/task.py
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
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = 'tasks/mlp_lvl2_autograd_modules/artifacts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────
def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id":    "mlp_lvl2_autograd_modules",
        "series":     "Neural Networks (MLP)",
        "level":      2,
        "algorithm":  "MLP (nn.Module + Autograd)",
        "dataset":    "MNIST",
        "framework":  "PyTorch",
        "output_dir": OUTPUT_DIR,
    }

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(
    batch_size: int = 256,
    val_fraction: float = 0.1,
    data_root: str = '/tmp/mnist_mlp',
) -> Dict[str, Any]:
    """
    Returns dict with train_loader, val_loader, test_loader.
    Train is 90% of official MNIST train split; val is the remaining 10%.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(root=data_root, train=True,  download=True, transform=transform)
    test_ds    = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    n_val   = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return {
        'train_loader': train_loader,
        'val_loader':   val_loader,
        'test_loader':  test_loader,
        'n_train':      n_train,
        'n_val':        n_val,
        'n_test':       len(test_ds),
        'input_dim':    784,
        'num_classes':  10,
    }

class MLPClassifier(nn.Module):
    """
    Fully-connected MLP with BatchNorm and Dropout.
    Architecture: Linear → [BN] → ReLU → Dropout  (×L) → Linear

    Provides sklearn-style .fit() / .predict() / .save() / .load() methods.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple = (512, 256, 128),
        num_classes: int = 10,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 10,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the model; return history dict with loss and accuracy curves."""
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            scheduler.step()

            train_loss = total_loss / total
            train_acc  = correct / total
            history['train_losses'].append(train_loss)
            history['train_acc'].append(train_acc)

            if val_loader is not None:
                val_m = evaluate(self, val_loader)
                history['val_losses'].append(val_m['loss'])
                history['val_acc'].append(val_m['accuracy'])
            else:
                history['val_losses'].append(None)
                history['val_acc'].append(None)

            if verbose and (epoch % 2 == 0 or epoch == 1):
                val_str = (f"val_loss={history['val_losses'][-1]:.4f}  "
                           f"val_acc={history['val_acc'][-1]:.4f}"
                           if val_loader else "")
                print(f"  Epoch [{epoch}/{epochs}]  "
                      f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  {val_str}")

        return history

    def predict(self, x) -> np.ndarray:
        """Return class predictions as numpy array."""
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


def build_model(input_dim: int = 784, num_classes: int = 10) -> MLPClassifier:
    return MLPClassifier(input_dim=input_dim, num_classes=num_classes)

def train(
    model: MLPClassifier,
    dataloaders: Dict[str, Any],
    epochs: int = 10,
    lr: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, Any]:
    return model.fit(
        dataloaders['train_loader'],
        val_loader=dataloaders['val_loader'],
        epochs=epochs,
        lr=lr,
        verbose=verbose,
    )

def evaluate(model: MLPClassifier, loader: DataLoader) -> Dict[str, float]:
    """
    Compute loss, accuracy, MSE (probs vs one-hot), and R² on the given loader.
    Returns dict: {loss, accuracy, mse, r2}.
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

def predict(model: MLPClassifier, x) -> np.ndarray:
    return model.predict(x)

def save_artifacts(
    model: MLPClassifier,
    history: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    model.save(os.path.join(OUTPUT_DIR, 'model.pt'))

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Plot: training loss & accuracy curves ──
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    if history['val_losses'][0] is not None:
        plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Cross-Entropy Loss')
    plt.title('Training Loss'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if history['val_acc'][0] is not None:
        plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_lvl2_training_curves.png'), dpi=150)
    plt.close()

    # ── Plot: metrics bar summary ──
    val_m = metrics['val']
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = ['Accuracy', 'R²', '1 - MSE']
    values = [val_m['accuracy'], max(val_m['r2'], 0), 1 - val_m['mse']]
    bars   = ax.bar(names, values, color=['steelblue', 'seagreen', 'salmon'])
    ax.set_ylim(0, 1.05)
    ax.set_title('Validation Metrics Summary')
    ax.set_ylabel('Score')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_lvl2_val_metrics.png'), dpi=150)
    plt.close()

    print(f"  Artifacts saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    print('=' * 60)
    print('MLP Classifier on MNIST')
    print('=' * 60)

    set_seed(42)
    metadata = get_task_metadata()
    print(f"\nTask:    {metadata['task_id']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Device:  {device}")

    ACC_THRESHOLD = 0.97
    EPOCHS        = 10

    # ── [1/4] Data ──
    print('\n[1/4] Loading and preprocessing data...')
    dataloaders = make_dataloaders(batch_size=256)
    print(f"  Train samples : {dataloaders['n_train']}")
    print(f"  Val   samples : {dataloaders['n_val']}")
    print(f"  Test  samples : {dataloaders['n_test']}")

    # ── [2/4] Build & train ──
    print('\n[2/4] Building and training MLP...')
    model = build_model()
    print(f"  Architecture: {model}")
    history = train(model, dataloaders, epochs=EPOCHS, lr=1e-3, verbose=True)

    # ── [3/4] Evaluate on train AND val AND test ──
    print('\n[3/4] Evaluating on train and validation splits...')
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

    # ── [4/4] Save artifacts ──
    print('\n[4/4] Saving artifacts...')
    all_metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
    save_artifacts(model, history, all_metrics)

    # ── Quality assertions ──
    print('\n--- Quality Assertions ---')
    print(f"  val  accuracy = {val_metrics['accuracy']:.4f}  (threshold > {ACC_THRESHOLD})")
    print(f"  test accuracy = {test_metrics['accuracy']:.4f}  (threshold > {ACC_THRESHOLD})")
    print(f"  val  MSE      = {val_metrics['mse']:.6f}")
    print(f"  val  R²       = {val_metrics['r2']:.6f}")
    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")
    print('=' * 60)

    try:
        assert val_metrics['accuracy']  > ACC_THRESHOLD, \
            f"Val accuracy {val_metrics['accuracy']:.4f} <= {ACC_THRESHOLD}"
        assert test_metrics['accuracy'] > ACC_THRESHOLD, \
            f"Test accuracy {test_metrics['accuracy']:.4f} <= {ACC_THRESHOLD}"
        assert val_metrics['mse'] < 0.05, \
            f"Val MSE {val_metrics['mse']:.4f} too high"
        print('Task completed successfully!')
        print('=' * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f'\nFAIL -- {e}')
        sys.exit(1)
