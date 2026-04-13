"""
Task 1: Multi-Layer Perceptron (MLP) as a Neural Network
Dataset: MNIST (handwritten digits)
Protocol: pytorch_task_v1
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

class MLP(nn.Module):
    """Fully-connected MLP: Input → 256 → 128 → 64 → 10 (classes)."""

    def __init__(self, input_dim: int = 784, hidden_dims=(256, 128, 64), num_classes: int = 10):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    # Hyper-parameters
    EPOCHS        = 5
    BATCH_SIZE    = 256
    LR            = 1e-3
    TARGET_ACC    = 0.95          # 95 % on test set → pass
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Task 1 – MLP]  device={DEVICE}")

    # Data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root="/tmp/mnist", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model, loss, optimiser
    model     = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        acc  = evaluate(model, test_loader, DEVICE)
        scheduler.step()
        print(f"  Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  test_acc={acc:.4f}")

    final_acc = evaluate(model, test_loader, DEVICE)
    print(f"\n[Task 1 – MLP]  Final test accuracy: {final_acc:.4f}  (target ≥ {TARGET_ACC})")

    if final_acc >= TARGET_ACC:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
