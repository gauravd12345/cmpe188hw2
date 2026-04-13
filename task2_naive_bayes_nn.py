"""
Task 2: Naive Bayes implemented as a Neural Network
Dataset: 20 Newsgroups (text classification, bag-of-words features)
Protocol: pytorch_task_v1

Naive Bayes as a NN:
  • Log class-prior  → learnable bias on the output layer
  • Log likelihoods  → learnable weight matrix (one row per class, one col per feature)
  • Forward pass     → W @ x + b  (exactly the Naive Bayes log-posterior, unnormalised)
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

class NaiveBayesNN(nn.Module):
    """
    Differentiable Naive Bayes as a single linear layer.
    Weights represent log-likelihoods; we learn them via gradient descent.
    """

    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        # Raw (unconstrained) parameters – made non-negative via softplus
        self.raw_weights = nn.Parameter(torch.randn(num_classes, vocab_size) * 0.01)
        self.bias        = nn.Parameter(torch.zeros(num_classes))          # log prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure weights stay non-negative (log-probability interpretation)
        W = torch.nn.functional.softplus(self.raw_weights)
        return x @ W.T + self.bias                                          # (B, C)

def numpy_to_tensor(arr) -> torch.Tensor:
    """Convert dense or sparse numpy/scipy array to float32 tensor."""
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return torch.tensor(arr, dtype=torch.float32)

def main():
    EPOCHS      = 20
    BATCH_SIZE  = 128
    LR          = 5e-3
    TARGET_ACC  = 0.70          # 70 % on test set → pass (20-class problem)
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Task 2 – Naive Bayes NN]  device={DEVICE}")

    # Data
    print("  Loading 20 Newsgroups …")
    news = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    X_text, y = news.data, news.target

    vectorizer = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = numpy_to_tensor(X_train)
    X_test_t  = numpy_to_tensor(X_test)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader  = DataLoader(
        TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH_SIZE
    )

    # Model
    vocab_size  = X_train_t.shape[1]
    num_classes = len(np.unique(y))
    model       = NaiveBayesNN(vocab_size, num_classes).to(DEVICE)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Training
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

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x_b, y_b in test_loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    correct += (model(x_b).argmax(1) == y_b).sum().item()
                    total   += y_b.size(0)
            print(f"  Epoch {epoch:>2}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={correct/total:.4f}")

    # Final evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            correct += (model(x_b).argmax(1) == y_b).sum().item()
            total   += y_b.size(0)
    final_acc = correct / total

    print(f"\n[Task 2 – Naive Bayes NN]  Final test accuracy: {final_acc:.4f}  (target ≥ {TARGET_ACC})")

    if final_acc >= TARGET_ACC:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
