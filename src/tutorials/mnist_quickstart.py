"""PyTorch Quickstart, run locally so Week 4 can record real numbers.

This follows the official Quickstart tutorial
(https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
Note the tutorial uses **FashionMNIST**, not the original digit MNIST -- same
shape (28x28 grayscale, 10 classes), harder task. ``--dataset mnist`` switches
to the digits if you want both.

It is kept in the repo so the figures in docs/week4 can be reproduced rather
than quoted from a screenshot.

Run from the src/ directory:
    python tutorials/mnist_quickstart.py
    python tutorials/mnist_quickstart.py --epochs 10 --lr 0.1   # one change

The six pieces to recognise here, because train.py has all six too:
  dataset   FashionMNIST            (train.py: the recorded mouse logs)
  input     a 28x28 image           (train.py: a window of deltas)
  target    a class index 0-9       (train.py: the next displacement)
  model     NeuralNetwork below     (train.py: DeltaGRU)
  loss      CrossEntropyLoss        (train.py: MSELoss)
  optimizer SGD                     (train.py: Adam)
"""
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "data", "tutorials")


class NeuralNetwork(nn.Module):
    """The tutorial's model: flatten, then three linear layers."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))


def train_epoch(loader, model, loss_fn, optimizer, device):
    model.train()
    last = 0.0
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        last = loss.item()
        if batch % 200 == 0:
            print(f"    loss {last:>7f}  [{batch * len(X):>5d}/"
                  f"{len(loader.dataset):>5d}]")
    return last


def test(loader, model, loss_fn, device):
    model.eval()
    n, batches = len(loader.dataset), len(loader)
    loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / n, loss / batches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("fashion", "mnist"),
                   default="fashion",
                   help="fashion = the tutorial's FashionMNIST (default)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    source = datasets.FashionMNIST if args.dataset == "fashion" else datasets.MNIST
    train_data = source(root=DATA, train=True, download=True,
                        transform=ToTensor())
    test_data = source(root=DATA, train=False, download=True,
                       transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch)

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    print(f"dataset: {args.dataset}  train {len(train_data)}  "
          f"test {len(test_data)}")
    print(f"device: {device}   epochs {args.epochs}  lr {args.lr}  "
          f"batch {args.batch}")
    print(f"params: {sum(q.numel() for q in model.parameters())}")
    for X, y in test_loader:
        print(f"input shape {tuple(X.shape)}  target shape {tuple(y.shape)} "
              f"dtype {y.dtype}")
        break

    final_train_loss, acc, tloss = 0.0, 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}\n{'-' * 30}")
        final_train_loss = train_epoch(train_loader, model, loss_fn,
                                       optimizer, device)
        acc, tloss = test(test_loader, model, loss_fn, device)
        print(f"  test accuracy {100 * acc:>0.1f}%  avg test loss {tloss:>8f}")

    print("\n=== record these in docs/week4 ===")
    print(f"dataset:            {args.dataset}")
    print(f"epochs:             {args.epochs}")
    print(f"learning rate:      {args.lr}")
    print(f"batch size:         {args.batch}")
    print(f"final train loss:   {final_train_loss:.6f}")
    print(f"test accuracy:      {100 * acc:.1f}%")
    print(f"avg test loss:      {tloss:.6f}")


if __name__ == "__main__":
    main()
