"""Train AssignmentCNN on CIFAR-10 and save the weights to app/models/cnn_cifar10.pth.

Run from the project root:
    uv run python train_cnn.py
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from app.cnn_model import AssignmentCNN, transform

MODEL_PATH = "app/models/cnn_cifar10.pth"


def get_datasets(image_folder=None):
    if image_folder:
        # Optional: CIFAR-10 stored as image folders (train/<class>/*.png, test/<class>/*.png)
        train = torchvision.datasets.ImageFolder(os.path.join(image_folder, "train"), transform=transform)
        test = torchvision.datasets.ImageFolder(os.path.join(image_folder, "test"), transform=transform)
    else:
        train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return train, test


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--image-folder", default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    train_dataset, test_dataset = get_datasets(args.image_folder)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss {running_loss / total:.4f} "
            f"| train acc {correct / total:.4f} | test acc {test_acc:.4f}"
        )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
