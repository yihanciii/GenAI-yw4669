"""Train the GAN on MNIST and save the generator to app/models/gan_generator.pth.

Run from the project root:
    uv run python train_gan.py
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from app.gan_model import NOISE_DIM, Discriminator, Generator

MODEL_PATH = "app/models/gan_generator.pth"
SAMPLE_DIR = "gan_samples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Scale pixels to [-1, 1] to match the generator's Tanh output
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    criterion = nn.BCELoss()
    # Adam with beta1 = 0.5 is the standard choice for GANs (DCGAN paper)
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, NOISE_DIM, device=device)  # to track progress
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    for epoch in range(args.epochs):
        G.train()
        D.train()
        for real, _ in loader:
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            # 1) Train Discriminator: real -> 1, fake -> 0
            z = torch.randn(b, NOISE_DIM, device=device)
            fake = G(z)
            loss_D = criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # 2) Train Generator: try to make D output 1 for fake images
            loss_G = criterion(D(fake), ones)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch + 1}/{args.epochs} | loss_D {loss_D.item():.4f} | loss_G {loss_G.item():.4f}")

        # Save a grid of samples after each epoch
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise)
        save_image(samples, f"{SAMPLE_DIR}/epoch_{epoch + 1:02d}.png", nrow=8, normalize=True, value_range=(-1, 1))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(G.state_dict(), MODEL_PATH)
    print(f"Saved generator to {MODEL_PATH}")


if __name__ == "__main__":
    main()
