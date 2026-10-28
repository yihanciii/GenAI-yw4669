"""GAN architecture for Assignment 3 (Part 1).

Generator:      noise (B, 100) -> FC -> 128 x 7 x 7 -> ConvT -> 64 x 14 x 14 -> ConvT -> 1 x 28 x 28
Discriminator:  image (B, 1, 28, 28) -> Conv -> 64 x 14 x 14 -> Conv -> 128 x 7 x 7 -> Linear -> probability
"""
import torch.nn as nn

NOISE_DIM = 100


class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super(Generator, self).__init__()
        # Fully connected layer: 100 -> 7*7*128, then reshape to 128 x 7 x 7
        self.fc = nn.Linear(noise_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            # 128 x 7 x 7 -> 64 x 14 x 14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 14 x 14 -> 1 x 28 x 28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # pixel values in [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)  # reshape
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # 1 x 28 x 28 -> 64 x 14 x 14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 14 x 14 -> 128 x 7 x 7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Flatten -> Linear -> single output, Sigmoid turns it into a probability
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(self.net(x))
