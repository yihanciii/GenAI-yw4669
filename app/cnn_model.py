"""CNN architecture for Assignment 2 (Part 1).

Input: RGB image 64x64x3
Conv(16, 3x3, s1, p1) -> ReLU -> MaxPool(2x2, s2)   : 64x64x3  -> 32x32x16
Conv(32, 3x3, s1, p1) -> ReLU -> MaxPool(2x2, s2)   : 32x32x16 -> 16x16x32
Flatten                                             : 16*16*32 = 8192
FC(100) -> ReLU -> FC(10)
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

IMAGE_SIZE = 64

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# CIFAR-10 images are 32x32, so we resize them to 64x64 to match the required input size
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


class AssignmentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 16 x 32 x 32
        x = self.pool(F.relu(self.conv2(x)))  # -> 32 x 16 x 16
        x = x.view(-1, 32 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
