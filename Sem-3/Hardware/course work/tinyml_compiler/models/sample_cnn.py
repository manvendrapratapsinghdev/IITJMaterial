"""
Sample CNN Models for TinyML Compiler Testing
===============================================
Provides models suitable for MNIST and CIFAR-10 classification.
"""

import torch
import torch.nn as nn


class SimpleCNN_MNIST(nn.Module):
    """
    Simple CNN for MNIST digit classification (28x28x1 input).
    Designed to fit within 256KB MCU RAM when quantized.
    
    Architecture:
      Conv2d(1, 16, 3, padding=1) → ReLU → MaxPool(2)
      Conv2d(16, 32, 3, padding=1) → ReLU → MaxPool(2)
      Flatten → Linear(32*7*7, 128) → ReLU → Linear(128, 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def get_sample_input():
        return torch.randn(1, 1, 28, 28)

    @staticmethod
    def get_input_shape():
        return (1, 1, 28, 28)


class TinyCNN_CIFAR10(nn.Module):
    """
    Tiny CNN for CIFAR-10 classification (32x32x3 input).
    Minimal architecture to fit MCU constraints.

    Architecture:
      Conv2d(3, 16, 3, padding=1) → ReLU → MaxPool(2)
      Conv2d(16, 32, 3, padding=1) → ReLU → MaxPool(2)
      Conv2d(32, 32, 3, padding=1) → ReLU → MaxPool(2)
      Flatten → Linear(32*4*4, 64) → ReLU → Linear(64, 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def get_sample_input():
        return torch.randn(1, 3, 32, 32)

    @staticmethod
    def get_input_shape():
        return (1, 3, 32, 32)


# Registry of available models
MODELS = {
    'mnist_cnn': SimpleCNN_MNIST,
    'cifar10_cnn': TinyCNN_CIFAR10,
}


if __name__ == "__main__":
    for name, ModelClass in MODELS.items():
        model = ModelClass()
        sample = model.get_sample_input()
        output = model(sample)
        total_params = sum(p.numel() for p in model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"{name}:")
        print(f"  Input shape:  {tuple(sample.shape)}")
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  Parameters:   {total_params:,}")
        print(f"  Size (FP32):  {total_size/1024:.1f} KB")
        print()
