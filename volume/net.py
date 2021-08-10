import torch
from torch import nn 
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class BenjaBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """Out channels should integer divisible by 4"""
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding='same')
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding='same')
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, padding='same')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, X):
        h = torch.cat([
            self.conv3x3(X),
            self.conv5x5(X),
            self.conv7x7(X),
            self.conv1x1(self.pool(X)),
        ], axis=1)

        h = self.relu(h)
        h = self.bn(h)
        return h


class Benja(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    
if __name__ == '__main__':
    block = BenjaBlock(1, 8)
    X = torch.randint(0,10, size=[1,1,6,6], dtype=torch.float32)
    block(X)

