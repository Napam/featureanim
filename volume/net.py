import torch
from torch import nn 
import warnings

from torch._C import BenchmarkExecutionStats
from torch.nn.modules.conv import Conv2d
warnings.filterwarnings("ignore", category=UserWarning)


class BenjaBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """Out channels should integer divisible by 4"""
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding='same', bias=False)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding='same', bias=False)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding='same', bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, padding='same', bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, X: torch.Tensor):
        H = torch.cat([
            self.conv3x3(X),
            self.conv5x5(X),
            self.conv7x7(X),
            self.conv1x1(self.pool(X)),
        ], axis=1)

        H = self.relu(H)
        H = self.bn(H)
        return H


class Benja(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(
            BenjaBlock( 1, 64),
            BenjaBlock(64, 64),
            nn.MaxPool2d(kernel_size=2),
            BenjaBlock(64, 64),
            BenjaBlock(64, 64),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.fc1 = nn.Linear(7*7*64, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 10)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, X: torch.Tensor):
        """Returns predicitions, and latest latent representation"""
        H = self.convs(X)
        H = self.fc1(H.view(len(X), -1))
        H = self.bn1(self.relu(H))
        H = self.fc2(H)
        H = self.bn2(self.relu(H))
        H = self.tanh(self.fc3(H))
        return self.sigmoid(self.fc4(H)), H


# class Benja(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.convs = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding='same'),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, padding='same'),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(14*14*64, 128)
#         )

#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 2)
#         self.fc4 = nn.Linear(2, 10)

#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, X: torch.Tensor):
#         """Returns predicitions, and latest latent representation"""
#         H = self.convs(X)
#         H = self.relu(self.fc2(H))
#         H = self.fc3(H)
#         return self.sigmoid(self.fc4(H)), H.detach()


def oneVsAll(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    bce = nn.BCELoss()
    classloss = 0
    for i in range(10):
        classloss += bce(y_pred[:,i], ((y_true == i)*1).to(torch.float32))
    return classloss

    
if __name__ == '__main__':
    X = torch.randint(0, 10, size=[4,1,28,28], dtype=torch.float32)
    model = Benja()
