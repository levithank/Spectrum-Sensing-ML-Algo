import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # 128 I samples + 128 Q samples = 256 inputs
        self.flatten = nn.Flatten()

        # 256 -> 256
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)

        # 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        # 128 -> 64
        self.fc3 = nn.Linear(128, 64)

        # 64 -> number of modulation classes
        self.out = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        # x shape: [batch, 2, 128]
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.out(x)

        return x