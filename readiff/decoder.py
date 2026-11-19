"""ViMo RVQ decoder for continuous 512-dimensional latents."""
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv1(self.relu(x))
        return self.conv2(self.relu(residual)) + x


class ResidualStack(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.model = nn.Sequential(*(ResidualBlock(width, dilation) for dilation in (9, 3, 1)))

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        blocks = [nn.Conv1d(width, width, 3, padding=1), nn.ReLU()]
        for _ in range(2):
            blocks.append(nn.Sequential(
                ResidualStack(width), nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(width, width, 3, padding=1)
            ))
        blocks.extend([nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
                       nn.Conv1d(width, 263, 3, padding=1)])
        self.model = nn.Sequential(*blocks)

    def forward(self, latent):
        return self.model(latent.transpose(1, 2)).transpose(1, 2)
