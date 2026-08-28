import torch.nn as nn


class ProposalHead(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)