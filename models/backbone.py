import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 feature extractor.

    Input:
        [B, 3, H, W]

    Output:
        [B, 512, H/32, W/32]
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(weights=weights)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

    def forward(self, x):
        return self.features(x)
