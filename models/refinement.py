import torch
import torch.nn as nn


class ProposalRefiner(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        connection_dim=64,
        num_classes=15,
        num_layers=2,
        num_heads=8,
    ):
        super().__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            batch_first=True,
        )

        self.transformer = nn.TransformerDecoder(
            layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(
        feature_dim,
        15,
        )

        self.x_offset = nn.Linear(feature_dim, 1)
        self.z = nn.Linear(feature_dim, 1)
        
        self.classifier = nn.Linear(
            feature_dim,
            num_classes,
        )

        self.connection = nn.Linear(
            feature_dim,
            connection_dim,
        )

    def forward(
        self,
        proposal_features,
        bev_features,
    ):
        memory = bev_features.flatten(2).transpose(1, 2)

        features = self.transformer(
            proposal_features,
            memory,
        )

        return {
    "x_offset": self.x_offset(features).squeeze(-1),
    "z": self.z(features).squeeze(-1),
    "class_logits": self.classifier(features),
    "connection": self.connection(features),
}