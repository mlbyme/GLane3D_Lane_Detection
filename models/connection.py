import torch
import torch.nn as nn


class ConnectionHead(nn.Module):
    def __init__(
        self,
        connection_dim=64,
        position_dim=32,
        hidden_dim=128,
        edge_dim=64,
    ):
        super().__init__()

        self.position_encoder = nn.Sequential(
            nn.Linear(2, position_dim),
            nn.ReLU(inplace=True),
            nn.Linear(position_dim, position_dim),
        )

        input_dim = connection_dim + position_dim

        self.origin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, edge_dim),
        )

        self.destination = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, edge_dim),
        )

        self.edge = nn.Linear(edge_dim, 1)

    def forward(self, points, connection_features):
        position = torch.stack(
            [
                points[..., 1],
                points[..., 0],
            ],
            dim=-1,
        )

        position_features = self.position_encoder(
            position
        )

        features = torch.cat(
            [connection_features, position_features],
            dim=-1,
        )

        origin = self.origin(features)
        destination = self.destination(features)

        pair_features = (
            origin.unsqueeze(2)
            * destination.unsqueeze(1)
        )

        adjacency_logits = self.edge(
            pair_features
        ).squeeze(-1)

        return adjacency_logits
