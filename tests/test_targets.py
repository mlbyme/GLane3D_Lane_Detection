import torch

from datasets.openlane import OpenLaneDataset
from datasets.targets import build_bev_targets
from models.bev import make_bev_anchor_grid


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

sample = dataset[0]

bev = make_bev_anchor_grid(
    height=56,
    width=32,
    forward_range=100.0,
    bev_width=34.0,
)

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
)

targets = build_bev_targets(
    sample["lane_lines"],
    bev,
    extrinsic,
)

positive = targets["score"] > 0

print("Score:", targets["score"].shape)
print("X offset:", targets["x_offset"].shape)
print("Z:", targets["z"].shape)

print("Positive anchors:", positive.sum().item())

if positive.any():
    print(
        "Offset range:",
        targets["x_offset"][positive].min().item(),
        "->",
        targets["x_offset"][positive].max().item(),
    )

    print(
        "Z range:",
        targets["z"][positive].min().item(),
        "->",
        targets["z"][positive].max().item(),
    )
