import torch

from datasets.openlane import OpenLaneDataset
from models.ipm import project_openlane_to_image


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

sample = dataset[0]

image = sample["image"]

K = torch.tensor(
    sample["intrinsic"],
    dtype=torch.float32,
)

lane = sample["lane_lines"][0]

xyz = torch.tensor(
    lane["xyz"],
    dtype=torch.float32,
).T

uv_gt = torch.tensor(
    lane["uv"],
    dtype=torch.float32,
).T

uv, depth = project_openlane_to_image(
    xyz,
    K,
)

error = torch.linalg.norm(
    uv - uv_gt,
    dim=1,
)

print(" NATIVE RESOLUTION PROJECTION TEST ")

print("Image:", image.size)
print("Points:", xyz.shape[0])

print("\nProjected first point:")
print(uv[0])

print("\nGround truth first point:")
print(uv_gt[0])

print("\nMean error:")
print(error.mean().item())

print("\nMax error:")
print(error.max().item())

print("\nDepth range:")
print(
    depth.min().item(),
    "->",
    depth.max().item(),
)