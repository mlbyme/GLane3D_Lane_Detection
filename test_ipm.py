import torch

from datasets.openlane import OpenLaneDataset
from datasets.transforms import ResizeWithIntrinsic

from models.ipm import (
    make_bev_grid,
    project_openlane_to_image,
)


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

resize = ResizeWithIntrinsic((360, 640))

sample = dataset[0]

image, K = resize(
    sample["image"],
    sample["intrinsic"],
)

K = K.float()

lane = sample["lane_lines"][0]

xyz = torch.tensor(
    lane["xyz"],
    dtype=torch.float32,
).T

uv_gt = torch.tensor(
    lane["uv"],
    dtype=torch.float32,
).T

# Resize ground-truth UV.
uv_gt[:, 0] *= 640.0 / 1920.0
uv_gt[:, 1] *= 360.0 / 1280.0

uv, depth = project_openlane_to_image(
    xyz,
    K,
)

error = torch.linalg.norm(
    uv - uv_gt,
    dim=1,
)

print("=== CORRECTED IPM TEST ===")

print("Image:", image.size)

print("\nBEV/3D point count:", xyz.shape[0])

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

# Ground-plane BEV grid.
bev = make_bev_grid(
    x_range=(0.0, 30.0),
    y_range=(-10.0, 10.0),
    z=0.0,
    x_step=0.5,
    y_step=0.5,
)

uv_bev, depth_bev = project_openlane_to_image(
    bev,
    K,
)

valid = (
    (depth_bev > 0)
    & (uv_bev[:, 0] >= 0)
    & (uv_bev[:, 0] < image.width)
    & (uv_bev[:, 1] >= 0)
    & (uv_bev[:, 1] < image.height)
)

print("\n=== BEV GRID ===")

print("BEV points:", bev.shape[0])

print(
    "Valid:",
    valid.sum().item(),
)

print(
    "Valid percentage:",
    100.0 * valid.float().mean().item(),
)
