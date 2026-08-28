from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.openlane import OpenLaneDataset
from models.bev import make_bev_anchor_grid
from models.ipm import (
    project_openlane_to_image,
    vehicle_to_camera,
)


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

OUTPUT_DIR = Path("outputs/bev")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


dataset = OpenLaneDataset(DATA_ROOT, split="training")
sample = dataset[0]

image = sample["image"]

K = torch.tensor(
    sample["intrinsic"],
    dtype=torch.float32,
)

bev = make_bev_anchor_grid(
    height=56,
    width=32,
    forward_range=100.0,
    bev_width=34.0,
)

vehicle_xyz = bev.reshape(-1, 3)

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
)

camera_xyz = vehicle_to_camera(
    vehicle_xyz,
    extrinsic,
)

uv, depth = project_openlane_to_image(
    camera_xyz,
    K,
)

forward = bev[..., 0].reshape(-1)

valid = (
    (depth > 0)
    & (uv[:, 0] >= 0)
    & (uv[:, 0] < image.width)
    & (uv[:, 1] >= 0)
    & (uv[:, 1] < image.height)
)

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(image)

for lane in sample["lane_lines"]:
    uv_gt = torch.tensor(
        lane["uv"],
        dtype=torch.float32,
    ).T

    visibility = torch.tensor(
        lane["visibility"],
        dtype=torch.float32,
    )

    lane_valid = visibility > 0.5
    uv_gt = uv_gt[lane_valid]

    if len(uv_gt) == 0:
        continue

    ax.plot(
        uv_gt[:, 0],
        uv_gt[:, 1],
        linewidth=2,
    )

ax.scatter(
    uv[valid, 0],
    uv[valid, 1],
    c=forward[valid].numpy(),
    s=8,
)

ax.set_title(
    "GLane3D BEV anchors projected onto OpenLane image"
)
ax.set_xlim(0, image.width)
ax.set_ylim(image.height, 0)
ax.set_xlabel("u")
ax.set_ylabel("v")

output_path = (
    OUTPUT_DIR
    / "05_vehicle_ground_projection_100m_34mwidth.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)
plt.close()

print("Saved:", output_path)
print("Image path:", sample["image_path"])
print("Total anchors:", bev.numel() // 3)
print("Visible anchors:", valid.sum().item())

print(
    "Forward range:",
    bev[..., 0].min().item(),
    "->",
    bev[..., 0].max().item(),
)

print(
    "Lateral range:",
    bev[..., 1].min().item(),
    "->",
    bev[..., 1].max().item(),
)