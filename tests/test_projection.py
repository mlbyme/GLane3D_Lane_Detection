import torch

from datasets.openlane import OpenLaneDataset
from datasets.transforms import ResizeWithIntrinsic


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

resize = ResizeWithIntrinsic((360, 640))

sample = dataset[0]

_, K = resize(
    sample["image"],
    sample["intrinsic"],
)

lane = sample["lane_lines"][0]

xyz = torch.tensor(
    lane["xyz"],
    dtype=torch.float32,
)

uv_gt = torch.tensor(
    lane["uv"],
    dtype=torch.float32,
)

# xyz shape is [3, N]
xyz = xyz.T

# uv shape is [2, N]
uv_gt = uv_gt.T

print("=== GROUND TRUTH PROJECTION CHECK ===")

print("XYZ shape:", xyz.shape)
print("UV shape:", uv_gt.shape)

print("\nFirst original XYZ:")
print(xyz[0])

print("\nFirst original UV:")
print(uv_gt[0])

# Resize the ground-truth image coordinates.
sx = 640.0 / 1920.0
sy = 360.0 / 1280.0

uv_resized = uv_gt.clone()
uv_resized[:, 0] *= sx
uv_resized[:, 1] *= sy

print("\nFirst resized UV:")
print(uv_resized[0])

print("\nXYZ ranges:")
print("X:", xyz[:, 0].min().item(), "->", xyz[:, 0].max().item())
print("Y:", xyz[:, 1].min().item(), "->", xyz[:, 1].max().item())
print("Z:", xyz[:, 2].min().item(), "->", xyz[:, 2].max().item())

print("\nResized UV ranges:")
print(
    "U:",
    uv_resized[:, 0].min().item(),
    "->",
    uv_resized[:, 0].max().item(),
)

print(
    "V:",
    uv_resized[:, 1].min().item(),
    "->",
    uv_resized[:, 1].max().item(),
)

# Test the standard camera-coordinate conversion:
#
# Waymo:
#   X forward
#   Y left
#   Z up
#
# Standard camera:
#   X right
#   Y down
#   Z forward
#
# Therefore:
#
# camera_x = -Waymo_Y
# camera_y = -Waymo_Z
# camera_z = Waymo_X

camera_xyz = torch.stack(
    [
        -xyz[:, 1],
        -xyz[:, 2],
        xyz[:, 0],
    ],
    dim=1,
)

depth = camera_xyz[:, 2]

uvw = (K @ camera_xyz.T).T

uv_projected = uvw[:, :2] / depth.unsqueeze(1)

print("\n=== STANDARD COORDINATE TEST ===")

print("Camera XYZ first point:")
print(camera_xyz[0])

print("\nProjected UV first point:")
print(uv_projected[0])

print("\nGround-truth resized UV first point:")
print(uv_resized[0])

print("\nProjected UV ranges:")
print(
    "U:",
    uv_projected[:, 0].min().item(),
    "->",
    uv_projected[:, 0].max().item(),
)

print(
    "V:",
    uv_projected[:, 1].min().item(),
    "->",
    uv_projected[:, 1].max().item(),
)

print("\nMean projection error:")
error = torch.linalg.norm(
    uv_projected - uv_resized,
    dim=1,
)

print(error.mean().item())

print("\nMax projection error:")
print(error.max().item())
