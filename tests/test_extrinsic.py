import torch

from datasets.openlane import OpenLaneDataset


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


dataset = OpenLaneDataset(DATA_ROOT, split="training")
sample = dataset[0]

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
)

print("=== CAMERA TO VEHICLE TEST ===")

for i, lane in enumerate(sample["lane_lines"]):
    xyz = torch.tensor(
        lane["xyz"],
        dtype=torch.float32,
    ).T

    visibility = torch.tensor(
        lane["visibility"],
        dtype=torch.float32,
    )

    xyz = xyz[visibility > 0.5]

    if len(xyz) == 0:
        continue

    ones = torch.ones(
        len(xyz),
        1,
        dtype=torch.float32,
    )

    xyz_h = torch.cat(
        [xyz, ones],
        dim=1,
    )

    vehicle = xyz_h @ extrinsic.T

    print(f"\nLane {i}")
    print(
        "Camera Z:",
        xyz[:, 2].min().item(),
        "->",
        xyz[:, 2].max().item(),
    )
    print(
        "Vehicle Z:",
        vehicle[:, 2].min().item(),
        "->",
        vehicle[:, 2].max().item(),
    )
