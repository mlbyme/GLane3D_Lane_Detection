import torch

from datasets.openlane import OpenLaneDataset


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

dataset = OpenLaneDataset(DATA_ROOT, split="training")
sample = dataset[0]

print("Intrinsic:")
print(torch.tensor(sample["intrinsic"]))

print("\nExtrinsic:")
print(torch.tensor(sample["extrinsic"]))

print("\nVisible lane Z ranges:")

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

    print(
        f"Lane {i}:",
        xyz[:, 2].min().item(),
        "->",
        xyz[:, 2].max().item(),
    )
