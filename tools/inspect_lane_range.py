import torch

from datasets.openlane import OpenLaneDataset


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

x_min = float("inf")
x_max = float("-inf")

y_min = float("inf")
y_max = float("-inf")

z_min = float("inf")
z_max = float("-inf")

point_count = 0

for index in range(min(len(dataset), 1000)):
    sample = dataset[index]

    for lane in sample["lane_lines"]:
        xyz = torch.tensor(
            lane["xyz"],
            dtype=torch.float32,
        ).T

        if xyz.numel() == 0:
            continue

        x_min = min(x_min, xyz[:, 0].min().item())
        x_max = max(x_max, xyz[:, 0].max().item())

        y_min = min(y_min, xyz[:, 1].min().item())
        y_max = max(y_max, xyz[:, 1].max().item())

        z_min = min(z_min, xyz[:, 2].min().item())
        z_max = max(z_max, xyz[:, 2].max().item())

        point_count += xyz.shape[0]

print("Samples:", min(len(dataset), 1000))
print("Points:", point_count)

print("\nOpenLane X (forward):")
print(x_min, "->", x_max)

print("\nOpenLane Y (left):")
print(y_min, "->", y_max)

print("\nOpenLane Z (up):")
print(z_min, "->", z_max)
