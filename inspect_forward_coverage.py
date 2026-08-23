import torch

from datasets.openlane import OpenLaneDataset


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

RANGES = [55.0, 75.0, 100.0]
NUM_SAMPLES = 5000


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

total_visible = 0
inside = {
    limit: 0
    for limit in RANGES
}

max_forward = 0.0


for index in range(min(NUM_SAMPLES, len(dataset))):
    sample = dataset[index]

    for lane in sample["lane_lines"]:
        xyz = torch.tensor(
            lane["xyz"],
            dtype=torch.float32,
        ).T

        visibility = torch.tensor(
            lane["visibility"],
            dtype=torch.float32,
        )

        valid = visibility > 0.5
        xyz = xyz[valid]

        if len(xyz) == 0:
            continue

        forward = xyz[:, 0]

        forward = forward[forward >= 0]

        if len(forward) == 0:
            continue

        total_visible += len(forward)

        max_forward = max(
            max_forward,
            forward.max().item(),
        )

        for limit in RANGES:
            inside[limit] += (
                forward <= limit
            ).sum().item()


print("Samples checked:", min(NUM_SAMPLES, len(dataset)))
print("Visible forward points:", total_visible)
print("Maximum forward distance:", max_forward)

print()

for limit in RANGES:
    count = inside[limit]

    percentage = (
        100.0 * count / total_visible
        if total_visible > 0
        else 0.0
    )

    print(
        f"0-{int(limit)} m:",
        count,
        f"({percentage:.2f}%)",
    )
