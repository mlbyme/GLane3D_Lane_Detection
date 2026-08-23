import torch

from datasets.openlane import OpenLaneDataset


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

LIMITS = [55.0, 75.0, 100.0]
NUM_SAMPLES = 5000


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

lane_count = 0

beyond = {
    limit: 0
    for limit in LIMITS
}

lane_max_values = []


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

        xyz = xyz[visibility > 0.5]

        if len(xyz) == 0:
            continue

        forward = xyz[:, 0]
        forward = forward[forward >= 0]

        if len(forward) == 0:
            continue

        lane_max = forward.max().item()

        lane_max_values.append(lane_max)
        lane_count += 1

        for limit in LIMITS:
            if lane_max > limit:
                beyond[limit] += 1


values = torch.tensor(lane_max_values)

print("Samples checked:", min(NUM_SAMPLES, len(dataset)))
print("Visible lanes:", lane_count)

print()

for limit in LIMITS:
    percentage = 100.0 * beyond[limit] / lane_count

    print(
        f"Lanes extending beyond {int(limit)} m:",
        beyond[limit],
        f"({percentage:.2f}%)",
    )

print()
print("Lane max-forward percentiles:")

for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    value = torch.quantile(values, q).item()

    print(
        f"{int(q * 100)}th percentile:",
        f"{value:.2f} m",
    )
