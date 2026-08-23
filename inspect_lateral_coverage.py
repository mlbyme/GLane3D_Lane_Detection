import torch

from datasets.openlane import OpenLaneDataset


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"
NUM_SAMPLES = 5000


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

lateral_values = []
lane_lateral_max = []


for index in range(min(NUM_SAMPLES, len(dataset))):
    sample = dataset[index]

    extrinsic = torch.tensor(
        sample["extrinsic"],
        dtype=torch.float32,
    )

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

        forward = vehicle[:, 0]
        lateral = vehicle[:, 1]

        valid = (
            (forward >= 0.0)
            & (forward <= 100.0)
        )

        lateral = lateral[valid]

        if len(lateral) == 0:
            continue

        abs_lateral = lateral.abs()

        lateral_values.append(abs_lateral)
        lane_lateral_max.append(
            abs_lateral.max()
        )


lateral_values = torch.cat(lateral_values)
lane_lateral_max = torch.stack(lane_lateral_max)

print("Samples checked:", min(NUM_SAMPLES, len(dataset)))
print("Visible points within 0-100 m:", len(lateral_values))
print()

print("Point absolute-lateral percentiles:")

for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    value = torch.quantile(
        lateral_values,
        q,
    ).item()

    print(
        f"{int(q * 100)}th percentile:",
        f"{value:.2f} m",
    )

print()
print("Lane max absolute-lateral percentiles:")

for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    value = torch.quantile(
        lane_lateral_max,
        q,
    ).item()

    print(
        f"{int(q * 100)}th percentile:",
        f"{value:.2f} m",
    )
