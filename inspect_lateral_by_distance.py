import torch

from datasets.openlane import OpenLaneDataset


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"
NUM_SAMPLES = 5000

BINS = [
    (0.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
    (60.0, 80.0),
    (80.0, 100.0),
]


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

values = {
    interval: []
    for interval in BINS
}


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
        lateral = vehicle[:, 1].abs()

        for start, end in BINS:
            valid = (
                (forward >= start)
                & (forward < end)
            )

            if valid.any():
                values[(start, end)].append(
                    lateral[valid]
                )


print(
    "Samples checked:",
    min(NUM_SAMPLES, len(dataset)),
)

for interval in BINS:
    start, end = interval

    print(
        f"\n{int(start)}-{int(end)} m:"
    )

    if not values[interval]:
        print("No points")
        continue

    lateral = torch.cat(
        values[interval]
    )

    print("Points:", len(lateral))

    for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
        value = torch.quantile(
            lateral,
            q,
        ).item()

        print(
            f"{int(q * 100)}th:",
            f"{value:.2f} m",
        )
