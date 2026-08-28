from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.openlane import OpenLaneDataset
from datasets.targets import interpolate_lane_to_rows
from models.bev import make_bev_anchor_grid


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

OUTPUT_DIR = Path("outputs/bev")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

sample = dataset[0]

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
)

bev = make_bev_anchor_grid(
    height=56,
    width=32,
    forward_range=100.0,
    bev_width=34.0,
)

row_forward = bev[:, 0, 0]

plt.figure(figsize=(10, 12))

plt.scatter(
    bev[..., 1].reshape(-1),
    bev[..., 0].reshape(-1),
    s=7,
    alpha=0.25,
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

    result = interpolate_lane_to_rows(
        xyz,
        visibility,
        row_forward,
        extrinsic,
    )

    if result is None:
        continue

    rows, target_x, target_z = result

    target_y = row_forward[rows]

    anchor_x = []

    for row, x in zip(rows, target_x):
        lateral = bev[row, :, 1]

        column = torch.argmin(
            torch.abs(lateral - x)
        )

        anchor_x.append(
            lateral[column]
        )

    anchor_x = torch.stack(anchor_x)

    line = plt.plot(
        target_x,
        target_y,
        linewidth=2,
    )[0]

    color = line.get_color()

    plt.scatter(
        target_x,
        target_y,
        s=24,
        color=color,
    )

    plt.scatter(
        anchor_x,
        target_y,
        marker="x",
        s=28,
        color=color,
    )

    for x0, x1, y in zip(
        anchor_x,
        target_x,
        target_y,
    ):
        plt.plot(
            [x0, x1],
            [y, y],
            linewidth=1,
            color=color,
        )

plt.xlabel("Lateral position (m)")
plt.ylabel("Forward position (m)")
plt.title(
    "BEV anchors, lane targets and lateral offsets"
)

output_path = (
    OUTPUT_DIR
    / "07_bev_anchor_target_alignment.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved:", output_path)