from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.openlane import OpenLaneDataset
from models.ipm import project_openlane_to_image


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


dataset = OpenLaneDataset(DATA_ROOT, split="training")
sample = dataset[0]

image = sample["image"]
K = torch.tensor(sample["intrinsic"], dtype=torch.float32)

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(image)

for lane in sample["lane_lines"]:
    uv_gt = torch.tensor(
        lane["uv"],
        dtype=torch.float32,
    ).T

    xyz = torch.tensor(
        lane["xyz"],
        dtype=torch.float32,
    ).T

    visibility = torch.tensor(
        lane["visibility"],
        dtype=torch.float32,
    )

    uv_proj, depth = project_openlane_to_image(xyz, K)

    valid = (depth > 0) & (visibility > 0.5)

    uv_gt = uv_gt[valid]
    uv_proj = uv_proj[valid]

    if len(uv_gt) == 0:
        continue

    gt_line = ax.plot(
        uv_gt[:, 0],
        uv_gt[:, 1],
        marker="o",
        markersize=3,
        linewidth=1,
    )[0]

    ax.scatter(
        uv_proj[:, 0],
        uv_proj[:, 1],
        marker="x",
        s=18,
        color=gt_line.get_color(),
    )

ax.set_title("OpenLane sample with GT UV and projected XYZ")
ax.set_xlim(0, image.width)
ax.set_ylim(image.height, 0)
ax.set_xlabel("u")
ax.set_ylabel("v")

output_path = OUTPUT_DIR / "sample_overlay.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", output_path)
print("Image size:", image.size)
print("Lane count:", len(sample["lane_lines"]))
print("Image path:", sample["image_path"])
