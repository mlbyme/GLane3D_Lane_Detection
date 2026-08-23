from pathlib import Path

import matplotlib.pyplot as plt

from models.bev import make_bev_anchor_grid


OUTPUT_DIR = Path("outputs/bev")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


bev = make_bev_anchor_grid(
    height=56,
    width=32,
    forward_range=100.0,
    bev_width=20.0,
)

forward = bev[..., 0].numpy()
lateral = bev[..., 1].numpy()

plt.figure(figsize=(8, 10))

for i in range(bev.shape[0]):
    plt.scatter(
        lateral[i],
        forward[i],
        s=8,
    )

plt.xlabel("Lateral position (m)")
plt.ylabel("Forward position (m)")
plt.title("GLane3D BEV anchor layout - 100 m")
plt.grid(True)
plt.axis("equal")

output_path = (
    OUTPUT_DIR
    / "04_metric_anchor_layout_100m_20mwidth.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved:", output_path)
print("Shape:", bev.shape)

print(
    "Forward range:",
    bev[..., 0].min().item(),
    "->",
    bev[..., 0].max().item(),
)

print(
    "Lateral range:",
    bev[..., 1].min().item(),
    "->",
    bev[..., 1].max().item(),
)