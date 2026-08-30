from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets.openlane import OpenLaneDataset
from models.glane3d import GLane3D
from models.graph import extract_lanes
from models.ipm import (
    project_openlane_to_image,
    vehicle_to_camera,
)


DATA_ROOT = (
    "/home/hp/datasets/openlane/"
    "openlane_v1_300"
)

CHECKPOINT_PATH = (
    "checkpoints/glane3d_epoch_03.pt"
)

OUTPUT_DIR = Path("outputs/predictions")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TARGET_SEGMENT = (
    "segment-1051897962568538022_238_170_258_170_with_camera_labels"
)

TARGET_STEM = (
    "150923225629601600"
)

# Raw GLane3D graph threshold.
# This is still being calibrated on validation data.
ADJACENCY_THRESHOLD = 0.20


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

transform = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -------------------------------------------------
# Load validation sample
# -------------------------------------------------
""""
dataset = OpenLaneDataset(
    DATA_ROOT,
    split="validation",
)

sample = dataset[SAMPLE_INDEX]

"""

dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

target_annotation = (
    Path(DATA_ROOT)
    / "annotations"
    / "lane3d_300"
    / "training"
    / TARGET_SEGMENT
    / f"{TARGET_STEM}.json"
)

print(
    "Target annotation:",
    target_annotation,
)

if not target_annotation.exists():
    raise RuntimeError(
        f"Annotation not found: {target_annotation}"
    )

try:
    sample_index = dataset.samples.index(
        target_annotation
    )
except ValueError:
    raise RuntimeError(
        "Annotation exists but is not present "
        "in OpenLaneDataset.samples"
    )

sample = dataset[sample_index]

print(
    "Found dataset index:",
    sample_index,
)

print(
    "Annotation:",
    target_annotation,
)

print(
    "Image:",
    sample["image_path"],
)

image = transform(
    sample["image"]
).unsqueeze(0).to(device)

intrinsic = torch.tensor(
    sample["intrinsic"],
    dtype=torch.float32,
    device=device,
).unsqueeze(0)

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
    device=device,
).unsqueeze(0)


# -------------------------------------------------
# Load trained model
# -------------------------------------------------

model = GLane3D().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


# -------------------------------------------------
# Inference
# -------------------------------------------------

with torch.inference_mode():
    output = model(
        image,
        intrinsic,
        extrinsic,
    )


points = output["strong_points"][0]

keep = output["keep_indices"][0]

strong_logits = (
    output["class_logits"][0, keep]
)

strong_probs = torch.softmax(
    strong_logits,
    dim=-1,
)

strong_scores, strong_classes = (
    strong_probs.max(dim=-1)
)

adjacency_logits = (
    output["adjacency_logits"][0]
)

adjacency = torch.sigmoid(
    adjacency_logits
)


print(
    "Validation sample:",
    sample_index,
)

print(
    "Strong keypoints:",
    len(points),
)

print(
    "GT lane instances:",
    len(sample["lane_lines"]),
)

print(
    "Adjacency range:",
    adjacency.min().item(),
    "->",
    adjacency.max().item(),
)


# -------------------------------------------------
# Threshold diagnostic
# -------------------------------------------------

thresholds = [
    0.60,
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
    0.15,
    0.10,
    0.05,
]

print()
print("THRESHOLD SWEEP")

for threshold in thresholds:
    graph_test = extract_lanes(
        points,
        adjacency_logits,
        threshold=threshold,
        min_points=2,
    )

    num_edges = int(
        graph_test["connections"]
        .sum()
        .item()
    )

    print(
        f"{threshold:.2f} | "
        f"edges {num_edges:5d} | "
        f"starts {len(graph_test['starts']):3d} | "
        f"ends {len(graph_test['ends']):3d} | "
        f"lanes {len(graph_test['lanes']):3d}"
    )


# -------------------------------------------------
# Extract raw graph at selected threshold
# -------------------------------------------------

graph = extract_lanes(
    points,
    adjacency_logits,
    threshold=ADJACENCY_THRESHOLD,
    min_points=2,
)

connections = graph["connections"]

print()
print(
    "Chosen threshold:",
    ADJACENCY_THRESHOLD,
)

print(
    "Extracted lanes:",
    len(graph["lanes"]),
)


# -------------------------------------------------
# Graph direction diagnostics
# -------------------------------------------------

edge_pairs = torch.nonzero(
    connections,
    as_tuple=False,
)

if len(edge_pairs) > 0:
    source = edge_pairs[:, 0]
    destination = edge_pairs[:, 1]

    forward_delta = (
        points[destination, 0]
        - points[source, 0]
    )

    forward_edges = int(
        (forward_delta > 1e-4)
        .sum()
        .item()
    )

    backward_edges = int(
        (forward_delta < -1e-4)
        .sum()
        .item()
    )

    same_row_edges = int(
        (
            forward_delta.abs()
            <= 1e-4
        )
        .sum()
        .item()
    )

else:
    forward_edges = 0
    backward_edges = 0
    same_row_edges = 0


print(
    "Forward edges:",
    forward_edges,
)

print(
    "Backward edges:",
    backward_edges,
)

print(
    "Same-row edges:",
    same_row_edges,
)


# -------------------------------------------------
# Extracted lane statistics
# -------------------------------------------------

lane_lengths = []
lane_spans = []

for lane in graph["lanes"]:
    lane_points = lane["points"]

    lane_lengths.append(
        len(lane["indices"])
    )

    span = (
        lane_points[:, 0].max()
        - lane_points[:, 0].min()
    )

    lane_spans.append(
        float(span.item())
    )


print(
    "Lane path lengths:",
    lane_lengths,
)

print(
    "Lane forward spans:",
    [
        round(span, 2)
        for span in lane_spans
    ],
)


# -------------------------------------------------
# BEV visualization
# -------------------------------------------------

points_cpu = (
    points
    .detach()
    .float()
    .cpu()
)

connections_cpu = (
    connections
    .detach()
    .cpu()
)

fig, ax = plt.subplots(
    figsize=(10, 12)
)


# All PointNMS keypoints
ax.scatter(
    points_cpu[:, 1],
    points_cpu[:, 0],
    s=12,
    alpha=0.35,
    label="Strong keypoints",
)


# All graph edges above threshold
for i in range(len(points_cpu)):
    neighbors = torch.where(
        connections_cpu[i]
    )[0]

    for j_tensor in neighbors:
        j = int(j_tensor.item())

        p1 = points_cpu[i]
        p2 = points_cpu[j]

        ax.plot(
            [
                p1[1].item(),
                p2[1].item(),
            ],
            [
                p1[0].item(),
                p2[0].item(),
            ],
            linewidth=0.8,
            alpha=0.25,
        )


# Dijkstra-extracted paths
for lane in graph["lanes"]:
    lane_points = (
        lane["points"]
        .detach()
        .float()
        .cpu()
    )

    ax.plot(
        lane_points[:, 1],
        lane_points[:, 0],
        linewidth=2.5,
    )

    ax.scatter(
        lane_points[:, 1],
        lane_points[:, 0],
        s=28,
    )


ax.set_xlabel(
    "Lateral position (m)"
)

ax.set_ylabel(
    "Forward position (m)"
)

ax.set_xlim(
    -17,
    17,
)

ax.set_ylim(
    0,
    100,
)

ax.set_title(
    "GLane3D raw predicted graph in BEV "
    f"(threshold={ADJACENCY_THRESHOLD:.2f})"
)

ax.grid(
    alpha=0.15
)

ax.legend(
    loc="upper right"
)


bev_output = (
    OUTPUT_DIR
    / "08_prediction_bev_epoch03.png"
)

fig.savefig(
    bev_output,
    dpi=160,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "Saved:",
    bev_output,
)


# Project ALL predicted 3D keypoints onto image

vehicle_points = (
    points
    .detach()
    .float()
    .to(device)
)

camera_points = vehicle_to_camera(
    vehicle_points,
    extrinsic[0],
)

uv, depth = project_openlane_to_image(
    camera_points,
    intrinsic[0],
)

valid = (
    (depth > 0)
    & (uv[:, 0] >= 0)
    & (
        uv[:, 0]
        < sample["image"].width
    )
    & (uv[:, 1] >= 0)
    & (
        uv[:, 1]
        < sample["image"].height
    )
)

uv_valid = (
    uv[valid]
    .detach()
    .float()
    .cpu()
)

scores_valid = (
    strong_scores[valid]
    .detach()
    .float()
    .cpu()
)

classes_valid = (
    strong_classes[valid]
    .detach()
    .cpu()
)

z_valid = (
    vehicle_points[valid, 2]
    .detach()
    .float()
    .cpu()
)

fig, ax = plt.subplots(
    figsize=(16, 10)
)

ax.imshow(
    sample["image"]
)

scatter = ax.scatter(
    uv_valid[:, 0],
    uv_valid[:, 1],
    c=z_valid,
    s=15 + 35 * scores_valid,
    alpha=0.9,
)

ax.axis("off")

ax.set_title(
    "GLane3D predicted 3D keypoints "
    "projected onto camera image"
)

colorbar = fig.colorbar(
    scatter,
    ax=ax,
    fraction=0.025,
    pad=0.01,
)

colorbar.set_label(
    "Predicted Z (m)"
)

keypoint_output = (
    OUTPUT_DIR
    / "10_predicted_3d_keypoints_epoch03.png"
)

fig.savefig(
    keypoint_output,
    dpi=160,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "Projected 3D keypoints:",
    len(uv_valid),
)

print(
    "Saved:",
    keypoint_output,
)


# Camera-image visualization

fig, ax = plt.subplots(
    figsize=(16, 10)
)

ax.imshow(
    sample["image"]
)


for lane in graph["lanes"]:
    vehicle_points = (
        lane["points"]
        .detach()
        .float()
        .to(device)
    )

    camera_points = (
        vehicle_to_camera(
            vehicle_points,
            extrinsic[0],
        )
    )

    uv, depth = (
        project_openlane_to_image(
            camera_points,
            intrinsic[0],
        )
    )

    valid = (
        (depth > 0)
        & (uv[:, 0] >= 0)
        & (
            uv[:, 0]
            < sample["image"].width
        )
        & (uv[:, 1] >= 0)
        & (
            uv[:, 1]
            < sample["image"].height
        )
    )

    uv = (
        uv[valid]
        .detach()
        .float()
        .cpu()
    )

    if len(uv) < 2:
        continue

    ax.plot(
        uv[:, 0],
        uv[:, 1],
        linewidth=3,
    )

    ax.scatter(
        uv[:, 0],
        uv[:, 1],
        s=22,
    )


ax.axis("off")

ax.set_title(
    "GLane3D epoch 3 raw graph prediction"
)


image_output = (
    OUTPUT_DIR
    / "09_prediction_overlay_epoch03.png"
)

fig.savefig(
    image_output,
    dpi=160,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "Saved:",
    image_output,
)