import torch
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)

from datasets.openlane import OpenLaneDataset
from datasets.targets import (
    build_connection_targets,
    build_gt_keypoints,
)
from models.glane3d import GLane3D
from models.matcher import match_keypoints


DATA_ROOT = (
    "/home/hp/datasets/openlane/"
    "openlane_v1_300"
)

CHECKPOINT_PATH = (
    "checkpoints/glane3d_epoch_03.pt"
)

SAMPLE_INDEX = 0


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


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="validation",
)

sample = dataset[SAMPLE_INDEX]


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


model = GLane3D().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


with torch.no_grad():
    output = model(
        image,
        intrinsic,
        extrinsic,
    )


# -------------------------------------------------
# Build GT keypoints
# -------------------------------------------------

row_forward = model.bev_grid[:, 0, 0]

gt = build_gt_keypoints(
    sample["lane_lines"],
    row_forward,
    extrinsic[0],
)

if gt is None or len(gt["points"]) == 0:
    raise RuntimeError(
        "No GT keypoints found for this sample"
    )


# -------------------------------------------------
# Second Hungarian matching
# -------------------------------------------------

proposal_rows = (
    output["proposal_indices"][0]
    // model.bev_grid.shape[1]
)

keep = output["keep_indices"][0]

points = (
    output["strong_points"][0]
)

matched_pred_2, matched_gt_2 = (
    match_keypoints(
        anchors=output["proposals"][
            0, keep
        ],
        refined_points=points,
        gt=gt,
        proposal_rows=proposal_rows[
            keep
        ],
        class_logits=output[
            "class_logits"
        ][0, keep],
        repeat=1,
    )
)


# -------------------------------------------------
# Build true connection target
# -------------------------------------------------

connection_target = (
    build_connection_targets(
        matched_pred_2,
        matched_gt_2,
        gt,
        num_predictions=len(
            points
        ),
    )
)


adjacency_logits = (
    output["adjacency_logits"][0]
)

connection_target = connection_target.to(
    adjacency_logits.device
)


# -------------------------------------------------
# Compare predicted probabilities
# -------------------------------------------------

probabilities = torch.sigmoid(
    adjacency_logits
)

positive_mask = (
    connection_target > 0
)

negative_mask = (
    connection_target == 0
)

positive_scores = probabilities[
    positive_mask
]

negative_scores = probabilities[
    negative_mask
]


print(
    "Validation sample:",
    SAMPLE_INDEX,
)

print(
    "Strong keypoints:",
    len(points),
)

print(
    "GT keypoints:",
    len(gt["points"]),
)

print(
    "Second Hungarian matches:",
    len(matched_pred_2),
)

print(
    "Positive GT edges:",
    positive_scores.numel(),
)

print(
    "Negative edges:",
    negative_scores.numel(),
)


if positive_scores.numel() == 0:
    raise RuntimeError(
        "No positive connection targets "
        "were created for this sample"
    )


print()
print("POSITIVE EDGE SCORES")

print(
    "Mean:",
    positive_scores.mean().item(),
)

print(
    "Median:",
    positive_scores.median().item(),
)

print(
    "Min:",
    positive_scores.min().item(),
)

print(
    "Max:",
    positive_scores.max().item(),
)


print()
print("NEGATIVE EDGE SCORES")

print(
    "Mean:",
    negative_scores.mean().item(),
)

print(
    "Median:",
    negative_scores.median().item(),
)

print(
    "Min:",
    negative_scores.min().item(),
)

print(
    "Max:",
    negative_scores.max().item(),
)


print()
print("THRESHOLD ANALYSIS")

for threshold in [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
]:
    positive_recall = (
        positive_scores
        >= threshold
    ).float().mean().item()

    false_positive_rate = (
        negative_scores
        >= threshold
    ).float().mean().item()

    positives_kept = int(
        (
            positive_scores
            >= threshold
        ).sum().item()
    )

    false_edges_kept = int(
        (
            negative_scores
            >= threshold
        ).sum().item()
    )

    print(
        f"{threshold:.2f} | "
        f"true edges kept "
        f"{positives_kept:4d}/"
        f"{positive_scores.numel():4d} | "
        f"recall "
        f"{positive_recall:.3f} | "
        f"false edges "
        f"{false_edges_kept:5d} | "
        f"FPR "
        f"{false_positive_rate:.5f}"
    )

positive_pairs = torch.nonzero(
    positive_mask,
    as_tuple=False,
)

false_high_mask = (
    negative_mask
    & (probabilities >= 0.20)
)

false_pairs = torch.nonzero(
    false_high_mask,
    as_tuple=False,
)


def print_pair_stats(
    name,
    pairs,
):
    if len(pairs) == 0:
        print(name, "none")
        return

    source = pairs[:, 0]
    target = pairs[:, 1]

    forward_delta = (
        points[target, 0]
        - points[source, 0]
    )

    lateral_delta = (
        points[target, 1]
        - points[source, 1]
    ).abs()

    print()
    print(name)
    print(
        "Count:",
        len(pairs),
    )

    print(
        "Forward delta mean:",
        forward_delta.mean().item(),
    )

    print(
        "Forward delta min/max:",
        forward_delta.min().item(),
        "->",
        forward_delta.max().item(),
    )

    print(
        "Lateral delta mean:",
        lateral_delta.mean().item(),
    )

    print(
        "Lateral delta min/max:",
        lateral_delta.min().item(),
        "->",
        lateral_delta.max().item(),
    )

    print(
        "Backward:",
        int(
            (forward_delta < 0)
            .sum()
            .item()
        ),
    )

    print(
        "Same/near row:",
        int(
            (
                forward_delta.abs()
                < 0.25
            )
            .sum()
            .item()
        ),
    )


print_pair_stats(
    "TRUE EDGES",
    positive_pairs,
)

print_pair_stats(
    "FALSE EDGES >= 0.20",
    false_pairs,
)