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
    build_proposal_target,
)
from models.glane3d import GLane3D
from models.losses import GLane3DLoss
from models.matcher import match_keypoints
from datasets.targets import NUM_LANE_CLASSES

DATA_ROOT = (
    "/home/hp/datasets/openlane/"
    "openlane_v1_300"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

sample = dataset[0]

transform = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

image = transform(
    sample["image"]
).unsqueeze(0).to(device)

intrinsic = torch.tensor(
    sample["intrinsic"],
    dtype=torch.float32,
    device=device,
)

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
    device=device,
)


model = GLane3D().to(device)
criterion = GLane3DLoss().to(device)

model.train()


# Ground-truth keypoints live on the same
# longitudinal rows as the BEV anchors.

row_forward = model.bev_grid[:, 0, 0]

gt = build_gt_keypoints(
    sample["lane_lines"],
    row_forward,
    extrinsic,
)

print(
    "Raw sample categories:",
    sorted({
        int(lane["category"])
        for lane in sample["lane_lines"]
    }),
)

print(
    "Mapped GT categories:",
    gt["categories"].tolist(),
)

print(
    "Mapped range:",
    gt["categories"].min().item(),
    "->",
    gt["categories"].max().item(),
)

assert gt["categories"].min() >= 0
assert (
    gt["categories"].max()
    < NUM_LANE_CLASSES
)

print("GT categories:", gt["categories"])
print(
    "Category range:",
    gt["categories"].min().item(),
    "->",
    gt["categories"].max().item(),
)

proposal_target = build_proposal_target(
    sample["lane_lines"],
    model.bev_grid,
    extrinsic,
    positive_radius=2.0,
).unsqueeze(0)


# Full model forward.

output = model(
    image,
    intrinsic,
    extrinsic,
)


# Hungarian matching #1:
# 512 proposals, GT repeated twice.

proposal_rows = (
    output["proposal_indices"][0]
    // model.bev_grid.shape[1]
)

matched_pred_1, matched_gt_1 = match_keypoints(
    anchors=output["proposals"][0],
    refined_points=output["refined_points"][0],
    gt=gt,
    proposal_rows=proposal_rows,
    class_logits=output["class_logits"][0],
    repeat=2,
)


# Hungarian matching #2:
# strongest PointNMS keypoints, no GT repetition.

keep = output["keep_indices"][0]

strong_anchors = output["proposals"][0, keep]

strong_rows = proposal_rows[keep]

strong_classes = output["class_logits"][0, keep]

matched_pred_2, matched_gt_2 = match_keypoints(
    anchors=strong_anchors,
    refined_points=output["strong_points"][0],
    gt=gt,
    proposal_rows=strong_rows,
    class_logits=strong_classes,
    repeat=1,
)


# Directed graph supervision is based on
# the second Hungarian matching.

connection_target = build_connection_targets(
    matched_pred_2,
    matched_gt_2,
    gt,
    num_predictions=len(
        output["strong_points"][0]
    ),
)


# Four GLane3D losses.

losses = criterion(
    seg_logits=output["seg_logits"],
    seg_target=proposal_target,
    x_offset=output["x_offset"][0],
    z=output["z"][0],
    class_logits=output["class_logits"][0],
    matched_pred=matched_pred_1,
    matched_gt=matched_gt_1,
    proposal_anchors=output["proposals"][0],
    gt_points=gt["points"],
    gt_classes=gt["categories"],
    adjacency_logits=output["adjacency_logits"][0],
    adjacency_target=connection_target,
)


model.zero_grad(set_to_none=True)
criterion.zero_grad(set_to_none=True)

losses["total"].backward()


print("Device:", device)

print(
    "GT keypoints:",
    len(gt["points"]),
)

print(
    "Proposal target positives:",
    int(proposal_target.sum().item()),
)

print(
    "First Hungarian matches:",
    len(matched_pred_1),
)

print(
    "Second Hungarian matches:",
    len(matched_pred_2),
)

print(
    "Positive connections:",
    int(connection_target.sum().item()),
)

print()

for name, value in losses.items():
    print(
        f"{name}:",
        value.item(),
    )


print("\nGradient check:")

modules = {
    "backbone": model.backbone,
    "proposal head": model.proposal_head,
    "refiner": model.refiner,
    "connection head": model.connection_head,
}

for name, module in modules.items():
    grad_norm = 0.0
    has_gradient = False

    for parameter in module.parameters():
        if parameter.grad is None:
            continue

        has_gradient = True
        grad_norm += parameter.grad.norm().item()

    print(
        f"{name}:",
        grad_norm
        if has_gradient
        else None,
    )

print(
    "loss weights:",
    criterion.log_vars.grad,
)