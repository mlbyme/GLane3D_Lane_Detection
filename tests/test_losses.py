import torch

from models.losses import GLane3DLoss


seg_logits = torch.randn(
    1,
    1,
    56,
    32,
)

seg_target = torch.zeros(
    1,
    56,
    32,
)

seg_target[:, 10, 10] = 1.0
seg_target[:, 11, 10] = 1.0


num_proposals = 512
num_classes = 11

anchors = torch.randn(
    num_proposals,
    3,
)

x_offset = torch.randn(
    num_proposals,
)

z = torch.randn(
    num_proposals,
)

class_logits = torch.randn(
    num_proposals,
    num_classes,
)

gt_points = torch.randn(
    20,
    3,
)

gt_classes = torch.randint(
    0,
    num_classes,
    (20,),
)

matched_pred = torch.arange(10)
matched_gt = torch.arange(10)

adjacency_logits = torch.randn(
    256,
    256,
)

adjacency_target = torch.zeros(
    256,
    256,
)

adjacency_target[
    torch.arange(9),
    torch.arange(1, 10),
] = 1.0


criterion = GLane3DLoss()

losses = criterion(
    seg_logits=seg_logits,
    seg_target=seg_target,
    x_offset=x_offset,
    z=z,
    class_logits=class_logits,
    matched_pred=matched_pred,
    matched_gt=matched_gt,
    proposal_anchors=anchors,
    gt_points=gt_points,
    gt_classes=gt_classes,
    adjacency_logits=adjacency_logits,
    adjacency_target=adjacency_target,
)

for name, value in losses.items():
    print(
        name,
        value.item(),
    )