import torch

from datasets.openlane import OpenLaneDataset
from datasets.targets import build_gt_keypoints
from models.bev import make_bev_anchor_grid
from models.matcher import match_keypoints


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

sample = dataset[0]

bev = make_bev_anchor_grid(
    height=56,
    width=32,
    forward_range=100.0,
    bev_width=34.0,
)

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
)

row_forward = bev[:, 0, 0]

gt = build_gt_keypoints(
    sample["lane_lines"],
    row_forward,
    extrinsic,
)

anchors = bev.reshape(-1, 3)

proposal_indices = torch.arange(
    min(512, len(anchors)),
)

proposal_anchors = anchors[
    proposal_indices
]

proposal_rows = (
    proposal_indices
    // bev.shape[1]
)

refined = proposal_anchors.clone()

pred_indices, gt_indices = match_keypoints(
    proposal_anchors,
    refined,
    gt,
    proposal_rows,
    repeat=2,
)

print("GT keypoints:", len(gt["points"]))
print("Proposals:", len(proposal_anchors))
print("Matches:", len(pred_indices))

print(
    "First matched proposal indices:",
    pred_indices[:10],
)

print(
    "First matched GT indices:",
    gt_indices[:10],
)
