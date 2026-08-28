import torch

from models.bev import make_bev_anchor_grid
from models.connection import ConnectionHead
from models.heads import ProposalHead
from models.pointnms import point_nms
from models.proposals import (
    gather_proposal_features,
    select_proposals,
)
from models.refinement import ProposalRefiner


bev_features = torch.randn(
    1,
    512,
    56,
    32,
)

bev = make_bev_anchor_grid(
    height=56,
    width=32,
    forward_range=100.0,
    bev_width=34.0,
)

proposal_head = ProposalHead()
refiner = ProposalRefiner()
connection_head = ConnectionHead()

score_map = proposal_head(bev_features)

proposals, _, indices = select_proposals(
    score_map,
    bev,
    num_proposals=512,
)

proposal_features = gather_proposal_features(
    bev_features,
    indices,
)

output = refiner(
    proposal_features,
    bev_features,
)

refined = proposals.clone()

refined[..., 1] += output["x_offset"]
refined[..., 2] = output["z"]

confidence = torch.softmax(
    output["class_logits"],
    dim=-1,
).max(dim=-1).values

keep = point_nms(
    refined[0],
    confidence[0],
    max_points=256,
)

points = refined[:, keep]
connection_features = output["connection"][:, keep]

adjacency_logits = connection_head(
    points,
    connection_features,
)

adjacency = torch.sigmoid(
    adjacency_logits
)

print("Kept points:", points.shape)
print(
    "Connection features:",
    connection_features.shape,
)
print(
    "Adjacency:",
    adjacency.shape,
)
print(
    "Adjacency range:",
    adjacency.min().item(),
    "->",
    adjacency.max().item(),
)
