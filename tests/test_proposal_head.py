import torch

from models.bev import make_bev_anchor_grid
from models.heads import ProposalHead
from models.proposals import select_proposals


features = torch.randn(
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

head = ProposalHead()

score_map = head(features)

proposals, scores, indices = select_proposals(
    score_map,
    bev,
    num_proposals=512,
)

print("Score map:", score_map.shape)
print("Proposals:", proposals.shape)
print("Scores:", scores.shape)
print("Indices:", indices.shape)