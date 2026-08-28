import torch
import torch.nn as nn

from models.backbone import ResNet18Backbone
from models.bev import make_bev_anchor_grid
from models.connection import ConnectionHead
from models.heads import ProposalHead
from models.ipm import sample_bev_features
from models.pointnms import point_nms
from models.proposals import (
    gather_proposal_features,
    select_proposals,
)
from models.refinement import ProposalRefiner
from datasets.targets import NUM_LANE_CLASSES


class GLane3D(nn.Module):
    def __init__(
        self,
        num_classes=NUM_LANE_CLASSES,
        num_proposals=512,
        max_keypoints=256,
    ):
        super().__init__()

        self.num_proposals = num_proposals
        self.max_keypoints = max_keypoints

        self.backbone = ResNet18Backbone(
            pretrained=True,
        )

        self.proposal_head = ProposalHead(
            in_channels=512,
        )

        self.refiner = ProposalRefiner(
            feature_dim=512,
            num_classes=num_classes,
        )

        self.connection_head = ConnectionHead()

        bev = make_bev_anchor_grid(
            height=56,
            width=32,
            forward_range=100.0,
            bev_width=34.0,
        )

        self.register_buffer(
            "bev_grid",
            bev,
            persistent=False,
        )

    def forward(
        self,
        image,
        intrinsic,
        extrinsic,
    ):
        if image.shape[0] != 1:
            raise ValueError(
                "Current implementation supports batch size 1"
            )

        if intrinsic.ndim == 3:
            intrinsic = intrinsic[0]

        if extrinsic.ndim == 3:
            extrinsic = extrinsic[0]

        front_features = self.backbone(
            image
        )

        image_size = (
            image.shape[-2],
            image.shape[-1],
        )

        bev_features, valid_bev = (
            sample_bev_features(
                front_features,
                self.bev_grid,
                intrinsic,
                extrinsic,
                image_size,
            )
        )

        seg_logits = self.proposal_head(
            bev_features
        )

        proposals, proposal_scores, indices = (
            select_proposals(
                seg_logits,
                self.bev_grid,
                num_proposals=self.num_proposals,
            )
        )

        proposal_features = (
            gather_proposal_features(
                bev_features,
                indices,
            )
        )

        output = self.refiner(
            proposal_features,
            bev_features,
        )

        refined = proposals.clone()

        refined[..., 1] = (
            refined[..., 1]
            + output["x_offset"]
        )

        refined[..., 2] = output["z"]

        class_probs = torch.softmax(
            output["class_logits"],
            dim=-1,
        )

        confidence = class_probs.max(
            dim=-1,
        ).values

        keep = point_nms(
            refined[0].detach(),
            confidence[0].detach(),
            max_points=self.max_keypoints,
        )

        strong_points = refined[:, keep]

        strong_connection = (
            output["connection"][:, keep]
        )

        adjacency_logits = (
            self.connection_head(
                strong_points,
                strong_connection,
            )
        )

        return {
            "bev_features": bev_features,
            "valid_bev": valid_bev,
            "seg_logits": seg_logits,
            "proposals": proposals,
            "proposal_scores": proposal_scores,
            "proposal_indices": indices,
            "x_offset": output["x_offset"],
            "z": output["z"],
            "class_logits": output["class_logits"],
            "connection_features": output["connection"],
            "refined_points": refined,
            "confidence": confidence,
            "keep_indices": keep,
            "strong_points": strong_points,
            "strong_connection_features": strong_connection,
            "adjacency_logits": adjacency_logits,
        }