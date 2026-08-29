import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    prob = torch.sigmoid(logits)

    pt = torch.where(
        targets > 0.5,
        prob,
        1.0 - prob,
    )

    alpha_t = torch.where(
        targets > 0.5,
        alpha,
        1.0 - alpha,
    )

    loss = alpha_t * (1.0 - pt).pow(gamma) * bce

    return loss.mean()


class GLane3DLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.log_vars = nn.Parameter(
            torch.zeros(4)
        )

    def forward(
        self,
        seg_logits,
        seg_target,
        x_offset,
        z,
        class_logits,
        matched_pred,
        matched_gt,
        proposal_anchors,
        gt_points,
        gt_classes,
        adjacency_logits=None,
        adjacency_target=None,
    ):
        loss_kp = F.binary_cross_entropy_with_logits(
            seg_logits.squeeze(1),
            seg_target,
        )
        
        has_matches = (
            len(matched_pred) > 0
        )

        if has_matches:
            pred_x = x_offset[matched_pred]
            pred_z = z[matched_pred]

            anchor_x = proposal_anchors[
                matched_pred,
                1,
            ]

            target_x = gt_points[
                matched_gt,
                1,
            ]

            target_z = gt_points[
                matched_gt,
                2,
            ]

            target_offset = (
                target_x - anchor_x
            )

            loss_reg = (
                F.l1_loss(
                    pred_x,
                    target_offset,
                )
                + F.l1_loss(
                    pred_z,
                    target_z,
                )
            )

            class_target = gt_classes[
                matched_gt
            ]

            loss_cls = F.cross_entropy(
                class_logits[matched_pred],
                class_target,
            )

        else:
            loss_reg = x_offset.sum() * 0.0
            loss_cls = class_logits.sum() * 0.0

        if (
            adjacency_logits is not None
            and adjacency_target is not None
        ):
            loss_conn = focal_loss(
                adjacency_logits,
                adjacency_target,
            )
        else:
            loss_conn = seg_logits.sum() * 0.0

        losses = torch.stack([
            loss_kp,
            loss_reg,
            loss_conn,
            loss_cls,
        ])

        active = torch.tensor(
            [
                1.0,
                float(has_matches),
                1.0,
                float(has_matches),
            ],
            dtype=losses.dtype,
            device=losses.device,
        )

        weights = torch.exp(
            -self.log_vars
        )

        weighted = (
            weights * losses
            + self.log_vars
        )

        total = (
            weighted * active
        ).sum()

        return {
            "total": total,
            "keypoint": loss_kp,
            "regression": loss_reg,
            "connection": loss_conn,
            "classification": loss_cls,
        }