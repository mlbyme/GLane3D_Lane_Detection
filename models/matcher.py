import torch
from scipy.optimize import linear_sum_assignment


def match_keypoints(
    anchors,
    refined_points,
    gt,
    proposal_rows,
    class_logits=None,
    repeat=2,
    max_anchor_distance=2.0,
    max_refined_distance=1.0,
    class_weight=1.0,
):
    if gt is None or len(gt["points"]) == 0:
        empty = torch.empty(
            0,
            dtype=torch.long,
            device=anchors.device,
        )
        return empty, empty

    gt_points = gt["points"]
    gt_rows = gt["rows"]
    gt_classes = gt["categories"]

    original_indices = torch.arange(
        len(gt_points),
        device=anchors.device,
    )

    if repeat > 1:
        gt_points = gt_points.repeat_interleave(
            repeat,
            dim=0,
        )
        gt_rows = gt_rows.repeat_interleave(
            repeat,
            dim=0,
        )
        gt_classes = gt_classes.repeat_interleave(
            repeat,
            dim=0,
        )
        original_indices = original_indices.repeat_interleave(
            repeat,
            dim=0,
        )

        if not (
        len(gt_points)
        == len(gt_rows)
        == len(gt_classes)
        == len(original_indices)
        ):
            raise RuntimeError(
            "GT fields have inconsistent lengths after repetition: "
            f"points={len(gt_points)}, "
            f"rows={len(gt_rows)}, "
            f"classes={len(gt_classes)}, "
            f"indices={len(original_indices)}"
        )

    pred_lateral = refined_points[:, 1]
    pred_z = refined_points[:, 2]

    anchor_lateral = anchors[:, 1]

    gt_lateral = gt_points[:, 1]
    gt_z = gt_points[:, 2]

    lateral_cost = torch.abs(
        pred_lateral[:, None]
        - gt_lateral[None, :]
    )

    z_cost = torch.abs(
        pred_z[:, None]
        - gt_z[None, :]
    )

    cost = lateral_cost + z_cost

    if class_logits is not None:
        class_probs = torch.softmax(
            class_logits,
            dim=-1,
        )

        if gt_classes.min() < 0:
            raise ValueError(
                "GT class index is negative"
            )

        if gt_classes.max() >= class_probs.shape[-1]:
            raise ValueError(
                "GT class index exceeds classifier size: "
                f"max={gt_classes.max().item()}, "
                f"num_classes={class_probs.shape[-1]}"
            )

        class_cost = -class_probs[
            :,
            gt_classes,
        ]

        cost = (
            cost
            + class_weight * class_cost
        )

    same_row = (
        proposal_rows[:, None]
        == gt_rows[None, :]
    )

    anchor_distance = torch.abs(
        anchor_lateral[:, None]
        - gt_lateral[None, :]
    )

    refined_distance = torch.abs(
        pred_lateral[:, None]
        - gt_lateral[None, :]
    )

    valid = (
        same_row
        & (anchor_distance <= max_anchor_distance)
        & (refined_distance <= max_refined_distance)
    )

    cost[~valid] = 1e6

    pred_indices, repeated_gt_indices = (
        linear_sum_assignment(
            cost.detach().cpu().numpy()
        )
    )

    pred_indices = torch.tensor(
        pred_indices,
        dtype=torch.long,
        device=anchors.device,
    )

    repeated_gt_indices = torch.tensor(
        repeated_gt_indices,
        dtype=torch.long,
        device=anchors.device,
    )

    matched_cost = cost[
        pred_indices,
        repeated_gt_indices,
    ]

    keep = matched_cost < 1e5

    pred_indices = pred_indices[keep]
    repeated_gt_indices = repeated_gt_indices[keep]

    gt_indices = original_indices[
        repeated_gt_indices
    ]

    return pred_indices, gt_indices