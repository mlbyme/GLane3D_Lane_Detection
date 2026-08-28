import torch
from torchvision.ops import nms


def point_nms(
    points,
    scores,
    max_points=256,
    thresh_x=1.5,
    thresh_y=0.1,
    scale=10.0,
):
    lateral = points[:, 1]
    forward = points[:, 0]

    x1 = torch.round(
        lateral * scale
        - (scale / 2) * thresh_x
    )

    x2 = torch.round(
        lateral * scale
        + (scale / 2) * thresh_x
    )

    y1 = torch.round(
        forward * scale
        - (scale / 2) * thresh_y
    )

    y2 = torch.round(
        forward * scale
        + (scale / 2) * thresh_y
    )

    boxes = torch.stack(
        [x1, y1, x2, y2],
        dim=1,
    )

    keep = nms(
        boxes,
        scores,
        iou_threshold=0.1,
    )

    return keep[:max_points]