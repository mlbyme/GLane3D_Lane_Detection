import torch

from models.ipm import camera_to_vehicle

OPENLANE_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21
]

CATEGORY_TO_INDEX = {
    category_id: index
    for index, category_id in enumerate(
        OPENLANE_CATEGORY_IDS
    )
}

NUM_LANE_CLASSES = len(
    OPENLANE_CATEGORY_IDS
)

def interpolate_lane_to_rows(
    xyz,
    visibility,
    row_forward,
    extrinsic,
):
    xyz = xyz[visibility > 0.5]

    if len(xyz) < 2:
        return None

    vehicle = camera_to_vehicle(
        xyz,
        extrinsic,
    )

    order = torch.argsort(vehicle[:, 0])
    vehicle = vehicle[order]

    forward = vehicle[:, 0]

    keep = torch.ones(
        len(forward),
        dtype=torch.bool,
        device=forward.device,
    )

    keep[1:] = (
        forward[1:] - forward[:-1]
    ).abs() > 1e-4

    vehicle = vehicle[keep]
    forward = vehicle[:, 0].contiguous()

    if len(vehicle) < 2:
        return None

    valid_rows = (
        (row_forward >= forward[0])
        & (row_forward <= forward[-1])
    )

    row_indices = torch.where(
        valid_rows
    )[0]

    if len(row_indices) == 0:
        return None

    rows = row_forward[row_indices]

    upper = torch.searchsorted(
        forward,
        rows,
    )

    upper = upper.clamp(
        1,
        len(forward) - 1,
    )

    lower = upper - 1

    y0 = forward[lower]
    y1 = forward[upper]

    t = (
        (rows - y0)
        / (y1 - y0).clamp(min=1e-6)
    )

    lateral = (
        vehicle[lower, 1]
        + t * (
            vehicle[upper, 1]
            - vehicle[lower, 1]
        )
    )

    z = (
        vehicle[lower, 2]
        + t * (
            vehicle[upper, 2]
            - vehicle[lower, 2]
        )
    )

    return row_indices, lateral, z

def build_bev_targets(
    lane_lines,
    bev,
    extrinsic,
):
    height, width = bev.shape[:2]

    score = torch.zeros(
        height,
        width,
        dtype=torch.float32,
        device=bev.device,
    )

    x_offset = torch.zeros_like(score)
    z_target = torch.zeros_like(score)

    row_forward = bev[:, 0, 0]

    for lane in lane_lines:
        xyz = torch.tensor(
            lane["xyz"],
            dtype=torch.float32,
            device=bev.device,
        ).T

        visibility = torch.tensor(
            lane["visibility"],
            dtype=torch.float32,
            device=bev.device,
        )

        result = interpolate_lane_to_rows(
            xyz,
            visibility,
            row_forward,
            extrinsic,
        )

        if result is None:
            continue

        rows, target_x, target_z = result

        for row, lateral, z in zip(
            rows,
            target_x,
            target_z,
        ):
            anchors = bev[row, :, 1]

            column = torch.argmin(
                torch.abs(anchors - lateral)
            )

            offset = lateral - anchors[column]

            if score[row, column] == 0:
                score[row, column] = 1.0
                x_offset[row, column] = offset
                z_target[row, column] = z
            elif abs(offset) < abs(x_offset[row, column]):
                x_offset[row, column] = offset
                z_target[row, column] = z

    return {
        "score": score,
        "x_offset": x_offset,
        "z": z_target,
    }

def build_gt_keypoints(
    lane_lines,
    row_forward,
    extrinsic,
):
    points = []
    rows = []
    lane_ids = []
    categories = []

    for lane_id, lane in enumerate(lane_lines):
        xyz = torch.tensor(
            lane["xyz"],
            dtype=torch.float32,
            device=row_forward.device,
        ).T

        visibility = torch.tensor(
            lane["visibility"],
            dtype=torch.float32,
            device=row_forward.device,
        )

        result = interpolate_lane_to_rows(
            xyz,
            visibility,
            row_forward,
            extrinsic,
        )

        if result is None:
            continue

        row_indices, lateral, z = result

        forward = row_forward[row_indices]

        lane_points = torch.stack(
            [
                forward,
                lateral,
                z,
            ],
            dim=-1,
        )

        points.append(lane_points)
        rows.append(row_indices)

        lane_ids.append(
            torch.full(
                (len(row_indices),),
                lane_id,
                dtype=torch.long,
                device=row_forward.device,
            )
        )

        raw_category = int(lane["category"])
        
        if raw_category == 0:
            continue
        
        if raw_category not in CATEGORY_TO_INDEX:
            raise ValueError(
                f"Unknown OpenLane category: {raw_category}"
            )

        category = CATEGORY_TO_INDEX[
            raw_category
        ]

        categories.append(
            torch.full(
                (len(row_indices),),
                category,
                dtype=torch.long,
                device=row_forward.device,
            )
        )

    if not points:
        return None

    return {
        "points": torch.cat(points),
        "rows": torch.cat(rows),
        "lane_ids": torch.cat(lane_ids),
        "categories": torch.cat(categories),
}

def build_connection_targets(
    matched_pred,
    matched_gt,
    gt,
    num_predictions,
):
    target = torch.zeros(
        num_predictions,
        num_predictions,
        dtype=torch.float32,
        device=matched_pred.device,
    )

    if len(matched_pred) == 0:
        return target

    lane_ids = gt["lane_ids"][matched_gt]
    rows = gt["rows"][matched_gt]

    for lane_id in lane_ids.unique():
        mask = lane_ids == lane_id

        pred = matched_pred[mask]
        lane_rows = rows[mask]

        if len(pred) < 2:
            continue

        order = torch.argsort(
            lane_rows
        )

        pred = pred[order]

        for i in range(len(pred) - 1):
            target[
                pred[i],
                pred[i + 1],
            ] = 1.0

    return target


##

def build_proposal_target(
    lane_lines,
    bev,
    extrinsic,
    positive_radius=2.0,
):
    height, width = bev.shape[:2]

    target = torch.zeros(
        height,
        width,
        dtype=torch.float32,
        device=bev.device,
    )

    row_forward = bev[:, 0, 0]

    for lane in lane_lines:
        xyz = torch.tensor(
            lane["xyz"],
            dtype=torch.float32,
            device=bev.device,
        ).T

        visibility = torch.tensor(
            lane["visibility"],
            dtype=torch.float32,
            device=bev.device,
        )

        result = interpolate_lane_to_rows(
            xyz,
            visibility,
            row_forward,
            extrinsic,
        )

        if result is None:
            continue

        rows, lateral, _ = result

        for row, target_x in zip(rows, lateral):
            anchor_x = bev[row, :, 1]

            distance = torch.abs(
                anchor_x - target_x
            )

            target[row, distance <= positive_radius] = 1.0

    return target