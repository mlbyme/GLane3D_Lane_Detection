import torch
import torch.nn.functional as F

def openlane_to_camera(xyz):
    """
    Convert OpenLane/Waymo coordinates to standard
    pinhole-camera coordinates.

    OpenLane / Waymo:
        X = forward
        Y = left
        Z = up

    Standard camera:
        X = right
        Y = down
        Z = forward

    Args:
        xyz: Tensor [..., 3]

    Returns:
        camera_xyz: Tensor [..., 3]
    """

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    return torch.stack(
        [
            -y,
            -z,
            x,
        ],
        dim=-1,
    )

def project_openlane_to_image(
    xyz,
    intrinsic,
):
    camera_xyz = openlane_to_camera(xyz)

    depth = camera_xyz[..., 2]
    safe_depth = depth.clamp(min=1e-6)

    uvw = torch.matmul(
        camera_xyz,
        intrinsic.transpose(-1, -2),
    )

    uv = (
        uvw[..., :2]
        / safe_depth.unsqueeze(-1)
    )

    return uv, depth


def make_bev_grid(
    x_range=(0.0, 30.0),
    y_range=(-10.0, 10.0),
    z=0.0,
    x_step=0.5,
    y_step=0.5,
):
    """
    Create an OpenLane/Waymo-coordinate BEV grid.

    X = forward
    Y = left
    Z = up
    """

    x = torch.arange(
        x_range[0],
        x_range[1],
        x_step,
        dtype=torch.float32,
    )

    y = torch.arange(
        y_range[0],
        y_range[1],
        y_step,
        dtype=torch.float32,
    )

    xx, yy = torch.meshgrid(
        x,
        y,
        indexing="ij",
    )

    zz = torch.full_like(xx, z)

    return torch.stack(
        [xx, yy, zz],
        dim=-1,
    ).reshape(-1, 3)

def vehicle_to_camera(
    xyz,
    extrinsic,
):
    vehicle_to_camera_matrix = (
        torch.linalg.inv(extrinsic)
    )

    ones = torch.ones(
        *xyz.shape[:-1],
        1,
        dtype=xyz.dtype,
        device=xyz.device,
    )

    xyz_h = torch.cat(
        [xyz, ones],
        dim=-1,
    )

    camera_h = torch.matmul(
        xyz_h,
        vehicle_to_camera_matrix.transpose(
            -1,
            -2,
        ),
    )

    return camera_h[..., :3]


def sample_bev_features(
    features,
    bev,
    intrinsic,
    extrinsic,
    image_size,
):
    height, width = image_size
    batch_size = features.shape[0]

    if intrinsic.ndim == 2:
        intrinsic = intrinsic.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

    if extrinsic.ndim == 2:
        extrinsic = extrinsic.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

    if intrinsic.shape[0] != batch_size:
        raise ValueError(
            "Intrinsic batch size does not match "
            "feature batch size"
        )

    if extrinsic.shape[0] != batch_size:
        raise ValueError(
            "Extrinsic batch size does not match "
            "feature batch size"
        )

    anchors = bev.reshape(
        1,
        -1,
        3,
    ).expand(
        batch_size,
        -1,
        -1,
    )

    camera_xyz = vehicle_to_camera(
        anchors,
        extrinsic,
    )

    uv, depth = project_openlane_to_image(
        camera_xyz,
        intrinsic,
    )

    valid = (
        (depth > 0)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < height)
    )

    x = (
        2.0 * uv[..., 0]
        / (width - 1)
        - 1.0
    )

    y = (
        2.0 * uv[..., 1]
        / (height - 1)
        - 1.0
    )

    grid = torch.stack(
        [x, y],
        dim=-1,
    )

    grid = grid.reshape(
        batch_size,
        bev.shape[0],
        bev.shape[1],
        2,
    )

    bev_features = F.grid_sample(
        features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    valid = valid.reshape(
        batch_size,
        bev.shape[0],
        bev.shape[1],
    )

    return bev_features, valid

def camera_to_vehicle(
    xyz,
    extrinsic,
):
    ones = torch.ones(
        *xyz.shape[:-1],
        1,
        dtype=xyz.dtype,
        device=xyz.device,
    )

    xyz_h = torch.cat(
        [xyz, ones],
        dim=-1,
    )

    vehicle_h = torch.matmul(
        xyz_h,
        extrinsic.transpose(
            -1,
            -2,
        ),
    )

    return vehicle_h[..., :3]