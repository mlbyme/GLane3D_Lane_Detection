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
    """
    Project OpenLane XYZ points into image coordinates.

    Args:
        xyz:
            Tensor [N, 3] in OpenLane/Waymo coordinates.

        intrinsic:
            Tensor [3, 3].

    Returns:
        uv:
            Tensor [N, 2].

        depth:
            Tensor [N].
    """

    camera_xyz = openlane_to_camera(xyz)

    depth = camera_xyz[..., 2]

    safe_depth = depth.clamp(min=1e-6)

    uvw = camera_xyz @ intrinsic.T

    uv = uvw[..., :2] / safe_depth.unsqueeze(-1)

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

def vehicle_to_camera(xyz, extrinsic):
    vehicle_to_camera_matrix = torch.linalg.inv(extrinsic)

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

    camera_h = xyz_h @ vehicle_to_camera_matrix.T

    return camera_h[..., :3]


def sample_bev_features(
    features,
    bev,
    intrinsic,
    extrinsic,
    image_size,
):
    height, width = image_size

    camera_xyz = vehicle_to_camera(
        bev.reshape(-1, 3),
        extrinsic,
    )

    uv, depth = project_openlane_to_image(
        camera_xyz,
        intrinsic,
    )

    valid = (
        (depth > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )

    x = 2.0 * uv[:, 0] / (width - 1) - 1.0
    y = 2.0 * uv[:, 1] / (height - 1) - 1.0

    grid = torch.stack(
        [x, y],
        dim=-1,
    )

    grid = grid.reshape(
        1,
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
        bev.shape[0],
        bev.shape[1],
    )

    return bev_features, valid

def camera_to_vehicle(xyz, extrinsic):
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

    vehicle_h = xyz_h @ extrinsic.T

    return vehicle_h[..., :3]