import torch


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
