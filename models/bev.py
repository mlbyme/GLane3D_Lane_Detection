import torch


def make_bev_anchor_grid(
    height=56,
    width=32,
    forward_range=100.0,
    bev_width=34.0,
):
    spacing = torch.linspace(
        0.5,
        1.5,
        height - 1,
        dtype=torch.float32,
    )

    spacing = spacing / spacing.sum()
    spacing = spacing * forward_range

    forward = torch.zeros(
        height,
        dtype=torch.float32,
    )

    forward[1:] = torch.cumsum(
        spacing,
        dim=0,
    )

    lateral_rows = []

    for i in range(height):
        t = i / (height - 1)

        half_width = bev_width * (
            0.25 + 0.25 * t
        )

        lateral = torch.linspace(
            -half_width,
            half_width,
            width,
            dtype=torch.float32,
        )

        lateral_rows.append(lateral)

    lateral = torch.stack(lateral_rows)

    forward = forward[:, None].expand(
        height,
        width,
    )

    ground = torch.zeros_like(forward)

    return torch.stack(
        [
            forward,
            lateral,
            ground,
        ],
        dim=-1,
    )