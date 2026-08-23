import torch


def make_custom_bev_grid(height=56, width=32, bev_width=20.0):
    spacing = torch.linspace(
        0.5,
        1.5,
        height - 1,
        dtype=torch.float32,
    )

    y = torch.zeros(height, dtype=torch.float32)
    y[1:] = torch.cumsum(spacing, dim=0)

    rows = []

    for i in range(height):
        t = i / (height - 1)

        x_start = -bev_width / 4 * (1.0 - t)
        x_end = bev_width / 4 * (1.0 - t)

        x_start -= bev_width / 2 * t
        x_end += bev_width / 2 * t

        x = torch.linspace(
            x_start,
            x_end,
            width,
            dtype=torch.float32,
        )

        rows.append(x)

    x = torch.stack(rows)

    y = y[:, None].expand(
        height,
        width,
    )

    return torch.stack(
        [x, y],
        dim=-1,
    )