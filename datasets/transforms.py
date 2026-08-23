import torch
from PIL import Image


class ResizeWithIntrinsic:
    def __init__(self, size):
        self.height = size[0]
        self.width = size[1]

    def __call__(self, image, intrinsic):
        old_width, old_height = image.size

        sx = self.width / old_width
        sy = self.height / old_height

        image = image.resize(
            (self.width, self.height),
            Image.BILINEAR,
        )

        K = torch.tensor(intrinsic, dtype=torch.float32).clone()

        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

        return image, K
