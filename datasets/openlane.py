import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class OpenLaneDataset(Dataset):
    def __init__(self, data_root, split="training"):
        self.data_root = Path(data_root)
        self.split = split

        if split not in {"training", "validation"}:
            raise ValueError("split must be 'training' or 'validation'")

        annotation_root = (
            self.data_root
            / "annotations"
            / "lane3d_300"
            / split
        )

        self.samples = sorted(annotation_root.rglob("*.json"))

        if not self.samples:
            raise RuntimeError(
                f"No annotation files found in {annotation_root}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        annotation_path = self.samples[index]

        with open(annotation_path) as f:
            annotation = json.load(f)

        image_path = self._get_image_path(annotation["file_path"])

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "lane_lines": annotation["lane_lines"],
            "intrinsic": annotation["intrinsic"],
            "extrinsic": annotation["extrinsic"],
            "image_path": str(image_path),
            "annotation_path": str(annotation_path),
        }

    def _get_image_path(self, file_path):
        relative_path = Path(file_path)

        image_root = (
            self.data_root
            / "images"
            / f"images_{self.split}_0"
        )

        image_path = image_root / relative_path

        if image_path.exists():
            return image_path

        segment_path = Path(*relative_path.parts[1:])

        matches = list(
            (self.data_root / "images").glob(
                f"images_{self.split}_*/{segment_path}"
            )
        )

        if len(matches) == 1:
            return matches[0]

        if not matches:
            raise FileNotFoundError(
                f"Could not find image for annotation path: {file_path}"
            )

        raise RuntimeError(
            f"Found multiple images for annotation path: {file_path}"
        )
