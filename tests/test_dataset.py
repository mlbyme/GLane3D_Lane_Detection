import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.openlane import OpenLaneDataset


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


image_transform = transforms.ToTensor()


def collate_fn(batch):
    images = torch.stack([
        image_transform(sample["image"])
        for sample in batch
    ])

    return {
        "images": images,
        "lane_lines": [sample["lane_lines"] for sample in batch],
        "intrinsics": [sample["intrinsic"] for sample in batch],
        "extrinsics": [sample["extrinsic"] for sample in batch],
        "image_paths": [sample["image_path"] for sample in batch],
        "annotation_paths": [sample["annotation_path"] for sample in batch],
    }


dataset = OpenLaneDataset(DATA_ROOT, split="training")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

batch = next(iter(loader))

print("=== BATCH ===")
print("Image tensor shape:", batch["images"].shape)
print("Image tensor dtype:", batch["images"].dtype)
print("Image tensor range:")
print("  min:", batch["images"].min().item())
print("  max:", batch["images"].max().item())

print("Lanes:", len(batch["lane_lines"][0]))
print("Intrinsic:")
print(torch.tensor(batch["intrinsics"][0]))

print("\nImage path:")
print(batch["image_paths"][0])

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())