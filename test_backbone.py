import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.openlane import OpenLaneDataset
from models.backbone import ResNet18Backbone


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


image_transform = transforms.Compose([
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
])


def collate_fn(batch):
    images = torch.stack([
        image_transform(sample["image"])
        for sample in batch
    ])

    return {
        "images": images,
        "lane_lines": [
            sample["lane_lines"]
            for sample in batch
        ],
        "intrinsics": [
            sample["intrinsic"]
            for sample in batch
        ],
        "extrinsics": [
            sample["extrinsic"]
            for sample in batch
        ],
    }


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn,
)

batch = next(iter(loader))

images = batch["images"].to(device)

model = ResNet18Backbone(
    pretrained=True,
).to(device)

model.eval()

with torch.no_grad():
    features = model(images)

print("=== RESNET-18 TEST ===")

print("Device:", device)

print("GPU:", torch.cuda.get_device_name(0))

print("\nInput:")
print(images.shape)
print(images.dtype)

print("\nOutput:")
print(features.shape)
print(features.dtype)

print("\nFeature range:")
print("min:", features.min().item())
print("max:", features.max().item())

print("\nGPU memory:")
print(
    round(
        torch.cuda.memory_allocated() / 1024**2,
        2,
    ),
    "MB allocated",
)
