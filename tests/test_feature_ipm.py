import torch
from torchvision import transforms

from datasets.openlane import OpenLaneDataset
from models.backbone import ResNet18Backbone
from models.bev import make_bev_anchor_grid
from models.ipm import sample_bev_features


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

sample = dataset[0]

image = image_transform(
    sample["image"]
).unsqueeze(0)

intrinsic = torch.tensor(
    sample["intrinsic"],
    dtype=torch.float32,
)

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
)

bev = make_bev_anchor_grid(
    height=56,
    width=32,
    forward_range=100.0,
    bev_width=34.0,
)

device = torch.device("cuda")

model = ResNet18Backbone(
    pretrained=True,
).to(device)

model.eval()

image = image.to(device)
intrinsic = intrinsic.to(device)
extrinsic = extrinsic.to(device)
bev = bev.to(device)

torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    front_features = model(image)

    bev_features, valid = sample_bev_features(
        front_features,
        bev,
        intrinsic,
        extrinsic,
        image_size=(1280, 1920),
    )

torch.cuda.synchronize()

print("Front-view features:")
print(front_features.shape)

print("\nBEV features:")
print(bev_features.shape)

print("\nValid BEV anchors:")
print(valid.sum().item(), "/", valid.numel())

print("\nBEV feature range:")
print(
    bev_features.min().item(),
    "->",
    bev_features.max().item(),
)

print("\nPeak GPU memory:")
print(
    round(
        torch.cuda.max_memory_allocated()
        / 1024**2,
        2,
    ),
    "MB",
)
