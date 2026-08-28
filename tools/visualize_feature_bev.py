from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from datasets.openlane import OpenLaneDataset
from models.backbone import ResNet18Backbone
from models.bev import make_bev_anchor_grid
from models.ipm import sample_bev_features


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

OUTPUT_DIR = Path("outputs/bev")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

with torch.no_grad():
    front_features = model(image)

    bev_features, valid = sample_bev_features(
        front_features,
        bev,
        intrinsic,
        extrinsic,
        image_size=(1280, 1920),
    )

feature_map = bev_features[0].mean(dim=0).cpu()
valid = valid.cpu()

feature_map = feature_map * valid

plt.figure(figsize=(10, 8))
plt.imshow(
    feature_map,
    origin="lower",
    aspect="auto",
    interpolation="bilinear",
)

plt.title("GLane3D BEV feature response")
plt.xlabel("Lateral anchor index")
plt.ylabel("Forward anchor index")
plt.colorbar(label="Mean feature activation")

output_path = (
    OUTPUT_DIR
    / "06_bev_feature_response_100m_34mwidth.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved:", output_path)
print("Feature shape:", bev_features.shape)
print(
    "Valid anchors:",
    valid.sum().item(),
    "/",
    valid.numel(),
)
