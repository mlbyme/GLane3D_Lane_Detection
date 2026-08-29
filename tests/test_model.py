import torch
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)

from datasets.openlane import OpenLaneDataset
from models.glane3d import GLane3D


DATA_ROOT = (
    "/home/hp/datasets/openlane/"
    "openlane_v1_300"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

sample = dataset[0]

transform = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

image = transform(
    sample["image"]
).unsqueeze(0).to(device)

intrinsic = torch.tensor(
    sample["intrinsic"],
    dtype=torch.float32,
    device=device,
)

extrinsic = torch.tensor(
    sample["extrinsic"],
    dtype=torch.float32,
    device=device,
)

model = GLane3D().to(device)
model.eval()

with torch.no_grad():
    output = model(
        image,
        intrinsic,
        extrinsic,
    )

print(
    "BEV:",
    output["bev_features"].shape,
)

print(
    "Segmentation:",
    output["seg_logits"].shape,
)

print(
    "Proposals:",
    output["proposals"].shape,
)

print(
    "Class logits:",
    output["class_logits"].shape,
)

print(
    "Refined:",
    output["refined_points"].shape,
)

print(
    "Strong points:",
    output["strong_points"][0].shape,
)

print(
    "Adjacency:",
    output["adjacency_logits"][0].shape,
)

print(
    "Keep indices:",
    output["keep_indices"][0].shape,
)