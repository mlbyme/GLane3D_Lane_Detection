import torch
from torchvision import transforms

from datasets.openlane import OpenLaneDataset
from models.backbone import ResNet18Backbone


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


dataset = OpenLaneDataset(DATA_ROOT, split="training")
sample = dataset[0]

image = image_transform(sample["image"]).unsqueeze(0)

device = torch.device("cuda")

model = ResNet18Backbone(pretrained=True)
model = model.to(device)
model.eval()

image = image.to(device)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    features = model(image)

torch.cuda.synchronize()

print("Device:", device)
print("Input shape:", image.shape)
print("Feature shape:", features.shape)
print("Feature dtype:", features.dtype)

print()
print(
    "Allocated:",
    round(torch.cuda.memory_allocated() / 1024**2, 2),
    "MB",
)
print(
    "Reserved:",
    round(torch.cuda.memory_reserved() / 1024**2, 2),
    "MB",
)
print(
    "Peak allocated:",
    round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    "MB",
)
