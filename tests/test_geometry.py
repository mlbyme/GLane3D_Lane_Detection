import torch

from datasets.openlane import OpenLaneDataset
from datasets.transforms import ResizeWithIntrinsic


DATA_ROOT = "/home/hp/datasets/openlane/openlane_v1_300"

dataset = OpenLaneDataset(
    DATA_ROOT,
    split="training",
)

resize = ResizeWithIntrinsic((360, 640))

sample = dataset[0]

image, K = resize(
    sample["image"],
    sample["intrinsic"],
)

print("=== GEOMETRY TEST ===")
print("Original image:", sample["image"].size)
print("Resized image:", image.size)

print("\nOriginal K:")
print(torch.tensor(sample["intrinsic"]))

print("\nResized K:")
print(K)

print("\nExpected:")
print("fx ≈ 686.54")
print("fy ≈ 579.27")
print("cx ≈ 317.47")
print("cy ≈ 178.29")
