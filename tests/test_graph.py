import torch

from models.graph import extract_lanes


points = torch.tensor([
    [10.0, 0.00, 0.0],
    [12.0, 0.05, 0.0],
    [14.0, 0.10, 0.0],
    [16.0, 0.15, 0.0],
])

adjacency = torch.zeros(4, 4)

adjacency[0, 1] = 0.95
adjacency[1, 2] = 0.90
adjacency[2, 3] = 0.93

eps = 1e-5

logits = torch.logit(
    adjacency.clamp(
        eps,
        1.0 - eps,
    )
)

result = extract_lanes(
    points,
    logits,
    threshold=0.5,
)

print("Starts:", result["starts"])
print("Ends:", result["ends"])
print("Number of lanes:", len(result["lanes"]))

for lane in result["lanes"]:
    print("Path:", lane["indices"])
    print("Points:")
    print(lane["points"])