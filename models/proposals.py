import torch


def select_proposals(
    score_map,
    bev_grid,
    num_proposals=512,
):
    scores = torch.sigmoid(score_map)

    batch_size = scores.shape[0]

    scores = scores.reshape(batch_size, -1)

    top_scores, top_indices = torch.topk(
        scores,
        k=min(num_proposals, scores.shape[1]),
        dim=1,
    )

    anchors = bev_grid.reshape(-1, 3)

    selected = []

    for b in range(batch_size):
        selected.append(
            anchors[top_indices[b]]
        )

    selected = torch.stack(selected)

    return selected, top_scores, top_indices

def gather_proposal_features(
    bev_features,
    indices,
):
    features = bev_features.flatten(2).transpose(1, 2)

    gather_indices = indices.unsqueeze(-1).expand(
        -1,
        -1,
        features.shape[-1],
    )

    return torch.gather(
        features,
        1,
        gather_indices,
    )