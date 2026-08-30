import heapq

import torch


def build_connections(
    points,
    adjacency,
    threshold=0.20,
    min_forward_delta=0.5,
    max_forward_delta=4.5,
    max_lateral_delta=0.8,
):
    """
    Build candidate directed edges.

    points:
        [N, 3] as
        [forward, lateral, z]

    adjacency:
        [N, N] probabilities
    """

    connections = (
        adjacency > threshold
    )

    num_points = len(points)

    source = points[:, None, :]
    destination = points[None, :, :]

    forward_delta = (
        destination[..., 0]
        - source[..., 0]
    )

    lateral_delta = (
        destination[..., 1]
        - source[..., 1]
    ).abs()

    geometry_valid = (
        (forward_delta >= min_forward_delta)
        & (forward_delta <= max_forward_delta)
        & (lateral_delta <= max_lateral_delta)
    )

    connections = (
        connections
        & geometry_valid
    )

    # Never connect a point to itself.
    diagonal = torch.eye(
        num_points,
        dtype=torch.bool,
        device=points.device,
    )

    connections = (
        connections
        & ~diagonal
    )

    return connections


def enforce_chain_structure(
    adjacency,
    connections,
):
    """
    Keep a sparse lane-chain graph.

    Each point gets at most:
        one outgoing edge
        one incoming edge

    Candidate edges are considered from
    highest probability to lowest.
    """

    pairs = torch.nonzero(
        connections,
        as_tuple=False,
    )

    if len(pairs) == 0:
        return connections

    scores = adjacency[
        pairs[:, 0],
        pairs[:, 1],
    ]

    order = torch.argsort(
        scores,
        descending=True,
    )

    selected = torch.zeros_like(
        connections
    )

    used_outgoing = set()
    used_incoming = set()

    for index in order:
        source = int(
            pairs[index, 0].item()
        )

        destination = int(
            pairs[index, 1].item()
        )

        if source in used_outgoing:
            continue

        if destination in used_incoming:
            continue

        selected[
            source,
            destination,
        ] = True

        used_outgoing.add(
            source
        )

        used_incoming.add(
            destination
        )

    return selected


def find_start_end_nodes(
    connections,
):
    incoming = connections.sum(
        dim=0
    )

    outgoing = connections.sum(
        dim=1
    )

    starts = torch.where(
        (incoming == 0)
        & (outgoing > 0)
    )[0]

    ends = torch.where(
        (incoming > 0)
        & (outgoing == 0)
    )[0]

    return starts, ends


def shortest_path(
    adjacency,
    connections,
    start,
    end,
):
    num_nodes = adjacency.shape[0]

    distances = [
        float("inf")
    ] * num_nodes

    previous = [
        -1
    ] * num_nodes

    distances[start] = 0.0

    queue = [
        (0.0, start)
    ]

    while queue:
        distance, node = (
            heapq.heappop(
                queue
            )
        )

        if (
            distance
            > distances[node]
        ):
            continue

        if node == end:
            break

        neighbors = torch.where(
            connections[node]
        )[0]

        for neighbor_tensor in neighbors:
            neighbor = int(
                neighbor_tensor.item()
            )

            probability = float(
                adjacency[
                    node,
                    neighbor,
                ].item()
            )

            edge_cost = (
                1.0 - probability
            )

            new_distance = (
                distance
                + edge_cost
            )

            if (
                new_distance
                < distances[neighbor]
            ):
                distances[
                    neighbor
                ] = new_distance

                previous[
                    neighbor
                ] = node

                heapq.heappush(
                    queue,
                    (
                        new_distance,
                        neighbor,
                    ),
                )

    if (
        distances[end]
        == float("inf")
    ):
        return None

    path = []
    node = end

    while node != -1:
        path.append(node)

        if node == start:
            break

        node = previous[node]

    if (
        not path
        or path[-1] != start
    ):
        return None

    return list(
        reversed(path)
    )


def extract_lanes(
    points,
    adjacency_logits,
    threshold=0.20,
    min_points=2,
    min_forward_delta=0.5,
    max_forward_delta=4.5,
    max_lateral_delta=0.8,
    enforce_chain=True,
):
    adjacency = torch.sigmoid(
        adjacency_logits
    )

    connections = build_connections(
        points=points,
        adjacency=adjacency,
        threshold=threshold,
        min_forward_delta=(
            min_forward_delta
        ),
        max_forward_delta=(
            max_forward_delta
        ),
        max_lateral_delta=(
            max_lateral_delta
        ),
    )

    if enforce_chain:
        connections = (
            enforce_chain_structure(
                adjacency,
                connections,
            )
        )

    starts, ends = (
        find_start_end_nodes(
            connections
        )
    )

    lanes = []

    for start_tensor in starts:
        start = int(
            start_tensor.item()
        )

        for end_tensor in ends:
            end = int(
                end_tensor.item()
            )

            path = shortest_path(
                adjacency,
                connections,
                start,
                end,
            )

            if path is None:
                continue

            if len(path) < min_points:
                continue

            lane_points = (
                points[path]
            )

            lanes.append({
                "indices": path,
                "points": lane_points,
            })

    return {
        "lanes": lanes,
        "adjacency": adjacency,
        "connections": connections,
        "starts": starts,
        "ends": ends,
    }