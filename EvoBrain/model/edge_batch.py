def time_major_edge_features(adjacency):
    batch_size, timesteps, num_nodes, _ = adjacency.shape
    return (
        adjacency.permute(1, 0, 2, 3)
        .reshape(timesteps, batch_size, num_nodes * num_nodes)
        .unsqueeze(-1)
    )


def canonicalize_eigenvector_signs(eigenvectors):
    indices = eigenvectors.abs().argmax(dim=0)
    anchors = eigenvectors.gather(0, indices.unsqueeze(0)).squeeze(0)
    signs = anchors.sign()
    signs = signs.masked_fill(signs == 0, 1)
    return eigenvectors * signs
