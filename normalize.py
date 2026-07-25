import numpy as np

# stats from this patient's train windows only, then reuse on the rest
N_NODES = 8
STD_FLOOR = 1e-8


def node_mean_and_std(train_windows):
    values = np.asarray(train_windows, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(f"train windows should be 4d, we got {values.ndim}d")
    n_windows, n_steps, n_nodes, n_bins = values.shape
    if n_nodes != N_NODES:
        raise ValueError(f"need {N_NODES} nodes, we got {n_nodes}")
    means = []
    stds = []
    node_index = 0
    while node_index < n_nodes:
        node_values = values[:, :, node_index, :]
        mean = float(np.mean(node_values))
        std = float(np.std(node_values))
        if std < STD_FLOOR:
            raise ValueError(f"node {node_index} std is ~0, cant zscore")
        means.append(mean)
        stds.append(std)
        node_index += 1
    return means, stds


def apply_node_zscore(windows, means, stds):
    values = np.asarray(windows, dtype=np.float64).copy()
    if len(means) != N_NODES or len(stds) != N_NODES:
        raise ValueError("means/stds need one number per node")
    node_index = 0
    while node_index < N_NODES:
        values[:, :, node_index, :] = (values[:, :, node_index, :] - means[node_index]) / stds[node_index]
        node_index += 1
    return values


def zscore_from_train(train_windows, val_windows, test_windows):
    means, stds = node_mean_and_std(train_windows)
    train_scaled = apply_node_zscore(train_windows, means, stds)
    val_scaled = apply_node_zscore(val_windows, means, stds)
    test_scaled = apply_node_zscore(test_windows, means, stds)
    return train_scaled, val_scaled, test_scaled, means, stds
