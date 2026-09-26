import numpy as np
from sklearn.feature_selection import mutual_info_regression

from graphs.pearson import (
    ELECTRICAL_SAMPLES_1S,
    ELECTRICAL_SAMPLES_4S,
    MOVEMENT_SAMPLES_4S,
    N_ELECTRICAL_NODES,
    N_MOVEMENT_NODES,
    N_NODES,
    series_for_pair,
)

MI_NEIGHBORS = 3
MI_RANDOM_STATE = 123


def mutual_information(series_a, series_b):
    if len(series_a) != len(series_b):
        raise ValueError(
            "mutual information needs the two series of the same length, "
            f"we got {len(series_a)} and {len(series_b)}"
        )
    if len(series_a) < 2:
        raise ValueError("need at least 2 samples or we cant do mutual information")

    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("mutual information needs 1d series")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError("cant do mutual information, a series has nan or inf")
    if np.max(a) == np.min(a) or np.max(b) == np.min(b):
        raise ValueError("cant do mutual information, a series is constant")

    values = mutual_info_regression(
        a.reshape(-1, 1),
        b,
        n_neighbors=MI_NEIGHBORS,
        random_state=MI_RANDOM_STATE,
    )
    return float(values[0])


def build_mi_graph(
    electrical_one_second,
    electrical_four_seconds,
    movement_four_seconds,
):
    electrical_1s = _require_shape(
        electrical_one_second, (N_ELECTRICAL_NODES, ELECTRICAL_SAMPLES_1S), "electrical_one_second"
    )
    electrical_4s = _require_shape(
        electrical_four_seconds,
        (N_ELECTRICAL_NODES, ELECTRICAL_SAMPLES_4S),
        "electrical_four_seconds",
    )
    movement_4s = _require_shape(
        movement_four_seconds, (N_MOVEMENT_NODES, MOVEMENT_SAMPLES_4S), "movement_four_seconds"
    )

    graph = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for node_a in range(N_NODES):
        for node_b in range(node_a, N_NODES):
            series_a, series_b = series_for_pair(
                node_a, node_b, electrical_1s, electrical_4s, movement_4s
            )
            value = mutual_information(series_a, series_b)
            graph[node_a, node_b] = value
            graph[node_b, node_a] = value
    return graph


def _require_shape(array, expected_shape, name):
    values = np.asarray(array, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError(f"{name} should be {expected_shape}, we got {values.shape}")
    return values
