import numpy as np


def save_split(path, features, adjacency, labels):
    features = np.asarray(features)
    adjacency = np.asarray(adjacency)
    labels = np.asarray(labels)
    if features.shape[0] != adjacency.shape[0] or features.shape[0] != labels.shape[0]:
        raise ValueError(
            "features, graphs and labels need the same number of windows, "
            f"we got {features.shape[0]}, {adjacency.shape[0]}, {labels.shape[0]}"
        )
    # one npz per split, not one file per window
    np.savez_compressed(
        path,
        features=features,
        adjacency=adjacency,
        labels=labels,
    )


def load_split(path):
    loaded = np.load(path)
    return loaded["features"], loaded["adjacency"], loaded["labels"]
