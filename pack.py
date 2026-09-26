import numpy as np

from data.labels import LABEL_INTERICTAL, LABEL_SOP


def save_split(path, features, adjacency, labels, ends=None):
    features = np.asarray(features)
    adjacency = np.asarray(adjacency)
    labels = np.asarray(labels)
    if features.shape[0] != adjacency.shape[0] or features.shape[0] != labels.shape[0]:
        raise ValueError(
            "features, graphs and labels need the same number of windows, "
            f"we got {features.shape[0]}, {adjacency.shape[0]}, {labels.shape[0]}"
        )
    # one npz per split, not one file per window
    if ends is None:
        np.savez_compressed(
            path,
            features=features,
            adjacency=adjacency,
            labels=labels,
        )
        return
    ends = np.asarray(ends)
    if ends.shape[0] != features.shape[0]:
        raise ValueError(
            "ends need the same number of windows as features, "
            f"we got {ends.shape[0]} and {features.shape[0]}"
        )
    np.savez_compressed(
        path,
        features=features,
        adjacency=adjacency,
        labels=labels,
        ends=ends,
    )


def load_split(path):
    loaded = np.load(path)
    ends = None
    if "ends" in loaded.files:
        ends = loaded["ends"]
    return loaded["features"], loaded["adjacency"], loaded["labels"], ends


def save_named_labels(path, features, adjacency, named_labels, ends=None):
    numbers = []
    index = 0
    while index < len(named_labels):
        numbers.append(label_to_class_number(named_labels[index]))
        index += 1
    save_split(
        path,
        features,
        adjacency,
        np.asarray(numbers, dtype=np.int64),
        ends=ends,
    )


def label_to_class_number(label):
    if label == LABEL_SOP:
        return 1
    if label == LABEL_INTERICTAL:
        return 0
    raise ValueError(f"cant pack label {label}, only sop and interictal")
