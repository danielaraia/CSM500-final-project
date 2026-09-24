"""Sensor tiers: zero the features of dropped nodes, keep the pack graphs."""

import numpy as np

from pearson import N_NODES

NODE_NAMES = (  # pack node order, set by edf.py load_run and glue.py _features_for_second
    "eeg_left",
    "eeg_right",
    "ecg",
    "emg",
    "neck_acc",
    "neck_gyr",
    "chest_acc",
    "chest_gyr",
)
SENSOR_TIERS = ("full", "headset", "eeg_acc", "eeg_only")
DEFAULT_TIER = "full"

_KEPT_NODES = {
    "full": tuple(range(N_NODES)),
    "headset": (0, 1, 4, 5),
    "eeg_acc": (0, 1, 4),
    "eeg_only": (0, 1),
}


def kept_nodes(tier):
    if tier not in _KEPT_NODES:
        raise ValueError(
            "sensors has to be full, headset, eeg_acc, or eeg_only, "
            f"we got {tier!r}"
        )
    return _KEPT_NODES[tier]


def mask_features(features, tier):
    kept = kept_nodes(tier)
    values = np.asarray(features)
    if values.ndim < 2 or values.shape[-2] != N_NODES:
        raise ValueError(
            f"features need {N_NODES} nodes on the second to last axis, "
            f"we got shape {values.shape}"
        )
    # full is a no-op: the same array goes back, no copy
    if len(kept) == N_NODES:
        return values
    masked = values.copy()
    dropped = [node for node in range(N_NODES) if node not in kept]
    masked[..., dropped, :] = 0.0
    return masked


def make_sensor_mask_fn(tier):
    kept_nodes(tier)
    # no fn at all for full, same as augment none
    if tier == DEFAULT_TIER:
        return None

    def mask_fn(features):
        return mask_features(features, tier)

    return mask_fn
