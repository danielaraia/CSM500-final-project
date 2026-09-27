import math

import numpy as np
import pytest

from evaluation.simple_baseline import (
    apply_zscore,
    fit_logistic,
    fit_on_train_windows,
    logits_for_windows,
    node_mean_energy,
    pick_threshold_from_val_or_monitor,
    predict_logits,
    threshold_score_map,
    zscore_from_train,
)


def test_node_mean_energy_is_per_node_mean_over_time_and_bins():
    features = np.zeros((2, 4, 8, 3))
    features[0, :, 0, :] = 2.0
    features[1, :, 1, :] = 4.0
    energy = node_mean_energy(features)
    assert energy.shape == (2, 8)
    assert energy[0, 0] == pytest.approx(2.0)
    assert energy[0, 1] == pytest.approx(0.0)
    assert energy[1, 1] == pytest.approx(4.0)


def test_node_mean_energy_rejects_wrong_rank():
    with pytest.raises(ValueError, match="4 axes"):
        node_mean_energy(np.zeros((2, 8)))


def test_zscore_fails_when_a_column_has_no_spread():
    x = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ValueError, match="no spread"):
        zscore_from_train(x)


def test_logistic_separates_two_clusters():
    rng = np.random.default_rng(123)
    neg = rng.normal(loc=0.0, scale=0.1, size=(20, 8))
    pos = rng.normal(loc=2.0, scale=0.1, size=(20, 8))
    x = np.concatenate([neg, pos], axis=0)
    y = np.concatenate([np.zeros(20), np.ones(20)])
    mean, std = zscore_from_train(x)
    z = apply_zscore(x, mean, std)
    weights = fit_logistic(z, y, pos_weight=1.0)
    logits = predict_logits(z, weights)
    assert logits[20:].min() > logits[:20].max()


def test_fit_on_train_does_not_use_held_out_windows():
    features = np.zeros((8, 60, 8, 128))
    node = 0
    while node < 8:
        window = 0
        while window < 8:
            features[window, :, node, :] = 0.1 * float(node) + 0.01 * float(window)
            window += 1
        node += 1
    index = 0
    while index < 4:
        features[index, :, 0, :] = 0.0
        index += 1
    while index < 8:
        features[index, :, 0, :] = 3.0
        index += 1
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    model = fit_on_train_windows(features[:6], labels[:6])
    again = fit_on_train_windows(features[:6], labels[:6])
    assert np.allclose(model["weights"], again["weights"])
    extra = features.copy()
    extra[6:8, :, 0, :] = 50.0
    other = fit_on_train_windows(extra, labels)
    train_logits = logits_for_windows(features[:6], model)
    assert train_logits[3] < train_logits[4]
    assert not np.allclose(model["weights"], other["weights"])


def test_one_class_train_fails():
    features = np.zeros((4, 60, 8, 128))
    features[:, :, 0, :] = np.linspace(0.0, 1.0, 4).reshape(4, 1, 1)
    with pytest.raises(ValueError, match="positive train windows"):
        fit_on_train_windows(features, [0, 0, 0, 0])


def test_threshold_map_has_only_val_and_monitor_seconds():
    score_at = threshold_score_map(
        [10, 11],
        [0.1, 0.2],
        [20, 21],
        [0.3, 0.4],
    )
    assert set(score_at.keys()) == {10, 11, 20, 21}
    assert 99 not in score_at


def test_operating_point_ignores_a_high_test_logit():
    seizures = [(80_000.0, 80_040.0)]
    duration = 8 * 3600
    val_ends = list(range(0, 400))
    val_logits = [0.1] * 400
    monitor_ends = list(range(1000, 1400))
    monitor_logits = [0.2] * 400
    threshold, info = pick_threshold_from_val_or_monitor(
        val_ends,
        val_logits,
        monitor_ends,
        monitor_logits,
        seizures,
        duration,
        0,
        duration,
    )
    assert info["threshold_set"] == "val"
    assert threshold != 99.0
    assert threshold < math.inf
