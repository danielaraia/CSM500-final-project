import numpy as np
import pytest

from normalize import apply_node_zscore, node_mean_and_std, zscore_from_train


def test_constant_node_fails():
    train = np.ones((2, 2, 8, 2))
    with pytest.raises(ValueError, match="std is ~0"):
        node_mean_and_std(train)


def test_zscore_on_a_node_with_two_values():
    train = np.zeros((2, 1, 8, 1))
    node_index = 0
    while node_index < 8:
        train[0, 0, node_index, 0] = 0.0
        train[1, 0, node_index, 0] = 2.0
        node_index += 1
    means, stds = node_mean_and_std(train)
    assert means[0] == pytest.approx(1.0)
    other = np.ones((1, 1, 8, 1))
    scaled = apply_node_zscore(other, means, stds)
    # (1 - 1) / 1 = 0
    assert scaled[0, 0, 0, 0] == pytest.approx(0.0)


def test_val_uses_train_mean_not_its_own():
    train = np.zeros((2, 1, 8, 1))
    node_index = 0
    while node_index < 8:
        train[0, 0, node_index, 0] = 0.0
        train[1, 0, node_index, 0] = 2.0
        node_index += 1
    val = np.full((1, 1, 8, 1), 100.0)
    test = np.full((1, 1, 8, 1), 3.0)
    train_z, val_z, test_z, means, stds = zscore_from_train(train, val, test)
    assert means[0] == pytest.approx(1.0)
    assert stds[0] == pytest.approx(1.0)
    # if we had used val's own stats, 100 would become 0
    assert val_z[0, 0, 0, 0] == pytest.approx(99.0)
    assert test_z[0, 0, 0, 0] == pytest.approx(2.0)
    assert train_z[0, 0, 0, 0] == pytest.approx(-1.0)
