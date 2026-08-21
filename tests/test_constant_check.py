import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

from constant_check import (
    checkpoint_health,
    class_mix,
    ends_time_ordered,
    graph_spread,
    labels_match_clock,
    state_dicts_max_abs_diff,
    tiny_batch_learning,
)


def test_mixed_labels_count_both_classes():
    mix = class_mix([0, 1, 0, 1, 1])
    assert mix["n_sop"] == 3
    assert mix["n_interictal"] == 2
    assert mix["mixed"] is True


def test_all_interictal_is_not_mixed():
    mix = class_mix([0, 0, 0])
    assert mix["n_sop"] == 0
    assert mix["n_interictal"] == 3
    assert mix["mixed"] is False


def test_bad_class_number_fails():
    with pytest.raises(ValueError, match="we got 2"):
        class_mix([0, 2])


def test_ends_in_clock_order_pass_and_scrambled_fail():
    assert ends_time_ordered([179, 189, 199]) is True
    assert ends_time_ordered([199, 189, 209]) is False
    assert ends_time_ordered([10, 10]) is False
    assert ends_time_ordered([42]) is True


def test_packed_labels_match_the_seizure_clock():
    onset = 10_000
    seizures = [(onset, onset + 100)]
    # sop is 7900 to 9700, interictal is far before the 60 min gap
    ends = [100, 8000]
    labels = [0, 1]
    assert labels_match_clock(ends, labels, seizures) == 0
    assert labels_match_clock(ends, [1, 1], seizures) == 1


def test_dead_graph_is_the_same_on_every_window():
    same = np.eye(8).reshape(1, 1, 8, 8)
    adj = np.repeat(same, 4, axis=0)
    adj = np.repeat(adj, 60, axis=1)
    spread = graph_spread(adj)
    assert spread["dead"] is True
    assert spread["n_identical_to_first"] == 4
    assert spread["max_mean_abs_diff"] == pytest.approx(0.0)


def test_live_graph_varies_across_windows():
    adj = np.zeros((3, 60, 8, 8))
    adj[0, :, 0, 1] = 0.2
    adj[0, :, 1, 0] = 0.2
    adj[1, :, 0, 1] = 0.5
    adj[1, :, 1, 0] = 0.5
    adj[2, :, 0, 1] = 0.9
    adj[2, :, 1, 0] = 0.9
    node = 0
    while node < 8:
        adj[:, :, node, node] = 1.0
        node += 1
    spread = graph_spread(adj)
    assert spread["dead"] is False
    assert spread["n_identical_to_first"] == 1
    assert spread["max_mean_abs_diff"] > 0.0
    assert spread["off_diag_std"] > 0.0


def test_tiny_batch_has_nonzero_grad_and_moves_weights():
    torch.manual_seed(123)
    batch = 8
    features = np.random.default_rng(123).normal(size=(batch, 60, 8, 128))
    adj = np.repeat(np.eye(8).reshape(1, 1, 8, 8), batch, axis=0)
    adj = np.repeat(adj, 60, axis=1)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    row = tiny_batch_learning(features, adj, labels, pos_weight=7.0, n_steps=3)
    assert row["grad_nonzero"] is True
    assert row["weight_moved"] is True
    assert row["weight_delta"] > 0.0
    assert row["grad_abs_sum"] > 0.0
    assert len(row["losses"]) == 3


def test_tiny_batch_fails_when_one_class_is_missing():
    features = np.zeros((4, 60, 8, 128))
    adj = np.zeros((4, 60, 8, 8))
    with pytest.raises(ValueError, match="both SOP and interictal"):
        tiny_batch_learning(features, adj, [0, 0, 0, 0], pos_weight=7.0)


def test_checkpoint_health_flags_nan_and_compares_two_states():
    folder = tempfile.mkdtemp()
    try:
        good = {
            "epoch": 1,
            "model": {"w": torch.tensor([[0.4, -0.1]])},
            "optimizer": {"state": {0: {"exp_avg": torch.tensor([[0.2]])}}},
            "monitor_loss": 1.2,
        }
        path = os.path.join(folder, "best.pth.tar")
        torch.save(good, path)
        row = checkpoint_health(path)
        assert row["finite"] is True
        assert row["n_nan_tensors"] == 0
        assert row["n_all_zero_tensors"] == 0
        assert row["optimizer_max_exp_avg"] == pytest.approx(0.2)
        other = {"w": torch.tensor([[0.9, -0.1]])}
        diff = state_dicts_max_abs_diff(good["model"], other)
        assert diff["n_equal_tensors"] == 0
        assert diff["max_abs_diff"] == pytest.approx(0.5)
    finally:
        shutil.rmtree(folder)
