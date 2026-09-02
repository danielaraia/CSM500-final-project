import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

from attention import (
    ATTENTION_ADDED_PARAMETERS,
    ATTENTION_HIDDEN,
    ATTENTION_METRIC,
    AttentionBackbone,
    learned_adjacency,
    make_graph_learner,
    prepare_attention_folder,
)
from config import LOCKED, read_config, run_folder_path, start_run
from pack import save_split


class _UsesAdj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(1, 1, bias=False)

    def forward(self, input_seq, seq_lengths, adj):
        pooled = adj.mean(dim=(1, 2, 3)).unsqueeze(-1)
        return self.w(pooled)


def test_learner_settings_match_the_lock():
    assert ATTENTION_HIDDEN == 32
    assert ATTENTION_METRIC == "self_attention"
    assert ATTENTION_ADDED_PARAMETERS == 8192
    learner = make_graph_learner()
    n_params = 0
    for parameter in learner.parameters():
        n_params += parameter.numel()
    assert n_params == ATTENTION_ADDED_PARAMETERS


def test_learned_adjacency_shape_and_row_sum():
    torch.manual_seed(123)
    learner = make_graph_learner()
    features = torch.randn(2, 4, LOCKED["num_nodes"], LOCKED["input_dim"])
    adj = learned_adjacency(learner, features)
    assert adj.shape == (2, 4, LOCKED["num_nodes"], LOCKED["num_nodes"])
    row_sums = adj.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)


def test_wrapper_ignores_packed_adjacency():
    torch.manual_seed(123)
    model = AttentionBackbone(_UsesAdj())
    batch = 3
    features = torch.randn(batch, 5, LOCKED["num_nodes"], LOCKED["input_dim"])
    lengths = torch.full((batch,), 5)
    packed_a = torch.zeros(batch, 5, LOCKED["num_nodes"], LOCKED["num_nodes"])
    packed_b = torch.ones(batch, 5, LOCKED["num_nodes"], LOCKED["num_nodes"])
    out_a = model(features, lengths, packed_a)
    out_b = model(features, lengths, packed_b)
    torch.testing.assert_close(out_a, out_b)
    assert not torch.equal(out_a, packed_a.mean(dim=(1, 2, 3)).unsqueeze(-1))


def test_learner_gets_a_gradient():
    torch.manual_seed(123)
    model = AttentionBackbone(_UsesAdj())
    features = torch.randn(2, 3, LOCKED["num_nodes"], LOCKED["input_dim"])
    lengths = torch.tensor([3, 3])
    packed = torch.zeros(2, 3, LOCKED["num_nodes"], LOCKED["num_nodes"])
    loss = model(features, lengths, packed).sum()
    loss.backward()
    found = False
    for parameter in model.learner.parameters():
        if parameter.grad is not None:
            found = True
            assert float(parameter.grad.abs().sum()) > 0.0
    assert found


def test_bad_feature_shape_fails():
    learner = make_graph_learner()
    with pytest.raises(ValueError, match="batch, time, nodes, bins"):
        learned_adjacency(learner, torch.randn(8, 128))


def test_attention_run_folder_is_not_pearson_or_mi():
    root = tempfile.mkdtemp()
    try:
        pearson_path = start_run(root, "sub-051", leaf="eval1")
        mi_path = start_run(
            root, "sub-051", extra={"graph_strategy": "mi"}, leaf="eval1"
        )
        attention_path = start_run(
            root, "sub-051", extra={"graph_strategy": "attention"}, leaf="eval1"
        )
        assert attention_path != pearson_path
        assert attention_path != mi_path
        assert attention_path == run_folder_path(
            root, "sub-051", "eval1", graph_strategy="attention"
        )
        loaded = read_config(os.path.join(attention_path, "config.json"))
        assert loaded["graph_strategy"] == "attention"
    finally:
        shutil.rmtree(root)


def test_prepare_attention_folder_links_the_pearson_splits():
    root = tempfile.mkdtemp()
    try:
        pearson = start_run(root, "sub-052", leaf="eval1")
        features = np.zeros((2, 60, 8, 128), dtype=np.float64)
        graphs = np.ones((2, 60, 8, 8), dtype=np.float64)
        labels = np.array([0, 1], dtype=np.int64)
        ends = np.array([70, 80], dtype=np.int64)
        for name in ("train", "monitor", "val", "test"):
            save_split(os.path.join(pearson, name + ".npz"), features, graphs, labels, ends)
        from metrics import write_metrics
        write_metrics(
            pearson,
            {
                "patient_id": "sub-052",
                "t1": 17929.0,
                "t2": 23108.0,
                "n_train_seizures": 1,
                "n_test_seizures": 1,
                "n_train_windows": 2,
                "n_monitor_windows": 2,
                "n_validation_windows": 2,
                "n_test_windows": 2,
                "packed_run": "07",
                "pack_scope": "seizure bearing run only",
            },
        )
        folder = prepare_attention_folder(root, "sub-052", pearson)
        assert folder == run_folder_path(
            root, "sub-052", "eval1", graph_strategy="attention"
        )
        loaded = read_config(os.path.join(folder, "config.json"))
        assert loaded["graph_strategy"] == "attention"
        assert loaded["attention_hidden"] == 32
        from metrics import read_metrics
        att_metrics = read_metrics(folder)
        assert att_metrics["t1"] == 17929.0
        assert att_metrics["t2"] == 23108.0
        assert att_metrics["n_test_seizures"] == 1
        train_src = np.load(os.path.join(pearson, "train.npz"))
        train_dst = np.load(os.path.join(folder, "train.npz"))
        np.testing.assert_array_equal(train_src["ends"], train_dst["ends"])
        np.testing.assert_array_equal(train_src["features"], train_dst["features"])
    finally:
        shutil.rmtree(root)
