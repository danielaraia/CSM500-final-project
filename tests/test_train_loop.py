import math
import os
import shutil
import tempfile

import pytest
import torch

from train_loop import monitor_bce, train_until_stop
from train_step import make_optimizer


class _TinyLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(1, 1, bias=True)

    def forward(self, input_seq, seq_lengths, adj):
        pooled = input_seq.mean(dim=(1, 2, 3)).unsqueeze(-1)
        return self.w(pooled)


def test_monitor_bce_is_finite_and_does_not_train():
    torch.manual_seed(123)
    model = _TinyLogits()
    features, adjacency, labels = _tiny_set(8)
    before = model.w.weight.detach().clone()
    loss = monitor_bce(model, features, adjacency, labels, 7.0, batch_size=8)
    after = model.w.weight.detach()
    assert math.isfinite(loss)
    assert torch.equal(before, after)


def test_zero_lr_stops_when_monitor_does_not_improve():
    torch.manual_seed(123)
    model = _TinyLogits()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    train = _tiny_set(8)
    monitor = _tiny_set(8)
    best_epoch, best_loss = train_until_stop(
        model,
        optimizer,
        train[0],
        train[1],
        train[2],
        monitor[0],
        monitor[1],
        monitor[2],
        pos_weight=7.0,
        max_epochs=20,
        patience=2,
        train_batch_size=8,
    )
    assert best_epoch == 1
    assert math.isfinite(best_loss)


def test_writes_last_and_best_checkpoints():
    torch.manual_seed(123)
    model = _TinyLogits()
    optimizer = make_optimizer(model)
    train = _tiny_set(8)
    monitor = _tiny_set(8)
    folder = tempfile.mkdtemp()
    try:
        train_until_stop(
            model,
            optimizer,
            train[0],
            train[1],
            train[2],
            monitor[0],
            monitor[1],
            monitor[2],
            pos_weight=7.0,
            max_epochs=1,
            patience=5,
            train_batch_size=8,
            checkpoint_dir=os.path.join(folder, "checkpoints"),
        )
        last = torch.load(
            os.path.join(folder, "checkpoints", "last.pth.tar"),
            weights_only=True,
        )
        best = torch.load(
            os.path.join(folder, "checkpoints", "best.pth.tar"),
            weights_only=True,
        )
        assert last["epoch"] == 1
        assert best["epoch"] == 1
        assert math.isfinite(best["monitor_loss"])
    finally:
        shutil.rmtree(folder)


def test_empty_train_fails():
    torch.manual_seed(123)
    model = _TinyLogits()
    optimizer = make_optimizer(model)
    empty_f = torch.zeros(0, 60, 8, 128)
    empty_a = torch.zeros(0, 60, 8, 8)
    empty_l = torch.zeros(0)
    monitor = _tiny_set(4)
    with pytest.raises(ValueError, match="train set is empty"):
        train_until_stop(
            model,
            optimizer,
            empty_f,
            empty_a,
            empty_l,
            monitor[0],
            monitor[1],
            monitor[2],
            pos_weight=7.0,
            max_epochs=1,
            patience=1,
            train_batch_size=8,
        )


def _tiny_set(n_windows):
    features = torch.randn(n_windows, 60, 8, 128)
    adjacency = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(n_windows, 60, 1, 1)
    labels = torch.zeros(n_windows)
    labels[-1] = 1.0
    return features, adjacency, labels
