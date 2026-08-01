import math
import os
import shutil
import tempfile

import pytest
import torch

from config import start_run
from metrics import read_metrics
from pack import save_split
from run import train_packed_run
from tiny_model import TinyLogits
from train_loop import monitor_bce, train_and_write_metrics, train_until_stop
from train_step import make_optimizer


def test_monitor_bce_is_finite_and_does_not_train():
    torch.manual_seed(123)
    model = TinyLogits()
    features, adjacency, labels = _tiny_set(8)
    before = model.w.weight.detach().clone()
    loss = monitor_bce(model, features, adjacency, labels, 7.0, batch_size=8)
    after = model.w.weight.detach()
    assert math.isfinite(loss)
    assert torch.equal(before, after)


def test_zero_lr_stops_when_monitor_does_not_improve():
    torch.manual_seed(123)
    model = TinyLogits()
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
    model = TinyLogits()
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


def test_writes_metrics_next_to_checkpoints():
    torch.manual_seed(123)
    model = TinyLogits()
    optimizer = make_optimizer(model)
    train = _tiny_set(8)
    monitor = _tiny_set(8)
    folder = tempfile.mkdtemp()
    try:
        train_and_write_metrics(
            folder,
            model,
            optimizer,
            train[0],
            train[1],
            train[2],
            monitor[0],
            monitor[1],
            monitor[2],
            extra={"patient_id": "sub-004"},
            max_epochs=1,
            patience=5,
            train_batch_size=8,
        )
        loaded = read_metrics(folder)
        assert loaded["patient_id"] == "sub-004"
        assert loaded["n_train_windows"] == 8
        assert loaded["n_monitor_windows"] == 8
        assert loaded["n_positive_train"] == 1
        assert loaded["n_negative_train"] == 7
        assert loaded["pos_weight"] == pytest.approx(7.0)
        assert loaded["negative_control"] == "none"
        assert math.isfinite(loaded["best_monitor_bce"])
        assert loaded["parameter_count"] > 0
        assert os.path.isfile(os.path.join(folder, "checkpoints", "best.pth.tar"))
    finally:
        shutil.rmtree(folder)


def test_empty_train_fails():
    torch.manual_seed(123)
    model = TinyLogits()
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


def test_train_packed_run_writes_metrics_next_to_config():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        folder = start_run(root, "sub-004")
        _write_split(folder, "train.npz", _tiny_set(8))
        _write_split(folder, "monitor.npz", _tiny_set(8))
        _write_empty_split(folder, "val.npz")
        _write_split(folder, "test.npz", _tiny_set(4))
        model = TinyLogits()
        train_packed_run(
            folder,
            model,
            make_optimizer(model),
            extra={"train_model": "laptop stand in"},
            max_epochs=1,
            patience=5,
            train_batch_size=8,
        )
        loaded = read_metrics(folder)
        assert loaded["n_train_windows"] == 8
        assert loaded["n_monitor_windows"] == 8
        assert loaded["n_validation_windows"] == 0
        assert loaded["n_test_windows"] == 4
        assert loaded["n_positive_train"] == 1
        assert loaded["n_negative_train"] == 7
        assert loaded["pos_weight"] == pytest.approx(7.0)
        assert loaded["train_model"] == "laptop stand in"
        assert loaded["negative_control"] == "none"
        assert math.isfinite(loaded["best_monitor_bce"])
        assert os.path.isfile(os.path.join(folder, "config.json"))
        assert os.path.isfile(os.path.join(folder, "checkpoints", "best.pth.tar"))
    finally:
        shutil.rmtree(root)


def test_train_packed_run_fails_when_train_split_is_missing():
    root = tempfile.mkdtemp()
    try:
        folder = start_run(root, "sub-004")
        model = TinyLogits()
        with pytest.raises(ValueError, match="split file is missing"):
            train_packed_run(
                folder,
                model,
                make_optimizer(model),
                max_epochs=1,
                patience=1,
            )
    finally:
        shutil.rmtree(root)


def _write_split(folder, name, tensors):
    features, adjacency, labels = tensors
    save_split(
        os.path.join(folder, name),
        features.detach().numpy(),
        adjacency.detach().numpy(),
        labels.detach().numpy(),
    )


def _write_empty_split(folder, name):
    save_split(
        os.path.join(folder, name),
        torch.zeros(0, 60, 8, 128).numpy(),
        torch.zeros(0, 60, 8, 8).numpy(),
        torch.zeros(0).numpy(),
    )


def _tiny_set(n_windows):
    features = torch.randn(n_windows, 60, 8, 128)
    adjacency = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(n_windows, 60, 1, 1)
    labels = torch.zeros(n_windows)
    labels[-1] = 1.0
    return features, adjacency, labels
