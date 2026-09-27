import json
import math
import os
import shutil
import tempfile

import torch

from core.config import start_run
from evaluation.epoch_trace import (
    class_counts,
    install_logged_train_loop,
    logit_zero_weighted_bce,
    train_until_stop_logged,
)
from packs.pack import save_split
from run import train_score_and_write_metrics
from model.tiny_model import TinyLogits
from model.train_step import make_optimizer


def test_logit_zero_matches_the_old_monitor_check():
    loss = logit_zero_weighted_bce(180, 151, 32.13333333333333)
    assert loss == math.log(2.0) * (32.13333333333333 * 180 + 151) / 331.0


def test_zero_lr_logs_patience_plus_one_epochs():
    torch.manual_seed(123)
    model = TinyLogits()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    train = _tiny_set(8)
    monitor = _tiny_set(8)
    folder = tempfile.mkdtemp()
    try:
        best_epoch, best_loss, n_epochs_run = train_until_stop_logged(
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
            checkpoint_dir=os.path.join(folder, "checkpoints"),
        )
        assert best_epoch == 1
        assert n_epochs_run == 3
        assert math.isfinite(best_loss)
        path = os.path.join(folder, "checkpoints", "epoch_log.json")
        with open(path) as handle:
            blob = json.load(handle)
        assert blob["n_epochs_logged"] == 3
        assert len(blob["epochs"]) == 3
        assert blob["epochs"][0]["epoch"] == 1
        assert blob["epochs"][2]["epoch"] == 3
        assert blob["epochs"][0]["best_epoch"] == 1
        assert blob["epochs"][2]["best_epoch"] == 1
        assert blob["epochs"][2]["wait"] == 2
        index = 0
        while index < len(blob["epochs"]):
            row = blob["epochs"][index]
            assert math.isfinite(row["train_bce"])
            assert math.isfinite(row["monitor_bce"])
            index += 1
        n_pos, n_neg = class_counts(monitor[2])
        assert blob["n_positive_monitor"] == n_pos
        assert blob["n_negative_monitor"] == n_neg
        assert blob["logit_zero_monitor_bce"] == logit_zero_weighted_bce(
            n_pos, n_neg, 7.0
        )
    finally:
        shutil.rmtree(folder)


def test_packed_train_uses_the_logged_loop_after_install():
    torch.manual_seed(123)
    import run

    original = run.train_until_stop
    root = tempfile.mkdtemp()
    try:
        install_logged_train_loop()
        folder = start_run(root, "sub-004")
        _write_split(folder, "train.npz", _tiny_set(8))
        _write_split(folder, "monitor.npz", _tiny_set(8))
        _write_empty_split(folder, "val.npz")
        _write_split(folder, "test.npz", _tiny_set(4))
        model = TinyLogits()
        train_score_and_write_metrics(
            folder,
            model,
            make_optimizer(model),
            extra={"patient_id": "sub-004"},
            clock_start=0.0,
            preprocess_seconds=0.0,
            max_epochs=1,
            patience=5,
            train_batch_size=8,
        )
        path = os.path.join(folder, "checkpoints", "epoch_log.json")
        assert os.path.isfile(path)
        with open(path) as handle:
            blob = json.load(handle)
        assert blob["n_epochs_logged"] == 1
    finally:
        run.train_until_stop = original
        shutil.rmtree(root)


def test_logged_loop_writes_best_checkpoint():
    torch.manual_seed(123)
    model = TinyLogits()
    optimizer = make_optimizer(model)
    train = _tiny_set(8)
    monitor = _tiny_set(8)
    folder = tempfile.mkdtemp()
    try:
        train_until_stop_logged(
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
        best = torch.load(
            os.path.join(folder, "checkpoints", "best.pth.tar"),
            weights_only=True,
        )
        assert best["epoch"] == 1
        assert math.isfinite(best["monitor_loss"])
    finally:
        shutil.rmtree(folder)


def _write_split(folder, name, tensors):
    features, adjacency, labels = tensors
    save_split(
        os.path.join(folder, name),
        features.detach().numpy(),
        adjacency.detach().numpy(),
        labels.detach().numpy(),
        ends=list(range(len(labels))),
    )


def _write_empty_split(folder, name):
    save_split(
        os.path.join(folder, name),
        torch.zeros(0, 60, 8, 128).numpy(),
        torch.zeros(0, 60, 8, 8).numpy(),
        torch.zeros(0).numpy(),
        ends=[],
    )


def _tiny_set(n_windows):
    features = torch.randn(n_windows, 60, 8, 128)
    adjacency = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(n_windows, 60, 1, 1)
    labels = torch.zeros(n_windows)
    labels[-1] = 1.0
    return features, adjacency, labels
