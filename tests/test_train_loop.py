import math
import os
import shutil
import tempfile
import time

import pytest
import torch

from config import start_run
from metrics import read_metrics, write_metrics
from pack import save_split
from run import train_packed_eval_run, train_packed_run, train_score_and_write_metrics
from tiny_model import TinyLogits
from train_loop import monitor_bce, score_windows, train_and_write_metrics, train_until_stop
from train_step import make_optimizer


def test_score_windows_counts_and_does_not_train():
    torch.manual_seed(123)
    model = TinyLogits()
    features, adjacency, _labels = _tiny_set(8)
    before = model.w.weight.detach().clone()
    n_scored = score_windows(model, features, adjacency, batch_size=8)
    after = model.w.weight.detach()
    assert n_scored == 8
    assert torch.equal(before, after)
    empty_f = torch.zeros(0, 60, 8, 128)
    empty_a = torch.zeros(0, 60, 8, 8)
    assert score_windows(model, empty_f, empty_a, batch_size=8) == 0


def test_train_score_and_write_metrics_has_wall_clock_keys():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        folder = start_run(root, "sub-004")
        _write_split(folder, "train.npz", _tiny_set(8))
        _write_split(folder, "monitor.npz", _tiny_set(8))
        _write_empty_split(folder, "val.npz")
        _write_split(folder, "test.npz", _tiny_set(4))
        model = TinyLogits()
        clock_start = time.perf_counter()
        _folder, metrics = train_score_and_write_metrics(
            folder,
            model,
            make_optimizer(model),
            extra={"n_steps_dropped_qc": 2, "n_steps_dropped_short_buffer": 3},
            clock_start=clock_start,
            preprocess_seconds=0.01,
            max_epochs=1,
            patience=5,
            train_batch_size=8,
            compile_seconds=1.5,
        )
        loaded = read_metrics(folder)
        assert loaded["n_seconds_scored"] == 4
        assert loaded["n_validation_windows_scored"] == 0
        assert loaded["n_test_windows_scored"] == 4
        assert loaded["preprocess_seconds"] == pytest.approx(0.01)
        assert loaded["train_seconds"] > 0
        assert loaded["score_seconds"] >= 0
        assert loaded["end_to_end_wall_clock"] > 0
        assert loaded["compile_seconds"] == pytest.approx(1.5)
        assert loaded["n_steps_dropped_qc"] == 2
        assert loaded["n_steps_dropped_short_buffer"] == 3
        assert loaded["n_positive_monitor"] == 1
        assert loaded["n_negative_monitor"] == 7
        assert loaded["patient_wall_clock_seconds"] == loaded["end_to_end_wall_clock"]
        assert loaded["graph_strategy"] == "pearson"
        assert loaded["sop_minutes"] == 30
        assert loaded["negative_control"] == "none"
        assert "auroc" not in loaded
        assert math.isfinite(loaded["best_monitor_bce"])
    finally:
        shutil.rmtree(root)


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


def test_train_packed_eval_run_merges_into_pack_metrics():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        folder = start_run(root, "sub-004", extra={"command": "python pack_eval.py"}, leaf="eval1")
        _write_split(folder, "train.npz", _tiny_set(8), ends=True)
        _write_split(folder, "monitor.npz", _tiny_set(8), ends=True)
        _write_empty_split(folder, "val.npz", ends=True)
        _write_split(folder, "test.npz", _tiny_set(4), ends=True)
        write_metrics(
            folder,
            {
                "command": "python pack_eval.py",
                "pack_seconds": 1.25,
                "n_test_windows": 4,
            },
        )
        model = TinyLogits()
        train_packed_eval_run(
            folder,
            model,
            make_optimizer(model),
            extra={"train_model": "laptop stand in", "command": "python train_eval1.py"},
            max_epochs=1,
            patience=5,
            train_batch_size=8,
        )
        loaded = read_metrics(folder)
        assert loaded["pack_command"] == "python pack_eval.py"
        assert loaded["pack_seconds"] == 1.25
        assert loaded["used_stored_ends"] is True
        assert loaded["scored_at_train_stride"] is False
        assert loaded["n_test_windows_scored"] == 4
        assert loaded["n_validation_windows_scored"] == 0
        assert loaded["n_test_ends"] == 4
        assert loaded["train_model"] == "laptop stand in"
        assert loaded["run_leaf"] == "eval1"
        assert math.isfinite(loaded["best_monitor_bce"])
        assert loaded["train_seconds"] > 0
        assert "auroc" not in loaded
        with pytest.raises(ValueError, match="train metrics already exist"):
            train_packed_eval_run(
                folder,
                TinyLogits(),
                make_optimizer(TinyLogits()),
                max_epochs=1,
                patience=1,
                train_batch_size=8,
            )
    finally:
        shutil.rmtree(root)


def test_train_packed_eval_run_fails_without_stored_ends():
    root = tempfile.mkdtemp()
    try:
        folder = start_run(root, "sub-004", leaf="eval1")
        _write_split(folder, "train.npz", _tiny_set(8))
        _write_split(folder, "monitor.npz", _tiny_set(8))
        _write_empty_split(folder, "val.npz")
        _write_split(folder, "test.npz", _tiny_set(4))
        write_metrics(folder, {"command": "python pack_eval.py"})
        model = TinyLogits()
        with pytest.raises(ValueError, match="no stored ends"):
            train_packed_eval_run(
                folder,
                model,
                make_optimizer(model),
                max_epochs=1,
                patience=1,
                train_batch_size=8,
            )
    finally:
        shutil.rmtree(root)


def test_train_packed_eval_run_fails_when_train_has_no_sop():
    root = tempfile.mkdtemp()
    try:
        folder = start_run(root, "sub-004", leaf="eval1")
        empty_pos = _tiny_set(8)
        empty_pos[2][:] = 0
        _write_split(folder, "train.npz", empty_pos, ends=True)
        _write_split(folder, "monitor.npz", _tiny_set(8), ends=True)
        _write_empty_split(folder, "val.npz", ends=True)
        _write_split(folder, "test.npz", _tiny_set(4), ends=True)
        write_metrics(folder, {"command": "python pack_eval.py"})
        model = TinyLogits()
        with pytest.raises(ValueError, match="gradient train has no sop"):
            train_packed_eval_run(
                folder,
                model,
                make_optimizer(model),
                max_epochs=1,
                patience=1,
                train_batch_size=8,
            )
    finally:
        shutil.rmtree(root)


def test_train_packed_eval_run_rejects_none_leaf():
    root = tempfile.mkdtemp()
    try:
        folder = start_run(root, "sub-004", leaf="none")
        model = TinyLogits()
        with pytest.raises(ValueError, match="only loads eval1 packs"):
            train_packed_eval_run(
                folder,
                model,
                make_optimizer(model),
                max_epochs=1,
                patience=1,
            )
    finally:
        shutil.rmtree(root)


def _write_split(folder, name, tensors, ends=False):
    features, adjacency, labels = tensors
    end_times = None
    if ends:
        end_times = list(range(len(labels)))
    save_split(
        os.path.join(folder, name),
        features.detach().numpy(),
        adjacency.detach().numpy(),
        labels.detach().numpy(),
        ends=end_times,
    )


def _write_empty_split(folder, name, ends=False):
    end_times = None
    if ends:
        end_times = []
    save_split(
        os.path.join(folder, name),
        torch.zeros(0, 60, 8, 128).numpy(),
        torch.zeros(0, 60, 8, 8).numpy(),
        torch.zeros(0).numpy(),
        ends=end_times,
    )


def _tiny_set(n_windows):
    features = torch.randn(n_windows, 60, 8, 128)
    adjacency = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(n_windows, 60, 1, 1)
    labels = torch.zeros(n_windows)
    labels[-1] = 1.0
    return features, adjacency, labels
