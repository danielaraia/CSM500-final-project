import os
import shutil
import tempfile

import pytest
import torch

from core.config import LOCKED, start_run
from metrics import read_metrics, write_metrics
from packs.pack import save_split
from model.pretrain import (
    find_pretrain_folders,
    load_pretrain_weights,
    pool_splits,
    pretrain_model,
)
from run import train_packed_eval_run
from model.tiny_model import TinyLogits


def test_pool_excludes_the_target_patient():
    root = tempfile.mkdtemp()
    try:
        _write_pack(root, "sub-001")
        _write_pack(root, "sub-002")
        _write_pack(root, "sub-003")
        folders = find_pretrain_folders(root, "sub-001", "pearson", 15)
        assert len(folders) == 2
        for folder in folders:
            assert "sub-001" not in folder
        info = pretrain_model(
            root,
            TinyLogits(),
            "sub-001",
            "pearson",
            15,
            os.path.join(root, "pretrain", "checkpoints"),
            max_epochs=1,
            patience=1,
            train_batch_size=8,
        )
        assert info["pretrain_pool"] == ["sub-002", "sub-003"]
        assert info["pretrain_pool_size"] == 2
        assert info["pretrain_n_train_windows"] == 16
        assert info["pretrain_n_monitor_windows"] == 16
    finally:
        shutil.rmtree(root)


def test_pool_is_empty_when_only_the_target_is_packed():
    root = tempfile.mkdtemp()
    try:
        _write_pack(root, "sub-001")
        assert find_pretrain_folders(root, "sub-001", "pearson", 15) == []
        with pytest.raises(ValueError, match="pretrain pool is empty"):
            pool_splits([])
        with pytest.raises(ValueError, match="pretrain pool is empty"):
            pretrain_model(
                root,
                TinyLogits(),
                "sub-001",
                "pearson",
                15,
                os.path.join(root, "pretrain", "checkpoints"),
                max_epochs=1,
                patience=1,
                train_batch_size=8,
            )
    finally:
        shutil.rmtree(root)


def test_pool_follows_the_sop_and_graph_of_the_run():
    root = tempfile.mkdtemp()
    try:
        _write_pack(root, "sub-001", sop_minutes=15)
        _write_pack(root, "sub-002", sop_minutes=30)
        _write_pack(root, "sub-003", sop_minutes=15, graph_strategy="mi")
        assert find_pretrain_folders(root, "sub-001", "pearson", 15) == []
        assert len(find_pretrain_folders(root, "sub-001", "pearson", 30)) == 1
        assert len(find_pretrain_folders(root, "sub-001", "mi", 15)) == 1
    finally:
        shutil.rmtree(root)


def test_attention_pretrain_pools_the_pearson_packs():
    root = tempfile.mkdtemp()
    try:
        _write_pack(root, "sub-001")
        _write_pack(root, "sub-002")
        folders = find_pretrain_folders(root, "sub-001", "attention", 15)
        assert len(folders) == 1
        assert "pearson" in folder_path_parts(folders[0])
    finally:
        shutil.rmtree(root)


def test_pretrain_checkpoint_round_trip():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        _write_pack(root, "sub-002")
        _write_pack(root, "sub-003")
        checkpoint_dir = os.path.join(root, "pretrain", "checkpoints")
        model = TinyLogits()
        info = pretrain_model(
            root,
            model,
            "sub-001",
            "pearson",
            15,
            checkpoint_dir,
            max_epochs=1,
            patience=1,
            train_batch_size=8,
        )
        torch.manual_seed(999)
        reloaded = TinyLogits()
        epoch = load_pretrain_weights(reloaded, checkpoint_dir)
        assert epoch == info["pretrain_best_epoch"]
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter.detach(), reloaded.state_dict()[name].detach())
        with pytest.raises(ValueError, match="pretrain checkpoint is missing"):
            load_pretrain_weights(reloaded, os.path.join(root, "nowhere"))
    finally:
        shutil.rmtree(root)


def test_fine_tune_after_pretrain_keeps_the_locked_monitor_logic():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        _write_pack(root, "sub-002")
        model = TinyLogits()
        pretrain_model(
            root,
            model,
            "sub-001",
            "pearson",
            15,
            os.path.join(root, "pretrain", "checkpoints"),
            max_epochs=1,
            patience=1,
            train_batch_size=8,
        )
        folder = start_run(
            root,
            "sub-001",
            extra={"graph_strategy": "pearson", "sop_minutes": 15},
            leaf="eval1-pre",
        )
        _write_split(folder, "train.npz", _tiny_set(8))
        _write_split(folder, "monitor.npz", _tiny_set(8))
        _write_split(folder, "val.npz", _tiny_set(4))
        _write_split(folder, "test.npz", _tiny_set(4))
        write_metrics(folder, {"command": "test"})
        # zero lr fine tune: the default monitor logic stops at patience + 1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        train_packed_eval_run(
            folder,
            model,
            optimizer,
            max_epochs=20,
            patience=2,
            train_batch_size=8,
        )
        metrics = read_metrics(folder)
        assert metrics["n_epochs_run"] == 3
        assert metrics["best_epoch"] == 1
        assert metrics["early_stopping_metric"] == LOCKED["early_stopping_metric"]
        assert metrics["run_leaf"] == "eval1-pre"
    finally:
        shutil.rmtree(root)


def folder_path_parts(folder):
    return folder.replace(os.sep, "/").split("/")


def _write_pack(root, patient_id, n_windows=8, sop_minutes=15, graph_strategy="pearson"):
    extra = {"graph_strategy": graph_strategy, "sop_minutes": sop_minutes}
    folder = start_run(root, patient_id, extra=extra, leaf="eval1")
    features, adjacency, labels = _tiny_set(n_windows)
    _write_split(folder, "train.npz", (features, adjacency, labels))
    _write_split(folder, "monitor.npz", (features, adjacency, labels))
    return folder


def _write_split(folder, name, tensors):
    features, adjacency, labels = tensors
    save_split(
        os.path.join(folder, name),
        features.detach().numpy(),
        adjacency.detach().numpy(),
        labels.detach().numpy(),
        ends=list(range(len(labels))),
    )


def _tiny_set(n_windows):
    features = torch.randn(n_windows, 60, 8, 128)
    adjacency = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(n_windows, 60, 1, 1)
    labels = torch.zeros(n_windows)
    labels[-1] = 1.0
    return features, adjacency, labels
