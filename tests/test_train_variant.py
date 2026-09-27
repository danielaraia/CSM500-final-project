import os
import shutil
import tempfile

import pytest
import torch

from data.augment import make_augment_fn
from core.config import read_config, start_run
from model.focal_loss import make_loss_fn
from metrics import read_metrics, write_metrics
from packs.pack import load_split, save_split
from core.recipe import normalize_recipe, recipe_leaf
from run import train_packed_eval_run
from model.tiny_model import TinyLogits
from model.train_step import make_optimizer
from train.train_variant import command_string, parse_args
from packs.variant_pack import prepare_variant_folder


def test_parse_args_reads_the_lever_flags():
    patient_id, graph_strategy, sop_minutes, recipe, leaf = parse_args(
        ["train_variant.py", "sub-051", "mi", "15", "--loss", "focal"]
    )
    assert (patient_id, graph_strategy, sop_minutes) == ("sub-051", "mi", 15)
    assert recipe == {
        "loss": "focal",
        "augmentation": "none",
        "pretrain": "none",
        "sensors": "full",
    }
    assert leaf == "eval1-focal"
    _p, _g, _s, recipe, leaf = parse_args(
        [
            "train_variant.py", "sub-039", "pearson", "5",
            "--augment", "jitter_scale", "--pretrain", "cross_patient",
        ]
    )
    assert recipe == {
        "loss": "weighted_bce",
        "augmentation": "jitter_scale",
        "pretrain": "cross_patient",
        "sensors": "full",
    }
    assert leaf == "eval1-aug-pre"


def test_parse_args_rejects_the_all_default_recipe():
    with pytest.raises(ValueError, match="train_sop.py"):
        parse_args(["train_variant.py", "sub-051", "mi", "15"])
    with pytest.raises(ValueError, match="train_sop.py"):
        parse_args(
            [
                "train_variant.py", "sub-051", "mi", "15",
                "--loss", "weighted_bce", "--augment", "none", "--pretrain", "none",
            ]
        )


def test_parse_args_rejects_bad_input():
    with pytest.raises(ValueError, match="usage"):
        parse_args(["train_variant.py", "sub-051"])
    with pytest.raises(ValueError, match="sop_minutes has to be"):
        parse_args(["train_variant.py", "sub-051", "mi", "20", "--loss", "focal"])
    with pytest.raises(ValueError, match="whole number"):
        parse_args(["train_variant.py", "sub-051", "mi", "fifteen"])
    with pytest.raises(ValueError, match="unknown flag"):
        parse_args(["train_variant.py", "sub-051", "mi", "15", "--gamma", "2"])
    with pytest.raises(ValueError, match="needs a value"):
        parse_args(["train_variant.py", "sub-051", "mi", "15", "--loss"])
    with pytest.raises(ValueError, match="given twice"):
        parse_args(
            ["train_variant.py", "sub-051", "mi", "15",
             "--loss", "focal", "--loss", "focal"]
        )
    with pytest.raises(ValueError, match="loss has to be"):
        parse_args(["train_variant.py", "sub-051", "mi", "15", "--loss", "cosine"])


def test_command_string_records_every_lever():
    recipe = normalize_recipe(loss="focal")
    command = command_string("sub-051", "mi", 15, recipe)
    assert command == (
        "python train_variant.py sub-051 mi 15 "
        "--loss focal --augment none --pretrain none"
    )


def test_prepare_variant_folder_logs_the_recipe_identity():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        source = _write_source_pack(root, "sub-051")
        recipe = normalize_recipe(loss="focal", augmentation="jitter_scale")
        leaf = recipe_leaf(recipe)
        folder = prepare_variant_folder(
            root, "sub-051", source, recipe, leaf, "mi", 15,
            "python train_variant.py sub-051 mi 15 --loss focal --augment jitter_scale --pretrain none",
        )
        config = read_config(os.path.join(folder, "config.json"))
        assert config["run_leaf"] == "eval1-focal-aug"
        assert config["loss"] == "focal"
        assert config["augmentation"] == "jitter_scale"
        assert config["pretrain"] == "none"
        assert config["recipe"] == "focal+jitter_scale+none"
        assert config["variant_of"] == source
        assert config["graph_strategy"] == "mi"
        assert config["sop_minutes"] == 15
        assert config["seed"] == 123
        for name in ("train", "monitor", "val", "test"):
            src = load_split(os.path.join(source, name + ".npz"))
            dst = load_split(os.path.join(folder, name + ".npz"))
            assert (src[0] == dst[0]).all()
        metrics = read_metrics(folder)
        assert metrics["packed_run"] == "01"
        assert metrics["loss"] == "focal"
        # train side keys from the source run must not leak into the variant
        assert "train_seconds" not in metrics
        assert "best_epoch" not in metrics
        with pytest.raises(ValueError, match="wont write over"):
            prepare_variant_folder(
                root, "sub-051", source, recipe, leaf, "mi", 15, "again"
            )
    finally:
        shutil.rmtree(root)


def test_prepare_variant_folder_cleans_up_when_a_split_is_missing():
    root = tempfile.mkdtemp()
    try:
        source = _write_source_pack(root, "sub-051")
        os.remove(os.path.join(source, "test.npz"))
        recipe = normalize_recipe(pretrain="cross_patient")
        with pytest.raises(ValueError, match="source split is missing"):
            prepare_variant_folder(
                root, "sub-051", source, recipe, "eval1-pre", "mi", 15, "test"
            )
        assert not os.path.isdir(
            os.path.join(
                root, "experiments", "runs", "within_patient",
                "sub-051", "mi", "sop15", "seed123", "eval1-pre",
            )
        )
    finally:
        shutil.rmtree(root)


def test_variant_train_end_to_end_on_tiny_model():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        source = _write_source_pack(root, "sub-051")
        recipe = normalize_recipe(loss="focal", augmentation="jitter_scale")
        leaf = recipe_leaf(recipe)
        folder = prepare_variant_folder(
            root, "sub-051", source, recipe, leaf, "mi", 15, "test"
        )
        model = TinyLogits()
        extra = {"command": "test", "loss": "focal", "augmentation": "jitter_scale"}
        train_packed_eval_run(
            folder,
            model,
            make_optimizer(model),
            extra=extra,
            max_epochs=1,
            patience=5,
            train_batch_size=8,
            loss_fn=make_loss_fn("focal"),
            augment_fn=make_augment_fn("jitter_scale"),
        )
        metrics = read_metrics(folder)
        assert metrics["loss"] == "focal"
        assert metrics["augmentation"] == "jitter_scale"
        assert metrics["run_leaf"] == "eval1-focal-aug"
        assert metrics["n_epochs_run"] == 1
        assert os.path.isfile(os.path.join(folder, "test_scores.npz"))
        assert os.path.isfile(os.path.join(folder, "checkpoints", "best.pth.tar"))
    finally:
        shutil.rmtree(root)


def test_default_train_path_is_unchanged_by_the_new_arguments():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        shared = {
            name: _tiny_set(n_windows)
            for name, n_windows in (
                ("train", 8), ("monitor", 8), ("val", 4), ("test", 4)
            )
        }
        folder_a = _write_source_pack(root, "sub-051", shared)
        folder_b = _write_source_pack(root, "sub-052", shared)
        torch.manual_seed(123)
        model_a = TinyLogits()
        train_packed_eval_run(
            folder_a, model_a, make_optimizer(model_a),
            max_epochs=2, patience=5, train_batch_size=8,
            replace_existing_train=True,
        )
        torch.manual_seed(123)
        model_b = TinyLogits()
        train_packed_eval_run(
            folder_b, model_b, make_optimizer(model_b),
            max_epochs=2, patience=5, train_batch_size=8,
            replace_existing_train=True,
            loss_fn=None, augment_fn=None,
        )
        metrics_a = read_metrics(folder_a)
        metrics_b = read_metrics(folder_b)
        assert metrics_a["best_monitor_bce"] == pytest.approx(
            metrics_b["best_monitor_bce"]
        )
        assert metrics_a["n_epochs_run"] == metrics_b["n_epochs_run"]
        for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(param_a.detach(), param_b.detach())
    finally:
        shutil.rmtree(root)


def _write_source_pack(root, patient_id, splits=None):
    folder = start_run(
        root,
        patient_id,
        extra={"graph_strategy": "mi", "sop_minutes": 15},
        leaf="eval1",
    )
    if splits is None:
        splits = {
            name: _tiny_set(n_windows)
            for name, n_windows in (
                ("train", 8), ("monitor", 8), ("val", 4), ("test", 4)
            )
        }
    for name in ("train", "monitor", "val", "test"):
        _write_split(folder, name + ".npz", splits[name])
    write_metrics(
        folder,
        {
            "command": "python pack_sop.py",
            "packed_run": "01",
            "train_seconds": 9.5,
            "best_epoch": 3,
        },
    )
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
