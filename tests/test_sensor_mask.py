import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

from alarms import load_window_scores
from core.config import start_run
from metrics import write_metrics
from pack import load_split, save_split
from pearson import N_NODES, pearson_correlation
from core.recipe import normalize_recipe, recipe_extra, recipe_leaf
from score_variant import parse_args as score_parse_args
from score_variant import score_variant_folder
from data.sensor_mask import (
    NODE_NAMES,
    kept_nodes,
    make_sensor_mask_fn,
    mask_features,
)
from tiny_model import TinyLogits
from train_loop import score_windows
from train_step import make_optimizer
from train_variant import command_string, parse_args
from run import train_packed_eval_run


def _features(n_windows=3):
    rng = np.random.default_rng(123)
    return rng.normal(size=(n_windows, 60, N_NODES, 128))


def test_node_order_matches_the_pack_layout():
    assert NODE_NAMES == (
        "eeg_left",
        "eeg_right",
        "ecg",
        "emg",
        "neck_acc",
        "neck_gyr",
        "chest_acc",
        "chest_gyr",
    )
    assert len(NODE_NAMES) == N_NODES


def test_kept_nodes_per_tier():
    assert kept_nodes("full") == tuple(range(N_NODES))
    assert kept_nodes("headset") == (0, 1, 4, 5)
    assert kept_nodes("eeg_acc") == (0, 1, 4)
    assert kept_nodes("eeg_only") == (0, 1)
    with pytest.raises(ValueError, match="sensors has to be"):
        kept_nodes("ears")


def test_mask_zeroes_only_the_dropped_nodes():
    features = _features()
    cases = {
        "headset": (0, 1, 4, 5),
        "eeg_acc": (0, 1, 4),
        "eeg_only": (0, 1),
    }
    for tier, kept in cases.items():
        masked = mask_features(features, tier)
        assert masked.shape == features.shape
        assert masked.dtype == features.dtype
        for node in range(N_NODES):
            if node in kept:
                assert np.array_equal(
                    masked[:, :, node, :], features[:, :, node, :]
                )
            else:
                assert np.all(masked[:, :, node, :] == 0.0)


def test_mask_returns_a_copy_and_leaves_the_input_untouched():
    features = _features()
    before = features.copy()
    masked = mask_features(features, "eeg_only")
    assert masked is not features
    assert np.array_equal(features, before)


def test_full_is_an_exact_no_op():
    features = _features()
    assert mask_features(features, "full") is features
    assert make_sensor_mask_fn("full") is None


def test_mask_is_deterministic():
    features = _features()
    first = mask_features(features, "headset")
    second = mask_features(features, "headset")
    assert np.array_equal(first, second)
    fn = make_sensor_mask_fn("headset")
    assert np.array_equal(fn(features), first)
    with pytest.raises(ValueError, match="sensors has to be"):
        make_sensor_mask_fn("ears")


def test_mask_rejects_a_bad_node_axis():
    with pytest.raises(ValueError, match="nodes"):
        mask_features(np.zeros((2, 60, 7, 128)), "eeg_only")


def test_masked_series_would_trip_the_pearson_guard():
    # dropped nodes become constant zero series, the exact input pearson refuses.
    # edges stay valid because the pack adjacency is built from full signals at
    # pack time and masking never recomputes it
    features = _features()
    adjacency = np.tile(np.eye(N_NODES), (3, 60, 1, 1))
    before = adjacency.copy()
    masked = mask_features(features, "eeg_only")
    assert np.array_equal(adjacency, before)
    for node in (2, 3, 4, 5, 6, 7):
        series = masked[:, :, node, :].reshape(-1)
        with pytest.raises(ValueError, match="a series is constant"):
            pearson_correlation(series, series)


def test_masked_run_scores_match_manual_masking():
    # if edges were rebuilt from masked features the pearson guard would raise,
    # so a finished run proves the pack graphs went to the model untouched
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        folder = _write_eval1_pack(root)
        model = TinyLogits()
        train_packed_eval_run(
            folder,
            model,
            make_optimizer(model),
            max_epochs=1,
            patience=5,
            train_batch_size=8,
            sensor_mask_fn=make_sensor_mask_fn("eeg_only"),
        )
        _ends, saved = load_window_scores(os.path.join(folder, "test_scores.npz"))
        test_f, test_g, _y, _e = load_split(os.path.join(folder, "test.npz"))
        expected = score_windows(model, mask_features(test_f, "eeg_only"), test_g)
        np.testing.assert_allclose(saved, expected.numpy(), rtol=1e-6)
        unmasked = score_windows(model, test_f, test_g)
        assert not np.allclose(saved, unmasked.numpy())
    finally:
        shutil.rmtree(root)


def test_default_run_is_unchanged_by_the_mask_argument():
    torch.manual_seed(123)
    root = tempfile.mkdtemp()
    try:
        shared = {
            name: _tiny_set(n_windows)
            for name, n_windows in (
                ("train", 8), ("monitor", 8), ("val", 4), ("test", 4)
            )
        }
        folder_a = _write_eval1_pack(root, "sub-051", shared)
        folder_b = _write_eval1_pack(root, "sub-052", shared)
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
            sensor_mask_fn=make_sensor_mask_fn("full"),
        )
        scores_a = load_window_scores(os.path.join(folder_a, "test_scores.npz"))[1]
        scores_b = load_window_scores(os.path.join(folder_b, "test_scores.npz"))[1]
        np.testing.assert_array_equal(scores_a, scores_b)
        for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(param_a.detach(), param_b.detach())
    finally:
        shutil.rmtree(root)


def test_recipe_leaf_and_extra_carry_the_tier():
    recipe = normalize_recipe(sensors="eeg_acc")
    assert recipe_leaf(recipe) == "eval1-eeg_acc"
    extra = recipe_extra(recipe)
    assert extra["sensors"] == "eeg_acc"
    assert extra["recipe"] == "weighted_bce+none+none+eeg_acc"
    recipe = normalize_recipe(loss="focal", sensors="eeg_only")
    assert recipe_leaf(recipe) == "eval1-focal-eeg_only"


def test_full_sensors_adds_nothing_to_the_run_identity():
    recipe = normalize_recipe(loss="focal")
    extra = recipe_extra(recipe)
    assert "sensors" not in extra
    assert extra["recipe"] == "focal+none+none"
    assert recipe_leaf(recipe) == "eval1-focal"
    with pytest.raises(ValueError, match="train_sop.py"):
        recipe_leaf(normalize_recipe(sensors="full"))


def test_train_variant_parse_args_reads_the_sensors_flag():
    _p, _g, _s, recipe, leaf = parse_args(
        ["train_variant.py", "sub-051", "pearson", "30", "--sensors", "eeg_only"]
    )
    assert recipe["sensors"] == "eeg_only"
    assert leaf == "eval1-eeg_only"
    _p, _g, _s, recipe, leaf = parse_args(
        ["train_variant.py", "sub-051", "pearson", "30", "--loss", "focal"]
    )
    assert recipe["sensors"] == "full"
    assert leaf == "eval1-focal"
    with pytest.raises(ValueError, match="sensors has to be"):
        parse_args(
            ["train_variant.py", "sub-051", "pearson", "30", "--sensors", "ears"]
        )


def test_command_string_adds_sensors_only_when_masked():
    recipe = normalize_recipe(loss="focal")
    assert "--sensors" not in command_string("sub-051", "mi", 15, recipe)
    recipe = normalize_recipe(sensors="headset")
    command = command_string("sub-051", "mi", 15, recipe)
    assert command.endswith("--sensors headset")


def test_score_variant_parse_args():
    assert score_parse_args(
        ["score_variant.py", "sub-051", "pearson", "30", "eval1"]
    ) == ("sub-051", "pearson", 30, "eval1", "full")
    assert score_parse_args(
        [
            "score_variant.py", "sub-051", "pearson", "30", "eval1-eeg_only",
            "--sensors", "eeg_only",
        ]
    ) == ("sub-051", "pearson", 30, "eval1-eeg_only", "eeg_only")
    with pytest.raises(ValueError, match="sensors has to be"):
        score_parse_args(
            ["score_variant.py", "sub-051", "pearson", "30", "eval1",
             "--sensors", "ears"]
        )
    with pytest.raises(ValueError, match="unknown flag"):
        score_parse_args(
            ["score_variant.py", "sub-051", "pearson", "30", "eval1",
             "--tier", "eeg_only"]
        )
    with pytest.raises(ValueError, match="usage"):
        score_parse_args(["score_variant.py", "sub-051"])


def test_score_variant_rejects_a_tier_mismatch():
    root = tempfile.mkdtemp()
    try:
        folder = start_run(
            root,
            "sub-051",
            extra={"graph_strategy": "pearson", "sop_minutes": 30, "sensors": "eeg_only"},
            leaf="eval1-eeg_only",
        )
        np.savez_compressed(os.path.join(folder, "test_scores.npz"), ends=[], logits=[])
        with pytest.raises(ValueError, match="config sensors do not match"):
            score_variant_folder(root, "sub-051", "pearson", 30, "eval1-eeg_only")
        with pytest.raises(ValueError, match="config sensors do not match"):
            score_variant_folder(
                root, "sub-051", "pearson", 30, "eval1-eeg_only", sensors="headset"
            )
    finally:
        shutil.rmtree(root)


def _write_eval1_pack(root, patient_id="sub-051", splits=None):
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
        features, adjacency, labels = splits[name]
        save_split(
            os.path.join(folder, name + ".npz"),
            features.detach().numpy(),
            adjacency.detach().numpy(),
            labels.detach().numpy(),
            ends=list(range(len(labels))),
        )
    write_metrics(folder, {"command": "python pack_sop.py", "packed_run": "01"})
    return folder


def _tiny_set(n_windows):
    torch.manual_seed(123)
    features = torch.randn(n_windows, 60, 8, 128)
    adjacency = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(n_windows, 60, 1, 1)
    labels = torch.zeros(n_windows)
    labels[-1] = 1.0
    return features, adjacency, labels
