import os
import shutil
import tempfile

import pytest

from metrics import load_gate_metrics, merge_metrics, read_metrics, write_metrics


def test_write_and_read_metrics():
    folder = tempfile.mkdtemp()
    try:
        write_metrics(folder, {"pos_weight": 7.0, "n_train_windows": 8})
        loaded = read_metrics(folder)
        assert loaded["pos_weight"] == 7.0
        assert loaded["n_train_windows"] == 8
        with pytest.raises(ValueError, match="wont write over"):
            write_metrics(folder, {"pos_weight": 1.0})
    finally:
        shutil.rmtree(folder)


def test_merge_metrics_keeps_pack_keys_and_saves_pack_command():
    folder = tempfile.mkdtemp()
    try:
        write_metrics(
            folder,
            {"command": "python pack_eval.py", "n_train_windows": 84, "pack_seconds": 1.5},
        )
        merge_metrics(
            folder,
            {"command": "python train_eval1.py", "train_seconds": 2.0, "best_monitor_bce": 1.1},
        )
        loaded = read_metrics(folder)
        assert loaded["pack_command"] == "python pack_eval.py"
        assert loaded["command"] == "python train_eval1.py"
        assert loaded["n_train_windows"] == 84
        assert loaded["pack_seconds"] == 1.5
        assert loaded["train_seconds"] == 2.0
        assert loaded["best_monitor_bce"] == 1.1
        with pytest.raises(ValueError, match="train metrics already exist"):
            merge_metrics(folder, {"train_seconds": 9.0})
    finally:
        shutil.rmtree(folder)


def test_missing_metrics_fails():
    folder = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="metrics file is missing"):
            read_metrics(folder)
    finally:
        shutil.rmtree(folder)


def test_missing_gate_metrics_fail():
    folder = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="p0 gate metrics file is missing"):
            load_gate_metrics(os.path.join(folder, "p0-gate-metrics.json"))
    finally:
        shutil.rmtree(folder)


def test_gate_metrics_missing_key_fails():
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    try:
        handle.write("{}\n")
        handle.close()
        with pytest.raises(ValueError, match="p0 gate metrics missing"):
            load_gate_metrics(handle.name)
    finally:
        os.remove(handle.name)
