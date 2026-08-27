import os
import shutil
import tempfile

import pytest

from config import LOCKED, read_config, run_folder_path, start_run, write_config


def test_locked_learning_rate_is_not_the_code_default():
    assert LOCKED["learning_rate"] == 0.0001
    assert LOCKED["max_seq_len"] == 60
    assert LOCKED["data_augment"] is False
    assert LOCKED["train_batch_size"] == 32
    assert LOCKED["num_eigenvectors"] == 4
    assert LOCKED["rnn_units"] == 64
    assert LOCKED["eeg_rms_rule"] == "drop a second only if both ears fail 13 to 150 uV"
    assert LOCKED["filter_call"] == "scipy.signal.butter SOS then sosfiltfilt"
    assert LOCKED["early_stopping_direction"] == "minimise"
    assert LOCKED["train_stride_seconds"] == 10
    assert LOCKED["eval_stride_seconds"] == 1


def test_write_and_read_config():
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    handle.close()
    try:
        write_config(handle.name, {"patient_id": "sub-004"})
        loaded = read_config(handle.name)
        assert loaded["patient_id"] == "sub-004"
        assert loaded["seed"] == 123
    finally:
        os.remove(handle.name)


def test_write_config_rejects_locked_overwrite():
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    handle.close()
    try:
        with pytest.raises(ValueError, match="locked config key"):
            write_config(handle.name, {"learning_rate": 0.0003})
    finally:
        os.remove(handle.name)


def test_start_run_writes_locked_config_and_wont_overwrite():
    root = tempfile.mkdtemp()
    try:
        path = start_run(root, "sub-004")
        loaded = read_config(os.path.join(path, "config.json"))
        assert loaded["patient_id"] == "sub-004"
        assert loaded["learning_rate"] == 0.0001
        assert loaded["seed"] == 123
        with pytest.raises(ValueError, match="wont write over"):
            start_run(root, "sub-004")
    finally:
        shutil.rmtree(root)


def test_start_run_leaf_eval1_is_a_new_folder():
    root = tempfile.mkdtemp()
    try:
        none_path = start_run(root, "sub-004")
        eval_path = start_run(root, "sub-004", leaf="eval1")
        assert none_path != eval_path
        assert eval_path == run_folder_path(root, "sub-004", "eval1")
        loaded = read_config(os.path.join(eval_path, "config.json"))
        assert loaded["run_leaf"] == "eval1"
        assert loaded["train_stride_seconds"] == 10
        assert loaded["eval_stride_seconds"] == 1
    finally:
        shutil.rmtree(root)


def test_mi_run_folder_is_not_pearson():
    root = tempfile.mkdtemp()
    try:
        pearson_path = start_run(root, "sub-051", leaf="eval1")
        mi_path = start_run(
            root, "sub-051", extra={"graph_strategy": "mi"}, leaf="eval1"
        )
        assert pearson_path != mi_path
        assert mi_path == run_folder_path(
            root, "sub-051", "eval1", graph_strategy="mi"
        )
        loaded = read_config(os.path.join(mi_path, "config.json"))
        assert loaded["graph_strategy"] == "mi"
        assert loaded["run_leaf"] == "eval1"
    finally:
        shutil.rmtree(root)


def test_write_config_rejects_unknown_graph_strategy():
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    handle.close()
    try:
        with pytest.raises(ValueError, match="pearson or mi"):
            write_config(handle.name, {"graph_strategy": "attention"})
    finally:
        os.remove(handle.name)
