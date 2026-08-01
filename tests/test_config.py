import os
import shutil
import tempfile

import pytest

from config import LOCKED, read_config, start_run, write_config


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
