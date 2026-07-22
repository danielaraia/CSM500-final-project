import os
import tempfile

from config import LOCKED, read_config, write_config


def test_locked_learning_rate_is_not_the_code_default():
    assert LOCKED["learning_rate"] == 0.0001
    assert LOCKED["max_seq_len"] == 60
    assert LOCKED["data_augment"] is False
    assert LOCKED["train_batch_size"] == 32


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
