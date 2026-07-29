import os
import shutil
import tempfile

import pytest

from metrics import read_metrics, write_metrics


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


def test_missing_metrics_fails():
    folder = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="metrics file is missing"):
            read_metrics(folder)
    finally:
        shutil.rmtree(folder)
