import os
import shutil
import tempfile

import numpy as np
import pytest

from data.features import N_FFT_BINS
from data.labels import LABEL_INTERICTAL, LABEL_SOP
from packs.pack import save_split
from packs.pack_monitor import (
    cut_monitoring_slice_at_clock,
    monitor_clock_from_train_hop10,
    rewrite_monitor_eval,
)
from graphs.pearson import N_NODES
from data.windows import WINDOW_SECONDS


def test_monitor_clock_cut_uses_hop_ten_quarter():
    ends = [59, 69, 79, 89, 90, 91, 92]
    n_windows = len(ends)
    features = np.zeros((n_windows, WINDOW_SECONDS, N_NODES, N_FFT_BINS))
    graphs = np.zeros((n_windows, WINDOW_SECONDS, N_NODES, N_NODES))
    labels = [LABEL_INTERICTAL] * 3 + [LABEL_SOP] * 2 + [LABEL_INTERICTAL] * 2
    clock = monitor_clock_from_train_hop10(ends)
    assert clock == 89
    train, monitor, start, n_positive, n_negative = cut_monitoring_slice_at_clock(
        features, graphs, ends, labels, clock
    )
    assert start == 89
    assert train[2] == [59, 69, 79]
    assert monitor[2] == [89, 90, 91, 92]
    assert n_positive == 2
    assert n_negative == 2


def test_rewrite_monitor_fails_when_folder_is_missing():
    with pytest.raises(ValueError, match="run folder is missing"):
        rewrite_monitor_eval(
            "/tmp/no-such-eval1-folder-please",
            "missing.edf",
            "missing.edf",
            "missing.edf",
            "missing.edf",
            "missing.tsv",
        )


def test_rewrite_monitor_fails_when_train_pack_is_missing():
    folder = tempfile.mkdtemp()
    try:
        with open(os.path.join(folder, "metrics.json"), "w") as handle:
            handle.write('{"t1": 100.0}\n')
        with pytest.raises(ValueError, match="train pack is missing"):
            rewrite_monitor_eval(
                folder,
                "missing.edf",
                "missing.edf",
                "missing.edf",
                "missing.edf",
                "missing.tsv",
            )
    finally:
        shutil.rmtree(folder)


def test_rewrite_monitor_fails_when_train_pack_has_no_ends():
    folder = tempfile.mkdtemp()
    try:
        with open(os.path.join(folder, "metrics.json"), "w") as handle:
            handle.write('{"t1": 100.0}\n')
        features = np.zeros((1, 1, 1, 1))
        graphs = np.zeros((1, 1, 1, 1))
        labels = np.zeros((1,), dtype=np.int64)
        save_split(os.path.join(folder, "train.npz"), features, graphs, labels)
        save_split(
            os.path.join(folder, "monitor.npz"),
            features,
            graphs,
            labels,
            ends=[100],
        )
        with pytest.raises(ValueError, match="train pack has no stored ends"):
            rewrite_monitor_eval(
                folder,
                "missing.edf",
                "missing.edf",
                "missing.edf",
                "missing.edf",
                "missing.tsv",
            )
    finally:
        shutil.rmtree(folder)
