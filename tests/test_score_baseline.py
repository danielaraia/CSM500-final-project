import os
import shutil
import tempfile

import numpy as np
import pytest

from packs.pack import save_split
from evaluation.report_baseline import plain_verdict
from evaluation.score_baseline import evo_from_metrics, load_strength_clock


def test_load_strength_clock_reads_ends_labels_and_node_means():
    folder = tempfile.mkdtemp()
    try:
        path = os.path.join(folder, "train.npz")
        features = np.zeros((3, 60, 8, 128))
        features[0, :, 0, :] = 2.0
        features[1, :, 1, :] = 4.0
        save_split(
            path,
            features,
            np.zeros((3, 60, 8, 8)),
            np.asarray([0, 1, 0]),
            ends=[10, 20, 30],
        )
        loaded = load_strength_clock(path)
        assert list(loaded["ends"]) == [10, 20, 30]
        assert list(loaded["labels"]) == [0, 1, 0]
        assert loaded["strength"].shape == (3, 8)
        assert loaded["strength"][0, 0] == pytest.approx(2.0)
        assert loaded["strength"][1, 1] == pytest.approx(4.0)
    finally:
        shutil.rmtree(folder)


def test_load_strength_clock_fails_without_ends():
    folder = tempfile.mkdtemp()
    try:
        path = os.path.join(folder, "train.npz")
        save_split(
            path,
            np.zeros((2, 60, 8, 128)),
            np.zeros((2, 60, 8, 8)),
            np.asarray([0, 1]),
        )
        with pytest.raises(ValueError, match="no stored ends"):
            load_strength_clock(path)
    finally:
        shutil.rmtree(folder)


def test_evo_from_metrics_copies_event_numbers():
    row = evo_from_metrics(
        {
            "event_sensitivity": 1.0,
            "n_seizures_caught": 1,
            "false_alarms_per_hour": 1.2,
            "test_still_overlap": True,
            "threshold_set": "val",
        }
    )
    assert row["event_sensitivity"] == pytest.approx(1.0)
    assert row["n_seizures_caught"] == 1
    assert row["false_alarms_per_hour"] == pytest.approx(1.2)
    assert row["test_still_overlap"] is True
    assert row["threshold_set"] == "val"


def test_plain_verdict_is_task_hard_when_test_still_overlaps():
    row = _patient_row(test_frac=0.988, criterion=False)
    assert plain_verdict([row]) == "task hard"


def test_plain_verdict_is_recipe_hard_when_most_sop_beats_interictal():
    row = _patient_row(test_frac=0.2, criterion=False)
    assert plain_verdict([row]) == "recipe hard"


def test_plain_verdict_is_recipe_hard_when_study_criterion_is_met():
    row = _patient_row(test_frac=1.0, criterion=True)
    assert plain_verdict([row]) == "recipe hard"


def _patient_row(test_frac, criterion):
    return {
        "baseline": {
            "study_criterion_met": criterion,
            "logit_overlap": {
                "test": {"frac_sop_at_or_under_interictal_max": test_frac}
            },
        }
    }
