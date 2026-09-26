import os
import tempfile

import numpy as np
import pytest

from data.labels import LABEL_INTERICTAL, LABEL_SOP, LABEL_UNUSED
from pack import label_to_class_number, load_split, save_named_labels, save_split


def test_round_trip_one_file_not_one_per_window():
    features = np.ones((3, 60, 8, 128))
    adjacency = np.eye(8).reshape(1, 1, 8, 8)
    adjacency = np.repeat(adjacency, 3, axis=0)
    adjacency = np.repeat(adjacency, 60, axis=1)
    labels = np.array([0, 1, 0])
    handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    handle.close()
    try:
        save_split(handle.name, features, adjacency, labels)
        f2, a2, l2, ends = load_split(handle.name)
        np.testing.assert_array_equal(f2, features)
        np.testing.assert_array_equal(l2, labels)
        assert f2.shape[0] == 3
        assert ends is None
    finally:
        os.remove(handle.name)


def test_sop_is_one_and_interictal_is_zero():
    assert label_to_class_number(LABEL_SOP) == 1
    assert label_to_class_number(LABEL_INTERICTAL) == 0


def test_unused_label_cant_be_packed():
    with pytest.raises(ValueError, match="cant pack label"):
        label_to_class_number(LABEL_UNUSED)


def test_named_labels_round_trip_as_zeros_and_ones():
    features = np.ones((2, 60, 8, 128))
    adjacency = np.zeros((2, 60, 8, 8))
    handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    handle.close()
    try:
        save_named_labels(
            handle.name, features, adjacency, [LABEL_INTERICTAL, LABEL_SOP]
        )
        _f2, _a2, labels, ends = load_split(handle.name)
        np.testing.assert_array_equal(labels, np.array([0, 1]))
        assert ends is None
    finally:
        os.remove(handle.name)


def test_saved_ends_match_the_clock():
    features = np.ones((2, 60, 8, 128))
    adjacency = np.zeros((2, 60, 8, 8))
    labels = np.array([0, 1])
    ends = np.array([69, 70])
    handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    handle.close()
    try:
        save_split(handle.name, features, adjacency, labels, ends=ends)
        packed = np.load(handle.name)
        np.testing.assert_array_equal(packed["ends"], ends)
        _f2, _a2, _labels, loaded_ends = load_split(handle.name)
        np.testing.assert_array_equal(loaded_ends, ends)
    finally:
        os.remove(handle.name)
