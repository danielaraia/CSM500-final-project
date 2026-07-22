import os
import tempfile

import numpy as np

from pack import load_split, save_split


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
        f2, a2, l2 = load_split(handle.name)
        np.testing.assert_array_equal(f2, features)
        np.testing.assert_array_equal(l2, labels)
        assert f2.shape[0] == 3
    finally:
        os.remove(handle.name)
