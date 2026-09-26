import os
import shutil
import tempfile

import numpy as np
import pytest

from alarms import save_window_scores
from logit_overlap import overlap_from_pack_and_scores, overlap_row, still_overlap
from packs.pack import save_split


def test_full_overlap_when_every_sop_logit_sits_under_the_interictal_max():
    labels = np.asarray([1, 1, 0, 0])
    logits = np.asarray([-0.2, -0.1, -0.05, 0.0])
    row = overlap_row(labels, logits)
    assert row["n_sop"] == 2
    assert row["n_interictal"] == 2
    assert row["sop_median"] == pytest.approx(-0.15)
    assert row["sop_max"] == pytest.approx(-0.1)
    assert row["interictal_median"] == pytest.approx(-0.025)
    assert row["interictal_max"] == pytest.approx(0.0)
    assert row["frac_sop_at_or_under_interictal_max"] == pytest.approx(1.0)
    assert still_overlap(row) is True


def test_separates_when_one_sop_logit_sits_above_every_interictal_logit():
    labels = np.asarray([1, 1, 0, 0])
    logits = np.asarray([0.4, -0.1, -0.2, 0.0])
    row = overlap_row(labels, logits)
    assert row["frac_sop_at_or_under_interictal_max"] == pytest.approx(0.5)
    assert still_overlap(row) is False


def test_empty_sop_does_not_invent_an_overlap_verdict():
    labels = np.asarray([0, 0, 0])
    logits = np.asarray([0.1, 0.2, 0.0])
    row = overlap_row(labels, logits)
    assert row["n_sop"] == 0
    assert row["frac_sop_at_or_under_interictal_max"] is None
    with pytest.raises(ValueError, match="cannot judge overlap"):
        still_overlap(row)


def test_sop_only_split_does_not_invent_an_overlap_verdict():
    labels = np.asarray([1, 1, 1])
    logits = np.asarray([0.1, 0.2, 0.0])
    row = overlap_row(labels, logits)
    assert row["n_sop"] == 3
    assert row["n_interictal"] == 0
    assert row["frac_sop_at_or_under_interictal_max"] is None
    with pytest.raises(ValueError, match="cannot judge overlap"):
        still_overlap(row)


def test_overlap_from_pack_checks_ends_match():
    folder = tempfile.mkdtemp()
    try:
        pack_path = os.path.join(folder, "test.npz")
        scores_path = os.path.join(folder, "test_scores.npz")
        save_split(
            pack_path,
            np.zeros((4, 60, 8, 128)),
            np.zeros((4, 60, 8, 8)),
            np.asarray([1, 1, 0, 0]),
            ends=[10, 11, 12, 13],
        )
        save_window_scores(scores_path, [10, 11, 12, 13], [-0.2, -0.1, -0.05, 0.0])
        row = overlap_from_pack_and_scores(pack_path, scores_path)
        assert still_overlap(row) is True
        save_window_scores(scores_path, [10, 11, 12, 99], [-0.2, -0.1, -0.05, 0.0])
        with pytest.raises(ValueError, match="pack ends do not match"):
            overlap_from_pack_and_scores(pack_path, scores_path)
    finally:
        shutil.rmtree(folder)
