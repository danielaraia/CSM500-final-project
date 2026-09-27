import json
import os
import shutil
import tempfile

import pytest

from evaluation.alarms import (
    lookback_seconds,
    silent_seconds,
    test_leads_from_seizures as leads_from_seizures,
    vote_need,
)
from evaluation.catch_coverage import catch_coverage_from_ends
from core.config import read_config, run_folder_path, start_run, write_config
from data.labels import (
    LABEL_INTERICTAL,
    LABEL_SOP,
    LABEL_UNUSED,
    SOP_SECONDS,
    SPH_SECONDS,
    label_second,
    seizures_with_full_sop,
    split_cut_times,
)
from packs.pack_sop import parse_args as parse_pack_args
from evaluation.report_sop import grid_row, plain_table
from evaluation.score_sop import run_grid
from train.train_sop import parse_args as parse_train_args

SOP_15 = 15 * 60
SOP_5 = 5 * 60


def test_label_second_at_15_minutes():
    onset = 10_000
    seizures = [(onset, 10_200)]
    sop_start = onset - SPH_SECONDS - SOP_15
    assert label_second(sop_start, seizures, SOP_15) == LABEL_SOP
    assert label_second(onset - SPH_SECONDS - 1, seizures, SOP_15) == LABEL_SOP
    assert label_second(sop_start - 1, seizures, SOP_15) == LABEL_UNUSED
    assert label_second(6_399, seizures, SOP_15) == LABEL_INTERICTAL


def test_label_second_at_5_minutes():
    onset = 10_000
    seizures = [(onset, 10_200)]
    sop_start = onset - SPH_SECONDS - SOP_5
    assert label_second(sop_start, seizures, SOP_5) == LABEL_SOP
    assert label_second(sop_start - 1, seizures, SOP_5) == LABEL_UNUSED


def test_shorter_sop_never_turns_old_sop_seconds_interictal():
    onset = 10_000
    seizures = [(onset, 10_200)]
    old_sop_start = onset - SPH_SECONDS - SOP_SECONDS
    t = old_sop_start
    while t < onset - SPH_SECONDS:
        assert label_second(t, seizures) == LABEL_SOP
        assert label_second(t, seizures, SOP_15) in (LABEL_SOP, LABEL_UNUSED)
        assert label_second(t, seizures, SOP_5) in (LABEL_SOP, LABEL_UNUSED)
        t += 100


def test_label_second_default_stays_30_minutes():
    onset = 10_000
    seizures = [(onset, 10_200)]
    assert label_second(8_000, seizures) == LABEL_SOP
    assert label_second(8_000, seizures, SOP_15) == LABEL_UNUSED
    assert label_second(8_000, seizures, SOP_5) == LABEL_UNUSED


def test_seizures_with_full_sop_follows_the_sop_length():
    seizures = [(1_300.0, 1_340.0), (9_000.0, 9_100.0), (20_000.0, 20_100.0)]
    assert len(seizures_with_full_sop(seizures)) == 2
    kept = seizures_with_full_sop(seizures, SOP_15)
    assert kept == [(1_300.0, 1_340.0), (9_000.0, 9_100.0), (20_000.0, 20_100.0)]
    assert len(seizures_with_full_sop(seizures, SOP_5)) == 3


def test_split_cut_times_move_with_the_sop_length():
    seizures = [
        (10_000, 10_100),
        (30_000, 30_100),
        (50_000, 50_100),
    ]
    t1_30, t2_30, n_train_30, n_test_30 = split_cut_times(seizures)
    t1_5, t2_5, n_train_5, n_test_5 = split_cut_times(seizures, SOP_5)
    assert (n_train_30, n_test_30) == (n_train_5, n_test_5)
    assert t2_30 == t2_5
    assert t1_5 <= t1_30


def test_lookback_and_vote_scale_with_the_sop_length():
    assert lookback_seconds(SOP_SECONDS) == 300
    assert vote_need(300) == 240
    assert lookback_seconds(SOP_15) == 150
    assert vote_need(150) == 120
    assert lookback_seconds(SOP_5) == 50
    assert vote_need(50) == 40
    assert silent_seconds(SPH_SECONDS, SOP_SECONDS) == 2100
    assert silent_seconds(SPH_SECONDS, SOP_15) == 1200
    assert silent_seconds(SPH_SECONDS, SOP_5) == 600


def test_test_leads_follow_the_sop_length():
    seizures = [
        (1_500.0, 1_540.0),
        (9_000.0, 9_100.0),
        (20_000.0, 20_100.0),
        (40_000.0, 40_100.0),
    ]
    _t1, _t2, leads_30, _m30, n_test_30 = leads_from_seizures(seizures)
    _t1, _t2, leads_5, _m5, n_test_5 = leads_from_seizures(seizures, SOP_5)
    assert n_test_30 == 1
    assert n_test_5 == 2
    assert leads_30 == [40_000.0]
    assert leads_5 == [20_000.0, 40_000.0]


def test_catch_coverage_uses_the_shorter_lookback():
    onset = 20_000
    ends = list(range(0, 25_000))
    dense = catch_coverage_from_ends(ends, [onset], 25_000, SOP_5)
    assert dense["alarm_lookback_seconds"] == 50
    assert dense["alarm_vote_need"] == 40
    assert dense["n_test_leads_evaluable"] == 1
    # a 1900 sec hole ends 500 sec before onset: 240 of 300 can never fill,
    # but 40 of 50 fills again before the hit interval closes
    gappy_ends = [e for e in ends if not (17_600 <= e <= 19_499)]
    gappy = catch_coverage_from_ends(gappy_ends, [onset], 25_000, SOP_5)
    assert gappy["n_test_leads_evaluable"] == 1
    gappy_30 = catch_coverage_from_ends(gappy_ends, [onset], 25_000)
    assert gappy_30["n_test_leads_evaluable"] == 0


def test_write_config_allows_15_and_5_and_rejects_20():
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    handle.close()
    try:
        write_config(handle.name, {"sop_minutes": 15})
        assert read_config(handle.name)["sop_minutes"] == 15
        write_config(handle.name, {"sop_minutes": 5})
        assert read_config(handle.name)["sop_minutes"] == 5
        with pytest.raises(ValueError, match="sop_minutes has to be"):
            write_config(handle.name, {"sop_minutes": 20})
    finally:
        os.remove(handle.name)


def test_sop_run_folders_do_not_collide():
    root = tempfile.mkdtemp()
    try:
        path_30 = start_run(root, "sub-004", leaf="eval1")
        path_15 = start_run(root, "sub-004", extra={"sop_minutes": 15}, leaf="eval1")
        path_5 = start_run(root, "sub-004", extra={"sop_minutes": 5}, leaf="eval1")
        assert path_30 != path_15 != path_5
        assert path_15 == run_folder_path(root, "sub-004", "eval1", sop_minutes=15)
        assert os.path.basename(os.path.dirname(os.path.dirname(path_15))) == "sop15"
        assert os.path.basename(os.path.dirname(os.path.dirname(path_5))) == "sop5"
        loaded = read_config(os.path.join(path_15, "config.json"))
        assert loaded["sop_minutes"] == 15
        assert loaded["seed"] == 123
    finally:
        shutil.rmtree(root)


def test_pack_sop_parse_args():
    assert parse_pack_args(
        ["pack_sop.py", "sub-004", "01", "mi", "15"]
    ) == ("sub-004", "01", "mi", 15)
    with pytest.raises(ValueError, match="usage"):
        parse_pack_args(["pack_sop.py", "sub-004"])
    with pytest.raises(ValueError, match="pearson or mi"):
        parse_pack_args(["pack_sop.py", "sub-004", "01", "cosine", "15"])
    with pytest.raises(ValueError, match="learned at train time"):
        parse_pack_args(["pack_sop.py", "sub-004", "01", "attention", "15"])
    with pytest.raises(ValueError, match="sop_minutes has to be"):
        parse_pack_args(["pack_sop.py", "sub-004", "01", "pearson", "20"])
    with pytest.raises(ValueError, match="whole number"):
        parse_pack_args(["pack_sop.py", "sub-004", "01", "pearson", "fifteen"])


def test_train_sop_parse_args_covers_the_whole_grid():
    grid = parse_train_args(["train_sop.py"])
    assert len(grid) == 42
    assert ("sub-015", "attention", 5) in grid
    assert ("sub-004", "pearson", 15) in grid
    assert parse_train_args(["train_sop.py", "sub-051", "mi", "15"]) == [
        ("sub-051", "mi", 15)
    ]
    with pytest.raises(ValueError, match="sop_minutes has to be"):
        parse_train_args(["train_sop.py", "sub-051", "mi", "20"])
    with pytest.raises(ValueError, match="usage"):
        parse_train_args(["train_sop.py", "sub-051"])


def test_score_sop_grid_matches_the_train_grid():
    grid = run_grid()
    assert len(grid) == 42
    assert ("sub-073", "attention", 5) in grid


def test_report_sop_grid_row_states():
    root = tempfile.mkdtemp()
    try:
        row = grid_row(root, "sub-004", "pearson", 15)
        assert row["status"] == "missing"
        folder = run_folder_path(root, "sub-004", "eval1", sop_minutes=15)
        os.makedirs(folder)
        with open(os.path.join(folder, "metrics.json"), "w") as handle:
            json.dump({"patient_id": "sub-004", "pack_seconds": 1.0}, handle)
        row = grid_row(root, "sub-004", "pearson", 15)
        assert row["status"] == "packed_not_trained"
        with open(os.path.join(folder, "metrics.json"), "w") as handle:
            json.dump(
                {
                    "patient_id": "sub-004",
                    "train_seconds": 2.0,
                    "n_seizures_caught": 1,
                    "n_seizures_missed": 0,
                    "n_seizures_evaluable": 1,
                    "n_seizures_unevaluable": 0,
                    "n_false_alarms": 0,
                    "false_alarms_per_hour": 0.0,
                    "threshold_set": "val",
                },
                handle,
            )
        row = grid_row(root, "sub-004", "pearson", 15)
        assert row["status"] == "scored"
        assert row["n_seizures_caught"] == 1
        table = plain_table([row])
        assert "sub-004" in table
        assert "scored" in table
    finally:
        shutil.rmtree(root)
