import json
import os
import tempfile

from pack_p6_relaxed import (
    PACK_SOP_MINUTES,
    PHASE,
    pack_patient,
    parse_args,
    relaxed_ledger_row,
    sop_list,
)


def _entry(**overrides):
    entry = {
        "patient_id": "sub-022",
        "run_id": "04",
        "longest_pass_sop_minutes": 15,
        "test_interictal_hours": 5.477,
        "fa_resolution": "limited",
        "ram_gib": 21.26,
        "needs_pod": True,
    }
    entry.update(overrides)
    return entry


def test_relaxed_ledger_row_carries_the_relaxed_gate_labels():
    row = relaxed_ledger_row(_entry(), 15, "packed", None, "msc-uol-r2:x")
    assert row["patient_id"] == "sub-022"
    assert row["run_id"] == "04"
    assert row["sop_minutes"] == 15
    assert row["status"] == "packed"
    assert row["phase"] == PHASE
    assert row["fa_resolution"] == "limited"
    assert row["test_interictal_hours"] == 5.477
    assert row["ts"]


def test_relaxed_ledger_row_marks_full_resolution():
    row = relaxed_ledger_row(
        _entry(fa_resolution="full", test_interictal_hours=12.45),
        10, "packed", None, "x",
    )
    assert row["fa_resolution"] == "full"
    assert row["test_interictal_hours"] == 12.45


def test_pack_patient_defers_big_ram_without_allow_big():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ledger.jsonl")
        pack_patient(_entry(), allow_big=False, ledger_path=path)
        with open(path) as handle:
            rows = [json.loads(line) for line in handle.read().splitlines()]
    assert len(rows) == len(PACK_SOP_MINUTES)
    assert [row["sop_minutes"] for row in rows] == [15, 10, 5]
    for row in rows:
        assert row["status"] == "deferred_big_ram"
        assert "allow-big" in row["reason"]
        assert row["r2_path"] is None
        assert row["fa_resolution"] == "limited"
        assert row["phase"] == PHASE


def test_sop_list_filters_to_one_sop():
    assert sop_list() == [15, 10, 5]
    assert sop_list(10) == [10]
    for bad in (30, 7):
        try:
            sop_list(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_pack_patient_defers_only_the_chosen_sop():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ledger.jsonl")
        pack_patient(_entry(), allow_big=False, only_sop=10, ledger_path=path)
        with open(path) as handle:
            rows = [json.loads(line) for line in handle.read().splitlines()]
    assert len(rows) == 1
    assert rows[0]["sop_minutes"] == 10
    assert rows[0]["status"] == "deferred_big_ram"


def test_parse_args():
    assert parse_args(["pack_p6_relaxed.py"]) == (None, None, False)
    assert parse_args(["pack_p6_relaxed.py", "--allow-big"]) == (None, None, True)
    assert parse_args(["pack_p6_relaxed.py", "sub-022", "04"]) == (
        ("sub-022", "04"),
        None,
        False,
    )
    assert parse_args(
        ["pack_p6_relaxed.py", "sub-022", "04", "--sop", "10", "--allow-big"]
    ) == (("sub-022", "04"), 10, True)
    try:
        parse_args(["pack_p6_relaxed.py", "--sop", "30"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
