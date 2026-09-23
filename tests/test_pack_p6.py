import json
import os
import tempfile

from pack_p6 import (
    append_ledger,
    classify_failure,
    ledger_row,
    load_cohort,
    parse_args,
    r2_pack_path,
    run_files_missing,
)


def test_r2_pack_path_matches_the_locked_layout():
    assert r2_pack_path("sub-043", 10) == (
        "msc-uol-r2:msc-uol-work/msc-uol/packs/within_patient/"
        "sub-043/pearson/sop10/seed123/eval1"
    )
    assert r2_pack_path("sub-043", 15).endswith("sub-043/pearson/sop15/seed123/eval1")
    assert r2_pack_path("sub-043", 5).endswith("sub-043/pearson/sop5/seed123/eval1")


def test_ledger_row_carries_the_brief_fields():
    row = ledger_row("sub-043", "08", 10, "packed", None, "msc-uol-r2:x")
    assert row["patient_id"] == "sub-043"
    assert row["sop_minutes"] == 10
    assert row["status"] == "packed"
    assert row["reason"] is None
    assert row["r2_path"] == "msc-uol-r2:x"
    assert row["ts"]
    metrics = {"n_train_windows": 55, "n_positive_train": 10, "unrelated": 1}
    row = ledger_row("sub-043", "08", 10, "packed", None, "x", metrics)
    assert row["n_train_windows"] == 55
    assert row["n_positive_train"] == 10
    assert "unrelated" not in row


def test_append_ledger_writes_one_json_line():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ledger.jsonl")
        append_ledger(ledger_row("sub-043", "08", 15, "qc_fail", "boom", None), path)
        append_ledger(ledger_row("sub-043", "08", 10, "packed", None, "x"), path)
        with open(path) as handle:
            lines = handle.read().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["status"] == "qc_fail"
        assert json.loads(lines[1])["status"] == "packed"


def test_classify_failure_known_qc_messages():
    text = "Traceback...\nValueError: gradient train has no sop windows, we got n_positive_train=0 n_negative_train=136\n"
    status, reason = classify_failure(text)
    assert status == "qc_fail"
    assert "n_positive_train=0" in reason
    text = "ValueError: monitoring slice is missing a class, we got n_positive=0 n_negative=26"
    status, reason = classify_failure(text)
    assert status == "qc_fail"
    assert "n_positive=0" in reason


def test_classify_failure_unknown_is_error():
    status, reason = classify_failure("Traceback...\nMemoryError: boom\n")
    assert status == "error"
    assert "MemoryError" in reason
    status, reason = classify_failure("")
    assert status == "error"
    assert reason == "packer died with no output"


def test_run_files_missing():
    with tempfile.TemporaryDirectory() as root:
        paths = {key: os.path.join(root, key) for key in ("eeg", "ecg", "emg", "mov", "events")}
        assert run_files_missing(paths) == ["eeg", "ecg", "emg", "mov", "events"]
        for key in ("eeg", "events"):
            with open(paths[key], "w") as handle:
                handle.write("x")
        assert run_files_missing(paths) == ["ecg", "emg", "mov"]


def test_load_cohort_reads_the_screen_output():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "cohort.json")
        with open(path, "w") as handle:
            json.dump({"cohort": [{"patient_id": "sub-043", "run_id": "08"}]}, handle)
        assert load_cohort(path) == [{"patient_id": "sub-043", "run_id": "08"}]


def test_parse_args():
    assert parse_args(["pack_p6.py"]) == (None, False)
    assert parse_args(["pack_p6.py", "--allow-big"]) == (None, True)
    assert parse_args(["pack_p6.py", "sub-043", "08"]) == (("sub-043", "08"), False)
    assert parse_args(["pack_p6.py", "sub-043", "08", "--allow-big"]) == (
        ("sub-043", "08"),
        True,
    )
