import json
import os
import subprocess
import tempfile

from pack_sop10_p6 import (
    EXISTING_COHORT,
    PHASE,
    SOP_MINUTES,
    pack_patient_sop10,
    parse_args,
    r2_pack_exists,
    sop10_ledger_row,
)


def test_existing_cohort_is_the_seven_p4_patients_on_their_runs():
    assert [patient for patient, _run, _ram in EXISTING_COHORT] == [
        "sub-004",
        "sub-015",
        "sub-039",
        "sub-051",
        "sub-052",
        "sub-073",
        "sub-113",
    ]
    runs = {patient: run for patient, run, _ram in EXISTING_COHORT}
    assert runs == {
        "sub-004": "01",
        "sub-015": "04",
        "sub-039": "06",
        "sub-051": "01",
        "sub-052": "07",
        "sub-073": "50",
        "sub-113": "04",
    }


def test_sop10_ledger_row_phase_and_sop():
    row = sop10_ledger_row("sub-051", "01", "packed", None, "msc-uol-r2:x")
    assert row["sop_minutes"] == SOP_MINUTES == 10
    assert row["phase"] == PHASE
    assert row["patient_id"] == "sub-051"
    assert row["run_id"] == "01"
    assert row["r2_path"] == "msc-uol-r2:x"
    assert row["ts"]


def test_r2_pack_exists_true_when_metrics_listed(monkeypatch):
    class Proc:
        returncode = 0
        stdout = "config.json\nmetrics.json\ntrain.npz\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc())
    assert r2_pack_exists("msc-uol-r2:bucket/some/path") is True


def test_r2_pack_exists_false_on_error_or_empty(monkeypatch):
    class Proc:
        returncode = 4
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc())
    assert r2_pack_exists("msc-uol-r2:bucket/missing") is False

    class EmptyProc:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: EmptyProc())
    assert r2_pack_exists("msc-uol-r2:bucket/empty") is False


def test_pack_patient_sop10_skips_when_r2_pack_exists(monkeypatch):
    monkeypatch.setattr(
        "pack_sop10_p6.r2_pack_exists", lambda r2_path: True
    )
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ledger.jsonl")
        pack_patient_sop10("sub-051", "01", 9.10, ledger_path=path)
        with open(path) as handle:
            rows = [json.loads(line) for line in handle.read().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "skipped_exists"
    assert row["r2_path"].endswith("sub-051/pearson/sop10/seed123/eval1")
    assert row["phase"] == PHASE


def test_pack_patient_sop10_defers_big_ram_without_allow_big(monkeypatch):
    monkeypatch.setattr(
        "pack_sop10_p6.r2_pack_exists", lambda r2_path: False
    )
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ledger.jsonl")
        pack_patient_sop10("sub-015", "04", 21.47, ledger_path=path)
        with open(path) as handle:
            rows = [json.loads(line) for line in handle.read().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "deferred_big_ram"
    assert "21.47" in row["reason"]
    assert "allow-big" in row["reason"]
    assert row["r2_path"] is None


def test_parse_args():
    assert parse_args(["pack_sop10_p6.py"]) == (None, False)
    assert parse_args(["pack_sop10_p6.py", "--allow-big"]) == (None, True)
    assert parse_args(["pack_sop10_p6.py", "sub-015"]) == ("sub-015", False)
    assert parse_args(["pack_sop10_p6.py", "sub-015", "--allow-big"]) == (
        "sub-015",
        True,
    )
