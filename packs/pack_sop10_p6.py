"""Pack the seven already handled patients at SOP 10. Skips any pack already in R2."""

import json
import os
import subprocess
import sys

from core.config import run_folder_path
from evaluation.metrics import read_metrics
from packs.pack_eval import seizeit2_run_paths
from packs.pack_p6 import (
    GRAPH_STRATEGY,
    append_ledger,
    download_run_files,
    ledger_row,
    pack_one,
    r2_pack_path,
    upload_pack,
)
from tools.screen_p6 import RAM_GIB_SKIP

LEDGER_NAME = "screening-ledger.jsonl"
SOP_MINUTES = 10
PHASE = "p6-pack-sop10"
# patient, run of record, clock ram estimate in GiB (screen_p6.screen_run on
# the local SeizeIT2 meta tree, 22 September 2026). The estimate cannot see
# signal level QC, so real pack usage comes in lower.
EXISTING_COHORT = (
    ("sub-004", "01", 20.38),
    ("sub-015", "04", 21.47),
    ("sub-039", "06", 7.51),
    ("sub-051", "01", 9.10),
    ("sub-052", "07", 6.23),
    ("sub-073", "50", 1.62),
    ("sub-113", "04", 10.71),
)


def r2_pack_exists(r2_path):
    proc = subprocess.run(
        ["rclone", "lsf", r2_path, "--max-depth", "1"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    return "metrics.json" in proc.stdout.split()


def sop10_ledger_row(patient_id, run_id, status, reason, r2_path, metrics=None):
    row = ledger_row(
        patient_id, run_id, SOP_MINUTES, status, reason, r2_path, metrics
    )
    row["phase"] = PHASE
    return row


def pack_patient_sop10(patient_id, run_id, ram_gib, allow_big=False,
                       ledger_path=LEDGER_NAME):
    r2_path = r2_pack_path(patient_id, SOP_MINUTES)
    if r2_pack_exists(r2_path):
        row = sop10_ledger_row(
            patient_id, run_id, "skipped_exists",
            "sop10 pack already in r2", r2_path,
        )
        append_ledger(row, ledger_path)
        print(json.dumps(row), flush=True)
        return
    if ram_gib >= RAM_GIB_SKIP and not allow_big:
        row = sop10_ledger_row(
            patient_id, run_id, "deferred_big_ram",
            f"ram estimate {ram_gib} GiB >= {RAM_GIB_SKIP} GiB, "
            "pack on a pod with --allow-big",
            None,
        )
        append_ledger(row, ledger_path)
        print(json.dumps(row), flush=True)
        return
    paths = seizeit2_run_paths(patient_id, run_id)
    downloaded = download_run_files(patient_id, run_id, paths)
    if downloaded:
        print(f"downloaded {patient_id} run {run_id}: {downloaded}", flush=True)
    folder = run_folder_path(
        ".", patient_id, "eval1",
        graph_strategy=GRAPH_STRATEGY, sop_minutes=SOP_MINUTES,
    )
    if os.path.isfile(os.path.join(folder, "metrics.json")):
        print(f"pack exists, uploading: {folder}", flush=True)
        status, reason = "packed", None
    else:
        status, reason, folder = pack_one(patient_id, run_id, SOP_MINUTES)
    metrics = None
    if status == "packed":
        upload_pack(folder, r2_path)
        metrics = read_metrics(folder)
    else:
        r2_path = None
    row = sop10_ledger_row(patient_id, run_id, status, reason, r2_path, metrics)
    append_ledger(row, ledger_path)
    print(json.dumps(row), flush=True)


def parse_args(argv):
    allow_big = "--allow-big" in argv
    rest = [a for a in argv[1:] if a != "--allow-big"]
    if len(rest) == 0:
        return None, allow_big
    if len(rest) == 1:
        return rest[0], allow_big
    raise ValueError(
        "usage: python pack_sop10_p6.py [patient_id] [--allow-big]"
    )


def main():
    only, allow_big = parse_args(sys.argv)
    for patient_id, run_id, ram_gib in EXISTING_COHORT:
        if only is not None and patient_id != only:
            continue
        pack_patient_sop10(patient_id, run_id, ram_gib, allow_big=allow_big)


if __name__ == "__main__":
    main()
