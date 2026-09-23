"""Pack the screening cohort at SOP 15, 10, 5 and upload to R2.

One ledger row per attempt; big RAM runs defer unless --allow-big.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from config import run_folder_path
from metrics import read_metrics
from pack_eval import seizeit2_run_paths
from screen_p6 import RAM_GIB_SKIP

COHORT_NAME = "p6-screening-cohort.json"
LEDGER_NAME = "screening-ledger.jsonl"
GRAPH_STRATEGY = "pearson"
PACK_SOP_MINUTES = (15, 10, 5)
R2_PREFIX = "msc-uol-r2:msc-uol-work/msc-uol/packs/within_patient"
OPENNEURO_PREFIX = "s3://openneuro.org/ds005873"

# packer ValueError messages that are honest data geometry, not bugs
QC_FAIL_MARKERS = (
    "gradient train has no sop windows",
    "monitoring slice is missing a class",
    "monitoring slice is empty",
    "didnt leave any train windows after the monitoring cut",
    "didnt keep any labelled windows",
    "no test windows",
    "no train windows",
    "edf has silent gaps",
    "need at least two usable seizures",
)


def load_cohort(path=COHORT_NAME):
    with open(path) as handle:
        blob = json.load(handle)
    return blob["cohort"]


def r2_pack_path(patient_id, sop_minutes, graph_strategy=GRAPH_STRATEGY):
    return (
        f"{R2_PREFIX}/{patient_id}/{graph_strategy}/sop{sop_minutes}/seed123/eval1"
    )


def ledger_row(patient_id, run_id, sop_minutes, status, reason, r2_path, metrics=None):
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": "p6-pack",
        "patient_id": patient_id,
        "run_id": run_id,
        "graph_strategy": GRAPH_STRATEGY,
        "sop_minutes": sop_minutes,
        "status": status,
        "reason": reason,
        "r2_path": r2_path,
    }
    if metrics:
        for key in (
            "n_train_windows",
            "n_monitor_windows",
            "n_validation_windows",
            "n_test_windows",
            "n_positive_train",
            "n_negative_train",
            "n_positive_monitor",
            "n_negative_monitor",
            "n_positive_val",
            "n_negative_val",
            "n_positive_test",
            "n_negative_test",
            "n_train_seizures",
            "n_test_seizures",
            "n_test_leads",
            "n_test_leads_evaluable",
            "n_test_leads_unevaluable",
            "pack_seconds",
        ):
            if key in metrics:
                row[key] = metrics[key]
    return row


def append_ledger(row, path=LEDGER_NAME):
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")


def classify_failure(output_text):
    # the packer raises ValueError; the last traceback line carries the message
    message = ""
    for line in reversed(output_text.strip().splitlines()):
        line = line.strip()
        if line:
            message = line
            break
    for marker in QC_FAIL_MARKERS:
        if marker in message:
            return "qc_fail", message
    return "error", message or "packer died with no output"


def run_files_missing(paths):
    return [key for key in ("eeg", "ecg", "emg", "mov", "events")
            if not os.path.isfile(paths[key])]


def download_run_files(patient_id, run_id, paths):
    missing = run_files_missing(paths)
    if not missing:
        return []
    stem = f"{patient_id}_ses-01_task-szMonitoring_run-{run_id}"
    for key in missing:
        if key == "events":
            remote = f"{OPENNEURO_PREFIX}/{patient_id}/ses-01/eeg/{stem}_events.tsv"
        else:
            remote = f"{OPENNEURO_PREFIX}/{patient_id}/ses-01/{key}/{stem}_{key}.edf"
        os.makedirs(os.path.dirname(paths[key]), exist_ok=True)
        subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", remote, paths[key]],
            check=True,
        )
    return missing


def pack_one(patient_id, run_id, sop_minutes):
    proc = subprocess.run(
        [sys.executable, "pack_sop.py", patient_id, run_id, GRAPH_STRATEGY,
         str(sop_minutes)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        status, reason = classify_failure(proc.stderr + "\n" + proc.stdout)
        return status, reason, None
    folder = run_folder_path(
        ".", patient_id, "eval1",
        graph_strategy=GRAPH_STRATEGY, sop_minutes=sop_minutes,
    )
    return "packed", None, folder


def upload_pack(folder, r2_path):
    subprocess.run(
        ["rclone", "copy", folder, r2_path, "--transfers", "4"],
        check=True,
    )


def pack_patient(entry, allow_big=False, ledger_path=LEDGER_NAME):
    patient_id = entry["patient_id"]
    run_id = entry["run_id"]
    if entry.get("needs_pod") and not allow_big:
        for sop_minutes in PACK_SOP_MINUTES:
            row = ledger_row(
                patient_id, run_id, sop_minutes, "deferred_big_ram",
                f"ram estimate {entry.get('ram_gib')} GiB >= {RAM_GIB_SKIP} GiB, "
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
    for sop_minutes in PACK_SOP_MINUTES:
        folder = run_folder_path(
            ".", patient_id, "eval1",
            graph_strategy=GRAPH_STRATEGY, sop_minutes=sop_minutes,
        )
        r2_path = r2_pack_path(patient_id, sop_minutes)
        if os.path.isfile(os.path.join(folder, "metrics.json")):
            print(f"pack exists, uploading: {folder}", flush=True)
            status, reason = "packed", None
        else:
            status, reason, folder = pack_one(patient_id, run_id, sop_minutes)
        metrics = None
        if status == "packed":
            upload_pack(folder, r2_path)
            metrics = read_metrics(folder)
        else:
            r2_path = None
        row = ledger_row(
            patient_id, run_id, sop_minutes, status, reason, r2_path, metrics
        )
        append_ledger(row, ledger_path)
        print(json.dumps(row), flush=True)


def parse_args(argv):
    allow_big = "--allow-big" in argv
    rest = [a for a in argv[1:] if a != "--allow-big"]
    if len(rest) == 0:
        return None, allow_big
    if len(rest) == 2:
        return (rest[0], rest[1]), allow_big
    raise ValueError(
        "usage: python pack_p6.py [patient_id run_id] [--allow-big]"
    )


def main():
    only, allow_big = parse_args(sys.argv)
    cohort = load_cohort()
    for entry in cohort:
        if only is not None and (entry["patient_id"], entry["run_id"]) != only:
            continue
        pack_patient(entry, allow_big=allow_big)


if __name__ == "__main__":
    main()
